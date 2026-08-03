"""
Live/Streaming Rhythm Detection Module

Two-stage real-time quantizer:
- Stage 1 (coarse, <50ms): per-note grid snap from BeatGrid for immediate display.
- Stage 2 (deferred, beats-adaptive): sequence-level Viterbi decoding over a
  trailing window. Runs on both binary (32nd) and ternary (32nd-triplet) grids
  and picks the lower-cost decode.

The BeatGrid (phase + period + subdivision) is the shared source of truth for
both stages. Phase anchors once when tempo confidence first crosses a
threshold; period tracks BPM updates continuously.
"""

import math
import os
import time
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
from display_chord_pairwise_model import canonicalize_pairwise_chord_group
from display_chord_pairwise_model import \
    load_pairwise_model as load_display_pairwise_model

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NOTE_TYPES = ['whole', 'half', 'quarter', 'eighth', '16th', '32nd']
NOTE_TYPE_BEATS = {
    'whole': 4.0, 'half': 2.0, 'quarter': 1.0,
    'eighth': 0.5, '16th': 0.25, '32nd': 0.125
}
BEATS_TO_TYPE = {v: k for k, v in NOTE_TYPE_BEATS.items()}

COARSE_CANDIDATES = [0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

REFINED_CANDIDATES = [
    (0.125, '32nd', False, False),
    (0.1667, '32nd', False, True),
    (0.1875, '32nd', True, False),
    (0.25, '16th', False, False),
    (0.333, '16th', False, True),
    (0.375, '16th', True, False),
    (0.5, 'eighth', False, False),
    (0.667, 'eighth', False, True),
    (0.75, 'eighth', True, False),
    (1.0, 'quarter', False, False),
    (1.333, 'quarter', False, True),
    (1.5, 'quarter', True, False),
    (2.0, 'half', False, False),
    (2.667, 'half', False, True),
    (3.0, 'half', True, False),
    (4.0, 'whole', False, False),
    (6.0, 'whole', True, False),
]

LIVE_MUSICAL_FRACS = {
    Fraction(1, 8):  ('32nd',    0.125,  False, False),
    Fraction(3, 16): ('32nd',    0.1875, True,  False),
    Fraction(1, 6):  ('16th',    1/6,    False, True),
    Fraction(1, 4):  ('16th',    0.25,   False, False),
    Fraction(1, 3):  ('eighth',  1/3,    False, True),
    Fraction(3, 8):  ('16th',    0.375,  True,  False),
    Fraction(1, 2):  ('eighth',  0.5,    False, False),
    Fraction(2, 3):  ('quarter', 2/3,    False, True),
    Fraction(3, 4):  ('eighth',  0.75,   True,  False),
    Fraction(1, 1):  ('quarter', 1.0,    False, False),
    Fraction(4, 3):  ('half',    4/3,    False, True),
    Fraction(3, 2):  ('quarter', 1.5,    True,  False),
    Fraction(2, 1):  ('half',    2.0,    False, False),
    Fraction(3, 1):  ('half',    3.0,    True,  False),
    Fraction(4, 1):  ('whole',   4.0,    False, False),
    Fraction(6, 1):  ('whole',   6.0,    True,  False),
}

_LIVE_FRAC_SORTED = sorted(
    [(float(f), f) for f in LIVE_MUSICAL_FRACS.keys()],
    key=lambda x: x[0]
)
_LIVE_FRAC_VALS = [x[0] for x in _LIVE_FRAC_SORTED]
_LIVE_FRAC_KEYS = [x[1] for x in _LIVE_FRAC_SORTED]
LIVE_TEMPO_ONSET_CLUSTER_TOLERANCE_SEC = 0.03
LIVE_TEMPO_NATURAL_MIN_BPM = 60.0
LIVE_TEMPO_NATURAL_MAX_BPM = 160.0
# The IOI cost/alignment metrics are octave-biased: dense 16th/32nd runs fit a
# doubled grid as eighths, so a fit-based selection cannot undo a 2x (or 1.5x)
# tempo error and the tracker collapses to 193/230/240 on busy passages. Fold the
# estimate back into the natural range as a prior. Env-gated for A/B testing.
LIVE_TEMPO_OCTAVE_GUARD = os.environ.get("LIVE_TEMPO_OCTAVE_GUARD", "1") != "0"
# Ternary tempo rescue. The binary tempo metric can't distinguish a triplet-
# eighth passage from its exact 0.75x binary alias (each 1/3-beat triplet reads
# as a clean 16th, each eighth as a dotted-16th), so a true-132 triplet run locks
# ~99-100 and the refine pass then engraves every triplet as a 16th. Rather than
# perturb the (heavily tuned) binary selection — which regresses ordinary binary
# music badly — we leave it untouched and add a narrow POST-HOC rescue: after the
# binary tracker picks a tempo, test the single specific hypothesis that the true
# tempo is 4/3x that (un-aliasing the 16th→triplet-eighth read), and switch ONLY
# when the rescaled tempo shows both a binary backbone (notes on the half grid)
# AND genuine sub-beat triplets riding on it. Pure binary music fails the triplet
# test; a bare triplet stream fails the anchor test — both are left unchanged.
LIVE_TEMPO_TERNARY_AWARE = os.environ.get("LIVE_TEMPO_TERNARY_AWARE", "1") != "0"
# Fraction of IOIs that must sit on a genuine sub-beat third (1/3, 2/3) at the
# rescaled tempo before a triplet reading is even considered.
LIVE_TEMPO_RESCUE_THIRD_FRAC = float(os.environ.get("LIVE_TEMPO_RESCUE_THIRD_FRAC", "0.18"))
# Fraction of IOIs that must anchor to the binary grid (a beat/half backbone) at
# the rescaled tempo — this is what separates real triplets-over-a-beat from a
# pure eighth run that merely aliases onto thirds.
LIVE_TEMPO_RESCUE_ANCHOR_FRAC = float(os.environ.get("LIVE_TEMPO_RESCUE_ANCHOR_FRAC", "0.15"))
LIVE_SCORE_DURATION_POLICY = os.environ.get("LIVE_SCORE_DURATION_POLICY", "ioi_same_voice")
LIVE_VOICE_ASSIGNMENT = os.environ.get("LIVE_VOICE_ASSIGNMENT", "per_hand")
# Re-express each display event's start_beat from its raw onset at the *final*
# reported tempo, snapped to 1/N of a beat. The per-note start_beat written
# during streaming is frozen at the tempo tracker's value at the moment the note
# was quantized; as the tracker later refines BPM, those beats drift relative to
# the final tempo the score is rendered at, distorting printed note values
# (the score-vs-MIDI tempo divergence). Recomputing from time_seconds at the
# final BPM removes that drift while still removing transcription jitter via the
# snap grid. 1/12 (eighth-triplet resolution) measured cleanest on gold12;
# set LIVE_DISPLAY_BEAT_SNAP_DIV=0 to restore the legacy frozen-grid behaviour.
LIVE_DISPLAY_BEAT_SNAP_DIV = int(os.environ.get("LIVE_DISPLAY_BEAT_SNAP_DIV", "12"))
DISPLAY_EVENT_DEDUPE_TOLERANCE_SEC = 0.05
DISPLAY_EVENT_GROUP_TOLERANCE_SEC = 0.03
DISPLAY_CHORD_RECONCILE_TOLERANCE_SEC = 0.01
DISPLAY_CHORD_EARLY_TIE_TOLERANCE_SEC = 0.003
MIDI_NOTE_NAMES = ('C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B')


def _fraction_snap(beats, max_denom=16):
    """Fallback snap for beat values that don't land on a grid table entry."""
    beats = max(0.0625, min(beats, 8.0))
    frac = Fraction(beats).limit_denominator(max_denom)
    frac_f = float(frac)
    best_idx = 0
    best_dist = abs(_LIVE_FRAC_VALS[0] - frac_f)
    for i in range(1, len(_LIVE_FRAC_VALS)):
        d = abs(_LIVE_FRAC_VALS[i] - frac_f)
        if d < best_dist:
            best_dist = d
            best_idx = i
    key = _LIVE_FRAC_KEYS[best_idx]
    if key in LIVE_MUSICAL_FRACS:
        note_type, note_beats, dotted, is_triplet = LIVE_MUSICAL_FRACS[key]
        if is_triplet or dotted:
            penalized = best_dist + (0.15 if is_triplet else 0.05)
            for j in range(max(0, best_idx - 2), min(len(_LIVE_FRAC_KEYS), best_idx + 3)):
                if j == best_idx:
                    continue
                alt_key = _LIVE_FRAC_KEYS[j]
                alt_info = LIVE_MUSICAL_FRACS.get(alt_key)
                if alt_info and not alt_info[2] and not alt_info[3]:
                    alt_dist = abs(_LIVE_FRAC_VALS[j] - frac_f)
                    if alt_dist < penalized:
                        return alt_info
        return (note_type, note_beats, dotted, is_triplet)

    best_beats = min(COARSE_CANDIDATES, key=lambda c: abs(beats - c))
    is_dotted = best_beats in [1.5, 3.0, 0.375, 0.1875, 0.75]
    base_beats = best_beats / 1.5 if is_dotted else best_beats
    note_type = 'quarter'
    for nt, nb in NOTE_TYPE_BEATS.items():
        if abs(base_beats - nb) < 0.01:
            note_type = nt
            break
    return (note_type, best_beats, is_dotted, False)


def _cluster_live_onset_times(
    notes: List[Dict],
    tolerance_sec: float = LIVE_TEMPO_ONSET_CLUSTER_TOLERANCE_SEC,
) -> List[float]:
    onset_times = sorted(
        float(note.get('time_seconds', 0.0) or 0.0)
        for note in (notes or [])
        if note.get('time_seconds') is not None
    )
    clustered: List[float] = []
    for onset_time in onset_times:
        if not clustered or onset_time - clustered[-1] > tolerance_sec:
            clustered.append(onset_time)
    return clustered


def estimate_bpm_from_notes(notes: List[Dict], seed_bpm: float = 0.0) -> float:
    """One-shot global tempo estimate from a note list, for stateless re-notation.

    The continuous /live/stream path never runs a tempo tracker, so the bpm the
    client sends to /live/refine is just its 120 default. Re-notating against 120
    aliases fast straight passages onto the triplet grid (a ~170bpm eighth at
    0.18s reads as a 0.167-beat triplet-eighth at 120) — which is exactly the
    "everything became triplets" failure. Here we rebuild the tempo from the note
    onsets themselves: cluster co-struck notes to one onset, seed a tracker with a
    coarse whole-piece IOI-histogram peak so its EMA doesn't lag from 120, then
    replay the onsets through the tuned tracker (octave guard + ternary rescue
    included) and read the settled tempo. Folds to the natural range like live.
    """
    onsets = _cluster_live_onset_times(notes)
    if len(onsets) < 5:
        return float(seed_bpm) if seed_bpm and seed_bpm > 1.0 else 120.0

    tracker = IncrementalTempoTracker()

    # Coarse global seed from the whole-piece IOI histogram (same candidate
    # divisors the tracker uses per-window), so the EMA starts near the truth
    # instead of crawling up from the 120 default over the first few updates.
    iois = np.diff(np.asarray(onsets, dtype=float))
    iois = iois[np.isfinite(iois) & (iois > 0)]
    candidates: List[float] = []
    for ioi in iois:
        for divisor in (0.25, 0.5, 1.0, 2.0, 4.0):
            bpm = 60.0 / (ioi / divisor)
            if tracker.min_bpm <= bpm <= tracker.max_bpm:
                candidates.append(bpm)
    if candidates:
        hist, edges = np.histogram(
            candidates, bins=80, range=(tracker.min_bpm, tracker.max_bpm)
        )
        peak = int(np.argmax(hist))
        seed = float((edges[peak] + edges[peak + 1]) / 2.0)
        tracker.current_bpm = seed
        tracker.initial_bpm = seed
        tracker.beat_grid.period = 60.0 / max(seed, 1.0)

    for onset in onsets:
        tracker.add_onset(onset)

    return float(tracker.current_bpm)


def _note_name_from_midi(midi_note: int) -> str:
    octave = (int(midi_note) // 12) - 1
    return f"{MIDI_NOTE_NAMES[int(midi_note) % 12]}{octave}"


def _event_midi_pitch(event: Dict) -> Optional[int]:
    try:
        if event.get('midi_note') is not None:
            return int(event.get('midi_note'))
        midi_notes = event.get('midi_notes') or []
        if midi_notes:
            return int(min(int(note) for note in midi_notes))
    except (TypeError, ValueError):
        return None
    return None


def _event_hand(event: Dict) -> str:
    hand = str(event.get('hand') or '').lower()
    if hand in ('bass', 'treble'):
        return hand
    pitch = _event_midi_pitch(event)
    return 'bass' if pitch is not None and pitch < 60 else 'treble'


def _voice_id_from_pitch(hand: str, pitch: Optional[int]) -> Tuple[str, int]:
    # 'per_hand': collapse each staff to a single notation lane (index 0 ->
    # MusicXML voice 1/2). The score's printed duration is the beat-IOI to the
    # next note in the SAME voice lane, and the GT/reference oracle carries no
    # voice info (renders as one voice per hand). Pitch-bucketed lanes therefore
    # both mismatch the reference voice number AND fragment a melodic line every
    # time it crosses a bucket boundary (over-extending its duration). One lane
    # per hand makes per-voice IOI == per-hand IOI, the correct default for the
    # mostly-monophonic-per-hand material we transcribe. gold12 score edit
    # accuracy 28.6 -> 41.1 vs 'pitch_lanes'. See memory
    # score_vs_midi_timing_divergence.
    if LIVE_VOICE_ASSIGNMENT == 'per_hand':
        return f"{hand}_voice_0", 0
    if pitch is None:
        index = 1
    elif hand == 'treble':
        if pitch >= 72:
            index = 0
        elif pitch >= 60:
            index = 1
        else:
            index = 2
    else:
        if pitch < 48:
            index = 0
        elif pitch < 60:
            index = 1
        else:
            index = 2
    return f"{hand}_voice_{index}", index


def _event_voice_id(event: Dict) -> str:
    voice_id = event.get('voice_id')
    if voice_id:
        return str(voice_id)
    hand = _event_hand(event)
    pitch = _event_midi_pitch(event)
    return _voice_id_from_pitch(hand, pitch)[0]


def assign_voice_ids(events: List[Dict]) -> List[Dict]:
    """Assign neutral notation lanes used by score-duration policies."""
    if LIVE_VOICE_ASSIGNMENT == 'off':
        return events
    for event in events or []:
        hand = _event_hand(event)
        pitch = _event_midi_pitch(event)
        voice_id, voice_index = _voice_id_from_pitch(hand, pitch)
        event.setdefault('voice_id', voice_id)
        event.setdefault('voice_index', voice_index)
        event.setdefault('voice_assignment', LIVE_VOICE_ASSIGNMENT)

        midi_notes = event.get('midi_notes') or []
        if midi_notes:
            voice_ids = []
            voice_indices = []
            for midi_note in midi_notes:
                try:
                    midi_value = int(midi_note)
                except (TypeError, ValueError):
                    continue
                note_hand = 'bass' if midi_value < 60 else 'treble'
                note_voice_id, note_voice_index = _voice_id_from_pitch(note_hand, midi_value)
                voice_ids.append(note_voice_id)
                voice_indices.append(note_voice_index)
            if voice_ids:
                event.setdefault('voice_ids', voice_ids)
                event.setdefault('voice_indices', voice_indices)
    return events


def _next_policy_onset(events: List[Dict], index: int, tolerance_sec: float = 1e-4) -> Optional[float]:
    if LIVE_SCORE_DURATION_POLICY not in ('ioi_same_hand', 'ioi_same_voice'):
        if index + 1 < len(events):
            try:
                return float(events[index + 1].get('time_seconds', 0.0) or 0.0)
            except (TypeError, ValueError):
                return None
        return None

    current = events[index]
    try:
        onset = float(current.get('time_seconds', 0.0) or 0.0)
    except (TypeError, ValueError):
        return None

    hand = _event_hand(current)
    voice_id = _event_voice_id(current)
    pitch = _event_midi_pitch(current)
    next_policy_onset = None
    next_same_pitch = None
    for candidate in events[index + 1:]:
        try:
            cand_onset = float(candidate.get('time_seconds', 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        if cand_onset - onset <= tolerance_sec:
            continue
        same_policy_lane = (
            _event_voice_id(candidate) == voice_id
            if LIVE_SCORE_DURATION_POLICY == 'ioi_same_voice'
            else _event_hand(candidate) == hand
        )
        if next_policy_onset is None and same_policy_lane:
            next_policy_onset = cand_onset
        cand_pitch = _event_midi_pitch(candidate)
        if pitch is not None and cand_pitch == pitch and next_same_pitch is None:
            next_same_pitch = cand_onset
        if next_policy_onset is not None and next_same_pitch is not None:
            break

    if next_policy_onset is None:
        return next_same_pitch
    if next_same_pitch is not None:
        return min(next_policy_onset, next_same_pitch)
    return next_policy_onset


def _display_event_time(event: Dict) -> float:
    return float(event.get('time_seconds', event.get('onset_time', 0.0)) or 0.0)


def _display_event_slot(event: Dict) -> Optional[int]:
    raw_idx = event.get('start_grid_idx')
    try:
        if raw_idx is not None:
            return int(raw_idx)
    except (TypeError, ValueError):
        pass

    raw_beat = event.get('start_beat')
    raw_subdivision = event.get('grid_subdivision')
    try:
        beat_value = float(raw_beat)
        subdivision = int(raw_subdivision)
    except (TypeError, ValueError):
        return None

    if subdivision <= 0:
        return None

    return int(round(beat_value * subdivision))


def _display_event_rank(event: Dict) -> int:
    source = str(event.get('_display_source') or '')
    if source == 'refined_note':
        return 3
    if source == 'note':
        return 2
    if source == 'chord_member':
        return 1
    return 0


def _display_event_score(event: Dict) -> tuple:
    return (
        _display_event_rank(event),
        1 if _display_event_slot(event) is not None else 0,
        float(event.get('quantization_confidence', 0.0) or 0.0),
        float(event.get('confidence', 0.0) or 0.0),
    )


def _display_event_sort_key(event: Dict) -> tuple:
    slot = _display_event_slot(event)
    try:
        midi_note = int(event.get('midi_note', 0) or 0)
    except (TypeError, ValueError):
        midi_note = 0

    return (
        0 if slot is not None else 1,
        slot if slot is not None else 0,
        _display_event_time(event),
        midi_note,
    )


def _sanitize_display_event(event: Dict) -> Dict:
    return {key: value for key, value in dict(event).items() if not str(key).startswith('_')}


def _display_chord_pitch_tuple(chord: Dict) -> tuple[int, ...]:
    pitches: List[int] = []
    for midi_note in chord.get('midi_notes') or []:
        try:
            pitches.append(int(midi_note))
        except (TypeError, ValueError):
            continue
    return tuple(sorted(pitches))


def _display_chords_conflict(
    left: Dict,
    right: Dict,
    overlap_threshold: float = 0.34,
) -> tuple[bool, float]:
    left_pitches = set(_display_chord_pitch_tuple(left))
    right_pitches = set(_display_chord_pitch_tuple(right))
    if not left_pitches or not right_pitches:
        return False, 0.0

    union = left_pitches | right_pitches
    if not union:
        return False, 0.0

    overlap = len(left_pitches & right_pitches) / len(union)
    subset_pair = left_pitches <= right_pitches or right_pitches <= left_pitches
    return (subset_pair or overlap >= overlap_threshold), overlap


def _select_display_chord_subset_heuristic(chord_group: List[Dict]) -> List[Dict]:
    if len(chord_group) > 7:
        earliest_time = min(_display_event_time(chord) for chord in chord_group)
        earliest_candidates = [
            chord
            for chord in chord_group
            if abs(_display_event_time(chord) - earliest_time) <= DISPLAY_CHORD_EARLY_TIE_TOLERANCE_SEC
        ]
        return [
            max(
                earliest_candidates,
                key=lambda chord: (
                    float(chord.get('confidence', 0.0) or 0.0),
                    len(_display_chord_pitch_tuple(chord)),
                ),
            )
        ]

    best_key = None
    best_subset = [
        max(
            chord_group,
            key=lambda chord: (
                float(chord.get('confidence', 0.0) or 0.0),
                len(_display_chord_pitch_tuple(chord)),
            ),
        )
    ]

    for subset_size in range(1, len(chord_group) + 1):
        for subset in combinations(chord_group, subset_size):
            confidence_score = sum(float(chord.get('confidence', 0.0) or 0.0) for chord in subset)
            density_bonus = 0.015 * sum(len(_display_chord_pitch_tuple(chord)) for chord in subset)
            conflict_penalty = 0.0
            for left, right in combinations(subset, 2):
                conflicts, overlap = _display_chords_conflict(left, right)
                if conflicts:
                    conflict_penalty += 1.1 + overlap

            score = confidence_score + density_bonus - conflict_penalty
            key = (score, -len(subset))
            if best_key is None or key > best_key:
                best_key = key
                best_subset = list(subset)

    return sorted(best_subset, key=_display_event_time)


def _select_display_chord_group(chord_group: List[Dict]) -> Dict[str, List[Dict]]:
    learned_model = load_display_pairwise_model()
    if learned_model is not None and 1 < len(chord_group) <= 7:
        canonical = canonicalize_pairwise_chord_group(chord_group, learned_model)
        return {
            'notes': list(canonical.get('notes') or ()),
            'chords': list(canonical.get('chords') or ()),
        }

    if len(chord_group) <= 1:
        return {
            'notes': [],
            'chords': list(chord_group),
        }

    return {
        'notes': [],
        'chords': _select_display_chord_subset_heuristic(chord_group),
    }


def _expand_display_chords_to_note_events(chords: List[Dict]) -> List[Dict]:
    note_events: List[Dict] = []

    for chord in chords or []:
        chord_dict = dict(chord)
        midi_notes = chord_dict.get('midi_notes') or []
        note_names = list(chord_dict.get('note_names') or [])
        voice_ids = list(chord_dict.get('voice_ids') or [])
        voice_indices = list(chord_dict.get('voice_indices') or [])

        for note_index, midi_note in enumerate(midi_notes):
            try:
                midi_value = int(midi_note)
            except (TypeError, ValueError):
                continue

            note_event = dict(chord_dict)
            note_event['midi_note'] = midi_value
            note_event['note_name'] = (
                note_names[note_index]
                if note_index < len(note_names)
                else _note_name_from_midi(midi_value)
            )
            if note_index < len(voice_ids):
                note_event['voice_id'] = voice_ids[note_index]
            if note_index < len(voice_indices):
                note_event['voice_index'] = voice_indices[note_index]
            note_event['_display_source'] = 'chord_member'
            note_events.append(note_event)

    return note_events


def _dedupe_display_note_events(
    note_events: List[Dict],
    time_tolerance_sec: float = DISPLAY_EVENT_DEDUPE_TOLERANCE_SEC,
) -> List[Dict]:
    deduped: List[Dict] = []

    for event in sorted(note_events or [], key=_display_event_sort_key):
        event_copy = dict(event)

        try:
            midi_note = int(event_copy.get('midi_note', 0) or 0)
        except (TypeError, ValueError):
            deduped.append(event_copy)
            continue

        event_copy['midi_note'] = midi_note
        event_slot = _display_event_slot(event_copy)
        event_time = _display_event_time(event_copy)

        duplicate_idx = None
        for idx in range(len(deduped) - 1, -1, -1):
            existing = deduped[idx]
            if existing.get('midi_note') != midi_note:
                continue

            existing_slot = _display_event_slot(existing)
            same_slot = (
                event_slot is not None
                and existing_slot is not None
                and event_slot == existing_slot
            )
            same_time = abs(_display_event_time(existing) - event_time) <= time_tolerance_sec
            if same_slot or same_time:
                duplicate_idx = idx
                break

        if duplicate_idx is None:
            deduped.append(event_copy)
            continue

        if _display_event_score(event_copy) > _display_event_score(deduped[duplicate_idx]):
            deduped[duplicate_idx] = event_copy

    return sorted(deduped, key=_display_event_sort_key)


def _group_display_note_events(note_events: List[Dict]) -> List[List[Dict]]:
    groups: List[List[Dict]] = []
    current_group: List[Dict] = []

    for event in sorted(note_events or [], key=_display_event_sort_key):
        if not current_group:
            current_group = [event]
            continue

        anchor = current_group[0]
        anchor_slot = _display_event_slot(anchor)
        event_slot = _display_event_slot(event)
        same_slot = (
            anchor_slot is not None
            and event_slot is not None
            and anchor_slot == event_slot
        )
        same_time = abs(_display_event_time(event) - _display_event_time(anchor)) <= DISPLAY_EVENT_GROUP_TOLERANCE_SEC

        if same_slot or ((anchor_slot is None or event_slot is None) and same_time):
            current_group.append(event)
            continue

        groups.append(current_group)
        current_group = [event]

    if current_group:
        groups.append(current_group)

    return groups


def _normalize_display_beats(events: List[Dict], bpm: float) -> None:
    """Snap each event's start_beat to its raw onset at the final reported tempo.

    See LIVE_DISPLAY_BEAT_SNAP_DIV: the streaming-time start_beat is frozen at the
    tempo-tracker value when the note was quantized, so it drifts from the final
    rendered tempo. Recomputing from time_seconds at the final BPM (snapped to a
    1/N-beat grid) removes that drift while preserving jitter removal."""
    div = LIVE_DISPLAY_BEAT_SNAP_DIV
    if div <= 0 or bpm <= 0:
        return
    beat_dur = 60.0 / bpm
    for event in events:
        raw = event.get('time_seconds', event.get('onset_time'))
        try:
            onset = float(raw)
        except (TypeError, ValueError):
            continue
        event['start_beat'] = round(onset / beat_dur * div) / div


def _build_display_surface(
    notes: List[Dict],
    chords: List[Dict],
    bpm: float = 0.0,
) -> Dict[str, List[Dict]]:
    sanitized_notes: List[Dict] = [_sanitize_display_event(note) for note in (notes or [])]
    sanitized_chords: List[Dict] = [_sanitize_display_event(chord) for chord in (chords or [])]
    _normalize_display_beats(sanitized_notes, bpm)
    _normalize_display_beats(sanitized_chords, bpm)
    assign_voice_ids(sanitized_notes)
    assign_voice_ids(sanitized_chords)

    generated_notes: List[Dict] = []
    reconciled_chords: List[Dict] = []
    chord_groups: List[List[Dict]] = []
    for chord in sorted(sanitized_chords, key=_display_event_time):
        chord_time = _display_event_time(chord)
        if (
            not chord_groups
            or (chord_time - _display_event_time(chord_groups[-1][0])) > DISPLAY_CHORD_RECONCILE_TOLERANCE_SEC
        ):
            chord_groups.append([chord])
            continue
        chord_groups[-1].append(chord)

    for chord_group in chord_groups:
        selected = _select_display_chord_group(chord_group)
        generated_notes.extend(selected.get('notes') or [])
        reconciled_chords.extend(selected.get('chords') or [])

    display_note_candidates: List[Dict] = []
    for note in sanitized_notes:
        note_copy = dict(note)
        note_copy['_display_source'] = 'note'
        display_note_candidates.append(note_copy)
    for note in generated_notes:
        note_copy = dict(note)
        note_copy['_display_source'] = 'note'
        display_note_candidates.append(note_copy)

    display_notes_internal = (
        _dedupe_display_note_events(display_note_candidates)
        if generated_notes
        else display_note_candidates
    )

    note_events: List[Dict] = list(display_notes_internal)
    note_events.extend(_expand_display_chords_to_note_events(reconciled_chords))

    return {
        'notes': [_sanitize_display_event(event) for event in display_notes_internal],
        'chords': reconciled_chords,
        'note_events': [_sanitize_display_event(event) for event in _dedupe_display_note_events(note_events)],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Beat Grid
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BeatGrid:
    """Phase-anchored beat grid. period scales with BPM, phase locks once."""
    phase: float = 0.0
    period: float = 0.5
    subdivision: int = 8
    anchored: bool = False
    anchor_confidence_threshold: float = 0.35

    def step_seconds(self) -> float:
        return self.period / self.subdivision if self.subdivision else self.period

    def grid_units(self, t: float) -> float:
        step = self.step_seconds()
        if step <= 0:
            return 0.0
        return (t - self.phase) / step

    def snap_idx(self, t: float) -> int:
        return int(round(self.grid_units(t)))

    def time_at_idx(self, idx: int) -> float:
        return self.phase + idx * self.step_seconds()

    def absolute_beat_at_idx(self, idx: int) -> float:
        """Return the snapped grid time as beats from recording/session start."""
        if self.period <= 0:
            return idx / max(self.subdivision, 1)
        return self.time_at_idx(idx) / self.period

    def beats_from(self, t: float) -> float:
        if self.period <= 0:
            return 0.0
        return (t - self.phase) / self.period

    def with_subdivision(self, sub: int) -> "BeatGrid":
        return BeatGrid(
            phase=self.phase,
            period=self.period,
            subdivision=sub,
            anchored=self.anchored,
            anchor_confidence_threshold=self.anchor_confidence_threshold,
        )


# Unit tables: number of grid units -> (note_type, beats, dotted, is_triplet)
_BINARY_UNIT_MAP = {
    1:  ('32nd',    0.125,  False, False),
    2:  ('16th',    0.25,   False, False),
    3:  ('16th',    0.375,  True,  False),
    4:  ('eighth',  0.5,    False, False),
    6:  ('eighth',  0.75,   True,  False),
    8:  ('quarter', 1.0,    False, False),
    12: ('quarter', 1.5,    True,  False),
    16: ('half',    2.0,    False, False),
    24: ('half',    3.0,    True,  False),
    32: ('whole',   4.0,    False, False),
    48: ('whole',   6.0,    True,  False),
}

_TERNARY_UNIT_MAP = {
    1:  ('32nd',    1/12,   False, True),
    2:  ('16th',    1/6,    False, True),
    3:  ('16th',    0.25,   False, False),
    4:  ('eighth',  1/3,    False, True),
    6:  ('eighth',  0.5,    False, False),
    8:  ('quarter', 2/3,    False, True),
    9:  ('eighth',  0.75,   True,  False),
    12: ('quarter', 1.0,    False, False),
    16: ('half',    4/3,    False, True),
    18: ('quarter', 1.5,    True,  False),
    24: ('half',    2.0,    False, False),
    32: ('whole',   8/3,    False, True),
    36: ('half',    3.0,    True,  False),
    48: ('whole',   4.0,    False, False),
    72: ('whole',   6.0,    True,  False),
}


def _units_to_musical(units: int, subdivision: int) -> Tuple[str, float, bool, bool]:
    table = _BINARY_UNIT_MAP if subdivision == 8 else _TERNARY_UNIT_MAP
    if units in table:
        return table[units]
    beats = units / max(subdivision, 1)
    return _fraction_snap(beats, max_denom=16)


def _unit_complexity_penalty(units: int, subdivision: int) -> float:
    """Cost added per transition by note-value complexity. Lower for plain values."""
    table = _BINARY_UNIT_MAP if subdivision == 8 else _TERNARY_UNIT_MAP
    if units not in table:
        return 0.6
    note_type, _, dotted, is_triplet = table[units]
    base = {
        'quarter': 0.0, 'eighth': 0.0, 'half': 0.0,
        '16th': 0.05, 'whole': 0.03, '32nd': 0.10,
    }.get(note_type, 0.2)
    if dotted:
        base += 0.07
    if is_triplet:
        base += 0.12
    return base


# ─────────────────────────────────────────────────────────────────────────────
# Incremental Tempo Tracker
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IncrementalTempoTracker:
    initial_bpm: float = 120.0
    min_bpm: float = 40.0
    max_bpm: float = 240.0
    window_iois: int = 32

    ioi_buffer: deque = field(default_factory=lambda: deque(maxlen=32))
    last_onset: float = -1.0
    current_bpm: float = 120.0
    confidence: float = 0.0
    _update_count: int = 0
    beat_grid: BeatGrid = field(default_factory=BeatGrid)

    def __post_init__(self):
        self.ioi_buffer = deque(maxlen=self.window_iois)
        self.current_bpm = self.initial_bpm
        self.beat_grid.period = 60.0 / self.current_bpm

    def reset(self):
        self.ioi_buffer.clear()
        self.last_onset = -1.0
        self.current_bpm = self.initial_bpm
        self.confidence = 0.0
        self._update_count = 0
        self.beat_grid = BeatGrid(period=60.0 / self.current_bpm)

    def add_onset(self, time_seconds: float) -> Tuple[float, float]:
        if self.last_onset >= 0:
            ioi = time_seconds - self.last_onset
            min_ioi = 60.0 / self.max_bpm / 4
            max_ioi = 60.0 / self.min_bpm * 4
            if min_ioi <= ioi <= max_ioi:
                self.ioi_buffer.append(ioi)
                self._update_count += 1
                if len(self.ioi_buffer) >= 4 and self._update_count % 2 == 0:
                    self._update_bpm()

        self.last_onset = time_seconds
        self.beat_grid.period = 60.0 / max(self.current_bpm, 1.0)

        if (
            not self.beat_grid.anchored
            and self.confidence >= self.beat_grid.anchor_confidence_threshold
        ):
            self.beat_grid.phase = time_seconds
            self.beat_grid.anchored = True

        return (self.current_bpm, self.confidence)

    def _update_bpm(self):
        if len(self.ioi_buffer) < 4:
            return
        iois = np.array(self.ioi_buffer)
        iois = iois[np.isfinite(iois) & (iois > 0)]
        if len(iois) < 4:
            return

        candidates = []
        for ioi in iois:
            for divisor in [0.25, 0.5, 1.0, 2.0, 4.0]:
                beat_dur = ioi / divisor
                bpm = 60.0 / beat_dur
                if self.min_bpm <= bpm <= self.max_bpm:
                    candidates.append(bpm)
        if not candidates:
            return
        candidates = np.array(candidates)
        hist, bin_edges = np.histogram(
            candidates, bins=80, range=(self.min_bpm, self.max_bpm)
        )
        peak_idx = np.argmax(hist)
        peak_bpm = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
        total = hist.sum()
        peak_count = hist[peak_idx]
        if peak_idx > 0:
            peak_count += hist[peak_idx - 1]
        if peak_idx < len(hist) - 1:
            peak_count += hist[peak_idx + 1]
        self.confidence = min(1.0, peak_count / max(total, 1) * 1.5)

        candidate_bpms = {
            round(float(bpm) * 2.0) / 2.0
            for bpm in candidates
            if self.min_bpm <= bpm <= self.max_bpm
        }
        candidate_bpms.add(round(float(peak_bpm) * 2.0) / 2.0)
        candidate_bpms.add(round(float(self.current_bpm) * 2.0) / 2.0)
        for base_bpm in [peak_bpm, self.current_bpm]:
            for mult in [0.5, 0.67, 0.75, 1.0, 1.33, 1.5, 2.0]:
                cand_bpm = float(base_bpm) * mult
                if self.min_bpm <= cand_bpm <= self.max_bpm:
                    candidate_bpms.add(round(cand_bpm * 2.0) / 2.0)

        tested_candidates = []
        best_bpm = peak_bpm
        best_cost = float('inf')
        best_alignment = float('-inf')
        for cand_bpm in sorted(candidate_bpms):
            cost = self._tempo_cost(iois, cand_bpm)
            alignment = self._alignment_score(iois, 60.0 / cand_bpm)
            tested_candidates.append((cand_bpm, cost, alignment))
            if (
                cost < best_cost - 0.005
                or (
                    abs(cost - best_cost) <= 0.005
                    and alignment > best_alignment + 1e-6
                )
            ):
                best_bpm = cand_bpm
                best_cost = cost
                best_alignment = alignment

        natural_candidates = [
            item for item in tested_candidates
            if LIVE_TEMPO_NATURAL_MIN_BPM <= item[0] <= LIVE_TEMPO_NATURAL_MAX_BPM
            and item[1] <= best_cost + 0.01
            and item[2] >= best_alignment * 0.97
        ]
        if natural_candidates and (
            best_bpm >= 180
            or best_bpm <= 55
            or natural_candidates[0][1] <= best_cost + 0.003
        ):
            natural_bpm, natural_cost, natural_alignment = min(
                natural_candidates,
                key=lambda item: (item[1], -item[2], abs(item[0] - self.current_bpm)),
            )
            best_bpm = natural_bpm
            best_cost = natural_cost
            best_alignment = natural_alignment

        # Octave-doubling guard. The cost/alignment metrics above genuinely favor
        # the doubled tempo on dense passages, so this prior — not a fit gate — is
        # what undoes it. Fold back into the natural range before smoothing.
        if LIVE_TEMPO_OCTAVE_GUARD:
            while (
                best_bpm > LIVE_TEMPO_NATURAL_MAX_BPM
                and best_bpm / 2.0 >= self.min_bpm
            ):
                best_bpm /= 2.0

        # Ternary rescue (see constant block): only un-aliases a triplet passage
        # the binary metric locked 0.75x low. No-op for binary music.
        if LIVE_TEMPO_TERNARY_AWARE:
            best_bpm = self._ternary_rescue(iois, best_bpm)

        alpha = 0.5 if self.confidence >= 0.5 else 0.2
        self.current_bpm = self.current_bpm * (1 - alpha) + best_bpm * alpha
        common_tempos = [60, 72, 80, 90, 100, 108, 120, 132, 140, 160, 180, 200]
        snap_tolerance = max(0.75, self.current_bpm * 0.01)
        for ct in common_tempos:
            if abs(self.current_bpm - ct) <= snap_tolerance:
                self.current_bpm = ct
                break

    @staticmethod
    def _tempo_cost(iois: np.ndarray, test_bpm: float) -> float:
        beat_period = 60.0 / max(test_bpm, 1.0)
        errors = []
        n_32nd = 0
        n_16th = 0
        for ioi in iois:
            beats = max(0.0625, min(float(ioi) / beat_period, 8.0))
            note_type, note_beats, _, _ = _fraction_snap(beats, max_denom=16)
            errors.append(abs(beats - note_beats) / max(note_beats, 1e-6))
            if note_type == '32nd':
                n_32nd += 1
            elif note_type == '16th':
                n_16th += 1

        if not errors:
            return 1.0

        base_error = float(np.mean(errors))
        n = len(errors)
        frac_32 = n_32nd / n
        frac_short = (n_32nd + n_16th) / n
        penalty = 0.0
        if frac_32 > 0.15:
            penalty += (frac_32 - 0.15) * 1.5
        if frac_32 > 0.50:
            penalty += (frac_32 - 0.50) * 2.0
        if frac_short > 0.50:
            penalty += (frac_short - 0.50) * 0.8
        return base_error + penalty

    @staticmethod
    def _alignment_score(iois, beat_period):
        score = 0.0
        for ioi in iois:
            ratio = ioi / beat_period
            nearest = round(ratio * 2) / 2
            if nearest < 0.25:
                continue
            dist = abs(ratio - nearest)
            score += math.exp(-(dist ** 2) / (2 * 0.08 ** 2))
        return score / len(iois)

    @staticmethod
    def _ternary_alignment_score(iois, beat_period):
        """Alignment that credits both the half grid and genuine sub-beat thirds.

        Used only to confirm a ternary rescue: a real triplet-over-a-beat texture
        scores far higher here at its true tempo than the plain (half-only) score
        does at the aliased binary tempo.
        """
        score = 0.0
        for ioi in iois:
            ratio = ioi / beat_period
            half = round(ratio * 2) / 2
            best = 0.0
            if half >= 0.25:
                best = math.exp(-((ratio - half) ** 2) / (2 * 0.08 ** 2))
            if ratio < 1.0:
                third = round(ratio * 3) / 3
                if third >= 0.25:
                    best = max(best, math.exp(-((ratio - third) ** 2) / (2 * 0.08 ** 2)))
            score += best
        return score / len(iois)

    def _ternary_rescue(self, iois, best_bpm: float) -> float:
        """Un-alias a triplet passage the binary metric locked to a 2:3 alias.

        A triplet-eighth run has two exact binary aliases: read as 16ths it locks
        0.75x low (true tempo = 4/3 x), read as eighths it locks 1.5x high (true
        tempo = 2/3 x). This tests both rescaled tempos and switches to one only
        when, there, the onsets show BOTH a binary backbone (a beat/half grid the
        triplets ride on) AND grouped sub-beat triplets, AND the ternary alignment
        clearly beats the binary alignment at the locked tempo. Every gate must
        pass, so ordinary binary music — no grouped sub-beat thirds at the rescaled
        tempo, or no anchor — is returned unchanged.
        """
        n = len(iois)
        if n < 4:
            return best_bpm
        binary_align = self._alignment_score(iois, 60.0 / max(best_bpm, 1.0))
        best_target = best_bpm
        best_gain = 0.05  # ternary alignment must beat binary by at least this
        for factor in (4.0 / 3.0, 2.0 / 3.0):
            target = best_bpm * factor
            # Only rescue into a musically plausible range.
            if not (LIVE_TEMPO_NATURAL_MIN_BPM <= target <= LIVE_TEMPO_NATURAL_MAX_BPM):
                continue
            period = 60.0 / target
            tol = 0.06
            is_third = []
            n_anchor = 0
            for ioi in iois:
                ratio = float(ioi) / period
                third = (
                    0.0 < ratio < 1.0
                    and (abs(ratio - 1.0 / 3) <= tol or abs(ratio - 2.0 / 3) <= tol)
                )
                is_third.append(third)
                # Binary backbone: an onset on the half/beat grid (nearest half-beat
                # multiple >= 0.5). Rounding, not a hard ratio>=0.5 floor, so a
                # half-beat note landing at 0.499 (target a hair off) still counts.
                if not third and ratio <= 8.0 and round(ratio * 2) >= 1 \
                        and abs(ratio * 2 - round(ratio * 2)) <= tol * 2:
                    n_anchor += 1

            # Count only GROUPED thirds — a third whose neighbour is also a third,
            # i.e. a run of >=2 equal sub-beat IOIs (a real triplet figure). This
            # rejects a dotted-eighth+16th binary rhythm, whose 16th aliases onto
            # 1/3 but sits ISOLATED between binary anchors, not in a run.
            n_grouped_third = 0
            for i, third in enumerate(is_third):
                if third and ((i > 0 and is_third[i - 1]) or (i < n - 1 and is_third[i + 1])):
                    n_grouped_third += 1

            if n_grouped_third / n < LIVE_TEMPO_RESCUE_THIRD_FRAC:
                continue
            if n_anchor / n < LIVE_TEMPO_RESCUE_ANCHOR_FRAC:
                continue
            gain = self._ternary_alignment_score(iois, period) - binary_align
            if gain > best_gain:
                best_gain = gain
                best_target = target
        return best_target

    def get_beat_duration(self) -> float:
        return 60.0 / self.current_bpm


# ─────────────────────────────────────────────────────────────────────────────
# Coarse Quantization (Stage 1)
# ─────────────────────────────────────────────────────────────────────────────

def _annotate_grid_position(note: Dict, grid: BeatGrid) -> None:
    if not grid.anchored:
        return
    onset = float(note.get('time_seconds', 0.0) or 0.0)
    idx = grid.snap_idx(onset)
    note['start_grid_idx'] = idx
    note['start_beat'] = grid.absolute_beat_at_idx(idx)
    note['grid_subdivision'] = grid.subdivision
    # Preserve the raw neural onset for runtime logic and expose a separate
    # snapped onset only for evaluation-side cluster grouping.
    note['cluster_metric_time_seconds'] = round(grid.time_at_idx(idx), 4)


def quantize_coarse(
    note: Dict,
    bpm: float,
    prev_notes: Optional[List[Dict]] = None,
    grid: Optional[BeatGrid] = None,
    next_onset_seconds: Optional[float] = None,
) -> Dict:
    """Single-note coarse quantization. Uses grid (start_idx -> next_idx) when
    available; falls back to fraction snap on duration_seconds otherwise."""
    beat_dur = 60.0 / max(bpm, 1.0)
    duration = float(note.get('duration_seconds', 0.0) or 0.0)
    onset = float(note.get('time_seconds', 0.0) or 0.0)
    policy_duration = None
    if next_onset_seconds is not None:
        try:
            policy_duration = max(0.0, float(next_onset_seconds) - onset)
        except (TypeError, ValueError):
            policy_duration = None

    used_grid = False
    note_type = 'quarter'
    best_beats = 1.0
    is_dotted = False

    if grid is not None and grid.anchored and next_onset_seconds is not None:
        this_idx = grid.snap_idx(onset)
        next_idx = grid.snap_idx(next_onset_seconds)
        if next_idx > this_idx:
            units = next_idx - this_idx
            cand_type, cand_beats, cand_dotted, cand_triplet = _units_to_musical(
                units, grid.subdivision
            )
            if not cand_triplet:
                note_type, best_beats, is_dotted = cand_type, cand_beats, cand_dotted
                used_grid = True

    if not used_grid:
        if policy_duration is not None and policy_duration > 0:
            duration = policy_duration
        if duration <= 0:
            duration = beat_dur
        beats = duration / beat_dur
        nt, nb, nd, nt_trip = _fraction_snap(beats, max_denom=8)
        if nt_trip:
            plain_beats = min(COARSE_CANDIDATES, key=lambda c: abs(beats - c))
            nd = plain_beats in [1.5, 3.0, 0.375, 0.1875, 0.75]
            base_beats = plain_beats / 1.5 if nd else plain_beats
            nt = 'quarter'
            for nm, nbv in NOTE_TYPE_BEATS.items():
                if abs(base_beats - nbv) < 0.01:
                    nt = nm
                    break
            nb = plain_beats
        note_type, best_beats, is_dotted = nt, nb, nd

    note['note_value'] = note_type
    note['note_divisions'] = best_beats
    note['dotted'] = is_dotted
    note['is_triplet'] = False
    note['triplet'] = False
    note['quantization_method'] = 'coarse_grid' if used_grid else 'coarse_live'
    if policy_duration is not None and policy_duration > 0:
        note['duration_source'] = LIVE_SCORE_DURATION_POLICY
        note['score_duration_seconds'] = round(policy_duration, 4)
    raw_beats = (duration / beat_dur) if duration > 0 else best_beats
    note['raw_beats'] = raw_beats
    note['quantization_confidence'] = max(
        0.5, 1.0 - abs(raw_beats - best_beats) / max(best_beats, 0.01)
    )

    if grid is not None:
        _annotate_grid_position(note, grid)

    return note


def quantize_batch_coarse(
    notes: List[Dict],
    bpm: float,
    grid: Optional[BeatGrid] = None,
) -> List[Dict]:
    assign_voice_ids(notes)
    for i, note in enumerate(notes):
        prev_notes = notes[max(0, i - 4):i] if i > 0 else None
        next_onset = _next_policy_onset(notes, i)
        quantize_coarse(
            note, bpm, prev_notes, grid=grid, next_onset_seconds=next_onset
        )
    return notes


# ─────────────────────────────────────────────────────────────────────────────
# Sequence-level Viterbi Decoder (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

def _decode_window_single(
    onsets: List[float],
    grid: BeatGrid,
    sigma_sec: float = 0.030,
    search_radius: int = 4,
) -> Tuple[List[int], float]:
    """Viterbi over one grid resolution. Returns (path, total_cost)."""
    if not onsets:
        return [], 0.0

    step = grid.step_seconds()
    sigma_units = max(sigma_sec / max(step, 1e-6), 0.5)

    candidates: List[List[int]] = []
    for t in onsets:
        center = grid.snap_idx(t)
        candidates.append(list(range(center - search_radius, center + search_radius + 1)))

    n = len(onsets)
    dp: List[Dict[int, float]] = [dict() for _ in range(n)]
    bp: List[Dict[int, Optional[int]]] = [dict() for _ in range(n)]

    obs0 = grid.grid_units(onsets[0])
    for idx in candidates[0]:
        cost = ((obs0 - idx) / sigma_units) ** 2
        dp[0][idx] = cost
        bp[0][idx] = None

    for i in range(1, n):
        obs = grid.grid_units(onsets[i])
        for idx in candidates[i]:
            emission = ((obs - idx) / sigma_units) ** 2
            best = math.inf
            best_prev = None
            for prev_idx, prev_cost in dp[i - 1].items():
                if prev_idx >= idx:
                    continue
                units = idx - prev_idx
                trans = _unit_complexity_penalty(units, grid.subdivision)
                if units > 96:
                    trans += 0.5
                total = prev_cost + emission + trans
                if total < best:
                    best = total
                    best_prev = prev_idx
            if best_prev is not None:
                dp[i][idx] = best
                bp[i][idx] = best_prev

    if not dp[-1]:
        return [grid.snap_idx(t) for t in onsets], math.inf

    last_idx = min(dp[-1].items(), key=lambda kv: kv[1])[0]
    total_cost = dp[-1][last_idx]
    path = [last_idx]
    for i in range(n - 1, 0, -1):
        prev = bp[i].get(path[-1])
        if prev is None:
            prev = path[-1] - 1
        path.append(prev)
    path.reverse()
    return path, total_cost


def decode_window_dual(
    onsets: List[float],
    base_grid: BeatGrid,
    last_note_duration: float = 0.5,
) -> Dict:
    """Decode an onset window with both 32nd (binary) and 32nd-triplet (ternary)
    grids. Returns the lower-cost decode plus per-note durations in grid units."""
    if not onsets:
        return {
            'subdivision': 8, 'indices': [], 'durations_units': [],
            'cost': 0.0, 'grid': base_grid,
        }

    binary_grid = base_grid.with_subdivision(8)
    ternary_grid = base_grid.with_subdivision(12)

    bin_path, bin_cost = _decode_window_single(onsets, binary_grid)
    ter_path, ter_cost = _decode_window_single(onsets, ternary_grid)

    ternary_bias = 0.25 * len(onsets)
    if ter_cost + ternary_bias < bin_cost:
        chosen = ternary_grid
        path = ter_path
        cost = ter_cost
    else:
        chosen = binary_grid
        path = bin_path
        cost = bin_cost

    durations_units: List[int] = []
    for i in range(len(path)):
        if i + 1 < len(path):
            durations_units.append(max(1, path[i + 1] - path[i]))
        else:
            est = max(1, int(round(last_note_duration / chosen.step_seconds())))
            durations_units.append(est)

    return {
        'subdivision': chosen.subdivision,
        'indices': path,
        'durations_units': durations_units,
        'cost': cost,
        'grid': chosen,
    }


def apply_window_decode(
    notes: List[Dict],
    decode: Dict,
    quantization_method: str = 'refined_window',
) -> None:
    """Annotate notes in-place from a dual-grid decode result."""
    grid: BeatGrid = decode['grid']
    sub = decode['subdivision']
    for note, idx, units in zip(notes, decode['indices'], decode['durations_units']):
        note_type, beats, dotted, is_triplet = _units_to_musical(units, sub)
        note['note_value'] = note_type
        note['note_divisions'] = beats
        note['dotted'] = dotted
        note['is_triplet'] = is_triplet
        note['triplet'] = is_triplet
        note['quantization_method'] = quantization_method
        note['start_grid_idx'] = idx
        note['start_beat'] = grid.absolute_beat_at_idx(idx)
        note['grid_subdivision'] = sub
        observed = float(note.get('time_seconds', 0.0) or 0.0)
        target_t = grid.time_at_idx(idx)
        note['cluster_metric_time_seconds'] = round(target_t, 4)
        timing_err = abs(observed - target_t)
        note['quantization_confidence'] = max(
            0.55, 1.0 - timing_err / max(grid.period, 0.01)
        )


def _note_onset_seconds(note: Dict) -> Optional[float]:
    for key in ('time_seconds', 'onset_time', 'onset'):
        v = note.get(key)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
    return None


def refine_notes_lookahead(
    notes: List[Dict],
    bpm: float,
    grid: Optional[BeatGrid] = None,
) -> Tuple[List[Dict], int]:
    """Lookahead-only re-notation of already-streamed notes.

    The streaming path (`quantizeStreamDurationBeats`, frontend) notates each
    note's ACOUSTIC duration (offset - onset = how long the key was physically
    held), snapped per-note with no lookahead. The intended score convention —
    used elsewhere by `quantize_coarse` via `next_onset` — is the grid-snapped
    INTER-ONSET INTERVAL to the next note in the same hand. Switching to IOI is
    exactly what the per-note acoustic snap cannot do without seeing the
    successor, and it is what this pass applies. It subsumes:

      * staccato fragmentation (short key-press -> 16th + rest) -> the notated
        value now fills to the next onset,
      * over-extended legato (held note overlapping the next) -> trimmed to IOI,
      * coherent TRIPLET groups (three consecutive ~1/3-beat IOIs snap ternary,
        then a coherence pass demotes any isolated triplet the per-note snap
        would have left dangling),
      * the trailing note of each hand, which has no successor and therefore
        keeps its acoustic duration (best available).

    A note is only rewritten when the IOI value actually differs from what the
    stream produced, so already-correct notes (acoustic ~= IOI) are untouched.
    Onsets/`start_beat` are left as the stream placed them; only note VALUES
    change. Mutates `notes` in place; returns (notes, changed_count).
    """
    valid = [n for n in notes if _note_onset_seconds(n) is not None]
    n = len(valid)
    if n < 1:
        return notes, 0
    beat = 60.0 / max(bpm, 1.0)

    order = sorted(range(n), key=lambda i: _note_onset_seconds(valid[i]))
    onset = [_note_onset_seconds(valid[i]) for i in order]
    hand = [_event_hand(valid[i]) for i in order]

    # next onset in the SAME hand (a bass note's value must not be cut by a
    # treble onset, and vice versa — matches the per_hand voice convention).
    next_same: List[Optional[float]] = [None] * n
    seen: Dict[str, float] = {}
    for pos in range(n - 1, -1, -1):
        next_same[pos] = seen.get(hand[pos])
        seen[hand[pos]] = onset[pos]

    # Positions (in sorted order) of the previous/next note in the SAME hand.
    # Needed by the triplet coherence pass: in a two-hand texture the immediate
    # sorted neighbours (pos±1) are usually the OTHER hand, so a triplet run in
    # one hand looks isolated unless we skip across to its same-hand partners.
    prev_same_pos: List[Optional[int]] = [None] * n
    next_same_pos: List[Optional[int]] = [None] * n
    last_pos: Dict[str, int] = {}
    for pos in range(n):
        h = hand[pos]
        if h in last_pos:
            prev_same_pos[pos] = last_pos[h]
            next_same_pos[last_pos[h]] = pos
        last_pos[h] = pos

    # IOI -> beats -> nearest musical value (triplet candidates included).
    # None = keep the streamed value (trailing note with no measurable duration).
    snap: List[Optional[Tuple[str, float, bool, bool]]] = [None] * n
    for pos in range(n):
        nxt = next_same[pos]
        if nxt is not None:
            ioi = nxt - onset[pos]
        else:
            # trailing note of this hand: no successor, so IOI is undefined. Fall
            # back to its own acoustic duration; if we don't even have that, leave
            # the streamed value alone rather than guess.
            ac = valid[order[pos]].get('duration')
            if ac is None:
                ac = valid[order[pos]].get('duration_seconds')
            if ac is None:
                continue
            ioi = float(ac)
        beats = min(4.0, max(1.0 / 12.0, ioi / beat))
        snap[pos] = _snap_beats_to_value(beats)

    # Coherence: a lone triplet cannot be engraved as a valid tuplet, so demote
    # any triplet without a same-hand triplet neighbour to its nearest binary
    # value. (Real triplet runs keep each other; strays fall back.)
    def _is_trip(p: int) -> bool:
        return snap[p] is not None and snap[p][3]

    for pos in range(n):
        if snap[pos] is None or not snap[pos][3]:
            continue
        p = prev_same_pos[pos]
        nx = next_same_pos[pos]
        prev_t = p is not None and _is_trip(p)
        next_t = nx is not None and _is_trip(nx)
        if not (prev_t or next_t):
            snap[pos] = _snap_beats_to_value(snap[pos][1], allow_triplet=False)

    # Group same-hand triplet runs into complete triples and assign
    # start/middle/end. The renderer (PianoSheetMusic.tsx) only emits a
    # <tuplet> bracket for a complete start->middle->end chain with matching
    # triplet_type on all three, and silently drops anything else — so a
    # `triplet: true` flag with no position/type here is invisible on the
    # score even though the beat-level math is correct. A run whose length
    # isn't a multiple of 3 has its tail demoted to the nearest binary value,
    # same as an isolated triplet above.
    triplet_position: List[Optional[str]] = [None] * n
    hand_positions: Dict[str, List[int]] = {}
    for pos in range(n):
        hand_positions.setdefault(hand[pos], []).append(pos)

    def _flush_run(run: List[int]) -> None:
        i = 0
        while i + 3 <= len(run):
            triplet_position[run[i]] = 'start'
            triplet_position[run[i + 1]] = 'middle'
            triplet_position[run[i + 2]] = 'end'
            i += 3
        for j in range(i, len(run)):
            p = run[j]
            snap[p] = _snap_beats_to_value(snap[p][1], allow_triplet=False)

    for positions in hand_positions.values():
        run: List[int] = []
        for pos in positions:
            if _is_trip(pos):
                run.append(pos)
            else:
                if run:
                    _flush_run(run)
                    run = []
        if run:
            _flush_run(run)

    changed = 0
    for pos in range(n):
        if snap[pos] is None:
            continue
        note = valid[order[pos]]
        note_value, divisions, dotted, is_trip = snap[pos]
        pos_label = triplet_position[pos] if is_trip else None
        differs = (
            note.get('note_value') != note_value
            or abs(float(note.get('note_divisions') or 0.0) - divisions) > 1e-6
            or bool(note.get('dotted', False)) != dotted
            or bool(note.get('triplet', False)) != is_trip
            or note.get('triplet_position') != pos_label
        )
        if not differs:
            continue
        note['note_value'] = note_value
        note['note_divisions'] = divisions
        note['dotted'] = dotted
        note['triplet'] = is_trip
        note['is_triplet'] = is_trip
        if is_trip:
            note['triplet_position'] = pos_label
            note['triplet_type'] = note_value
            note['actual_notes'] = 3
            note['normal_notes'] = 2
        else:
            note['triplet_position'] = None
            note['triplet_type'] = None
            note.pop('actual_notes', None)
            note.pop('normal_notes', None)
        note['_refined'] = True
        note['_refine_reason'] = 'ioi'
        changed += 1
    return notes, changed


# (beats -> value) candidate table; ordered fine->coarse doesn't matter, we pick
# the nearest by |beats - candidate|. Mirrors the frontend stream candidates plus
# their ternary partners so refinement and streaming share one vocabulary.
_VALUE_CANDIDATES: List[Tuple[float, str, bool, bool]] = [
    (4.0,     'whole',   False, False),
    (3.0,     'half',    True,  False),
    (8.0 / 3, 'whole',   False, True),
    (2.0,     'half',    False, False),
    (1.5,     'quarter', True,  False),
    (4.0 / 3, 'half',    False, True),
    (1.0,     'quarter', False, False),
    (0.75,    'eighth',  True,  False),
    (2.0 / 3, 'quarter', False, True),
    (0.5,     'eighth',  False, False),
    (0.375,   '16th',    True,  False),
    (1.0 / 3, 'eighth',  False, True),
    (0.25,    '16th',    False, False),
    (1.0 / 6, '16th',    False, True),
    (0.125,   '32nd',    False, False),
    (1.0 / 12, '32nd',   False, True),
]


def _snap_beats_to_value(
    beats: float, allow_triplet: bool = True
) -> Tuple[str, float, bool, bool]:
    """Nearest (note_value, beats, dotted, triplet) to a duration in beats."""
    pool = _VALUE_CANDIDATES if allow_triplet else [
        c for c in _VALUE_CANDIDATES if not c[3]
    ]
    best = min(pool, key=lambda c: abs(c[0] - beats))
    return best[1], best[0], best[2], best[3]


# ─────────────────────────────────────────────────────────────────────────────
# Deferred Refinement (Stage 2 - Background)
# ─────────────────────────────────────────────────────────────────────────────

def _apply_score_duration_policy_to_quantized_window(
    notes: List[Dict],
    bpm: float,
    grid: Optional[BeatGrid],
) -> None:
    """Replace visible score durations with policy IOIs while preserving onset decode."""
    if LIVE_SCORE_DURATION_POLICY not in ('ioi_same_hand', 'ioi_same_voice'):
        return
    if not notes:
        return

    assign_voice_ids(notes)
    beat_dur = 60.0 / max(bpm, 1.0)
    for i, note in enumerate(notes):
        next_onset = _next_policy_onset(notes, i)
        if next_onset is None:
            continue
        try:
            onset = float(note.get('time_seconds', 0.0) or 0.0)
        except (TypeError, ValueError):
            continue
        policy_duration = float(next_onset) - onset
        if policy_duration <= 0:
            continue

        start_idx = note.get('start_grid_idx')
        if grid is not None and start_idx is not None:
            try:
                units = max(1, grid.snap_idx(next_onset) - int(start_idx))
            except (TypeError, ValueError):
                units = 0
            if units > 0:
                note_type, beats, dotted, is_triplet = _units_to_musical(
                    units, grid.subdivision
                )
            else:
                note_type, beats, dotted, is_triplet = _fraction_snap(
                    policy_duration / beat_dur,
                    max_denom=8,
                )
        else:
            note_type, beats, dotted, is_triplet = _fraction_snap(
                policy_duration / beat_dur,
                max_denom=8,
            )

        note['note_value'] = note_type
        note['note_divisions'] = beats
        note['dotted'] = dotted
        note['is_triplet'] = is_triplet
        note['triplet'] = is_triplet
        note['duration_source'] = LIVE_SCORE_DURATION_POLICY
        note['score_duration_seconds'] = round(policy_duration, 4)
        note['quantization_method'] = f"refined_deferred_{LIVE_SCORE_DURATION_POLICY}"


@dataclass
class DeferredRefinementState:
    """Beats-adaptive refinement buffer. Delays and lookahead scale with BPM."""
    refinement_delay_beats: float = 2.0
    refinement_min_delay_sec: float = 0.25
    refinement_max_delay_sec: float = 4.0
    min_beats_for_refinement: float = 2.0
    min_notes_for_refinement: int = 2
    lookahead_beats: float = 2.0
    lookahead_notes_cap: int = 8

    pending_notes: List[Dict] = field(default_factory=list)
    refined_notes: List[Dict] = field(default_factory=list)
    last_refinement_time: float = 0.0
    _refinement_version: int = 0

    def __post_init__(self):
        self.pending_notes = []
        self.refined_notes = []

    def _delay_seconds(self, bpm: float) -> float:
        beat_dur = 60.0 / max(bpm, 1.0)
        d = self.refinement_delay_beats * beat_dur
        return max(self.refinement_min_delay_sec, min(d, self.refinement_max_delay_sec))

    def _lookahead_seconds(self, bpm: float) -> float:
        beat_dur = 60.0 / max(bpm, 1.0)
        return self.lookahead_beats * beat_dur

    def _meets_min_window(self, bpm: float) -> bool:
        if len(self.pending_notes) < self.min_notes_for_refinement:
            return False
        beat_dur = 60.0 / max(bpm, 1.0)
        first_t = float(self.pending_notes[0].get('time_seconds', 0.0) or 0.0)
        last_t = float(self.pending_notes[-1].get('time_seconds', 0.0) or 0.0)
        return ((last_t - first_t) / beat_dur) >= self.min_beats_for_refinement

    def add_note(self, note: Dict, current_time: float) -> None:
        note['_arrival_time'] = current_time
        note['_refined'] = False
        self.pending_notes.append(note)

    def add_notes(self, notes: List[Dict], current_time: float) -> None:
        for note in notes:
            self.add_note(note, current_time)

    def check_refinement(
        self,
        current_time: float,
        bpm: float,
        grid: Optional[BeatGrid] = None,
    ) -> Optional[List[Dict]]:
        if not self._meets_min_window(bpm):
            return None

        delay_sec = self._delay_seconds(bpm)
        cutoff_time = current_time - delay_sec
        ready_indices: List[int] = []
        for i, note in enumerate(self.pending_notes):
            arrival = note.get('_arrival_time', 0)
            if arrival <= cutoff_time and not note.get('_refined', False):
                ready_indices.append(i)

        if not ready_indices:
            return None

        if grid is None or not grid.anchored:
            grid = BeatGrid(period=60.0 / max(bpm, 1.0), anchored=False)

        last_ready = ready_indices[-1]
        ctx_start = max(0, ready_indices[0] - 4)
        ctx_end = min(len(self.pending_notes), last_ready + self.lookahead_notes_cap + 1)

        if grid.anchored:
            lookahead_sec = self._lookahead_seconds(bpm)
            last_ready_t = float(self.pending_notes[last_ready].get('time_seconds', 0.0) or 0.0)
            extended = ctx_end
            for j in range(last_ready + 1, len(self.pending_notes)):
                t_j = float(self.pending_notes[j].get('time_seconds', 0.0) or 0.0)
                if t_j - last_ready_t > lookahead_sec:
                    extended = j
                    break
            else:
                extended = len(self.pending_notes)
            ctx_end = min(len(self.pending_notes), max(ctx_end, extended))

        window = self.pending_notes[ctx_start:ctx_end]
        if len(window) < self.min_notes_for_refinement:
            return None

        ready_set = set(ready_indices)
        onsets = [float(n.get('time_seconds', 0.0) or 0.0) for n in window]
        last_dur = float(
            window[-1].get('duration_seconds', 60.0 / max(bpm, 1.0))
            or 60.0 / max(bpm, 1.0)
        )
        decode = decode_window_dual(onsets, grid, last_note_duration=last_dur)
        apply_window_decode(window, decode, quantization_method='refined_deferred')
        _apply_score_duration_policy_to_quantized_window(
            window,
            bpm,
            decode.get('grid'),
        )

        newly_refined: List[Dict] = []
        for offset, note in enumerate(window):
            absolute_idx = ctx_start + offset
            if absolute_idx in ready_set:
                note['_refined'] = True
                newly_refined.append(note)

        if newly_refined:
            self._refinement_version += 1
            self.last_refinement_time = current_time
            self.refined_notes.extend(newly_refined)

        return newly_refined if newly_refined else None

    def get_refinement_version(self) -> int:
        return self._refinement_version

    def get_next_refinement_delay_ms(
        self,
        current_time: float,
        bpm: float = 120.0,
    ) -> Optional[int]:
        if not self._meets_min_window(bpm):
            return None
        delay_sec = self._delay_seconds(bpm)
        next_due = None
        for note in self.pending_notes:
            if note.get('_refined', False):
                continue
            due_time = note.get('_arrival_time', current_time) + delay_sec
            if next_due is None or due_time < next_due:
                next_due = due_time
        if next_due is None:
            return None
        return max(0, int(math.ceil((next_due - current_time) * 1000)))

    def get_all_notes(self) -> List[Dict]:
        result = []
        refined_times = {n.get('time_seconds', 0) for n in self.refined_notes}
        for note in self.pending_notes:
            t = note.get('time_seconds', 0)
            if t in refined_times:
                for rn in self.refined_notes:
                    if abs(rn.get('time_seconds', 0) - t) < 0.001:
                        result.append(rn)
                        break
            else:
                result.append(note)
        return sorted(result, key=lambda n: n.get('time_seconds', 0))

    def clear(self):
        self.pending_notes.clear()
        self.refined_notes.clear()
        self._refinement_version = 0


# ─────────────────────────────────────────────────────────────────────────────
# Live Transcription Session
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LiveTranscriptionSession:
    session_id: str

    tempo_tracker: IncrementalTempoTracker = field(
        default_factory=IncrementalTempoTracker
    )
    refinement_state: DeferredRefinementState = field(
        default_factory=DeferredRefinementState
    )

    created_at: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    _last_notified_version: int = 0

    coarse_notes: List[Dict] = field(default_factory=list)
    coarse_chords: List[Dict] = field(default_factory=list)

    @property
    def beat_grid(self) -> BeatGrid:
        return self.tempo_tracker.beat_grid

    def grid_payload(self) -> Dict:
        g = self.beat_grid
        return {
            "phase": g.phase,
            "period": g.period,
            "subdivision": g.subdivision,
            "anchored": g.anchored,
        }

    def process_notes(self, notes: List[Dict], chords: List[Dict] = None) -> Dict:
        current_time = time.time()
        self.last_update = current_time

        timing_events = list(notes or [])
        if chords:
            timing_events.extend(chords)

        for onset in _cluster_live_onset_times(timing_events):
            self.tempo_tracker.add_onset(onset)

        bpm = self.tempo_tracker.current_bpm
        grid = self.tempo_tracker.beat_grid

        quantize_batch_coarse(notes, bpm, grid=grid)
        self.coarse_notes.extend(notes)

        if chords:
            quantize_batch_coarse(chords, bpm, grid=grid)
            self.coarse_chords.extend(chords)

        self.refinement_state.add_notes(notes, current_time)

        refined = self.refinement_state.check_refinement(current_time, bpm, grid=grid)

        needs_refresh = False
        current_version = self.refinement_state.get_refinement_version()
        if current_version > self._last_notified_version:
            needs_refresh = True
            self._last_notified_version = current_version

        return {
            'coarse_notes': notes,
            'coarse_chords': chords or [],
            'bpm': bpm,
            'bpm_confidence': self.tempo_tracker.confidence,
            'beat_grid': self.grid_payload(),
            'needs_refresh': needs_refresh,
            'refined_notes': refined,
            'refinement_version': current_version,
        }

    def get_all_notes(self) -> List[Dict]:
        return self.refinement_state.get_all_notes()

    def get_display_state(self) -> Dict[str, List[Dict]]:
        return _build_display_surface(
            self.get_all_notes(),
            self.coarse_chords,
            self.tempo_tracker.current_bpm,
        )

    def get_current_bpm(self) -> Tuple[float, float]:
        return (self.tempo_tracker.current_bpm, self.tempo_tracker.confidence)

    def get_next_refinement_delay_ms(
        self,
        current_time: Optional[float] = None,
    ) -> Optional[int]:
        effective_time = current_time if current_time is not None else time.time()
        return self.refinement_state.get_next_refinement_delay_ms(
            effective_time,
            self.tempo_tracker.current_bpm,
        )

    def force_refinement(self) -> Optional[List[Dict]]:
        bpm = self.tempo_tracker.current_bpm
        grid = self.tempo_tracker.beat_grid
        current_time = time.time()
        old_delay_beats = self.refinement_state.refinement_delay_beats
        old_min_delay = self.refinement_state.refinement_min_delay_sec
        old_min_beats = self.refinement_state.min_beats_for_refinement
        old_min_notes = self.refinement_state.min_notes_for_refinement
        try:
            self.refinement_state.refinement_delay_beats = 0.0
            self.refinement_state.refinement_min_delay_sec = 0.0
            self.refinement_state.min_beats_for_refinement = 0.0
            self.refinement_state.min_notes_for_refinement = 1
            return self.refinement_state.check_refinement(
                current_time + 10,
                bpm,
                grid=grid,
            )
        finally:
            self.refinement_state.refinement_delay_beats = old_delay_beats
            self.refinement_state.refinement_min_delay_sec = old_min_delay
            self.refinement_state.min_beats_for_refinement = old_min_beats
            self.refinement_state.min_notes_for_refinement = old_min_notes

    def reset(self):
        self.tempo_tracker.reset()
        self.refinement_state.clear()
        self.coarse_notes.clear()
        self.coarse_chords.clear()
        self._last_notified_version = 0


# ─────────────────────────────────────────────────────────────────────────────
# Session Management
# ─────────────────────────────────────────────────────────────────────────────

_live_sessions: Dict[str, LiveTranscriptionSession] = {}


def get_or_create_session(session_id: str) -> LiveTranscriptionSession:
    if session_id not in _live_sessions:
        _live_sessions[session_id] = LiveTranscriptionSession(session_id=session_id)
    return _live_sessions[session_id]


def delete_session(session_id: str) -> bool:
    if session_id in _live_sessions:
        del _live_sessions[session_id]
        return True
    return False


def cleanup_stale_sessions(max_age_seconds: float = 3600.0) -> int:
    current = time.time()
    stale = [
        sid for sid, sess in _live_sessions.items()
        if current - sess.last_update > max_age_seconds
    ]
    for sid in stale:
        del _live_sessions[sid]
    return len(stale)
