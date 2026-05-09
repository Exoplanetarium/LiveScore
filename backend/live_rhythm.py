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
import time
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Tuple

import numpy as np

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

        best_bpm = peak_bpm
        best_score = self._alignment_score(iois, 60.0 / peak_bpm)
        for mult in [0.5, 2.0]:
            alt_bpm = peak_bpm * mult
            if self.min_bpm <= alt_bpm <= self.max_bpm:
                alt_score = self._alignment_score(iois, 60.0 / alt_bpm)
                if alt_score > best_score * 1.05:
                    best_bpm = alt_bpm
                    best_score = alt_score

        alpha = 0.3 if self.confidence > 0.5 else 0.1
        self.current_bpm = self.current_bpm * (1 - alpha) + best_bpm * alpha
        common_tempos = [60, 72, 80, 90, 100, 108, 120, 132, 140, 160, 180, 200]
        for ct in common_tempos:
            if abs(self.current_bpm - ct) < 3:
                self.current_bpm = ct
                break

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
    note['start_beat'] = idx / max(grid.subdivision, 1)
    note['grid_subdivision'] = grid.subdivision


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
    for i, note in enumerate(notes):
        prev_notes = notes[max(0, i - 4):i] if i > 0 else None
        next_onset = None
        if i + 1 < len(notes):
            try:
                next_onset = float(notes[i + 1].get('time_seconds', 0.0) or 0.0)
            except (TypeError, ValueError):
                next_onset = None
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
        note['start_beat'] = idx / max(sub, 1)
        note['grid_subdivision'] = sub
        observed = float(note.get('time_seconds', 0.0) or 0.0)
        target_t = grid.time_at_idx(idx)
        timing_err = abs(observed - target_t)
        note['quantization_confidence'] = max(
            0.55, 1.0 - timing_err / max(grid.period, 0.01)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Deferred Refinement (Stage 2 - Background)
# ─────────────────────────────────────────────────────────────────────────────

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

        for note in notes:
            onset = note.get('time_seconds', 0)
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
        self.refinement_state.refinement_delay_beats = 0.0
        self.refinement_state.refinement_min_delay_sec = 0.0
        self.refinement_state.min_beats_for_refinement = 0.0
        refined = self.refinement_state.check_refinement(current_time + 10, bpm, grid=grid)
        self.refinement_state.refinement_delay_beats = old_delay_beats
        self.refinement_state.refinement_min_delay_sec = old_min_delay
        self.refinement_state.min_beats_for_refinement = old_min_beats
        return refined

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
