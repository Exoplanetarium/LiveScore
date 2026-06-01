"""Quick accuracy test for the trained ensemble or mel baseline model."""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

# Import rhythm quantization functions
from detect_note import compute_rhythm_coherence, detect_tempo_from_onsets
from detect_note import duration_to_note_value as dn_duration_to_note_value
from detect_note import (quantize_rhythm_from_ioi, quantize_rhythm_ml,
                         smooth_erratic_rhythm)
# decode_note_events is always from train_ensemble (mel baseline also uses it)
from rhythm_training.train_ensemble import decode_note_events

DEFAULT_STRICT_ONSET_TOLS_MS = (10, 20, 30)


def _load_model_module(model_type: str):
    """Return (MODEL_PATH, build_fn, extractor_cls, constants) for the chosen model."""
    if model_type == "mel":
        from rhythm_training.train_mel_baseline import (
            HOP_LENGTH, MIDI_OFFSET, MODEL_PATH, NOTE_VALUE_BEATS,
            NOTE_VALUE_CLASSES, NOTE_VALUE_NAMES, PIANO_KEYS, SAMPLE_RATE,
            MelFeatureExtractor, _build_model_from_config)
        return dict(
            MODEL_PATH=MODEL_PATH, build=_build_model_from_config,
            ExtractorClass=MelFeatureExtractor, extractor_kwargs={},
            HOP_LENGTH=HOP_LENGTH, MIDI_OFFSET=MIDI_OFFSET,
            NOTE_VALUE_CLASSES=NOTE_VALUE_CLASSES, NOTE_VALUE_NAMES=NOTE_VALUE_NAMES,
            NOTE_VALUE_BEATS=NOTE_VALUE_BEATS,
            PIANO_KEYS=PIANO_KEYS, SAMPLE_RATE=SAMPLE_RATE,
        )
    else:
        from rhythm_training.train_ensemble import (HOP_LENGTH, MIDI_OFFSET,
                                                    MODEL_PATH,
                                                    NOTE_VALUE_BEATS,
                                                    NOTE_VALUE_CLASSES,
                                                    NOTE_VALUE_NAMES,
                                                    PIANO_KEYS, SAMPLE_RATE,
                                                    MultiResFeatureExtractor,
                                                    _build_model_from_config)
        return dict(
            MODEL_PATH=MODEL_PATH, build=_build_model_from_config,
            ExtractorClass=MultiResFeatureExtractor, extractor_kwargs={},
            HOP_LENGTH=HOP_LENGTH, MIDI_OFFSET=MIDI_OFFSET,
            NOTE_VALUE_CLASSES=NOTE_VALUE_CLASSES, NOTE_VALUE_NAMES=NOTE_VALUE_NAMES,
            NOTE_VALUE_BEATS=NOTE_VALUE_BEATS,
            PIANO_KEYS=PIANO_KEYS, SAMPLE_RATE=SAMPLE_RATE,
        )


def load_midi_notes(midi_path, extend_with_pedal=True):
    """Load note events from MIDI file with duration info.
    
    Args:
        midi_path: Path to MIDI file
        extend_with_pedal: If True, extend note durations based on sustain pedal.
            This makes durations match acoustic duration rather than key-release.
    """
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(midi_path)
    
    # Extract sustain pedal events (CC#64)
    pedal_events = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for cc in inst.control_changes:
            if cc.number == 64:  # Sustain pedal
                pedal_events.append({
                    'time': cc.time,
                    'value': cc.value,
                    'is_down': cc.value >= 64
                })
    
    # Sort pedal events by time
    pedal_events.sort(key=lambda e: e['time'])
    
    # Build pedal state intervals: list of (start, end) when pedal is down
    pedal_intervals = []
    pedal_down_time = None
    for event in pedal_events:
        if event['is_down'] and pedal_down_time is None:
            pedal_down_time = event['time']
        elif not event['is_down'] and pedal_down_time is not None:
            pedal_intervals.append((pedal_down_time, event['time']))
            pedal_down_time = None
    # Handle pedal held to end
    if pedal_down_time is not None:
        pedal_intervals.append((pedal_down_time, float('inf')))
    
    def find_pedal_release(note_end):
        """Find when pedal releases after note_end, or return note_end if no pedal."""
        for start, end in pedal_intervals:
            if start <= note_end <= end:
                return end
        return note_end
    
    notes = []
    for inst_idx, inst in enumerate(midi.instruments):
        if inst.is_drum:
            continue
        for note in inst.notes:
            original_end = note.end

            if extend_with_pedal and pedal_intervals:
                # Extend note to pedal release if pedal is down at note end
                extended_end = find_pedal_release(note.end)
                # Cap extension to reasonable max (e.g., 10 seconds from onset)
                max_end = note.start + 10.0
                actual_end = min(extended_end, max_end)
            else:
                actual_end = note.end

            notes.append({
                'onset_time': note.start,
                'offset_time': actual_end,
                'duration': actual_end - note.start,
                'midi_note': note.pitch,
                'velocity': note.velocity,
                'original_duration': original_end - note.start,  # Keep original for debugging
                'midi_track': inst_idx,  # Preserve track for hand assignment
            })
    notes.sort(key=lambda n: (n['onset_time'], n['midi_note']))
    return notes, midi


def get_midi_tempo(midi):
    """Extract tempo from MIDI file tempo changes."""
    import pretty_midi
    tempo_times, tempos = midi.get_tempo_changes()
    if len(tempos) > 0:
        # Return the most common tempo (weighted by duration)
        if len(tempos) == 1:
            return tempos[0]
        # For multiple tempos, use the one that covers most time
        # Simple: just use the first one for now
        return tempos[0]
    return 120.0  # Default


def duration_to_note_value(duration_sec, bpm):
    """Convert a duration in seconds to a quantized note value."""
    beat_duration = 60.0 / bpm
    beats = duration_sec / beat_duration
    
    # Available note values with their beat durations
    note_values = [
        ('whole', 4.0), ('half', 2.0), ('quarter', 1.0),
        ('eighth', 0.5), ('16th', 0.25), ('32nd', 0.125),
        ('half_dotted', 3.0), ('quarter_dotted', 1.5), ('eighth_dotted', 0.75),
    ]
    
    # Find closest
    best_match = 'quarter'
    best_diff = float('inf')
    for name, val in note_values:
        diff = abs(beats - val)
        if diff < best_diff:
            best_diff = diff
            best_match = name
    
    return best_match, beats


def compute_note_metrics(pred_notes, gt_notes, onset_tol=0.05, duration_tol=0.2):
    """
    Compute precision, recall, F1 for note detection.
    A note is matched if onset is within tolerance and pitch matches.
    
    Also computes rhythm metrics: how many matched notes have correct duration.
    """
    matched = 0
    rhythm_matched = 0  # matched notes with correct rhythm
    gt_matched = set()
    matched_pairs = []  # (pred, gt) pairs for rhythm analysis
    
    for pred in pred_notes:
        for i, gt in enumerate(gt_notes):
            if i in gt_matched:
                continue
            if (abs(pred['onset_time'] - gt['onset_time']) <= onset_tol
                    and pred['midi_note'] == gt['midi_note']):
                matched += 1
                gt_matched.add(i)
                matched_pairs.append((pred, gt))
                
                # Check duration match (within tolerance ratio)
                pred_dur = pred.get('duration', pred.get('offset_time', pred['onset_time'] + 0.5) - pred['onset_time'])
                gt_dur = gt.get('duration', gt['offset_time'] - gt['onset_time'])
                if gt_dur > 0:
                    dur_ratio = min(pred_dur, gt_dur) / max(pred_dur, gt_dur)
                    if dur_ratio >= (1.0 - duration_tol):
                        rhythm_matched += 1
                break
    
    precision = matched / len(pred_notes) if pred_notes else 0
    recall = matched / len(gt_notes) if gt_notes else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Rhythm F1: of matched notes, how many have correct duration?
    rhythm_precision = rhythm_matched / matched if matched > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matched': matched,
        'predicted': len(pred_notes),
        'ground_truth': len(gt_notes),
        'rhythm_matched': rhythm_matched,
        'rhythm_precision': rhythm_precision,
        'matched_pairs': matched_pairs,
    }


def compute_onset_tolerance_sweep(pred_notes, gt_notes, onset_tolerances_ms=None):
    """Evaluate onset-only note metrics at multiple strict timing tolerances."""
    tolerances_ms = onset_tolerances_ms or DEFAULT_STRICT_ONSET_TOLS_MS
    sweep = {}
    for tol_ms in sorted({int(max(1, round(float(value)))) for value in tolerances_ms}):
        label = f'{tol_ms}ms'
        metrics = compute_note_metrics(pred_notes, gt_notes, onset_tol=tol_ms / 1000.0)
        sweep[label] = {
            'onset_tol_ms': tol_ms,
            **metrics,
        }
    return sweep


def normalize_note_value(note_type, dotted=False):
    """Normalize note value to standard format for comparison."""
    # Handle dotted suffix format from local function
    if '_dotted' in str(note_type):
        return note_type.replace('_dotted', ''), True
    return note_type, dotted


def compute_model_head_metrics(pred_notes, gt_notes, bpm, onset_tol=0.05,
                                model_nv_beats=None, model_nv_names=None):
    """Evaluate the model's note_value head directly against GT.

    Only considers notes that have a 'model_note_value' field.
    Compares against track-based IOI GT using the MODEL's own class system
    (not dn_duration_to_note_value) to avoid train/eval boundary mismatch.
    Also reports per-class accuracy breakdown.
    """
    beat_duration = 60.0 / bpm

    # Build the model's class lookup for beats
    base_note_value_beats = {
        'whole': 4.0, 'half': 2.0, 'quarter': 1.0, 'eighth': 0.5,
        '16th': 0.25, '32nd': 0.125,
    }

    # If model class system provided, use it for GT quantization
    if model_nv_beats is not None and model_nv_names is not None:
        import numpy as _np
        log_nv = _np.log2(model_nv_beats)

        def _quantize_gt(dur_sec):
            """Quantize GT duration using the model's own class boundaries."""
            ioi_beats = max(0.0625, min(8.0, dur_sec / beat_duration))
            class_idx = int(_np.argmin(_np.abs(log_nv - _np.log2(ioi_beats))))
            name = model_nv_names[class_idx]
            if name.startswith('dotted_'):
                return name[7:], True, name
            return name, False, name
    else:
        def _quantize_gt(dur_sec):
            """Fallback: use dn_duration_to_note_value."""
            result = dn_duration_to_note_value(dur_sec, bpm=bpm)
            base = result['type']
            dot = result.get('dotted', False)
            label = f"{'dotted_' if dot else ''}{base}"
            return base, dot, label

    # Build track-based IOI map for GT
    gt_sorted = sorted(gt_notes, key=lambda n: n['onset_time'])
    has_track_info = any('midi_track' in n for n in gt_sorted)
    gt_hand_indices = {}
    for i, n in enumerate(gt_sorted):
        hand = f'track_{n.get("midi_track", 0)}' if has_track_info else (
            'bass' if n['midi_note'] < 60 else 'treble')
        gt_hand_indices.setdefault(hand, []).append(i)

    gt_ioi_map = {}
    for hand, indices in gt_hand_indices.items():
        for j in range(len(indices) - 1):
            curr, nxt = gt_sorted[indices[j]], gt_sorted[indices[j + 1]]
            gt_ioi_map[(curr['onset_time'], curr['midi_note'])] = nxt['onset_time'] - curr['onset_time']

    # Match pred to GT and compare
    matched_pairs = []
    gt_matched = set()
    for pred in pred_notes:
        if 'model_note_value' not in pred:
            continue
        for i, gt in enumerate(gt_notes):
            if i in gt_matched:
                continue
            if (abs(pred['onset_time'] - gt['onset_time']) <= onset_tol
                    and pred['midi_note'] == gt['midi_note']):
                matched_pairs.append((pred, gt))
                gt_matched.add(i)
                break

    if not matched_pairs:
        return {'n_matched': 0, 'ioi_accuracy': 0.0, 'sustain_accuracy': 0.0,
                'avg_beat_error': 0.0, 'per_class': {}}

    exact_ioi = 0
    exact_sustain = 0
    beat_errors = []
    per_class_correct = {}
    per_class_total = {}
    confusion = {}  # (pred, gt) -> count

    for pred, gt in matched_pairs:
        # Parse model prediction
        raw_name = pred['model_note_value']
        if raw_name.startswith('dotted_'):
            pred_base, pred_dot = raw_name[7:], True
        else:
            pred_base, pred_dot = raw_name, False
        pred_beats = base_note_value_beats.get(pred_base, 1.0) * (1.5 if pred_dot else 1.0)

        # GT: IOI-based (track-aware) — quantized using MODEL's class system
        gt_ioi = gt_ioi_map.get((gt['onset_time'], gt['midi_note']))
        if gt_ioi is not None and gt_ioi > 0.03:
            gt_ioi_dur = min(gt_ioi, beat_duration * 6.0)
        else:
            gt_ioi_dur = gt.get('duration', gt['offset_time'] - gt['onset_time'])
        gt_base, gt_dot, gt_label = _quantize_gt(gt_ioi_dur)
        gt_beats = base_note_value_beats.get(gt_base, 1.0) * (1.5 if gt_dot else 1.0)

        # GT: sustain-based
        gt_sus_dur = gt.get('original_duration', gt.get('duration', gt['offset_time'] - gt['onset_time']))
        gt_sus_base, gt_sus_dot, _ = _quantize_gt(gt_sus_dur)

        # Exact match
        if pred_base == gt_base and pred_dot == gt_dot:
            exact_ioi += 1
        if pred_base == gt_sus_base and pred_dot == gt_sus_dot:
            exact_sustain += 1
        beat_errors.append(abs(pred_beats - gt_beats))

        # Per-class tracking
        per_class_total[gt_label] = per_class_total.get(gt_label, 0) + 1
        if pred_base == gt_base and pred_dot == gt_dot:
            per_class_correct[gt_label] = per_class_correct.get(gt_label, 0) + 1

        # Confusion
        confusion[(raw_name, gt_label)] = confusion.get((raw_name, gt_label), 0) + 1

    n = len(matched_pairs)
    per_class = {}
    for cls in sorted(per_class_total, key=lambda c: -per_class_total[c]):
        total = per_class_total[cls]
        correct = per_class_correct.get(cls, 0)
        per_class[cls] = {'correct': correct, 'total': total, 'accuracy': correct / total}

    # Top confusions
    top_confusions = sorted(confusion.items(), key=lambda x: -x[1])[:10]

    return {
        'n_matched': n,
        'ioi_accuracy': exact_ioi / n,
        'sustain_accuracy': exact_sustain / n,
        'avg_beat_error': np.mean(beat_errors) if beat_errors else 0.0,
        'per_class': per_class,
        'top_confusions': top_confusions,
    }


def compute_rhythm_metrics(pred_notes, gt_notes, bpm, onset_tol=0.05, debug=False):
    """
    Compute rhythm-specific metrics comparing quantized note values.

    Uses the `note_value` field from prediction (set by quantization pipeline)
    and quantizes GT duration for comparison.

    Reports separate metrics for:
    - IOI-based ground truth (using MIDI track for hand assignment when available)
    - Sustain-based ground truth (MIDI key-release duration)
    - Beat sum validity (do predicted note values fill valid measures?)
    """
    matched_pairs = []

    for pred in pred_notes:
        for gt in gt_notes:
            if (abs(pred['onset_time'] - gt['onset_time']) <= onset_tol
                    and pred['midi_note'] == gt['midi_note']):
                matched_pairs.append((pred, gt))
                break

    if not matched_pairs:
        return {
            'note_value_accuracy': 0.0,
            'ioi_note_value_accuracy': 0.0,
            'sustain_note_value_accuracy': 0.0,
            'avg_beat_error': 0.0,
            'avg_sustain_beat_error': 0.0,
            'rhythm_f1': 0.0,
            'n_matched': 0,
            'beat_sum_validity': 0.0,
        }

    beat_duration = 60.0 / bpm

    # --- Build per-hand IOI map using MIDI track assignment (not pitch split) ---
    gt_sorted = sorted(gt_notes, key=lambda n: n['onset_time'])

    # Determine if MIDI track info is available
    has_track_info = any('midi_track' in n for n in gt_sorted)

    gt_hand_indices = {}
    for i, n in enumerate(gt_sorted):
        if has_track_info:
            hand = f'track_{n.get("midi_track", 0)}'
        else:
            hand = 'bass' if n['midi_note'] < 60 else 'treble'
        if hand not in gt_hand_indices:
            gt_hand_indices[hand] = []
        gt_hand_indices[hand].append(i)

    gt_ioi_map = {}  # (onset_time, midi_note) -> per-hand IOI in seconds
    for hand, indices in gt_hand_indices.items():
        for j in range(len(indices) - 1):
            curr = gt_sorted[indices[j]]
            nxt = gt_sorted[indices[j + 1]]
            ioi = nxt['onset_time'] - curr['onset_time']
            gt_ioi_map[(curr['onset_time'], curr['midi_note'])] = ioi

    # --- Also build pitch-split IOI map for backward-compatible metric ---
    gt_pitch_hand_indices = {}
    for i, n in enumerate(gt_sorted):
        hand = 'bass' if n['midi_note'] < 60 else 'treble'
        if hand not in gt_pitch_hand_indices:
            gt_pitch_hand_indices[hand] = []
        gt_pitch_hand_indices[hand].append(i)

    gt_pitch_ioi_map = {}
    for hand, indices in gt_pitch_hand_indices.items():
        for j in range(len(indices) - 1):
            curr = gt_sorted[indices[j]]
            nxt = gt_sorted[indices[j + 1]]
            ioi = nxt['onset_time'] - curr['onset_time']
            gt_pitch_ioi_map[(curr['onset_time'], curr['midi_note'])] = ioi

    # Note value to beats mapping (includes dotted variants from model head)
    note_value_beats = {
        'whole': 4.0, 'half': 2.0, 'quarter': 1.0, 'eighth': 0.5,
        '16th': 0.25, '32nd': 0.125,
        'dotted_whole': 6.0, 'dotted_half': 3.0, 'dotted_quarter': 1.5,
        'dotted_eighth': 0.75, 'dotted_16th': 0.375, 'dotted_32nd': 0.1875,
    }

    def _parse_note_value(name, dotted_flag):
        """Parse a note value name into (base_type, is_dotted) for comparison.

        Handles both formats:
          - model head: 'dotted_quarter' with dotted_flag ignored
          - pipeline: 'quarter' with dotted_flag=True
        """
        if name and name.startswith('dotted_'):
            return name[len('dotted_'):], True
        return name, dotted_flag

    ioi_exact_match = 0
    sustain_exact_match = 0
    pitch_ioi_exact_match = 0
    ioi_beat_errors = []
    sustain_beat_errors = []

    if debug and matched_pairs:
        print(f"    [DEBUG] First 5 rhythm comparisons (BPM={bpm}, track-based hand={'yes' if has_track_info else 'no'}):")

    for idx, (pred, gt) in enumerate(matched_pairs):
        # --- Prediction note value (pipeline only, ignore model head) ---
        pred_val = pred.get('note_value')
        pred_dotted = pred.get('dotted', False)
        if pred_val is None:
            pred_dur = pred.get('duration', pred.get('offset_time', pred['onset_time'] + 0.5) - pred['onset_time'])
            pred_result = dn_duration_to_note_value(pred_dur, bpm=bpm)
            pred_val = pred_result['type']
            pred_dotted = pred_result.get('dotted', False)

        # Normalize combined dotted names (e.g. 'dotted_quarter' -> 'quarter', True)
        pred_val, pred_dotted = _parse_note_value(pred_val, pred_dotted)

        pred_beats_val = note_value_beats.get(pred_val, 1.0)
        if pred_dotted:
            pred_beats_val *= 1.5

        # --- GT: IOI-based (track-aware) ---
        gt_ioi = gt_ioi_map.get((gt['onset_time'], gt['midi_note']))
        if gt_ioi is not None and gt_ioi > 0.03:
            max_dur = beat_duration * 6.0
            gt_ioi_dur = min(gt_ioi, max_dur)
        else:
            gt_ioi_dur = gt.get('duration', gt['offset_time'] - gt['onset_time'])
        gt_ioi_result = dn_duration_to_note_value(gt_ioi_dur, bpm=bpm)
        gt_ioi_val = gt_ioi_result['type']
        gt_ioi_dotted = gt_ioi_result.get('dotted', False)

        gt_ioi_beats_val = note_value_beats.get(gt_ioi_val, 1.0)
        if gt_ioi_dotted:
            gt_ioi_beats_val *= 1.5

        # --- GT: IOI-based (pitch-split, backward compatible) ---
        gt_pitch_ioi = gt_pitch_ioi_map.get((gt['onset_time'], gt['midi_note']))
        if gt_pitch_ioi is not None and gt_pitch_ioi > 0.03:
            gt_pitch_dur = min(gt_pitch_ioi, beat_duration * 6.0)
        else:
            gt_pitch_dur = gt.get('duration', gt['offset_time'] - gt['onset_time'])
        gt_pitch_result = dn_duration_to_note_value(gt_pitch_dur, bpm=bpm)
        gt_pitch_val = gt_pitch_result['type']
        gt_pitch_dotted = gt_pitch_result.get('dotted', False)

        # --- GT: Sustain-based (MIDI key-release duration) ---
        gt_sustain_dur = gt.get('original_duration', gt.get('duration', gt['offset_time'] - gt['onset_time']))
        gt_sustain_result = dn_duration_to_note_value(gt_sustain_dur, bpm=bpm)
        gt_sustain_val = gt_sustain_result['type']
        gt_sustain_dotted = gt_sustain_result.get('dotted', False)

        gt_sustain_beats_val = note_value_beats.get(gt_sustain_val, 1.0)
        if gt_sustain_dotted:
            gt_sustain_beats_val *= 1.5

        # Debug output
        if debug and idx < 5:
            pred_str = f"{'dotted ' if pred_dotted else ''}{pred_val}"
            gt_ioi_str = f"{'dotted ' if gt_ioi_dotted else ''}{gt_ioi_val}"
            gt_sus_str = f"{'dotted ' if gt_sustain_dotted else ''}{gt_sustain_val}"
            m_ioi = (pred_val == gt_ioi_val and pred_dotted == gt_ioi_dotted)
            m_sus = (pred_val == gt_sustain_val and pred_dotted == gt_sustain_dotted)
            print(f"      [{idx}] pred={pred_str}({pred_beats_val:.2f}b), "
                  f"gt_ioi={gt_ioi_str}({gt_ioi_beats_val:.2f}b, match={m_ioi}), "
                  f"gt_sustain={gt_sus_str}({gt_sustain_beats_val:.2f}b, match={m_sus})")

        # Exact matches
        if pred_val == gt_ioi_val and pred_dotted == gt_ioi_dotted:
            ioi_exact_match += 1
        if pred_val == gt_sustain_val and pred_dotted == gt_sustain_dotted:
            sustain_exact_match += 1
        if pred_val == gt_pitch_val and pred_dotted == gt_pitch_dotted:
            pitch_ioi_exact_match += 1

        # Beat errors
        ioi_beat_errors.append(abs(pred_beats_val - gt_ioi_beats_val))
        sustain_beat_errors.append(abs(pred_beats_val - gt_sustain_beats_val))

    n = len(matched_pairs)
    ioi_accuracy = ioi_exact_match / n
    sustain_accuracy = sustain_exact_match / n
    pitch_ioi_accuracy = pitch_ioi_exact_match / n

    # --- Beat sum validity: do predicted note values form valid measures? ---
    # Group predicted notes by measure (4 beats per measure at the detected BPM)
    measure_duration = beat_duration * 4  # assuming 4/4 time
    measure_sums = {}
    for note in pred_notes:
        onset = note.get('onset_time', note.get('time_seconds', 0))
        measure_idx = int(onset / measure_duration)
        nv = note.get('note_value', 'quarter')
        beats = note_value_beats.get(nv, 1.0)
        if note.get('dotted', False):
            beats *= 1.5
        if note.get('triplet', False):
            beats *= 2.0 / 3.0
        measure_sums[measure_idx] = measure_sums.get(measure_idx, 0.0) + beats

    # A valid measure sums to exactly 4 beats (in 4/4). Allow 10% tolerance.
    valid_measures = 0
    total_measures = len(measure_sums)
    for m_idx, total_beats in measure_sums.items():
        if abs(total_beats - 4.0) <= 0.4:  # within 10% of 4 beats
            valid_measures += 1
    beat_sum_validity = valid_measures / total_measures if total_measures > 0 else 0.0

    return {
        # Primary: IOI-based with track-aware hand split (decoupled from prediction)
        'note_value_accuracy': ioi_accuracy,
        'ioi_note_value_accuracy': ioi_accuracy,
        # Backward-compatible: IOI-based with pitch-split (same bias as prediction)
        'pitch_split_ioi_accuracy': pitch_ioi_accuracy,
        # Independent: sustain-based (MIDI key-release, no IOI dependency)
        'sustain_note_value_accuracy': sustain_accuracy,
        'avg_beat_error': np.mean(ioi_beat_errors) if ioi_beat_errors else 0.0,
        'avg_sustain_beat_error': np.mean(sustain_beat_errors) if sustain_beat_errors else 0.0,
        'rhythm_f1': ioi_accuracy,
        'n_matched': n,
        'beat_sum_validity': beat_sum_validity,
    }


def apply_rhythm_quantization(notes, bpm, use_coherence=True, debug=False):
    """
    Apply rhythm quantization pipeline to predicted notes.
    
    Args:
        notes: List of note dicts with onset_time, offset_time, midi_note
        bpm: Detected tempo
        use_coherence: Whether to apply coherence smoothing
        debug: Print debug info
        
    Returns:
        Notes with quantized rhythm info added
    """
    if not notes:
        return notes
    
    # Convert to format expected by quantize functions
    for note in notes:
        note['time_seconds'] = note['onset_time']
        note['duration_seconds'] = note.get('duration', 
            note.get('offset_time', note['onset_time'] + 0.5) - note['onset_time'])
        note['pitch'] = note['midi_note']
        # Assign hand based on pitch (rough split at middle C = 60)
        note['hand'] = 'bass' if note['midi_note'] < 60 else 'treble'
    
    # Apply ML-based quantization (production path: Transformer → MLP → heuristic fallback)
    quantized = quantize_rhythm_ml(notes.copy(), bpm, debug=debug)
    
    if use_coherence:
        # Apply coherence analysis and smoothing
        coherence_info = compute_rhythm_coherence(quantized, window_size=8, bpm=bpm, debug=debug)
        if coherence_info['erratic_indices']:
            quantized = smooth_erratic_rhythm(quantized, coherence_info, bpm, debug=debug)
    
    # Copy back quantization results
    for i, note in enumerate(notes):
        if i < len(quantized):
            note['note_value'] = quantized[i].get('note_value', 'quarter')
            note['note_divisions'] = quantized[i].get('note_divisions', 1.0)
            note['dotted'] = quantized[i].get('dotted', False)
            note['coherence_smoothed'] = 'coherence-smoothed' in quantized[i].get('quantization_method', '')
            # Copy acoustic-based values for comparison
            note['acoustic_note_value'] = quantized[i].get('acoustic_note_value', note['note_value'])
            note['acoustic_dotted'] = quantized[i].get('acoustic_dotted', note.get('dotted', False))
    
    return notes


def test_on_sample(model_type: str = "ensemble", strict_onset_tols_ms=None):
    """Test model on a sample from MAESTRO test set."""
    import librosa

    mod = _load_model_module(model_type)
    MODEL_PATH = mod['MODEL_PATH']
    SAMPLE_RATE = mod['SAMPLE_RATE']
    HOP_LENGTH = mod['HOP_LENGTH']
    MIDI_OFFSET = mod['MIDI_OFFSET']
    PIANO_KEYS = mod['PIANO_KEYS']
    NOTE_VALUE_NAMES = mod['NOTE_VALUE_NAMES']

    # Load test index
    index_path = os.path.join(os.path.dirname(__file__),
                              'rhythm_training', 'ensemble_index', 'test_index.json')

    if not os.path.exists(index_path):
        print("Test index not found. Run: python train_ensemble.py --prepare")
        return

    with open(index_path) as f:
        test_idx = json.load(f)

    pieces = test_idx['pieces']
    if not pieces:
        print("No test pieces found!")
        return

    print(f"Found {len(pieces)} test pieces")
    print(f"Model type: {model_type}")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        return

    checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    config = checkpoint.get('config', {})

    print(f"Model info:")
    print(f"  - Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  - Val loss: {checkpoint.get('val_loss', '?'):.4f}")
    print(f"  - Onset F1 (validation): {checkpoint.get('onset_f1', '?'):.3f}")
    print(f"  - Frame F1 (validation): {checkpoint.get('frame_f1', '?'):.3f}")

    # Build extractor based on model type
    if model_type == "mel":
        extractor = mod['ExtractorClass'](
            sr=config.get('sample_rate', SAMPLE_RATE),
            hop_length=config.get('hop_length', HOP_LENGTH),
            device=device,
        )
    else:
        extractor = mod['ExtractorClass'](
            sr=config.get('sample_rate', SAMPLE_RATE),
            hop_length=config.get('hop_length', HOP_LENGTH),
            device=device,
            hop_lengths=config.get('hop_lengths', None),
        )

    model = mod['build'](config)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.to(device)
    model.eval()
    
    # Test on first 3 pieces (first 30 seconds each)
    n_test = min(3, len(pieces))
    all_metrics = []
    
    for i in range(n_test):
        piece = pieces[i]
        audio_path = piece['audio']
        midi_path = piece['midi']
        
        if not os.path.exists(audio_path):
            print(f"Audio not found: {audio_path}")
            continue
        
        print(f"\n[{i+1}/{n_test}] {piece.get('title', 'Unknown')[:50]}...")
        
        # Load 30 seconds of audio
        test_duration = 30.0
        sr = config.get('sample_rate', SAMPLE_RATE)
        
        audio, _ = librosa.load(audio_path, sr=sr, mono=True, duration=test_duration)
        print(f"  Audio: {len(audio)/sr:.1f}s @ {sr}Hz")
        
        # Load ground truth MIDI (filter to same time range)
        gt_notes_all, midi_obj = load_midi_notes(midi_path, extend_with_pedal=False)
        gt_notes = [n for n in gt_notes_all if n['onset_time'] < test_duration]
        gt_bpm = get_midi_tempo(midi_obj)
        print(f"  Ground truth: {len(gt_notes)} notes, BPM={gt_bpm:.1f}")
        
        # Run inference
        audio_t = torch.from_numpy(audio).float().to(device)
        
        with torch.no_grad():
            features = extractor.extract(audio_t)  # (1, T, 373)
            out = model(features)

            onset_p = torch.sigmoid(out['onset_logits'][0]).cpu().numpy()
            frame_p = torch.sigmoid(out['frame_logits'][0]).cpu().numpy()
            velocity = out['velocity'][0].cpu().numpy()
            # Note-value logits: (T, 88, 6) if head exists
            nv_logits = out.get('note_value_logits')
            if nv_logits is not None:
                nv_logits = nv_logits[0].cpu().numpy()  # (T, 88, 6)
            # Neural tempo prediction
            neural_bpm = out['tempo_bpm'][0].item() if 'tempo_bpm' in out else None
        
        # Decode notes - use raw frame-based duration (no IOI extension)
        # Lower frame threshold to 0.1 to extend durations for sustained notes
        pred_notes = decode_note_events(
            onset_p, frame_p, velocity,
            sr=sr, hop=config.get('hop_length', HOP_LENGTH),
            onset_threshold=0.7,
            frame_threshold=0.1,  # Very low to catch sustained notes
            min_note_duration=0.05,
            min_velocity=15,
            use_peak_picking=True,
            filter_harmonics=True,
            extend_to_next_onset=False,  # Use frame-based duration only
        )

        # Attach model-predicted note values directly (bypasses quantization)
        has_nv_head = nv_logits is not None
        if has_nv_head:
            hop_sec = config.get('hop_length', HOP_LENGTH) / sr
            for note in pred_notes:
                onset_frame = int(note['onset_time'] / hop_sec)
                key = note['midi_note'] - MIDI_OFFSET
                if 0 <= onset_frame < nv_logits.shape[0] and 0 <= key < PIANO_KEYS:
                    class_idx = int(nv_logits[onset_frame, key].argmax())
                    note['model_note_value'] = NOTE_VALUE_NAMES[class_idx]
                    note['model_note_value_class'] = class_idx

        print(f"  Predicted: {len(pred_notes)} notes"
              f"{' (with model note-values)' if has_nv_head else ''}")
        
        # Compute basic onset+pitch metrics
        metrics = compute_note_metrics(pred_notes, gt_notes, onset_tol=0.05)
        strict_onset_metrics = compute_onset_tolerance_sweep(
            pred_notes,
            gt_notes,
            onset_tolerances_ms=strict_onset_tols_ms,
        )
        
        print(f"  Note Detection:")
        print(f"    Precision: {metrics['precision']:.3f}")
        print(f"    Recall:    {metrics['recall']:.3f}")
        print(f"    F1:        {metrics['f1']:.3f}")
        print(
            "  Strict Onset F1: "
            + " ".join(
                f"{label}={strict_metrics['f1']:.3f}"
                for label, strict_metrics in strict_onset_metrics.items()
            )
        )
        
        # Detect BPM from onsets for rhythm quantization
        onset_times = [n['onset_time'] for n in pred_notes]
        vel_list = [n.get('velocity', 64) for n in pred_notes]
        tempo_result = detect_tempo_from_onsets(onset_times, velocities=vel_list)
        detected_bpm = tempo_result['bpm']
        print(f"  Detected BPM (DSP): {detected_bpm:.1f} (GT: {gt_bpm:.1f}, confidence: {tempo_result['confidence']:.2f})")
        if neural_bpm is not None:
            print(f"  Detected BPM (Neural): {neural_bpm:.1f} (GT: {gt_bpm:.1f}, error: {abs(neural_bpm - gt_bpm):.1f})")
        
        # Test rhythm quantization with DETECTED BPM (no coherence)
        import copy
        pred_no_coherence = copy.deepcopy(pred_notes)
        pred_no_coherence = apply_rhythm_quantization(pred_no_coherence, detected_bpm, use_coherence=False)
        rhythm_metrics_no_coh = compute_rhythm_metrics(pred_no_coherence, gt_notes, gt_bpm, debug=(i == 0))
        
        # Test rhythm quantization with DETECTED BPM (with coherence)
        pred_with_coherence = copy.deepcopy(pred_notes)
        pred_with_coherence = apply_rhythm_quantization(pred_with_coherence, detected_bpm, use_coherence=True)
        rhythm_metrics_coh = compute_rhythm_metrics(pred_with_coherence, gt_notes, gt_bpm, debug=False)
        
        # Test rhythm quantization with GROUND TRUTH BPM
        pred_gt_bpm = copy.deepcopy(pred_notes)
        pred_gt_bpm = apply_rhythm_quantization(pred_gt_bpm, gt_bpm, use_coherence=True)
        rhythm_metrics_gt = compute_rhythm_metrics(pred_gt_bpm, gt_notes, gt_bpm)

        # Test rhythm quantization with NEURAL BPM (if available)
        rhythm_metrics_neural = None
        if neural_bpm is not None:
            pred_neural_bpm = copy.deepcopy(pred_notes)
            pred_neural_bpm = apply_rhythm_quantization(pred_neural_bpm, neural_bpm, use_coherence=True)
            rhythm_metrics_neural = compute_rhythm_metrics(pred_neural_bpm, gt_notes, gt_bpm)
        
        # Count how many notes were smoothed
        n_smoothed = sum(1 for n in pred_with_coherence if n.get('coherence_smoothed', False))
        
        # Compute coherence scores
        coherence_no_coh = compute_rhythm_coherence(pred_no_coherence, window_size=8, bpm=detected_bpm)
        coherence_coh = compute_rhythm_coherence(pred_with_coherence, window_size=8, bpm=detected_bpm)
        
        print(f"  Rhythm (detected BPM={detected_bpm:.0f}, no coherence):")
        print(f"    IOI accuracy (track):   {rhythm_metrics_no_coh['ioi_note_value_accuracy']:.3f}")
        print(f"    IOI accuracy (pitch60): {rhythm_metrics_no_coh['pitch_split_ioi_accuracy']:.3f}")
        print(f"    Sustain accuracy:       {rhythm_metrics_no_coh['sustain_note_value_accuracy']:.3f}")
        print(f"    Avg beat error:         {rhythm_metrics_no_coh['avg_beat_error']:.3f}")
        print(f"    Beat sum validity:      {rhythm_metrics_no_coh['beat_sum_validity']:.3f}")
        print(f"    Coherence score:        {coherence_no_coh['global_coherence']:.3f}")
        print(f"    Erratic notes:          {len(coherence_no_coh['erratic_indices'])}")

        print(f"  Rhythm (detected BPM={detected_bpm:.0f}, WITH coherence):")
        print(f"    IOI accuracy (track):   {rhythm_metrics_coh['ioi_note_value_accuracy']:.3f}")
        print(f"    IOI accuracy (pitch60): {rhythm_metrics_coh['pitch_split_ioi_accuracy']:.3f}")
        print(f"    Sustain accuracy:       {rhythm_metrics_coh['sustain_note_value_accuracy']:.3f}")
        print(f"    Avg beat error:         {rhythm_metrics_coh['avg_beat_error']:.3f}")
        print(f"    Beat sum validity:      {rhythm_metrics_coh['beat_sum_validity']:.3f}")
        print(f"    Coherence score:        {coherence_coh['global_coherence']:.3f}")
        print(f"    Notes smoothed:         {n_smoothed}")

        print(f"  Rhythm (GROUND TRUTH BPM={gt_bpm:.0f}):")
        print(f"    IOI accuracy (track):   {rhythm_metrics_gt['ioi_note_value_accuracy']:.3f}")
        print(f"    Sustain accuracy:       {rhythm_metrics_gt['sustain_note_value_accuracy']:.3f}")
        print(f"    Avg beat error:         {rhythm_metrics_gt['avg_beat_error']:.3f}")
        print(f"    Beat sum validity:      {rhythm_metrics_gt['beat_sum_validity']:.3f}")

        if rhythm_metrics_neural is not None:
            print(f"  Rhythm (NEURAL BPM={neural_bpm:.0f}):")
            print(f"    IOI accuracy (track):   {rhythm_metrics_neural['ioi_note_value_accuracy']:.3f}")
            print(f"    Sustain accuracy:       {rhythm_metrics_neural['sustain_note_value_accuracy']:.3f}")
            print(f"    Avg beat error:         {rhythm_metrics_neural['avg_beat_error']:.3f}")
            print(f"    Beat sum validity:      {rhythm_metrics_neural['beat_sum_validity']:.3f}")

        # Compute improvement
        acc_diff = rhythm_metrics_coh['note_value_accuracy'] - rhythm_metrics_no_coh['note_value_accuracy']
        coh_diff = coherence_coh['global_coherence'] - coherence_no_coh['global_coherence']
        gt_improvement = rhythm_metrics_gt['note_value_accuracy'] - rhythm_metrics_coh['note_value_accuracy']
        print(f"  Improvement:")
        print(f"    Coherence smoothing:  {acc_diff:+.3f}")
        print(f"    GT BPM vs detected:   {gt_improvement:+.3f}")
        if rhythm_metrics_neural is not None:
            neural_improvement = rhythm_metrics_neural['note_value_accuracy'] - rhythm_metrics_coh['note_value_accuracy']
            print(f"    Neural BPM vs DSP:    {neural_improvement:+.3f}")

        # Model note_value head evaluation (zero-cost predictions from forward pass)
        model_head_metrics = None
        if has_nv_head:
            model_head_metrics = compute_model_head_metrics(
                pred_notes, gt_notes, gt_bpm,
                model_nv_beats=mod.get('NOTE_VALUE_BEATS'),
                model_nv_names=mod.get('NOTE_VALUE_NAMES'),
            )
            print(f"  MODEL NOTE-VALUE HEAD (direct, no pipeline):")
            print(f"    IOI accuracy (track):   {model_head_metrics['ioi_accuracy']:.3f}")
            print(f"    Sustain accuracy:       {model_head_metrics['sustain_accuracy']:.3f}")
            print(f"    Avg beat error:         {model_head_metrics['avg_beat_error']:.3f}")
            print(f"    Notes evaluated:        {model_head_metrics['n_matched']}")
            if model_head_metrics['per_class']:
                print(f"    Per-class accuracy:")
                for cls, info in model_head_metrics['per_class'].items():
                    print(f"      {cls:>20s}: {info['accuracy']:.3f}  ({info['correct']}/{info['total']})")
            if model_head_metrics.get('top_confusions'):
                print(f"    Top confusions (pred -> gt):")
                for (p, g), count in model_head_metrics['top_confusions'][:5]:
                    print(f"      {p:>20s} -> {g:<20s}: {count}")

        # Store metrics
        metrics['rhythm_no_coherence'] = rhythm_metrics_no_coh
        metrics['rhythm_with_coherence'] = rhythm_metrics_coh
        metrics['rhythm_gt_bpm'] = rhythm_metrics_gt
        metrics['rhythm_neural_bpm'] = rhythm_metrics_neural
        metrics['coherence_before'] = coherence_no_coh['global_coherence']
        metrics['coherence_after'] = coherence_coh['global_coherence']
        metrics['n_smoothed'] = n_smoothed
        metrics['detected_bpm'] = detected_bpm
        metrics['neural_bpm'] = neural_bpm
        metrics['gt_bpm'] = gt_bpm
        metrics['model_head'] = model_head_metrics
        metrics['strict_onset_metrics'] = strict_onset_metrics
        all_metrics.append(metrics)
    
    # Average metrics
    if all_metrics:
        print("\n" + "=" * 60)
        print("OVERALL RESULTS")
        print("=" * 60)
        avg_p = np.mean([m['precision'] for m in all_metrics])
        avg_r = np.mean([m['recall'] for m in all_metrics])
        avg_f1 = np.mean([m['f1'] for m in all_metrics])
        print(f"Note Detection:")
        print(f"  Avg Precision: {avg_p:.3f}")
        print(f"  Avg Recall:    {avg_r:.3f}")
        print(f"  Avg F1:        {avg_f1:.3f}")
        strict_labels = sorted(
            {
                label
                for metrics in all_metrics
                for label in (metrics.get('strict_onset_metrics') or {}).keys()
            },
            key=lambda label: int(label.rstrip('ms')),
        )
        if strict_labels:
            strict_parts = []
            for label in strict_labels:
                values = [
                    metrics['strict_onset_metrics'][label]['f1']
                    for metrics in all_metrics
                    if label in metrics.get('strict_onset_metrics', {})
                ]
                if values:
                    strict_parts.append(f"{label}={np.mean(values):.3f}")
            if strict_parts:
                print(f"  Strict Onset F1: {' '.join(strict_parts)}")
        
        # Rhythm metrics
        avg_ioi_no_coh = np.mean([m['rhythm_no_coherence']['ioi_note_value_accuracy'] for m in all_metrics])
        avg_ioi_coh = np.mean([m['rhythm_with_coherence']['ioi_note_value_accuracy'] for m in all_metrics])
        avg_ioi_gt = np.mean([m['rhythm_gt_bpm']['ioi_note_value_accuracy'] for m in all_metrics])
        avg_pitch_no_coh = np.mean([m['rhythm_no_coherence']['pitch_split_ioi_accuracy'] for m in all_metrics])
        avg_pitch_coh = np.mean([m['rhythm_with_coherence']['pitch_split_ioi_accuracy'] for m in all_metrics])
        avg_sustain_no_coh = np.mean([m['rhythm_no_coherence']['sustain_note_value_accuracy'] for m in all_metrics])
        avg_sustain_coh = np.mean([m['rhythm_with_coherence']['sustain_note_value_accuracy'] for m in all_metrics])
        avg_sustain_gt = np.mean([m['rhythm_gt_bpm']['sustain_note_value_accuracy'] for m in all_metrics])
        avg_err_no_coh = np.mean([m['rhythm_no_coherence']['avg_beat_error'] for m in all_metrics])
        avg_err_coh = np.mean([m['rhythm_with_coherence']['avg_beat_error'] for m in all_metrics])
        avg_err_gt = np.mean([m['rhythm_gt_bpm']['avg_beat_error'] for m in all_metrics])
        avg_bsv_no_coh = np.mean([m['rhythm_no_coherence']['beat_sum_validity'] for m in all_metrics])
        avg_bsv_coh = np.mean([m['rhythm_with_coherence']['beat_sum_validity'] for m in all_metrics])
        avg_bsv_gt = np.mean([m['rhythm_gt_bpm']['beat_sum_validity'] for m in all_metrics])
        avg_c_before = np.mean([m['coherence_before'] for m in all_metrics])
        avg_c_after = np.mean([m['coherence_after'] for m in all_metrics])
        total_smoothed = sum(m['n_smoothed'] for m in all_metrics)

        print(f"\nRhythm Quantization:")
        print(f"  Detected BPM (no coherence):")
        print(f"    IOI accuracy (track):   {avg_ioi_no_coh:.3f}")
        print(f"    IOI accuracy (pitch60): {avg_pitch_no_coh:.3f}")
        print(f"    Sustain accuracy:       {avg_sustain_no_coh:.3f}")
        print(f"    Avg beat error:         {avg_err_no_coh:.3f}")
        print(f"    Beat sum validity:      {avg_bsv_no_coh:.3f}")
        print(f"    Coherence score:        {avg_c_before:.3f}")
        print(f"  Detected BPM (WITH coherence):")
        print(f"    IOI accuracy (track):   {avg_ioi_coh:.3f}")
        print(f"    IOI accuracy (pitch60): {avg_pitch_coh:.3f}")
        print(f"    Sustain accuracy:       {avg_sustain_coh:.3f}")
        print(f"    Avg beat error:         {avg_err_coh:.3f}")
        print(f"    Beat sum validity:      {avg_bsv_coh:.3f}")
        print(f"    Coherence score:        {avg_c_after:.3f}")
        print(f"    Total notes smoothed:   {total_smoothed}")
        print(f"  GROUND TRUTH BPM:")
        print(f"    IOI accuracy (track):   {avg_ioi_gt:.3f}")
        print(f"    Sustain accuracy:       {avg_sustain_gt:.3f}")
        print(f"    Avg beat error:         {avg_err_gt:.3f}")
        print(f"    Beat sum validity:      {avg_bsv_gt:.3f}")

        # Neural BPM results (if model has tempo head)
        neural_results = [m for m in all_metrics if m.get('rhythm_neural_bpm') is not None]
        if neural_results:
            avg_ioi_neural = np.mean([m['rhythm_neural_bpm']['ioi_note_value_accuracy'] for m in neural_results])
            avg_err_neural = np.mean([m['rhythm_neural_bpm']['avg_beat_error'] for m in neural_results])
            avg_bsv_neural = np.mean([m['rhythm_neural_bpm']['beat_sum_validity'] for m in neural_results])
            avg_bpm_err = np.mean([abs(m['neural_bpm'] - m['gt_bpm']) for m in neural_results])
            avg_dsp_err = np.mean([abs(m['detected_bpm'] - m['gt_bpm']) for m in all_metrics])
            print(f"  NEURAL BPM:")
            print(f"    IOI accuracy (track):   {avg_ioi_neural:.3f}")
            print(f"    Avg beat error:         {avg_err_neural:.3f}")
            print(f"    Beat sum validity:      {avg_bsv_neural:.3f}")
            print(f"    Avg BPM error (neural): {avg_bpm_err:.1f}")
            print(f"    Avg BPM error (DSP):    {avg_dsp_err:.1f}")

        print(f"\nCOHERENCE SMOOTHING IMPACT:")
        acc_improvement = (avg_ioi_coh - avg_ioi_no_coh) * 100
        coh_improvement = (avg_c_after - avg_c_before) * 100
        gt_improvement = (avg_ioi_gt - avg_ioi_coh) * 100
        print(f"  IOI accuracy (track): {acc_improvement:+.1f}% points")
        print(f"  Coherence score:      {coh_improvement:+.1f}% points")
        print(f"\nGROUND TRUTH BPM IMPROVEMENT:")
        print(f"  vs detected BPM:      {gt_improvement:+.1f}% points")
        print(f"\nEVALUATION BIAS CHECK:")
        bias_no_coh = (avg_pitch_no_coh - avg_ioi_no_coh) * 100
        bias_coh = (avg_pitch_coh - avg_ioi_coh) * 100
        print(f"  Pitch-split vs track-split IOI gap (no coh):   {bias_no_coh:+.1f}% points")
        print(f"  Pitch-split vs track-split IOI gap (with coh): {bias_coh:+.1f}% points")
        ioi_vs_sustain = (avg_ioi_coh - avg_sustain_coh) * 100
        print(f"  IOI vs sustain accuracy gap (with coh):        {ioi_vs_sustain:+.1f}% points")

        # Model head results
        head_results = [m for m in all_metrics if m.get('model_head') is not None]
        if head_results:
            avg_head_ioi = np.mean([m['model_head']['ioi_accuracy'] for m in head_results])
            avg_head_sustain = np.mean([m['model_head']['sustain_accuracy'] for m in head_results])
            avg_head_err = np.mean([m['model_head']['avg_beat_error'] for m in head_results])
            pipeline_vs_head = (avg_head_ioi - avg_ioi_coh) * 100

            print(f"\nMODEL NOTE-VALUE HEAD (zero-cost, from forward pass):")
            print(f"  Avg IOI accuracy (track):   {avg_head_ioi:.3f}")
            print(f"  Avg sustain accuracy:       {avg_head_sustain:.3f}")
            print(f"  Avg beat error:             {avg_head_err:.3f}")
            print(f"  vs pipeline (IOI track):    {pipeline_vs_head:+.1f}% points")

            # Aggregate per-class
            all_cls_correct = {}
            all_cls_total = {}
            for m in head_results:
                for cls, info in m['model_head']['per_class'].items():
                    all_cls_total[cls] = all_cls_total.get(cls, 0) + info['total']
                    all_cls_correct[cls] = all_cls_correct.get(cls, 0) + info['correct']
            print(f"  Per-class accuracy (aggregated):")
            for cls in sorted(all_cls_total, key=lambda c: -all_cls_total[c]):
                total = all_cls_total[cls]
                correct = all_cls_correct.get(cls, 0)
                print(f"    {cls:>20s}: {correct/total:.3f}  ({correct}/{total})")

            # Aggregate top confusions
            all_conf = {}
            for m in head_results:
                for (p, g), count in m['model_head'].get('top_confusions', []):
                    all_conf[(p, g)] = all_conf.get((p, g), 0) + count
            top_conf = sorted(all_conf.items(), key=lambda x: -x[1])[:10]
            if top_conf:
                print(f"  Top confusions (pred -> gt):")
                for (p, g), count in top_conf:
                    print(f"    {p:>20s} -> {g:<20s}: {count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test transcription model accuracy")
    parser.add_argument('--model', choices=['ensemble', 'mel'], default='ensemble',
                        help='Which model to test (default: ensemble)')
    parser.add_argument('--strict-onset-tols-ms', nargs='*', type=int,
                        default=list(DEFAULT_STRICT_ONSET_TOLS_MS),
                        help='Strict onset tolerances in milliseconds for live-paper comparisons (default: 10 20 30)')
    args = parser.parse_args()
    test_on_sample(model_type=args.model, strict_onset_tols_ms=args.strict_onset_tols_ms)
