import os
import warnings
from copy import deepcopy

warnings.filterwarnings('ignore', module='librosa.*')

# consistency between local and server
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # macOS, harmless elsewhere
os.environ.setdefault("BLIS_NUM_THREADS", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ["OPENBLAS_CORETYPE"] = "HASWELL" 

#! DIAGNOSTICS
import math
from fractions import Fraction

import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
from numba import njit
from scipy.optimize import nnls
from scipy.signal import butter, get_window
from scipy.signal import istft as scipy_istft
from scipy.signal import medfilt, resample_poly, sosfiltfilt
from scipy.signal import stft as scipy_stft

#* ─── GPU Acceleration ─────────────────────────────────────────────────────────
try:
    import torch
    _HAS_TORCH = True
    _CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    _HAS_TORCH = False
    _CUDA_AVAILABLE = False

USE_GPU = _HAS_TORCH and _CUDA_AVAILABLE

if USE_GPU:
    try:
        from gpu_ops import USE_GPU as _GPU_OPS_READY
        from gpu_ops import (fused_noise_reduce,
                             get_gpu_enhanced_mel_transcriber,
                             get_gpu_enhanced_mel_transcriber_status,
                             get_gpu_mel_baseline_transcriber,
                             get_gpu_mel_baseline_transcriber_status,
                             get_gpu_rhythm_model, get_gpu_transcriber,
                             get_gpu_transcriber_status,
                             get_gpu_transformer_model,
                             gpu_batch_multiband_gate, gpu_compute_stft_once,
                             gpu_cqt, gpu_extract_features,
                             gpu_extract_features_v2, gpu_magnitude_and_flux,
                             gpu_multiband_spectral_gate,
                             parallel_process_hands, print_gpu_info)
        USE_GPU = _GPU_OPS_READY
        print_gpu_info()
    except ImportError as e:
        print(f"[GPU] gpu_ops import failed: {e}, using CPU fallback")
        USE_GPU = False

#* ─── Note Value Constants ────────────────────────────────────────────────────
NOTE_VALUE_BEATS = {
    'whole': 4.0, 'half': 2.0, 'quarter': 1.0, 'eighth': 0.5,
    '16th': 0.25, '32nd': 0.125,
}
NOTE_VALUE_BEATS_WITH_SUBDIVISIONS = {
    **NOTE_VALUE_BEATS,
    'quarter_triplet': 2/3, 'eighth_triplet': 1/3, '16th_triplet': 1/6,
    'dotted_half': 3.0, 'dotted_quarter': 1.5, 'dotted_eighth': 0.75, 'dotted_16th': 0.375,
}
NOTE_VALUES_LIST = [
    ('whole', 4.0), ('half', 2.0), ('quarter', 1.0),
    ('eighth', 0.5), ('16th', 0.25), ('32nd', 0.125),
]

# ─── Fraction-based quantization table ───────────────────────────────────────
# Maps musical beat fractions to (note_type, beats, dotted, is_triplet).
# Used by fraction_quantize() for continuous, threshold-free note assignment.
MUSICAL_FRACTIONS = {
    Fraction(1, 8):  ('32nd',    0.125,  False, False),
    Fraction(3, 16): ('32nd',    0.1875, True,  False),
    Fraction(1, 6):  ('16th',    1/6,    False, True),   # 16th triplet
    Fraction(1, 4):  ('16th',    0.25,   False, False),
    Fraction(1, 3):  ('eighth',  1/3,    False, True),   # eighth triplet
    Fraction(3, 8):  ('16th',    0.375,  True,  False),
    Fraction(1, 2):  ('eighth',  0.5,    False, False),
    Fraction(2, 3):  ('quarter', 2/3,    False, True),   # quarter triplet
    Fraction(3, 4):  ('eighth',  0.75,   True,  False),
    Fraction(1, 1):  ('quarter', 1.0,    False, False),
    Fraction(4, 3):  ('half',    4/3,    False, True),   # half triplet
    Fraction(3, 2):  ('quarter', 1.5,    True,  False),
    Fraction(2, 1):  ('half',    2.0,    False, False),
    Fraction(3, 1):  ('half',    3.0,    True,  False),
    Fraction(4, 1):  ('whole',   4.0,    False, False),
    Fraction(6, 1):  ('whole',   6.0,    True,  False),
}

# Pre-sorted list of (fraction_float, fraction_obj) for fast nearest-lookup
_MUSICAL_FRAC_SORTED = sorted(
    [(float(f), f) for f in MUSICAL_FRACTIONS.keys()],
    key=lambda x: x[0]
)
_MUSICAL_FRAC_VALS = np.array([x[0] for x in _MUSICAL_FRAC_SORTED])
_MUSICAL_FRAC_KEYS = [x[1] for x in _MUSICAL_FRAC_SORTED]


def _normalize_ensemble_note_value(note_dict):
    """Convert ensemble dotted names ('dotted_eighth') to standard name + dotted flag.

    The ensemble model outputs names like 'dotted_eighth', 'dotted_quarter', etc.
    but NOTE_VALUE_BEATS only has keys like 'eighth', 'quarter'. Without normalization,
    every dotted prediction silently falls back to 1.0 (quarter note) via .get() default.
    """
    nv = note_dict.get('note_value', '')
    if nv.startswith('dotted_'):
        base = nv[7:]  # strip 'dotted_' prefix
        note_dict['note_value'] = base
        note_dict['dotted'] = True
        note_dict['note_divisions'] = NOTE_VALUE_BEATS.get(base, 1.0) * 1.5
    else:
        note_dict['dotted'] = note_dict.get('dotted', False)
        base_beats = NOTE_VALUE_BEATS.get(nv, 1.0)
        note_dict['note_divisions'] = base_beats * 1.5 if note_dict['dotted'] else base_beats


#* ─── ML Rhythm Model (lazy loaded) ───────────────────────────────────────────
_rhythm_model = None
_rhythm_model_loaded = False

def get_rhythm_model():
    """Lazy load the ML rhythm quantization model."""
    global _rhythm_model, _rhythm_model_loaded
    
    if _rhythm_model_loaded:
        return _rhythm_model
    
    try:
        from rhythm_training.rhythm_model import RhythmQuantizerMLP

        # Try to find the model file
        model_paths = [
            os.path.join(os.path.dirname(__file__), 'rhythm_training', 'rhythm_model.npz'),
            os.path.join(os.path.dirname(__file__), 'rhythm_model.npz'),
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                _rhythm_model = RhythmQuantizerMLP(hidden_size=128)
                _rhythm_model.load(path)
                print(f"[Rhythm ML] Loaded model from {path}")
                break
        
        if _rhythm_model is None:
            print("[Rhythm ML] Model file not found, using heuristic quantization")
    except Exception as e:
        print(f"[Rhythm ML] Failed to load model: {e}")
        _rhythm_model = None
    
    _rhythm_model_loaded = True
    return _rhythm_model


def load_audio_deterministic(path, target_sr=44100):
    # Read raw PCM deterministically
    y, sr = sf.read(path, dtype="float32", always_2d=True)  # shape (N, ch)
    y = y.mean(axis=1).astype(np.float32, copy=False)       # force mono by ourselves

    if sr != target_sr:
        g = math.gcd(sr, target_sr)
        up, down = target_sr // g, sr // g
        y = resample_poly(y, up, down).astype(np.float32, copy=False)  # deterministic polyphase
    return y, target_sr

#* ─── Constants ────────────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
FRAME_SIZE  = 2048
HOP_SIZE    = 512
FFT_SIZE    = 2048
MAG_SIZE    = FFT_SIZE // 2 + 1
CQT_BINS    = 88
WINDOW_TYPE = 'hann'

CHORD_INTERVALS = {
    'maj':  [0, 4, 7],
    'min':  [0, 3, 7],
    'dim':  [0, 3, 6],
    'aug':  [0, 4, 8],
    'sus2': [0, 2, 7],
    'sus4': [0, 5, 7],
    'maj7': [0, 4, 7, 11],
    'min7': [0, 3, 7, 10],
    'dom7': [0, 4, 7, 10],
}
ROOTS = ['C', 'C#', 'D', 'D#', 'E', 'F', 
            'F#', 'G', 'G#', 'A', 'A#', 'B']
test_benchmark = 'test_C3_E3_G3_C4_Cmaj_Fmaj7_Gdom7_Caug_Cmin.wav'

# Precompute MIDI → Hz for CQT bins 21…108
bin_freq = np.array([440.0 * 2**((m - 69)/12) for m in np.arange(21, 21 + CQT_BINS)])

#* ─── Utility Functions ──────────────────────────────────────────────────────
def note_to_name(note):
    """Convert MIDI note number to a string name."""
    if note < 21 or note > 108:
        raise ValueError(f"Note {note} out of range (21-108)")
    octave = (note // 12) - 1
    pitch_class = note % 12
    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F',
                        'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f"{pitch_classes[pitch_class]}{octave}"

def frames_to_seconds(frames, sr, hop_length):
    """Convert frame indices to time in seconds."""
    return (np.asarray(frames) * hop_length) / float(sr)


def duration_to_note_value(duration_seconds, bpm=120, debug=False):
    """
    Convert duration in seconds to a note value based on tempo.

    Uses fraction-based quantization: converts beats to a rational number
    via Fraction.limit_denominator(), then maps to the nearest musical
    note value. This is mathematically continuous — similar inputs always
    produce similar outputs, eliminating threshold cliff-edge artifacts.

    Falls back to log-distance matching for values outside the fraction table.

    Args:
        duration_seconds: Duration of the note in seconds
        bpm: Beats per minute (default 120)
        debug: Print debug info for this quantization

    Returns:
        dict with 'type', 'divisions', 'beats', 'dotted', 'is_triplet',
        'raw_beats', 'quantization_error'
    """
    beat_duration = 60.0 / bpm
    beats = duration_seconds / beat_duration

    # Clamp to valid range
    beats = max(0.0625, min(beats, 8.0))

    # ── Primary: fraction-based lookup ──
    # Find nearest musical fraction using vectorized distance
    idx = np.argmin(np.abs(_MUSICAL_FRAC_VALS - beats))
    nearest_frac = _MUSICAL_FRAC_KEYS[idx]
    nearest_val = float(nearest_frac)

    # Check if the fraction match is close enough (within ~40% log-distance)
    if nearest_val > 0 and beats > 0:
        log_dist = abs(math.log2(beats / nearest_val))
    else:
        log_dist = abs(beats - nearest_val)

    if log_dist < 0.5 and nearest_frac in MUSICAL_FRACTIONS:
        note_type, note_beats, dotted, is_triplet = MUSICAL_FRACTIONS[nearest_frac]

        # Apply penalties: if a non-triplet/non-dotted candidate is almost as close,
        # prefer it (same penalty logic as before)
        if is_triplet or dotted:
            penalized_dist = log_dist + (0.15 if is_triplet else 0.05)
            # Check if a simpler candidate is within range
            for j in range(max(0, idx - 2), min(len(_MUSICAL_FRAC_KEYS), idx + 3)):
                alt_frac = _MUSICAL_FRAC_KEYS[j]
                if alt_frac == nearest_frac:
                    continue
                alt_info = MUSICAL_FRACTIONS.get(alt_frac)
                if alt_info and not alt_info[2] and not alt_info[3]:  # not dotted, not triplet
                    alt_val = float(alt_frac)
                    alt_dist = abs(math.log2(beats / alt_val)) if alt_val > 0 else 999
                    if alt_dist < penalized_dist:
                        note_type, note_beats, dotted, is_triplet = alt_info
                        nearest_val = alt_val
                        break

        q_error = abs(beats - note_beats) / note_beats if note_beats > 0 else 0

        if debug:
            margin_ms = abs(beats - note_beats) * beat_duration * 1000
            triplet_str = "triplet " if is_triplet else ""
            dotted_str = "dotted " if dotted else ""
            print(f"[Duration] {duration_seconds*1000:.1f}ms = {beats:.4f} beats @ {bpm} BPM "
                  f"-> {dotted_str}{triplet_str}{note_type} ({note_beats} beats, "
                  f"margin: {margin_ms:.1f}ms) [fraction]")

        return {
            'type': note_type,
            'divisions': note_beats,
            'beats': note_beats,
            'dotted': dotted,
            'is_triplet': is_triplet,
            'raw_beats': beats,
            'quantization_error': q_error,
        }

    # ── Fallback: log-distance matching for edge cases ──
    note_values = [
        ('whole', 6.0, True, False),
        ('whole', 4.0, False, False),
        ('half', 3.0, True, False),
        ('half', 4/3, False, True),
        ('half', 2.0, False, False),
        ('quarter', 1.5, True, False),
        ('quarter', 2/3, False, True),
        ('quarter', 1.0, False, False),
        ('eighth', 0.75, True, False),
        ('eighth', 1/3, False, True),
        ('eighth', 0.5, False, False),
        ('16th', 0.375, True, False),
        ('16th', 1/6, False, True),
        ('16th', 0.25, False, False),
        ('32nd', 0.1875, True, False),
        ('32nd', 1/12, False, True),
        ('32nd', 0.125, False, False),
    ]

    TRIPLET_PENALTY = 0.15
    DOTTED_PENALTY = 0.05

    best_match = None
    best_distance = float('inf')

    for note_type, note_beats, dotted, is_triplet in note_values:
        if beats > 0 and note_beats > 0:
            log_distance = abs(math.log2(beats / note_beats))
        else:
            log_distance = abs(beats - note_beats)

        if is_triplet:
            log_distance += TRIPLET_PENALTY
        if dotted:
            log_distance += DOTTED_PENALTY

        if log_distance < best_distance:
            best_distance = log_distance
            best_match = (note_type, note_beats, dotted, is_triplet)

    if best_match:
        note_type, note_beats, dotted, is_triplet = best_match

        if debug:
            margin_ms = abs(beats - note_beats) * beat_duration * 1000
            triplet_str = "triplet " if is_triplet else ""
            dotted_str = "dotted " if dotted else ""
            print(f"[Duration] {duration_seconds*1000:.1f}ms = {beats:.4f} beats @ {bpm} BPM "
                  f"-> {dotted_str}{triplet_str}{note_type} ({note_beats} beats, "
                  f"margin: {margin_ms:.1f}ms) [fallback]")

        return {
            'type': note_type,
            'divisions': note_beats,
            'beats': note_beats,
            'dotted': dotted,
            'is_triplet': is_triplet,
            'raw_beats': beats,
            'quantization_error': abs(beats - note_beats) / note_beats if note_beats > 0 else 0,
        }

    return {
        'type': '32nd',
        'divisions': 0.125,
        'beats': 0.125,
        'dotted': False,
        'is_triplet': False,
        'raw_beats': beats,
        'quantization_error': abs(beats - 0.125) / 0.125,
    }


def detect_ornaments(notes, bpm, debug=False):
    """
    Detect ornamental passages like trills, turns, mordents, and grace notes.
    
    TRILLS: Rapid alternation between two adjacent pitches (1-2 semitones apart)
    - Minimum 4 notes alternating between 2 pitches
    - Notes must be fast (typically 32nd or faster relative to tempo)
    - Collapses trill into single note with 'trill' ornament marker
    
    GRACE NOTES: Very short notes immediately before a main note
    - Single very short note (< 1/8 beat) followed by longer note
    - Usually stepwise or small interval
    
    TURNS/MORDENTS: Short melodic figures around a note
    - 3-4 notes that return to starting pitch
    
    Args:
        notes: List of note dicts with 'time_seconds', 'midi_note', 'duration_seconds'
        bpm: Tempo in BPM
        debug: Print debug info
        
    Returns:
        Modified notes list with ornaments detected and collapsed
    """
    if len(notes) < 3:
        return notes
    
    beat_duration = 60.0 / bpm
    sorted_notes = sorted(notes, key=lambda n: n.get('time_seconds', 0))
    
    # Track which notes to remove (they're part of ornaments)
    notes_to_remove = set()
    
    if debug:
        print(f"[Ornament] Scanning {len(sorted_notes)} notes for ornaments (BPM={bpm}, beat={beat_duration*1000:.0f}ms)")
        # Show first 20 notes to understand the data
        print(f"[Ornament] First notes (time, pitch, ioi_to_next):")
        for idx in range(min(20, len(sorted_notes))):
            n = sorted_notes[idx]
            t = n.get('time_seconds', 0)
            p = n.get('midi_note', 0)
            name = n.get('note_name', '?')
            if idx < len(sorted_notes) - 1:
                next_t = sorted_notes[idx + 1].get('time_seconds', 0)
                ioi_ms = (next_t - t) * 1000
                ioi_beats = (next_t - t) / beat_duration
            else:
                ioi_ms = 0
                ioi_beats = 0
            print(f"    {idx}: t={t:.3f}s, {name}(midi={p}), ioi={ioi_ms:.0f}ms ({ioi_beats:.2f}beats)")
    
    # ============ TRILL DETECTION ============
    # Look for rapid alternation between 2 adjacent pitches
    i = 0
    while i < len(sorted_notes) - 3:
        # Check if this could be the start of a trill
        note1 = sorted_notes[i]
        note2 = sorted_notes[i + 1]
        
        pitch1 = note1.get('midi_note', 60)
        pitch2 = note2.get('midi_note', 60)
        
        # Trills alternate between notes 1-3 semitones apart (half step to minor 3rd)
        interval = abs(pitch2 - pitch1)
        if interval < 1 or interval > 3:
            i += 1
            continue
        
        # Check timing - notes must be reasonably fast
        ioi = note2.get('time_seconds', 0) - note1.get('time_seconds', 0)
        ioi_beats = ioi / beat_duration
        
        # Trill notes are typically very fast - both in beats AND absolute time
        # Must be 8th notes or faster (0.5 beats or less) AND under 200ms
        # The absolute time check prevents false positives at slow tempos
        if ioi_beats > 0.6 or ioi > 0.2:  # 200ms minimum for trill notes
            i += 1
            continue
        
        if debug:
            print(f"  [Trill?] Starting at note {i}: pitch {pitch1}->{pitch2}, interval={interval}, ioi={ioi_beats:.2f} beats")
        
        # Now scan forward to find how long the trill continues
        trill_notes = [i, i + 1]
        last_pitch = pitch2  # The last pitch we saw
        consecutive_same = 0  # Track consecutive same-pitch notes (detection artifacts)
        
        for j in range(i + 2, len(sorted_notes)):
            next_note = sorted_notes[j]
            next_pitch = next_note.get('midi_note', 60)
            
            # Check timing is still reasonably fast (both in beats and absolute)
            prev_time = sorted_notes[j - 1].get('time_seconds', 0)
            curr_time = next_note.get('time_seconds', 0)
            next_ioi = curr_time - prev_time
            next_ioi_beats = next_ioi / beat_duration
            
            if next_ioi_beats > 0.7 or next_ioi > 0.25:  # Timing broke the trill (250ms max)
                if debug:
                    print(f"    Trill broke at note {j}: ioi too slow ({next_ioi_beats:.2f} beats)")
                break
            
            # Check if it's one of the two trill pitches
            if next_pitch in (pitch1, pitch2):
                if next_pitch == last_pitch:
                    # Same pitch repeated - might be detection artifact
                    consecutive_same += 1
                    if consecutive_same > 1:
                        # Too many same notes in a row - not a trill anymore
                        if debug:
                            print(f"    Trill broke at note {j}: too many consecutive {next_pitch}")
                        break
                    # Allow one repeated note (detection artifact)
                    trill_notes.append(j)
                else:
                    # Proper alternation
                    trill_notes.append(j)
                    last_pitch = next_pitch
                    consecutive_same = 0
            else:
                if debug:
                    expected = pitch1 if last_pitch == pitch2 else pitch2
                    print(f"    Trill broke at note {j}: pitch {next_pitch} not in trill ({pitch1}, {pitch2})")
                break
        
        # Need at least 3 notes for a valid trill (main + aux + main)
        if len(trill_notes) >= 3:
            # Calculate total trill duration
            first_time = sorted_notes[trill_notes[0]].get('time_seconds', 0)
            last_note_idx = trill_notes[-1]
            last_time = sorted_notes[last_note_idx].get('time_seconds', 0)
            last_dur = sorted_notes[last_note_idx].get('duration_seconds', 0.1)
            total_duration = (last_time + last_dur) - first_time
            
            # Mark the first note as having a trill
            principal_note = sorted_notes[trill_notes[0]]
            trill_to_pitch = pitch2 if pitch1 == principal_note.get('midi_note') else pitch1
            
            principal_note['ornament'] = 'trill'
            principal_note['trill_to'] = trill_to_pitch
            principal_note['trill_interval'] = interval
            principal_note['trill_notes_count'] = len(trill_notes)
            principal_note['duration_seconds'] = total_duration  # Extend duration
            
            # Quantize the trill duration
            trill_note_val = duration_to_note_value(total_duration, bpm=bpm, debug=False)
            principal_note['note_value'] = trill_note_val['type']
            principal_note['note_divisions'] = trill_note_val['beats']
            principal_note['dotted'] = trill_note_val.get('dotted', False)
            
            # Mark subsequent trill notes for removal
            for idx in trill_notes[1:]:
                notes_to_remove.add(idx)
            
            if debug:
                note_name = principal_note.get('note_name', '?')
                print(f"[Ornament] ✓ Trill detected: {note_name} with {len(trill_notes)} alternations "
                      f"({interval} semitones), duration={total_duration*1000:.0f}ms")
            
            # Skip past the trill
            i = trill_notes[-1] + 1
        else:
            i += 1
    
    # ============ GRACE NOTE DETECTION ============
    # DISABLED: Grace notes cause more problems than they solve in current pipeline
    # Very short notes are quantized to 32nd notes instead
    # To re-enable, set skip_grace_notes = False
    skip_grace_notes = True
    
    i = 0
    while i < len(sorted_notes) - 1 and not skip_grace_notes:
        if i in notes_to_remove:
            i += 1
            continue

        note = sorted_notes[i]
        next_note = sorted_notes[i + 1]

        if i + 1 in notes_to_remove:
            i += 1
            continue

        duration = note.get('duration_seconds', 0.5)
        duration_beats = duration / beat_duration

        # Check if this note fits reasonably well as a 32nd note (0.125 beats)
        # If the quantization error to 32nd note is small, prefer 32nd note over grace
        thirty_second_beats = 0.125
        quant_error_32nd = abs(duration_beats - thirty_second_beats) / thirty_second_beats if thirty_second_beats > 0 else 999
        fits_as_32nd = quant_error_32nd < 0.7  # More lenient - within 70% of a 32nd note duration

        # Grace notes must be EXTREMELY short AND meet many strict conditions:
        # - Less than 1/4 of a 32nd note in beats (< 0.03 beats) 
        # - Under 30ms absolute time (truly imperceptible as a rhythmic note)
        # - Must NOT fit well as a quantized 32nd note
        # - Must NOT be part of a simultaneous chord (check for notes within 20ms)
        
        # Check if this note is likely part of a chord (other notes at ~same time)
        note_time = note.get('time_seconds', 0)
        is_chord_note = False
        for j in range(max(0, i - 3), min(len(sorted_notes), i + 3)):
            if j != i and j not in notes_to_remove:
                other_time = sorted_notes[j].get('time_seconds', 0)
                if abs(other_time - note_time) < 0.025:  # Within 25ms = chord
                    is_chord_note = True
                    break
        
        # Skip grace note detection for chord notes entirely
        if is_chord_note:
            i += 1
            continue
        
        if duration_beats < 0.03 and duration < 0.03 and not fits_as_32nd:
            # Check the interval to the next note - grace notes are STEPWISE only
            pitch1 = note.get('midi_note', 60)
            pitch2 = next_note.get('midi_note', 60)
            interval = abs(pitch2 - pitch1)

            # Grace notes should be stepwise (1-2 semitones) or at most a small skip (3 semitones)
            if interval <= 3:
                # Check that next note is MUCH longer (at least 8x)
                next_dur_beats = next_note.get('duration_seconds', 0.5) / beat_duration

                if next_dur_beats >= duration_beats * 8:
                    # This is likely a grace note
                    note['ornament'] = 'grace'
                    note['grace_type'] = 'acciaccatura' if interval <= 2 else 'appoggiatura'
                    note['note_value'] = 'grace'
                    note['note_divisions'] = 0  # Grace notes don't count in rhythm

                    if debug:
                        note_name = note.get('note_name', '?')
                        print(f"[Ornament] Grace note detected: {note_name} -> "
                              f"{next_note.get('note_name', '?')} "
                              f"(dur={duration*1000:.1f}ms, {duration_beats:.4f} beats)")
        i += 1
    
    # ============ MORDENT DETECTION ============
    # 3 notes: main -> auxiliary -> main (all fast)
    i = 0
    while i < len(sorted_notes) - 2:
        if i in notes_to_remove:
            i += 1
            continue
        
        note1 = sorted_notes[i]
        note2 = sorted_notes[i + 1]
        note3 = sorted_notes[i + 2]
        
        if (i + 1) in notes_to_remove or (i + 2) in notes_to_remove:
            i += 1
            continue
        
        pitch1 = note1.get('midi_note', 60)
        pitch2 = note2.get('midi_note', 60)
        pitch3 = note3.get('midi_note', 60)
        
        # Mordent: returns to starting pitch
        if pitch1 == pitch3 and pitch1 != pitch2:
            interval = abs(pitch2 - pitch1)
            
            # Usually 1-2 semitones
            if interval in (1, 2):
                # Check all three notes are fast
                time1 = note1.get('time_seconds', 0)
                time2 = note2.get('time_seconds', 0)
                time3 = note3.get('time_seconds', 0)
                
                total_time = time3 - time1
                total_beats = total_time / beat_duration
                
                # All three notes should fit in about 1 beat or less
                # AND total time should be under 500ms (absolute threshold)
                if total_beats <= 1.0 and total_time <= 0.5:
                    # This is a mordent
                    mordent_type = 'upper' if pitch2 > pitch1 else 'lower'
                    
                    # Extend first note to cover the mordent
                    note1['ornament'] = f'mordent_{mordent_type}'
                    note1['mordent_note'] = pitch2
                    original_end = time3 + note3.get('duration_seconds', 0.1)
                    note1['duration_seconds'] = original_end - time1
                    
                    # Re-quantize
                    mord_note_val = duration_to_note_value(note1['duration_seconds'], bpm=bpm)
                    note1['note_value'] = mord_note_val['type']
                    note1['note_divisions'] = mord_note_val['beats']
                    note1['dotted'] = mord_note_val.get('dotted', False)
                    
                    # Mark notes 2 and 3 for removal
                    notes_to_remove.add(i + 1)
                    notes_to_remove.add(i + 2)
                    
                    if debug:
                        note_name = note1.get('note_name', '?')
                        print(f"[Ornament] {mordent_type.capitalize()} mordent detected on {note_name}")
                    
                    i += 3
                    continue
        
        i += 1
    
    # ============ TURN DETECTION ============
    # 4 notes: main -> upper -> main -> lower (or reverse)
    i = 0
    while i < len(sorted_notes) - 3:
        if i in notes_to_remove:
            i += 1
            continue
        
        note1 = sorted_notes[i]
        note2 = sorted_notes[i + 1]
        note3 = sorted_notes[i + 2]
        note4 = sorted_notes[i + 3]
        
        if any((i + j) in notes_to_remove for j in range(1, 4)):
            i += 1
            continue
        
        pitch1 = note1.get('midi_note', 60)
        pitch2 = note2.get('midi_note', 60)
        pitch3 = note3.get('midi_note', 60)
        pitch4 = note4.get('midi_note', 60)
        
        # Turn pattern: main -> upper -> main -> lower OR main -> lower -> main -> upper
        is_turn = False
        turn_type = None
        
        if pitch3 == pitch1:  # Returns to main note
            if pitch2 > pitch1 and pitch4 < pitch1:
                # Upper turn: main -> up -> main -> down
                is_turn = True
                turn_type = 'upper'
            elif pitch2 < pitch1 and pitch4 > pitch1:
                # Inverted turn: main -> down -> main -> up
                is_turn = True
                turn_type = 'inverted'
        
        if is_turn:
            # Check intervals are small (1-2 semitones typically)
            int1 = abs(pitch2 - pitch1)
            int2 = abs(pitch4 - pitch1)
            
            if int1 <= 3 and int2 <= 3:
                # Check timing - should be fast (both in beats and absolute)
                time1 = note1.get('time_seconds', 0)
                time4 = note4.get('time_seconds', 0)
                total_time = time4 - time1
                total_beats = total_time / beat_duration
                
                # Must fit in 1.5 beats AND under 600ms absolute
                if total_beats <= 1.5 and total_time <= 0.6:
                    note1['ornament'] = f'turn_{turn_type}'
                    original_end = time4 + note4.get('duration_seconds', 0.1)
                    note1['duration_seconds'] = original_end - time1
                    
                    # Re-quantize
                    turn_note_val = duration_to_note_value(note1['duration_seconds'], bpm=bpm)
                    note1['note_value'] = turn_note_val['type']
                    note1['note_divisions'] = turn_note_val['beats']
                    note1['dotted'] = turn_note_val.get('dotted', False)
                    
                    for j in range(1, 4):
                        notes_to_remove.add(i + j)
                    
                    if debug:
                        note_name = note1.get('note_name', '?')
                        print(f"[Ornament] {turn_type.capitalize()} turn detected on {note_name}")
                    
                    i += 4
                    continue
        
        i += 1
    
    # Remove the ornamental notes that were collapsed
    result = [n for idx, n in enumerate(sorted_notes) if idx not in notes_to_remove]
    
    if debug and notes_to_remove:
        print(f"[Ornament] Collapsed {len(notes_to_remove)} ornamental notes into main notes")
    
    return result


def enforce_bar_sum(notes, bpm, beats_per_bar=4.0, debug=False):
    """
    Adjust note values so each bar sums to exactly beats_per_bar using ILP.

    For each bar, formulates a small integer linear program:
      - Each non-triplet note can be assigned any value from the LADDER
      - Objective: minimize total cost (weighted deviation from current value)
      - Constraint: sum of chosen values == beats_per_bar

    Falls back to greedy single-step adjustment if scipy ILP fails or if the
    bar has too few adjustable notes.

    Args:
        notes:         List of note dicts (must already have 'note_value',
                       'dotted', 'time_seconds', and ideally 'quantization_error').
        bpm:           Current tempo.
        beats_per_bar: How many quarter-note beats fill one bar (default 4.0).
        debug:         Print per-bar adjustment info.

    Returns:
        The same list, modified in place, with note values adjusted.
    """
    if len(notes) < 2 or beats_per_bar <= 0:
        return notes

    from scipy.optimize import linprog
    from scipy.sparse import eye as speye

    beat_duration = 60.0 / bpm

    LADDER = [
        ('32nd',    0.125,  False),
        ('32nd',    0.1875, True),
        ('16th',    0.25,   False),
        ('16th',    0.375,  True),
        ('eighth',  0.5,    False),
        ('eighth',  0.75,   True),
        ('quarter', 1.0,    False),
        ('quarter', 1.5,    True),
        ('half',    2.0,    False),
        ('half',    3.0,    True),
        ('whole',   4.0,    False),
        ('whole',   6.0,    True),
    ]
    LADDER_BEATS = np.array([b for (_, b, _) in LADDER])
    N_LADDER = len(LADDER)

    def _note_beats(n):
        base = NOTE_VALUE_BEATS.get(n.get('note_value', 'quarter'), 1.0)
        if n.get('dotted', False):
            base *= 1.5
        return base

    def _ladder_index(n):
        nv = n.get('note_value', 'quarter')
        dot = n.get('dotted', False)
        for idx, (lt, _, ld) in enumerate(LADDER):
            if lt == nv and ld == dot:
                return idx
        return -1

    def _set_from_ladder(n, idx):
        lt, lb, ld = LADDER[idx]
        n['note_value'] = lt
        n['dotted'] = ld
        n['note_divisions'] = lb

    sorted_notes = sorted(notes, key=lambda n: n.get('time_seconds', 0))
    first_onset = sorted_notes[0].get('time_seconds', 0.0)

    # Assign each note to a bar
    bars = {}
    for i, n in enumerate(sorted_notes):
        onset = n.get('time_seconds', 0.0) - first_onset
        bar_idx = int(onset / (beats_per_bar * beat_duration))
        bars.setdefault(bar_idx, []).append(i)

    tol = beats_per_bar * 0.015
    adjustments = 0

    for bar_idx in sorted(bars):
        indices = bars[bar_idx]
        bar_total = sum(_note_beats(sorted_notes[i]) for i in indices)
        has_rest = any(sorted_notes[i].get('has_rest_after', False) for i in indices)
        deficit = beats_per_bar - bar_total

        if abs(deficit) <= tol:
            continue
        # Only fix under-fill if no rests present
        if deficit > tol and has_rest:
            continue

        # Collect adjustable notes (skip triplets, grace notes)
        adjustable = []
        fixed_beats = 0.0
        for i in indices:
            n = sorted_notes[i]
            if n.get('is_triplet', False) or n.get('note_value') == 'grace':
                fixed_beats += _note_beats(n)
                continue
            li = _ladder_index(n)
            if li < 0:
                fixed_beats += _note_beats(n)
                continue
            adjustable.append((i, li, n.get('quantization_error', 0.5)))

        if not adjustable:
            continue

        target = beats_per_bar - fixed_beats
        if target <= 0:
            continue

        K = len(adjustable)

        # ── ILP formulation ──
        # For each adjustable note k, we have N_LADDER binary variables x[k,j]
        # indicating "note k is assigned ladder position j".
        # Total variables: K * N_LADDER
        # Constraint 1: for each k, sum_j x[k,j] = 1 (pick exactly one)
        # Constraint 2: sum_k sum_j LADDER_BEATS[j] * x[k,j] = target
        # Objective: minimize sum_k sum_j cost[k,j] * x[k,j]
        #   where cost[k,j] = |LADDER_BEATS[j] - LADDER_BEATS[current_j]| * (1 + q_error)

        n_vars = K * N_LADDER

        # Build cost vector
        c = np.zeros(n_vars)
        for k, (ni, cur_li, q_err) in enumerate(adjustable):
            cur_beats = LADDER_BEATS[cur_li]
            weight = 1.0 + q_err  # higher q_error = cheaper to adjust
            for j in range(N_LADDER):
                c[k * N_LADDER + j] = abs(LADDER_BEATS[j] - cur_beats) / weight

        # Equality constraints: A_eq @ x = b_eq
        # K "pick one" constraints + 1 "sum to target" constraint
        n_eq = K + 1
        A_eq = np.zeros((n_eq, n_vars))
        b_eq = np.zeros(n_eq)

        # Pick-one constraints
        for k in range(K):
            A_eq[k, k * N_LADDER:(k + 1) * N_LADDER] = 1.0
            b_eq[k] = 1.0

        # Sum-to-target constraint
        for k in range(K):
            for j in range(N_LADDER):
                A_eq[K, k * N_LADDER + j] = LADDER_BEATS[j]
        b_eq[K] = target

        # Bounds: 0 <= x <= 1 (relaxed LP; round to nearest integer)
        bounds = [(0, 1)] * n_vars

        try:
            result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

            if result.success:
                x = result.x
                for k, (ni, cur_li, _) in enumerate(adjustable):
                    chosen_j = np.argmax(x[k * N_LADDER:(k + 1) * N_LADDER])
                    if chosen_j != cur_li:
                        _set_from_ladder(sorted_notes[ni], chosen_j)
                        adjustments += 1
                        if debug:
                            print(f"[BarSum ILP] bar {bar_idx}: note {ni} "
                                  f"{LADDER[cur_li][0]}->{LADDER[chosen_j][0]}")
                continue
        except Exception:
            pass

        # ── Greedy fallback ──
        adjustable.sort(key=lambda x: -x[2])  # worst confidence first
        remaining = deficit
        for ni, li, _ in adjustable:
            if abs(remaining) <= tol:
                break
            n = sorted_notes[ni]
            cur_beats = LADDER_BEATS[li]
            if remaining > 0 and li < N_LADDER - 1:
                new_beats = LADDER_BEATS[li + 1]
                gain = new_beats - cur_beats
                if gain <= remaining + tol:
                    _set_from_ladder(n, li + 1)
                    remaining -= gain
                    adjustments += 1
            elif remaining < 0 and li > 0:
                new_beats = LADDER_BEATS[li - 1]
                loss = cur_beats - new_beats
                if loss <= -remaining + tol:
                    _set_from_ladder(n, li - 1)
                    remaining += loss
                    adjustments += 1

    if debug and adjustments:
        print(f"[BarSum ILP] adjusted {adjustments} notes across {len(bars)} bars "
              f"(time_sig={beats_per_bar} beats/bar)")

    return notes


def post_process_rhythm_unified(notes, bpm, debug=False):
    """
    Unified single-pass rhythm post-processing.

    Replaces the previous 4-function chain:
      fill_gaps_with_ioi -> smooth_rhythm_gaps -> reduce_rest_entropy -> detect_and_normalize_runs

    In a SINGLE pass over the note list, this function:
      1. Computes all IOIs and builds a sliding-window statistical model
      2. For each note decides: extend to fill gap, insert rest, or keep as-is
      3. Detects runs of similar-duration notes and normalizes outliers

    All thresholds are TEMPO-RELATIVE rather than fixed.

    Args:
        notes: List of note dicts (already quantized with note_value)
        bpm: Tempo in BPM
        debug: Print debug info

    Returns:
        Modified notes with unified post-processing applied
    """
    if len(notes) < 2:
        return notes

    beat_duration = 60.0 / bpm

    # ── Tempo-relative thresholds ──
    # At 60 BPM beat=1s, at 120 BPM beat=0.5s, at 180 BPM beat=0.33s
    # "Minimum meaningful gap" scales with tempo: ~1/8 of a beat
    min_gap_beats = 0.125  # 32nd note — below this, always extend
    # "Maximum fill gap" — extend notes for gaps up to this size
    # Larger at slow tempos (more time between notes), smaller at fast tempos
    max_fill_beats = min(2.0, max(0.5, 2.0 * (120.0 / bpm)))
    # At 60 BPM -> 4.0 (capped at 2.0), at 120 BPM -> 2.0, at 180 BPM -> 1.33
    # "Phrase boundary" — gaps above this are always rests
    phrase_boundary_beats = max(2.0, 6.0 * (60.0 / bpm))
    # At 60 BPM -> 6.0, at 120 BPM -> 3.0, at 180 BPM -> 2.0

    sorted_notes = sorted(notes, key=lambda n: n.get('time_seconds', 0))

    # ── Pre-compute IOIs for the whole sequence ──
    n_notes = len(sorted_notes)
    iois = []
    for i in range(n_notes - 1):
        onset_a = sorted_notes[i].get('time_seconds', 0)
        onset_b = sorted_notes[i + 1].get('time_seconds', 0)
        iois.append(onset_b - onset_a)
    iois.append(iois[-1] if iois else beat_duration)  # pad last

    # ── Pre-compute sliding-window statistics (Hampel identifier) ──
    WINDOW = 8
    K_THRESHOLD = 3.0
    MAD_SCALE = 1.4826

    def _is_outlier_ioi(idx):
        """Return True if IOI[idx] is a statistical outlier in its neighbourhood."""
        lo = max(0, idx - WINDOW // 2)
        hi = min(len(iois), idx + WINDOW // 2 + 1)
        window = sorted(iois[lo:hi])
        n = len(window)
        if n < 3:
            window = sorted(iois)
            n = len(window)
        median = window[n // 2]
        deviations = sorted(abs(x - median) for x in window)
        mad = deviations[len(deviations) // 2]
        sigma_est = mad * MAD_SCALE
        if sigma_est < 1e-6:
            sigma_est = median * 0.25
        threshold = median + K_THRESHOLD * sigma_est
        return iois[idx] > threshold

    # ── Detect runs of similar IOIs (for run normalization) ──
    IOI_SIMILARITY = 0.20

    # Find contiguous segments where IOIs are similar
    ioi_runs = []
    run_start = 0
    for i in range(1, len(iois) - 1):  # exclude padded last
        prev_ioi = iois[i - 1]
        curr_ioi = iois[i]
        if prev_ioi > 0 and curr_ioi > 0:
            ratio = curr_ioi / prev_ioi
            if 1.0 - IOI_SIMILARITY <= ratio <= 1.0 + IOI_SIMILARITY:
                continue
        run_len = i - run_start + 1
        if run_len >= 4:
            ioi_runs.append((run_start, run_start + run_len))
        run_start = i
    # Final run
    final_run_len = n_notes - run_start
    if final_run_len >= 3:
        ioi_runs.append((run_start, n_notes))

    # Also find runs based on assigned note values
    value_runs = []
    run_start = 0
    prev_bts = NOTE_VALUE_BEATS.get(sorted_notes[0].get('note_value', 'quarter'), 1.0)
    for i in range(1, n_notes):
        curr_bts = NOTE_VALUE_BEATS.get(sorted_notes[i].get('note_value', 'quarter'), 1.0)
        if prev_bts > 0 and curr_bts > 0:
            ratio = curr_bts / prev_bts
            if 0.67 <= ratio <= 1.5:
                prev_bts = curr_bts
                continue
        if i - run_start >= 4:
            value_runs.append((run_start, i))
        run_start = i
        prev_bts = curr_bts
    if n_notes - run_start >= 3:
        value_runs.append((run_start, n_notes))

    # Merge overlapping runs
    all_runs = sorted(ioi_runs + value_runs, key=lambda x: x[0])
    merged_runs = []
    for run in all_runs:
        if merged_runs and run[0] <= merged_runs[-1][1]:
            merged_runs[-1] = (merged_runs[-1][0], max(merged_runs[-1][1], run[1]))
        else:
            merged_runs.append(run)

    # Build set of notes in runs, with their target value
    run_targets = {}  # note_index -> (target_type, target_beats, target_dotted)
    for run_start_idx, run_end_idx in merged_runs:
        run_notes = sorted_notes[run_start_idx:run_end_idx]
        run_iois = [iois[i] for i in range(run_start_idx, min(run_end_idx - 1, len(iois)))]
        if not run_iois:
            continue

        median_ioi = np.median(run_iois)
        run_note_val = duration_to_note_value(median_ioi, bpm=bpm, debug=False)
        run_type = run_note_val['type']
        run_beats = run_note_val['beats']
        run_dotted = run_note_val.get('dotted', False)

        # Also check majority vote
        value_counts = {}
        for note in run_notes:
            val = note.get('note_value', 'quarter')
            dotted = note.get('dotted', False)
            key = (val, dotted)
            value_counts[key] = value_counts.get(key, 0) + 1

        most_common = max(value_counts.items(), key=lambda x: x[1])
        most_common_type, most_common_dotted = most_common[0]
        most_common_count = most_common[1]

        ioi_error = run_note_val.get('quantization_error', 1.0)
        majority_pct = most_common_count / len(run_notes)

        if ioi_error < 0.15 and run_type == most_common_type:
            final_type, final_beats, final_dotted = run_type, run_beats, run_dotted
        elif majority_pct >= 0.4:
            final_type = most_common_type
            final_dotted = most_common_dotted
            final_beats = NOTE_VALUE_BEATS.get(final_type, 0.5)
            if final_dotted:
                final_beats *= 1.5
        else:
            final_type, final_beats, final_dotted = run_type, run_beats, run_dotted

        for j in range(run_start_idx, run_end_idx):
            run_targets[j] = (final_type, final_beats, final_dotted)

    # ── SINGLE PASS: process each note ──
    fills = 0
    extensions = 0
    rest_removals = 0
    run_normalizations = 0

    for i in range(n_notes):
        note = sorted_notes[i]
        note_beats = note.get('note_divisions', 1.0)

        # ── Step A: Run normalization (outlier notes in runs) ──
        # Skip notes with deliberate rests — their shorter value is intentional
        if i in run_targets and not note.get('has_rest_after', False):
            # Preserve pre-tagged run notes and high-confidence ensemble predictions
            if note.get('quantization_method') == 'run_tagged':
                pass  # pre-quantization run tag takes priority
            elif note.get('quantization_method') == 'ensemble_kept' \
               and note.get('quantization_confidence', 0) >= 0.60:
                pass  # keep ensemble prediction
            else:
                target_type, target_beats, target_dotted = run_targets[i]
                if note.get('note_value') != target_type or note.get('dotted', False) != target_dotted:
                    note['note_value'] = target_type
                    note['note_divisions'] = target_beats
                    note['dotted'] = target_dotted
                    note['is_triplet'] = False
                    note['triplet'] = False
                    note.pop('triplet_position', None)
                    note.pop('actual_notes', None)
                    note.pop('normal_notes', None)
                    note['run_normalized'] = True
                    note_beats = target_beats
                    run_normalizations += 1

        # ── Step B: Gap analysis (fill, extend, or rest) ──
        if i >= n_notes - 1:
            continue  # Last note — no gap to analyze

        ioi = iois[i]
        ioi_beats = ioi / beat_duration
        gap_beats = ioi_beats - note_beats

        # B1: Tiny or no gap — nothing to do
        if gap_beats <= min_gap_beats:
            note['has_rest_after'] = False
            note.pop('rest_duration', None)
            continue

        # B2: Beyond phrase boundary — always keep rest
        if gap_beats > phrase_boundary_beats:
            note['has_rest_after'] = True
            note['rest_duration'] = ioi - note_beats * beat_duration
            continue

        # B3: Statistical outlier test (from reduce_rest_entropy logic)
        is_outlier = _is_outlier_ioi(i)

        # B4: If already marked as rest, keep it — deliberate rest assignments
        # should not be undone by gap filling. Only override if gap is tiny (B1).
        if note.get('has_rest_after', False):
            continue

        # B4.5: Don't extend ensemble-predicted notes to fill gaps
        if note.get('quantization_method') == 'ensemble_kept' \
           and note.get('quantization_confidence', 0) >= 0.60:
            continue

        # B5: Within fillable range — try to extend note
        if gap_beats <= max_fill_beats:
            # Try IOI-based re-quantization (from fill_gaps_with_ioi logic)
            ioi_val = duration_to_note_value(ioi, bpm=bpm, debug=False)
            ioi_error = ioi_val.get('quantization_error', 1.0)

            if ioi_error < 0.3:
                # Clean quantization to IOI — extend note
                note['note_value'] = ioi_val['type']
                note['note_divisions'] = ioi_val['divisions']
                note['dotted'] = ioi_val.get('dotted', False)
                note['has_rest_after'] = False
                note.pop('rest_duration', None)
                fills += 1
                continue

            # Try extending to nearest larger standard value (from smooth_rhythm_gaps logic)
            new_duration_beats = note_beats + gap_beats
            new_note_val = extend_to_nearest_value(
                note_beats, new_duration_beats, note.get('is_triplet', False)
            )
            if new_note_val:
                note['note_value'] = new_note_val['type']
                note['note_divisions'] = new_note_val['beats']
                note['dotted'] = new_note_val.get('dotted', False)
                note['has_rest_after'] = False
                note.pop('rest_duration', None)
                extensions += 1
                continue

        # B6: Medium gap, not an outlier — still try to extend by re-quantizing IOI
        # But don't override pre-tagged run notes — their value is already correct
        if not is_outlier and note.get('quantization_method') != 'run_tagged':
            ioi_val = duration_to_note_value(ioi, bpm=bpm, debug=False)
            ioi_error = ioi_val.get('quantization_error', 1.0)
            if ioi_error < 0.35:
                note['note_value'] = ioi_val['type']
                note['note_divisions'] = ioi_val['divisions']
                note['dotted'] = ioi_val.get('dotted', False)
                note['has_rest_after'] = False
                note.pop('rest_duration', None)
                note['quantization_method'] = note.get('quantization_method', '') + ' (rest removed)'
                rest_removals += 1
                continue

        # B7: Fallback — significant gap that's an outlier or can't be filled cleanly
        note['has_rest_after'] = True
        note['rest_duration'] = ioi - note_beats * beat_duration

    if debug:
        print(f"  [Unified Post-Process] IOI fills: {fills}, extensions: {extensions}, "
              f"rest removals: {rest_removals}, run normalizations: {run_normalizations}")

    return sorted_notes


def detect_and_normalize_runs(notes, bpm, debug=False):
    """
    Detect runs of similar-duration notes and normalize them to the same value.
    
    MUSICAL INSIGHT: When a pianist plays a run of eighth notes, timing will vary
    slightly, but they're ALL eighths. If 7 out of 8 notes in a sequence quantize
    to eighths and 1 quantizes to a dotted 16th, that outlier should become an eighth.
    
    This uses TWO approaches:
    1. IOI-based: Find consecutive notes with similar inter-onset intervals
    2. Value-based: Find consecutive notes assigned similar note values
    
    Args:
        notes: List of note dicts (already with note_value assigned)
        bpm: Tempo in BPM
        debug: Print debug info
        
    Returns:
        Modified notes list with normalized runs
    """
    if len(notes) < 3:
        return notes
    
    beat_duration = 60.0 / bpm
    sorted_notes = sorted(notes, key=lambda n: n.get('time_seconds', 0))
    
    # Calculate IOIs
    times = [n.get('time_seconds', 0) for n in sorted_notes]
    iois = []
    for i in range(len(times) - 1):
        iois.append(times[i + 1] - times[i])

    # Map note values to their beat duration for comparison
    
    # APPROACH 1: Find runs based on IOI similarity (within 35%)
    IOI_SIMILARITY = 0.20
    
    # Find contiguous segments where IOIs are similar
    runs = []
    run_start = 0
    
    for i in range(len(iois)):
        if i == 0:
            continue
        
        prev_ioi = iois[i - 1]
        curr_ioi = iois[i]
        
        if prev_ioi > 0 and curr_ioi > 0:
            ratio = curr_ioi / prev_ioi
            if 1.0 - IOI_SIMILARITY <= ratio <= 1.0 + IOI_SIMILARITY:
                continue  # Similar, keep building run
        
        # End of run
        run_len = i - run_start + 1  # +1 because i is IOI index, we want note count
        if run_len >= 4:
            runs.append((run_start, run_start + run_len))
        run_start = i
    
    # Final run
    run_len = len(sorted_notes) - run_start
    if run_len >= 3:
        runs.append((run_start, len(sorted_notes)))
    
    # APPROACH 2: Also find runs based on assigned note values
    # Group consecutive notes with the same or adjacent note values
    value_runs = []
    run_start = 0
    prev_beats = NOTE_VALUE_BEATS.get(sorted_notes[0].get('note_value', 'quarter'), 1.0)
    
    for i in range(1, len(sorted_notes)):
        curr_beats = NOTE_VALUE_BEATS.get(sorted_notes[i].get('note_value', 'quarter'), 1.0)
        
        # Allow notes within factor of 2 to be in same run (handles dotted variants)
        if prev_beats > 0 and curr_beats > 0:
            ratio = curr_beats / prev_beats
            if 0.67 <= ratio <= 1.5:
                prev_beats = curr_beats
                continue
        
        # End of run
        if i - run_start >= 4:
            value_runs.append((run_start, i))
        run_start = i
        prev_beats = curr_beats
    
    if len(sorted_notes) - run_start >= 3:
        value_runs.append((run_start, len(sorted_notes)))
    
    # Merge the two approaches - use whichever finds longer runs
    all_runs = runs + value_runs
    
    # Sort by start, then merge overlapping runs
    all_runs.sort(key=lambda x: x[0])
    merged_runs = []
    for run in all_runs:
        if merged_runs and run[0] <= merged_runs[-1][1]:
            # Overlapping - extend previous run
            merged_runs[-1] = (merged_runs[-1][0], max(merged_runs[-1][1], run[1]))
        else:
            merged_runs.append(run)
    
    if debug and merged_runs:
        print(f"[Run Detection] Found {len(merged_runs)} runs in {len(sorted_notes)} notes")
    
    # Process each run
    for run_start, run_end in merged_runs:
        run_notes = sorted_notes[run_start:run_end]
        
        # Get the IOIs in this run
        run_iois = [iois[i] for i in range(run_start, min(run_end - 1, len(iois)))]
        
        if not run_iois:
            continue
            
        # Calculate the median IOI (robust to outliers)
        median_ioi = np.median(run_iois)
        
        # Quantize the median IOI to get the "true" note value for this run
        run_note_val = duration_to_note_value(median_ioi, bpm=bpm, debug=False)
        run_type = run_note_val['type']
        run_beats = run_note_val['beats']
        run_dotted = run_note_val.get('dotted', False)
        
        # Also check what the majority of notes are assigned to
        value_counts = {}
        for note in run_notes:
            val = note.get('note_value', 'quarter')
            dotted = note.get('dotted', False)
            key = (val, dotted)
            value_counts[key] = value_counts.get(key, 0) + 1
        
        # Find the most common value
        most_common = max(value_counts.items(), key=lambda x: x[1])
        most_common_type, most_common_dotted = most_common[0]
        most_common_count = most_common[1]
        
        # Decide: use median IOI value or most common assigned value?
        # Use whichever has more support
        ioi_error = run_note_val.get('quantization_error', 1.0)
        majority_pct = most_common_count / len(run_notes)
        
        # If median IOI strongly supports a value (low error) and majority agrees, use it
        # Otherwise use the majority vote
        if ioi_error < 0.15 and run_type == most_common_type:
            final_type = run_type
            final_beats = run_beats
            final_dotted = run_dotted
        elif majority_pct >= 0.4:
            # Use majority vote
            final_type = most_common_type
            final_dotted = most_common_dotted
            final_beats = NOTE_VALUE_BEATS.get(final_type, 0.5)
            if final_dotted:
                final_beats *= 1.5
        else:
            # Use median IOI
            final_type = run_type
            final_beats = run_beats
            final_dotted = run_dotted
        
        # Count outliers
        outliers = sum(1 for n in run_notes 
                      if n.get('note_value') != final_type or n.get('dotted', False) != final_dotted)
        
        if debug and outliers > 0:
            print(f"  Run [{run_start}:{run_end}]: {len(run_notes)} notes -> all {final_type}"
                  f"{'.' if final_dotted else ''} (normalizing {outliers} outliers)")
        
        # Normalize all notes in the run
        for note in run_notes:
            if note.get('note_value') != final_type or note.get('dotted', False) != final_dotted:
                note['note_value'] = final_type
                note['note_divisions'] = final_beats
                note['dotted'] = final_dotted
                note['is_triplet'] = False  # Runs are usually not triplets unless detected
                note['triplet'] = False
                note.pop('triplet_position', None)
                note.pop('actual_notes', None)
                note.pop('normal_notes', None)
                note['run_normalized'] = True
    
    return sorted_notes


def detect_beats_neural(audio_path_or_array, sr=None, debug=False):
    """
    Use neural network beat tracking for accurate beat grid detection.
    
    This uses librosa's beat_track with a pretrained model, which is more
    accurate than simple onset-based tempo detection.
    
    Args:
        audio_path_or_array: Path to audio file OR pre-loaded numpy array
        sr: Sample rate (required if passing array, ignored if passing path)
        debug: Print debug info
    
    Returns:
        dict with 'beats' (array of beat times in seconds), 'bpm', 'confidence'
    """
    # Use 22050Hz for beat detection - sufficient for tempo and 2x faster
    BEAT_SR = 22050
    BEAT_HOP = 512
    
    try:
        # Load audio or use pre-loaded
        if isinstance(audio_path_or_array, (str, bytes)) or hasattr(audio_path_or_array, '__fspath__'):
            y, sr = load_audio_deterministic(audio_path_or_array, target_sr=BEAT_SR)
        else:
            y = audio_path_or_array
            if sr is None:
                raise ValueError("Sample rate (sr) must be provided when passing audio array")
            # Downsample if needed for faster beat tracking
            if sr != BEAT_SR:
                gcd = math.gcd(sr, BEAT_SR)
                y = resample_poly(y, BEAT_SR // gcd, sr // gcd).astype(np.float32, copy=False)
                sr = BEAT_SR
        
        # Use librosa's beat tracker (uses a pretrained model under the hood)
        # This is more accurate than simple autocorrelation
        tempo, beat_frames = librosa.beat.beat_track(
            y=y, sr=sr, 
            hop_length=BEAT_HOP,
            start_bpm=120,
            tightness=100,  # How tightly to adhere to tempo estimate
            trim=True
        )
        
        # Convert frames to times
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=BEAT_HOP)
        
        # Calculate confidence from beat strength consistency
        # Get onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=BEAT_HOP)
        
        # Sample onset envelope at beat positions
        if len(beat_frames) > 0:
            beat_strengths = onset_env[beat_frames[beat_frames < len(onset_env)]]
            # Confidence = how consistent beat strengths are (low std = high confidence)
            if len(beat_strengths) > 1:
                cv = np.std(beat_strengths) / (np.mean(beat_strengths) + 1e-6)
                confidence = max(0.3, min(1.0, 1.0 - cv))
            else:
                confidence = 0.5
        else:
            confidence = 0.3
        
        if debug:
            print(f"[Beat Detection] Found {len(beat_times)} beats at {tempo:.1f} BPM (confidence: {confidence:.2f})")
            if len(beat_times) > 4:
                print(f"[Beat Detection] First 4 beats: {beat_times[:4]}")
        
        return {
            'beats': beat_times,
            'bpm': float(tempo) if np.isscalar(tempo) else float(tempo[0]),
            'confidence': confidence,
            'beat_interval': 60.0 / (float(tempo) if np.isscalar(tempo) else float(tempo[0]))
        }
    except Exception as e:
        if debug:
            print(f"[Beat Detection] Error: {e}")
        return {
            'beats': np.array([]),
            'bpm': 120.0,
            'confidence': 0.0,
            'beat_interval': 0.5
        }


def cross_validate_with_acoustic_duration(notes, bpm, debug=False):
    """
    Cross-validate quantized note values against acoustic durations.

    After quantization assigns note_value based on IOI/grid, this checks
    whether the acoustic duration (how long the key was actually held)
    is consistent with the assigned value. If the note_value implies
    2 beats but the key was only held for ~1 beat, the note should
    probably be shorter with a rest after.
    """
    if len(notes) < 2:
        return notes

    beat_duration = 60.0 / bpm
    sorted_notes = sorted(notes, key=lambda n: n.get('time_seconds', 0))
    corrections = 0

    for note in sorted_notes:
        assigned_beats = note.get('note_divisions', 1.0)
        acoustic_dur = note.get('duration_seconds', 0.5)
        acoustic_beats = acoustic_dur / beat_duration

        # Skip grace notes or notes already marked with acoustic method
        if note.get('ornament') == 'grace':
            continue
        method = note.get('quantization_method', '')
        if 'acoustic' in method:
            continue
        # Don't second-guess high-confidence ensemble predictions or run-tagged notes
        if method == 'ensemble_kept' or method == 'run_tagged':
            continue

        # Check: does the acoustic duration justify the assigned note value?
        # If the note rings for less than 60% of the assigned duration,
        # AND the assigned value is at least 1 beat,
        # consider correcting.
        ratio = acoustic_beats / assigned_beats if assigned_beats > 0 else 1.0

        if ratio < 0.60 and assigned_beats >= 1.0:
            acoustic_val = duration_to_note_value(acoustic_dur, bpm=bpm, debug=False)
            acoustic_error = acoustic_val.get('quantization_error', 1.0)

            # Only correct if the acoustic value quantizes cleanly
            if acoustic_error < 0.20:
                gap_dur = (assigned_beats - acoustic_val['beats']) * beat_duration
                gap_val = duration_to_note_value(gap_dur, bpm=bpm, debug=False)
                gap_error = gap_val.get('quantization_error', 1.0)

                # Both the note AND the rest must quantize cleanly
                if gap_error < 0.25:
                    note['note_value'] = acoustic_val['type']
                    note['note_divisions'] = acoustic_val['beats']
                    note['dotted'] = acoustic_val.get('dotted', False)
                    note['has_rest_after'] = True
                    note['rest_duration'] = gap_dur
                    note['quantization_method'] = method + ' (acoustic-corrected)'
                    corrections += 1

    if debug and corrections > 0:
        print(f"  [Acoustic Cross-Validation] Corrected {corrections} notes")

    return sorted_notes


def tag_runs_pre_quantization(notes, bpm, debug=False):
    """
    Pre-quantization run detection: tag consecutive fast notes that should
    share a single note value.

    Runs through sorted notes once, grouping those with similar IOIs
    (within 30% of each other). When 3+ consecutive notes form a run,
    computes the median IOI, maps it to a note value, and tags every note
    in the run with 'run_note_value' / 'run_note_divisions' / 'run_dotted'.

    The quantizer can then use these tags directly instead of quantizing
    each note independently (which produces a noisy mix of 16ths and 32nds
    for what should be uniform sixteenths).

    Runs in O(n) — single pass, no allocations beyond the tag fields.
    """
    if len(notes) < 3:
        return notes

    beat_duration = 60.0 / bpm
    sorted_notes = sorted(notes, key=lambda n: n.get('time_seconds', 0))

    # Compute IOIs
    times = [n.get('time_seconds', 0) for n in sorted_notes]
    n = len(times)

    # Maximum IOI to consider "fast" — anything above a quarter note is not a run
    max_run_ioi = beat_duration * 1.1  # slightly above quarter note

    # Detect runs: groups of 8+ notes with similar IOIs
    # We allow 30% deviation between consecutive IOIs
    SIMILARITY = 0.30
    MIN_RUN_LEN = 8

    run_start = 0
    run_iois = []

    def finalize_run(start, end, iois):
        """Tag notes [start, end) with the run's median note value."""
        if end - start < MIN_RUN_LEN:
            return
        median_ioi = float(np.median(iois))
        if median_ioi > max_run_ioi or median_ioi < beat_duration * 0.04:
            return  # not a fast run, or impossibly fast

        note_val = duration_to_note_value(median_ioi, bpm=bpm, debug=False)
        run_type = note_val['type']
        run_beats = note_val['beats']
        run_dotted = note_val.get('dotted', False)

        if debug:
            print(f"  [Run] notes [{start}:{end}] ({end-start} notes), "
                  f"median IOI={median_ioi*1000:.0f}ms -> {run_type}"
                  f"{'.' if run_dotted else ''}")

        for j in range(start, end):
            sorted_notes[j]['run_note_value'] = run_type
            sorted_notes[j]['run_note_divisions'] = run_beats
            sorted_notes[j]['run_dotted'] = run_dotted

    for i in range(n - 1):
        ioi = times[i + 1] - times[i]

        if ioi > max_run_ioi or ioi < 0.01:
            # Break: this gap is too large or negative
            finalize_run(run_start, i + 1, run_iois)
            run_start = i + 1
            run_iois = []
            continue

        if run_iois:
            prev_ioi = run_iois[-1]
            if prev_ioi > 0:
                ratio = ioi / prev_ioi
                if not (1.0 - SIMILARITY <= ratio <= 1.0 + SIMILARITY):
                    # IOI changed too much — finalize current run
                    finalize_run(run_start, i + 1, run_iois)
                    run_start = i
                    run_iois = [ioi]
                    continue

        run_iois.append(ioi)

    # Finalize last run (include last note)
    finalize_run(run_start, n, run_iois)

    return sorted_notes


def time_to_local_beat(time_seconds, beat_times):
    """Map a wall-clock time to a fractional beat index on a non-uniform beat grid."""
    beat_times = np.asarray(beat_times, dtype=float)
    if len(beat_times) < 2:
        return 0.0

    if time_seconds <= beat_times[0]:
        interval = max(beat_times[1] - beat_times[0], 1e-6)
        return (time_seconds - beat_times[0]) / interval

    idx = np.searchsorted(beat_times, time_seconds, side='right') - 1
    if idx >= len(beat_times) - 1:
        interval = max(beat_times[-1] - beat_times[-2], 1e-6)
        return (len(beat_times) - 1) + ((time_seconds - beat_times[-1]) / interval)

    interval = max(beat_times[idx + 1] - beat_times[idx], 1e-6)
    return idx + ((time_seconds - beat_times[idx]) / interval)


def local_beat_duration_at(time_seconds, beat_times, fallback=None):
    """Estimate the local beat duration near a wall-clock time."""
    beat_times = np.asarray(beat_times, dtype=float)
    if len(beat_times) < 2:
        return fallback if fallback is not None else 0.5

    if time_seconds <= beat_times[0]:
        return max(beat_times[1] - beat_times[0], 1e-6)

    idx = np.searchsorted(beat_times, time_seconds, side='right') - 1
    if idx >= len(beat_times) - 1:
        return max(beat_times[-1] - beat_times[-2], 1e-6)

    return max(beat_times[idx + 1] - beat_times[idx], 1e-6)


def build_regularized_local_beat_grid(detected_beats, fallback_beats, target_beat_interval,
                                      confidence=0.0, debug=False):
    """Regularize detected beats into a smooth local tempo curve for quantization."""
    fallback_beats = np.asarray(fallback_beats, dtype=float)
    detected_beats = np.asarray(detected_beats, dtype=float)

    if len(detected_beats) < 2:
        return fallback_beats
    if len(fallback_beats) < 2:
        return detected_beats

    detected_beats = np.unique(np.round(detected_beats, 6))
    target_beat_interval = max(float(target_beat_interval), 1e-6)
    raw_intervals = np.diff(detected_beats)
    if len(raw_intervals) == 0:
        return detected_beats

    smoothed_intervals = []
    for index in range(len(raw_intervals)):
        lo = max(0, index - 1)
        hi = min(len(raw_intervals), index + 2)
        local_median = float(np.median(raw_intervals[lo:hi]))
        blended = 0.75 * local_median + 0.25 * target_beat_interval
        blended = min(max(blended, target_beat_interval * 0.6), target_beat_interval * 1.6)
        smoothed_intervals.append(blended)

    regularized = [float(detected_beats[0])]
    for interval in smoothed_intervals:
        regularized.append(regularized[-1] + interval)
    regularized = np.asarray(regularized, dtype=float)

    detected_weight = min(max(0.35 + 0.45 * confidence, 0.35), 0.85)
    if len(regularized) == len(detected_beats):
        regularized = ((1.0 - detected_weight) * regularized) + (detected_weight * detected_beats)

    min_interval = max(target_beat_interval * 0.25, 1e-3)
    for index in range(1, len(regularized)):
        if regularized[index] <= regularized[index - 1] + min_interval:
            regularized[index] = regularized[index - 1] + min_interval

    if fallback_beats[0] < regularized[0] - min_interval:
        prefix = fallback_beats[fallback_beats < regularized[0] - min_interval]
        regularized = np.concatenate([prefix, regularized])
    if fallback_beats[-1] > regularized[-1] + min_interval:
        suffix = fallback_beats[fallback_beats > regularized[-1] + min_interval]
        regularized = np.concatenate([regularized, suffix])

    if debug:
        raw_std = float(np.std(raw_intervals)) if len(raw_intervals) > 1 else 0.0
        reg_std = float(np.std(np.diff(regularized))) if len(regularized) > 2 else 0.0
        print(f"[Local Tempo Curve] raw_std={raw_std*1000:.1f}ms, regularized_std={reg_std*1000:.1f}ms, beats={len(regularized)}")

    return regularized


def apply_backend_timing_authority(notes, chords, beat_times):
    """Attach backend-authored beat positions so the renderer can avoid re-quantizing."""
    if len(beat_times) < 2:
        return

    for event in notes + chords:
        onset = event.get('time_seconds', 0.0)
        start_beat = event.get('grid_start_beat_candidate')
        if start_beat is None:
            start_beat = time_to_local_beat(onset, beat_times)
        start_beat = round(float(start_beat) * 24) / 24

        note_beats = max(float(event.get('note_divisions', 0.0) or 0.0), 0.0)
        event['start_beat'] = start_beat
        event['end_beat'] = round((start_beat + note_beats) * 24) / 24
        event['local_beat_duration'] = local_beat_duration_at(onset, beat_times)
        if event.get('has_rest_after') and event.get('rest_duration'):
            local_rest_beats = max(event['rest_duration'] / max(event['local_beat_duration'], 1e-6), 0.0)
            event['rest_after_beats'] = round(local_rest_beats * 24) / 24
        else:
            event['rest_after_beats'] = 0.0
        event['timing_authority'] = 'backend_local_beat_grid'


def quantize_to_beat_grid(notes, beat_times, bpm, subdivision_info=None, debug=False):
    """
    Quantize note timings to a detected beat grid for more accurate rhythm.

    KEY INSIGHT: In real notation, notes usually extend to the next note onset.
    Pianists release keys early, but the NOTATION shows longer durations.
    We use IOI (inter-onset interval) as the primary duration indicator,
    and only insert rests for truly significant gaps.

    Uses compensating quantization: tracks cumulative drift between quantized
    beat positions and raw onset times, and biases quantization to reduce drift
    when it exceeds a threshold (1/8 beat). Only compensates when it would
    actually reduce drift — never pushes in the wrong direction.

    Args:
        notes: List of note dicts with 'time_seconds'
        beat_times: Array of beat times in seconds (from beat detection)
        bpm: Detected tempo in BPM
        subdivision_info: Global subdivision info (used as fallback)
        debug: Print debug info

    Returns:
        Modified notes list with improved note_value assignments
    """
    if len(notes) == 0 or len(beat_times) < 2:
        return quantize_rhythm_from_ioi(notes, bpm, debug)

    beat_duration = 60.0 / bpm

    # Sort notes by time
    sorted_notes = sorted(notes, key=lambda n: n.get('time_seconds', 0))

    # ── Compensating quantization state ──
    # Tracks cumulative beat position from quantized durations vs. expected
    # position from raw onset times. When drift exceeds DRIFT_THRESHOLD,
    # we bias the effective_duration to pull it back.
    cumulative_quantized_beats = time_to_local_beat(
        sorted_notes[0].get('time_seconds', 0), beat_times
    )
    DRIFT_THRESHOLD = 0.125  # 1/8 beat — only compensate beyond this
    DRIFT_MAX_CORRECTION = 0.25  # never adjust by more than 1/4 beat at once
    
    # Check global subdivision info
    global_uses_triplets = subdivision_info.get('uses_triplets', False) if subdivision_info else False
    
    # Define grid subdivisions
    subdivisions = [1.0, 0.5, 0.25, 0.125]
    if global_uses_triplets:
        subdivisions.extend([1/3, 2/3])
    
    # Build grid from beat times
    grid_points = []
    grid_beats = []
    for i in range(len(beat_times) - 1):
        beat_start = beat_times[i]
        beat_end = beat_times[i + 1]
        beat_len = beat_end - beat_start
        
        for subdiv in subdivisions:
            fractions = np.arange(0, 1.0, subdiv)
            points = fractions * beat_len + beat_start
            grid_points.extend(points.tolist())
            grid_beats.extend((i + fractions).tolist())
    
    grid_points.append(beat_times[-1])
    grid_beats.append(float(len(beat_times) - 1))

    ordered_pairs = sorted(zip(grid_points, grid_beats), key=lambda pair: pair[0])
    deduped_points = []
    deduped_beats = []
    for point, beat_index in ordered_pairs:
        if deduped_points and abs(point - deduped_points[-1]) < 1e-6:
            continue
        deduped_points.append(float(point))
        deduped_beats.append(float(beat_index))
    grid_points = np.array(deduped_points)
    grid_beats = np.array(deduped_beats)
    
    if debug:
        print(f"[Grid Quantize] Built grid with {len(grid_points)} points from {len(beat_times)} beats")
        print(f"[Grid Quantize] Using IOI-primary duration (acoustic duration as secondary)")
    
    for i, note in enumerate(sorted_notes):
        onset = note.get('time_seconds', 0)
        acoustic_duration = note.get('duration_seconds', 0.5)
        acoustic_offset = note.get('offset_seconds', onset + acoustic_duration)
        if acoustic_offset <= onset:
            acoustic_offset = onset + acoustic_duration

        # ── Compute expected beat position from raw onset ──
        expected_beat = time_to_local_beat(onset, beat_times)
        note['onset_snap_error'] = abs(cumulative_quantized_beats - expected_beat)

        # Snap onset to nearest local grid point and keep it for renderer authority.
        grid_idx = int(np.argmin(np.abs(grid_points - onset)))
        snapped_onset = float(grid_points[grid_idx])
        snapped_beat = float(grid_beats[grid_idx])
        note['snapped_onset_seconds'] = snapped_onset
        note['grid_start_beat_candidate'] = round(snapped_beat * 24) / 24
        note['local_beat_duration'] = local_beat_duration_at(onset, beat_times, beat_duration)

        if i < len(sorted_notes) - 1:
            next_onset = sorted_notes[i + 1].get('time_seconds', onset + acoustic_duration)
            next_grid_idx = int(np.argmin(np.abs(grid_points - next_onset)))
            next_snapped_beat = float(grid_beats[next_grid_idx])
            onset_candidate_beats = max(next_snapped_beat - snapped_beat, 0.125)
            ioi = next_onset - onset
        else:
            ioi = acoustic_duration
            onset_candidate_beats = max(
                time_to_local_beat(acoustic_offset, beat_times) - snapped_beat,
                0.125,
            )

        acoustic_candidate_beats = max(
            time_to_local_beat(acoustic_offset, beat_times) - snapped_beat,
            acoustic_duration / max(note['local_beat_duration'], 1e-6),
            0.125,
        )

        gap_beats = max(onset_candidate_beats - acoustic_candidate_beats, 0.0)
        overlap_beats = max(acoustic_candidate_beats - onset_candidate_beats, 0.0)
        sustain_ratio = acoustic_candidate_beats / max(onset_candidate_beats, 0.125)
        rest_threshold_beats = max(0.25, 0.5 * (120.0 / bpm))
        clear_rest_threshold_beats = max(rest_threshold_beats, 0.5)
        rest_val = duration_to_note_value(gap_beats * beat_duration, bpm=bpm, debug=False)
        rest_quantizes_cleanly = gap_beats > 0.001 and rest_val.get('quantization_error', 1.0) < 0.2

        note['ioi_beats'] = onset_candidate_beats
        note['acoustic_beats'] = acoustic_candidate_beats

        # ── Preserve high-confidence ensemble predictions ──
        ensemble_conf = note.get('note_value_confidence', 0)
        has_ensemble_nv = (note.get('note_value_source') == 'ensemble'
                           and 'note_value' in note)
        # Save original ensemble values before grid snapping overwrites them
        _saved_ensemble_nv = None
        if has_ensemble_nv:
            _saved_ensemble_nv = (
                note.get('note_value'),
                note.get('note_divisions', 1.0),
                note.get('dotted', False),
            )
        if has_ensemble_nv and ensemble_conf >= 0.60 and 'run_note_value' not in note:
            if note.get('note_divisions', 1.0) > onset_candidate_beats + (1 / 48):
                capped_val = duration_to_note_value_contextual(
                    onset_candidate_beats * beat_duration,
                    bpm=bpm,
                    subdivision_info=subdivision_info or {},
                    debug=False,
                )
                note['note_value'] = capped_val['type']
                note['note_divisions'] = capped_val['divisions']
                note['dotted'] = capped_val.get('dotted', False)
                note['is_triplet'] = capped_val.get('is_triplet', False)
                note['quantization_method'] = 'ensemble_capped_to_onset'
            else:
                note['quantization_method'] = 'ensemble_kept'
            note['quantization_confidence'] = ensemble_conf
            note['duration_source'] = 'ensemble'
            # note_divisions and dotted already set by _normalize_ensemble_note_value
            # Still determine has_rest_after from IOI analysis
            note['has_rest_after'] = (
                gap_beats >= clear_rest_threshold_beats
                and sustain_ratio < 0.55
                and rest_quantizes_cleanly
            )
            if note['has_rest_after']:
                note['rest_duration'] = max(
                    (onset_candidate_beats - note.get('note_divisions', 1.0)) * beat_duration,
                    0.0,
                )
            else:
                note.pop('rest_duration', None)
            # Advance cumulative beat position for drift tracking
            cumulative_quantized_beats += note.get('note_divisions', 1.0)
            if note.get('has_rest_after') and note.get('rest_duration'):
                rest_beats = round(note['rest_duration'] / beat_duration * 8) / 8
                cumulative_quantized_beats += rest_beats
            if debug and i < 10:
                print(f"  Note {i}: ensemble_kept {note['note_value']}"
                      f"{' (dotted)' if note.get('dotted') else ''}"
                      f" conf={ensemble_conf:.2f}")
            continue  # skip grid snapping for this note

        # Get LOCAL subdivision context
        local_subdiv_info = get_local_subdivision_info(sorted_notes, i, bpm, window_notes=20)
        
        # CORE LOGIC: start from onset-driven durations, but reconcile them with
        # acoustic offsets so notes do not get spuriously elongated and hold up
        # the rest of the music. Clear gaps become rests; ambiguous gaps are blended.
        if overlap_beats > (1 / 24):
            # The acoustic offset extends past the next onset. The current
            # renderer cannot safely overlap same-staff voices yet, so keep the
            # note long, but cap it at the next onset so later events still flow.
            effective_beats = onset_candidate_beats
            method = 'onset_capped_overlap'
            note['duration_source'] = 'overlap_capped'
            note['has_rest_after'] = False
            note.pop('rest_duration', None)
        elif gap_beats <= (1 / 24):
            effective_beats = onset_candidate_beats
            method = 'onset_primary'
            note['duration_source'] = 'onset'
            note['has_rest_after'] = False
            note.pop('rest_duration', None)
        elif (
            gap_beats >= clear_rest_threshold_beats
            and sustain_ratio < 0.55
            and rest_quantizes_cleanly
        ):
            effective_beats = acoustic_candidate_beats
            method = 'offset_reconciled (rest follows)'
            note['duration_source'] = 'offset_guarded'
            note['has_rest_after'] = True
            note['rest_duration'] = gap_beats * beat_duration
        else:
            if sustain_ratio >= 0.72:
                effective_beats = onset_candidate_beats
                note['duration_source'] = 'onset'
            else:
                # Bias ambiguous cases late rather than short. Leave only a small
                # safety margin before the next onset instead of splitting the gap.
                safety_margin_beats = min(1 / 24, gap_beats * 0.2)
                effective_beats = max(
                    acoustic_candidate_beats,
                    onset_candidate_beats - safety_margin_beats,
                )
                effective_beats = min(onset_candidate_beats, effective_beats)
                note['duration_source'] = 'reconciled_long_bias'

            leftover_beats = max(onset_candidate_beats - effective_beats, 0.0)
            note['has_rest_after'] = (
                leftover_beats >= clear_rest_threshold_beats
                and sustain_ratio < 0.65
                and rest_quantizes_cleanly
            )
            if note['has_rest_after']:
                note['rest_duration'] = leftover_beats * beat_duration
            else:
                note.pop('rest_duration', None)
            method = 'onset_offset_reconciled'

        effective_beats = max(min(effective_beats, onset_candidate_beats), 0.125)
        min_duration = beat_duration * 0.125  # 32nd note minimum
        effective_duration = max(effective_beats * beat_duration, min_duration)

        # ── Compensating quantization: bias effective_duration to reduce drift ──
        # Only apply when drift is significant AND correction would reduce it.
        drift = cumulative_quantized_beats - expected_beat  # positive = we're ahead
        if abs(drift) > DRIFT_THRESHOLD:
            # Compute correction: shrink duration if ahead, grow if behind
            correction = -drift  # negate: if ahead, want shorter note
            correction = max(-DRIFT_MAX_CORRECTION, min(DRIFT_MAX_CORRECTION, correction))
            candidate_beats = (effective_duration / beat_duration) + correction
            # Safety: only apply if (a) it stays positive and (b) it actually
            # reduces drift (don't push in the wrong direction)
            if min_duration / beat_duration < candidate_beats <= onset_candidate_beats:
                original_beats = effective_duration / beat_duration
                # Will this note's quantized value actually be different and closer?
                new_quantized = cumulative_quantized_beats + candidate_beats
                old_quantized = cumulative_quantized_beats + original_beats
                if i < len(sorted_notes) - 1:
                    next_expected = time_to_local_beat(
                        sorted_notes[i + 1].get('time_seconds', 0), beat_times
                    )
                else:
                    next_expected = expected_beat + original_beats
                new_drift = abs(new_quantized - next_expected)
                old_drift = abs(old_quantized - next_expected)
                if new_drift < old_drift:
                    effective_duration = candidate_beats * beat_duration
                    note['drift_corrected'] = True
                    note['drift_correction_beats'] = correction

        # Convert to note value using LOCAL context
        note_val = duration_to_note_value_contextual(
            effective_duration, bpm=bpm,
            subdivision_info=local_subdiv_info, debug=False
        )
        
        note['note_value'] = note_val['type']
        note['note_divisions'] = note_val['divisions']
        note['dotted'] = note_val.get('dotted', False)
        note['is_triplet'] = note_val.get('is_triplet', False)
        note['quantization_method'] = method
        note['raw_beats'] = note_val.get('raw_beats', 0)
        note['quantization_error'] = note_val.get('quantization_error', 0)

        if i < len(sorted_notes) - 1 and note.get('note_divisions', 1.0) > onset_candidate_beats + (1 / 48):
            capped_val = duration_to_note_value_contextual(
                onset_candidate_beats * beat_duration,
                bpm=bpm,
                subdivision_info=local_subdiv_info,
                debug=False,
            )
            note['note_value'] = capped_val['type']
            note['note_divisions'] = capped_val['divisions']
            note['dotted'] = capped_val.get('dotted', False)
            note['is_triplet'] = capped_val.get('is_triplet', False)
            note['raw_beats'] = capped_val.get('raw_beats', onset_candidate_beats)
            note['quantization_error'] = capped_val.get('quantization_error', 0)
            note['quantization_method'] = note.get('quantization_method', '') + ' (capped)'

        # ── Override with pre-tagged run value if available ──
        # Run detection already identified this note as part of a fast passage
        # that should share a uniform note value. Use the run's median-derived
        # value instead of the per-note quantized one.
        if 'run_note_value' in note:
            note['note_value'] = note['run_note_value']
            note['note_divisions'] = note['run_note_divisions']
            note['dotted'] = note.get('run_dotted', False)
            note['is_triplet'] = False
            note['quantization_method'] = 'run_tagged'
            note['has_rest_after'] = False  # runs don't have internal rests

        # ── Graduated ensemble trust: for medium confidence (0.40-0.60), ──
        # prefer ensemble if its error is comparable to grid's
        if has_ensemble_nv and 0.40 <= ensemble_conf < 0.60 and _saved_ensemble_nv is not None:
            saved_nv, saved_beats, saved_dotted = _saved_ensemble_nv
            raw_beats = effective_duration / beat_duration
            grid_error = note_val.get('quantization_error', 1.0)
            ensemble_error = abs(raw_beats - saved_beats) / max(saved_beats, 0.01)
            if ensemble_error <= grid_error * 1.5:
                note['note_value'] = saved_nv
                note['note_divisions'] = saved_beats
                note['dotted'] = saved_dotted
                note['quantization_method'] = 'ensemble_biased'

        if debug and i < 10:
            triplet_str = " (triplet)" if note_val.get('is_triplet') else ""
            drift_str = f" drift={drift:+.3f}b" if abs(drift) > DRIFT_THRESHOLD else ""
            print(f"  Note {i}: acoustic={acoustic_duration*1000:.0f}ms, ioi={ioi*1000:.0f}ms "
                  f"-> {note['note_value']}{triplet_str} ({note['quantization_method']}){drift_str}")

        # ── Advance cumulative beat position for drift tracking ──
        cumulative_quantized_beats += note.get('note_divisions', 1.0)
        if note.get('has_rest_after') and note.get('rest_duration'):
            rest_beats = round(note['rest_duration'] / beat_duration * 8) / 8
            cumulative_quantized_beats += rest_beats

    return sorted_notes


def compute_notation_proximity_score(notes, chords, bpm, debug=False):
    """
    Measure how closely the quantized notation matches the raw detected timing.

    For each note (sorted by onset), compares:
      - Expected beat position (from raw time_seconds / beat_duration)
      - Cumulative beat position (from summing quantized note_divisions + rests)

    Returns a dict of aggregate metrics plus per-note onset_drift_beats/ms fields
    stored on each note dict.

    Args:
        notes: List of note dicts (already quantized, with note_divisions etc.)
        chords: List of chord dicts (same structure)
        bpm: Detected BPM
        debug: Print per-note drift info

    Returns:
        dict with mean/max/median drift in ms, pct within 32nd/16th/8th,
        and per-hand breakdowns.
    """
    beat_duration = 60.0 / bpm

    def _score_hand(items, hand_label):
        if not items:
            return {}
        sorted_items = sorted(items, key=lambda n: n.get('time_seconds', 0))
        cumulative_beats = 0.0
        errors_beats = []

        for item in sorted_items:
            onset = item.get('time_seconds', 0)
            expected_beat = onset / beat_duration
            drift = abs(cumulative_beats - expected_beat)
            item['onset_drift_beats'] = drift
            item['onset_drift_ms'] = drift * beat_duration * 1000
            errors_beats.append(drift)

            # Advance by quantized duration
            cumulative_beats += item.get('note_divisions', 1.0)
            # Add rest if present
            if item.get('has_rest_after') and item.get('rest_duration'):
                rest_beats = round(item['rest_duration'] / beat_duration * 8) / 8
                cumulative_beats += rest_beats

        errors_ms = [e * beat_duration * 1000 for e in errors_beats]
        n = len(errors_beats)

        result = {
            'mean_drift_ms': sum(errors_ms) / n,
            'max_drift_ms': max(errors_ms),
            'median_drift_ms': sorted(errors_ms)[n // 2],
            'pct_within_32nd': sum(1 for e in errors_beats if e <= 0.125) / n,
            'pct_within_16th': sum(1 for e in errors_beats if e <= 0.25) / n,
            'pct_within_8th': sum(1 for e in errors_beats if e <= 0.5) / n,
            'num_notes': n,
        }

        if debug:
            print(f"  [{hand_label}] mean={result['mean_drift_ms']:.1f}ms, "
                  f"max={result['max_drift_ms']:.1f}ms, "
                  f"within 16th={result['pct_within_16th']*100:.0f}%, "
                  f"within 8th={result['pct_within_8th']*100:.0f}%")

        return result

    bass_notes = [n for n in notes if n.get('hand') == 'bass']
    treble_notes = [n for n in notes if n.get('hand') == 'treble']
    bass_chords = [c for c in chords if c.get('hand') == 'bass']
    treble_chords = [c for c in chords if c.get('hand') == 'treble']

    if debug:
        print(f"\n[Proximity Score] BPM={bpm:.0f}, beat={beat_duration*1000:.0f}ms")

    scores = {
        'treble_notes': _score_hand(treble_notes, 'Treble Notes'),
        'bass_notes': _score_hand(bass_notes, 'Bass Notes'),
        'treble_chords': _score_hand(treble_chords, 'Treble Chords'),
        'bass_chords': _score_hand(bass_chords, 'Bass Chords'),
    }

    # Overall aggregate
    all_items = notes + chords
    if all_items:
        all_drifts = [n.get('onset_drift_ms', 0) for n in all_items
                      if 'onset_drift_ms' in n]
        if all_drifts:
            n = len(all_drifts)
            scores['overall'] = {
                'mean_drift_ms': sum(all_drifts) / n,
                'max_drift_ms': max(all_drifts),
                'median_drift_ms': sorted(all_drifts)[n // 2],
                'pct_within_32nd': sum(1 for d in all_drifts if d <= 0.125 * beat_duration * 1000) / n,
                'pct_within_16th': sum(1 for d in all_drifts if d <= 0.25 * beat_duration * 1000) / n,
                'pct_within_8th': sum(1 for d in all_drifts if d <= 0.5 * beat_duration * 1000) / n,
                'num_notes': n,
            }
            if debug:
                o = scores['overall']
                print(f"  [Overall] mean={o['mean_drift_ms']:.1f}ms, "
                      f"max={o['max_drift_ms']:.1f}ms, "
                      f"within 16th={o['pct_within_16th']*100:.0f}%, "
                      f"within 8th={o['pct_within_8th']*100:.0f}%")

    return scores


def detect_dominant_subdivisions(notes, bpm, debug=False, window_seconds=4.0):
    """
    Analyze notes within a time window to find the LOCAL rhythmic subdivisions.
    
    This helps prevent human timing jitter from causing incorrect quantization.
    Uses a sliding window approach so different sections of a piece can have
    different subdivision patterns (e.g., verse in eighths, chorus in triplets).
    
    Args:
        notes: List of note dicts with 'time_seconds' and 'duration_seconds'
        bpm: Detected tempo
        debug: Print debug info
        window_seconds: Size of analysis window in seconds (default 4.0 = ~4-8 bars)
    
    Returns:
        dict with 'subdivision_weights' mapping note types to weights (0-1),
        'primary_subdivision' (most common), 'uses_triplets' (bool)
    """
    if len(notes) < 3:
        return {
            'subdivision_weights': {},
            'primary_subdivision': 'quarter',
            'uses_triplets': False,
            'uses_dotted': False
        }
    
    beat_duration = 60.0 / bpm
    
    # Calculate IOIs
    times = sorted([n.get('time_seconds', 0) for n in notes])
    iois = np.diff(times)
    
    # Also consider detected durations
    durations = [n.get('duration_seconds', 0.5) for n in notes]
    
    # Combine IOIs and durations for analysis
    all_intervals = list(iois) + durations
    
    # Define subdivision templates (in beats)
    subdivisions = NOTE_VALUE_BEATS_WITH_SUBDIVISIONS
    
    # Count how many intervals fall near each subdivision
    # Use TIGHT tolerance (15%) to avoid false positives
    tolerance = 0.15
    counts = {name: 0 for name in subdivisions}
    
    for interval in all_intervals:
        interval_beats = interval / beat_duration
        
        for name, subdiv_beats in subdivisions.items():
            if subdiv_beats > 0:
                error = abs(interval_beats - subdiv_beats) / subdiv_beats
                if error <= tolerance:
                    # Weight by how close it is (closer = much higher weight)
                    # Exponential weighting: perfect match = 1.0, at tolerance = ~0.1
                    weight = math.exp(-error * 15)
                    counts[name] += weight
    
    # Normalize to weights (0-1)
    total = sum(counts.values())
    if total > 0:
        weights = {name: count / total for name, count in counts.items()}
    else:
        weights = {name: 0 for name in subdivisions}
    
    # Find primary subdivision (excluding dotted and triplets for simplicity)
    basic_subdivs = ['whole', 'half', 'quarter', 'eighth', '16th', '32nd']
    primary = max(basic_subdivs, key=lambda x: weights.get(x, 0))
    
    # Require SIGNIFICANT triplet presence (>15% of weighted intervals)
    triplet_weight = weights.get('quarter_triplet', 0) + weights.get('eighth_triplet', 0) + weights.get('16th_triplet', 0)
    uses_triplets = triplet_weight > 0.15
    
    # Require significant dotted note presence
    dotted_weight = weights.get('dotted_half', 0) + weights.get('dotted_quarter', 0) + weights.get('dotted_eighth', 0) + weights.get('dotted_16th', 0)
    uses_dotted = dotted_weight > 0.15
    
    if debug:
        print(f"\n[Subdivision Analysis] Analyzed {len(all_intervals)} intervals")
        print(f"  Primary subdivision: {primary}")
        print(f"  Uses triplets: {uses_triplets} ({triplet_weight*100:.1f}%)")
        print(f"  Uses dotted: {uses_dotted} ({dotted_weight*100:.1f}%)")
        top_5 = sorted(weights.items(), key=lambda x: -x[1])[:5]
        print(f"  Top subdivisions: {[(n, f'{w*100:.1f}%') for n, w in top_5]}")
    
    return {
        'subdivision_weights': {k: float(v) for k, v in weights.items()},
        'primary_subdivision': str(primary),
        'uses_triplets': bool(uses_triplets),
        'uses_dotted': bool(uses_dotted)
    }


def get_local_subdivision_info(notes, note_index, bpm, window_notes=20):
    """
    Get subdivision info for a LOCAL window around a specific note.
    
    Instead of analyzing the whole piece, this looks at nearby notes
    to determine what subdivisions are being used in THIS section.
    
    Args:
        notes: List of all notes (sorted by time)
        note_index: Index of the current note
        bpm: Tempo in BPM  
        window_notes: Number of notes to include in window (±half on each side, default 20)
    
    Returns:
        Subdivision info dict for the local context
    """
    half_window = window_notes // 2
    start_idx = max(0, note_index - half_window)
    end_idx = min(len(notes), note_index + half_window + 1)
    
    window_notes_list = notes[start_idx:end_idx]
    
    if len(window_notes_list) < 3:
        # Not enough context, return permissive defaults
        return {
            'subdivision_weights': {},
            'primary_subdivision': 'eighth',
            'uses_triplets': True,  # Allow everything if not enough context
            'uses_dotted': True
        }
    
    return detect_dominant_subdivisions(window_notes_list, bpm, debug=False)


def duration_to_note_value_contextual(duration_seconds, bpm, subdivision_info, debug=False):
    """
    Convert duration to note value with STRONG context-aware bias.
    
    This aggressively prefers note values that are common in the local context,
    making the system robust to human timing jitter. The philosophy is:
    - If the piece uses eighths, a slightly-off eighth should stay an eighth
    - Unusual subdivisions (dotted 32nds, etc.) require strong evidence
    - Simpler note values are preferred when timing is ambiguous
    
    Args:
        duration_seconds: Duration in seconds
        bpm: Tempo in BPM
        subdivision_info: Output from get_local_subdivision_info()
        debug: Print debug info
    
    Returns:
        Same format as duration_to_note_value()
    """
    beat_duration = 60.0 / bpm
    beats = duration_seconds / beat_duration
    
    weights = subdivision_info.get('subdivision_weights', {})
    uses_triplets = subdivision_info.get('uses_triplets', False)
    uses_dotted = subdivision_info.get('uses_dotted', False)
    primary = subdivision_info.get('primary_subdivision', 'eighth')
    
    # All possible note values (type, beats, dotted, is_triplet, complexity_penalty)
    # Complexity penalty: higher = less likely to be chosen unless strong evidence
    note_values = [
        ('whole', 4.0, False, False, 0.0),
        ('half', 2.0, False, False, 0.0),
        ('quarter', 1.0, False, False, 0.0),
        ('eighth', 0.5, False, False, 0.0),
        ('16th', 0.25, False, False, 0.1),   # Slightly penalize fast notes
        ('32nd', 0.125, False, False, 0.3),  # More penalty - rare in most music
    ]
    
    # Add triplets only if detected in the local context (>10% of notes)
    if uses_triplets:
        note_values.extend([
            ('quarter', 2/3, False, True, 0.1),   # Quarter triplet
            ('eighth', 1/3, False, True, 0.1),    # Eighth triplet
            ('16th', 1/6, False, True, 0.2),      # 16th triplet
        ])
    
    # Add dotted notes only if detected in the local context
    if uses_dotted:
        note_values.extend([
            ('half', 3.0, True, False, 0.1),      # Dotted half
            ('quarter', 1.5, True, False, 0.1),   # Dotted quarter
            ('eighth', 0.75, True, False, 0.15),  # Dotted eighth
            ('16th', 0.375, True, False, 0.25),   # Dotted 16th - quite rare
        ])
    
    # Find best match with strong context-aware scoring
    best_match = None
    best_score = float('-inf')
    
    for note_type, note_beats, dotted, is_triplet, complexity in note_values:
        if note_beats <= 0:
            continue
            
        # Calculate timing error as percentage
        error_pct = abs(beats - note_beats) / note_beats if note_beats > 0 else 10
        
        # TIGHT tolerance: anything beyond 30% error is unlikely to be this note value
        if error_pct > 0.35:
            continue  # Skip completely if too far off
        
        # Base score: how close is the timing? (exponential decay)
        # Perfect match = 1.0, 15% error ≈ 0.5, 30% error ≈ 0.1
        timing_score = math.exp(-error_pct * 8)
        
        # Context boost: how common is this subdivision in the local context?
        if is_triplet:
            weight_key = f"{note_type}_triplet"
        elif dotted:
            weight_key = f"dotted_{note_type}"
        else:
            weight_key = note_type
        
        context_weight = weights.get(weight_key, 0.0)
        
        # STRONG context boost: common subdivisions get big bonus
        # Rare subdivisions get penalty if context weight is low
        if context_weight > 0.15:
            context_boost = 1.0 + context_weight * 2  # Max ~1.8x for dominant patterns
        elif context_weight > 0.05:
            context_boost = 0.9 + context_weight * 1.5
        else:
            context_boost = 0.6  # Mild penalty, not harsh exclusion

        # Primary subdivision gets modest boost
        if note_type == primary and not dotted and not is_triplet:
            context_boost *= 1.2
        
        # Apply complexity penalty (prefer simpler rhythms when ambiguous)
        complexity_factor = 1.0 - complexity
        
        # Final score
        final_score = timing_score * context_boost * complexity_factor
        
        if debug:
            print(f"    {note_type}{'.' if dotted else ''}{'^3' if is_triplet else ''}: "
                  f"err={error_pct:.1%}, timing={timing_score:.3f}, "
                  f"ctx={context_boost:.2f}, final={final_score:.3f}")
        
        if final_score > best_score:
            best_score = final_score
            best_match = (note_type, note_beats, dotted, is_triplet)
    
    # Fallback if nothing matched (shouldn't happen often)
    if best_match is None:
        # Default to closest simple subdivision
        simple_subdivs = [(4.0, 'whole'), (2.0, 'half'), (1.0, 'quarter'), 
                         (0.5, 'eighth'), (0.25, '16th'), (0.125, '32nd')]
        closest = min(simple_subdivs, key=lambda x: abs(beats - x[0]))
        best_match = (closest[1], closest[0], False, False)
    
    note_type, note_beats, dotted, is_triplet = best_match
    
    result = {
        'type': note_type,
        'divisions': note_beats,
        'beats': note_beats,
        'dotted': dotted,
        'is_triplet': is_triplet,
        'raw_beats': beats,
        'quantization_error': abs(beats - note_beats) / note_beats if note_beats > 0 else 0
    }
    
    return result


def quantize_rhythm_ml(notes, bpm, debug=False, ensemble_confidence_threshold=0.60):
    """
    Quantize note rhythms using the trained ML model.

    If notes already have 'note_value' from ensemble model with high confidence,
    those predictions are preserved. Otherwise, transformer/MLP refinement is used.

    Uses GPU-batched inference when CUDA is available, otherwise falls back
    to CPU numpy model. Falls back to heuristic if model is not available.

    Args:
        notes: List of note dicts with 'time_seconds', 'duration_seconds', 'midi_note'
        bpm: Detected tempo in BPM
        debug: Print debug info
        ensemble_confidence_threshold: Notes with ensemble confidence >= this are kept

    Returns:
        Modified notes list with ML-based note_value assignments
    """
    if len(notes) == 0:
        return notes

    # Sort notes by time
    sorted_notes = sorted(notes, key=lambda n: n.get('time_seconds', 0))
    
    # Map ensemble note_value names to standard names (they match already)
    # ensemble uses: ['32nd', '16th', 'eighth', 'quarter', 'half', 'whole']
    # transformer expects same names

    # ── GPU Path: Try Transformer (sequence-aware) first, then MLP fallback ──
    if USE_GPU:
        # --- Transformer: sequence-aware rhythm + rest prediction ---
        transformer_model = get_gpu_transformer_model()
        if transformer_model is not None and transformer_model.initialized:
            features = gpu_extract_features_v2(sorted_notes, bpm, use_ioi_as_duration=True)

            if debug:
                print(f"\n[Rhythm Transformer GPU] Processing {len(sorted_notes)} notes "
                      f"at {bpm} BPM (sequence model)")

            predictions = transformer_model.predict_batch(features)

            n_kept_ensemble = 0
            n_refined = 0
            for i, (note, pred) in enumerate(zip(sorted_notes, predictions)):
                ensemble_conf = note.get('note_value_confidence', 0)
                has_ensemble_nv = note.get('note_value_source') == 'ensemble' and 'note_value' in note
                
                if has_ensemble_nv and ensemble_conf >= ensemble_confidence_threshold:
                    # Keep ensemble prediction, add transformer's extra fields
                    # Preserve dotted flag from normalizer if ensemble predicted dotted
                    if not note.get('dotted', False):
                        note['dotted'] = pred['dotted']
                    note['is_triplet'] = pred['is_triplet']
                    note['has_rest_after'] = pred['has_rest']
                    note['quantization_method'] = 'ensemble_kept'
                    note['quantization_confidence'] = ensemble_conf
                    note['rest_confidence'] = pred.get('rest_confidence', 0)
                    base_beats = NOTE_VALUE_BEATS.get(note['note_value'], 1.0)
                    note['note_divisions'] = base_beats * 1.5 if note.get('dotted') else base_beats
                    n_kept_ensemble += 1
                else:
                    # Use transformer prediction (refine or replace)
                    note['note_value'] = pred['note_type']
                    note['note_divisions'] = pred['beats']
                    note['dotted'] = pred['dotted']
                    note['is_triplet'] = pred['is_triplet']
                    note['has_rest_after'] = pred['has_rest']
                    note['quantization_method'] = 'transformer_gpu'
                    note['quantization_confidence'] = pred['confidence']
                    note['rest_confidence'] = pred.get('rest_confidence', 0)
                    n_refined += 1

                if pred['has_rest'] and i < len(sorted_notes) - 1:
                    next_onset = sorted_notes[i + 1].get('time_seconds', 0)
                    onset = note.get('time_seconds', 0)
                    ioi = next_onset - onset
                    note['rest_duration'] = max(ioi - note['note_divisions'] * (60.0 / bpm), 0)

                if debug and i < 10:
                    dur = note.get('duration_seconds', 0) * 1000
                    rest_str = " [REST]" if note.get('has_rest_after') else ""
                    src = "ENSEMBLE" if note.get('quantization_method') == 'ensemble_kept' else "TRANS"
                    print(f"  Note {i}: {dur:.0f}ms -> {note['note_value']}"
                          f"{' (dotted)' if note.get('dotted') else ''}"
                          f"{' (triplet)' if note.get('is_triplet') else ''}"
                          f" [{src}] (conf={note.get('quantization_confidence', 0):.2f}){rest_str}")

            if debug:
                print(f"  [Summary] Kept {n_kept_ensemble} ensemble predictions, "
                      f"refined {n_refined} with transformer")

            sorted_notes = enforce_triplet_groups(sorted_notes, debug=debug)
            sorted_notes = post_process_rhythm_unified(sorted_notes, bpm, debug=debug)
            sorted_notes = enforce_bar_sum(sorted_notes, bpm, debug=debug)
            return sorted_notes

        # --- MLP fallback: per-note inference ---
        gpu_model = get_gpu_rhythm_model()
        if gpu_model is not None and gpu_model.initialized:
            # Vectorized feature extraction
            features = gpu_extract_features(sorted_notes, bpm, use_ioi_as_duration=True)

            if debug:
                print(f"\n[Rhythm ML GPU] Processing {len(sorted_notes)} notes at {bpm} BPM (batch inference)")

            # Single batched forward pass on GPU for ALL notes
            predictions = gpu_model.predict_batch(features)

            # Update notes (MLP path also respects high-confidence ensemble)
            n_kept_ensemble = 0
            for i, (note, pred) in enumerate(zip(sorted_notes, predictions)):
                ensemble_conf = note.get('note_value_confidence', 0)
                has_ensemble_nv = note.get('note_value_source') == 'ensemble' and 'note_value' in note
                
                if has_ensemble_nv and ensemble_conf >= ensemble_confidence_threshold:
                    # Keep ensemble prediction
                    if not note.get('dotted', False):
                        note['dotted'] = pred['dotted']
                    note['is_triplet'] = pred['is_triplet']
                    note['quantization_method'] = 'ensemble_kept'
                    note['quantization_confidence'] = ensemble_conf
                    base_beats = NOTE_VALUE_BEATS.get(note['note_value'], 1.0)
                    note['note_divisions'] = base_beats * 1.5 if note.get('dotted') else base_beats
                    n_kept_ensemble += 1
                else:
                    note['note_value'] = pred['note_type']
                    note['note_divisions'] = pred['beats']
                    note['dotted'] = pred['dotted']
                    note['is_triplet'] = pred['is_triplet']
                    note['quantization_method'] = 'ml_gpu'
                    note['quantization_confidence'] = pred['confidence']

                if debug and i < 10:
                    dur = note.get('duration_seconds', 0) * 1000
                    print(f"  Note {i}: {dur:.0f}ms -> {pred['note_type']}"
                          f"{' (dotted)' if pred['dotted'] else ''}"
                          f"{' (triplet)' if pred['is_triplet'] else ''}"
                          f" (conf={pred['confidence']:.2f})")

            if debug:
                print(f"  [Summary] Kept {n_kept_ensemble} ensemble predictions")

            sorted_notes = enforce_triplet_groups(sorted_notes, debug=debug)
            sorted_notes = post_process_rhythm_unified(sorted_notes, bpm, debug=debug)
            sorted_notes = enforce_bar_sum(sorted_notes, bpm, debug=debug)
            return sorted_notes

    # ── CPU Path: Original sequential inference ──
    model = get_rhythm_model()

    if model is None:
        if debug:
            print("[Rhythm ML] Model not available, using heuristic")
        return quantize_rhythm_from_ioi(notes, bpm, debug)

    try:
        from rhythm_training.rhythm_model import extract_features_for_ml
    except ImportError:
        if debug:
            print("[Rhythm ML] Feature extraction not available, using heuristic")
        return quantize_rhythm_from_ioi(notes, bpm, debug)

    features = extract_features_for_ml(sorted_notes, bpm, use_ioi_as_duration=True)

    if debug:
        print(f"\n[Rhythm ML] Processing {len(sorted_notes)} notes at {bpm} BPM")

    predictions = model.predict(features)

    if isinstance(predictions, dict):
        predictions = [predictions]

    for i, (note, pred) in enumerate(zip(sorted_notes, predictions)):
        note['note_value'] = pred['note_type']
        note['note_divisions'] = pred['beats']
        note['dotted'] = pred['dotted']
        note['is_triplet'] = pred['is_triplet']
        note['quantization_method'] = 'ml'
        note['quantization_confidence'] = pred['confidence']

        if debug and i < 10:
            dur = note.get('duration_seconds', 0) * 1000
            print(f"  Note {i}: {dur:.0f}ms -> {pred['note_type']}"
                  f"{' (dotted)' if pred['dotted'] else ''}"
                  f"{' (triplet)' if pred['is_triplet'] else ''}"
                  f" (conf={pred['confidence']:.2f})")

    sorted_notes = enforce_triplet_groups(sorted_notes, debug=debug)
    sorted_notes = post_process_rhythm_unified(sorted_notes, bpm, debug=debug)

    # Apply rhythm coherence smoothing to reduce erratic sections
    sorted_notes = apply_coherence_smoothing(sorted_notes, bpm, debug=debug)

    sorted_notes = enforce_bar_sum(sorted_notes, bpm, debug=debug)

    return sorted_notes


def smooth_rhythm_gaps(notes, bpm, debug=False, max_gap_beats=1.0):
    """
    Smooth out unnatural gaps in rhythm by extending notes to fill small gaps.
    
    This ensures notes flow naturally by extending their duration when the 
    quantized duration creates small gaps to the next note. Gaps smaller than
    max_gap_beats are filled by extending the previous note.
    
    For phrase boundaries (larger gaps > 1 beat), gaps are preserved as rests.
    
    Args:
        notes: List of notes with 'note_divisions' (in beats)
        bpm: Tempo in BPM
        debug: Print debug info
        max_gap_beats: Maximum gap in beats to smooth (default 0.5 = eighth note)
    
    Returns:
        Modified notes with smoothed durations
    """
    if len(notes) < 2:
        return notes
    
    beat_duration = 60.0 / bpm
    adjustments = 0
    
    for i in range(len(notes) - 1):
        note = notes[i]
        next_note = notes[i + 1]
        
        # Calculate gap between end of this note and start of next note
        note_start = note.get('time_seconds', 0)
        note_duration_beats = note.get('note_divisions', 1.0)
        note_end_beat = note_start / beat_duration + note_duration_beats
        
        next_start_beat = next_note.get('time_seconds', 0) / beat_duration
        
        gap_beats = next_start_beat - note_end_beat
        
        # If there's a small gap, extend this note to fill it
        if 0 < gap_beats <= max_gap_beats:
            # Choose the next larger standard note value that fills the gap
            new_duration_beats = note_duration_beats + gap_beats
            
            # Find the nearest standard note value
            new_note_val = extend_to_nearest_value(note_duration_beats, new_duration_beats, 
                                                    note.get('is_triplet', False))
            
            if new_note_val:
                old_type = note.get('note_value', 'quarter')
                note['note_value'] = new_note_val['type']
                note['note_divisions'] = new_note_val['beats']
                note['dotted'] = new_note_val.get('dotted', False)
                adjustments += 1
                
                if debug and adjustments <= 5:
                    print(f"  [Smooth] Note {i}: {old_type} -> {new_note_val['type']} "
                          f"(filled {gap_beats:.3f} beat gap)")
        
        # Large gaps (> 1 beat) are likely phrase boundaries - leave as rests
        # Medium gaps (> max_gap_beats but < 1 beat) - could be intentional rests
    
    if debug and adjustments > 0:
        print(f"  [Rhythm Smoothing] Extended {adjustments} notes to fill small gaps")
    
    return notes


def extend_to_nearest_value(current_beats, target_beats, is_triplet=False):
    """
    Find the nearest standard note value that's >= target_beats.
    
    Prefers simple note values (quarter, eighth) over complex ones (dotted 16th).
    """
    # Standard note values in order of preference
    if is_triplet:
        candidates = [
            {'type': 'whole', 'beats': 8/3, 'dotted': False},
            {'type': 'half', 'beats': 4/3, 'dotted': False},
            {'type': 'quarter', 'beats': 2/3, 'dotted': False},
            {'type': 'eighth', 'beats': 1/3, 'dotted': False},
        ]
    else:
        candidates = [
            {'type': 'whole', 'beats': 4.0, 'dotted': False},
            {'type': 'half', 'beats': 2.0, 'dotted': False},
            {'type': 'quarter', 'beats': 1.0, 'dotted': False},
            {'type': 'eighth', 'beats': 0.5, 'dotted': False},
            {'type': '16th', 'beats': 0.25, 'dotted': False},
            # Dotted values
            {'type': 'half', 'beats': 3.0, 'dotted': True},
            {'type': 'quarter', 'beats': 1.5, 'dotted': True},
            {'type': 'eighth', 'beats': 0.75, 'dotted': True},
            {'type': '16th', 'beats': 0.375, 'dotted': True},
        ]
    
    # Find smallest candidate that's >= target and > current
    best = None
    best_diff = float('inf')
    
    for c in candidates:
        if c['beats'] >= target_beats - 0.01 and c['beats'] > current_beats + 0.01:
            diff = c['beats'] - target_beats
            # Prefer non-dotted and give slight preference to simpler values
            complexity = 0.1 if c['dotted'] else 0
            score = diff + complexity
            if score < best_diff:
                best_diff = score
                best = c

    return best


def fill_gaps_with_ioi(notes, bpm, max_fill_beats=2.0, debug=False):
    """
    Fill gaps between notes by re-quantizing to the inter-onset interval.

    After ML or heuristic quantization, notes may have durations shorter than
    the time to the next note, creating unwanted gaps/rests in notation.
    This detects those gaps and re-quantizes using the IOI (time to next note)
    as the target duration, naturally extending notes to fill gaps.

    Only fills gaps up to max_fill_beats. Larger gaps are treated as
    intentional phrase boundaries or rests.

    Args:
        notes: List of note dicts with 'time_seconds', 'note_divisions'
        bpm: Tempo in BPM
        max_fill_beats: Maximum gap in beats to fill (default 1.0)
        debug: Print debug info

    Returns:
        Modified notes with gaps filled
    """
    if len(notes) < 2:
        return notes

    beat_duration = 60.0 / bpm
    sorted_notes = sorted(notes, key=lambda n: n.get('time_seconds', 0))
    fills = 0

    for i in range(len(sorted_notes) - 1):
        note = sorted_notes[i]
        next_note = sorted_notes[i + 1]

        onset = note.get('time_seconds', 0)
        next_onset = next_note.get('time_seconds', 0)
        ioi = next_onset - onset
        ioi_beats = ioi / beat_duration

        note_beats = note.get('note_divisions', 1.0)
        gap_beats = ioi_beats - note_beats

        # Only fill gaps that are significant but not too large (phrase boundary)
        if gap_beats > 0.06 and gap_beats <= max_fill_beats:
            # Re-quantize using IOI as the target duration
            ioi_val = duration_to_note_value(ioi, bpm=bpm)
            ioi_error = ioi_val.get('quantization_error', 1.0)

            # Accept if the IOI quantizes cleanly to a standard note value
            if ioi_error < 0.3:
                old_type = note.get('note_value', 'quarter')
                note['note_value'] = ioi_val['type']
                note['note_divisions'] = ioi_val['divisions']
                note['dotted'] = ioi_val.get('dotted', False)
                note['has_rest_after'] = False
                note.pop('rest_duration', None)
                fills += 1

                if debug and fills <= 5:
                    print(f"  [IOI Fill] Note {i}: {old_type}({note_beats:.2f}b) -> "
                          f"{ioi_val['type']}({ioi_val['divisions']:.2f}b), "
                          f"filled {gap_beats:.2f}b gap")

        # Clear rest flags for any gap within fill range
        if 0 < gap_beats <= max_fill_beats:
            note['has_rest_after'] = False
            note.pop('rest_duration', None)

    if debug and fills > 0:
        print(f"  [IOI Fill] Filled {fills} gaps by extending to IOI duration")

    return sorted_notes


def reduce_rest_entropy(notes, bpm, debug=False):
    """
    Statistically identify genuine phrase boundaries and remove spurious rests.

    Instead of heuristic thresholds, this uses robust statistics on the IOI
    (inter-onset interval) distribution to classify each gap as either:
      - A normal continuation (note extended, no rest)
      - A genuine phrase boundary (rest kept)

    Method:
      1. Compute all IOIs in a local sliding window (8 notes).
      2. Compute the window's median IOI and MAD (median absolute deviation).
      3. A gap is a statistical outlier (= phrase boundary) when:
             IOI  >  median + k * MAD      (k = 3.0, ~99.7th percentile)
         This adapts automatically to the local rhythmic density: fast passages
         have a low median so only truly large gaps qualify; slow passages have
         a high median so normal gaps aren't flagged.
      4. Outliers that also land on a strong metric position (beat 1 or 3) are
         high-confidence phrase boundaries — these keep rests unconditionally.
      5. Everything else: extend the note to cover the gap (remove rest).

    This is a standard robust outlier detection technique (Hampel identifier)
    applied to the IOI time series.

    Args:
        notes: List of note dicts with note_divisions, has_rest_after, etc.
        bpm: Tempo in BPM
        debug: Print debug info

    Returns:
        Modified notes with statistically validated rests only.
    """
    if len(notes) < 2:
        return notes

    beat_duration = 60.0 / bpm
    sorted_notes = sorted(notes, key=lambda n: n.get('time_seconds', 0))

    # ── Step 1: Compute all IOIs ──
    iois = []
    for i in range(len(sorted_notes) - 1):
        onset_a = sorted_notes[i].get('time_seconds', 0)
        onset_b = sorted_notes[i + 1].get('time_seconds', 0)
        iois.append(onset_b - onset_a)
    iois.append(iois[-1] if iois else beat_duration)  # pad last

    # ── Step 2: Sliding-window outlier detection ──
    WINDOW = 8          # notes of context
    K_THRESHOLD = 3.0   # MAD multiplier (~99.7% for Gaussian)
    MAD_SCALE = 1.4826  # consistency constant:  MAD * 1.4826 ≈ σ for Normal

    def _local_outlier(idx):
        """Return True if IOI[idx] is a statistical outlier in its neighbourhood."""
        lo = max(0, idx - WINDOW // 2)
        hi = min(len(iois), idx + WINDOW // 2 + 1)
        window = sorted(iois[lo:hi])
        n = len(window)
        if n < 3:
            # Not enough context — fall back to global median
            window = sorted(iois)
            n = len(window)
        median = window[n // 2]
        # MAD = median of |x_i - median|
        deviations = sorted(abs(x - median) for x in window)
        mad = deviations[len(deviations) // 2]
        # Scaled MAD approximates standard deviation for normal distributions
        sigma_est = mad * MAD_SCALE
        if sigma_est < 1e-6:
            # All IOIs nearly identical — any gap > 2× median is an outlier
            sigma_est = median * 0.25
        threshold = median + K_THRESHOLD * sigma_est
        return iois[idx] > threshold, median, sigma_est

    removed = 0
    kept = 0

    for i in range(len(sorted_notes) - 1):
        note = sorted_notes[i]
        next_note = sorted_notes[i + 1]

        onset = note.get('time_seconds', 0)
        next_onset = next_note.get('time_seconds', 0)
        ioi = next_onset - onset
        note_beats = note.get('note_divisions', 1.0)
        rest_dur = note.get('rest_duration', 0)
        rest_beats = rest_dur / beat_duration if beat_duration > 0 else 0

        # Skip notes that already have no rest flag
        if not note.get('has_rest_after', False):
            continue

        # ── Statistical test ──
        is_outlier, local_med, local_sigma = _local_outlier(i)

        # ── Metric position of rest ──
        note_end_time = onset + note_beats * beat_duration
        rest_start_beat = (note_end_time / beat_duration) % 4
        on_strong_beat = (abs(rest_start_beat - round(rest_start_beat)) < 0.15 and
                          int(round(rest_start_beat)) % 2 == 0)

        # ── Decision ──
        # Keep rest only if the IOI is a statistical outlier
        if is_outlier:
            kept += 1
            if debug and kept <= 5:
                z_score = (ioi - local_med) / local_sigma if local_sigma > 1e-9 else 0
                print(f"  [Stats] KEEP rest after note {i}: IOI={ioi*1000:.0f}ms, "
                      f"median={local_med*1000:.0f}ms, z={z_score:.1f}, "
                      f"beat={rest_start_beat:.1f}, strong={on_strong_beat}")
            continue

        # ── Remove rest — extend note to cover the gap ──
        ioi_val = duration_to_note_value(ioi, bpm=bpm)
        ioi_error = ioi_val.get('quantization_error', 1.0)

        if ioi_error < 0.35:
            note['note_value'] = ioi_val['type']
            note['note_divisions'] = ioi_val['divisions']
            note['dotted'] = ioi_val.get('dotted', False)
        else:
            note['note_divisions'] = note_beats + rest_beats

        note['has_rest_after'] = False
        note.pop('rest_duration', None)
        note['quantization_method'] = note.get('quantization_method', '') + ' (rest removed)'
        removed += 1

        if debug and removed <= 5:
            z_score = (ioi - local_med) / local_sigma if local_sigma > 1e-9 else 0
            print(f"  [Stats] REMOVE rest after note {i}: IOI={ioi*1000:.0f}ms, "
                  f"median={local_med*1000:.0f}ms, z={z_score:.1f}")

    if debug:
        print(f"  [Rest Entropy] Removed {removed} spurious rests, "
              f"kept {kept} statistically significant phrase boundaries")

    return sorted_notes


def apply_coherence_smoothing(notes, bpm, debug=False):
    """Apply rhythm coherence analysis and smooth erratic sections.

    NOTE: Disabled — empirical testing shows coherence smoothing
    degrades accuracy by ~8% on MAESTRO test set.
    """
    return notes
    return notes


def compute_rhythm_coherence(notes, window_size=8, bpm=120.0, debug=False):
    """
    Compute a rhythm coherence score over sliding windows.
    
    Inspired by the brain's predictive pulse model: the auditory system maintains
    an internal rhythm expectation and detects "surprises" when notes deviate.
    This function quantifies that deviation as a coherence score.
    
    Metrics computed per window:
      1. Note value entropy: Shannon entropy of note value distribution (0=uniform, low=consistent)
      2. IOI variance: Coefficient of variation of inter-onset intervals
      3. Beat grid deviation: How far notes fall from quantized beat positions
    
    Returns a score in [0, 1] where 1 = perfectly coherent, 0 = maximally erratic.
    Also returns per-note scores for targeted smoothing.
    
    Args:
        notes: List of note dicts with 'note_value', 'note_divisions', 'time_seconds'
        window_size: Number of notes in the sliding window
        bpm: Tempo in BPM (used for beat grid deviation calculation)
        debug: Print debug info
        
    Returns:
        dict with:
          - 'global_coherence': Overall coherence score [0, 1]
          - 'per_note_coherence': List of per-note coherence values
          - 'erratic_indices': List of note indices flagged as erratic
          - 'window_coherences': List of per-window coherence scores
    """
    if len(notes) < 3:
        return {
            'global_coherence': 1.0,
            'per_note_coherence': [1.0] * len(notes),
            'erratic_indices': [],
            'window_coherences': []
        }
    
    sorted_notes = sorted(notes, key=lambda n: n.get('time_seconds', 0))
    
    # Standard note value to numeric mapping for entropy calculation
    
    def _compute_window_coherence(window_notes):
        """Compute coherence metrics for a single window."""
        if len(window_notes) < 2:
            return 1.0, {}
        
        # ── 1. Note value entropy ──
        # Count note value frequencies
        value_counts = {}
        for n in window_notes:
            nv = n.get('note_value', 'quarter')
            is_dotted = n.get('dotted', False)
            key = f"{nv}_dotted" if is_dotted else nv
            value_counts[key] = value_counts.get(key, 0) + 1
        
        # Shannon entropy: H = -sum(p * log2(p))
        total = sum(value_counts.values())
        entropy = 0.0
        for count in value_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log2(p)
        
        # Normalize entropy: max entropy = log2(n) when all values different
        max_entropy = np.log2(len(window_notes)) if len(window_notes) > 1 else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Convert to coherence (1 - normalized_entropy) so low entropy = high coherence
        entropy_coherence = 1.0 - normalized_entropy
        
        # ── 2. IOI coefficient of variation ──
        onsets = [n.get('time_seconds', 0) for n in window_notes]
        iois = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]
        
        if len(iois) > 0:
            ioi_mean = np.mean(iois)
            ioi_std = np.std(iois)
            cv = ioi_std / ioi_mean if ioi_mean > 0 else 0  # coefficient of variation
            # CV of 0 = perfectly regular; CV > 1 = very erratic
            # Map to coherence: use sigmoid-like transformation
            ioi_coherence = 1.0 / (1.0 + cv * 2)  # cv=0 -> 1.0, cv=0.5 -> 0.5, cv=1 -> 0.33
        else:
            ioi_coherence = 1.0
        
        # ── 3. Beat grid deviation ──
        # How well do note_divisions align with the IOIs?
        deviations = []
        for i in range(len(window_notes) - 1):
            n = window_notes[i]
            expected_beats = n.get('note_divisions', 1.0)
            actual_ioi = iois[i] if i < len(iois) else 0
            # Compare in relative terms
            if actual_ioi > 0 and expected_beats > 0:
                beat_dur = 60.0 / bpm
                actual_beats = actual_ioi / beat_dur
                ratio = min(actual_beats, expected_beats) / max(actual_beats, expected_beats)
                deviations.append(ratio)
        
        if deviations:
            grid_coherence = np.mean(deviations)
        else:
            grid_coherence = 1.0
        
        # ── Combined coherence ──
        # Weight: entropy matters most (note value consistency), then IOI regularity
        combined = (
            0.5 * entropy_coherence + 
            0.3 * ioi_coherence + 
            0.2 * grid_coherence
        )
        
        metrics = {
            'entropy': entropy,
            'normalized_entropy': normalized_entropy,
            'entropy_coherence': entropy_coherence,
            'ioi_cv': cv if len(iois) > 0 else 0,
            'ioi_coherence': ioi_coherence,
            'grid_coherence': grid_coherence,
            'value_distribution': value_counts
        }
        
        return combined, metrics
    
    # ── Compute per-window coherence ──
    window_coherences = []
    half_window = window_size // 2
    
    for i in range(len(sorted_notes)):
        lo = max(0, i - half_window)
        hi = min(len(sorted_notes), i + half_window + 1)
        window = sorted_notes[lo:hi]
        coherence, _ = _compute_window_coherence(window)
        window_coherences.append(coherence)
    
    # ── Detect erratic notes using z-score ──
    # Notes with coherence significantly below the local median are flagged
    per_note_coherence = window_coherences.copy()
    erratic_indices = []
    
    if len(window_coherences) >= 4:
        median_coherence = np.median(window_coherences)
        mad = np.median([abs(c - median_coherence) for c in window_coherences])
        sigma = mad * 1.4826  # MAD to std conversion
        
        # Use TWO criteria - either statistical outlier OR absolute low coherence
        # 1. Statistical: coherence more than 1.5σ below median (lowered from 2σ)
        stat_threshold = median_coherence - 1.5 * sigma if sigma > 0.01 else median_coherence * 0.8
        # 2. Absolute: coherence below 0.65 is always erratic
        abs_threshold = 0.65
        
        for i, coherence in enumerate(window_coherences):
            if coherence < stat_threshold or coherence < abs_threshold:
                erratic_indices.append(i)
    
    # Also flag notes whose value differs from both neighbors (isolated oddities)
    for i in range(1, len(sorted_notes) - 1):
        if i in erratic_indices:
            continue
        curr_val = sorted_notes[i].get('note_value', 'quarter')
        prev_val = sorted_notes[i-1].get('note_value', 'quarter')
        next_val = sorted_notes[i+1].get('note_value', 'quarter')
        if curr_val != prev_val and curr_val != next_val and prev_val == next_val:
            # This note is a lone outlier sandwiched between matching values
            erratic_indices.append(i)
    
    erratic_indices = sorted(set(erratic_indices))  # Remove duplicates
    
    global_coherence = np.mean(window_coherences) if window_coherences else 1.0
    
    if debug:
        print(f"  [Rhythm Coherence] Global: {global_coherence:.3f}, "
              f"Erratic notes: {len(erratic_indices)}/{len(notes)}")
        if erratic_indices and len(erratic_indices) <= 5:
            for idx in erratic_indices[:5]:
                print(f"    Note {idx}: coherence={window_coherences[idx]:.3f}, "
                      f"value={sorted_notes[idx].get('note_value', '?')}")
    
    return {
        'global_coherence': global_coherence,
        'per_note_coherence': per_note_coherence,
        'erratic_indices': erratic_indices,
        'window_coherences': window_coherences
    }


def smooth_erratic_rhythm(notes, coherence_info, bpm, debug=False):
    """
    Smooth erratic rhythm sections by promoting locally dominant note values.
    
    When the rhythm becomes "erratic" (coherence drops), this re-quantizes
    those notes to match the most common note value in their neighborhood,
    mimicking how the brain's predictive pulse snaps outliers to the expected beat.
    
    Only applies smoothing if:
      1. The note is flagged as erratic
      2. There's a reasonably dominant note value in the local window (>35% occurrence)
      3. The change doesn't wildly increase quantization error (within 50%)
    
    Args:
        notes: List of note dicts
        coherence_info: Dict from compute_rhythm_coherence()
        bpm: Tempo in BPM
        debug: Print debug info
        
    Returns:
        Modified notes with erratic rhythms smoothed
    """
    erratic_indices = coherence_info.get('erratic_indices', [])
    if not erratic_indices:
        return notes
    
    sorted_notes = sorted(notes, key=lambda n: n.get('time_seconds', 0))
    beat_duration = 60.0 / bpm
    smoothed = 0
    
    # Standard note values and their beat durations
    
    for idx in erratic_indices:
        if idx < 0 or idx >= len(sorted_notes):
            continue
        
        note = sorted_notes[idx]
        
        # Skip grace notes
        if note.get('is_grace_note', False):
            continue

        # Skip notes with deliberate rests — their shorter value is intentional
        if note.get('has_rest_after', False):
            continue
        
        # Get local window (±4 notes)
        lo = max(0, idx - 4)
        hi = min(len(sorted_notes), idx + 5)
        window = sorted_notes[lo:hi]
        
        # Count note values in window (excluding current note)
        value_counts = {}
        for i, n in enumerate(window):
            if lo + i == idx:  # Skip current note
                continue
            nv = n.get('note_value', 'quarter')
            dotted = n.get('dotted', False)
            key = (nv, dotted)
            value_counts[key] = value_counts.get(key, 0) + 1
        
        if not value_counts:
            continue
        
        # Find dominant value (lowered to >35% of window, was 50%)
        total = sum(value_counts.values())
        dominant_key, dominant_count = max(value_counts.items(), key=lambda x: x[1])
        
        if dominant_count / total < 0.35:
            # No clear dominant value - don't smooth
            continue
        
        dominant_value, dominant_dotted = dominant_key
        
        # Find the beat duration for dominant value
        dominant_beats = None
        for nv, beats in NOTE_VALUES_LIST:
            if nv == dominant_value:
                dominant_beats = beats * (1.5 if dominant_dotted else 1.0)
                break
        
        if dominant_beats is None:
            continue
        
        # Check if changing to dominant value is reasonable
        # (raw duration should be within 50% of dominant value - was 40%)
        raw_beats = note.get('raw_beats', note.get('note_divisions', 1.0))
        ratio = min(raw_beats, dominant_beats) / max(raw_beats, dominant_beats) if dominant_beats > 0 else 0
        
        if ratio < 0.5:
            # Too different - don't force the change
            continue
        
        # Apply smoothing
        old_value = note.get('note_value', 'quarter')
        old_dotted = note.get('dotted', False)
        
        if old_value != dominant_value or old_dotted != dominant_dotted:
            note['note_value'] = dominant_value
            note['note_divisions'] = dominant_beats
            note['dotted'] = dominant_dotted
            note['quantization_method'] = note.get('quantization_method', '') + ' (coherence-smoothed)'
            smoothed += 1
            
            if debug and smoothed <= 5:
                print(f"  [Coherence Smooth] Note {idx}: {old_value}{'.' if old_dotted else ''} -> "
                      f"{dominant_value}{'.' if dominant_dotted else ''} "
                      f"(dominant: {dominant_count}/{total} in window)")
    
    if debug and smoothed > 0:
        print(f"  [Coherence Smooth] Smoothed {smoothed} erratic notes")
    
    return sorted_notes


def enforce_triplet_groups(notes, debug=False):
    """
    Ensure triplets only appear in valid groups of exactly 3 consecutive notes
    of the same duration.

    Rules:
    - Only consecutive notes with the same note_value can form a triplet group
    - A change in note_value breaks the run
    - Only groups of exactly 3 are valid; for runs of 6, 9, etc. each sub-group
      of 3 is independent (but all must be same note_value)
    - Remainders (e.g. 2 leftover notes in a run of 5) get stripped
    - Grace notes are never triplets

    Args:
        notes: List of notes with 'is_triplet' field
        debug: Print debug info

    Returns:
        Modified notes with isolated triplets converted to non-triplets
    """
    if len(notes) < 3:
        for note in notes:
            note['is_triplet'] = False
        return notes

    # Find runs of consecutive triplets with the SAME note_value
    # A change in note_value breaks the run
    triplet_runs = []
    run_start = None
    run_note_value = None

    for i, note in enumerate(notes):
        # Grace notes can never be triplets
        if note.get('ornament') == 'grace':
            note['is_triplet'] = False
            if run_start is not None:
                triplet_runs.append((run_start, i))
                run_start = None
                run_note_value = None
            continue

        if note.get('is_triplet', False):
            current_value = note.get('note_value', 'quarter')
            if run_start is None:
                run_start = i
                run_note_value = current_value
            elif current_value != run_note_value:
                # Different note value breaks the run
                triplet_runs.append((run_start, i))
                run_start = i
                run_note_value = current_value
        else:
            if run_start is not None:
                triplet_runs.append((run_start, i))
                run_start = None
                run_note_value = None

    # Handle run that extends to end
    if run_start is not None:
        triplet_runs.append((run_start, len(notes)))

    # For each run, keep only the first (run_len // 3) * 3 notes as triplets
    # Strip the remainder
    removed_count = 0
    for start, end in triplet_runs:
        run_len = end - start
        usable = (run_len // 3) * 3
        if usable < 3:
            # Not enough for even one triplet group - strip all
            for i in range(start, end):
                notes[i]['is_triplet'] = False
            removed_count += run_len
        elif usable < run_len:
            # Keep the first 'usable' notes, strip the remainder
            for i in range(start + usable, end):
                notes[i]['is_triplet'] = False
            removed_count += (run_len - usable)

    if debug and removed_count > 0:
        print(f"  [Triplet cleanup] Removed {removed_count} isolated triplet markings")

    return notes


def strip_triplets_from_grace_notes(items):
    """
    Strip triplet markings from grace notes - they should never be triplets.
    Works for both notes and chords.

    Args:
        items: List of note or chord dicts

    Returns:
        The same list with triplet info removed from grace notes
    """
    for item in items:
        if item.get('ornament') == 'grace' and (item.get('triplet') or item.get('is_triplet')):
            item['triplet'] = False
            item['is_triplet'] = False
            item.pop('triplet_position', None)
            item.pop('triplet_type', None)
            item.pop('actual_notes', None)
            item.pop('normal_notes', None)
    return items


def quantize_rhythm_from_ioi(notes, bpm, debug=False):
    """
    Quantize note rhythms using IOI-primary approach.
    
    KEY PRINCIPLE: In real notation, notes usually extend to the next note onset.
    Rests should be minimized - only insert them for truly intentional gaps.
    
    Args:
        notes: List of note dicts with 'time_seconds' and 'duration_seconds'
        bpm: Detected tempo in BPM
        debug: Print debug info
    
    Returns:
        Modified notes list with improved note_value assignments
    """
    if len(notes) < 2:
        # Not enough notes to use IOI, fall back to duration-based
        for note in notes:
            dur = note.get('duration_seconds', 0.5)
            note_val = duration_to_note_value(dur, bpm=bpm, debug=debug)
            note['note_value'] = note_val['type']
            note['note_divisions'] = note_val['divisions']
            note['dotted'] = note_val['dotted']
            note['quantization_method'] = 'duration'
        return notes
    
    beat_duration = 60.0 / bpm

    # Sort notes by time
    sorted_notes = sorted(notes, key=lambda n: n.get('time_seconds', 0))

    # ── Compute per-hand IOIs ──
    # Cross-hand IOI captures inter-hand spacing, not note duration.
    # Per-hand IOI is a much better proxy for the musical note duration.
    hand_indices = {}  # hand -> list of indices in sorted_notes
    for i, note in enumerate(sorted_notes):
        hand = note.get('hand', 'treble')
        if hand not in hand_indices:
            hand_indices[hand] = []
        hand_indices[hand].append(i)

    per_hand_ioi = [None] * len(sorted_notes)
    for hand, indices in hand_indices.items():
        for j in range(len(indices) - 1):
            curr_idx = indices[j]
            next_idx = indices[j + 1]
            ioi = (sorted_notes[next_idx].get('time_seconds', 0)
                   - sorted_notes[curr_idx].get('time_seconds', 0))
            per_hand_ioi[curr_idx] = ioi

    # Global IOI as fallback (for last-in-hand notes)
    times = [n.get('time_seconds', 0) for n in sorted_notes]
    global_iois = np.diff(times)

    # ── Tempo-relative thresholds ──
    min_ioi_sec = max(0.05, beat_duration * 0.2)
    min_ioi_beats = min_ioi_sec / beat_duration
    min_acoustic_beats = max(0.1, 0.2 * (120.0 / bpm))
    MAX_NOTE_BEATS = 6.0

    if debug:
        print(f"\n[IOI Quantization] {len(notes)} notes, BPM={bpm}")
        print(f"  Beat duration: {beat_duration*1000:.1f}ms")
        print(f"  Hands: {', '.join(f'{h}={len(idx)}' for h, idx in hand_indices.items())}")
        print(f"  Using per-hand IOI as primary duration")

    # Process each note
    for i, note in enumerate(sorted_notes):
        acoustic_dur = note.get('duration_seconds', 0.5)
        acoustic_beats = acoustic_dur / beat_duration

        # Prefer per-hand IOI; fall back to global IOI; fall back to acoustic
        ioi_sec = per_hand_ioi[i]
        if ioi_sec is None:
            # Last note in this hand — use global IOI if available
            if i < len(global_iois):
                ioi_sec = global_iois[i]
            else:
                ioi_sec = acoustic_dur  # last note overall
        ioi_beats = ioi_sec / beat_duration

        if ioi_beats < min_ioi_beats or ioi_sec < 0.03:
            # Near-simultaneous / overlapping — use max of acoustic and IOI
            use_dur = max(acoustic_dur, ioi_sec)
            note_val = duration_to_note_value(use_dur, bpm=bpm, debug=False)
            method = 'acoustic (overlap)'
            note['has_rest_after'] = False
        elif ioi_beats > MAX_NOTE_BEATS:
            # Very long gap — cap at max note duration, rest after
            capped = beat_duration * MAX_NOTE_BEATS
            note_val = duration_to_note_value(capped, bpm=bpm, debug=False)
            method = 'ioi-capped (long gap)'
            note['has_rest_after'] = True
            note['rest_duration'] = ioi_sec - capped
        elif acoustic_beats < min_acoustic_beats:
            # Grace note / very short — extend to IOI but cap at half beat
            note_val = duration_to_note_value(min(ioi_sec, beat_duration * 0.5), bpm=bpm, debug=False)
            method = 'ioi (short note)'
            note['has_rest_after'] = False
        else:
            # DEFAULT: Use per-hand IOI as note duration.
            # This matches real notation: notes extend to next onset.
            note_val = duration_to_note_value(ioi_sec, bpm=bpm, debug=False)
            method = 'ioi'
            note['has_rest_after'] = False

        if debug:
            ph = "ph" if per_hand_ioi[i] is not None else "gl"
            print(f"  Note {i}: acoustic={acoustic_dur*1000:.0f}ms, ioi={ioi_sec*1000:.0f}ms({ph}) "
                  f"-> {note_val['type']} ({method})")
        
        note['note_value'] = note_val['type']
        note['note_divisions'] = note_val['divisions']
        note['dotted'] = note_val.get('dotted', False)
        note['quantization_method'] = method
        note['raw_beats'] = note_val.get('raw_beats', 0)
        note['quantization_error'] = note_val.get('quantization_error', 0)
        # Store acoustic-based value separately (never overwritten by post-processing)
        # This is useful for accuracy evaluation against MIDI ground truth
        acoustic_val = duration_to_note_value(acoustic_dur, bpm=bpm, debug=False)
        note['acoustic_note_value'] = acoustic_val['type']
        note['acoustic_dotted'] = acoustic_val.get('dotted', False)

    sorted_notes = post_process_rhythm_unified(sorted_notes, bpm, debug=debug)

    # Apply rhythm coherence smoothing to reduce erratic sections
    sorted_notes = apply_coherence_smoothing(sorted_notes, bpm, debug=debug)

    sorted_notes = enforce_bar_sum(sorted_notes, bpm, debug=debug)

    return sorted_notes


def quantize_rhythm_sequence(notes, chords, bpm, debug=False, use_ml=True):
    """
    Quantize rhythms for both notes and chords.
    
    Handles notes and chords in separate hands independently to avoid
    cross-hand IOI confusion.
    
    Args:
        notes: List of note dicts
        chords: List of chord dicts  
        bpm: Detected tempo
        debug: Print debug info
        use_ml: If True, use ML-based quantization; otherwise use heuristic
    
    Returns:
        Tuple of (quantized_notes, quantized_chords)
    """
    # Choose quantization function
    quantize_fn = quantize_rhythm_ml if use_ml else quantize_rhythm_from_ioi
    fn_name = "ML" if use_ml else "IOI"
    
    # Separate by hand if available
    bass_notes = [n for n in notes if n.get('hand') == 'bass']
    treble_notes = [n for n in notes if n.get('hand') == 'treble']
    other_notes = [n for n in notes if n.get('hand') not in ('bass', 'treble')]
    
    bass_chords = [c for c in chords if c.get('hand') == 'bass']
    treble_chords = [c for c in chords if c.get('hand') == 'treble']
    other_chords = [c for c in chords if c.get('hand') not in ('bass', 'treble')]
    
    # Quantize each group separately
    if debug:
        print(f"\n[Rhythm Quantization - {fn_name}] Bass: {len(bass_notes)} notes, {len(bass_chords)} chords")
    if bass_notes:
        bass_notes = quantize_fn(bass_notes, bpm, debug=debug)
    if bass_chords:
        bass_chords = quantize_fn(bass_chords, bpm, debug=debug)
    
    if debug:
        print(f"\n[Rhythm Quantization - {fn_name}] Treble: {len(treble_notes)} notes, {len(treble_chords)} chords")
    if treble_notes:
        treble_notes = quantize_fn(treble_notes, bpm, debug=debug)
    if treble_chords:
        treble_chords = quantize_fn(treble_chords, bpm, debug=debug)
    
    if debug and other_notes:
        print(f"\n[Rhythm Quantization - {fn_name}] Other: {len(other_notes)} notes, {len(other_chords)} chords")
    if other_notes:
        other_notes = quantize_fn(other_notes, bpm, debug=debug)
    if other_chords:
        other_chords = quantize_fn(other_chords, bpm, debug=debug)
    
    # Merge back
    all_notes = bass_notes + treble_notes + other_notes
    all_chords = bass_chords + treble_chords + other_chords
    
    # Sort by time
    all_notes.sort(key=lambda n: n.get('time_seconds', 0))
    all_chords.sort(key=lambda c: c.get('time_seconds', 0))
    
    return all_notes, all_chords


def detect_tempo_from_onsets(onset_times, velocities=None, min_bpm=50, max_bpm=200):
    """
    Detect tempo from onset times using velocity-weighted autocorrelation.

    For polyphonic music (e.g. piano with both hands), raw onset spacing picks
    up inter-hand note density rather than the musical beat.  This function:
    1. Builds a velocity-weighted pulse train (louder notes = stronger pulse)
    2. Autocorrelates via FFT to find periodic peaks
    3. Generates sub-harmonic candidates (BPM/2) for every fast peak
    4. Scores candidates by IOI alignment to integer beat multiples
    5. Strongly prefers musically natural tempos (60-160 BPM range)

    Args:
        onset_times: List/array of onset times in seconds
        velocities:  Optional list/array of velocities (0-127) per onset.
                     Louder onsets weight more heavily in autocorrelation.
        min_bpm: Minimum BPM to consider (default 50)
        max_bpm: Maximum BPM to consider (default 200)

    Returns:
        dict with 'bpm' (detected tempo), 'confidence' (0-1), 'beat_interval' (seconds)
    """
    if len(onset_times) < 3:
        return {'bpm': 120, 'confidence': 0.0, 'beat_interval': 0.5}

    onset_times = np.sort(np.array(onset_times, dtype=float))

    # Normalize velocities (default = uniform)
    if velocities is not None:
        vel = np.array(velocities, dtype=float)
        vel = vel / (vel.max() + 1e-12)
        # Floor at 0.3 so quiet notes still contribute
        vel = np.clip(vel, 0.3, 1.0)
    else:
        vel = np.ones(len(onset_times))

    # ── Step 1: IOIs ──
    iois = np.diff(onset_times)
    min_interval = 60.0 / max_bpm * 0.25
    max_interval = 60.0 / min_bpm * 4
    valid_iois = iois[(iois >= min_interval) & (iois <= max_interval)]

    if len(valid_iois) < 2:
        return {'bpm': 120, 'confidence': 0.0, 'beat_interval': 0.5}

    # ── Step 2: Velocity-weighted autocorrelation ──
    duration = onset_times[-1] - onset_times[0]
    if duration <= 0:
        return {'bpm': 120, 'confidence': 0.0, 'beat_interval': 0.5}

    bin_size = 0.005  # 5ms resolution
    n_bins = int(duration / bin_size) + 1
    pulse = np.zeros(n_bins)
    for j, t in enumerate(onset_times):
        idx = int((t - onset_times[0]) / bin_size)
        if 0 <= idx < n_bins:
            pulse[idx] = vel[j]  # Velocity-weighted (not binary)

    n_fft = 1
    while n_fft < 2 * n_bins:
        n_fft *= 2
    F = np.fft.rfft(pulse, n=n_fft)
    ac = np.fft.irfft(F * np.conj(F))[:n_bins]
    ac = ac / (ac[0] + 1e-12)

    # ── Step 3: Find peaks in autocorrelation ──
    min_lag = int(60.0 / max_bpm / bin_size)
    max_lag = int(60.0 / min_bpm / bin_size)
    max_lag = min(max_lag, n_bins - 1)

    if min_lag >= max_lag or min_lag >= len(ac):
        median_ioi = float(np.median(valid_iois))
        bpm = int(np.clip(60.0 / median_ioi, min_bpm, max_bpm))
        return {'bpm': bpm, 'confidence': 0.2, 'beat_interval': 60.0 / bpm}

    ac_segment = ac[min_lag:max_lag + 1]
    lags = np.arange(min_lag, max_lag + 1)

    peaks = []
    for i in range(1, len(ac_segment) - 1):
        if ac_segment[i] > ac_segment[i - 1] and ac_segment[i] > ac_segment[i + 1]:
            if ac_segment[i] > 0.05:
                lag = lags[i]
                period = lag * bin_size
                bpm_candidate = 60.0 / period
                peaks.append((bpm_candidate, period, ac_segment[i]))

    if not peaks:
        median_ioi = float(np.median(valid_iois))
        bpm = int(np.clip(60.0 / median_ioi, min_bpm, max_bpm))
        return {'bpm': bpm, 'confidence': 0.2, 'beat_interval': 60.0 / bpm}

    # ── Step 4: Generate sub-harmonic candidates ──
    # For every fast peak (>100 BPM), also evaluate BPM/2 as a candidate.
    # In polyphonic piano, the autocorrelation often fires at the note-density
    # level (fast) rather than the beat level (slower).
    all_candidates = list(peaks)
    for bpm_c, period, ac_str in peaks:
        if bpm_c > 100:
            half_bpm = bpm_c / 2
            half_period = period * 2
            if half_bpm >= min_bpm:
                # Check AC strength at the sub-harmonic lag
                half_lag = int(half_period / bin_size)
                if half_lag < len(ac):
                    ac_at_half = ac[half_lag]
                else:
                    ac_at_half = 0.0
                all_candidates.append((half_bpm, half_period, max(ac_at_half, ac_str * 0.5)))
        # Also try 2/3 of BPM for triplet-related tempos
        if bpm_c > 130:
            two_thirds_bpm = bpm_c * 2 / 3
            two_thirds_period = period * 1.5
            if two_thirds_bpm >= min_bpm:
                t3_lag = int(two_thirds_period / bin_size)
                if t3_lag < len(ac):
                    ac_at_t3 = ac[t3_lag]
                else:
                    ac_at_t3 = 0.0
                all_candidates.append((two_thirds_bpm, two_thirds_period, max(ac_at_t3, ac_str * 0.4)))

    # ── Step 5: Score all candidates by IOI alignment ──
    def score_candidate(period):
        """Score how well IOIs align to multiples of this beat period."""
        total_score = 0.0
        for ioi in valid_iois:
            ratio = ioi / period
            nearest_half = round(ratio * 2) / 2
            if nearest_half < 0.25:
                continue
            dist = abs(ratio - nearest_half)
            alignment = math.exp(-(dist ** 2) / (2 * 0.08 ** 2))
            if abs(nearest_half - round(nearest_half)) < 0.01:
                alignment *= 1.3
            total_score += alignment
        return total_score / len(valid_iois)

    scored = []
    for bpm_c, period, ac_strength in all_candidates:
        alignment_score = score_candidate(period)
        combined = 0.4 * ac_strength + 0.6 * alignment_score
        scored.append((bpm_c, period, combined, ac_strength, alignment_score))

    scored.sort(key=lambda x: -x[2])

    # ── Step 6: Prefer musically sensible tempo ──
    # Widen tolerance to 70%: if anything in 60-160 BPM scores within 30%
    # of the best, strongly prefer it (fast tempos are usually subdivisions)
    best = scored[0]
    for candidate in scored[:8]:
        bpm_c, period, combined, _, _ = candidate
        if 60 <= bpm_c <= 160 and combined >= best[2] * 0.70:
            best = candidate
            break

    best_bpm = best[0]
    best_confidence = min(1.0, best[2] * 2)

    # ── Step 7: Octave disambiguation ──
    # Check if 2x or 0.5x the detected tempo gives better IOI alignment.
    # Piano autocorrelation often locks onto note density (2x) rather than beat.
    best_period = best[1]
    best_alignment = score_candidate(best_period)
    for mult in [0.5, 2.0]:
        alt_bpm = best_bpm * mult
        if min_bpm <= alt_bpm <= max_bpm:
            alt_period = 60.0 / alt_bpm
            alt_alignment = score_candidate(alt_period)
            # Prefer the alternative only if clearly better alignment
            # AND it falls in a more natural range
            if alt_alignment > best_alignment * 1.05:
                best_bpm = alt_bpm
                best_period = alt_period
                best_alignment = alt_alignment

    # ── Step 8: Snap to nearest common BPM ──
    common_bpms = [50, 54, 58, 60, 63, 66, 69, 72, 76, 80, 84, 88, 92, 96,
                   100, 104, 108, 112, 116, 120, 126, 132, 138, 144, 152,
                   160, 168, 176, 184, 192, 200]
    closest_common = min(common_bpms, key=lambda x: abs(x - best_bpm))
    if abs(closest_common - best_bpm) / best_bpm < 0.05:
        best_bpm = closest_common
    else:
        best_bpm = round(best_bpm / 2) * 2

    bpm = int(np.clip(best_bpm, min_bpm, max_bpm))
    beat_interval = 60.0 / bpm

    print(f"[Tempo] Detected {bpm} BPM (beat = {beat_interval:.3f}s, confidence = {best_confidence:.2f})")
    print(f"[Tempo] IOI stats: {len(valid_iois)} intervals, median={np.median(valid_iois):.3f}s, mean={np.mean(valid_iois):.3f}s")
    if len(scored) >= 3:
        print(f"[Tempo] Top candidates: {scored[0][0]:.0f}BPM(score={scored[0][2]:.3f}) "
              f"{scored[1][0]:.0f}BPM(score={scored[1][2]:.3f}) "
              f"{scored[2][0]:.0f}BPM(score={scored[2][2]:.3f})")

    return {
        'bpm': bpm,
        'confidence': round(best_confidence, 2),
        'beat_interval': round(beat_interval, 4)
    }


def refine_tempo_by_quantization(notes, initial_bpm, min_bpm=40, max_bpm=240):
    """
    Refine tempo by testing candidates and picking the one with lowest quantization error.
    
    The initial tempo detection uses histogram analysis which can pick up subdivisions
    or multiples of the true beat. This function tests related tempos (0.33x to 2.0x) 
    and picks the one that minimizes quantization error.
    
    Args:
        notes: List of note dicts with 'time_seconds' and 'duration_seconds'
        initial_bpm: Initially detected BPM
        min_bpm: Minimum valid BPM (default 40 for slow pieces like Bach fugues)
        max_bpm: Maximum valid BPM
    
    Returns:
        dict with refined 'bpm', 'confidence', 'beat_interval', 'refinement_factor'
    """
    if len(notes) < 3:
        return {
            'bpm': initial_bpm,
            'confidence': 0.5,
            'beat_interval': 60.0 / initial_bpm,
            'refinement_factor': 1.0
        }

    # Tempo should be inferred from successive onset clusters, not every raw
    # polyphonic event. Sorting and collapsing near-simultaneous events avoids
    # note stacks and note+chord duplicates from masquerading as ultra-fast IOIs.
    onset_cluster_tolerance = 0.03
    event_times = sorted(
        float(note.get('time_seconds', 0.0) or 0.0)
        for note in notes
        if note.get('time_seconds') is not None
    )

    clustered_times = []
    for time_seconds in event_times:
        if not clustered_times or time_seconds - clustered_times[-1] > onset_cluster_tolerance:
            clustered_times.append(time_seconds)

    if len(clustered_times) < 3:
        return {
            'bpm': initial_bpm,
            'confidence': 0.5,
            'beat_interval': 60.0 / initial_bpm,
            'refinement_factor': 1.0
        }
    
    # Test these multipliers of the initial tempo
    # These correspond to common tempo confusions:
    # 0.33 = initial detection found 16th notes instead of quarter notes
    # 0.5 = initial detection found eighth notes instead of quarter notes
    # 0.67 = confusing dotted quarters with quarters
    # 0.75 = confusing compound meter 
    # 1.33 = inverse of 0.75
    # 1.5 = confusing half notes with dotted half notes
    # 2.0 = double time
    # More granular multipliers for better coverage
    multipliers = [0.25, 0.33, 0.4, 0.5, 0.67, 0.75, 1.0, 1.33, 1.5, 2.0, 3.0]
    
    # Calculate IOIs from clustered onset times and ignore gaps outside the
    # plausible beat/subdivision search window.
    iois = np.diff(np.array(clustered_times, dtype=float))
    max_interval = 60.0 / min_bpm * 4.0
    iois = iois[(iois > 0) & (iois <= max_interval)]
    
    if len(iois) == 0:
        return {
            'bpm': initial_bpm,
            'confidence': 0.5,
            'beat_interval': 60.0 / initial_bpm,
            'refinement_factor': 1.0
        }
    
    # Two-pass approach: first find best from initial, then refine from that
    def calc_error_at_bpm(test_bpm):
        """Calculate mean quantization error at a given tempo, penalizing
        implausibly short note-value distributions (sign of tempo doubling)."""
        errors = []
        n_32nd = 0
        n_16th = 0
        for ioi in iois:
            val = duration_to_note_value(ioi, bpm=test_bpm)
            errors.append(val.get('quantization_error', 0))
            if val['type'] == '32nd':
                n_32nd += 1
            elif val['type'] == '16th':
                n_16th += 1
        if not errors:
            return 1.0
        base_error = np.mean(errors)
        # Penalize tempos that force too many notes into tiny values.
        # Real music rarely has >20% 32nds or >50% 16th-or-shorter.
        # Use a steep, progressive penalty so "mostly 32nds" is unmistakable.
        n = len(errors)
        frac_32 = n_32nd / n
        frac_short = (n_32nd + n_16th) / n
        penalty = 0.0
        if frac_32 > 0.15:
            penalty += (frac_32 - 0.15) * 1.5   # aggressive: +15% per 10% excess 32nds
        if frac_32 > 0.50:
            penalty += (frac_32 - 0.50) * 2.0   # extra steep above 50% 32nds
        if frac_short > 0.50:
            penalty += (frac_short - 0.50) * 0.8  # +8% per 10% excess short notes
        return base_error + penalty
    
    best_bpm = initial_bpm
    best_error = calc_error_at_bpm(initial_bpm)
    best_mult = 1.0
    
    # Track all tested tempos for debugging
    tested = [(initial_bpm, 1.0, best_error)]
    
    for mult in multipliers:
        test_bpm = initial_bpm * mult
        
        # Skip if outside valid range
        if test_bpm < min_bpm or test_bpm > max_bpm:
            continue
        
        mean_error = calc_error_at_bpm(test_bpm)
        tested.append((test_bpm, mult, mean_error))
        
        # Pick the tempo with lowest error
        # Only prefer multiplier closer to 1.0 if errors are VERY similar (within 0.5%)
        if mean_error < best_error - 0.005:
            best_error = mean_error
            best_bpm = test_bpm
            best_mult = mult
    
    # Sort tested tempos by error for debugging
    tested_sorted = sorted(tested, key=lambda x: x[2])
    print(f"[Tempo Search] Top candidates: ", end="")
    for bpm, mult, err in tested_sorted[:3]:
        print(f"{bpm:.0f}BPM({err*100:.1f}%) ", end="")
    print()
    
    # Second pass: test refinements of the best tempo (1.5x, 0.67x)
    # This catches cases where we need a compound adjustment
    second_pass_mults = [0.67, 0.75, 1.33, 1.5]
    for mult in second_pass_mults:
        test_bpm = best_bpm * mult
        if test_bpm < min_bpm or test_bpm > max_bpm:
            continue
        
        mean_error = calc_error_at_bpm(test_bpm)
        tested.append((test_bpm, best_mult * mult, mean_error))
        
        if mean_error < best_error - 0.005:
            best_error = mean_error
            best_bpm = test_bpm
            best_mult = best_mult * mult
    
    # Round to nice BPM (include slow tempos for baroque music)
    nice_bpms = [40, 44, 46, 48, 50, 52, 54, 56, 58, 60, 63, 66, 69, 72, 76, 80, 84, 88, 92, 96, 
                 100, 104, 108, 112, 116, 120, 126, 132, 138, 144, 150, 
                 156, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240]
    closest_nice = min(nice_bpms, key=lambda x: abs(x - best_bpm))
    if abs(closest_nice - best_bpm) / best_bpm < 0.03:
        best_bpm = closest_nice
    else:
        best_bpm = round(best_bpm)
    
    # Confidence is higher if error is low
    confidence = max(0.3, min(1.0, 1.0 - best_error * 2))
    
    if best_mult != 1.0:
        print(f"[Tempo Refinement] Adjusted {initial_bpm:.0f} → {best_bpm:.0f} BPM "
              f"(×{best_mult:.2f}, error: {best_error*100:.1f}%)")
    
    return {
        'bpm': int(best_bpm),
        'confidence': round(confidence, 2),
        'beat_interval': round(60.0 / best_bpm, 4),
        'refinement_factor': best_mult
    }


def refine_tempo_onset_grid(notes, chords, initial_bpm, beat_times=None,
                            sweep_pct=0.10, step=0.5, debug=False):
    """
    Fine-grained tempo refinement using onset-to-grid alignment.

    For each candidate BPM, lays down an ideal grid starting at t=0 with
    optimal phase and measures how well note onsets align to it. Returns the
    BPM with lowest mean circular alignment error, plus synthetic regular
    beat_times generated from that BPM.

    This runs even when neural beat detection succeeds — librosa's beat_track
    gives non-uniform beat positions that follow performer rubato, but MusicXML
    quantization needs a single fixed BPM. A ±10% sweep finds the BPM where
    fixed-grid quantization distorts note durations least.

    Args:
        notes: list of note dicts with 'time_seconds'
        chords: list of chord dicts with 'time_seconds'
        initial_bpm: BPM from beat detection
        beat_times: optional array of detected beat times (for fallback)
        sweep_pct: fraction of initial_bpm to sweep (default 10%)
        step: BPM step size (default 0.5)
        debug: print diagnostics

    Returns:
        dict with 'bpm', 'beat_interval', 'beat_times', 'confidence',
               'grid_error', 'phase_offset'
    """
    # Gather all onset times
    all_events = notes + chords
    onsets = np.array(sorted(set(
        e.get('time_seconds', 0) for e in all_events if e.get('time_seconds', 0) > 0
    )))

    if len(onsets) < 4:
        beat_int = 60.0 / initial_bpm
        duration = onsets[-1] if len(onsets) > 0 else 10.0
        return {
            'bpm': initial_bpm,
            'beat_interval': beat_int,
            'beat_times': np.arange(0, duration + beat_int, beat_int),
            'confidence': 0.5,
            'grid_error': 1.0,
            'phase_offset': 0.0,
        }

    duration = onsets[-1] + 1.0  # extend slightly past last note

    def grid_alignment_error(bpm):
        """Compute mean alignment error and optimal phase for a given BPM.

        For each onset, compute its fractional position within a beat period.
        Use circular statistics to find the optimal phase, then compute mean
        absolute distance to nearest grid point.
        """
        beat_period = 60.0 / bpm
        # Fractional beat positions (0 to 1)
        phases = (onsets / beat_period) % 1.0

        # Circular mean to find optimal phase offset
        # Convert to angles, compute mean angle
        angles = phases * 2 * np.pi
        mean_sin = np.mean(np.sin(angles))
        mean_cos = np.mean(np.cos(angles))
        mean_angle = np.arctan2(mean_sin, mean_cos)
        phase_offset = (mean_angle / (2 * np.pi)) % 1.0

        shifted = (phases - phase_offset) % 1.0

        # Let slower tempi explain onsets via simple subdivisions instead of
        # forcing every fast subdivision to become its own beat.
        subdivision_specs = (
            (np.array([0.0]), 0.0),
            (np.array([0.0, 0.5]), beat_period * 0.035),
            (np.array([0.0, 1 / 3, 2 / 3]), beat_period * 0.05),
            (np.array([0.0, 0.25, 0.5, 0.75]), beat_period * 0.08),
        )

        onset_costs = []
        for onset_phase in shifted:
            best_cost = beat_period * 0.5
            for subdivision_points, penalty in subdivision_specs:
                distances = np.abs(
                    ((onset_phase - subdivision_points + 0.5) % 1.0) - 0.5
                )
                best_cost = min(
                    best_cost,
                    float(np.min(distances)) * beat_period + penalty,
                )
            onset_costs.append(best_cost)

        mean_err = float(np.mean(onset_costs))

        return mean_err, phase_offset * beat_period

    def build_candidate_beat_times(candidate_bpm, candidate_phase):
        beat_int = 60.0 / candidate_bpm
        start = candidate_phase % beat_int
        return np.arange(start, duration + beat_int, beat_int)

    def probe_notation_candidate(candidate_bpm, candidate_phase):
        candidate_beat_times = build_candidate_beat_times(candidate_bpm, candidate_phase)
        candidate_notes = deepcopy(notes)
        candidate_chords = deepcopy(chords)
        candidate_events = candidate_notes + candidate_chords
        candidate_subdivision_info = detect_dominant_subdivisions(
            candidate_events,
            candidate_bpm,
            debug=False,
        )

        def quantize_items(items):
            if not items:
                return []
            items = tag_runs_pre_quantization(items, candidate_bpm, debug=False)
            items = quantize_to_beat_grid(
                items,
                candidate_beat_times,
                candidate_bpm,
                candidate_subdivision_info,
                debug=False,
            )
            items = cross_validate_with_acoustic_duration(items, candidate_bpm, debug=False)
            items = post_process_rhythm_unified(items, candidate_bpm, debug=False)
            items = apply_coherence_smoothing(items, candidate_bpm, debug=False)
            return items

        hand_labels = ('bass', 'treble')
        quantized_notes = []
        quantized_chords = []
        for hand_label in hand_labels:
            quantized_notes.extend(
                quantize_items([n for n in candidate_notes if n.get('hand') == hand_label])
            )
            quantized_chords.extend(
                quantize_items([c for c in candidate_chords if c.get('hand') == hand_label])
            )

        quantized_notes.extend(
            quantize_items([
                n for n in candidate_notes
                if n.get('hand') not in hand_labels
            ])
        )
        quantized_chords.extend(
            quantize_items([
                c for c in candidate_chords
                if c.get('hand') not in hand_labels
            ])
        )

        quantized_notes = sorted(quantized_notes, key=lambda item: item.get('time_seconds', 0))
        quantized_chords = sorted(quantized_chords, key=lambda item: item.get('time_seconds', 0))
        proximity = compute_notation_proximity_score(
            quantized_notes,
            quantized_chords,
            candidate_bpm,
            debug=False,
        )
        overall = proximity.get('overall') or {}
        return {
            'mean_drift_ms': float(overall.get('mean_drift_ms', float('inf'))),
            'pct_within_16th': float(overall.get('pct_within_16th', 0.0)),
            'pct_within_8th': float(overall.get('pct_within_8th', 0.0)),
        }

    # Sweep BPM candidates
    lo = max(30, initial_bpm * (1 - sweep_pct))
    hi = min(300, initial_bpm * (1 + sweep_pct))
    candidates = np.arange(lo, hi + step * 0.5, step)

    best_bpm = initial_bpm
    best_error, best_phase = grid_alignment_error(initial_bpm)
    tested_candidates = [(float(initial_bpm), float(best_error), float(best_phase))]

    for c_bpm in candidates:
        err, phase = grid_alignment_error(c_bpm)
        tested_candidates.append((float(c_bpm), float(err), float(phase)))
        if err < best_error - 0.001:  # require meaningful improvement
            best_error = err
            best_bpm = c_bpm
            best_phase = phase

    # Also test slower ratio adjustments against both the original tempo and the
    # current best local candidate. Without this second base, a local sweep can
    # climb to (for example) 244 BPM and never compare against the meaningful
    # half-tempo alternative at 122 BPM.
    ratio_bases = {float(initial_bpm), float(best_bpm)}
    ratio_multipliers = [0.5, 2 / 3, 0.75]
    tested_ratio_bpms = {
        round(float(cand_bpm) * 1000)
        for cand_bpm, _, _ in tested_candidates
    }
    for base_bpm in ratio_bases:
        for mult in ratio_multipliers:
            c_bpm = base_bpm * mult
            c_bpm_key = round(float(c_bpm) * 1000)
            if c_bpm_key in tested_ratio_bpms:
                continue
            tested_ratio_bpms.add(c_bpm_key)
            if not (30 <= c_bpm <= 300):
                continue

            err, phase = grid_alignment_error(c_bpm)
            tested_candidates.append((float(c_bpm), float(err), float(phase)))
            if err < best_error - 0.001:
                best_error = err
                best_bpm = c_bpm
                best_phase = phase

    # If the search still lands near the ceiling, prefer a slower natural-range
    # candidate when its grid fit is effectively tied.
    if best_bpm >= 200:
        tie_tolerance = max(0.012, (60.0 / best_bpm) * 0.05)
        natural_candidates = [
            (cand_bpm, cand_error, cand_phase)
            for cand_bpm, cand_error, cand_phase in tested_candidates
            if 60 <= cand_bpm <= 160
            and cand_bpm <= best_bpm / 1.25
            and cand_error <= best_error + tie_tolerance
        ]
        if natural_candidates:
            natural_bpm, natural_error, natural_phase = min(
                natural_candidates,
                key=lambda item: (item[1], -item[0]),
            )
            best_bpm = natural_bpm
            best_error = natural_error
            best_phase = natural_phase

    # Raw onset-grid fit can still prefer doubled tempo even when the resulting
    # quantized notation is much worse. Before finalizing a fast tempo, probe a
    # few slower natural-range candidates through the actual quantization path
    # and switch only when the downstream score quality improves materially.
    if best_bpm >= 140 and len(onsets) >= 6:
        probe_candidates = [(float(best_bpm), float(best_error), float(best_phase))]
        seen_probe_bpms = {round(float(best_bpm) * 1000)}
        slower_candidates = sorted(
            [
                (cand_bpm, cand_error, cand_phase)
                for cand_bpm, cand_error, cand_phase in tested_candidates
                if 60 <= cand_bpm < best_bpm
                and cand_bpm <= best_bpm / 1.25
            ],
            key=lambda item: (item[1], -item[0]),
        )
        for cand_bpm, cand_error, cand_phase in slower_candidates:
            cand_key = round(float(cand_bpm) * 1000)
            if cand_key in seen_probe_bpms:
                continue
            probe_candidates.append((float(cand_bpm), float(cand_error), float(cand_phase)))
            seen_probe_bpms.add(cand_key)
            if len(probe_candidates) >= 4:
                break

        probe_results = []
        for cand_bpm, cand_error, cand_phase in probe_candidates:
            notation_score = probe_notation_candidate(cand_bpm, cand_phase)
            probe_results.append({
                'bpm': cand_bpm,
                'grid_error': cand_error,
                'phase': cand_phase,
                **notation_score,
            })

        current_probe = probe_results[0]
        best_probe = min(
            probe_results,
            key=lambda item: (
                item['mean_drift_ms'],
                -item['pct_within_16th'],
                item['grid_error'],
            ),
        )
        if (
            best_probe['bpm'] != current_probe['bpm']
            and (
                best_probe['mean_drift_ms'] <= current_probe['mean_drift_ms'] - 40.0
                or best_probe['pct_within_16th'] >= current_probe['pct_within_16th'] + 0.25
            )
        ):
            best_bpm = best_probe['bpm']
            best_error = best_probe['grid_error']
            best_phase = best_probe['phase']
            if debug:
                print(
                    f"[Tempo Notation Probe] Prefer {best_bpm:.1f} BPM over "
                    f"{current_probe['bpm']:.1f} BPM "
                    f"(mean drift {current_probe['mean_drift_ms']:.1f}ms "
                    f"→ {best_probe['mean_drift_ms']:.1f}ms)"
                )

    # Round to nearest 0.5 BPM
    best_bpm = round(best_bpm * 2) / 2

    # Snap to "nice" BPM if very close (within 1%)
    nice_bpms = [40, 44, 46, 48, 50, 52, 54, 56, 58, 60, 63, 66, 69, 72, 76,
                 80, 84, 88, 92, 96, 100, 104, 108, 112, 116, 120, 126, 132,
                 138, 144, 150, 156, 160, 168, 176, 184, 192, 200]
    closest_nice = min(nice_bpms, key=lambda x: abs(x - best_bpm))
    if abs(closest_nice - best_bpm) / best_bpm < 0.01:
        best_bpm = closest_nice

    # Generate synthetic regular beat times from the refined BPM + phase
    beat_int = 60.0 / best_bpm
    synthetic_beats = build_candidate_beat_times(best_bpm, best_phase)

    # Confidence based on alignment quality
    # Perfect alignment: error ~0; worst case: error = beat_int/4
    relative_error = best_error / (beat_int / 4)  # normalize to [0, 1]
    confidence = max(0.3, min(1.0, 1.0 - relative_error))

    if debug or abs(best_bpm - initial_bpm) > 0.5:
        print(f"[Tempo Grid Search] {initial_bpm:.1f} → {best_bpm:.1f} BPM "
              f"(grid error: {best_error*1000:.1f}ms, phase: {best_phase*1000:.0f}ms, "
              f"confidence: {confidence:.2f})")

    return {
        'bpm': best_bpm,
        'beat_interval': beat_int,
        'beat_times': synthetic_beats,
        'confidence': confidence,
        'grid_error': best_error,
        'phase_offset': best_phase,
    }


def detect_triplets(notes, bpm=120, tolerance=0.15):
    """
    Detect triplet patterns in a sequence of notes.
    Triplets are EXACTLY 3 notes played in the time of 2 regular notes.

    STRICT REQUIREMENTS:
    1. Must have exactly 3 consecutive notes (or a multiple of 3, split into groups)
    2. Both inter-note spacings must be nearly equal to each other
    3. Both spacings must match a known triplet pattern for the tempo
    4. The note before (if any) must NOT have the same spacing
    5. The note after the run must NOT have the same spacing
    6. All notes must be valid (have time_seconds)
    7. Grace notes are excluded from triplet detection
    8. Rests can be part of a triplet group but if all 3 are rests, use a single rest

    Args:
        notes: List of note dictionaries with 'time_seconds' field
        bpm: Beats per minute
        tolerance: Timing tolerance as fraction (0.15 = 15% tolerance)

    Returns:
        List of notes with triplet information added
    """
    if len(notes) < 3:
        return notes

    beat_duration = 60.0 / bpm

    # Triplet patterns: (type, expected_spacing_between_notes)
    triplet_patterns = [
        ('half', beat_duration * 4 / 3),      # 3 in time of 2 half notes
        ('quarter', beat_duration * 2 / 3),   # 3 in time of 2 quarters
        ('eighth', beat_duration / 3),         # 3 in time of 2 eighths
        ('16th', beat_duration / 6),           # 3 in time of 2 16ths
        ('32nd', beat_duration / 12),          # 3 in time of 2 32nds
    ]

    # Track which note indices are part of a triplet
    triplet_assigned = set()

    i = 0
    while i <= len(notes) - 3:  # Need at least 3 notes from position i
        if i in triplet_assigned:
            i += 1
            continue

        # Skip grace notes - they cannot be part of triplets
        if notes[i].get('ornament') == 'grace':
            i += 1
            continue

        # Get the first 3 notes (skipping grace notes in between)
        candidates = []
        ci = i
        while ci < len(notes) and len(candidates) < 3:
            if notes[ci].get('ornament') == 'grace':
                ci += 1
                continue
            candidates.append(ci)
            ci += 1

        if len(candidates) < 3:
            i += 1
            continue

        idx0, idx1, idx2 = candidates[0], candidates[1], candidates[2]
        note0 = notes[idx0]
        note1 = notes[idx1]
        note2 = notes[idx2]

        # VALIDATION: All notes must have valid time_seconds
        t0 = note0.get('time_seconds')
        t1 = note1.get('time_seconds')
        t2 = note2.get('time_seconds')

        if t0 is None or t1 is None or t2 is None:
            i += 1
            continue

        # Calculate spacings between consecutive notes
        spacing1 = t1 - t0
        spacing2 = t2 - t1

        # VALIDATION: Spacings must be positive
        if spacing1 <= 0 or spacing2 <= 0:
            i += 1
            continue

        # VALIDATION: The two spacings must be very close to each other
        avg_spacing = (spacing1 + spacing2) / 2
        if avg_spacing < 0.02:  # Too fast to be meaningful
            i += 1
            continue

        spacing_diff = abs(spacing1 - spacing2) / avg_spacing
        if spacing_diff > tolerance:
            # Spacings are too different - not a triplet
            i += 1
            continue

        # Try to match a triplet pattern
        matched = False
        for triplet_type, expected_spacing in triplet_patterns:
            tol = expected_spacing * tolerance

            # VALIDATION: Both spacings must match expected triplet spacing
            if abs(spacing1 - expected_spacing) > tol:
                continue
            if abs(spacing2 - expected_spacing) > tol:
                continue

            # VALIDATION: Note BEFORE must NOT have the same spacing (ensures start of run)
            if idx0 > 0:
                # Find previous non-grace note
                prev_idx = idx0 - 1
                while prev_idx >= 0 and notes[prev_idx].get('ornament') == 'grace':
                    prev_idx -= 1
                if prev_idx >= 0:
                    t_prev = notes[prev_idx].get('time_seconds')
                    if t_prev is not None:
                        spacing_before = t0 - t_prev
                        if spacing_before > 0 and abs(spacing_before - expected_spacing) <= tol:
                            # Previous note has same spacing - we're in the middle of something
                            continue

            # Count how many consecutive notes share this spacing (the full run)
            run_indices = [idx0, idx1, idx2]
            scan = idx2
            while True:
                # Find next non-grace note
                next_scan = scan + 1
                while next_scan < len(notes) and notes[next_scan].get('ornament') == 'grace':
                    next_scan += 1
                if next_scan >= len(notes):
                    break
                t_curr = notes[scan].get('time_seconds')
                t_next = notes[next_scan].get('time_seconds')
                if t_curr is None or t_next is None:
                    break
                sp = t_next - t_curr
                if sp > 0 and abs(sp - expected_spacing) <= tol:
                    run_indices.append(next_scan)
                    scan = next_scan
                else:
                    break

            # VALIDATION: Note AFTER the run must NOT have the same spacing
            last_run_idx = run_indices[-1]
            next_after = last_run_idx + 1
            while next_after < len(notes) and notes[next_after].get('ornament') == 'grace':
                next_after += 1
            if next_after < len(notes):
                t_last = notes[last_run_idx].get('time_seconds')
                t_after = notes[next_after].get('time_seconds')
                if t_last is not None and t_after is not None:
                    spacing_after = t_after - t_last
                    if spacing_after > 0 and abs(spacing_after - expected_spacing) <= tol:
                        # Run extends further - but we already scanned it, shouldn't happen
                        continue

            run_len = len(run_indices)

            # Only use the portion that's a multiple of 3
            usable = (run_len // 3) * 3
            if usable < 3:
                continue

            triplet_beats = {
                'half': 4/3,
                'quarter': 2/3,
                'eighth': 1/3,
                '16th': 1/6,
                '32nd': 1/12,
            }[triplet_type]

            # Split into groups of 3 and mark each group
            for g in range(0, usable, 3):
                gi0 = run_indices[g]
                gi1 = run_indices[g + 1]
                gi2 = run_indices[g + 2]

                triplet_assigned.add(gi0)
                triplet_assigned.add(gi1)
                triplet_assigned.add(gi2)

                notes[gi0].update({
                    'triplet': True,
                    'triplet_position': 'start',
                    'triplet_type': triplet_type,
                    'actual_notes': 3,
                    'normal_notes': 2,
                    'note_value': triplet_type,
                    'note_divisions': triplet_beats,
                    'dotted': False
                })
                notes[gi1].update({
                    'triplet': True,
                    'triplet_position': 'middle',
                    'triplet_type': triplet_type,
                    'actual_notes': 3,
                    'normal_notes': 2,
                    'note_value': triplet_type,
                    'note_divisions': triplet_beats,
                    'dotted': False
                })
                notes[gi2].update({
                    'triplet': True,
                    'triplet_position': 'end',
                    'triplet_type': triplet_type,
                    'actual_notes': 3,
                    'normal_notes': 2,
                    'note_value': triplet_type,
                    'note_divisions': triplet_beats,
                    'dotted': False
                })

            matched = True
            # Advance past the entire run (usable portion)
            i = run_indices[usable - 1] + 1
            break

        if not matched:
            i += 1

    return notes


def detect_triplets_in_chords(chords, bpm=120, tolerance=0.15):
    """
    Detect triplet patterns in a sequence of chords.

    STRICT REQUIREMENTS (same as detect_triplets):
    1. Must have 3 consecutive chords (or multiple of 3, split into groups)
    2. Both inter-chord spacings must be nearly equal to each other
    3. Both spacings must match a known triplet pattern for the tempo
    4. The chord before (if any) must NOT have the same spacing
    5. The chord after the run must NOT have the same spacing
    6. All chords must have valid time_seconds
    """
    if len(chords) < 3:
        return chords

    beat_duration = 60.0 / bpm

    triplet_patterns = [
        ('half', beat_duration * 4 / 3),
        ('quarter', beat_duration * 2 / 3),
        ('eighth', beat_duration / 3),
        ('16th', beat_duration / 6),
        ('32nd', beat_duration / 12),
    ]

    triplet_assigned = set()

    i = 0
    while i <= len(chords) - 3:
        if i in triplet_assigned:
            i += 1
            continue

        chord0 = chords[i]
        chord1 = chords[i + 1]
        chord2 = chords[i + 2]

        # VALIDATION: All chords must have valid time_seconds
        t0 = chord0.get('time_seconds')
        t1 = chord1.get('time_seconds')
        t2 = chord2.get('time_seconds')

        if t0 is None or t1 is None or t2 is None:
            i += 1
            continue

        # Calculate spacings
        spacing1 = t1 - t0
        spacing2 = t2 - t1

        # VALIDATION: Spacings must be positive
        if spacing1 <= 0 or spacing2 <= 0:
            i += 1
            continue

        # VALIDATION: The two spacings must be very close to each other
        avg_spacing = (spacing1 + spacing2) / 2
        if avg_spacing < 0.02:  # Too fast
            i += 1
            continue

        spacing_diff = abs(spacing1 - spacing2) / avg_spacing
        if spacing_diff > tolerance:
            i += 1
            continue

        # Try to match a triplet pattern
        matched = False
        for triplet_type, expected_spacing in triplet_patterns:
            tol = expected_spacing * tolerance

            # VALIDATION: Both spacings must match expected triplet spacing
            if abs(spacing1 - expected_spacing) > tol:
                continue
            if abs(spacing2 - expected_spacing) > tol:
                continue

            # VALIDATION: Chord BEFORE must NOT have the same spacing (ensures start of run)
            if i > 0:
                t_prev = chords[i - 1].get('time_seconds')
                if t_prev is not None:
                    spacing_before = t0 - t_prev
                    if spacing_before > 0 and abs(spacing_before - expected_spacing) <= tol:
                        continue

            # Count how many consecutive chords share this spacing (the full run)
            run_indices = [i, i + 1, i + 2]
            scan = i + 2
            while scan + 1 < len(chords):
                t_curr = chords[scan].get('time_seconds')
                t_next = chords[scan + 1].get('time_seconds')
                if t_curr is None or t_next is None:
                    break
                sp = t_next - t_curr
                if sp > 0 and abs(sp - expected_spacing) <= tol:
                    run_indices.append(scan + 1)
                    scan += 1
                else:
                    break

            # VALIDATION: Chord AFTER the run must NOT have the same spacing
            last_run_idx = run_indices[-1]
            if last_run_idx + 1 < len(chords):
                t_last = chords[last_run_idx].get('time_seconds')
                t_after = chords[last_run_idx + 1].get('time_seconds')
                if t_last is not None and t_after is not None:
                    spacing_after = t_after - t_last
                    if spacing_after > 0 and abs(spacing_after - expected_spacing) <= tol:
                        continue

            run_len = len(run_indices)

            # Only use the portion that's a multiple of 3
            usable = (run_len // 3) * 3
            if usable < 3:
                continue

            triplet_beats = {
                'half': 4/3,
                'quarter': 2/3,
                'eighth': 1/3,
                '16th': 1/6,
                '32nd': 1/12,
            }[triplet_type]

            # Split into groups of 3 and mark each group
            for g in range(0, usable, 3):
                gi0 = run_indices[g]
                gi1 = run_indices[g + 1]
                gi2 = run_indices[g + 2]

                triplet_assigned.add(gi0)
                triplet_assigned.add(gi1)
                triplet_assigned.add(gi2)

                for idx, pos in [(gi0, 'start'), (gi1, 'middle'), (gi2, 'end')]:
                    chords[idx].update({
                        'triplet': True,
                        'triplet_position': pos,
                        'triplet_type': triplet_type,
                        'actual_notes': 3,
                        'normal_notes': 2,
                        'note_value': triplet_type,
                        'note_divisions': triplet_beats,
                        'dotted': False
                    })

            matched = True
            i = run_indices[usable - 1] + 1
            break

        if not matched:
            i += 1

    return chords

#* ─── Spectral Gate Noise Filter ──────────────────────────────────────────────
def spectral_gate_filter(audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=512, 
                         noise_floor_percentile=15, gate_threshold_db=-15, 
                         softness=3.0):
    """
    Optimized non-ML spectral gate using scipy STFT and vectorized operations.
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        n_fft: FFT size for STFT
        hop_length: Hop size for STFT
        noise_floor_percentile: Percentile for noise floor estimation (lower = more conservative)
        gate_threshold_db: Threshold above noise floor in dB
        softness: Softness parameter for sigmoid gate (higher = softer transition)
    
    Returns:
        Filtered audio signal
    
    Time complexity: O(n log n) - dominated by FFT operations
    """
    # Use scipy's faster STFT
    window = get_window('hann', n_fft, fftbins=True)
    _, _, stft_data = scipy_stft(audio, fs=sr, window=window, nperseg=n_fft, 
                                  noverlap=n_fft-hop_length, return_onesided=True)
    
    # Compute magnitude (keep as float32 for speed)
    magnitude = np.abs(stft_data).astype(np.float32)
    
    # Fast noise floor estimation - vectorized percentile per bin
    noise_floor = np.percentile(magnitude, noise_floor_percentile, axis=1, keepdims=True).astype(np.float32)
    
    # Compute gate threshold (vectorized)
    gate_threshold_linear = 10.0 ** (gate_threshold_db / 20.0)
    threshold = noise_floor * gate_threshold_linear
    
    # Apply soft mask (fully vectorized sigmoid)
    ratio = magnitude / (threshold + 1e-10)
    soft_mask = 1.0 / (1.0 + np.exp(-softness * (ratio - 1.0)))
    
    # Apply mask
    filtered_magnitude = magnitude * soft_mask
    
    # Reconstruct with original phase
    phase = np.angle(stft_data)
    filtered_stft = filtered_magnitude * np.exp(1j * phase)
    
    # Faster inverse STFT using scipy
    _, filtered_audio = scipy_istft(filtered_stft, fs=sr, window=window, nperseg=n_fft, 
                                     noverlap=n_fft-hop_length, input_onesided=True)
    
    # Ensure output length matches input
    if len(filtered_audio) > len(audio):
        filtered_audio = filtered_audio[:len(audio)]
    elif len(filtered_audio) < len(audio):
        filtered_audio = np.pad(filtered_audio, (0, len(audio) - len(filtered_audio)), mode='constant')
    
    return filtered_audio.astype(np.float32)


#* ─── Improved Multi-Band Spectral Gate ───────────────────────────────────────
def multiband_spectral_gate(audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=512,
                            noise_estimation_seconds=0.1, gate_threshold_db=-12,
                            min_gate_threshold_db=-40):
    """
    Multi-band spectral gate with improved noise estimation.
    
    Key improvements over basic spectral gate:
    1. Estimates noise from quietest frames (not just percentile)
    2. Uses different thresholds for different frequency bands
    3. Preserves transients better with attack-aware gating
    4. Sub-bass and ultra-high suppression to remove rumble and hiss
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        n_fft: FFT size for STFT
        hop_length: Hop size for STFT
        noise_estimation_seconds: Duration to use for noise floor estimation
        gate_threshold_db: Threshold above noise floor in dB (main band)
        min_gate_threshold_db: Minimum absolute threshold to prevent over-gating
    
    Returns:
        Filtered audio signal, noise_removed_db (for logging)
    """
    # Compute STFT
    window = get_window('hann', n_fft, fftbins=True)
    _, _, stft_data = scipy_stft(audio, fs=sr, window=window, nperseg=n_fft, 
                                  noverlap=n_fft-hop_length, return_onesided=True)
    
    magnitude = np.abs(stft_data).astype(np.float32)
    n_bins, n_frames = magnitude.shape
    freqs = np.fft.rfftfreq(n_fft, 1.0/sr)
    
    # --- IMPROVED NOISE ESTIMATION ---
    # Find the quietest frames (lowest RMS) to estimate noise floor
    frame_rms = np.sqrt(np.mean(magnitude**2, axis=0))
    n_noise_frames = max(5, int(noise_estimation_seconds * sr / hop_length))
    quietest_frame_indices = np.argsort(frame_rms)[:n_noise_frames]
    
    # Noise floor = mean magnitude of quietest frames per frequency bin
    noise_floor = np.mean(magnitude[:, quietest_frame_indices], axis=1, keepdims=True)
    noise_floor = np.maximum(noise_floor, 1e-10)  # Prevent division by zero
    
    # --- MULTI-BAND THRESHOLDS ---
    # Different frequency regions need different treatment
    # Sub-bass (<40Hz): Aggressive gating - mostly room rumble
    # Bass (40-200Hz): Moderate gating - musical content
    # Mid (200-2000Hz): Conservative gating - most musical content
    # High (2000-8000Hz): Moderate gating - harmonics
    # Ultra-high (>8000Hz): Aggressive gating - mostly hiss
    
    band_thresholds = np.ones(n_bins) * gate_threshold_db
    
    for i, f in enumerate(freqs):
        if f < 40:
            band_thresholds[i] = gate_threshold_db - 10  # More aggressive (lower threshold = more gating)
        elif f < 200:
            band_thresholds[i] = gate_threshold_db - 3
        elif f < 2000:
            band_thresholds[i] = gate_threshold_db  # Most conservative for musical content
        elif f < 8000:
            band_thresholds[i] = gate_threshold_db - 3
        else:
            band_thresholds[i] = gate_threshold_db - 8  # Aggressive for high-frequency hiss
    
    band_thresholds = band_thresholds.reshape(-1, 1)
    
    # Convert thresholds to linear scale
    threshold_linear = 10.0 ** (band_thresholds / 20.0)
    min_threshold_linear = 10.0 ** (min_gate_threshold_db / 20.0)
    
    # Compute adaptive threshold per bin
    threshold = np.maximum(noise_floor * threshold_linear, min_threshold_linear)
    
    # --- SOFT GATING WITH ATTACK PRESERVATION ---
    # Compute signal-to-noise ratio
    snr = magnitude / threshold
    
    # Detect transients (sudden increases in energy)
    frame_energy = np.sum(magnitude**2, axis=0)
    energy_diff = np.diff(frame_energy, prepend=frame_energy[0])
    transient_mask = energy_diff > np.percentile(energy_diff, 90)
    
    # Soft gate with sigmoid (softer transition = less artifacts)
    softness = 4.0
    gate_mask = 1.0 / (1.0 + np.exp(-softness * (snr - 1.0)))
    
    # Preserve transients - reduce gating during attacks
    for i, is_transient in enumerate(transient_mask):
        if is_transient:
            # Blend toward unity (less gating) during transients
            gate_mask[:, i] = gate_mask[:, i] * 0.5 + 0.5
    
    # --- APPLY GATE ---
    filtered_magnitude = magnitude * gate_mask
    
    # Compute noise removal stats
    original_power = np.sum(magnitude**2)
    filtered_power = np.sum(filtered_magnitude**2)
    noise_removed_db = 10 * np.log10(original_power / (filtered_power + 1e-10))
    
    # Reconstruct signal
    phase = np.angle(stft_data)
    filtered_stft = filtered_magnitude * np.exp(1j * phase)
    
    _, filtered_audio = scipy_istft(filtered_stft, fs=sr, window=window, nperseg=n_fft, 
                                     noverlap=n_fft-hop_length, input_onesided=True)
    
    # Match output length
    if len(filtered_audio) > len(audio):
        filtered_audio = filtered_audio[:len(audio)]
    elif len(filtered_audio) < len(audio):
        filtered_audio = np.pad(filtered_audio, (0, len(audio) - len(filtered_audio)), mode='constant')
    
    return filtered_audio.astype(np.float32), float(noise_removed_db)

#* ─── Persistent Tone Removal ────────────────────────────────────────────────
def remove_persistent_tones(audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=512,
                            persistence_percentile=10, subtraction_strength=0.8,
                            min_freq=30, max_freq=4000):
    """
    Remove background tones that persist throughout the recording.
    
    Background noise like HVAC, electrical hum (60Hz), room resonance, etc. stays
    at roughly constant levels throughout the recording. Real music has
    varying dynamics.
    
    Approach: Use the LOW percentile (e.g. 10th) of each frequency bin as the
    "floor" that's always there. Only consider frequencies in the musical range
    where persistent noise actually matters.
    
    Args:
        audio: Input audio signal
        sr: Sample rate  
        n_fft: FFT size
        hop_length: Hop size
        persistence_percentile: LOW percentile to use as noise floor (10 = 10th percentile)
        subtraction_strength: How aggressively to remove (0-1)
        min_freq: Minimum frequency to consider (Hz)
        max_freq: Maximum frequency to consider (Hz) - above this is less relevant
    
    Returns:
        Filtered audio, noise_reduction_db
    """
    window = get_window('hann', n_fft, fftbins=True)
    _, _, stft_data = scipy_stft(audio, fs=sr, window=window, nperseg=n_fft, 
                                  noverlap=n_fft-hop_length, return_onesided=True)
    
    magnitude = np.abs(stft_data).astype(np.float32)
    phase = np.angle(stft_data)
    freqs = np.fft.rfftfreq(n_fft, 1/sr)
    
    # Use low percentile as the "persistent floor" for each frequency bin
    persistent_floor = np.percentile(magnitude, persistence_percentile, axis=1, keepdims=True)
    
    # Get median and max for analysis
    median_magnitude = np.median(magnitude, axis=1, keepdims=True)
    max_magnitude = np.max(magnitude, axis=1, keepdims=True)
    
    # Overall signal level for threshold
    overall_median = np.median(magnitude)
    overall_max = np.max(magnitude)
    
    # Persistence score: how close is the floor to the median?
    persistence_score = persistent_floor / (median_magnitude + 1e-10)
    
    # Create frequency mask - only consider musical range
    freq_mask = (freqs >= min_freq) & (freqs <= max_freq)
    
    # Require MINIMUM ABSOLUTE ENERGY to be considered persistent
    # Must be at least 5% of overall median - this filters out near-silent bins
    min_energy_threshold = 0.05 * overall_median
    has_energy = (persistent_floor[:, 0] > min_energy_threshold)
    
    # Combined mask: in frequency range AND has real energy AND high persistence
    high_persistence = (persistence_score[:, 0] > 0.2) & freq_mask & has_energy
    n_persistent = np.sum(high_persistence)
    
    print(f"[Persistent Tone] Analyzing {np.sum(freq_mask)} bins in {min_freq}-{max_freq} Hz range...")
    print(f"[Persistent Tone] Overall magnitude: median={overall_median:.6f}, max={overall_max:.6f}")
    print(f"[Persistent Tone] Min energy threshold: {min_energy_threshold:.6f}")
    print(f"[Persistent Tone] Found {n_persistent} bins with persistence > 0.2 AND sufficient energy")
    
    # Check if the recording is very clean (low persistent noise)
    total_persistent_energy = np.sum(persistent_floor[high_persistence, 0])
    total_signal_energy = np.sum(median_magnitude[:, 0])
    persistent_ratio = total_persistent_energy / (total_signal_energy + 1e-10)
    print(f"[Persistent Tone] Persistent energy ratio: {persistent_ratio:.4f} ({persistent_ratio*100:.2f}% of signal)")
    
    if n_persistent > 0:
        persistent_indices = np.where(high_persistence)[0]
        persistent_freqs = freqs[persistent_indices]
        scores = persistence_score[persistent_indices, 0]
        floors = persistent_floor[persistent_indices, 0]
        sorted_idx = np.argsort(-scores)[:10]
        print(f"[Persistent Tone] Top persistent frequencies:")
        for i in sorted_idx:
            print(f"  {persistent_freqs[i]:.1f} Hz: score={scores[i]:.3f}, floor={floors[i]:.6f}")
    
    # Build subtraction mask - only subtract from identified persistent bins
    subtraction_mask = np.zeros_like(persistent_floor)
    subtraction_mask[high_persistence, :] = subtraction_strength * persistence_score[high_persistence, :]
    
    magnitude_cleaned = np.maximum(
        magnitude - subtraction_mask * persistent_floor,
        magnitude * 0.01  # Floor at 1% to prevent complete nulling
    )
    
    # Compute reduction stats
    original_power = np.sum(magnitude**2)
    cleaned_power = np.sum(magnitude_cleaned**2)
    noise_reduction_db = 10 * np.log10(original_power / (cleaned_power + 1e-10))
    
    # Reconstruct
    filtered_stft = magnitude_cleaned * np.exp(1j * phase)
    _, filtered_audio = scipy_istft(filtered_stft, fs=sr, window=window, nperseg=n_fft, 
                                     noverlap=n_fft-hop_length, input_onesided=True)
    
    # Match output length
    if len(filtered_audio) > len(audio):
        filtered_audio = filtered_audio[:len(audio)]
    elif len(filtered_audio) < len(audio):
        filtered_audio = np.pad(filtered_audio, (0, len(audio) - len(filtered_audio)), mode='constant')
    
    return filtered_audio.astype(np.float32), float(noise_reduction_db)

#* ─── Wiener Filter (Second Layer) ───────────────────────────────────────────
def wiener_filter(audio, sr=SAMPLE_RATE, n_fft=2048, hop_length=512,
                  noise_estimation_frames=10, oversubtraction_factor=1.5):
    """
    Wiener-style spectral gain filter for noise reduction.
    
    This filter estimates the SNR (Signal-to-Noise Ratio) per frequency bin
    and applies a gain based on the Wiener filter formula:
        Gain = max(0, 1 - (noise_power / signal_power) * factor)
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        n_fft: FFT size for STFT
        hop_length: Hop size for STFT
        noise_estimation_frames: Number of initial frames to use for noise estimation
        oversubtraction_factor: Factor to amplify noise subtraction (>1 = more aggressive)
    
    Returns:
        Filtered audio signal
    
    Time complexity: O(n log n) - dominated by FFT operations
    """
    # Use scipy's faster STFT
    window = get_window('hann', n_fft, fftbins=True)
    _, _, stft_data = scipy_stft(audio, fs=sr, window=window, nperseg=n_fft, 
                                  noverlap=n_fft-hop_length, return_onesided=True)
    
    # Compute power spectrum (magnitude squared)
    power = np.abs(stft_data).astype(np.float32) ** 2
    
    # Estimate noise power from initial frames (assumed to be quieter)
    # Use minimum statistics across a sliding window for better noise tracking
    noise_frames = min(noise_estimation_frames, power.shape[1])
    
    # Use minimum across initial frames as noise estimate
    noise_power = np.min(power[:, :noise_frames], axis=1, keepdims=True).astype(np.float32)
    
    # For better tracking, also use a running minimum across all frames
    # This helps when noise characteristics change over time
    running_min_window = 20  # frames
    noise_power_running = np.zeros_like(power)
    
    for i in range(power.shape[1]):
        start_idx = max(0, i - running_min_window // 2)
        end_idx = min(power.shape[1], i + running_min_window // 2 + 1)
        noise_power_running[:, i:i+1] = np.min(power[:, start_idx:end_idx], 
                                                axis=1, keepdims=True)
    
    # Use the maximum of initial estimate and running minimum
    # (ensures we don't underestimate noise)
    noise_power_estimate = np.maximum(noise_power, noise_power_running)
    
    # Compute Wiener gain: G = max(0, 1 - α * (N / S))
    # where N is noise power, S is signal power, α is oversubtraction factor
    snr_ratio = (noise_power_estimate * oversubtraction_factor) / (power + 1e-10)
    wiener_gain = np.maximum(0.0, 1.0 - snr_ratio)
    
    # Apply gain to magnitude (not power), so take sqrt of gain
    # This gives smoother results
    magnitude_gain = np.sqrt(wiener_gain)
    
    # Apply gain to original STFT
    filtered_stft = stft_data * magnitude_gain
    
    # Inverse STFT using scipy
    _, filtered_audio = scipy_istft(filtered_stft, fs=sr, window=window, nperseg=n_fft, 
                                     noverlap=n_fft-hop_length, input_onesided=True)
    
    # Ensure output length matches input
    if len(filtered_audio) > len(audio):
        filtered_audio = filtered_audio[:len(audio)]
    elif len(filtered_audio) < len(audio):
        filtered_audio = np.pad(filtered_audio, (0, len(audio) - len(filtered_audio)), mode='constant')
    
    return filtered_audio.astype(np.float32)

#* ─── OPTIMIZED: Compute-once STFT/CQT helpers ──────────────────────────────
def compute_stft_once(audio, sr=SAMPLE_RATE, n_fft=FFT_SIZE, hop_length=HOP_SIZE):
    """
    Compute STFT once for reuse throughout analysis.
    CRITICAL: Must match frame_audio() + compute_magnitude() behavior EXACTLY.

    Uses GPU-accelerated STFT when CUDA is available, otherwise falls back
    to manual framing (matching numpy FFT behavior exactly).
    """
    if USE_GPU:
        return gpu_compute_stft_once(audio, sr, n_fft, hop_length, WINDOW_TYPE)

    # CPU fallback: manual framing to match frame_audio() exactly
    window = get_window(WINDOW_TYPE, n_fft, fftbins=True)

    num_frames = 1 + (len(audio) - n_fft) // hop_length

    # Pre-allocate STFT matrix (use complex128 to match np.fft.rfft default precision)
    stft_data = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.complex128)

    for i in range(num_frames):
        frame = audio[i * hop_length : i * hop_length + n_fft]
        windowed_frame = frame * window
        stft_data[:, i] = np.fft.rfft(windowed_frame, n=n_fft)

    magnitude = np.abs(stft_data)
    phase = np.angle(stft_data)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    return stft_data, magnitude, phase, freqs

def apply_filters_to_magnitude(magnitude, freqs, 
                               noise_floor_percentile=25, gate_threshold_db=-20,
                               suppress_sub_bass=True, sub_bass_cutoff=27.5):
    """Apply spectral gate directly to magnitude (Wiener filter temporarily disabled)"""
    # Spectral gate only
    noise_floor = np.percentile(magnitude, noise_floor_percentile, axis=1, keepdims=True)
    gate_threshold_linear = 10.0 ** (gate_threshold_db / 20.0)
    threshold = noise_floor * gate_threshold_linear
    ratio = magnitude / (threshold + 1e-10)
    gate_mask = 1.0 / (1.0 + np.exp(-5.0 * (ratio - 1.0)))
    
    # Sub-bass suppression
    if suppress_sub_bass:
        sub_bass_mask = freqs[:, np.newaxis] < sub_bass_cutoff
        gate_mask = np.where(sub_bass_mask, gate_mask * 0.03, gate_mask)
    
    magnitude_filtered = magnitude * gate_mask
    
    return magnitude_filtered

def compute_flux_from_magnitude(magnitude):
    """Compute spectral flux from pre-computed magnitude"""
    diffs = np.diff(magnitude, axis=1)
    flux = np.sum(np.square(np.clip(diffs, 0, None)), axis=0)
    return np.concatenate(([0.], flux))

#* ─── Frequency Range Separation ─────────────────────────────────────────────
def split_frequency_ranges(audio, sr=SAMPLE_RATE, split_midi=60):
    """
    Split audio into bass (left hand) and treble (right hand) frequency ranges.
    Uses bandpass filtering around a split point (default: C4 / MIDI 60 / 261.6 Hz).
    
    Args:
        audio: Input audio signal
        sr: Sample rate
        split_midi: MIDI note number to split at (default 60 = middle C)
    
    Returns:
        (bass_audio, treble_audio): Two filtered versions of the input
    """
    split_freq = 440.0 * 2**((split_midi - 69) / 12)  # Convert MIDI to Hz
    
    # Design filters (4th order Butterworth)
    # Bass: lowpass at split frequency
    sos_bass = butter(4, split_freq, btype='low', fs=sr, output='sos')
    bass_audio = sosfiltfilt(sos_bass, audio)
    
    # Treble: highpass at split frequency
    sos_treble = butter(4, split_freq, btype='high', fs=sr, output='sos')
    treble_audio = sosfiltfilt(sos_treble, audio)
    
    print(f"[Frequency Split] Split at MIDI {split_midi} ({split_freq:.1f} Hz)")
    print(f"[Frequency Split] Bass RMS: {np.sqrt(np.mean(bass_audio**2)):.4f}, Treble RMS: {np.sqrt(np.mean(treble_audio**2)):.4f}")
    
    return bass_audio, treble_audio

#* ─── Read + Noise Reduction Pipeline ────────────────────────────────────────
def read_wav(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr}")
    
    input_rms = np.sqrt(np.mean(audio**2))
    print(f"[Noise Pipeline] Input audio: {len(audio)} samples, RMS: {input_rms:.4f}")

    if USE_GPU:
        # ── GPU: Fused noise pipeline (1 STFT instead of 2 separate ones) ──
        audio, persistent_db, gate_db = fused_noise_reduce(
            audio, sr=sr, n_fft=FFT_SIZE, hop_length=HOP_SIZE,
            persistence_percentile=10, subtraction_strength=0.8,
            persistent_min_freq=30, persistent_max_freq=4000,
            noise_estimation_seconds=0.15, gate_threshold_db=-10,
            min_gate_threshold_db=-50
        )
    else:
        # ── CPU: Sequential noise pipeline (original) ──
        # Step 0: Remove persistent background tones (HVAC, 60Hz hum, room resonance)
        audio, persistent_db = remove_persistent_tones(
            audio, sr=sr, n_fft=FFT_SIZE, hop_length=HOP_SIZE,
            persistence_percentile=10,
            subtraction_strength=0.8
        )
        print(f"[Noise Pipeline] After persistent tone removal: {persistent_db:.2f} dB removed")

        # Step 1: Multi-band spectral gate
        audio, noise_removed_db = multiband_spectral_gate(
            audio, sr=sr, n_fft=FFT_SIZE, hop_length=HOP_SIZE,
            noise_estimation_seconds=0.15,
            gate_threshold_db=-10,
            min_gate_threshold_db=-50
        )
        print(f"[Noise Pipeline] After multiband spectral gate: {noise_removed_db:.2f} dB noise removed")

    # Step 2: noisereduce for residual non-stationary noise
    # (noisereduce has its own internal STFT, can't fuse further)
    audio_before = audio.copy()
    audio = nr.reduce_noise(
        y=audio, sr=sr,
        stationary=False,
        n_fft=FFT_SIZE,
        hop_length=HOP_SIZE,
        prop_decrease=0.6
    )
    rms_reduction = np.sqrt(np.mean(audio_before**2)) - np.sqrt(np.mean(audio**2))
    print(f"[Noise Pipeline] After noisereduce: RMS reduction = {rms_reduction:.4f}")

    # Step 3: High-pass filter to remove any remaining sub-bass rumble
    sos = butter(2, 30, btype='high', fs=sr, output='sos')
    audio = sosfiltfilt(sos, audio)

    final_rms = np.sqrt(np.mean(audio**2))
    total_reduction_db = 20 * np.log10(input_rms / (final_rms + 1e-10))
    print(f"[Noise Pipeline] Final RMS: {final_rms:.4f}, Total reduction: {total_reduction_db:.2f} dB")

    return audio.astype(np.float32)

#* ─── Frame Audio ───────────────────────────────────────────────────────────
def frame_audio(audio):
    win = get_window(WINDOW_TYPE, FRAME_SIZE, fftbins=True)
    num_frames = 1 + (len(audio) - FRAME_SIZE)//HOP_SIZE
    frames = np.stack([
        audio[i*HOP_SIZE : i*HOP_SIZE+FRAME_SIZE] * win
        for i in range(num_frames)
    ])
    return frames  # shape (T, FRAME_SIZE)

#* ─── Magnitude & Flux ──────────────────────────────────────────────────────
def compute_magnitude(frame):
    X = np.fft.rfft(frame, n=FFT_SIZE)
    return np.abs(X)  # shape (MAG_SIZE,)

def compute_flux(mags):
    diffs = np.diff(mags, axis=0)
    flux = np.sum(np.square(np.clip(diffs, 0, None)), axis=1)
    return np.concatenate(([0.], flux))  # pad to same length

def normalize(v):
    mx = np.max(v)
    return v/mx if mx>0 else v

def find_onsets(flux, window=50, K=2.0):
    """Find onsets using adaptive threshold. K=2.0 means 2 std devs above mean."""
    onsets = []
    buf = []
    for t in range(1, len(flux)-1):
        buf.append(flux[t])
        if len(buf)>window: buf.pop(0)
        μ = np.mean(buf)
        s = np.std(buf)
        thresh = μ + K*s
        if flux[t]>flux[t-1] and flux[t]>flux[t+1] and flux[t]>thresh:
            onsets.append(t)
    return onsets


def find_onsets_with_slope_validation(flux, window=50, K=2.0, min_slope_ratio=0.3, 
                                       slope_window=3, debug=False):
    """
    Find onsets with additional slope/sharpness validation.
    
    Real piano attacks have sharp transients - the energy rises very quickly.
    Noise-induced false onsets tend to have gradual energy buildup.
    
    Args:
        flux: Spectral flux array (normalized)
        window: Window size for adaptive threshold
        K: Number of std devs above mean for threshold
        min_slope_ratio: Minimum ratio of onset slope to peak value (0-1)
                        Higher = stricter (requires sharper attacks)
        slope_window: Number of frames before onset to measure slope
        debug: Print debug info
    
    Returns:
        List of validated onset frame indices
    """
    # First, find candidate onsets using standard method
    candidates = []
    buf = []
    for t in range(1, len(flux)-1):
        buf.append(flux[t])
        if len(buf) > window: 
            buf.pop(0)
        μ = np.mean(buf)
        s = np.std(buf)
        thresh = μ + K * s
        if flux[t] > flux[t-1] and flux[t] > flux[t+1] and flux[t] > thresh:
            candidates.append(t)
    
    if debug:
        print(f"[Slope] Found {len(candidates)} candidate onsets")
    
    # Validate each candidate by checking attack slope
    validated = []
    for onset_frame in candidates:
        # Get the pre-onset frames (before the peak)
        start_frame = max(0, onset_frame - slope_window)
        
        if start_frame >= onset_frame:
            # Not enough frames before onset, accept it
            validated.append(onset_frame)
            continue
        
        # Measure the slope: how much did flux increase leading up to onset?
        pre_onset_values = flux[start_frame:onset_frame + 1]
        
        if len(pre_onset_values) < 2:
            validated.append(onset_frame)
            continue
        
        # Calculate slope as (peak - baseline) / peak
        # This gives us a ratio: 1.0 = came from silence, 0.0 = no change
        baseline = np.min(pre_onset_values[:-1])  # Min value before peak
        peak = flux[onset_frame]
        
        if peak < 1e-6:
            # Very weak onset, skip
            continue
        
        slope_ratio = (peak - baseline) / peak
        
        if slope_ratio >= min_slope_ratio:
            validated.append(onset_frame)
            if debug:
                print(f"[Slope] ✓ Onset at frame {onset_frame}: slope_ratio={slope_ratio:.3f} >= {min_slope_ratio}")
        else:
            if debug:
                print(f"[Slope] ✗ Rejected frame {onset_frame}: slope_ratio={slope_ratio:.3f} < {min_slope_ratio}")
    
    if debug:
        print(f"[Slope] Validated {len(validated)}/{len(candidates)} onsets")
    
    return validated

#* ─── CQT & HPS Pitch Picker ────────────────────────────────────────────────
def compute_cqt(frame):
    # returns length-CQT_BINS magnitude vector
    # librosa.cqt returns complex; we take abs
    # Use a more conservative approach that avoids warnings
    C = np.abs(librosa.cqt(
        frame, sr=SAMPLE_RATE, 
        hop_length=HOP_SIZE,  # Use standard hop size
        n_bins=CQT_BINS, 
        bins_per_octave=12,
        fmin=bin_freq[0],  # Start from C1 (around 32.7 Hz)
        filter_scale=1.0,  # Default filter scale
        norm=1,  # L1 normalization
        window='hann'
    ))
    # Take mean across time dimension if we have multiple time steps
    if C.ndim > 1:
        C = np.mean(C, axis=1)
    return C.flatten()

def pick_pitches_HPS(cqt_mag, max_voices=4, max_h=5):
    # precompute harmonic offsets once
    global _hps_offsets
    if '_hps_offsets' not in globals():
        offsets = np.zeros((CQT_BINS, max_h+1), int)
        for b in range(CQT_BINS):
            for h in range(1, max_h+1):
                tgt = bin_freq[b]*h
                offsets[b,h] = np.argmin(np.abs(bin_freq - tgt))
        _hps_offsets = offsets

    residual = cqt_mag.copy()
    notes = []
    # Normalize CQT for better thresholding
    max_mag = np.max(cqt_mag)
    if max_mag == 0:
        return notes
    
    for voice in range(max_voices):
        hps = []
        for b in range(CQT_BINS):
            # Calculate HPS score with emphasis on fundamental strength
            fundamental_mag = residual[b]
            if fundamental_mag < 0.02 * max_mag:  # Skip only extremely weak fundamentals
                hps.append(-1e6)
                continue
                
            # Start with fundamental strength (heavily weighted)
            hps_score = np.log(fundamental_mag + 1e-8) * 3.0  # 3x weight for fundamental
            
            # Add harmonic support with decreasing weights
            harmonic_support = 0
            for h in range(2, max_h + 1):  # Start from 2nd harmonic
                idx = _hps_offsets[b, h]
                if idx < len(residual) and idx != b:  # Don't double-count fundamental
                    weight = 1.0 / (h * h)  # Quadratic decay for higher harmonics
                    harmonic_support += np.log(residual[idx] + 1e-8) * weight
            
            # Final score: fundamental + harmonic support
            hps_score += harmonic_support * 0.5  # Harmonics contribute 50% as much
            hps.append(hps_score)
        
        best = np.argmax(hps)
        
        # Threshold for additional voices — allow softer accompaniment through
        # Voice 0 (melody): 8% of max; voices 1+ (accompaniment): 10% of max
        min_threshold = np.log(0.08 * max_mag) if voice == 0 else np.log(0.10 * max_mag)
        
        if hps[best] < min_threshold:
            break
            
        notes.append(21 + best)
        
        # More aggressive harmonic subtraction
        for h in range(1, max_h+1):
            idx = _hps_offsets[best,h]
            if idx < len(residual):
                # Subtract more aggressively for higher harmonics
                subtraction_factor = 0.9 / h
                residual[idx] = max(0, residual[idx] - cqt_mag[idx] * subtraction_factor)
    
    return sorted(notes)

#* ─── Harmonic mixture model (BIC selection) ──────────────────────────────
def _precompute_fft_freqs(sr=SAMPLE_RATE, n_fft=FFT_SIZE):
    return np.fft.rfftfreq(n_fft, 1.0/sr)

_FFT_FREQS = _precompute_fft_freqs()

def _midi_to_hz(m): 
    return 440.0 * 2**((m - 69)/12)

def _template_for_midi(midi, freqs=_FFT_FREQS, H=8, sigma_bins=1.5):
    """Return a length-|freqs| template for this midi's harmonic series."""
    f0 = _midi_to_hz(midi)
    # convert a 'sigma' in Hz to bins dynamically per harmonic
    t = np.zeros_like(freqs, dtype=np.float32)
    for h in range(1, H+1):
        fh = h * f0
        if fh >= freqs[-1]: 
            break
        # nearest bin and gaussian spread (in bins)
        k = np.argmin(np.abs(freqs - fh))
        # wider spread for higher harmonics (slightly)
        sig_bins = sigma_bins * (1 + 0.1*(h-1))
        # local gaussian without allocating full vector (clip to a small window)
        rad = int(3*sig_bins) + 1
        lo = max(0, k - rad); hi = min(len(freqs), k + rad + 1)
        local = np.arange(lo, hi) - k
        t[lo:hi] += (1.0/h) * np.exp(-0.5*(local/sig_bins)**2)
    # normalize to unit norm so gains are meaningful
    nrm = np.linalg.norm(t) + 1e-12
    return t / nrm

# Cache templates for speed
_TEMPLATE_CACHE = {}
def get_template(midi):
    key = (int(midi), FFT_SIZE, SAMPLE_RATE)
    if key not in _TEMPLATE_CACHE:
        _TEMPLATE_CACHE[key] = _template_for_midi(int(midi))
    return _TEMPLATE_CACHE[key]

def _fit_nonneg_mixture(x, midis, iters=6):
    """Fast NNLS via multiplicative updates on a small template set."""
    if len(midis) == 0:
        return np.array([]), np.sum(x*x)
    D = np.stack([get_template(m) for m in midis], axis=1).astype(np.float32)  # (B,K)
    a = np.maximum(D.T @ x, 1e-8)  # init by projection
    Dt = D.T
    for _ in range(iters):
        num = Dt @ x
        den = Dt @ (D @ a) + 1e-12
        a *= num / den
    recon = D @ a
    err = np.sum((x - recon)**2)
    return a, err

def _bic(err, B, dof):
    # err = sum of squared residuals, B = #bins in x, dof approx = #active gains + K (rough)
    return B * np.log(max(err, 1e-18) / B) + dof * np.log(B)

def _salience_candidates_from_fft(mag, top=8, H=6):
    """
    Improved salience: score each MIDI by fundamental strength + harmonic support.
    Uses peak detection with parabolic interpolation for sub-bin accuracy.
    """
    midi_lo, midi_hi = 21, 108
    freqs = _FFT_FREQS
    
    # Find actual peaks in the spectrum with parabolic interpolation
    # This gives us sub-bin frequency accuracy
    peaks = []  # List of (bin_index, interpolated_freq, magnitude)
    mag_thresh = 0.04 * np.max(mag)  # Low threshold for soft accompaniment detection
    for i in range(1, len(mag) - 1):
        if mag[i] > mag[i-1] and mag[i] > mag[i+1] and mag[i] > mag_thresh:
            # Parabolic interpolation for sub-bin accuracy
            y1, y2, y3 = mag[i-1], mag[i], mag[i+1]
            # Fit parabola: offset from center bin
            denom = y1 - 2*y2 + y3
            if abs(denom) > 1e-10:
                delta = 0.5 * (y1 - y3) / denom
                delta = np.clip(delta, -0.5, 0.5)  # Sanity check
            else:
                delta = 0.0
            
            interp_bin = i + delta
            interp_freq = interp_bin * (freqs[1] - freqs[0])  # bin_width * bin_index
            interp_mag = y2 - 0.25 * (y1 - y3) * delta  # Interpolated magnitude
            peaks.append((i, interp_freq, interp_mag))
    
    scores = []
    for m in range(midi_lo, midi_hi + 1):
        f0 = _midi_to_hz(m)
        
        # Find the closest peak to this MIDI's fundamental frequency
        # Use interpolated frequencies for better accuracy
        closest_peak = None
        min_freq_diff = float('inf')
        for bin_idx, peak_freq, peak_mag in peaks:
            freq_diff = abs(peak_freq - f0)
            if freq_diff < min_freq_diff:
                min_freq_diff = freq_diff
                closest_peak = (bin_idx, peak_freq, peak_mag)
        
        # Tolerance: half a semitone = f0 * (2^(1/24) - 1) ≈ 2.9% of f0
        semitone_tolerance = f0 * 0.029  # Half semitone
        has_fund_peak = closest_peak is not None and min_freq_diff < semitone_tolerance
        
        # Score based on energy at fundamental frequency
        f0_bin = np.argmin(np.abs(freqs - f0))
        fund_energy = 0.0
        for offset in range(-2, 3):  # ±2 bins around f0
            bin_idx = f0_bin + offset
            if 0 <= bin_idx < len(mag):
                weight = 1.0 - 0.3 * abs(offset)  # Center weighted
                fund_energy += mag[bin_idx] * weight
        
        # Bonus if we found an interpolated peak very close to expected f0
        peak_bonus = 0.0
        if has_fund_peak:
            # Stronger bonus for closer match (linear falloff)
            closeness = 1.0 - (min_freq_diff / semitone_tolerance)
            peak_bonus = closest_peak[2] * closeness * 0.5  # Use peak magnitude
        
        # Add harmonic support (but less weight than fundamental)
        harmonic_support = 0.0
        for h in range(2, H + 1):
            fh = f0 * h
            if fh >= freqs[-1]:
                break
            h_bin = np.argmin(np.abs(freqs - fh))
            # Less weight for higher harmonics
            h_weight = 0.4 / h
            for offset in range(-1, 2):  # ±1 bin around harmonic
                bin_idx = h_bin + offset
                if 0 <= bin_idx < len(mag):
                    harmonic_support += mag[bin_idx] * h_weight * (1.0 - 0.3 * abs(offset))
        
        # Check for subharmonic evidence (is this note actually a harmonic of a lower note?)
        subharmonic_penalty = 0.0
        for sub_h in [2, 3]:
            sub_f0 = f0 / sub_h
            if sub_f0 >= freqs[1]:  # Above DC
                sub_bin = np.argmin(np.abs(freqs - sub_f0))
                # Check if there's significant energy at the potential fundamental
                sub_energy = sum(mag[max(0, sub_bin-1):min(len(mag), sub_bin+2)])
                if sub_energy > fund_energy * 0.3:  # If subharmonic is reasonably strong
                    subharmonic_penalty += sub_energy * 0.5 / sub_h
        
        # Final score: fund energy + peak bonus + harmonic support - subharmonic penalty
        score = fund_energy * (1.5 if has_fund_peak else 0.8) + peak_bonus + harmonic_support - subharmonic_penalty
        scores.append((score, m, fund_energy, peak_bonus, harmonic_support, subharmonic_penalty, has_fund_peak))
    
    scores.sort(reverse=True, key=lambda x: x[0])
    return [(s[1], s[0], s[2], s[3], s[4], s[5], s[6]) for s in scores[:top]]  # (midi, score, fund, peak_bonus, harm, subharm, has_peak)

def _temporal_filter_candidates(mag_frames, min_consecutive=2, top_per_frame=12, H=6, debug=False):
    """
    Run salience detection on multiple frames and filter to candidates
    that appear in at least `min_consecutive` consecutive frames.
    
    This acts as a simple HMM-like temporal filter to reject spurious 
    single-frame detections (noise, transients, harmonics).
    
    Args:
        mag_frames: List of 1D magnitude arrays (one per frame)
        min_consecutive: Minimum consecutive frames a candidate must appear in
        top_per_frame: How many top candidates to consider per frame
        H: Number of harmonics for salience calculation
        debug: Print debug info
    
    Returns:
        List of (midi, avg_score) for candidates that pass temporal filter,
        sorted by average score descending
    """
    if len(mag_frames) < min_consecutive:
        # Not enough frames - fall back to single-frame detection on average
        avg_mag = np.mean(mag_frames, axis=0) if len(mag_frames) > 0 else mag_frames[0]
        results = _salience_candidates_from_fft(avg_mag, top=top_per_frame, H=H)
        return [(midi, score) for midi, score, *_ in results]
    
    # Run salience detection on each frame
    frame_candidates = []
    for i, mag in enumerate(mag_frames):
        # Normalize each frame independently
        mag_norm = mag.astype(np.float32).copy()
        if mag_norm.max() > 0:
            mag_norm /= (mag_norm.max() + 1e-12)
        
        results = _salience_candidates_from_fft(mag_norm, top=top_per_frame, H=H)
        # Store as dict: midi -> score
        frame_cands = {midi: score for midi, score, *_ in results}
        frame_candidates.append(frame_cands)
    
    # Find candidates that appear in min_consecutive consecutive frames
    # A MIDI is "present" in a frame if it's in the top candidates
    all_midis = set()
    for fc in frame_candidates:
        all_midis.update(fc.keys())
    
    persistent_candidates = []
    
    for midi in all_midis:
        # Check for consecutive runs
        max_consecutive = 0
        current_consecutive = 0
        total_score = 0.0
        appearances = 0
        
        for fc in frame_candidates:
            if midi in fc:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
                total_score += fc[midi]
                appearances += 1
            else:
                current_consecutive = 0
        
        if max_consecutive >= min_consecutive:
            avg_score = total_score / appearances if appearances > 0 else 0
            persistent_candidates.append((midi, avg_score, max_consecutive, appearances))
    
    # Sort by average score (descending)
    persistent_candidates.sort(key=lambda x: x[1], reverse=True)
    
    if debug and persistent_candidates:
        print(f"  [Temporal Filter] {len(all_midis)} unique candidates across {len(mag_frames)} frames")
        print(f"  [Temporal Filter] {len(persistent_candidates)} passed (>={min_consecutive} consecutive frames):")
        for midi, avg_score, max_consec, appearances in persistent_candidates[:8]:
            note_name = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][midi % 12] + str(midi // 12 - 1)
            print(f"    {note_name:4s}: avg_score={avg_score:.4f}, max_consecutive={max_consec}, appearances={appearances}/{len(mag_frames)}")
    
    return [(midi, avg_score) for midi, avg_score, *_ in persistent_candidates]


def estimate_voices_bic(mag_window, max_K=3, H=8, debug=False, mag_frames=None, use_temporal_filter=True):
    """
    mag_window: 1D FFT magnitude you'd like to explain (ideally averaged over ±1 frame around the onset).
    mag_frames: Optional list of individual frame magnitudes for temporal filtering.
                If provided and use_temporal_filter=True, candidates must appear in 2+ frames.
    Returns: dict with {'K', 'midis', 'gains', 'bic', 'err'}
    """
    # Normalize the target spectrum so BIC compares apples to apples
    x = mag_window.astype(np.float32).copy()
    if x.max() > 0:
        x /= (x.max() + 1e-12)
    B = len(x)

    # Apply temporal filtering if we have multiple frames
    if use_temporal_filter and mag_frames is not None and len(mag_frames) >= 2:
        # Use temporal filter - only consider candidates in 2+ consecutive frames
        temporal_results = _temporal_filter_candidates(
            mag_frames, min_consecutive=2, top_per_frame=12, H=H, debug=debug
        )
        
        if temporal_results:
            # Use temporally-filtered candidates
            cand_midis = [midi for midi, score in temporal_results[:12]]
            # Build salience_info from temporal results
            salience_info = {midi: (score, True) for midi, score in temporal_results}
            
            if debug:
                print(f"  [BIC] Using {len(cand_midis)} temporally-filtered candidates")
        else:
            # No candidates passed temporal filter - fall back to single-frame
            if debug:
                print(f"  [BIC] No candidates passed temporal filter, using single-frame detection")
            cand_results = _salience_candidates_from_fft(x, top=12, H=H)
            salience_info = {midi: (score, has_peak) for midi, score, fund, peak_bonus, harm, subharm, has_peak in cand_results}
            cand_midis = [r[0] for r in cand_results]
    else:
        # No temporal filtering - use single-frame salience (original behavior)
        cand_results = _salience_candidates_from_fft(x, top=12, H=H)  # Get more candidates for filtering
        salience_info = {midi: (score, has_peak) for midi, score, fund, peak_bonus, harm, subharm, has_peak in cand_results}
        cand_midis = [r[0] for r in cand_results]
        
        if debug:
            print(f"\n  [Salience] Top candidates from FFT:")
            for midi, score, fund, peak_bonus, harm, subharm, has_peak in cand_results[:8]:
                note_name = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][midi % 12] + str(midi // 12 - 1)
                f0 = 440.0 * 2**((midi - 69)/12)
                print(f"    {note_name:4s} (MIDI {midi:3d}, {f0:6.1f}Hz): score={score:.4f} "
                      f"[fund={fund:.4f}, peak_bonus={peak_bonus:.4f}, harm={harm:.4f}, subharm_penalty={subharm:.4f}, has_peak={has_peak}]")
    
    # Keep top 8 candidates (no octave filtering - let CQT validation handle it)
    cand_midis = cand_midis[:8]

    best = {'K': 0, 'midis': [], 'gains': np.array([]), 'bic': _bic(np.sum(x*x), B, 0), 'err': float(np.sum(x*x))}
    # Try K=1..max_K by taking top-K candidates; refine by pruning tiny gains
    for K in range(1, max_K+1):
        midis = cand_midis[:K]
        gains, err = _fit_nonneg_mixture(x, midis, iters=6)
        # prune near-zero components and recompute (optional)
        keep = gains > (0.015 * gains.max())  # Allow soft accompaniment notes through
        if keep.any() and keep.sum() < K:
            midis = [m for m, k in zip(midis, keep) if k]
            gains, err = _fit_nonneg_mixture(x, midis, iters=4)
            K_eff = len(midis)
        else:
            K_eff = K
        bic = _bic(err, B, dof=K_eff*2.0)  # Slightly higher penalty for more notes
        
        if debug:
            midi_names = [['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][m % 12] + str(m // 12 - 1) for m in midis]
            print(f"  [BIC] K={K_eff}: midis={midi_names}, gains={gains[:K_eff]}, err={err:.6f}, bic={bic:.2f} {'← BEST' if bic < best['bic'] else ''}")
        
        if bic < best['bic']:
            best = {'K': K_eff, 'midis': midis, 'gains': gains[:K_eff], 'bic': bic, 'err': float(err)}
    
    if debug:
        best_names = [['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][m % 12] + str(m // 12 - 1) for m in best['midis']]
        print(f"  [BIC] Final selection: K={best['K']}, midis={best_names}")
    
    # Include salience_info so CQT validation can trust FFT peaks
    best['salience_info'] = salience_info
    return best

def validate_midi_with_cqt(midi_candidates, cqt_mag, tolerance_semitones=1, debug=False, salience_info=None, cqt_prev=None):
    """
    Cross-validate MIDI candidates against CQT peaks.
    
    Strategy: BIC tells us WHAT notes are playing (onset detection), CQT tells us 
    the EXACT pitch. When BIC and CQT disagree by 1 semitone, trust CQT's pitch.
    
    1. For each BIC candidate, find the nearest CQT peak
    2. If within tolerance, use the CQT peak's MIDI (more accurate pitch)
    3. If no CQT support and no harmonic evidence, reject the candidate
    """
    if cqt_mag is None or len(cqt_mag) == 0:
        return midi_candidates
    
    # CQT starts at A0 (MIDI 21) - the lowest piano key
    CQT_MIDI_OFFSET = 21  # A0 = MIDI 21
    
    # Find significant peaks in CQT with their magnitudes
    max_cqt = np.max(cqt_mag)
    threshold = 0.05 * max_cqt  # Low threshold to catch soft accompaniment notes
    cqt_peaks = []  # List of (midi, magnitude)
    cqt_peak_bins = []
    for i in range(1, len(cqt_mag) - 1):
        if cqt_mag[i] > cqt_mag[i-1] and cqt_mag[i] > cqt_mag[i+1] and cqt_mag[i] > threshold:
            cqt_peaks.append((CQT_MIDI_OFFSET + i, cqt_mag[i]))
            cqt_peak_bins.append(i)
    
    cqt_peak_midis = [p[0] for p in cqt_peaks]
    
    if debug:
        peak_names = [['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][m % 12] + str(m // 12 - 1) for m in cqt_peak_midis[:10]]
        print(f"  [CQT Validation] CQT peaks found: {peak_names} (bins: {cqt_peak_bins[:10]}, MIDI: {cqt_peak_midis[:10]})")
    
    validated = []
    
    for midi in midi_candidates:
        # Find the nearest CQT peak to this BIC candidate
        nearest_peak = None
        nearest_dist = float('inf')
        for peak_midi, peak_mag in cqt_peaks:
            dist = abs(midi - peak_midi)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_peak = (peak_midi, peak_mag)
        
        # Check harmonic support in CQT (for missing fundamental cases)
        harmonic_support = 0
        for h in [2, 3]:
            harmonic_midi = midi + 12 * int(np.log2(h))
            if any(abs(harmonic_midi - peak) <= tolerance_semitones for peak in cqt_peak_midis):
                harmonic_support += 1
        
        if nearest_peak and nearest_dist <= tolerance_semitones:
            # BIC candidate has CQT support - keep BIC's pitch (don't correct)
            if midi not in validated:
                validated.append(midi)
                if debug:
                    bic_name = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][midi % 12] + str(midi // 12 - 1)
                    print(f"    {bic_name}: ✓ KEPT (cqt_confirmed)")
        elif harmonic_support >= 2:
            # Missing fundamental case - keep BIC's pitch
            if midi not in validated:
                validated.append(midi)
                if debug:
                    note_name = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][midi % 12] + str(midi // 12 - 1)
                    print(f"    {note_name}: ✓ KEPT (harmonic_support={harmonic_support})")
        else:
            # No CQT support and no harmonic evidence - reject
            if debug:
                note_name = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][midi % 12] + str(midi // 12 - 1)
                print(f"    {note_name}: ✗ REJECTED (no CQT support, nearest peak {nearest_dist} semitones away)")
    
    # Sort by MIDI number for consistent output
    validated.sort()
    
    # Fallback: if nothing validated, keep the top BIC candidate
    result = validated if validated else midi_candidates[:1] if midi_candidates else []
    
    if debug:
        result_names = [['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][m % 12] + str(m // 12 - 1) for m in result]
        print(f"  [CQT Validation] Final validated: {result_names}")
    
    return result

#* ─── Chord Detection ────────────────────────────────────────────────────────
def make_templates():
    """Build normalized pitch-class templates for common chords."""
    templates, labels = [], []
    for i, root in enumerate(ROOTS):
        for quality, intervals in CHORD_INTERVALS.items():
            vec = np.zeros(12, dtype=float)
            for interval in intervals:
                vec[(i + interval) % 12] = 1.0
            vec /= np.linalg.norm(vec)
            templates.append(vec)
            labels.append(f"{root}:{quality}")
    return np.stack(templates), labels

CH_TEMPLATES, CH_LABELS = make_templates()
# single‐note templates = identity
NOTE_TEMPLATES = np.eye(12)

def extract_chroma(audio, sr, hop_length=512):
    C = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
    return C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-6)

def _cqt_center_freqs(n_bins, bpo, fmin):
    # Center frequencies for each CQT bin
    idx = np.arange(n_bins, dtype=float)
    return fmin * (2.0 ** (idx / float(bpo)))

def _harmonic_bins(b0, bpo, H=3):
    # CQT is log2 frequency; multiply by h -> + bpo*log2(h)
    offs = [0.0] + [np.log2(h) * bpo for h in range(2, H+1)]
    return [int(round(b0 + off)) for off in offs]

def estimate_bass_bin(mag, t0, *, bpo=12, fmin=librosa.note_to_hz('A0'), frames_ahead=3, lowpass_hz=220.0,
                      q_thresh=0.75, min_sep_bins=2, H=3, w=(1.0, 0.6, 0.4)):
    """
    Robust bass-bin estimator around onset frame t0 (causal average).
    - Uses median over frames [t0 .. t0+frames_ahead] to avoid pre-transition smear
    - Low-pass to <= lowpass_hz to focus on bass
    - Selects candidates above a robust quantile, enforces local-peak separation
    - Scores candidates by harmonicity (energy at 1x/2x/3x)
    - Fixes 'missing fundamental' by promoting +1 octave if needed
    """
    n_bins, n_frames = mag.shape
    t1 = min(n_frames, t0 + max(1, frames_ahead))
    S = np.median(mag[:, t0:t1], axis=1)  # robust, causal
    if not np.any(S):
        return None

    freqs = _cqt_center_freqs(n_bins, bpo, fmin)
    low_mask = freqs <= lowpass_hz
    S_low = S * low_mask.astype(float)

    # Robust threshold (quantile over nonzeros)
    nz = S_low[S_low > 0]
    if nz.size == 0:
        return None
    thr = np.quantile(nz, q_thresh)
    cand = np.where(S_low >= thr)[0]
    if cand.size == 0:
        return int(np.argmax(S_low))  # fallback

    # Keep local maxima & enforce min separation to avoid dense clusters
    keep = []
    for b in cand:
        l = max(0, b-1); r = min(n_bins-1, b+1)
        if S_low[b] >= S_low[l] and S_low[b] >= S_low[r]:
            if not keep or (b - keep[-1]) >= min_sep_bins:
                keep.append(b)
    if not keep:
        keep = cand.tolist()

    # Harmonicity score: sum of energies at 1x,2x,3x (with weights)
    def hscore(b0):
        bins = _harmonic_bins(b0, bpo, H=H)
        s = 0.0
        for k, bb in enumerate(bins):
            if 0 <= bb < n_bins:
                s += (w[k] if k < len(w) else 1.0) * S[min(bb, n_bins-1)]
        return s

    best = max(keep, key=hscore)

    # Missing-fundamental fix: if energy 1 octave above is much stronger, shift up
    one_oct = best + int(round(bpo))
    if one_oct < n_bins and S[best] < 0.5 * S[one_oct]:
        best = one_oct

    return int(best)

def bin_to_octave(bin_idx, *, bpo=12, fmin=librosa.note_to_hz('A0'), a4=440.0):
    """
    Map CQT bin -> musical octave using actual CQT config.
    Octave number uses MIDI convention: C4 in octave 4, etc.
    """
    if bin_idx is None:
        return None
    freq = fmin * (2.0 ** (bin_idx / float(bpo)))
    midi = 69.0 + 12.0 * np.log2(freq / float(a4))
    return int(np.floor(midi / 12.0) - 1)

def detect_true_bass_pc(mag_frame, floor_frac=0.1):
    """Find the semitone class of the lowest active bin in a magnitude spectrum."""
    thresh = mag_frame.max() * floor_frac
    active_bins = np.where(mag_frame >= thresh)[0]
    if active_bins.size == 0:
        return None
    # lowest active bin index modulo 12 gives the pitch class
    return int(active_bins.min().item() % 12)

def _bin_to_pc(bin_idx, *, bpo=12, fmin=librosa.note_to_hz('A0'), a4=440.0):
    """
    Map a CQT bin index to a pitch-class (0..11) via frequency -> MIDI -> PC.
    """
    freq = fmin * (2.0 ** (bin_idx / float(bpo)))
    midi = 69.0 + 12.0 * np.log2(freq / float(a4))
    return int(round(midi)) % 12

def detect_bass_pc_conf(mag, t0, *, bpo=12, fmin=librosa.note_to_hz('A0'), frames_ahead=2, lowpass_hz=220.0):
    """
    Causal, low-band bass PC + confidence.
    Returns (bass_pc, bass_rel) where bass_rel in [0,1] is bass bin energy
    relative to low-band max. If no clear bass, returns (None, 0.0).
    """
    n_bins, n_frames = mag.shape
    t1 = min(n_frames, t0 + max(1, frames_ahead))
    S = np.median(mag[:, t0:t1], axis=1)

    freqs = _cqt_center_freqs(n_bins, bpo, fmin)
    mask = (freqs <= lowpass_hz).astype(float)
    S_low = S * mask

    if not np.any(S_low):
        return (None, 0.0)

    b = int(np.argmax(S_low))
    bass_rel = float(S_low[b] / (S_low.max() + 1e-12))
    return (_bin_to_pc(b, bpo=bpo, fmin=fmin), bass_rel)

def chord_tone_pcs(root_pc, quality):
    """Return ordered chord tone pitch-classes for a given quality.

    The order is root, 3rd, 5th [, 7th] to map to inversion names.
    """
    q = (quality or "").lower()
    if "maj7" in q:
        iv = (0, 4, 7, 11)
    elif "dom7" in q or q == "7":
        iv = (0, 4, 7, 10)
    elif "m7b5" in q or "half" in q:
        iv = (0, 3, 6, 10)
    elif "min7" in q or (("m7" in q) and ("m7b5" not in q)):
        iv = (0, 3, 7, 10)
    elif "dim7" in q:
        iv = (0, 3, 6, 9)
    elif "aug" in q:
        iv = (0, 4, 8)
    elif "dim" in q:
        iv = (0, 3, 6)
    elif "min" in q:
        iv = (0, 3, 7)
    else:
        iv = (0, 4, 7)
    return [ (root_pc + i) % 12 for i in iv ]

def compute_inversion(root_pc, quality, pc_energies, mag, 
                      t0, *, bpo=12, fmin=librosa.note_to_hz('A0'),
                      min_bass_rel=0.60, bass_vs_root=0.85):
    """
    Smart inversion decision returning 'root'|'first'|'second'|'third'.

    Uses a causal low-band bass estimate and compares bass strength to the
    root; falls back to 'root' when ambiguous. Never returns 'slash'.
    """
    # 1) robust bass pc + confidence
    bass_pc, bass_rel = detect_bass_pc_conf(mag, t0, bpo=bpo, fmin=fmin)
    if bass_pc is None:
        return "root"

    tones = chord_tone_pcs(root_pc, quality)

    # 2) If bass isn't a chord tone, prefer root
    if bass_pc not in tones:
        return "root"

    # 3) Strength checks
    root_e = float(pc_energies[root_pc]) if pc_energies is not None else 1.0
    bass_e = float(pc_energies[bass_pc]) if pc_energies is not None else bass_rel

    if bass_rel < min_bass_rel:
        return "root"
    if pc_energies is not None and bass_e < bass_vs_root * root_e:
        return "root"

    # 4) Map bass tone to inversion
    pos = tones.index(bass_pc)
    if len(tones) == 3:
        return ["root", "first", "second"][pos] if pos < 3 else "root"
    else:
        return ["root", "first", "second", "third"][pos] if pos < 4 else "root"

def has_seventh_bic(x_chroma, root_pc, quality='maj7'):
    """
    Decide if the 7th is present using NNLS + BIC.
    x_chroma: (12,) normalized chroma for the onset (median over ~3 frames is ok)
    root_pc: 0..11 (C=0)
    quality: 'maj7' or 'dom7'
    Returns: (keep_seventh: bool, debug: dict)
    """
    # chord bins
    tri = [(root_pc + i) % 12 for i in (0, 4, 7)]         # major triad basis
    sev_int = 11 if quality == 'maj7' else 10
    sev = (root_pc + sev_int) % 12

    # design matrices (12 x k), columns are one-hots
    A3 = np.eye(12)[:, tri]                # k=3
    A4 = np.eye(12)[:, tri + [sev]]        # k=4

    # NNLS fits
    w3, _ = nnls(A3, x_chroma)
    r3 = x_chroma - A3 @ w3
    sse3 = float(np.dot(r3, r3))

    w4, _ = nnls(A4, x_chroma)
    r4 = x_chroma - A4 @ w4
    sse4 = float(np.dot(r4, r4))

    # BIC (N=12 features, k params = number of active tones)
    N = 12
    bic3 = N * np.log(sse3 / N + 1e-12) + 3 * np.log(N)
    bic4 = N * np.log(sse4 / N + 1e-12) + 4 * np.log(N)

    keep = (bic4 < bic3)  # prefer maj7/dom7 only if it wins after penalty
    dbg = dict(sse3=sse3, sse4=sse4, bic3=bic3, bic4=bic4, w3=w3, w4=w4)
    return keep, dbg

#* ─── YIN Pitch Detection ────────────────────────────────────────────────

def detect_pitch_yin_enhanced(frame, min_freq=50, max_freq=800, threshold=0.1, debug=False):
    """
    Enhanced YIN algorithm for pitch detection with better parameter tuning
    and low-frequency sensitivity. Based on the original YIN paper by 
    de Cheveigné & Kawahara (2002) with improvements for musical note detection.
    """
    # Use longer frame for better low frequency resolution
    if len(frame) < 4096:
        # Zero-pad to get better frequency resolution for low notes
        padded_frame = np.zeros(4096) 
        padded_frame[:len(frame)] = frame
        frame = padded_frame
    
    # Apply window and remove DC
    windowed = frame * np.hanning(len(frame))
    windowed = windowed - np.mean(windowed)
    
    # Calculate the range of periods to search
    min_period = int(SAMPLE_RATE / max_freq)
    max_period = int(SAMPLE_RATE / min_freq)
    max_period = min(max_period, len(windowed) // 2)
    
    if min_period >= max_period or max_period <= min_period + 10:
        return None
    
    # Step 1: Difference function (squared difference) and CMND using JIT
    diff_func = _yin_diff(windowed, min_period, max_period)
    cmnd_func = _yin_cmnd(diff_func)

    # Step 3: Absolute threshold - find first minimum below threshold
    adaptive_threshold = threshold
    best_period = None
    
    for tau in range(min_period, max_period):
        if cmnd_func[tau] < adaptive_threshold:
            # Look for local minimum around this point
            local_min_tau = tau
            local_min_val = cmnd_func[tau]
            
            # Search in a small window for the actual minimum
            search_start = max(min_period, tau - 3)
            search_end = min(max_period, tau + 4)
            
            for t in range(search_start, search_end):
                if cmnd_func[t] < local_min_val:
                    local_min_val = cmnd_func[t]
                    local_min_tau = t
            
            best_period = local_min_tau
            break
    
    # Step 4: If no period found with absolute threshold, use best local minimum
    # with bias towards lower frequencies (longer periods)
    if best_period is None:
        min_value = float('inf')
        for tau in range(min_period + 1, max_period - 1):
            # Check for local minimum
            if (cmnd_func[tau] < cmnd_func[tau - 1] and 
                cmnd_func[tau] < cmnd_func[tau + 1]):
                
                # Add bias towards lower frequencies for musical notes
                # Lower frequencies (longer periods) get a slight preference
                freq = SAMPLE_RATE / tau
                if freq < 200:  # Below G3
                    bias_factor = 0.9  # 10% preference for low frequencies
                elif freq < 100:  # Below C3  
                    bias_factor = 0.8  # 20% preference for very low frequencies
                else:
                    bias_factor = 1.0
                
                adjusted_value = cmnd_func[tau] * bias_factor
                
                if adjusted_value < min_value:
                    min_value = adjusted_value
                    best_period = tau
        
        # Only accept if the minimum is reasonable
        if best_period is None or cmnd_func[best_period] > 0.8:
            if debug:
                print(f"    YIN: No reliable period found (min_value={cmnd_func[best_period] if best_period else 'N/A'})")
            return None
    
    # Step 5: Parabolic interpolation for sub-sample accuracy
    if best_period and min_period < best_period < max_period - 1:
        # Parabolic interpolation around the minimum
        y1 = cmnd_func[best_period - 1]
        y2 = cmnd_func[best_period]
        y3 = cmnd_func[best_period + 1]
        
        # Fit parabola and find minimum
        a = (y1 - 2*y2 + y3) / 2
        b = (y3 - y1) / 2
        
        if a > 0:  # Parabola opens upward
            x_min = -b / (2 * a)
            if -0.5 <= x_min <= 0.5:  # Reasonable interpolation
                best_period_interp = best_period + x_min
            else:
                best_period_interp = best_period
        else:
            best_period_interp = best_period
    else:
        best_period_interp = best_period
    
    if best_period_interp is None or best_period_interp <= 0:
        return None
    
    # Convert period to frequency
    fundamental_freq = SAMPLE_RATE / best_period_interp
    
    if debug:
        print(f"    YIN: period={best_period_interp:.2f}, freq={fundamental_freq:.1f}Hz, "
              f"confidence={1-cmnd_func[int(best_period)]:.3f}")
    
    # Convert to MIDI
    if fundamental_freq > 0:
        midi_note = 69 + 12 * np.log2(fundamental_freq / 440.0)
        midi_note = int(round(midi_note))
        if 21 <= midi_note <= 108:
            return midi_note
    
    return None

# JIT-compiled difference function for YIN
@njit
def _yin_diff(windowed, min_period, max_period):
    N = windowed.shape[0]
    diff_func = np.zeros(max_period + 1)
    for tau in range(min_period, max_period + 1):
        s = 0.0
        for j in range(N - tau):
            diff = windowed[j] - windowed[j + tau]
            s += diff * diff
        diff_func[tau] = s
    return diff_func

# JIT-compiled cumulative mean normalized difference function
@njit
def _yin_cmnd(diff_func):
    N = diff_func.shape[0]
    cmnd = np.ones(N)
    cumsum = 0.0
    for tau in range(1, N):
        cumsum += diff_func[tau]
        if cumsum > 0.0:
            cmnd[tau] = diff_func[tau] * tau / cumsum
        else:
            cmnd[tau] = 1.0
    return cmnd

#* ─── Robust Pitch Detection with Octave Error Correction ─────────────────
def detect_pitch_robust(frame, cqt_mag, fft_mag, freqs, debug=False):
    """
    Robust pitch detection that combines multiple methods and corrects octave errors
    """
    candidates = []
    
    # Method 1: FFT-based detection
    fft_note, _, _ = detect_fundamental_from_fft(frame)
    if fft_note:
        candidates.append(('FFT', fft_note, 440.0 * 2**((fft_note - 69)/12)))
    
    # Method 2: CQT Simple method
    simple_note = detect_fundamental_simple(cqt_mag)
    if simple_note:
        candidates.append(('CQT_Simple', simple_note, 440.0 * 2**((simple_note - 69)/12)))
    
    # Method 3: HPS method
    hps_notes = pick_pitches_HPS(cqt_mag, max_voices=1)
    if hps_notes:
        candidates.append(('HPS', hps_notes[0], 440.0 * 2**((hps_notes[0] - 69)/12)))
      # Method 4: Autocorrelation-based pitch detection
    autocorr_note = detect_pitch_autocorrelation(frame)
    if autocorr_note:
        candidates.append(('Autocorr', autocorr_note, 440.0 * 2**((autocorr_note - 69)/12)))
    
    # Method 5: YIN enhanced algorithm - especially good for low frequencies
    yin_note = detect_pitch_yin_enhanced(frame, min_freq=50, max_freq=800, threshold=0.15, debug=debug)
    if yin_note:
        candidates.append(('YIN', yin_note, 440.0 * 2**((yin_note - 69)/12)))
    
    if not candidates:
        return None, "No detection"
    
    if debug:
        print(f"  Pitch candidates:")
        for method, note, freq in candidates:
            def midi_to_name_local(m):
                names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                return f"{names[m%12]}{(m//12)-1}"
            print(f"    {method}: {midi_to_name_local(note)} (MIDI {note}, {freq:.1f}Hz)")
    
    # Octave error correction: check for consensus at different octaves
    octave_groups = {}
    for method, note, freq in candidates:
        # Group by note class (C, C#, D, etc.) regardless of octave
        note_class = note % 12
        if note_class not in octave_groups:
            octave_groups[note_class] = []
        octave_groups[note_class].append((method, note, freq))
    
    # Find the note class with most votes
    best_note_class = max(octave_groups.keys(), key=lambda k: len(octave_groups[k]))
    octave_candidates = octave_groups[best_note_class]
    
    if debug:
        def midi_to_name_local(m):
            names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            return f"{names[m%12]}{(m//12)-1}"
        print(f"  Best note class: {['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][best_note_class]}")
        print(f"  Octave candidates:")
        for method, note, freq in octave_candidates:
            print(f"    {method}: {midi_to_name_local(note)} ({freq:.1f}Hz)")
    
    if len(octave_candidates) == 1:
        # Only one candidate, trust it
        method, final_note, final_freq = octave_candidates[0]
        return final_note, f"{method}"
      # Multiple candidates for the same note class - pick the most reasonable octave
    # Use harmonic analysis to determine the correct octave, with method weighting
    best_score = -1
    best_candidate = None
    
    for method, note, freq in octave_candidates:
        # Score based on harmonic strength in the spectrum
        harmonic_score = score_harmonic_fit(freq, fft_mag, freqs)
        
        # Apply method weighting based on frequency range and method reliability
        method_weight = 1.0
        if freq < 150:  # Low frequencies - YIN and Autocorr are generally better
            if method == 'YIN':
                method_weight = 1.3  # 30% bonus for YIN on low frequencies
            elif method == 'Autocorr':
                method_weight = 1.2  # 20% bonus for autocorrelation on low frequencies
            elif method == 'FFT':
                method_weight = 1.1  # Small bonus for FFT (good fundamental detection)
        elif freq < 300:  # Mid frequencies
            if method == 'YIN':
                method_weight = 1.1  # Small bonus for YIN
            elif method == 'HPS':
                method_weight = 1.1  # HPS works well in mid range
        # Higher frequencies - all methods weighted equally
        
        final_score = harmonic_score * method_weight
        
        if debug:
            def midi_to_name_local(m):
                names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                return f"{names[m%12]}{(m//12)-1}"
            print(f"    {midi_to_name_local(note)} harmonic score: {harmonic_score:.3f}, "
                  f"method weight: {method_weight:.2f}, final: {final_score:.3f}")
        
        if final_score > best_score:
            best_score = final_score
            best_candidate = (method, note, freq)
    
    if best_candidate:
        method, final_note, final_freq = best_candidate
        return final_note, f"{method} (octave-corrected)"
    
    # Fallback: use the median octave
    notes = [note for _, note, _ in octave_candidates]
    final_note = int(np.median(notes))
    return final_note, "Median octave"

def detect_pitch_autocorrelation(frame, min_freq=60, max_freq=800):
    """
    Enhanced autocorrelation-based pitch detection with better low frequency sensitivity
    """
    # Use longer frame for better low frequency resolution
    if len(frame) < 4096:
        # Zero-pad to get better frequency resolution for low notes
        padded_frame = np.zeros(4096)
        padded_frame[:len(frame)] = frame
        frame = padded_frame
    
    # Apply window and remove DC
    windowed = frame * np.hanning(len(frame))
    windowed = windowed - np.mean(windowed)
    
    # Autocorrelation
    autocorr = np.correlate(windowed, windowed, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    
    # Normalize
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]
    
    # Find the range of lags corresponding to our frequency range
    min_lag = int(SAMPLE_RATE / max_freq)
    max_lag = int(SAMPLE_RATE / min_freq)
    max_lag = min(max_lag, len(autocorr) - 1)
    
    if min_lag >= max_lag or max_lag <= min_lag + 10:
        return None
    
    # Find peaks in the autocorrelation function
    search_range = autocorr[min_lag:max_lag]
    if len(search_range) == 0:
        return None
    
    # Look for peaks above threshold, with bias towards low frequencies
    threshold = 0.25  # Slightly lower threshold for better sensitivity
    peaks = []
    
    for i in range(1, len(search_range) - 1):
        if (search_range[i] > search_range[i-1] and 
            search_range[i] > search_range[i+1] and 
            search_range[i] > threshold):
            actual_lag = min_lag + i
            freq = SAMPLE_RATE / actual_lag
            correlation = search_range[i]
            
            # Apply bias towards lower frequencies for musical note detection
            if freq < 150:  # Below about D#3
                bias_factor = 1.2  # 20% boost for low frequencies
            elif freq < 100:  # Below C3
                bias_factor = 1.3  # 30% boost for very low frequencies
            else:
                bias_factor = 1.0
            
            adjusted_correlation = correlation * bias_factor
            peaks.append((actual_lag, adjusted_correlation, freq, correlation))
    
    if not peaks:
        return None
    
    # Sort by adjusted correlation strength and take the best
    peaks.sort(key=lambda x: x[1], reverse=True)
    best_lag, adj_corr, best_freq, orig_corr = peaks[0]
    
    # Only accept if original correlation is reasonably strong
    if orig_corr < 0.2:
        return None
    
    # Convert to MIDI
    if best_freq > 0:
        midi_note = 69 + 12 * np.log2(best_freq / 440.0)
        midi_note = int(round(midi_note))
        if 21 <= midi_note <= 108:
            return midi_note
    
    return None

def score_harmonic_fit(fundamental_freq, fft_mag, freqs):
    """
    Score how well a fundamental frequency fits the harmonic content in the FFT
    Enhanced for low-frequency notes where the fundamental might be weak
    """
    score = 0
    fundamental_strength = 0
    harmonic_strength = 0
    
    # Check the fundamental first
    fund_bin = np.argmin(np.abs(freqs - fundamental_freq))
    if fund_bin < len(fft_mag):
        window_start = max(0, fund_bin - 2)
        window_end = min(len(fft_mag), fund_bin + 3)
        fundamental_strength = np.max(fft_mag[window_start:window_end])
    
    # Check harmonics - extend search for low frequencies
    harmonic_count = 0
    max_harmonics = 8 if fundamental_freq < 150 else 5  # More harmonics for low frequencies
    
    for h in range(2, max_harmonics + 1):
        harmonic_freq = fundamental_freq * h
        if harmonic_freq < SAMPLE_RATE / 2:
            # Find the closest bin
            bin_idx = np.argmin(np.abs(freqs - harmonic_freq))
            if bin_idx < len(fft_mag):
                # Wider window for low frequencies to account for frequency uncertainty
                window_width = 3 if fundamental_freq > 100 else 5
                window_start = max(0, bin_idx - window_width)
                window_end = min(len(fft_mag), bin_idx + window_width + 1)
                max_mag_in_window = np.max(fft_mag[window_start:window_end])
                
                # Lower threshold for low frequencies where harmonics might be more prominent
                threshold = 0.05 * np.max(fft_mag) if fundamental_freq < 150 else 0.1 * np.max(fft_mag)
                
                if max_mag_in_window > threshold:
                    # Weight lower harmonics more heavily, especially for low frequencies
                    if fundamental_freq < 100:  # Very low frequencies
                        weight = 1.5 / h  # Extra weight for low freq harmonics
                    else:
                        weight = 1.0 / h
                    
                    harmonic_strength += max_mag_in_window * weight
                    harmonic_count += 1
    
    # Enhanced scoring for low frequencies:
    # For low frequencies, harmonics are often stronger than the fundamental
    if fundamental_freq < 150:  # Below about D#3
        if harmonic_count >= 2:
            # Strong harmonic evidence for low frequencies - weight harmonics heavily
            score = fundamental_strength * 0.2 + harmonic_strength * 0.8
        else:
            # Weak harmonic evidence - still check fundamental
            score = fundamental_strength * 0.6 + harmonic_strength * 0.4
    else:
        # Standard scoring for higher frequencies
        if harmonic_count > 0:
            score = fundamental_strength * 0.4 + harmonic_strength * 0.6
        else:
            score = fundamental_strength
    
    # Bonus for having multiple harmonics (indicates a pitched sound)
    if harmonic_count >= 3:
        score *= 1.2  # 20% bonus for rich harmonic content
    
    return score

#* ─── Fundamental Frequency Detection ────────────────────────────────────────
def detect_fundamental_simple(cqt_mag, min_confidence=0.1, debug=False):
    """Simple fundamental detection - find the strongest peak that has harmonics, accounting for missing fundamental"""
    max_mag = np.max(cqt_mag)
    if max_mag < min_confidence:
        return None
    
    # Find peaks above threshold
    threshold = 0.2 * max_mag
    candidates = []
    
    if debug:
        print(f"  Debug: max_mag={max_mag:.3f}, threshold={threshold:.3f}")
        print(f"  Debug: Analyzing candidates...")
    
    # Check each possible fundamental frequency
    for i in range(len(cqt_mag)):
        midi_note = 21 + i
        freq = 440.0 * 2**((midi_note - 69)/12)
        
        # Score this as a potential fundamental
        fundamental_mag = cqt_mag[i] if cqt_mag[i] > threshold * 0.5 else 0  # Allow weaker fundamentals
        
        # Check for harmonics (2x, 3x, 4x, 5x)
        harmonic_score = fundamental_mag * 2.0  # Weight actual fundamental if present
        harmonic_count = 0
        harmonic_details = []
        
        for h in [2, 3, 4, 5]:
            harmonic_freq = freq * h
            # Find closest bin to harmonic
            harmonic_bin = np.argmin(np.abs(bin_freq - harmonic_freq))
            if harmonic_bin < len(cqt_mag):
                harmonic_mag = cqt_mag[harmonic_bin]
                if harmonic_mag > threshold * 0.3:  # Lower threshold for harmonics
                    # Weight harmonics by strength and inverse of harmonic number
                    weight = 1.0 / h
                    harmonic_score += harmonic_mag * weight
                    harmonic_count += 1
                    harmonic_details.append((h, harmonic_mag, weight))
        
        # Only consider if we have at least 2 harmonics (including potential fundamental)
        if harmonic_count >= 1 or fundamental_mag > 0:
            candidates.append((midi_note, harmonic_score, harmonic_count, fundamental_mag, harmonic_details))
            
            # Debug output for promising candidates
            if debug and (fundamental_mag > threshold * 0.3 or harmonic_score > threshold):
                def midi_to_name_local(m):
                    names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                    return f"{names[m%12]}{(m//12)-1}"
                
                print(f"    Candidate {midi_to_name_local(midi_note)} (MIDI {midi_note}): fund_mag={fundamental_mag:.3f}, score={harmonic_score:.3f}, harmonics={harmonic_count}")
    
    if not candidates:
        if debug:
            print("  Debug: No candidates found")
        return None
    
    # Sort by harmonic score and return the best
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_note = candidates[0][0]
    
    if debug:
        print(f"  Debug: Top 3 candidates:")
        for i, (note, score, count, fund_mag, _) in enumerate(candidates[:3]):
            def midi_to_name_local(m):
                names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                return f"{names[m%12]}{(m//12)-1}"
            print(f"    {i+1}. {midi_to_name_local(note)} (MIDI {note}): score={score:.3f}, fund_mag={fund_mag:.3f}")
    
    # Special check for missing fundamental: only apply if the detected fundamental is very weak
    # and there's strong evidence of harmonics without a clear fundamental
    strongest_peak_idx = np.argmax(cqt_mag)
    strongest_note = 21 + strongest_peak_idx
    
    if debug:
        def midi_to_name_local(m):
            names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            return f"{names[m%12]}{(m//12)-1}"
        print(f"  Debug: Strongest peak is {midi_to_name_local(strongest_note)} (MIDI {strongest_note}) with mag={cqt_mag[strongest_peak_idx]:.3f}")
    
    # Only consider missing fundamental if the best candidate has very weak fundamental support
    # and there's strong evidence of harmonics without a clear fundamental
    best_note_idx = best_note - 21
    if (0 <= best_note_idx < len(cqt_mag) and 
        cqt_mag[best_note_idx] < 0.3 * max_mag):  # Only if fundamental is quite weak
        
        if debug:
            print(f"  Debug: Best candidate has weak fundamental ({cqt_mag[best_note_idx]:.3f} < {0.3 * max_mag:.3f}), checking for missing fundamental...")
        
        # If strongest peak could be a 2nd harmonic, check the octave below
        potential_fundamental = strongest_note - 12  # One octave down
        if 21 <= potential_fundamental <= 108:
            fund_idx = potential_fundamental - 21
            if fund_idx >= 0 and fund_idx < len(cqt_mag):
                # Check if this lower note has good harmonic support
                fund_freq = 440.0 * 2**((potential_fundamental - 69)/12)
                harmonic_support = 0
                for h in [2, 3, 4]:
                    harmonic_freq = fund_freq * h
                    harmonic_bin = np.argmin(np.abs(bin_freq - harmonic_freq))
                    if harmonic_bin < len(cqt_mag):
                        harmonic_support += cqt_mag[harmonic_bin]
                
                if debug:
                    def midi_to_name_local(m):
                        names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                        return f"{names[m%12]}{(m//12)-1}"
                    print(f"  Debug: Checking potential missing fundamental {midi_to_name_local(potential_fundamental)} (MIDI {potential_fundamental})")
                    print(f"  Debug: harmonic_support={harmonic_support:.3f}, fund_mag={cqt_mag[fund_idx]:.3f}")
                    print(f"  Debug: Conditions: harm_sup > 4*fund? {harmonic_support > 4 * cqt_mag[fund_idx]}, harm_sup > 0.7*max? {harmonic_support > max_mag * 0.7}, fund < 0.2*max? {cqt_mag[fund_idx] < 0.2 * max_mag}")
                
                # Only use sub-octave if harmonics are MUCH stronger and fundamental is truly missing
                if (harmonic_support > 4 * cqt_mag[fund_idx] and 
                    harmonic_support > max_mag * 0.7 and
                    cqt_mag[fund_idx] < 0.2 * max_mag):
                    if debug:
                        print(f"  Debug: Using missing fundamental correction!")
                    return potential_fundamental
    
    if debug:
        def midi_to_name_local(m):
            names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            return f"{names[m%12]}{(m//12)-1}"
        print(f"  Debug: Final result: {midi_to_name_local(best_note)} (MIDI {best_note})")
    
    return best_note

def detect_fundamental_from_fft(frame, min_freq=40, max_freq=600):
    """Detect fundamental frequency directly from FFT magnitude spectrum with harmonic analysis"""
    # Compute FFT
    fft_mag = compute_magnitude(frame)
    
    # Convert bin indices to frequencies
    freqs = np.fft.rfftfreq(FFT_SIZE, 1/SAMPLE_RATE)
    
    # Find the range of bins corresponding to our frequency range
    min_bin = int(min_freq * FFT_SIZE / SAMPLE_RATE)
    max_bin = int(max_freq * FFT_SIZE / SAMPLE_RATE)
    min_bin = max(1, min_bin)  # Avoid DC component
    max_bin = min(len(fft_mag)-1, max_bin)
    
    # Find peaks in the magnitude spectrum
    threshold = 0.1 * np.max(fft_mag)
    peaks = []
    for i in range(min_bin, max_bin):
        if (fft_mag[i] > fft_mag[i-1] and 
            fft_mag[i] > fft_mag[i+1] and 
            fft_mag[i] > threshold):
            peaks.append((i, fft_mag[i], freqs[i]))
    
    if not peaks:
        return None, freqs, fft_mag
    
    # For each peak, calculate a "fundamental score" based on harmonic content
    candidates = []
    for peak_bin, peak_mag, peak_freq in peaks:
        # Score based on fundamental strength and harmonic support
        fundamental_score = peak_mag
        
        # Check for harmonics (2x, 3x, 4x, 5x)
        harmonic_support = 0
        for h in [2, 3, 4, 5]:
            harmonic_freq = peak_freq * h
            if harmonic_freq < SAMPLE_RATE / 2:
                # Find the closest bin to this harmonic
                harmonic_bin = int(harmonic_freq * FFT_SIZE / SAMPLE_RATE)
                if harmonic_bin < len(fft_mag):
                    # Look for peak around this bin (±2 bins)
                    search_range = range(max(0, harmonic_bin-2), min(len(fft_mag), harmonic_bin+3))
                    max_harmonic_mag = max(fft_mag[search_range])
                    # Weight harmonics less as they get higher
                    harmonic_support += max_harmonic_mag / (h * h)
        
        # Check if this might be a harmonic of a lower frequency
        subharmonic_penalty = 0
        for sub_h in [2, 3, 4]:
            sub_freq = peak_freq / sub_h
            if sub_freq >= min_freq:
                sub_bin = int(sub_freq * FFT_SIZE / SAMPLE_RATE)
                if 0 <= sub_bin < len(fft_mag):
                    # If there's a strong peak at half/third/quarter frequency, penalize this peak
                    search_range = range(max(0, sub_bin-2), min(len(fft_mag), sub_bin+3))
                    if search_range:
                        max_sub_mag = max(fft_mag[search_range])
                        if max_sub_mag > 0.5 * peak_mag:  # If subharmonic is strong
                            subharmonic_penalty += max_sub_mag * sub_h
        
        # Final score: fundamental + harmonic support - subharmonic penalty
        total_score = fundamental_score + harmonic_support * 0.3 - subharmonic_penalty * 0.5
        candidates.append((peak_freq, total_score, peak_mag))
    
    if not candidates:
        return None, freqs, fft_mag
    
    # Sort by total score and take the best
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_freq = candidates[0][0]
    
    # Convert frequency to MIDI note
    if best_freq > 0:
        midi_note = 69 + 12 * np.log2(best_freq / 440.0)
        midi_note = int(round(midi_note))
        if 21 <= midi_note <= 108:
            return midi_note, freqs, fft_mag
    
    return None, freqs, fft_mag

#* ─── Ringing note cancellation ────────────────────────────────────────
class RingTrack:
    __slots__ = ("midi", "f0", "harm_gains", "last_time")
    def __init__(self, midi, gains_h, t_now):
        self.midi = int(midi)
        self.f0   = 440.0 * 2**((self.midi - 69)/12)
        self.harm_gains = np.array(gains_h, dtype=np.float32)  # length H (h=1..H)
        self.last_time  = float(t_now)

ACTIVE_TRACKS = []  # global list of RingTrack
HARM_COUNT = 8
FUND_PROTECT_BW_BINS = 2  # don't cancel around f0 when there is a fresh onset

def harmonic_atom(freqs, f, sigma_bins=1.5, bw=3):
    k = np.argmin(np.abs(freqs - f))
    lo = max(0, k - bw); hi = min(len(freqs), k + bw + 1)
    local = np.arange(lo, hi) - k
    atom = np.zeros_like(freqs, dtype=np.float32)
    g = np.exp(-0.5*(local/sigma_bins)**2)
    atom[lo:hi] = g / (np.linalg.norm(g)+1e-12)
    return atom

def prev_template_matrix(freqs, tracks, t_now, tau_fund=0.35, tau_harm=0.22, protect_bins=None):
    """
    Build dictionary D_prev for all harmonics of active tracks at time t_now.
    Columns: each is a single harmonic atom with its current decayed gain cap.
    Returns D_prev (B,K), caps (K,), meta list [(track_idx, h), ...]
    """
    cols, caps, meta = [], [], []
    for ti, tr in enumerate(tracks):
        dt = max(0.0, t_now - tr.last_time)
        for h in range(1, HARM_COUNT+1):
            fh = tr.f0 * h
            if fh >= SAMPLE_RATE/2: break
            atom = harmonic_atom(freqs, fh)
            # decay: fundamentals ring longer than harmonics
            tau = tau_fund if h == 1 else tau_harm
            cap = tr.harm_gains[h-1] * np.exp(-dt / tau)
            # optional: protect bins around any "fresh onset" region (fundamental only)
            if protect_bins is not None and h == 1:
                k = np.argmin(np.abs(freqs - fh))
                lo = max(0, k - FUND_PROTECT_BW_BINS); hi = min(len(freqs), k + FUND_PROTECT_BW_BINS + 1)
                atom[lo:hi] = 0.0  # do not cancel fundamentals near fresh onset
            if cap > 1e-4 and np.any(atom):
                cols.append(atom); caps.append(cap); meta.append((ti, h))
    if not cols:
        return None, None, None
    D_prev = np.stack(cols, axis=1)
    return D_prev, np.array(caps, dtype=np.float32), meta

def nnls_capped(D, x, caps, iters=6):
    """Multiplicative NNLS with per-coefficient upper bounds (caps)."""
    if D is None: 
        return None
    a = np.minimum(np.maximum(D.T @ x, 0.0), caps.copy())
    Dt = D.T
    for _ in range(iters):
        num = Dt @ x
        den = Dt @ (D @ a) + 1e-12
        a *= num / den
        a = np.minimum(a, caps)
    return a

def cancel_ringing(mag_window, freqs, onset_midi_seeds=None):
    """
    Predict-and-subtract previous notes. Returns residual spectrum and
    the per-track updated harmonic gains (to store back).
    """
    if not ACTIVE_TRACKS:
        return mag_window, {}
    # protect fundamentals near fresh seeds (don’t suppress a re-strike)
    protect_bins = None
    if onset_midi_seeds:
        protect_bins = set([int(m) for m in onset_midi_seeds])

    # build prev dictionary
    D_prev, caps, meta = prev_template_matrix(freqs, ACTIVE_TRACKS, t_now=0.0, protect_bins=protect_bins)
    if D_prev is None:
        return mag_window, {}

    # fit previous-only contribution under caps
    a = nnls_capped(D_prev, mag_window, caps, iters=6)
    recon_prev = D_prev @ a
    resid = np.maximum(0.0, mag_window - recon_prev)

    # gather updated per-track harmonic gains
    updated = {}
    if a is not None:
        # sum contributions per (track,h)
        for coeff, (ti,h) in zip(a, meta):
            updated.setdefault(ti, np.zeros(HARM_COUNT, dtype=np.float32))
            updated[ti][h-1] += float(coeff)

    return resid, updated

#* ─── Offset Estimation ────────────────────────────────────────────────────────
def _event_energy_series(chroma, pcs, start_f, end_f):
    """Energy track for this note/chord = max chroma of its pitch classes."""
    pcs = list({p % 12 for p in pcs})
    return np.max(chroma[pcs, start_f:end_f], axis=0)

def estimate_offsets_from_chroma(
    onsets_frames,         # e.g. np.array of onset frame indices
    event_midis,           # list same length as onsets; each item int or list[int]
    chroma,                # shape (12, T) float
    lookahead_frames=2048//512 * 128,  # search up to ~128 beats (tweak)
    decay_ratio=0.20,                # end when energy < decay_ratio * local_peak
    abs_floor=0.06,                  # and also below an absolute floor
    hysteresis_L=3,                  # require L consecutive low frames
    guard_next_onset=True            # don’t cross strongly into next onset
):
    """
    Returns list of (onset_f, offset_f) per event (offset_f is exclusive).
    Works for single notes or chords (we track max energy across chord tones).
    """
    T = chroma.shape[1]
    onsets = np.asarray(onsets_frames, dtype=int)
    results = []

    for i, f0 in enumerate(onsets):
        pcs = event_midis[i]
        if not isinstance(pcs, (list, tuple, np.ndarray)):
            pcs = [int(pcs)]
        pcs = [int(m) % 12 for m in pcs]

        # define search window
        f1_limit = min(T, f0 + lookahead_frames)
        if guard_next_onset and i + 1 < len(onsets):
            f1_limit = min(f1_limit, onsets[i + 1])

        if f0 >= f1_limit - 1:
            results.append((f0, f0 + 1))
            continue

        e = _event_energy_series(chroma, pcs, f0, f1_limit)  # length W
        if not np.any(e):
            results.append((f0, min(f0 + 1, T)))
            continue

        # local peak over a short growth window to be robust at onset
        grow_win = min(6, e.size)  # ~ few frames
        local_peak = float(np.max(e[:grow_win]))
        thr = max(decay_ratio * local_peak, abs_floor)

        low_run = 0
        off_rel = e.size  # default to end of window
        for t_rel in range(grow_win, e.size):
            if e[t_rel] < thr:
                low_run += 1
                if low_run >= hysteresis_L:
                    off_rel = t_rel - hysteresis_L + 1
                    break
            else:
                # reset if energy resurges (legato/sustain)
                low_run = 0

        onset_f = int(f0)
        offset_f = int(min(f0 + off_rel, f1_limit))
        # never zero duration
        if offset_f <= onset_f:
            offset_f = min(onset_f + 1, T)
        results.append((onset_f, offset_f))

    return results

#* ─── Main Analysis Function ──────────────────────────────────────────────────
def detect_single_note_frame(frame, debug=False):
    def midi_to_name(m):
        names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        return f"{names[m%12]}{(m//12)-1}"

    # Method 1: FFT-based detection
    fft_note, freqs, fft_mag = detect_fundamental_from_fft(frame)
    if debug and fft_note:
        print(f"  FFT method: {midi_to_name(fft_note)} (MIDI {fft_note})")
    
    # Method 2: CQT-based detection
    cqt_mag = compute_cqt(frame)
    
    # Method 3: Simple CQT method
    simple_note = detect_fundamental_simple(cqt_mag)
    if debug and simple_note:
        print(f"  Simple method: {midi_to_name(simple_note)}")
    
    # Method 4: HPS method
    hps_notes = pick_pitches_HPS(cqt_mag, max_voices=1)
    if debug and hps_notes:
        print(f"  HPS method: {midi_to_name(hps_notes[0])}")
    
    # Method 5: Robust detection with octave correction
    robust_note, robust_method = detect_pitch_robust(frame, cqt_mag, fft_mag, freqs, debug)
    if debug and robust_note:
        print(f"  Robust method: {midi_to_name(robust_note)} (via {robust_method})")
    
    # Choose the best method (prefer robust method if available)
    if robust_note:
        final_note = robust_note
        method_used = f"Robust ({robust_method})"
        confidence = 0.9
    elif simple_note and hps_notes and simple_note == hps_notes[0]:
        # Both CQT methods agree
        final_note = simple_note
        method_used = "CQT (consensus)"
        confidence = 0.8
    elif simple_note:
        # Use simple CQT method as primary
        final_note = simple_note
        method_used = "CQT (simple)"
        confidence = 0.7
    elif hps_notes:
        # Fall back to HPS
        final_note = hps_notes[0]
        method_used = "CQT (HPS)"
        confidence = 0.6
    elif fft_note:
        # FFT as last resort
        final_note = fft_note
        method_used = "FFT"
        confidence = 0.5
    else:
        final_note = None
        method_used = "None"
        confidence = 0.0
    
    # Add note information if detected
    if final_note:
        note_info = {
            "time_seconds": 0,
            "frame_index": 0,
            "midi_note": int(final_note),
            "note_name": midi_to_name(final_note),
            "frequency_hz": round(440.0 * 2**((final_note - 69)/12), 2),
            "method": method_used,
            "confidence": confidence
        }
    else:
        note_info = None
        
    return note_info

def detect_chord_multiframe(chroma, mag, frame_idx, num_frames=3, debug=False):
    """
    Multi-frame chord detection for improved accuracy.
    
    Args:
        chroma: Chroma features (12, T)
        mag: CQT magnitude (bins, T) 
        frame_idx: Starting frame index
        num_frames: Number of frames to analyze (default 3)
        debug: Enable debug logging
    
    Returns:
        Chord detection result or None
    """
    # Ensure we don't go beyond available frames
    end_frame = min(frame_idx + num_frames, chroma.shape[1])
    actual_frames = end_frame - frame_idx
    
    if actual_frames < 1:
        return None
    
    # Average chroma across multiple frames for stability
    c_frames = chroma[:, frame_idx:end_frame]
    c_frame = np.mean(c_frames, axis=1)  # Average across time dimension
    
    if debug:
        print(f"[MultiFrame] Analyzing frames {frame_idx}-{end_frame-1} ({actual_frames} frames)")
        print(f"[MultiFrame] Chroma energy distribution: {np.round(c_frame, 3)}")

    score = 0

    # 2) Chroma-peak ratio check
    sorted_bins = np.sort(c_frame)[::-1]
    if sorted_bins[1] >= 0.5 * sorted_bins[0]:
        score += 1

    # 3) Template vs note score
    note_score = NOTE_TEMPLATES.dot(c_frame).max()
    chord_scores = CH_TEMPLATES.dot(c_frame)
    best_chord_score = chord_scores.max()
    if best_chord_score > note_score + 0.1:
        score += 1

    if debug:
        print(f"[MultiFrame] frame={frame_idx}, pts={score}/2, "
              f"ratio={sorted_bins[1]/sorted_bins[0]:.2f}, "
              f"chord_score={best_chord_score:.2f} vs note_score={note_score:.2f}")

    if score <= 1:
        return None

    # Chord result
    ci = int(np.argmax(chord_scores).item())
    chord_label = CH_LABELS[ci]
    root_pc = ROOTS.index(chord_label.split(':')[0])
    
    # Post-filter 7th chord extensions with logging
    quality = chord_label.split(':')[1]
    if quality in ('maj7', 'dom7'):
        keep7, dbg7 = has_seventh_bic(c_frame, root_pc, quality)
        if debug:
            print(f"[7th BIC] {quality}: keep={keep7}  bic3={dbg7['bic3']:.2f}  bic4={dbg7['bic4']:.2f}")
        if not keep7:
            original_label = chord_label
            chord_label = f"{ROOTS[root_pc]}:maj"
            ci = CH_LABELS.index(chord_label)
            # use the same chroma you judged on (c_med) for a consistent score
            best_chord_score = CH_TEMPLATES[ci].dot(c_frame)
            if debug:
                print(f"           Downgrading {original_label} → {chord_label}")
    elif quality in ('min7', 'dim7'):
        keep7, dbg7 = has_seventh_bic(c_frame, root_pc, quality)
        if debug:
            print(f"[7th BIC] {quality}: keep={keep7}  bic3={dbg7['bic3']:.2f}  bic4={dbg7['bic4']:.2f}")
        if not keep7:
            original_label = chord_label
            chord_label = f"{ROOTS[root_pc]}:min"
            ci = CH_LABELS.index(chord_label)
            # use the same chroma you judged on (c_med) for a consistent score
            best_chord_score = CH_TEMPLATES[ci].dot(c_frame)
            if debug:
                print(f"           Downgrading {original_label} → {chord_label}")
                
    # Use the middle frame for bass/inversion detection (most stable)
    middle_frame = frame_idx + actual_frames // 2

    # pc_energies is the chroma vector (pitch-class energies) for the frame
    pc_energies = c_frame  # already averaged across frames above

    # Use the smarter inversion computation which uses causal low-band bass
    try:
        inv = compute_inversion(
            root_pc=root_pc,
            quality=quality,
            pc_energies=pc_energies,
            mag=mag,
            t0=middle_frame,
            bpo=12,
            fmin=librosa.note_to_hz('A0')
        )
    except Exception:
        # Fallback to simple bass estimate if compute_inversion fails
        bass_pc = detect_true_bass_pc(mag[:, middle_frame])
        bass_bin = estimate_bass_bin(mag, middle_frame)
        octave  = bin_to_octave(bass_bin)
        if bass_pc is None or bass_pc == root_pc:
            inv = 'root'
        elif bass_pc in {(root_pc+3)%12, (root_pc+4)%12}:
            inv = 'first'
        elif bass_pc == (root_pc+7)%12:
            inv = 'second'
        else:
            inv = 'root'

    # compute octave for reporting (keep previous logic)
    bass_bin = estimate_bass_bin(mag, middle_frame)
    octave  = bin_to_octave(bass_bin)

    if debug:
        print(f"  ➤ DETECTED CHORD: {chord_label}, octave={octave}, inversion={inv}, confidence={best_chord_score:.3f}")
    
    return {
        "type": "chord",
        "label": chord_label,  
        "chord_quality": chord_label.split(':')[1],  # E.g., "maj", "min"
        "inversion": inv,
        "octave": octave,
        "confidence": best_chord_score,
        "note_score": note_score,
        "frames_analyzed": actual_frames
    }

#* ─── OPTIMIZED Analysis Pipeline (compute STFT/CQT once) ────────────────────
def analyze_audio_optimized(wav_path_or_array, debug=False, use_ml_rhythm=True):
    """
    Optimized analysis: compute STFT and CQT once, reuse everywhere.
    ~4x faster than standard pipeline with identical accuracy.
    
    Args:
        use_ml_rhythm: Use ML-based rhythm quantization (default: True)
    """
    try:
        # 1) Load and prepare audio
        if isinstance(wav_path_or_array, str):
            audio = read_wav(wav_path_or_array)
        else:
            audio = wav_path_or_array
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
    except Exception as e:
        return {"error": f"Failed to read audio: {str(e)}"}
    
    # 2) COMPUTE STFT ONCE - reuse for onset detection and pitch analysis
    # Note: audio is already filtered by read_wav (spectral gate + HPF)
    stft_data, magnitude, phase, freqs = compute_stft_once(audio)
    
    # 3) Onset detection from pre-computed magnitude (no additional filtering needed)
    flux = compute_flux_from_magnitude(magnitude)
    flux = normalize(flux)
    onsets = find_onsets(flux)
    
    # 4) COMPUTE CQT ONCE - reuse for chord/pitch detection (on filtered audio)
    if USE_GPU:
        C_full = gpu_cqt(audio, sr=SAMPLE_RATE, n_bins=CQT_BINS,
                         bins_per_octave=12, fmin=librosa.note_to_hz('A0'),
                         hop_length=HOP_SIZE)
    else:
        C_full = np.abs(librosa.cqt(
            y=audio, sr=SAMPLE_RATE, hop_length=HOP_SIZE,
            n_bins=CQT_BINS, bins_per_octave=12,
            fmin=librosa.note_to_hz('A0')
        ))
    
    # 5) COMPUTE CHROMA - use extract_chroma to match regular pipeline exactly
    chroma = extract_chroma(audio, SAMPLE_RATE, hop_length=HOP_SIZE)
    
    # 6) Estimate offsets from chroma
    try:
        event_midis = []
        for f in onsets:
            if 0 <= f < chroma.shape[1]:
                pc = int(np.argmax(chroma[:, f]))
                event_midis.append(pc)
            else:
                event_midis.append(0)
        offsets_frames = estimate_offsets_from_chroma(onsets, event_midis, chroma)
    except Exception:
        offsets_frames = [(f, f+1) for f in onsets]
    
    # Results structure
    results = {
        "onsets": [],
        "notes": [],
        "chords": [],
        "analysis_summary": {
            "total_onsets": len(onsets),
            "duration_seconds": float(len(audio) / SAMPLE_RATE),
            "sample_rate": int(SAMPLE_RATE)
        }
    }
    
    # 7) Process each onset - NO NEW FFT/CQT COMPUTATIONS
    for i, onset_frame in enumerate(onsets):
        time_seconds = onset_frame * HOP_SIZE / SAMPLE_RATE
        
        if debug:
            print(f"\n{'='*60}")
            print(f"ONSET #{i+1} at {time_seconds:.3f}s (frame {onset_frame})")
            print(f"{'='*60}")
        
        # Get offset/duration from chroma analysis
        try:
            oframe = int(offsets_frames[i][1])
            osec = round(oframe * HOP_SIZE / SAMPLE_RATE, 3)
            dur = round(osec - time_seconds, 3)
        except Exception:
            oframe = int(onset_frame + 1)
            osec = round((onset_frame + 1) * HOP_SIZE / SAMPLE_RATE, 3)
            dur = round(osec - time_seconds, 3)
        
        onset_info = {
            "time_seconds": round(time_seconds, 3),
            "frame_index": int(onset_frame),
            "offset_frame": oframe,
            "offset_seconds": osec,
            "duration_seconds": dur
        }
        results["onsets"].append(onset_info)
        
        # Extract onset-centered magnitude from pre-computed STFT
        # Also collect individual frames for temporal filtering (HMM-like approach)
        if 1 <= onset_frame < magnitude.shape[1] - 1:
            mag_window = np.mean(magnitude[:, onset_frame-1:onset_frame+2], axis=1)
            # Individual frames for temporal filtering
            mag_frames_raw = [magnitude[:, onset_frame-1], magnitude[:, onset_frame], magnitude[:, onset_frame+1]]
        else:
            mag_window = magnitude[:, min(onset_frame, magnitude.shape[1]-1)]
            mag_frames_raw = [mag_window]  # Single frame fallback
        
        # Ringing cancellation + BIC voice estimation with temporal filtering
        resid, _ = cancel_ringing(mag_window, freqs)
        # Apply ringing cancellation to each frame for temporal filtering
        mag_frames_resid = [cancel_ringing(mf, freqs)[0] for mf in mag_frames_raw]
        bic_est = estimate_voices_bic(resid, max_K=3, H=8, debug=debug, mag_frames=mag_frames_resid)
        
        K = bic_est['K']
        midi_set = bic_est['midis']
        salience_info = bic_est.get('salience_info', {})
        
        # For single notes, cross-validate with CQT peaks
        # For chords (K >= 2), trust BIC - CQT validation can incorrectly reject chord tones
        cqt_idx = min(onset_frame + 1, C_full.shape[1] - 1)
        if K == 1 and midi_set and cqt_idx < C_full.shape[1]:
            cqt_frame = C_full[:, cqt_idx]
            midi_set = validate_midi_with_cqt(midi_set, cqt_frame, tolerance_semitones=1, debug=debug)
            K = len(midi_set)  # Update K after validation
        
        is_chord_final = (K >= 2)
        
        # If NOT a chord but multiple notes detected, pick highest salience for single note
        if not is_chord_final and K > 1 and salience_info:
            midi_set_scored = [(m, salience_info.get(m, (0, False))[0]) for m in midi_set]
            midi_set_scored.sort(key=lambda x: x[1], reverse=True)
            best_midi = midi_set_scored[0][0]
            if debug:
                print(f"  [Single Note Selection] Picking highest salience: {['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][best_midi % 12] + str(best_midi // 12 - 1)}")
            midi_set = [best_midi]
            K = 1
        
        if debug and midi_set:
            final_names = [['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][m % 12] + str(m // 12 - 1) for m in midi_set]
            print(f"  [FINAL] Detected notes: {final_names}")
        
        if is_chord_final:
            # Chord detection using pre-computed chroma and C_full
            res = detect_chord_multiframe(chroma, C_full, onset_frame, num_frames=1, debug=debug)
            if res:
                # Add the actual MIDI notes from BIC analysis
                # This gives us the real notes played, not just a chord template
                res["midi_notes"] = [int(m) for m in midi_set] if midi_set else []
                res.update({
                    "time_seconds": round(time_seconds, 3),
                    "frame_index": int(onset_frame),
                    "offset_seconds": osec,
                    "duration_seconds": dur
                })
                results["chords"].append(res)
            else:
                # Chord detection failed, treat as single note - use MIDI from BIC
                m = midi_set[0] if midi_set else None
                if m is not None:
                    results["notes"].append({
                        "time_seconds": round(time_seconds, 3),
                        "frame_index": int(onset_frame),
                        "midi_note": int(m),
                        "note_name": note_to_name(int(m)),
                        "frequency_hz": round(_midi_to_hz(int(m)), 2),
                        "method": "HarmonicMixture(BIC)",
                        "confidence": 0.9,
                        "offset_seconds": osec,
                        "duration_seconds": dur
                    })
                # Note: If m is None here, we skip the note (no frame available for detect_single_note_frame)
        else:
            # Single note from BIC
            m = midi_set[0] if midi_set else None
            if m is not None:
                results["notes"].append({
                    "time_seconds": round(time_seconds, 3),
                    "frame_index": int(onset_frame),
                    "midi_note": int(m),
                    "note_name": note_to_name(int(m)),
                    "frequency_hz": round(_midi_to_hz(int(m)), 2),
                    "method": "BIC",
                    "confidence": 0.9,
                    "offset_seconds": osec,
                    "duration_seconds": dur
                })
            # Note: If m is None here, we skip the note (no frame available for detect_single_note_frame)
    
    # Filter out notes/chords that are too short (0.05 seconds or less)
    results["notes"] = [n for n in results["notes"] if n.get("duration_seconds", 0) > 0.05]
    results["chords"] = [c for c in results["chords"] if c.get("duration_seconds", 0) > 0.05]
    
    # Deduplicate chords at nearly the same time with same label
    TIME_TOLERANCE = 0.02  # 20ms - tight enough to only dedupe true duplicates
    dedupe_chords = []
    seen_chords = set()
    for c in results["chords"]:
        key = (round(c["time_seconds"] / TIME_TOLERANCE), c.get("label", ""))
        if key not in seen_chords:
            seen_chords.add(key)
            dedupe_chords.append(c)
    results["chords"] = dedupe_chords
    
    # Deduplicate notes at nearly the same time with same MIDI value
    dedupe_notes = []
    seen_notes = set()
    for n in results["notes"]:
        key = (round(n["time_seconds"] / TIME_TOLERANCE), n["midi_note"])
        if key not in seen_notes:
            seen_notes.add(key)
            dedupe_notes.append(n)
    results["notes"] = dedupe_notes
    
    # Also filter out notes that are already part of chords at the same time
    notes_in_chords = set()
    for c in results["chords"]:
        chord_time_key = round(c["time_seconds"] / TIME_TOLERANCE)
        for midi in c.get("midi_notes", []):
            notes_in_chords.add((chord_time_key, midi))
    
    results["notes"] = [
        n for n in results["notes"]
        if (round(n["time_seconds"] / TIME_TOLERANCE), n["midi_note"]) not in notes_in_chords
    ]
    
    # Detect tempo from onset times
    onset_times = [o["time_seconds"] for o in results["onsets"]]
    tempo_info = detect_tempo_from_onsets(onset_times)
    initial_bpm = tempo_info['bpm']
    
    # Refine tempo by testing candidates based on quantization error
    # This helps when the histogram picks up subdivisions instead of the true beat
    all_events = results["notes"] + results["chords"]
    refined_tempo = refine_tempo_by_quantization(all_events, initial_bpm)
    detected_bpm = refined_tempo['bpm']
    
    # Add tempo info to results
    results["analysis_summary"]["detected_bpm"] = detected_bpm
    results["analysis_summary"]["initial_bpm"] = initial_bpm
    results["analysis_summary"]["tempo_confidence"] = refined_tempo['confidence']
    results["analysis_summary"]["beat_interval"] = refined_tempo['beat_interval']
    results["analysis_summary"]["tempo_refinement_factor"] = refined_tempo.get('refinement_factor', 1.0)
    
    # RE-QUANTIZE rhythms using ML or IOI-based approach for better accuracy
    method_name = "ML" if use_ml_rhythm else "IOI"
    print(f"\n[Rhythm] Re-quantizing with {method_name} approach at {detected_bpm} BPM...")
    results["notes"], results["chords"] = quantize_rhythm_sequence(
        results["notes"], results["chords"], detected_bpm, debug=debug, use_ml=use_ml_rhythm
    )
    
    # Detect triplets (must be after regular note values are assigned)
    # Sort by time for triplet detection
    results["notes"] = sorted(results["notes"], key=lambda x: x.get("time_seconds", 0))
    results["chords"] = sorted(results["chords"], key=lambda x: x.get("time_seconds", 0))
    
    # Apply triplet detection (modifies notes/chords in place) - use detected BPM
    detect_triplets(results["notes"], bpm=detected_bpm, tolerance=0.20)
    detect_triplets_in_chords(results["chords"], bpm=detected_bpm, tolerance=0.20)
    strip_triplets_from_grace_notes(results["notes"])
    
    # Update summary
    results["analysis_summary"].update({
        "total_notes": len(results["notes"]),
        "total_chords": len(results["chords"])
    })
    
    return results


#* ─── Independent Two-Hands Analysis ─────────────────────────────────────────
def analyze_audio_independent_hands(wav_path_or_array, debug=False, split_midi=60, use_ml_rhythm=True):
    """
    Analyze audio with INDEPENDENT onset detection for bass and treble hands.
    
    Args:
        use_ml_rhythm: Use ML-based rhythm quantization (default: True)
    
    This enables detecting rhythmically independent parts, such as:
    - A held bass chord while treble plays a melody
    - Different rhythmic patterns in left and right hands
    - Sustained bass notes with staccato treble notes
    
    The process:
    1. Load and preprocess audio (noise reduction, etc.)
    2. Split into bass and treble frequency bands
    3. Detect onsets INDEPENDENTLY in each band
    4. Analyze each band's onsets using filtered audio
    5. Merge results with proper hand labeling
    
    Args:
        wav_path_or_array: Audio file path or numpy array
        debug: Enable debug output
        split_midi: MIDI note to split at (default 60 = middle C)
    
    Returns:
        Results with independently detected bass and treble notes/chords
    """
    print(f"\n{'='*70}")
    print("🎹 INDEPENDENT TWO-HANDS ANALYSIS")
    print(f"   Split point: MIDI {split_midi} ({440.0 * 2**((split_midi - 69) / 12):.1f} Hz)")
    print(f"   Bass and treble will have INDEPENDENT rhythm detection")
    print(f"{'='*70}\n")
    
    try:
        # 1) Load and prepare audio
        if isinstance(wav_path_or_array, str):
            audio = read_wav(wav_path_or_array)
        else:
            audio = wav_path_or_array.copy()
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
    except Exception as e:
        return {"error": f"Failed to read audio: {str(e)}"}
    
    # 2) Split audio into bass and treble frequency bands
    split_freq = 440.0 * 2**((split_midi - 69) / 12)
    
    # Design crossover filters (4th order Linkwitz-Riley style for flat sum)
    sos_bass = butter(4, split_freq, btype='low', fs=SAMPLE_RATE, output='sos')
    sos_treble = butter(4, split_freq, btype='high', fs=SAMPLE_RATE, output='sos')
    
    bass_audio = sosfiltfilt(sos_bass, audio).astype(np.float32)
    treble_audio = sosfiltfilt(sos_treble, audio).astype(np.float32)
    
    bass_rms = np.sqrt(np.mean(bass_audio**2))
    treble_rms = np.sqrt(np.mean(treble_audio**2))
    print(f"[Split] Bass RMS: {bass_rms:.4f}, Treble RMS: {treble_rms:.4f}")
    
    # 2b) Apply per-band noise reduction
    # After splitting, band-specific noise becomes more visible and can be targeted
    print(f"[Noise] Applying per-band noise reduction...")

    if USE_GPU:
        # GPU: Batched spectral gate for both bands simultaneously
        gate_results = gpu_batch_multiband_gate(
            [bass_audio, treble_audio], sr=SAMPLE_RATE,
            n_fft=FFT_SIZE, hop_length=HOP_SIZE,
            noise_estimation_seconds=0.15,
            gate_threshold_db=-8,  # Use bass threshold (less aggressive)
            min_gate_threshold_db=-45
        )
        bass_audio, bass_nr_db = gate_results[0]
        treble_audio, treble_nr_db = gate_results[1]

        # noisereduce still runs on CPU (has internal STFT)
        bass_audio = nr.reduce_noise(
            y=bass_audio, sr=SAMPLE_RATE, stationary=False,
            n_fft=FFT_SIZE, hop_length=HOP_SIZE, prop_decrease=0.5
        ).astype(np.float32)
        treble_audio = nr.reduce_noise(
            y=treble_audio, sr=SAMPLE_RATE, stationary=False,
            n_fft=FFT_SIZE, hop_length=HOP_SIZE, prop_decrease=0.6
        ).astype(np.float32)
    else:
        # CPU: Sequential per-band processing
        # Bass band: Focus on low-frequency rumble
        bass_audio, bass_nr_db = multiband_spectral_gate(
            bass_audio, sr=SAMPLE_RATE, n_fft=FFT_SIZE, hop_length=HOP_SIZE,
            noise_estimation_seconds=0.15,
            gate_threshold_db=-8,
            min_gate_threshold_db=-45
        )
        bass_audio = nr.reduce_noise(
            y=bass_audio, sr=SAMPLE_RATE, stationary=False,
            n_fft=FFT_SIZE, hop_length=HOP_SIZE, prop_decrease=0.5
        ).astype(np.float32)

        # Treble band: Focus on high-frequency hiss and transient noise
        treble_audio, treble_nr_db = multiband_spectral_gate(
            treble_audio, sr=SAMPLE_RATE, n_fft=FFT_SIZE, hop_length=HOP_SIZE,
            noise_estimation_seconds=0.15,
            gate_threshold_db=-10,
            min_gate_threshold_db=-50
        )
        treble_audio = nr.reduce_noise(
            y=treble_audio, sr=SAMPLE_RATE, stationary=False,
            n_fft=FFT_SIZE, hop_length=HOP_SIZE, prop_decrease=0.6
        ).astype(np.float32)
    
    bass_rms_after = np.sqrt(np.mean(bass_audio**2))
    treble_rms_after = np.sqrt(np.mean(treble_audio**2))
    print(f"[Noise] After per-band NR - Bass RMS: {bass_rms_after:.4f} (gate: {bass_nr_db:.1f}dB), Treble RMS: {treble_rms_after:.4f} (gate: {treble_nr_db:.1f}dB)")
    
    # 3) Detect onsets INDEPENDENTLY for each band
    print(f"\n[Bass] Detecting onsets in bass band (with slope validation)...")

    if USE_GPU:
        # GPU: Compute magnitude and flux for both bands in a single batched operation
        bass_mag_stft, bass_flux = gpu_magnitude_and_flux(bass_audio, n_fft=FFT_SIZE, hop_length=HOP_SIZE, window_type=WINDOW_TYPE)
        treble_mag_stft, treble_flux = gpu_magnitude_and_flux(treble_audio, n_fft=FFT_SIZE, hop_length=HOP_SIZE, window_type=WINDOW_TYPE)

        # Transpose magnitude for compatibility: gpu returns (freq, frames), mags array is (frames, freq)
        bass_mags = bass_mag_stft.T
        treble_mags = treble_mag_stft.T
    else:
        bass_frames = frame_audio(bass_audio)
        bass_mags = np.array([compute_magnitude(f) for f in bass_frames])
        bass_flux = normalize(compute_flux(bass_mags))

        treble_frames = frame_audio(treble_audio)
        treble_mags = np.array([compute_magnitude(f) for f in treble_frames])
        treble_flux = normalize(compute_flux(treble_mags))

    # Use slope validation for bass - helps filter noise-induced false onsets
    # K=2.0 matches treble sensitivity so soft accompaniment onsets aren't missed
    bass_onsets = find_onsets_with_slope_validation(
        bass_flux, K=2.0, min_slope_ratio=0.2, slope_window=3, debug=debug
    )
    print(f"[Bass] Found {len(bass_onsets)} validated onsets")

    print(f"\n[Treble] Detecting onsets in treble band...")
    treble_onsets = find_onsets(treble_flux, K=2.0)  # Standard threshold for treble
    print(f"[Treble] Found {len(treble_onsets)} onsets")
    
    # 3b) Detect tempo from combined onset times
    all_onset_times = sorted(set(
        [o * HOP_SIZE / SAMPLE_RATE for o in bass_onsets] +
        [o * HOP_SIZE / SAMPLE_RATE for o in treble_onsets]
    ))
    tempo_info = detect_tempo_from_onsets(all_onset_times)
    detected_bpm = tempo_info['bpm']
    tempo_confidence = tempo_info['confidence']
    beat_interval = tempo_info['beat_interval']
    print(f"\n[Tempo] Detected BPM: {detected_bpm} (confidence: {tempo_confidence:.2f}, beat interval: {beat_interval:.3f}s)")
    
    # 4) Compute shared resources for analysis
    # Full audio chroma for chord quality detection
    chroma_full = extract_chroma(audio, SAMPLE_RATE, hop_length=HOP_SIZE)
    
    # CQT on full audio for pitch detection
    if USE_GPU:
        C_full = gpu_cqt(audio, sr=SAMPLE_RATE, n_bins=CQT_BINS,
                         bins_per_octave=12, fmin=librosa.note_to_hz('A0'),
                         hop_length=HOP_SIZE)
    else:
        C_full = np.abs(librosa.cqt(
            y=audio, sr=SAMPLE_RATE, hop_length=HOP_SIZE,
            n_bins=CQT_BINS, bins_per_octave=12,
            fmin=librosa.note_to_hz('A0')
        ))

    # IMPORTANT: Use FULL AUDIO STFT for pitch detection (BIC needs full harmonic spectrum)
    # Band-filtered audio loses harmonics which causes pitch errors
    _, full_magnitude, _, full_freqs = compute_stft_once(audio)
    
    # Estimate offsets for each band using band-specific chroma
    bass_chroma = extract_chroma(bass_audio, SAMPLE_RATE, hop_length=HOP_SIZE)
    treble_chroma = extract_chroma(treble_audio, SAMPLE_RATE, hop_length=HOP_SIZE)
    
    # Results structure
    results = {
        "onsets": [],
        "notes": [],
        "chords": [],
        "analysis_summary": {
            "duration_seconds": float(len(audio) / SAMPLE_RATE),
            "sample_rate": int(SAMPLE_RATE),
            "bass_onsets": len(bass_onsets),
            "treble_onsets": len(treble_onsets),
            "split_midi": split_midi,
            "independent_hands": True
        }
    }
    
    def process_onset(onset_frame, magnitude, chroma, freqs, hand_label, midi_filter_fn):
        """Process a single onset for a specific hand."""
        time_seconds = onset_frame * HOP_SIZE / SAMPLE_RATE
        
        if debug:
            print(f"\n{'='*60}")
            print(f"{hand_label} ONSET at {time_seconds:.3f}s (frame {onset_frame})")
            print(f"{'='*60}")
        
        # Get magnitude window around onset
        # Also collect individual frames for temporal filtering (HMM-like approach)
        if 1 <= onset_frame < magnitude.shape[1] - 1:
            mag_window = np.mean(magnitude[:, onset_frame-1:onset_frame+2], axis=1)
            # Individual frames for temporal filtering
            mag_frames_raw = [magnitude[:, onset_frame-1], magnitude[:, onset_frame], magnitude[:, onset_frame+1]]
        else:
            mag_window = magnitude[:, min(onset_frame, magnitude.shape[1]-1)]
            mag_frames_raw = [mag_window]  # Single frame fallback
        
        # Ringing cancellation + BIC voice estimation with temporal filtering
        resid, _ = cancel_ringing(mag_window, freqs)
        # Apply ringing cancellation to each frame for temporal filtering
        mag_frames_resid = [cancel_ringing(mf, freqs)[0] for mf in mag_frames_raw]
        bic_est = estimate_voices_bic(resid, max_K=6, H=8, debug=debug, mag_frames=mag_frames_resid)  # Allow more voices for chords
        
        K = bic_est['K']
        midi_set = bic_est['midis']
        salience_info = bic_est.get('salience_info', {})
        
        # For single notes, cross-validate with CQT peaks
        # For chords (K >= 2), trust BIC - CQT validation can incorrectly reject chord tones
        cqt_idx = min(onset_frame + 2, C_full.shape[1] - 1)
        if K == 1 and midi_set and cqt_idx < C_full.shape[1]:
            cqt_frame = C_full[:, cqt_idx]
            midi_set = validate_midi_with_cqt(midi_set, cqt_frame, tolerance_semitones=1, debug=debug)
            K = len(midi_set)  # Update K after validation
        
        # Filter MIDI notes to only include those in the correct range
        midi_set = [m for m in midi_set if midi_filter_fn(m)]
        K = len(midi_set)
        
        # Check if this is a chord BEFORE any reduction
        is_chord = (K >= 2)
        
        # If NOT a chord but multiple notes detected, pick the one with highest FFT salience score
        if not is_chord and K > 1 and salience_info:
            # Sort by salience score (descending) and keep only the top one
            midi_set_scored = [(m, salience_info.get(m, (0, False))[0]) for m in midi_set]
            midi_set_scored.sort(key=lambda x: x[1], reverse=True)
            best_midi = midi_set_scored[0][0]
            if debug:
                print(f"  [Single Note Selection] Multiple notes detected, picking highest salience: {[['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][m % 12] + str(m // 12 - 1) for m, s in midi_set_scored]} -> {['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][best_midi % 12] + str(best_midi // 12 - 1)}")
            midi_set = [best_midi]
            K = 1
        
        if debug and midi_set:
            final_names = [['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][m % 12] + str(m // 12 - 1) for m in midi_set]
            print(f"  [FINAL] {hand_label} detected notes: {final_names}")
        
        if K == 0:
            if debug:
                print(f"  [FINAL] No valid notes in {hand_label} range")
            return None, None  # No valid notes in this range
        
        # Estimate offset using band-specific chroma
        try:
            pc = int(np.argmax(chroma[:, onset_frame])) if onset_frame < chroma.shape[1] else 0
            offsets = estimate_offsets_from_chroma([onset_frame], [pc], chroma)
            oframe = int(offsets[0][1])
            osec = round(oframe * HOP_SIZE / SAMPLE_RATE, 3)
            dur = round(osec - time_seconds, 3)
        except Exception:
            oframe = onset_frame + 10  # Default ~0.1s duration
            osec = round(oframe * HOP_SIZE / SAMPLE_RATE, 3)
            dur = round(osec - time_seconds, 3)
        
        dur = max(dur, 0.05)  # Minimum duration
        note_val = duration_to_note_value(dur, bpm=detected_bpm)  # Use detected tempo
        
        if is_chord:
            # Try detect_chord_multiframe for chord quality/label detection
            res = detect_chord_multiframe(chroma_full, C_full, onset_frame, num_frames=1, debug=False)
            
            if res:
                # Chord confirmed by chroma-based detection
                # Calculate octave from lowest MIDI note
                lowest_midi = int(min(midi_set))
                octave = (lowest_midi // 12) - 1  # MIDI octave calculation
                
                # Build chord from BIC-detected MIDI notes (which are already filtered to correct range)
                chord = {
                    "type": "chord",
                    "time_seconds": round(time_seconds, 3),
                    "frame_index": int(onset_frame),
                    "midi_notes": [int(m) for m in sorted(midi_set)],
                    "note_names": [note_to_name(int(m)) for m in sorted(midi_set)],
                    "root": note_to_name(int(lowest_midi)),
                    "octave": res.get("octave", octave),
                    "frequencies_hz": [round(_midi_to_hz(int(m)), 2) for m in sorted(midi_set)],
                    "method": f"BIC ({hand_label})",
                    "label": res.get("label", f"{note_to_name(lowest_midi)}:?"),
                    "chord_quality": res.get("chord_quality", "unknown"),
                    "inversion": res.get("inversion", "root"),
                    "confidence": res.get("confidence", 0.8),
                    "offset_seconds": osec,
                    "duration_seconds": dur,
                    "hand": hand_label,
                    "note_value": note_val["type"],
                    "note_divisions": note_val["divisions"],
                    "dotted": note_val["dotted"]
                }
                return None, chord
            
            # detect_chord_multiframe returned None - not a confirmed chord
            # Fall back to single note (use the strongest/first MIDI from BIC)
        
        # Single note (either K==1, or chord detection failed)
        m = midi_set[0]
        note = {
            "time_seconds": round(time_seconds, 3),
            "frame_index": int(onset_frame),
            "midi_note": int(m),
            "note_name": note_to_name(int(m)),
            "frequency_hz": round(_midi_to_hz(int(m)), 2),
            "method": f"BIC ({hand_label})",
            "confidence": 0.9,
            "offset_seconds": osec,
            "duration_seconds": dur,
            "hand": hand_label,
            "note_value": note_val["type"],
            "note_divisions": note_val["divisions"],
            "dotted": note_val["dotted"]
        }
        return note, None
    
    # 5) Process bass onsets - use FULL magnitude for pitch, but filter by MIDI range
    print(f"\n[Bass] Processing {len(bass_onsets)} bass onsets...")
    bass_notes = []
    bass_chords = []
    
    for onset in bass_onsets:
        note, chord = process_onset(
            onset, full_magnitude, bass_chroma, full_freqs,
            "bass", lambda m: m < split_midi
        )
        if note:
            bass_notes.append(note)
        if chord:
            bass_chords.append(chord)
    
    print(f"[Bass] Detected {len(bass_notes)} notes, {len(bass_chords)} chords")
    
    # 6) Process treble onsets - use FULL magnitude for pitch, but filter by MIDI range
    print(f"\n[Treble] Processing {len(treble_onsets)} treble onsets...")
    treble_notes = []
    treble_chords = []
    
    for onset in treble_onsets:
        note, chord = process_onset(
            onset, full_magnitude, treble_chroma, full_freqs,
            "treble", lambda m: m >= split_midi
        )
        if note:
            treble_notes.append(note)
        if chord:
            treble_chords.append(chord)
    
    print(f"[Treble] Detected {len(treble_notes)} notes, {len(treble_chords)} chords")
    
    # 7) Merge results (sorted by time)
    all_notes = bass_notes + treble_notes
    all_notes.sort(key=lambda x: x["time_seconds"])
    
    all_chords = bass_chords + treble_chords
    all_chords.sort(key=lambda x: x["time_seconds"])
    
    results["notes"] = all_notes
    results["chords"] = all_chords
    
    # Filter out notes/chords that are too short
    results["notes"] = [n for n in results["notes"] if n.get("duration_seconds", 0) > 0.05]
    results["chords"] = [c for c in results["chords"] if c.get("duration_seconds", 0) > 0.05]
    
    # Deduplicate chords at nearly the same time with same label
    TIME_TOLERANCE = 0.02  # 20ms - tight enough to only dedupe true duplicates
    dedupe_chords = []
    seen_chords = set()
    for c in results["chords"]:
        key = (round(c["time_seconds"] / TIME_TOLERANCE), c.get("label", ""))
        if key not in seen_chords:
            seen_chords.add(key)
            dedupe_chords.append(c)
    results["chords"] = dedupe_chords
    
    # Deduplicate notes at nearly the same time with same MIDI value
    # This prevents dissonant clumps from duplicate detections
    dedupe_notes = []
    seen_notes = set()  # (rounded_time, midi_note)
    for n in results["notes"]:
        key = (round(n["time_seconds"] / TIME_TOLERANCE), n["midi_note"])
        if key not in seen_notes:
            seen_notes.add(key)
            dedupe_notes.append(n)
    results["notes"] = dedupe_notes
    
    # Also filter out notes that are already part of chords at the same time
    notes_in_chords = set()
    for c in results["chords"]:
        chord_time_key = round(c["time_seconds"] / TIME_TOLERANCE)
        for midi in c.get("midi_notes", []):
            notes_in_chords.add((chord_time_key, midi))
    
    results["notes"] = [
        n for n in results["notes"]
        if (round(n["time_seconds"] / TIME_TOLERANCE), n["midi_note"]) not in notes_in_chords
    ]
    
    # RE-QUANTIZE rhythms using IOI-based approach for better accuracy at fast tempos
    # This is more reliable than duration-based quantization because onset detection
    # is more accurate than offset detection
    
    # Refine tempo by testing alternatives and picking the one with lowest quantization error
    all_items = results["notes"] + results["chords"]
    refined_tempo = refine_tempo_by_quantization(all_items, detected_bpm)
    if refined_tempo['bpm'] != detected_bpm:
        detected_bpm = refined_tempo['bpm']
        beat_interval = refined_tempo['beat_interval']
        tempo_confidence = refined_tempo['confidence']
    
    method_name = "ML" if use_ml_rhythm else "IOI"
    print(f"\n[Rhythm] Re-quantizing with {method_name} approach at {detected_bpm} BPM...")
    results["notes"], results["chords"] = quantize_rhythm_sequence(
        results["notes"], results["chords"], detected_bpm, debug=debug, use_ml=use_ml_rhythm
    )
    
    # Detect triplets separately for bass and treble (to avoid cross-hand triplet detection)
    bass_notes_list = [n for n in results["notes"] if n.get("hand") == "bass"]
    treble_notes_list = [n for n in results["notes"] if n.get("hand") == "treble"]
    bass_chords_list = [c for c in results["chords"] if c.get("hand") == "bass"]
    treble_chords_list = [c for c in results["chords"] if c.get("hand") == "treble"]
    
    # Sort by time and detect triplets (using detected BPM)
    bass_notes_list = sorted(bass_notes_list, key=lambda x: x.get("time_seconds", 0))
    treble_notes_list = sorted(treble_notes_list, key=lambda x: x.get("time_seconds", 0))
    bass_chords_list = sorted(bass_chords_list, key=lambda x: x.get("time_seconds", 0))
    treble_chords_list = sorted(treble_chords_list, key=lambda x: x.get("time_seconds", 0))
    
    detect_triplets(bass_notes_list, bpm=detected_bpm, tolerance=0.20)
    detect_triplets(treble_notes_list, bpm=detected_bpm, tolerance=0.20)
    detect_triplets_in_chords(bass_chords_list, bpm=detected_bpm, tolerance=0.20)
    detect_triplets_in_chords(treble_chords_list, bpm=detected_bpm, tolerance=0.20)
    strip_triplets_from_grace_notes(bass_notes_list)
    strip_triplets_from_grace_notes(treble_notes_list)

    # Merge back (already sorted)
    results["notes"] = sorted(bass_notes_list + treble_notes_list, key=lambda x: x.get("time_seconds", 0))
    results["chords"] = sorted(bass_chords_list + treble_chords_list, key=lambda x: x.get("time_seconds", 0))
    
    # Update summary (including tempo info)
    results["analysis_summary"].update({
        "total_onsets": len(bass_onsets) + len(treble_onsets),
        "total_notes": len(results["notes"]),
        "total_chords": len(results["chords"]),
        "detected_bpm": float(detected_bpm),
        "tempo_confidence": float(tempo_confidence),
        "beat_interval": float(beat_interval),
        "bass_notes": len([n for n in results["notes"] if n.get("hand") == "bass"]),
        "treble_notes": len([n for n in results["notes"] if n.get("hand") == "treble"]),
        "bass_chords": len([c for c in results["chords"] if c.get("hand") == "bass"]),
        "treble_chords": len([c for c in results["chords"] if c.get("hand") == "treble"])
    })
    
    print(f"\n{'='*70}")
    print(f"✓ Independent hands analysis complete:")
    print(f"   Tempo:  {detected_bpm:.0f} BPM (confidence: {tempo_confidence:.2f})")
    print(f"   Bass:   {results['analysis_summary']['bass_notes']} notes, {results['analysis_summary']['bass_chords']} chords")
    print(f"   Treble: {results['analysis_summary']['treble_notes']} notes, {results['analysis_summary']['treble_chords']} chords")
    print(f"{'='*70}\n")
    
    return results


def analyze_audio_split_ranges(wav_path_or_array, debug=False, split_midi=60, use_ml_rhythm=True):
    """
    Analyze audio with harmonic subtraction first, then categorize notes into bass/treble.
    This performs harmonic cancellation on the full spectrum, then splits results by MIDI range.
    
    NOTE: This method uses shared onset detection, so bass and treble share the same rhythm.
    For independent rhythm detection (e.g., held bass chord with moving treble melody),
    use analyze_audio_independent_hands() instead.
    
    Args:
        wav_path_or_array: Audio file path or numpy array
        debug: Enable debug output
        split_midi: MIDI note to split at (default 60 = middle C)
        use_ml_rhythm: Use ML-based rhythm quantization (default: True)
    
    Returns:
        Results with notes categorized by bass/treble range
    """
    # 1) Analyze full audio with harmonic subtraction
    print(f"[Split Analysis] Analyzing full audio with harmonic subtraction...")
    results = analyze_audio_optimized(wav_path_or_array, debug=debug, use_ml_rhythm=use_ml_rhythm)
    
    # 2) Categorize detected notes into bass and treble
    bass_notes = []
    treble_notes = []
    
    for note in results.get("notes", []):
        if note["midi_note"] < split_midi:
            bass_notes.append(note)
        else:
            treble_notes.append(note)
    
    # 3) Similarly categorize chords
    bass_chords = []
    treble_chords = []
    
    for chord in results.get("chords", []):
        # Categorize chord by its root note or lowest note
        midi_notes = chord.get("midi_notes", [])
        if midi_notes:
            lowest_midi = min(midi_notes)
            if lowest_midi < split_midi:
                bass_chords.append(chord)
            else:
                treble_chords.append(chord)
    
    # 4) Update results with bass/treble breakdown
    results["analysis_summary"]["bass_notes"] = len(bass_notes)
    results["analysis_summary"]["treble_notes"] = len(treble_notes)
    results["analysis_summary"]["bass_chords"] = len(bass_chords)
    results["analysis_summary"]["treble_chords"] = len(treble_chords)
    
    print(f"[Split Analysis] Categorized results: {len(bass_notes)} bass + {len(treble_notes)} treble = {len(results['notes'])} total notes")
    
    return results


#* ─── NEURAL NETWORK TRANSCRIPTION (piano_transcription_inference) ───────────
def analyze_audio_neural(wav_path, debug=False, split_midi=60, device='cpu', use_ml_rhythm=True):
    """
    High-accuracy polyphonic piano transcription.

    Tries custom trained model first (better soft note detection),
    then falls back to ByteDance's piano_transcription_inference.

    Args:
        wav_path: Path to audio file (must be a file path, not array)
        debug: Enable debug output
        split_midi: MIDI note to split bass/treble hands (default 60 = middle C)
        device: 'cuda' for GPU or 'cpu' for CPU inference
        use_ml_rhythm: Use ML-based rhythm quantization (default: True)

    Returns:
        Results dict compatible with existing format (notes, chords, analysis_summary)
    """
    import time
    neural_timings = {}
    t_neural_start = time.perf_counter()
    audio_full_sr = None  # Store full-rate audio for beat detection (avoid reloading)
    
    # ── Try ensemble multi-resolution model first (fastest, GPU-parallel) ──
    use_ensemble = False
    ensemble_sr = 16000
    ensemble_label = 'Mel Baseline'
    ensemble_onset_threshold = 0.4
    ensemble_frame_threshold = 0.5
    ensemble_offset_threshold = None
    ensemble_duplicate_window_sec = 0.04
    ensemble_merge_gap_sec = 0.0

    def _float_env(name, default):
        try:
            return float(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return default

    def _int_env(name, default):
        try:
            return int(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return default

    def _bool_env(name, default):
        raw_value = os.environ.get(name)
        if raw_value is None:
            return default
        normalized = str(raw_value).strip().lower()
        if normalized in {'1', 'true', 'yes', 'y', 'on'}:
            return True
        if normalized in {'0', 'false', 'no', 'n', 'off'}:
            return False
        return default

    if USE_GPU:
        ensemble_model = get_gpu_enhanced_mel_transcriber()
        if ensemble_model is not None and ensemble_model.initialized:
            use_ensemble = True
            ensemble_sr = ensemble_model.config.get('sample_rate', 16000)
            ensemble_label = 'Enhanced Mel'
            ensemble_onset_threshold = _float_env('ENHANCED_MEL_ONSET_THRESHOLD', 0.75)
            ensemble_frame_threshold = _float_env('ENHANCED_MEL_FRAME_THRESHOLD', 0.50)
            ensemble_offset_threshold = _float_env('ENHANCED_MEL_OFFSET_THRESHOLD', 0.35)
            ensemble_min_velocity = _int_env('ENHANCED_MEL_MIN_VELOCITY', 8)
            ensemble_filter_harmonics = _bool_env('ENHANCED_MEL_FILTER_HARMONICS', False)
            ensemble_duplicate_window_sec = _float_env('ENHANCED_MEL_DUPLICATE_WINDOW_SEC', 0.04)
            ensemble_merge_gap_sec = _float_env('ENHANCED_MEL_MERGE_GAP_SEC', 0.0)
        else:
            ensemble_model = get_gpu_mel_baseline_transcriber()
            if ensemble_model is not None and ensemble_model.initialized:
                use_ensemble = True
                ensemble_sr = ensemble_model.config.get('sample_rate', 16000)
                ensemble_min_velocity = 15
                ensemble_filter_harmonics = True

    if use_ensemble:
        print(f"\n{'='*70}")
        print(f"NEURAL TRANSCRIPTION ({ensemble_label})")
        print(f"   Device: {device}")
        print(f"{'='*70}\n")

        try:
            # Load audio using fast soundfile reader (not librosa which uses ffmpeg)
            # Load at full sample rate (44100) for beat detection,
            # then resample to 16kHz for ensemble model
            t0 = time.perf_counter()
            print(f"[Neural] Loading audio: {wav_path}")
            audio_full_sr, _ = load_audio_deterministic(wav_path, target_sr=SAMPLE_RATE)
            neural_timings['audio_load_ms'] = (time.perf_counter() - t0) * 1000
            print(f"[Neural] Audio load time: {neural_timings['audio_load_ms']:.1f}ms (at {SAMPLE_RATE}Hz, soundfile)")
            
            # Resample for ensemble model (fast polyphase resampling)
            t0 = time.perf_counter()
            # Use scipy's resample_poly for deterministic fast resampling
            gcd = math.gcd(SAMPLE_RATE, ensemble_sr)
            up, down = ensemble_sr // gcd, SAMPLE_RATE // gcd
            audio = resample_poly(audio_full_sr, up, down).astype(np.float32, copy=False)
            neural_timings['resample_ms'] = (time.perf_counter() - t0) * 1000
            print(f"[Neural] Resample to {ensemble_sr}Hz: {neural_timings['resample_ms']:.1f}ms")
            
            duration_seconds = len(audio) / ensemble_sr
            sample_rate_used = ensemble_sr
            print(f"[Neural] Audio duration: {duration_seconds:.2f}s")

            t0 = time.perf_counter()
            print(f"[Neural] Running {ensemble_label.lower()} inference...")
            transcribe_kwargs = {
                'onset_threshold': ensemble_onset_threshold,
                'frame_threshold': ensemble_frame_threshold,
                'min_velocity': ensemble_min_velocity,
                'filter_harmonics': ensemble_filter_harmonics,
                'duplicate_window_sec': ensemble_duplicate_window_sec,
                'merge_gap_sec': ensemble_merge_gap_sec,
            }
            if ensemble_offset_threshold is not None:
                transcribe_kwargs['offset_threshold'] = ensemble_offset_threshold
            transcribed_dict = ensemble_model.transcribe(audio, **transcribe_kwargs)
            neural_timings['transcribe_ms'] = (time.perf_counter() - t0) * 1000
            
            # Extract detailed inference timings if available
            if '_inference_timing_ms' in transcribed_dict:
                neural_timings['inference_detail'] = transcribed_dict['_inference_timing_ms']
                print(f"[Neural] Inference breakdown: {transcribed_dict['_inference_timing_ms']}")

            note_events = transcribed_dict.get('est_note_events', [])
            print(f"[Neural] {ensemble_label} detected {len(note_events)} note events in {neural_timings['transcribe_ms']:.1f}ms")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Neural] {ensemble_label} failed: {e}, falling back to custom model")
            use_ensemble = False

    # ── Try custom trained model second (velocity-weighted, better soft detect) ──
    use_custom = False
    custom_sr = 16000
    if not use_ensemble and USE_GPU:
        custom_model = get_gpu_transcriber()
        if custom_model is not None and custom_model.initialized:
            use_custom = True
            custom_sr = custom_model.config.get('sample_rate', 16000)

    if use_custom:
        print(f"\n{'='*70}")
        print("NEURAL TRANSCRIPTION (Custom velocity-weighted model)")
        print(f"   Device: {device}")
        print(f"{'='*70}\n")

        try:
            print(f"[Neural] Loading audio: {wav_path}")
            audio, _ = librosa.load(path=wav_path, sr=custom_sr, mono=True)
            duration_seconds = len(audio) / custom_sr
            sample_rate_used = custom_sr
            print(f"[Neural] Audio duration: {duration_seconds:.2f}s, sample rate: {custom_sr}")

            print(f"[Neural] Running custom model inference...")
            transcribed_dict = custom_model.transcribe(
                audio,
                onset_threshold=0.35,   # slightly lower for soft note sensitivity
                frame_threshold=0.25,
            )

            note_events = transcribed_dict.get('est_note_events', [])
            print(f"[Neural] Custom model detected {len(note_events)} note events")

        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[Neural] Custom model failed: {e}, falling back to ByteDance")
            use_custom = False

    if not use_ensemble and not use_custom:
        try:
            from piano_transcription_inference import (PianoTranscription,
                                                       sample_rate)
        except ImportError:
            return {"error": "piano_transcription_inference not installed. Run: pip install piano_transcription_inference"}

        print(f"\n{'='*70}")
        print("NEURAL TRANSCRIPTION (ByteDance Piano Transcription)")
        print(f"   Device: {device}")
        print(f"{'='*70}\n")

        try:
            print(f"[Neural] Loading audio: {wav_path}")
            audio, _ = librosa.load(path=wav_path, sr=sample_rate, mono=True)
            duration_seconds = len(audio) / sample_rate
            sample_rate_used = sample_rate
            print(f"[Neural] Audio duration: {duration_seconds:.2f}s, sample rate: {sample_rate}")

            print(f"[Neural] Initializing ByteDance transcriptor...")
            transcriptor = PianoTranscription(device=device, checkpoint_path=None)

            print(f"[Neural] Running inference...")
            transcribed_dict = transcriptor.transcribe(audio, midi_path=None)

            note_events = transcribed_dict.get('est_note_events', [])
            print(f"[Neural] Detected {len(note_events)} note events")

        except Exception as e:
            import traceback
            traceback.print_exc()
            return {"error": f"Neural transcription failed: {str(e)}"}

    if debug:
        for i, event in enumerate(note_events[:20]):
            onset = event['onset_time']
            offset = event['offset_time']
            pitch = event['midi_note']
            velocity = event.get('velocity', 64)
            note_name = note_to_name(int(pitch))
            print(f"  {i+1}. {note_name} (MIDI {pitch}): {onset:.3f}s - {offset:.3f}s, vel={velocity}")
        if len(note_events) > 20:
            print(f"  ... and {len(note_events) - 20} more")
    
    # Convert note_events to our format
    event_groups = _group_neural_note_events_by_onset(note_events)
    
    print(f"[Neural] Grouped into {len(event_groups)} onset events")
    t0 = time.perf_counter()
    
    # Convert groups to notes and chords
    notes = []
    chords = []
    all_onset_times = []
    
    for group in event_groups:
        # Use average onset time for the group
        avg_onset = sum(e['onset_time'] for e in group) / len(group)
        all_onset_times.append(avg_onset)
        
        # Get MIDI notes, sorted low to high
        midi_notes = sorted([int(e['midi_note']) for e in group])
        
        # Duration: use minimum offset time minus onset (conservative)
        min_offset = min(e['offset_time'] for e in group)
        duration = max(0.05, min_offset - avg_onset)
        
        # Average velocity (convert to 0-1 confidence)
        avg_velocity = sum(e.get('velocity', 64) for e in group) / len(group)
        confidence = avg_velocity / 127.0
        
        # Determine hand assignment based on lowest note
        lowest_midi = min(midi_notes)
        hand = "bass" if lowest_midi < split_midi else "treble"
        
        if len(midi_notes) == 1:
            # Single note
            m = midi_notes[0]
            note_dict = {
                "time_seconds": round(avg_onset, 3),
                "midi_note": m,
                "note_name": note_to_name(m),
                "frequency_hz": round(440.0 * 2**((m - 69)/12), 2),
                "method": "neural",
                "confidence": round(confidence, 3),
                "offset_seconds": round(min_offset, 3),
                "duration_seconds": round(duration, 3),
                "hand": hand,
            }
            # Copy ensemble note_value predictions if available
            if 'note_value_name' in group[0]:
                note_dict['note_value'] = group[0]['note_value_name']
                note_dict['note_value_confidence'] = group[0].get('note_value_confidence', 0.5)
                note_dict['note_value_source'] = 'ensemble'
                _normalize_ensemble_note_value(note_dict)
            notes.append(note_dict)
        else:
            # Chord (2+ simultaneous notes)
            # Calculate octave from lowest note (MIDI octave: note // 12 - 1)
            octave = (lowest_midi // 12) - 1
            
            # Try to identify chord quality using existing chord detection logic
            chord_dict = {
                "time_seconds": round(avg_onset, 3),
                "midi_notes": midi_notes,
                "note_names": [note_to_name(m) for m in midi_notes],
                "root": note_to_name(lowest_midi),
                "octave": octave,
                "inversion": "root",  # Neural model doesn't detect inversions, assume root position
                "method": "neural",
                "confidence": round(confidence, 3),
                "offset_seconds": round(min_offset, 3),
                "duration_seconds": round(duration, 3),
                "hand": hand,
                "label": _identify_chord_label(midi_notes),
            }
            # Copy ensemble note_value predictions if available (use most confident from group)
            nv_events = [e for e in group if 'note_value_name' in e]
            if nv_events:
                best_nv = max(nv_events, key=lambda e: e.get('note_value_confidence', 0))
                chord_dict['note_value'] = best_nv['note_value_name']
                chord_dict['note_value_confidence'] = best_nv.get('note_value_confidence', 0.5)
                chord_dict['note_value_source'] = 'ensemble'
                _normalize_ensemble_note_value(chord_dict)
            chords.append(chord_dict)
    
    print(f"[Neural] Converted to: {len(notes)} single notes, {len(chords)} chords")
    neural_timings['note_conversion_ms'] = (time.perf_counter() - t0) * 1000
    print(f"[TIMING] note_conversion={neural_timings['note_conversion_ms']:.1f}ms")
    
    # Detect tempo using neural beat tracking (more accurate than IOI histogram)
    t0 = time.perf_counter()
    print(f"\n[Beat Detection] Running neural beat tracking...")
    # Use pre-loaded audio if available (saves ~10s of re-loading)
    if audio_full_sr is not None:
        print(f"[Beat Detection] Using pre-loaded audio (skipping file reload)")
        beat_info = detect_beats_neural(audio_full_sr, sr=SAMPLE_RATE, debug=debug)
    else:
        print(f"[Beat Detection] Loading audio from file...")
        beat_info = detect_beats_neural(wav_path, debug=debug)
    detected_beat_times = beat_info['beats']
    beat_times = detected_beat_times
    detected_bpm = beat_info['bpm']
    tempo_confidence = beat_info['confidence']
    beat_interval = beat_info['beat_interval']
    neural_timings['beat_detection_ms'] = (time.perf_counter() - t0) * 1000
    print(f"[TIMING] beat_detection={neural_timings['beat_detection_ms']:.1f}ms")
    
    # If beat detection failed or low confidence, fall back to onset-based
    if len(beat_times) < 4 or tempo_confidence < 0.4:
        print(f"[Beat Detection] Low confidence ({tempo_confidence:.2f}), falling back to IOI-based tempo")
        tempo_info = detect_tempo_from_onsets(all_onset_times)
        detected_bpm = tempo_info['bpm']
        tempo_confidence = tempo_info['confidence']
        beat_interval = tempo_info['beat_interval']
        detected_beat_times = np.array([])  # Clear beat times to use IOI quantization
        beat_times = np.array([])
        
        # Refine tempo
        all_events = notes + chords
        refined_tempo = refine_tempo_by_quantization(all_events, detected_bpm)
        detected_bpm = refined_tempo['bpm']
        tempo_confidence = refined_tempo['confidence']
        beat_interval = refined_tempo['beat_interval']
    
    # Quantize rhythms using beat grid if available, otherwise IOI-based
    t0 = time.perf_counter()
    
    # Ornament post-processing is disabled so rhythm quantization operates
    # directly on the decoded notes while note-value accuracy is being tuned.
    neural_timings['ornament_detection_ms'] = 0.0
    print(f"[TIMING] ornament_detection={neural_timings['ornament_detection_ms']:.1f}ms (disabled)")
    
    # Now analyze subdivision patterns across all notes for context-aware quantization
    t0 = time.perf_counter()
    all_events = notes + chords
    subdivision_info = detect_dominant_subdivisions(all_events, detected_bpm, debug=debug)
    neural_timings['subdivision_analysis_ms'] = (time.perf_counter() - t0) * 1000

    # Fine-grained tempo refinement via onset-to-grid alignment sweep
    # Runs even when beat detection succeeded — finds the fixed BPM that
    # minimizes quantization distortion across all note onsets
    t0 = time.perf_counter()
    grid_result = refine_tempo_onset_grid(
        notes, chords, detected_bpm, beat_times=beat_times, debug=debug
    )
    if abs(grid_result['bpm'] - detected_bpm) > 0.5:
        print(f"[Tempo Refined] {detected_bpm:.1f} → {grid_result['bpm']:.1f} BPM "
              f"(grid error: {grid_result['grid_error']*1000:.1f}ms)")
    detected_bpm = grid_result['bpm']
    beat_interval = grid_result['beat_interval']
    # Build a regularized local beat curve for quantization. This preserves
    # short-window tempo variation while smoothing beat tracker jitter.
    synthetic_beat_times = grid_result['beat_times']
    beat_times = build_regularized_local_beat_grid(
        detected_beat_times,
        synthetic_beat_times,
        grid_result['beat_interval'],
        confidence=tempo_confidence,
        debug=debug,
    )
    beat_grid_source = 'local_curve' if len(detected_beat_times) >= 4 else 'synthetic'
    neural_timings['tempo_grid_search_ms'] = (time.perf_counter() - t0) * 1000

    t_quant_start = time.perf_counter()
    print(f"\n[Rhythm] Quantizing at {detected_bpm} BPM...")
    if len(beat_times) >= 4:
        # Use beat-grid quantization (more accurate)
        print(f"[Rhythm] Using beat-grid quantization with {len(beat_times)} detected beats")

        # Separate by hand for independent quantization
        bass_notes = [n for n in notes if n.get('hand') == 'bass']
        treble_notes = [n for n in notes if n.get('hand') == 'treble']
        bass_chords = [c for c in chords if c.get('hand') == 'bass']
        treble_chords = [c for c in chords if c.get('hand') == 'treble']

        # Pre-tag runs (fast scale passages) so quantizer assigns uniform values
        if bass_notes:
            bass_notes = tag_runs_pre_quantization(bass_notes, detected_bpm, debug=debug)
        if treble_notes:
            treble_notes = tag_runs_pre_quantization(treble_notes, detected_bpm, debug=debug)
        if bass_chords:
            bass_chords = tag_runs_pre_quantization(bass_chords, detected_bpm, debug=debug)
        if treble_chords:
            treble_chords = tag_runs_pre_quantization(treble_chords, detected_bpm, debug=debug)

        # Parallel beat-grid quantization + unified post-processing for bass/treble
        if USE_GPU and bass_notes and treble_notes:
            def _quantize_and_normalize(notes_list, bt, bpm, si, debug_flag):
                result = quantize_to_beat_grid(notes_list, bt, bpm, si, debug=debug_flag)
                result = cross_validate_with_acoustic_duration(result, bpm, debug=debug_flag)
                result = post_process_rhythm_unified(result, bpm, debug=debug_flag)
                result = apply_coherence_smoothing(result, bpm, debug=debug_flag)
                return result

            bass_notes, treble_notes = parallel_process_hands(
                _quantize_and_normalize, _quantize_and_normalize,
                (bass_notes, beat_times, detected_bpm, subdivision_info, debug),
                (treble_notes, beat_times, detected_bpm, subdivision_info, debug)
            )
        else:
            if bass_notes:
                bass_notes = quantize_to_beat_grid(bass_notes, beat_times, detected_bpm, subdivision_info, debug=debug)
                bass_notes = cross_validate_with_acoustic_duration(bass_notes, detected_bpm, debug=debug)
                bass_notes = post_process_rhythm_unified(bass_notes, detected_bpm, debug=debug)
                bass_notes = apply_coherence_smoothing(bass_notes, detected_bpm, debug=debug)
            if treble_notes:
                treble_notes = quantize_to_beat_grid(treble_notes, beat_times, detected_bpm, subdivision_info, debug=debug)
                treble_notes = cross_validate_with_acoustic_duration(treble_notes, detected_bpm, debug=debug)
                treble_notes = post_process_rhythm_unified(treble_notes, detected_bpm, debug=debug)
                treble_notes = apply_coherence_smoothing(treble_notes, detected_bpm, debug=debug)

        # Chords can also run in parallel
        if USE_GPU and bass_chords and treble_chords:
            def _quantize_chords(chords_list, bt, bpm, si, debug_flag):
                result = quantize_to_beat_grid(chords_list, bt, bpm, si, debug=debug_flag)
                result = cross_validate_with_acoustic_duration(result, bpm, debug=debug_flag)
                result = post_process_rhythm_unified(result, bpm, debug=debug_flag)
                result = apply_coherence_smoothing(result, bpm, debug=debug_flag)
                return result

            bass_chords, treble_chords = parallel_process_hands(
                _quantize_chords, _quantize_chords,
                (bass_chords, beat_times, detected_bpm, subdivision_info, debug),
                (treble_chords, beat_times, detected_bpm, subdivision_info, debug)
            )
        else:
            if bass_chords:
                bass_chords = quantize_to_beat_grid(bass_chords, beat_times, detected_bpm, subdivision_info, debug=debug)
                bass_chords = cross_validate_with_acoustic_duration(bass_chords, detected_bpm, debug=debug)
                bass_chords = post_process_rhythm_unified(bass_chords, detected_bpm, debug=debug)
                bass_chords = apply_coherence_smoothing(bass_chords, detected_bpm, debug=debug)
            if treble_chords:
                treble_chords = quantize_to_beat_grid(treble_chords, beat_times, detected_bpm, subdivision_info, debug=debug)
                treble_chords = cross_validate_with_acoustic_duration(treble_chords, detected_bpm, debug=debug)
                treble_chords = post_process_rhythm_unified(treble_chords, detected_bpm, debug=debug)
                treble_chords = apply_coherence_smoothing(treble_chords, detected_bpm, debug=debug)
        
        notes = sorted(bass_notes + treble_notes, key=lambda x: x.get('time_seconds', 0))
        chords = sorted(bass_chords + treble_chords, key=lambda x: x.get('time_seconds', 0))
        neural_timings['rhythm_quantization_ms'] = (time.perf_counter() - t_quant_start) * 1000
        print(f"[TIMING] rhythm_quantization={neural_timings['rhythm_quantization_ms']:.1f}ms (beat-grid)")
    else:
        # Fall back to ML/IOI-based quantization
        method_name = "ML" if use_ml_rhythm else "IOI"
        print(f"[Rhythm] Using {method_name}-based quantization (no beat grid available)")
        notes, chords = quantize_rhythm_sequence(notes, chords, detected_bpm, debug=debug, use_ml=use_ml_rhythm)
        # post_process_rhythm_unified (including run normalization) is already
        # called inside quantize_rhythm_from_ioi / quantize_rhythm_ml
        neural_timings['rhythm_quantization_ms'] = (time.perf_counter() - t_quant_start) * 1000
        print(f"[TIMING] rhythm_quantization={neural_timings['rhythm_quantization_ms']:.1f}ms ({method_name}-based)")
    
    # Detect triplets (separately for bass and treble)
    t0 = time.perf_counter()
    bass_notes_list = [n for n in notes if n.get("hand") == "bass"]
    treble_notes_list = [n for n in notes if n.get("hand") == "treble"]
    bass_chords_list = [c for c in chords if c.get("hand") == "bass"]
    treble_chords_list = [c for c in chords if c.get("hand") == "treble"]
    
    bass_notes_list = sorted(bass_notes_list, key=lambda x: x.get("time_seconds", 0))
    treble_notes_list = sorted(treble_notes_list, key=lambda x: x.get("time_seconds", 0))
    bass_chords_list = sorted(bass_chords_list, key=lambda x: x.get("time_seconds", 0))
    treble_chords_list = sorted(treble_chords_list, key=lambda x: x.get("time_seconds", 0))
    
    detect_triplets(bass_notes_list, bpm=detected_bpm, tolerance=0.20)
    detect_triplets(treble_notes_list, bpm=detected_bpm, tolerance=0.20)
    detect_triplets_in_chords(bass_chords_list, bpm=detected_bpm, tolerance=0.20)
    detect_triplets_in_chords(treble_chords_list, bpm=detected_bpm, tolerance=0.20)
    strip_triplets_from_grace_notes(bass_notes_list)
    strip_triplets_from_grace_notes(treble_notes_list)

    # Merge back
    notes = sorted(bass_notes_list + treble_notes_list, key=lambda x: x.get("time_seconds", 0))
    chords = sorted(bass_chords_list + treble_chords_list, key=lambda x: x.get("time_seconds", 0))

    # Sync is_triplet -> triplet: ML paths set 'is_triplet' but frontend reads 'triplet'.
    # detect_triplets() sets 'triplet' for IOI-detected groups. Propagate ML predictions
    # for notes where triplet wasn't already set by IOI detection.
    for n in notes:
        if n.get('is_triplet', False) and not n.get('triplet', False):
            n['triplet'] = True
    for c in chords:
        if c.get('is_triplet', False) and not c.get('triplet', False):
            c['triplet'] = True

    apply_backend_timing_authority(notes, chords, beat_times)

    neural_timings['triplet_detection_ms'] = (time.perf_counter() - t0) * 1000
    print(f"[TIMING] triplet_detection={neural_timings['triplet_detection_ms']:.1f}ms")

    # Compute notation proximity score (how close quantized rhythm is to raw timing)
    t_prox = time.perf_counter()
    proximity_score = compute_notation_proximity_score(notes, chords, detected_bpm, debug=True)
    neural_timings['proximity_score_ms'] = (time.perf_counter() - t_prox) * 1000

    # Build results
    results = {
        "notes": notes,
        "chords": chords,
        "onsets": [{"time_seconds": t, "frame_index": int(t * SAMPLE_RATE / HOP_SIZE)} for t in all_onset_times],
        "analysis_summary": {
            "duration_seconds": round(duration_seconds, 3),
            "sample_rate": int(sample_rate_used),
            "total_onsets": len(event_groups),
            "total_notes": len(notes),
            "total_chords": len(chords),
            "detected_bpm": float(detected_bpm),
            "tempo_confidence": float(tempo_confidence),
            "beat_interval": float(beat_interval),
            "bass_notes": len([n for n in notes if n.get("hand") == "bass"]),
            "treble_notes": len([n for n in notes if n.get("hand") == "treble"]),
            "bass_chords": len([c for c in chords if c.get("hand") == "bass"]),
            "treble_chords": len([c for c in chords if c.get("hand") == "treble"]),
            "method": "neural (piano_transcription_inference)",
            "device": device,
            "rhythm_method": "beat_grid" if len(beat_times) >= 4 else "ioi",
            "beat_grid_source": beat_grid_source,
            "local_tempo_curve": beat_grid_source == 'local_curve',
            "backend_timing_authority": True,
            "primary_subdivision": subdivision_info.get('primary_subdivision', 'quarter'),
            "uses_triplets": subdivision_info.get('uses_triplets', False),
            "uses_dotted": subdivision_info.get('uses_dotted', False),
        },
        "proximity_score": proximity_score,
    }

    # Add timing info to results (for bottleneck analysis)
    neural_timings['total_ms'] = (time.perf_counter() - t_neural_start) * 1000
    neural_timings['real_time_factor'] = neural_timings['total_ms'] / (duration_seconds * 1000)
    results['_neural_timing_ms'] = neural_timings

    # Print full timing breakdown
    print(f"\n[TIMING] Full breakdown:")
    for key, val in neural_timings.items():
        if key not in ('inference_detail', 'real_time_factor'):
            print(f"   {key}: {val:.1f}ms" if isinstance(val, (int, float)) else f"   {key}: {val}")

    print(f"\n{'='*70}")
    print(f"✓ Neural transcription complete:")
    print(f"   Tempo:  {detected_bpm:.0f} BPM (confidence: {tempo_confidence:.2f})")
    print(f"   Rhythm: {subdivision_info.get('primary_subdivision', 'quarter')}s, "
          f"triplets={'yes' if subdivision_info.get('uses_triplets') else 'no'}, "
          f"dotted={'yes' if subdivision_info.get('uses_dotted') else 'no'}")
    print(f"   Bass:   {results['analysis_summary']['bass_notes']} notes, {results['analysis_summary']['bass_chords']} chords")
    print(f"   Treble: {results['analysis_summary']['treble_notes']} notes, {results['analysis_summary']['treble_chords']} chords")
    if proximity_score.get('overall'):
        o = proximity_score['overall']
        print(f"   Score:  mean_drift={o['mean_drift_ms']:.1f}ms, "
              f"within_16th={o['pct_within_16th']*100:.0f}%, "
              f"within_8th={o['pct_within_8th']*100:.0f}%")
    print(f"   Timing: total={neural_timings['total_ms']:.1f}ms, RTF={neural_timings['real_time_factor']:.4f}")
    print(f"{'='*70}\n")
    
    return results


def _identify_chord_label(midi_notes):
    """
    Helper to identify chord label from MIDI notes.
    Returns a string like 'C:maj' or 'G:min7'.
    """
    if len(midi_notes) < 2:
        return f"{note_to_name(midi_notes[0])}:single" if midi_notes else "unknown"
    
    # Get pitch classes (0-11)
    pcs = sorted(set([m % 12 for m in midi_notes]))
    root_pc = min(midi_notes) % 12
    root_name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][root_pc]
    
    # Calculate intervals from root
    intervals = sorted([(pc - root_pc) % 12 for pc in pcs])
    
    # Match against known chord patterns
    chord_patterns = {
        (0, 4, 7): 'maj',
        (0, 3, 7): 'min',
        (0, 3, 6): 'dim',
        (0, 4, 8): 'aug',
        (0, 2, 7): 'sus2',
        (0, 5, 7): 'sus4',
        (0, 4, 7, 11): 'maj7',
        (0, 3, 7, 10): 'min7',
        (0, 4, 7, 10): 'dom7',
        (0, 3, 6, 10): 'm7b5',
        (0, 3, 6, 9): 'dim7',
        (0, 4, 7, 9): 'maj6',
        (0, 3, 7, 9): 'min6',
    }
    
    intervals_tuple = tuple(intervals)
    if intervals_tuple in chord_patterns:
        return f"{root_name}:{chord_patterns[intervals_tuple]}"
    
    # Try matching subsets (for incomplete voicings)
    for pattern, name in chord_patterns.items():
        if all(i in intervals for i in pattern):
            return f"{root_name}:{name}"
    
    # Fallback
    return f"{root_name}:chord"


def _env_float(name, default):
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name, default=False):
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    normalized = str(raw_value).strip().lower()
    if normalized in {'1', 'true', 'yes', 'y', 'on'}:
        return True
    if normalized in {'0', 'false', 'no', 'n', 'off'}:
        return False
    return default


_NEURAL_SIMULTANEOUS_BASE_TOLERANCE_SEC = _env_float('LIVE_NEURAL_GROUP_BASE_TOLERANCE_SEC', 0.03)
_NEURAL_SIMULTANEOUS_MIN_TOLERANCE_SEC = _env_float('LIVE_NEURAL_GROUP_MIN_TOLERANCE_SEC', 0.012)
_NEURAL_SIMULTANEOUS_GROUP_SHRINK_SEC = _env_float('LIVE_NEURAL_GROUP_SHRINK_SEC', 0.004)
_NEURAL_SIMULTANEOUS_STEP_RATIO = _env_float('LIVE_NEURAL_GROUP_STEP_RATIO', 0.50)
_NEURAL_GROUP_PRUNE_ENABLED = _env_bool('LIVE_NEURAL_GROUP_PRUNE_ENABLED', False)
_NEURAL_GROUP_PRUNE_MIN_SIZE = max(2, int(_env_float('LIVE_NEURAL_GROUP_PRUNE_MIN_SIZE', 3)))
_NEURAL_GROUP_PRUNE_ABS_ONSET = _env_float('LIVE_NEURAL_GROUP_PRUNE_ABS_ONSET', 0.55)
_NEURAL_GROUP_PRUNE_MEDIAN_RATIO = _env_float('LIVE_NEURAL_GROUP_PRUNE_MEDIAN_RATIO', 0.55)


def _adaptive_neural_group_tolerances(group_size):
    extra_voices = max(0, int(group_size) - 2)
    span_tolerance = max(
        _NEURAL_SIMULTANEOUS_MIN_TOLERANCE_SEC,
        _NEURAL_SIMULTANEOUS_BASE_TOLERANCE_SEC
        - (extra_voices * _NEURAL_SIMULTANEOUS_GROUP_SHRINK_SEC),
    )
    step_tolerance = max(
        _NEURAL_SIMULTANEOUS_MIN_TOLERANCE_SEC,
        span_tolerance * _NEURAL_SIMULTANEOUS_STEP_RATIO,
    )
    return span_tolerance, step_tolerance


def _group_neural_note_events_by_onset(note_events):
    # Dense polyphonic clips tend to overmerge when every group gets the same
    # fixed simultaneity window. Shrink the allowed span as a group grows.
    note_events_sorted = sorted(note_events or [], key=lambda event: event['onset_time'])
    event_groups = []
    current_group = []

    for event in note_events_sorted:
        if not current_group:
            current_group = [event]
            continue

        onset = float(event.get('onset_time', 0.0) or 0.0)
        group_start = float(current_group[0].get('onset_time', 0.0) or 0.0)
        previous_onset = float(current_group[-1].get('onset_time', group_start) or group_start)
        span_tolerance, step_tolerance = _adaptive_neural_group_tolerances(len(current_group))

        within_group_span = (onset - group_start) <= span_tolerance
        near_previous_attack = (onset - previous_onset) <= step_tolerance

        if within_group_span and near_previous_attack:
            current_group.append(event)
            continue

        event_groups.append(current_group)
        current_group = [event]

    if current_group:
        event_groups.append(current_group)

    return event_groups


def _prune_neural_group_confidence_outliers(group):
    if not _NEURAL_GROUP_PRUNE_ENABLED or len(group) < _NEURAL_GROUP_PRUNE_MIN_SIZE:
        return group

    onset_probs = [float(event.get('onset_prob', 0.5) or 0.0) for event in group]
    if not onset_probs:
        return group

    median_prob = float(np.median(np.asarray(onset_probs, dtype=np.float64)))
    keep = [
        event
        for event, onset_prob in zip(group, onset_probs)
        if not (
            onset_prob < _NEURAL_GROUP_PRUNE_ABS_ONSET
            and onset_prob < median_prob * _NEURAL_GROUP_PRUNE_MEDIAN_RATIO
        )
    ]
    return keep or group


def _convert_neural_note_events_to_results(note_events, split_midi=60):
    """Convert frame-level neural note events into the live note/chord payload shape."""
    event_groups = _group_neural_note_events_by_onset(note_events)

    notes = []
    chords = []
    onsets = []

    for group in event_groups:
        group = _prune_neural_group_confidence_outliers(group)
        avg_onset = sum(event['onset_time'] for event in group) / len(group)
        midi_notes = sorted(int(event['midi_note']) for event in group)
        min_offset = min(event['offset_time'] for event in group)
        duration = max(0.05, min_offset - avg_onset)
        avg_velocity = sum(event.get('velocity', 64) for event in group) / len(group)
        confidence = avg_velocity / 127.0
        lowest_midi = min(midi_notes)
        hand = 'bass' if lowest_midi < split_midi else 'treble'

        onsets.append({
            'time_seconds': round(avg_onset, 3),
            'offset_seconds': round(min_offset, 3),
            'duration_seconds': round(duration, 3),
        })

        if len(midi_notes) == 1:
            midi_note = midi_notes[0]
            note_dict = {
                'time_seconds': round(avg_onset, 3),
                'midi_note': midi_note,
                'note_name': note_to_name(midi_note),
                'frequency_hz': round(440.0 * 2 ** ((midi_note - 69) / 12), 2),
                'method': 'neural_live',
                'confidence': round(confidence, 3),
                'offset_seconds': round(min_offset, 3),
                'duration_seconds': round(duration, 3),
                'hand': hand,
            }
            if 'note_value_name' in group[0]:
                note_dict['note_value'] = group[0]['note_value_name']
                note_dict['note_value_confidence'] = group[0].get('note_value_confidence', 0.5)
                note_dict['note_value_source'] = 'ensemble'
                _normalize_ensemble_note_value(note_dict)
            notes.append(note_dict)
            continue

        octave = (lowest_midi // 12) - 1
        chord_dict = {
            'time_seconds': round(avg_onset, 3),
            'midi_notes': midi_notes,
            'note_names': [note_to_name(midi_note) for midi_note in midi_notes],
            'root': note_to_name(lowest_midi),
            'octave': octave,
            'inversion': 'root',
            'method': 'neural_live',
            'confidence': round(confidence, 3),
            'offset_seconds': round(min_offset, 3),
            'duration_seconds': round(duration, 3),
            'hand': hand,
            'label': _identify_chord_label(midi_notes),
        }
        sorted_group = sorted(group, key=lambda event: int(event['midi_note']))
        chord_dict['note_probabilities'] = [float(event.get('onset_prob', 0.5)) for event in sorted_group]
        nv_events = [event for event in group if 'note_value_name' in event]
        if nv_events:
            best_nv = max(nv_events, key=lambda event: event.get('note_value_confidence', 0))
            chord_dict['note_value'] = best_nv['note_value_name']
            chord_dict['note_value_confidence'] = best_nv.get('note_value_confidence', 0.5)
            chord_dict['note_value_source'] = 'ensemble'
            _normalize_ensemble_note_value(chord_dict)
        chords.append(chord_dict)

    return {
        'onsets': onsets,
        'notes': notes,
        'chords': chords,
        'event_groups': len(event_groups),
    }


def _select_live_neural_onset_threshold(audio_chunk, base_onset_threshold, enabled=True):
    """Adjust live onset sensitivity from cheap chunk loudness stats only."""
    experiment = 'adaptive_onset_loudness_v1' if enabled else 'fixed_onset_baseline'

    if audio_chunk is None:
        return float(base_onset_threshold), {
            'experiment': experiment,
            'profile': 'no_audio',
            'chunk_rms': 0.0,
            'peak_level': 0.0,
            'crest_factor': 0.0,
        }

    audio = np.asarray(audio_chunk, dtype=np.float32)
    if audio.size == 0:
        return float(base_onset_threshold), {
            'experiment': experiment,
            'profile': 'empty_audio',
            'chunk_rms': 0.0,
            'peak_level': 0.0,
            'crest_factor': 0.0,
        }

    audio64 = audio.astype(np.float64, copy=False)
    abs_audio = np.abs(audio64)
    chunk_rms = float(np.sqrt(np.mean(audio64 * audio64)))
    peak_level = float(np.max(abs_audio))
    crest_factor = float(peak_level / max(chunk_rms, 1e-6))

    selected = float(base_onset_threshold)
    profile = 'fixed_baseline' if not enabled else 'baseline_nominal'

    if enabled:
        if chunk_rms < 0.024 and peak_level < 0.45:
            selected = max(0.30, base_onset_threshold - 0.04)
            profile = 'soft_sparse_recall'
        elif chunk_rms > 0.110 or (chunk_rms > 0.060 and crest_factor < 2.30):
            precision_cap = 0.46 if base_onset_threshold <= 0.50 else 0.95
            selected = min(precision_cap, base_onset_threshold + 0.02)
            profile = 'loud_dense_precision'

    return float(selected), {
        'experiment': experiment,
        'profile': profile,
        'chunk_rms': chunk_rms,
        'peak_level': peak_level,
        'crest_factor': crest_factor,
    }


def analyze_audio_live_neural(audio_or_path, sr=SAMPLE_RATE, debug=False, split_midi=60, device='cuda', adaptive_onset_threshold=True):
    """Run a minimal array-based neural transcription path for live chunk updates."""
    import time

    def _format_loader_status(label, status):
        reason = str(status.get('reason') or 'unknown')
        details = [f"{label}={reason}"]
        selected_path = status.get('selected_path')
        last_error = status.get('last_error')
        if selected_path:
            details.append(f"path={selected_path}")
        if last_error:
            details.append(f"error={last_error}")
        return ', '.join(details)

    timings = {}
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    if isinstance(audio_or_path, str):
        audio_full_sr, sr = load_audio_deterministic(audio_or_path, target_sr=sr)
    else:
        audio_full_sr = np.asarray(audio_or_path, dtype=np.float32)
        if audio_full_sr.ndim > 1:
            audio_full_sr = np.mean(audio_full_sr, axis=1)
        audio_full_sr = audio_full_sr.astype(np.float32, copy=False)
    timings['neural_audio_prepare'] = (time.perf_counter() - t0) * 1000

    if audio_full_sr.size == 0:
        return {'error': 'Live neural transcription received empty audio.'}

    model_name = None
    note_events = []
    model_sr = sr
    inference_detail = {}
    selected_onset_threshold = 0.0
    onset_threshold_profile = 'not_used'
    onset_threshold_experiment = (
        'adaptive_onset_loudness_v1'
        if adaptive_onset_threshold
        else 'fixed_onset_baseline'
    )
    enhanced_status = {
        'reason': 'not_attempted',
        'selected_path': None,
        'last_error': None,
    }
    mel_status = {
        'reason': 'not_attempted',
        'selected_path': None,
        'last_error': None,
    }
    custom_status = {
        'reason': 'not_attempted',
        'selected_path': None,
        'last_error': None,
    }

    if USE_GPU:
        ensemble_model = get_gpu_enhanced_mel_transcriber()
        enhanced_status = get_gpu_enhanced_mel_transcriber_status()
        if ensemble_model is not None and ensemble_model.initialized:
            model_name = 'enhanced_mel'
            model_sr = int(ensemble_model.config.get('sample_rate', 16000))

            t0 = time.perf_counter()
            if sr != model_sr:
                gcd = math.gcd(sr, model_sr)
                up, down = model_sr // gcd, sr // gcd
                model_audio = resample_poly(audio_full_sr, up, down).astype(np.float32, copy=False)
            else:
                model_audio = audio_full_sr
            timings['neural_resample'] = (time.perf_counter() - t0) * 1000

            try:
                _enhanced_base_thr = float(os.environ.get('LIVE_ENHANCED_ONSET_BASE', '0.75'))
            except (TypeError, ValueError):
                _enhanced_base_thr = 0.75
            try:
                _enhanced_offset_thr = float(os.environ.get('LIVE_ENHANCED_OFFSET_BASE', '0.35'))
            except (TypeError, ValueError):
                _enhanced_offset_thr = 0.35
            try:
                _enhanced_min_velocity = int(os.environ.get('LIVE_ENHANCED_MIN_VELOCITY', '8'))
            except (TypeError, ValueError):
                _enhanced_min_velocity = 8
            try:
                _enhanced_duplicate_window_sec = float(os.environ.get('LIVE_ENHANCED_DUPLICATE_WINDOW_SEC', '0.04'))
            except (TypeError, ValueError):
                _enhanced_duplicate_window_sec = 0.04
            try:
                _enhanced_merge_gap_sec = float(os.environ.get('LIVE_ENHANCED_MERGE_GAP_SEC', '0.0'))
            except (TypeError, ValueError):
                _enhanced_merge_gap_sec = 0.0
            _enhanced_filter_harmonics_raw = str(
                os.environ.get('LIVE_ENHANCED_FILTER_HARMONICS', '0')
            ).strip().lower()
            _enhanced_filter_harmonics = _enhanced_filter_harmonics_raw in {
                '1', 'true', 'yes', 'y', 'on'
            }
            selected_onset_threshold, threshold_debug = _select_live_neural_onset_threshold(
                model_audio,
                _enhanced_base_thr,
                enabled=adaptive_onset_threshold,
            )
            onset_threshold_profile = str(threshold_debug.get('profile') or 'baseline_nominal')
            onset_threshold_experiment = str(
                threshold_debug.get('experiment') or onset_threshold_experiment
            )
            timings['neural_onset_threshold_base'] = _enhanced_base_thr
            timings['neural_onset_threshold_selected'] = selected_onset_threshold
            timings['neural_offset_threshold'] = _enhanced_offset_thr
            timings['neural_min_velocity'] = _enhanced_min_velocity
            timings['neural_filter_harmonics'] = bool(_enhanced_filter_harmonics)
            timings['neural_duplicate_window_sec'] = _enhanced_duplicate_window_sec
            timings['neural_merge_gap_sec'] = _enhanced_merge_gap_sec
            timings['neural_chunk_rms'] = float(threshold_debug.get('chunk_rms') or 0.0)
            timings['neural_chunk_peak'] = float(threshold_debug.get('peak_level') or 0.0)
            timings['neural_chunk_crest_factor'] = float(threshold_debug.get('crest_factor') or 0.0)

            t0 = time.perf_counter()
            transcribed_dict = ensemble_model.transcribe(
                model_audio,
                onset_threshold=selected_onset_threshold,
                offset_threshold=_enhanced_offset_thr,
                frame_threshold=0.5,
                min_velocity=_enhanced_min_velocity,
                duplicate_window_sec=_enhanced_duplicate_window_sec,
                merge_gap_sec=_enhanced_merge_gap_sec,
                filter_harmonics=_enhanced_filter_harmonics,
            )
            timings['neural_transcribe'] = (time.perf_counter() - t0) * 1000
            inference_detail = transcribed_dict.get('_inference_timing_ms') or {}
            note_events = transcribed_dict.get('est_note_events', [])

    if model_name is None and USE_GPU:
        ensemble_model = get_gpu_mel_baseline_transcriber()
        mel_status = get_gpu_mel_baseline_transcriber_status()
        if ensemble_model is not None and ensemble_model.initialized:
            model_name = 'mel_baseline'
            model_sr = int(ensemble_model.config.get('sample_rate', 16000))

            t0 = time.perf_counter()
            if sr != model_sr:
                gcd = math.gcd(sr, model_sr)
                up, down = model_sr // gcd, sr // gcd
                model_audio = resample_poly(audio_full_sr, up, down).astype(np.float32, copy=False)
            else:
                model_audio = audio_full_sr
            timings['neural_resample'] = (time.perf_counter() - t0) * 1000

            # Base onset threshold. With the longer LIVE_CONTEXT_SEC window the
            # model is well-calibrated, so a higher base (0.46) sharply improves
            # precision / display cluster F1 at a negligible recall cost vs the old
            # short-chunk value of 0.38. Tunable via LIVE_ONSET_BASE.
            try:
                _mel_base_thr = float(os.environ.get('LIVE_ONSET_BASE', '0.46'))
            except (TypeError, ValueError):
                _mel_base_thr = 0.46
            selected_onset_threshold, threshold_debug = _select_live_neural_onset_threshold(
                model_audio,
                _mel_base_thr,
                enabled=adaptive_onset_threshold,
            )
            onset_threshold_profile = str(threshold_debug.get('profile') or 'baseline_nominal')
            onset_threshold_experiment = str(
                threshold_debug.get('experiment') or onset_threshold_experiment
            )
            timings['neural_onset_threshold_base'] = _mel_base_thr
            timings['neural_onset_threshold_selected'] = selected_onset_threshold
            timings['neural_chunk_rms'] = float(threshold_debug.get('chunk_rms') or 0.0)
            timings['neural_chunk_peak'] = float(threshold_debug.get('peak_level') or 0.0)
            timings['neural_chunk_crest_factor'] = float(threshold_debug.get('crest_factor') or 0.0)

            t0 = time.perf_counter()
            transcribed_dict = ensemble_model.transcribe(
                model_audio,
                onset_threshold=selected_onset_threshold,
                frame_threshold=0.5,
            )
            timings['neural_transcribe'] = (time.perf_counter() - t0) * 1000
            inference_detail = transcribed_dict.get('_inference_timing_ms') or {}
            note_events = transcribed_dict.get('est_note_events', [])

    if model_name is None and USE_GPU:
        custom_model = get_gpu_transcriber()
        custom_status = get_gpu_transcriber_status()
        if custom_model is not None and custom_model.initialized:
            model_name = 'custom_velocity_weighted'
            model_sr = int(custom_model.config.get('sample_rate', 16000))

            t0 = time.perf_counter()
            if sr != model_sr:
                gcd = math.gcd(sr, model_sr)
                up, down = model_sr // gcd, sr // gcd
                model_audio = resample_poly(audio_full_sr, up, down).astype(np.float32, copy=False)
            else:
                model_audio = audio_full_sr
            timings['neural_resample'] = (time.perf_counter() - t0) * 1000

            selected_onset_threshold, threshold_debug = _select_live_neural_onset_threshold(
                model_audio,
                0.33,
                enabled=adaptive_onset_threshold,
            )
            onset_threshold_profile = str(threshold_debug.get('profile') or 'baseline_nominal')
            onset_threshold_experiment = str(
                threshold_debug.get('experiment') or onset_threshold_experiment
            )
            timings['neural_onset_threshold_base'] = 0.33
            timings['neural_onset_threshold_selected'] = selected_onset_threshold
            timings['neural_chunk_rms'] = float(threshold_debug.get('chunk_rms') or 0.0)
            timings['neural_chunk_peak'] = float(threshold_debug.get('peak_level') or 0.0)
            timings['neural_chunk_crest_factor'] = float(threshold_debug.get('crest_factor') or 0.0)

            t0 = time.perf_counter()
            transcribed_dict = custom_model.transcribe(
                model_audio,
                onset_threshold=selected_onset_threshold,
                frame_threshold=0.25,
            )
            timings['neural_transcribe'] = (time.perf_counter() - t0) * 1000
            note_events = transcribed_dict.get('est_note_events', [])

    if model_name is None:
        if not USE_GPU:
            error_message = 'Live neural transcription unavailable: gpu_ops reports cuda_unavailable.'
            error_code = 'cuda_unavailable'
        else:
            error_message = (
                'Live neural transcription unavailable: '
                + _format_loader_status('enhanced_mel', enhanced_status)
                + '; '
                + _format_loader_status('mel_baseline', mel_status)
                + '; '
                + _format_loader_status('custom_transcriber', custom_status)
            )
            error_code = 'no_gpu_transcriber_initialized'
        return {
            'error': error_message,
            'error_code': error_code,
            'loader_status': {
                'enhanced_mel': enhanced_status,
                'mel_baseline': mel_status,
                'custom_transcriber': custom_status,
            },
        }

    t0 = time.perf_counter()
    converted = _convert_neural_note_events_to_results(note_events, split_midi=split_midi)
    timings['neural_note_conversion'] = (time.perf_counter() - t0) * 1000

    duration_seconds = float(len(audio_full_sr) / sr) if sr > 0 else 0.0
    total_ms = (time.perf_counter() - total_start) * 1000
    timings['neural_total'] = total_ms
    timings['neural_audio_duration'] = duration_seconds * 1000.0
    timings['neural_real_time_factor'] = (total_ms / (duration_seconds * 1000.0)) if duration_seconds > 0 else 0.0

    inference_key_map = {
        'audio_to_gpu': 'neural_audio_to_gpu',
        'feature_extraction': 'neural_feature_extraction',
        'model_inference': 'neural_model_inference',
        'decode_notes': 'neural_decode_notes',
        'total': 'neural_model_total',
        'audio_duration_ms': 'neural_model_audio_duration',
        'n_frames': 'neural_model_frames',
        'n_chunks': 'neural_model_chunks',
        'real_time_factor': 'neural_model_real_time_factor',
    }
    for key, value in inference_detail.items():
        if not isinstance(value, (int, float)):
            continue
        timings[inference_key_map.get(key, f'neural_model_{key}')] = float(value)

    if debug:
        print(
            f"[Live Neural] model={model_name} notes={len(converted['notes'])} chords={len(converted['chords'])} "
            f"onset_threshold={selected_onset_threshold:.2f} profile={onset_threshold_profile} "
            f"total={total_ms:.1f}ms rtf={timings['neural_real_time_factor']:.2f}x"
        )

    return {
        'onsets': converted['onsets'],
        'notes': converted['notes'],
        'chords': converted['chords'],
        'analysis_summary': {
            'total_onsets': len(converted['onsets']),
            'total_notes': len(converted['notes']),
            'total_chords': len(converted['chords']),
            'duration_seconds': duration_seconds,
            'sample_rate': int(sr),
            'analysis_path': 'live_neural',
            'neural_model': model_name,
            'event_groups': converted['event_groups'],
            'live_onset_threshold_experiment': onset_threshold_experiment,
            'live_onset_threshold_profile': onset_threshold_profile,
            'live_onset_threshold': round(float(selected_onset_threshold), 3),
        },
        '_timing_ms': {key: round(float(value), 3) for key, value in timings.items()},
    }


def analyze_audio(wav_path_or_array, debug=False, use_split=True, independent_hands=True, use_neural=False, device='cpu', use_ml_rhythm=True):
    """
    Main audio analysis function.
    
    Args:
        wav_path_or_array: Audio file path or numpy array
        debug: Enable debug output
        use_split: If True, use frequency range splitting to separate left/right hand (default: True)
        independent_hands: If True and use_split is True, detect bass and treble rhythms 
                          independently (enables held bass chord + moving treble melody).
                          If False, uses shared onset detection (default: True)
        use_neural: If True, use neural network transcription (requires piano_transcription_inference).
                   This typically gives much higher accuracy but requires more memory/compute.
        device: Device for neural inference ('cuda' for GPU, 'cpu' for CPU). Only used if use_neural=True.
        use_ml_rhythm: If True, use ML-based rhythm quantization (default: True).
                       Falls back to heuristic if model not available.
    
    For the legacy frame-by-frame pipeline, use analyze_audio_legacy().
    """
    if use_neural:
        # Neural network transcription (requires file path, not array)
        if isinstance(wav_path_or_array, str):
            return analyze_audio_neural(wav_path_or_array, debug=debug, device=device, use_ml_rhythm=use_ml_rhythm)
        else:
            return {"error": "Neural transcription requires a file path, not an array. Save to a temp file first."}
    
    if use_split:
        if independent_hands:
            return analyze_audio_independent_hands(wav_path_or_array, debug=debug, use_ml_rhythm=use_ml_rhythm)
        else:
            return analyze_audio_split_ranges(wav_path_or_array, debug=debug, use_ml_rhythm=use_ml_rhythm)
    else:
        return analyze_audio_optimized(wav_path_or_array, debug=debug, use_ml_rhythm=use_ml_rhythm)

def analyze_audio_legacy(wav_path_or_array, debug=False):
    """
    Legacy analysis pipeline (kept for backwards compatibility).
    Uses frame-by-frame FFT computation.
    For production use, prefer analyze_audio() which uses the optimized pipeline.
    """
    try:
        # 1) Load audio
        if isinstance(wav_path_or_array, str):
            audio = read_wav(wav_path_or_array)
        else:
            audio = wav_path_or_array
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
    except Exception as e:
        return {"error": f"Failed to read audio: {str(e)}"}
    
    # 2) Onset detection with noise reduction
    frames = frame_audio(audio)
    mags = np.array([compute_magnitude(f) for f in frames])
    flux = normalize(compute_flux(mags))
    onsets = find_onsets(flux)
    
    # 3) Precompute chroma & full-range CQT for chord detection
    chroma = extract_chroma(audio, SAMPLE_RATE, hop_length=HOP_SIZE)
    if USE_GPU:
        C_full = gpu_cqt(audio, sr=SAMPLE_RATE, n_bins=CQT_BINS,
                         bins_per_octave=12, fmin=librosa.note_to_hz('A0'),
                         hop_length=HOP_SIZE)
    else:
        C_full = np.abs(librosa.cqt(
            y=audio, sr=SAMPLE_RATE, hop_length=HOP_SIZE,
            n_bins=CQT_BINS, bins_per_octave=12,
            fmin=librosa.note_to_hz('A0')
        ))
    # estimate offsets from chroma (fast pre-pass) — no console logs in API path
    try:
        event_midis = []
        for f in onsets:
            if 0 <= f < chroma.shape[1]:
                pc = int(np.argmax(chroma[:, f]))
                event_midis.append(pc)
            else:
                event_midis.append(0)
        offsets_frames = estimate_offsets_from_chroma(onsets, event_midis, chroma)
    except Exception:
        offsets_frames = [(f, f+1) for f in onsets]

    # Results structure
    results = {
        "onsets": [],
        "notes": [],
        "chords": [],
        "analysis_summary": {
            "total_onsets": len(onsets),
            "duration_seconds": float(len(audio) / SAMPLE_RATE),
            "sample_rate": int(SAMPLE_RATE)
        }
    }
    
    # 5) Process each onset
    for i, onset in enumerate(onsets):
        idx = min(onset, len(frames)-1)
        frame = frames[idx]
        
        # Convert frame index to time
        time_seconds = onset * HOP_SIZE / SAMPLE_RATE
                
        # Add onset information
        # attach estimated offset and duration (from chroma analysis)
        try:
            oframe = int(offsets_frames[i][1])
            osec = round(oframe * HOP_SIZE / SAMPLE_RATE, 3)
            dur = round(osec - time_seconds, 3)
        except Exception:
            oframe = int(onset + 1)
            osec = round((onset + 1) * HOP_SIZE / SAMPLE_RATE, 3)
            dur = round(osec - time_seconds, 3)
        onset_info = {
            "time_seconds": round(time_seconds, 3),
            "frame_index": int(onset),
            "offset_frame": oframe,
            "offset_seconds": osec,
            "duration_seconds": dur
        }
        results["onsets"].append(onset_info)
        
        # 1) Build a tiny onset-centered spectrum (average 2 frames ahead for stability)
        # Also collect individual frames for temporal filtering (HMM-like approach)
        fft_mag_center = compute_magnitude(frames[idx])
        if 1 <= idx < len(frames)-1:
            fft_mag_prev   = compute_magnitude(frames[idx-1])
            fft_mag_next   = compute_magnitude(frames[idx+1])
            mag_window = (fft_mag_prev + fft_mag_center + fft_mag_next) / 3.0
            mag_frames_raw = [fft_mag_prev, fft_mag_center, fft_mag_next]
        else:
            mag_window = fft_mag_center
            mag_frames_raw = [fft_mag_center]

        # ringing cancellation
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0/SAMPLE_RATE)
        resid, updated = cancel_ringing(mag_window, freqs)
        # Apply ringing cancellation to each frame for temporal filtering
        mag_frames_resid = [cancel_ringing(mf, freqs)[0] for mf in mag_frames_raw]
        # 2) Explain it with K harmonic sources chosen by BIC (with temporal filtering)
        bic_est = estimate_voices_bic(resid, max_K=3, H=8, mag_frames=mag_frames_resid)
        K = bic_est['K']
        midi_set = bic_est['midis']
        salience_info = bic_est.get('salience_info', {})  # For selecting best note when multiple detected
        
        # For single notes, cross-validate with CQT peaks
        # For chords (K >= 2), trust BIC - CQT validation can incorrectly reject chord tones
        cqt_idx = min(onset + 2, C_full.shape[1] - 1)
        if K == 1 and midi_set and cqt_idx < C_full.shape[1]:
            cqt_frame = C_full[:, cqt_idx]
            midi_set = validate_midi_with_cqt(midi_set, cqt_frame)
            K = len(midi_set)  # Update K after validation

        is_chord_final = (K >= 2)
        
        # If NOT a chord but multiple notes detected, pick highest salience for single note
        if not is_chord_final and len(midi_set) > 1:
            midi_set = sorted(midi_set, key=lambda m: salience_info.get(m, (0,))[0], reverse=True)[:1]
            K = 1

        if is_chord_final:            
            # Use existing chord detection for labeling and inversion analysis
            res = detect_chord_multiframe(chroma, C_full, onset, num_frames=1, debug=True)
            if res is not None:
                # Add the actual MIDI notes from BIC analysis
                res["midi_notes"] = [int(m) for m in midi_set] if midi_set else []
                res.update({"time_seconds": round(time_seconds, 3), "frame_index": int(onset)})
                # copy onset offset metadata
                ofmeta = results["onsets"][i]
                res.update({"offset_seconds": ofmeta.get("offset_seconds"), "duration_seconds": ofmeta.get("duration_seconds"), "offset_frame": ofmeta.get("offset_frame")})
                results["chords"].append(res)
            else:
                # Single note → use the MIDI from BIC analysis
                m = midi_set[0] if midi_set else None
                if m is None:
                    res_note = detect_single_note_frame(frame, debug=False)
                    if res_note:
                        res_note.update({"time_seconds": round(time_seconds, 3), "frame_index": int(onset)})
                        # attach onset offset metadata
                        ofmeta = results["onsets"][i]
                        res_note.update({"offset_seconds": ofmeta.get("offset_seconds"), "duration_seconds": ofmeta.get("duration_seconds"), "offset_frame": ofmeta.get("offset_frame")})
                else:
                    note_name = note_to_name(m)
                    frequency = _midi_to_hz(m)
                    res_note = {
                        "time_seconds": round(time_seconds, 3),
                        "frame_index": int(onset),
                        "midi_note": int(m),
                        "note_name": note_name,
                        "frequency_hz": round(frequency, 2),
                        "method": "HarmonicMixture(BIC)",
                        "confidence": 0.9
                    }
                    # attach onset offset metadata
                    ofmeta = results["onsets"][i]
                    res_note.update({"offset_seconds": ofmeta.get("offset_seconds"), "duration_seconds": ofmeta.get("duration_seconds"), "offset_frame": ofmeta.get("offset_frame")})

                if res_note: 
                    results["notes"].append(res_note)
        else:            
            # Single note → use the MIDI from BIC analysis
            m = midi_set[0] if midi_set else None
            if m is None:
                res_note = detect_single_note_frame(frame, debug=False)
                if res_note:
                    res_note.update({"time_seconds": round(time_seconds, 3), "frame_index": int(onset)})
            else:
                note_name = note_to_name(m)
                frequency = _midi_to_hz(m)
                
                res_note = {
                    "time_seconds": round(time_seconds, 3),
                    "frame_index": int(onset),
                    "midi_note": int(m),
                    "note_name": note_name,
                    "frequency_hz": round(frequency, 2),
                    "method": "HarmonicMixture(BIC)",
                    "confidence": 0.9
                }

            if res_note:
                results["notes"].append(res_note)

    # Update analysis summary
    results["analysis_summary"].update({
        "total_notes": len(results["notes"]),
        "total_chords": len(results["chords"])
    })


    return results

#* ─── Command-line Analysis Function ───────────────────────────────────────────
def analyze_audio_cmdline(wav_path_or_array, use_legacy=False, use_split=True, split_midi=60, independent_hands=True, use_neural=True, device='cuda'):
    """
    Command-line focused audio analysis with both single note and chord detection.
    Includes detailed console logging of the analysis process and thresholds.
    
    Args:
        wav_path_or_array: Audio file path or numpy array
        use_legacy: Use old frame-by-frame pipeline (default: False)
        use_split: Use frequency range splitting to separate left/right hand (default: True)
        split_midi: MIDI note to split at when use_split=True (default: 60 = middle C)
        independent_hands: If True, detect bass/treble rhythms independently (default: True)
        use_neural: If True, use neural network transcription for higher accuracy (default: False)
        device: Device for neural inference - 'cuda' for GPU, 'cpu' for CPU (default: 'cpu')
    """
    # Neural network transcription (highest accuracy)
    if use_neural:
        if isinstance(wav_path_or_array, str):
            return analyze_audio_neural(wav_path_or_array, debug=True, split_midi=split_midi, device=device)
        else:
            print("ERROR: Neural transcription requires a file path, not an array.")
            return {"error": "Neural transcription requires a file path"}
    
    if not use_legacy:
        if use_split:
            if independent_hands:
                # Use independent hands analysis - bass and treble have separate onset detection
                return analyze_audio_independent_hands(wav_path_or_array, debug=True, split_midi=split_midi)
            
            # Shared onset detection with categorization
            print("\n" + "="*70)
            print("🎹 BASS/TREBLE CATEGORIZATION ENABLED (SHARED RHYTHM)")
            print(f"   Split point: MIDI {split_midi} ({440.0 * 2**((split_midi - 69) / 12):.1f} Hz)")
            print("   Bass (left hand) < split point, Treble (right hand) >= split point")
            print("   Analysis: Full spectrum with harmonic subtraction, then categorize")
            print("="*70 + "\n")
            
            # Analyze full audio with harmonic subtraction
            print("\n" + "-"*70)
            print("🎼 ANALYZING FULL AUDIO WITH HARMONIC SUBTRACTION")
            print("-"*70)
            results = analyze_audio_optimized(wav_path_or_array, debug=True)
            
            print(f"\n✓ Full audio analysis complete:")
            print(f"   Total onsets detected: {results['analysis_summary']['total_onsets']}")
            print(f"   Total notes detected: {len(results.get('notes', []))}")
            print(f"   Total chords detected: {len(results.get('chords', []))}")
            
            # Categorize notes into bass and treble
            print("\n" + "-"*70)
            print("🔀 CATEGORIZING NOTES BY RANGE")
            print("-"*70)
            
            bass_notes = []
            treble_notes = []
            
            for note in results.get("notes", []):
                if note["midi_note"] < split_midi:
                    bass_notes.append(note)
                else:
                    treble_notes.append(note)
            
            # Categorize chords
            bass_chords = []
            treble_chords = []
            
            for chord in results.get("chords", []):
                midi_notes = chord.get("midi_notes", [])
                if midi_notes:
                    lowest_midi = min(midi_notes)
                    if lowest_midi < split_midi:
                        bass_chords.append(chord)
                    else:
                        treble_chords.append(chord)
            
            # Update summary
            results["analysis_summary"]["bass_notes"] = len(bass_notes)
            results["analysis_summary"]["treble_notes"] = len(treble_notes)
            results["analysis_summary"]["bass_chords"] = len(bass_chords)
            results["analysis_summary"]["treble_chords"] = len(treble_chords)
            
            print(f"\n✓ Categorization complete:")
            print(f"   Bass notes (< MIDI {split_midi}): {len(bass_notes)}")
            if bass_notes:
                print(f"      Range: {min(n['note_name'] for n in bass_notes)} to {max(n['note_name'] for n in bass_notes)}")
            print(f"   Treble notes (>= MIDI {split_midi}): {len(treble_notes)}")
            if treble_notes:
                print(f"      Range: {min(n['note_name'] for n in treble_notes)} to {max(n['note_name'] for n in treble_notes)}")
            print(f"   Bass chords: {len(bass_chords)}")
            print(f"   Treble chords: {len(treble_chords)}")
            
            print("\n" + "="*70)
            print("📊 FINAL NOTE SEQUENCE (by time)")
            print("="*70)
            for i, note in enumerate(results["notes"][:20], 1):  # Show first 20
                hand = "🎼 Bass" if note["midi_note"] < split_midi else "🎹 Treble"
                print(f"{i:3d}. {note['time_seconds']:6.2f}s - {hand:10s} - {note['note_name']:4s} (MIDI {note['midi_note']:3d}) - {note['confidence']*100:.0f}%")
            
            if len(results["notes"]) > 20:
                print(f"     ... and {len(results['notes']) - 20} more notes")
            print("="*70 + "\n")
            
            return results
        else:
            return analyze_audio_optimized(wav_path_or_array)

    try:
        # 1) Load audio
        if isinstance(wav_path_or_array, str):
            audio = read_wav(wav_path_or_array)
            print(f"✓ Loaded audio file: {wav_path_or_array}")
        else:
            audio = wav_path_or_array
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            print("✓ Loaded audio array")
    except Exception as e:
        print(f"✗ Failed to read audio: {str(e)}")
        return {"error": f"Failed to read audio: {str(e)}"}

    print(f"Audio duration: {len(audio) / SAMPLE_RATE:.2f}s")
    
    # 2) Onset detection with noise reduction
    print("🔍 Detecting onsets...")
    frames = frame_audio(audio)
    mags = np.array([compute_magnitude(f) for f in frames])
    flux = normalize(compute_flux(mags))
    onsets = find_onsets(flux)

    
    print(f"✓ Found {len(onsets)} onsets at frames: {onsets}")
    
    # 3) Precompute chroma & full-range CQT for chord detection
    print("🎼 Computing chroma and CQT features...")
    chroma = extract_chroma(audio, SAMPLE_RATE, hop_length=HOP_SIZE)
    if USE_GPU:
        C_full = gpu_cqt(audio, sr=SAMPLE_RATE, n_bins=CQT_BINS,
                         bins_per_octave=12, fmin=librosa.note_to_hz('A0'),
                         hop_length=HOP_SIZE)
    else:
        C_full = np.abs(librosa.cqt(
            y=audio, sr=SAMPLE_RATE, hop_length=HOP_SIZE,
            n_bins=CQT_BINS, bins_per_octave=12,
            fmin=librosa.note_to_hz('A0')
        ))
    print(f"✓ Chroma shape: {chroma.shape}, CQT shape: {C_full.shape}")
    # --- estimate offsets from chroma for each onset (fast pre-pass)
    try:
        event_midis = []
        for f in onsets:
            if 0 <= f < chroma.shape[1]:
                # representative pitch-class for this event: strongest chroma bin
                pc = int(np.argmax(chroma[:, f]))
                event_midis.append(pc)
            else:
                event_midis.append(0)
        offsets_frames = estimate_offsets_from_chroma(onsets, event_midis, chroma)
    except Exception as e:
        print(f"[OFFSETS] estimate_offsets_from_chroma failed: {e}")
        offsets_frames = [(f, f+1) for f in onsets]
    
    # Results structure
    results = {
        "onsets": [],
        "notes": [],
        "chords": [],
        "analysis_summary": {
            "total_onsets": len(onsets),
            "duration_seconds": float(len(audio) / SAMPLE_RATE),
            "sample_rate": int(SAMPLE_RATE)
        }
    }
    
    # 5) Process each onset
    print("🎵 Analyzing each onset...")
    for i, onset in enumerate(onsets):
        idx = min(onset, len(frames)-1)
        frame = frames[idx]
        
        # Convert frame index to time
        time_seconds = onset * HOP_SIZE / SAMPLE_RATE
        
        print(f"\n=== ONSET {i+1}/{len(onsets)} at frame {onset} ({time_seconds:.2f}s) ===")
        
        # Add onset information
        onset_info = {
            "time_seconds": round(time_seconds, 3),
            "frame_index": int(onset)
        }
        # attach estimated offset and duration (from chroma analysis)
        try:
            oframe = int(offsets_frames[i][1])
            osec = round(oframe * HOP_SIZE / SAMPLE_RATE, 3)
            dur = round(osec - time_seconds, 3)
        except Exception:
            oframe = int(onset + 1)
            osec = round((onset + 1) * HOP_SIZE / SAMPLE_RATE, 3)
            dur = round(osec - time_seconds, 3)
        onset_info.update({"offset_frame": oframe, "offset_seconds": osec, "duration_seconds": dur})
        results["onsets"].append(onset_info)
        print(f"  [OFFSETS] estimated offset={osec}s (frame {oframe}), duration={dur}s")
        
        # 1) Build a tiny onset-centered spectrum (average 2 frames ahead for stability)
        # Also collect individual frames for temporal filtering (HMM-like approach)
        print(f"  🔬 Building onset-centered spectrum...")
        fft_mag_center = compute_magnitude(frames[idx])
        if 1 <= idx < len(frames)-1:
            fft_mag_prev   = compute_magnitude(frames[idx-1])
            fft_mag_next   = compute_magnitude(frames[idx+1])
            mag_window = (fft_mag_prev + fft_mag_center + fft_mag_next) / 3.0
            mag_frames_raw = [fft_mag_prev, fft_mag_center, fft_mag_next]
            print(f"     Using 3-frame average (frames {idx-1}-{idx+1}) for stability + temporal filtering")
        else:
            mag_window = fft_mag_center
            mag_frames_raw = [fft_mag_center]
            print(f"     Using single frame {idx} (edge case)")
        
        spectrum_energy = np.sum(mag_window)
        max_magnitude = np.max(mag_window)
        print(f"     Spectrum energy: {spectrum_energy:.2f}, max magnitude: {max_magnitude:.4f}")

        # ringing cancellation
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0/SAMPLE_RATE)
        resid, updated = cancel_ringing(mag_window, freqs)
        # Apply ringing cancellation to each frame for temporal filtering
        mag_frames_resid = [cancel_ringing(mf, freqs)[0] for mf in mag_frames_raw]
        # 2) Explain it with K harmonic sources chosen by BIC (with temporal filtering)
        print(f"  🎼 Performing BIC harmonic mixture analysis with temporal filtering...")
        bic_est = estimate_voices_bic(resid, max_K=3, H=8, mag_frames=mag_frames_resid)
        K = bic_est['K']
        midi_set = bic_est['midis']
        bic_value = bic_est['bic']
        fit_error = bic_est['err']

        is_chord_final = (K >= 2)
        
        print(f"     BIC Analysis Results:")
        print(f"     - Optimal voices (K): {K}")
        print(f"     - Detected MIDI notes: {midi_set}")
        print(f"     - Fit error: {fit_error:.4f}")
        print(f"     - BIC score: {bic_value:.2f}")
        print(f"     - Classification: {'CHORD' if is_chord_final else 'SINGLE NOTE'}")
        
        # Convert MIDI notes to note names for better readability
        if midi_set:
            note_names = [note_to_name(int(m)) for m in midi_set]
            print(f"     - Note names: {note_names}")

        if is_chord_final:
            print(f"  🎹 Analyzing as CHORD...")
            print(f"     Detected {K} simultaneous voices: {note_names}")
            
            # Convert to pitch classes for chord identification
            pcs = sorted([(m % 12) for m in midi_set])
            print(f"     Pitch classes: {pcs}")
            
            # Use existing chord detection for labeling and inversion analysis
            res = detect_chord_multiframe(chroma, C_full, onset, num_frames=1, debug=True)
            if res is not None:
                # Add the actual MIDI notes from BIC analysis
                res["midi_notes"] = [int(m) for m in midi_set] if midi_set else []
                res.update({"time_seconds": round(time_seconds, 3), "frame_index": int(onset)})
                # copy onset offset metadata
                ofmeta = results["onsets"][i]
                res.update({"offset_seconds": ofmeta.get("offset_seconds"), "duration_seconds": ofmeta.get("duration_seconds"), "offset_frame": ofmeta.get("offset_frame")})
                results["chords"].append(res)
                print(f"     ✓ Added chord: {res['label']} ({res['inversion']} inversion)")
            else:
                print(f"     ✗ Chord detection failed despite BIC indicating polyphony")
                # go to single note detection
                print(f"  🎵 Analyzing as SINGLE NOTE instead of CHORD...")
            
                # Single note → use the MIDI from BIC analysis
                m = midi_set[0] if midi_set else None
                if m is None:
                    print(f"     BIC didn't detect any notes, falling back to robust detection...")
                    res_note = detect_single_note_frame(frame, debug=False)
                    if res_note:
                        res_note.update({"time_seconds": round(time_seconds, 3), "frame_index": int(onset)})
                        # copy onset offset metadata
                        ofmeta = results["onsets"][i]
                        res_note.update({"offset_seconds": ofmeta.get("offset_seconds"), "duration_seconds": ofmeta.get("duration_seconds"), "offset_frame": ofmeta.get("offset_frame")})
                        print(f"     ✓ Fallback detection: {res_note['note_name']} ({res_note['method']})")
                    else:
                        print(f"     ✗ No note detected by any method")
                else:
                    note_name = note_to_name(m)
                    frequency = _midi_to_hz(m)
                    print(f"     BIC detected: {note_name} (MIDI {m}, {frequency:.1f}Hz)")
                    
                    res_note = {
                        "time_seconds": round(time_seconds, 3),
                        "frame_index": int(onset),
                        "midi_note": int(m),
                        "note_name": note_name,
                        "frequency_hz": round(frequency, 2),
                        "method": "HarmonicMixture(BIC)",
                        "confidence": 0.9
                    }
                    print(f"     ✓ Added note with high confidence")

                if res_note:
                    # attach onset offset metadata
                    ofmeta = results["onsets"][i]
                    res_note.update({"offset_seconds": ofmeta.get("offset_seconds"), "duration_seconds": ofmeta.get("duration_seconds"), "offset_frame": ofmeta.get("offset_frame")})
                    results["notes"].append(res_note)

        else:
            print(f"  🎵 Analyzing as SINGLE NOTE...")
            
            # Single note → use the MIDI from BIC analysis
            m = midi_set[0] if midi_set else None
            if m is None:
                print(f"     BIC didn't detect any notes, falling back to robust detection...")
                res_note = detect_single_note_frame(frame, debug=False)
                if res_note:
                    res_note.update({"time_seconds": round(time_seconds, 3), "frame_index": int(onset)})
                    print(f"     ✓ Fallback detection: {res_note['note_name']} ({res_note['method']})")
                else:
                    print(f"     ✗ No note detected by any method")
            else:
                note_name = note_to_name(m)
                frequency = _midi_to_hz(m)
                print(f"     BIC detected: {note_name} (MIDI {m}, {frequency:.1f}Hz)")
                
                res_note = {
                    "time_seconds": round(time_seconds, 3),
                    "frame_index": int(onset),
                    "midi_note": int(m),
                    "note_name": note_name,
                    "frequency_hz": round(frequency, 2),
                    "method": "HarmonicMixture(BIC)",
                    "confidence": 0.9
                }
                print(f"     ✓ Added note with high confidence")

            if res_note: 
                ofmeta = results["onsets"][i]
                res_note.update({
                    "offset_seconds": ofmeta.get("offset_seconds"),
                    "duration_seconds": ofmeta.get("duration_seconds"),
                    "offset_frame": ofmeta.get("offset_frame")
                })
                results["notes"].append(res_note)
        
        print(f"   🎯 FINAL DECISION: {'CHORD' if is_chord_final else 'SINGLE NOTE'}")
        print(f"      Method: BIC Harmonic Mixture Analysis (K={K})")
        if midi_set:
            print(f"      Detected: {note_names}")
        else:
            print(f"      No clear musical content detected")

    # Update analysis summary
    results["analysis_summary"].update({
        "total_notes": len(results["notes"]),
        "total_chords": len(results["chords"])
    })
    
    print(f"\n🎼 Analysis Complete!")
    print(f"   Total onsets: {len(results['onsets'])}")
    print(f"   Notes detected: {len(results['notes'])}")
    print(f"   Chords detected: {len(results['chords'])}")
    
    return results


#* ─── Second Pass: Soft Note Gap-Fill Detection ─────────────────────────────
def second_pass_gap_fill(wav_path_or_array, existing_notes, existing_chords,
                          min_gap_seconds=0.25, soft_K=1.2, debug=False):
    """
    Second pass detection to find soft notes that were missed in gaps.
    
    This uses a lower onset detection threshold (K=1.2 instead of K=2.0)
    but only searches within gaps between existing notes.
    
    Args:
        wav_path_or_array: Audio file path or numpy array
        existing_notes: List of notes from first pass
        existing_chords: List of chords from first pass  
        min_gap_seconds: Minimum gap duration to search for soft notes (default: 0.25s)
        soft_K: Softer threshold for onset detection (default: 1.2 std devs)
        debug: Print debug info
        
    Returns:
        dict with 'notes' and 'chords' containing only the NEW detections
    """
    print(f"\n🔍 Second Pass: Searching for soft notes in gaps (K={soft_K})")
    
    # Load audio
    try:
        if isinstance(wav_path_or_array, str):
            audio = read_wav(wav_path_or_array)
        else:
            audio = wav_path_or_array
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
    except Exception as e:
        return {"error": f"Failed to read audio: {str(e)}", "notes": [], "chords": []}
    
    duration_seconds = len(audio) / SAMPLE_RATE
    
    # Combine notes and chords to find all covered time ranges
    all_events = []
    for n in existing_notes:
        t = n.get('time_seconds', 0)
        d = n.get('duration_seconds', 0.1)
        all_events.append((t, t + d))
    
    for c in existing_chords:
        t = c.get('time_seconds', 0)
        d = c.get('duration_seconds', 0.1)
        all_events.append((t, t + d))
    
    # Sort by start time and find gaps
    all_events.sort(key=lambda x: x[0])
    
    gaps = []
    prev_end = 0.0
    for start, end in all_events:
        if start > prev_end + min_gap_seconds:
            gaps.append((prev_end, start))
        prev_end = max(prev_end, end)
    
    # Check gap at the end
    if duration_seconds > prev_end + min_gap_seconds:
        gaps.append((prev_end, duration_seconds))
    
    if debug:
        print(f"[Second Pass] Found {len(gaps)} gaps >= {min_gap_seconds}s")
        for i, (gs, ge) in enumerate(gaps):
            print(f"  Gap {i+1}: {gs:.2f}s - {ge:.2f}s ({ge-gs:.2f}s)")
    
    if not gaps:
        print("[Second Pass] No significant gaps found")
        return {"notes": [], "chords": []}
    
    # Compute full spectral flux with softer onset detection
    stft_data, magnitude, phase, freqs = compute_stft_once(audio)
    flux = compute_flux_from_magnitude(magnitude)
    flux = normalize(flux)
    
    # Find onsets with lower threshold (more sensitive)
    soft_onsets = find_onsets(flux, window=50, K=soft_K)
    
    if debug:
        print(f"[Second Pass] Found {len(soft_onsets)} soft onsets with K={soft_K}")
    
    # Filter to only onsets within gaps
    gap_onsets = []
    for onset_frame in soft_onsets:
        time_sec = onset_frame * HOP_SIZE / SAMPLE_RATE
        for gap_start, gap_end in gaps:
            if gap_start <= time_sec <= gap_end:
                gap_onsets.append(onset_frame)
                break
    
    if debug:
        print(f"[Second Pass] {len(gap_onsets)} onsets fall within gaps")
    
    if not gap_onsets:
        print("[Second Pass] No new onsets in gaps")
        return {"notes": [], "chords": []}
    
    # Compute CQT for pitch detection
    if USE_GPU:
        C_full = gpu_cqt(audio, sr=SAMPLE_RATE, n_bins=CQT_BINS,
                         bins_per_octave=12, fmin=librosa.note_to_hz('A0'),
                         hop_length=HOP_SIZE)
    else:
        C_full = np.abs(librosa.cqt(
            y=audio, sr=SAMPLE_RATE, hop_length=HOP_SIZE,
            n_bins=CQT_BINS, bins_per_octave=12,
            fmin=librosa.note_to_hz('A0')
        ))
    
    # Process each gap onset
    new_notes = []
    new_chords = []
    
    for i, onset_frame in enumerate(gap_onsets):
        time_seconds = onset_frame * HOP_SIZE / SAMPLE_RATE
        
        if debug:
            print(f"\n  [Gap Onset {i+1}] at {time_seconds:.3f}s")
        
        # Get CQT slice for pitch detection
        cqt_idx = min(onset_frame + 1, C_full.shape[1] - 1)
        cqt_slice = C_full[:, cqt_idx]
        
        # Detect pitches using HPS (Harmonic Product Spectrum)
        midi_notes = pick_pitches_HPS(cqt_slice, max_voices=4, max_h=5)
        
        if not midi_notes:
            if debug:
                print(f"    No pitches detected")
            continue
        
        # Estimate duration (simple heuristic: until next onset or gap end)
        next_event_time = duration_seconds
        for gap_start, gap_end in gaps:
            if gap_start <= time_seconds <= gap_end:
                next_event_time = gap_end
                break
        
        for j, next_onset in enumerate(gap_onsets):
            next_time = next_onset * HOP_SIZE / SAMPLE_RATE
            if next_time > time_seconds:
                next_event_time = min(next_event_time, next_time)
                break
        
        duration = min(next_event_time - time_seconds, 2.0)  # Cap at 2 seconds
        duration = max(duration, 0.05)  # Minimum 50ms
        
        # Determine hand (bass/treble based on pitch)
        avg_midi = sum(midi_notes) / len(midi_notes)
        hand = "bass" if avg_midi < 60 else "treble"
        
        # Helper to convert MIDI to note name
        def _midi_to_name(m):
            names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
            return names[m % 12] + str(m // 12 - 1)
        
        if len(midi_notes) == 1:
            # Single note
            midi = midi_notes[0]
            note_name = _midi_to_name(midi)
            freq = 440.0 * (2 ** ((midi - 69) / 12))
            
            new_notes.append({
                'time_seconds': float(time_seconds),
                'midi_note': int(midi),
                'note_name': note_name,
                'frequency_hz': float(freq),
                'duration_seconds': float(duration),
                'offset_seconds': float(time_seconds + duration),
                'hand': hand,
                'method': 'second_pass_soft',
                'confidence': 0.6,  # Lower confidence for second pass
            })
            
            if debug:
                print(f"    → Note: {note_name} (MIDI {midi}) dur={duration:.2f}s")
        
        else:
            # Multiple pitches = chord
            note_names = [_midi_to_name(m) for m in midi_notes]
            
            new_chords.append({
                'time_seconds': float(time_seconds),
                'midi_notes': [int(m) for m in midi_notes],
                'note_names': note_names,
                'duration_seconds': float(duration),
                'offset_seconds': float(time_seconds + duration),
                'hand': hand,
                'method': 'second_pass_soft',
                'confidence': 0.5,
            })
            
            if debug:
                print(f"    → Chord: {note_names} dur={duration:.2f}s")
    
    print(f"\n✅ Second Pass Complete: Found {len(new_notes)} notes, {len(new_chords)} chords")
    
    return {
        "notes": new_notes,
        "chords": new_chords,
    }


#* ─── Main Pipeline ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Use absolute path to audio file
    wav_path = os.path.join(os.path.dirname(__file__), 'audio', "test_fugue1_cmajor.wav")
    print(f"🎹 Piano Note Detection - Command Line")
    print(f"Reading audio from: {wav_path}")
    try:
        audio = read_wav(wav_path)
    except Exception as e:
        print(f"Failed to open audio file: {e}")
        exit()
    
    results = analyze_audio_cmdline(wav_path, use_legacy=False, use_split=True, independent_hands=True, use_neural=True, device='cuda')

    if "error" not in results:
        print("\n" + "="*50)
        print("FINAL RESULTS:")
        print("="*50)
        
        # Print notes
        if results["notes"]:
            print("NOTES:")
            for note in results["notes"]:
                off = note.get("offset_seconds")
                dur = note.get("duration_seconds")
                off_str = f"{off:.2f}s" if off is not None else "N/A"
                dur_str = f"{dur:.2f}s" if dur is not None else "N/A"
                print(f"  {note['time_seconds']:6.2f}s -> {off_str} (dur {dur_str}): {note['note_name']:>4} ({note['frequency_hz']:6.1f}Hz) - {note['method']}")
        
        # Print chords
        if results["chords"]:
            print("\nCHORDS:")
            for chord in results["chords"]:
                off = chord.get("offset_seconds")
                dur = chord.get("duration_seconds")
                off_str = f"{off:.2f}s" if off is not None else "N/A"
                dur_str = f"{dur:.2f}s" if dur is not None else "N/A"
                label = chord.get('label', 'unknown')
                octave = chord.get('octave', '?')
                inversion = chord.get('inversion', 'unknown')
                confidence = chord.get('confidence', 0.0)
                hand = chord.get('hand', '')
                note_names = chord.get('note_names', [])
                print(f"  {chord['time_seconds']:6.2f}s -> {off_str} (dur {dur_str}): {label:>12} oct {octave} ({inversion}) [{hand}] conf={confidence:.2f} - {note_names}")
        
        if not results["notes"] and not results["chords"]:
            print("  No notes or chords detected")
        
        if not results["notes"] and not results["chords"]:
            print("  No notes or chords detected")
            print("  No notes or chords detected")
            print("  No notes or chords detected")
