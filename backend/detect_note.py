import os
import warnings

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

def duration_to_note_value(duration_seconds, bpm=120):
    """
    Convert duration in seconds to a note value based on tempo.
    Supports: whole, half, quarter, eighth, 16th, 32nd notes and their dotted versions.
    
    Args:
        duration_seconds: Duration of the note in seconds
        bpm: Beats per minute (default 120)
    
    Returns:
        dict with 'type' (MusicXML note type), 'divisions' (duration in divisions),
        'beats' (duration in beats), and 'dotted' (boolean)
    """
    # Calculate beat duration in seconds
    beat_duration = 60.0 / bpm  # Duration of one quarter note
    
    # Calculate how many beats this note is
    beats = duration_seconds / beat_duration
    
    # Note values in beats (from longest to shortest)
    # Format: (min_threshold, max_threshold, type, beats, dotted)
    # Thresholds use midpoints between adjacent note values for best matching
    note_values = [
        # Dotted whole (6 beats) - threshold: > 5 beats
        (5.0, float('inf'), 'whole', 6.0, True),
        # Whole (4 beats) - threshold: 3.5 to 5 beats
        (3.5, 5.0, 'whole', 4.0, False),
        # Dotted half (3 beats) - threshold: 2.75 to 3.5 beats
        (2.75, 3.5, 'half', 3.0, True),
        # Half (2 beats) - threshold: 1.75 to 2.75 beats
        (1.75, 2.75, 'half', 2.0, False),
        # Dotted quarter (1.5 beats) - threshold: 1.25 to 1.75 beats
        (1.25, 1.75, 'quarter', 1.5, True),
        # Quarter (1 beat) - threshold: 0.875 to 1.25 beats
        (0.875, 1.25, 'quarter', 1.0, False),
        # Dotted eighth (0.75 beats) - threshold: 0.625 to 0.875 beats
        (0.625, 0.875, 'eighth', 0.75, True),
        # Eighth (0.5 beats) - threshold: 0.4375 to 0.625 beats
        (0.4375, 0.625, 'eighth', 0.5, False),
        # Dotted 16th (0.375 beats) - threshold: 0.3125 to 0.4375 beats
        (0.3125, 0.4375, '16th', 0.375, True),
        # 16th (0.25 beats) - threshold: 0.1875 to 0.3125 beats
        (0.1875, 0.3125, '16th', 0.25, False),
        # Dotted 32nd (0.1875 beats) - threshold: 0.15625 to 0.1875 beats
        (0.15625, 0.1875, '32nd', 0.1875, True),
        # 32nd (0.125 beats) - threshold: < 0.15625 beats
        (0, 0.15625, '32nd', 0.125, False),
    ]
    
    for min_thresh, max_thresh, note_type, note_beats, dotted in note_values:
        if min_thresh <= beats < max_thresh:
            return {
                'type': note_type,
                'divisions': note_beats,
                'beats': note_beats,
                'dotted': dotted
            }
    
    # Fallback to 32nd note for very short durations
    return {'type': '32nd', 'divisions': 0.125, 'beats': 0.125, 'dotted': False}


def detect_tempo_from_onsets(onset_times, min_bpm=50, max_bpm=200):
    """
    Detect tempo from onset times using Inter-Onset Interval (IOI) analysis.
    
    This uses a histogram-based approach to find the most common beat interval.
    Since the user can adjust by 2x or 0.5x, we only need to get within that range.
    
    Args:
        onset_times: List/array of onset times in seconds
        min_bpm: Minimum BPM to consider (default 50)
        max_bpm: Maximum BPM to consider (default 200)
    
    Returns:
        dict with 'bpm' (detected tempo), 'confidence' (0-1), 'beat_interval' (seconds)
    """
    if len(onset_times) < 3:
        # Not enough onsets to detect tempo
        return {'bpm': 120, 'confidence': 0.0, 'beat_interval': 0.5}
    
    onset_times = np.sort(np.array(onset_times))
    
    # Calculate Inter-Onset Intervals (IOIs)
    iois = np.diff(onset_times)
    
    # Filter out very short intervals (likely grace notes or detection errors)
    # and very long intervals (likely rests or held notes)
    min_interval = 60.0 / max_bpm  # e.g., 0.3s at 200 BPM
    max_interval = 60.0 / min_bpm  # e.g., 1.2s at 50 BPM
    
    valid_iois = iois[(iois >= min_interval * 0.5) & (iois <= max_interval * 2)]
    
    if len(valid_iois) < 2:
        return {'bpm': 120, 'confidence': 0.0, 'beat_interval': 0.5}
    
    # Use histogram to find the most common interval
    # Bin width of ~25ms gives good resolution while clustering similar intervals
    bin_width = 0.025
    bins = np.arange(min_interval * 0.5, max_interval * 2 + bin_width, bin_width)
    hist, bin_edges = np.histogram(valid_iois, bins=bins)
    
    # Also consider half and double intervals (for eighth notes / half notes)
    # This helps when the piece uses mostly eighth notes or half notes
    half_iois = valid_iois / 2
    double_iois = valid_iois * 2
    
    # Filter extended IOIs to valid range
    half_iois = half_iois[(half_iois >= min_interval) & (half_iois <= max_interval)]
    double_iois = double_iois[(double_iois >= min_interval) & (double_iois <= max_interval)]
    
    # Create weighted histogram including half/double intervals
    all_candidate_iois = np.concatenate([
        valid_iois[(valid_iois >= min_interval) & (valid_iois <= max_interval)],
        half_iois,
        double_iois
    ])
    
    if len(all_candidate_iois) < 2:
        # Fallback: use median of valid IOIs
        beat_interval = float(np.median(valid_iois))
        bpm = 60.0 / beat_interval
        bpm = np.clip(bpm, min_bpm, max_bpm)
        return {'bpm': round(bpm), 'confidence': 0.3, 'beat_interval': beat_interval}
    
    # Histogram on all candidates
    hist_all, _ = np.histogram(all_candidate_iois, bins=bins)
    
    # Find the peak (most common interval)
    peak_idx = np.argmax(hist_all)
    peak_interval = (bin_edges[peak_idx] + bin_edges[peak_idx + 1]) / 2
    
    # Refine by taking weighted average around the peak
    peak_start = max(0, peak_idx - 2)
    peak_end = min(len(hist_all), peak_idx + 3)
    weights = hist_all[peak_start:peak_end]
    intervals = (bin_edges[peak_start:peak_end] + bin_edges[peak_start+1:peak_end+1]) / 2
    
    if np.sum(weights) > 0:
        refined_interval = np.average(intervals, weights=weights)
    else:
        refined_interval = peak_interval
    
    # Calculate BPM
    bpm = 60.0 / refined_interval
    
    # Snap to "nice" BPM values (multiples of 2 or 5)
    # Common tempos: 60, 72, 80, 90, 100, 108, 120, 132, 140, 160, 180
    nice_bpms = [50, 54, 58, 60, 63, 66, 69, 72, 76, 80, 84, 88, 92, 96, 
                 100, 104, 108, 112, 116, 120, 126, 132, 138, 144, 150, 
                 156, 160, 168, 176, 184, 192, 200]
    
    # Find closest nice BPM
    closest_bpm = min(nice_bpms, key=lambda x: abs(x - bpm))
    
    # Only snap if we're close (within 5%)
    if abs(closest_bpm - bpm) / bpm < 0.05:
        bpm = closest_bpm
    else:
        bpm = round(bpm)
    
    # Ensure BPM is in valid range
    bpm = int(np.clip(bpm, min_bpm, max_bpm))
    
    # Calculate confidence based on how concentrated the histogram peak is
    peak_count = hist_all[peak_idx]
    total_count = np.sum(hist_all)
    confidence = min(1.0, (peak_count / total_count) * 3)  # Scale so 33% concentration = 1.0
    
    # Boost confidence if half/double intervals also cluster at the same beat
    beat_interval = 60.0 / bpm
    
    print(f"[Tempo] Detected {bpm} BPM (beat = {beat_interval:.3f}s, confidence = {confidence:.2f})")
    print(f"[Tempo] IOI stats: {len(valid_iois)} intervals, median={np.median(valid_iois):.3f}s, mean={np.mean(valid_iois):.3f}s")
    
    return {
        'bpm': bpm,
        'confidence': round(confidence, 2),
        'beat_interval': round(beat_interval, 4)
    }


def detect_triplets(notes, bpm=120, tolerance=0.15):
    """
    Detect triplet patterns in a sequence of notes.
    Triplets are EXACTLY 3 notes played in the time of 2 regular notes.
    
    STRICT REQUIREMENTS:
    1. Must have exactly 3 consecutive notes
    2. Both inter-note spacings must be nearly equal to each other
    3. Both spacings must match a known triplet pattern for the tempo
    4. The note before (if any) must NOT have the same spacing
    5. The note after (if any) must NOT have the same spacing
    6. All 3 notes must be valid (have time_seconds)
    
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
        
        # Get all 3 notes
        note0 = notes[i]
        note1 = notes[i + 1]
        note2 = notes[i + 2]
        
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
            
            # VALIDATION: Note BEFORE must NOT have the same spacing (ensures start of triplet)
            if i > 0:
                t_prev = notes[i - 1].get('time_seconds')
                if t_prev is not None:
                    spacing_before = t0 - t_prev
                    if spacing_before > 0 and abs(spacing_before - expected_spacing) <= tol:
                        # Previous note has same spacing - we're in the middle of something
                        continue
            
            # VALIDATION: Note AFTER must NOT have the same spacing (ensures end of triplet)
            if i + 3 < len(notes):
                t_next = notes[i + 3].get('time_seconds')
                if t_next is not None:
                    spacing_after = t_next - t2
                    if spacing_after > 0 and abs(spacing_after - expected_spacing) <= tol:
                        # Next note has same spacing - this is more than 3 notes
                        continue
            
            # ALL VALIDATIONS PASSED - This is a valid triplet of exactly 3 notes
            triplet_assigned.add(i)
            triplet_assigned.add(i + 1)
            triplet_assigned.add(i + 2)
            
            triplet_beats = {
                'half': 4/3,
                'quarter': 2/3,
                'eighth': 1/3,
                '16th': 1/6,
                '32nd': 1/12,
            }[triplet_type]
            
            # Mark all 3 notes
            note0.update({
                'triplet': True,
                'triplet_position': 'start',
                'triplet_type': triplet_type,
                'actual_notes': 3,
                'normal_notes': 2,
                'note_value': triplet_type,
                'note_divisions': triplet_beats,
                'dotted': False
            })
            note1.update({
                'triplet': True,
                'triplet_position': 'middle',
                'triplet_type': triplet_type,
                'actual_notes': 3,
                'normal_notes': 2,
                'note_value': triplet_type,
                'note_divisions': triplet_beats,
                'dotted': False
            })
            note2.update({
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
            i += 3  # Skip past the triplet
            break
        
        if not matched:
            i += 1
    
    return notes


def detect_triplets_in_chords(chords, bpm=120, tolerance=0.15):
    """
    Detect triplet patterns in a sequence of chords.
    
    STRICT REQUIREMENTS (same as detect_triplets):
    1. Must have exactly 3 consecutive chords
    2. Both inter-chord spacings must be nearly equal to each other
    3. Both spacings must match a known triplet pattern for the tempo
    4. The chord before (if any) must NOT have the same spacing
    5. The chord after (if any) must NOT have the same spacing
    6. All 3 chords must have valid time_seconds
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
            
            # VALIDATION: Chord BEFORE must NOT have the same spacing (ensures start of triplet)
            if i > 0:
                t_prev = chords[i - 1].get('time_seconds')
                if t_prev is not None:
                    spacing_before = t0 - t_prev
                    if spacing_before > 0 and abs(spacing_before - expected_spacing) <= tol:
                        # Previous chord has same spacing - not start of triplet
                        continue
            
            # VALIDATION: Chord AFTER must NOT have the same spacing (ensures end of triplet)
            if i + 3 < len(chords):
                t_next = chords[i + 3].get('time_seconds')
                if t_next is not None:
                    spacing_after = t_next - t2
                    if spacing_after > 0 and abs(spacing_after - expected_spacing) <= tol:
                        # Next chord has same spacing - more than 3 notes
                        continue
            
            # ALL VALIDATIONS PASSED - This is a valid triplet of exactly 3 chords
            triplet_assigned.add(i)
            triplet_assigned.add(i + 1)
            triplet_assigned.add(i + 2)
            
            triplet_beats = {
                'half': 4/3,
                'quarter': 2/3,
                'eighth': 1/3,
                '16th': 1/6,
                '32nd': 1/12,
            }[triplet_type]
            
            for idx, pos in [(i, 'start'), (i + 1, 'middle'), (i + 2, 'end')]:
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
            i += 3
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
    
    scipy.signal.stft() doesn't perfectly replicate manual windowing + FFT.
    Solution: Manually replicate the exact same operations in vectorized form.
    """
    # Use the SAME window as frame_audio()
    window = get_window(WINDOW_TYPE, n_fft, fftbins=True)
    
    # Manual framing to match frame_audio() exactly
    num_frames = 1 + (len(audio) - n_fft) // hop_length
    
    # Pre-allocate STFT matrix (use complex128 to match np.fft.rfft default precision)
    stft_data = np.zeros((n_fft // 2 + 1, num_frames), dtype=np.complex128)
    
    # Compute FFT for each frame (matching compute_magnitude exactly)
    for i in range(num_frames):
        frame = audio[i * hop_length : i * hop_length + n_fft]
        windowed_frame = frame * window
        stft_data[:, i] = np.fft.rfft(windowed_frame, n=n_fft)
    
    # Keep float64 precision to match regular pipeline (compute_magnitude returns float64)
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
    
    # Step 0: Remove persistent background tones (HVAC, 60Hz hum, room resonance)
    # These are constant-pitch noise sources that stay on throughout the recording
    audio, persistent_db = remove_persistent_tones(
        audio, sr=sr, n_fft=FFT_SIZE, hop_length=HOP_SIZE,
        persistence_percentile=10,  # 10th percentile = floor present in 90% of frames
        subtraction_strength=0.8  # Subtract 80% of persistent floor
    )
    print(f"[Noise Pipeline] After persistent tone removal: {persistent_db:.2f} dB removed")
    
    # Step 1: Multi-band spectral gate (our improved implementation)
    # This handles frequency-specific noise (rumble, hiss) with transient preservation
    audio, noise_removed_db = multiband_spectral_gate(
        audio, sr=sr, n_fft=FFT_SIZE, hop_length=HOP_SIZE,
        noise_estimation_seconds=0.15,  # Use 150ms for noise estimation
        gate_threshold_db=-10,  # dB above noise floor to keep
        min_gate_threshold_db=-50  # Absolute minimum threshold
    )
    print(f"[Noise Pipeline] After multiband spectral gate: {noise_removed_db:.2f} dB noise removed")
    
    # Step 2: noisereduce for residual non-stationary noise
    # Use less aggressive settings since spectral gate already did heavy lifting
    audio_before = audio.copy()
    audio = nr.reduce_noise(
        y=audio, sr=sr, 
        stationary=False, 
        n_fft=FFT_SIZE, 
        hop_length=HOP_SIZE, 
        prop_decrease=0.6  # Less aggressive (was 0.8)
    )
    rms_reduction = np.sqrt(np.mean(audio_before**2)) - np.sqrt(np.mean(audio**2))
    print(f"[Noise Pipeline] After noisereduce: RMS reduction = {rms_reduction:.4f}")
    
    # Step 3: High-pass filter to remove any remaining sub-bass rumble
    # Use a proper Butterworth HPF instead of simple one-pole
    sos = butter(2, 30, btype='high', fs=sr, output='sos')  # 30 Hz cutoff
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
            if fundamental_mag < 0.05 * max_mag:  # Skip very weak fundamentals
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
        
        # More stringent threshold for additional voices
        min_threshold = np.log(0.08 * max_mag) if voice == 0 else np.log(0.4 * max_mag)
        
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
    mag_thresh = 0.1 * np.max(mag)
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

def estimate_voices_bic(mag_window, max_K=3, H=8, debug=False):
    """
    mag_window: 1D FFT magnitude you'd like to explain (ideally averaged over ±1 frame around the onset).
    Returns: dict with {'K', 'midis', 'gains', 'bic', 'err'}
    """
    # Normalize the target spectrum so BIC compares apples to apples
    x = mag_window.astype(np.float32).copy()
    if x.max() > 0:
        x /= (x.max() + 1e-12)
    B = len(x)

    # Propose ~8 MIDI candidates via FFT salience (fast, no thresholds)
    cand_results = _salience_candidates_from_fft(x, top=12, H=H)  # Get more candidates for filtering
    
    # Build salience info dict for CQT validation: MIDI -> (score, has_peak)
    salience_info = {midi: (score, has_peak) for midi, score, fund, peak_bonus, harm, subharm, has_peak in cand_results}
    
    if debug:
        print(f"\n  [Salience] Top candidates from FFT:")
        for midi, score, fund, peak_bonus, harm, subharm, has_peak in cand_results[:8]:
            note_name = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'][midi % 12] + str(midi // 12 - 1)
            f0 = 440.0 * 2**((midi - 69)/12)
            print(f"    {note_name:4s} (MIDI {midi:3d}, {f0:6.1f}Hz): score={score:.4f} "
                  f"[fund={fund:.4f}, peak_bonus={peak_bonus:.4f}, harm={harm:.4f}, subharm_penalty={subharm:.4f}, has_peak={has_peak}]")
    
    cand_midis = [r[0] for r in cand_results]  # Extract just MIDI numbers
    
    # Keep top 8 candidates (no octave filtering - let CQT validation handle it)
    cand_midis = cand_midis[:8]

    best = {'K': 0, 'midis': [], 'gains': np.array([]), 'bic': _bic(np.sum(x*x), B, 0), 'err': float(np.sum(x*x))}
    # Try K=1..max_K by taking top-K candidates; refine by pruning tiny gains
    for K in range(1, max_K+1):
        midis = cand_midis[:K]
        gains, err = _fit_nonneg_mixture(x, midis, iters=6)
        # prune near-zero components and recompute (optional)
        keep = gains > (0.05 * gains.max())  # Increased from 0.02 to 0.05 for stricter pruning
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
    threshold = 0.15 * max_cqt
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
def analyze_audio_optimized(wav_path_or_array, debug=False):
    """
    Optimized analysis: compute STFT and CQT once, reuse everywhere.
    ~4x faster than standard pipeline with identical accuracy.
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
        if 1 <= onset_frame < magnitude.shape[1] - 1:
            mag_window = np.mean(magnitude[:, onset_frame-1:onset_frame+2], axis=1)
        else:
            mag_window = magnitude[:, min(onset_frame, magnitude.shape[1]-1)]
        
        # Ringing cancellation + BIC voice estimation
        resid, _ = cancel_ringing(mag_window, freqs)
        bic_est = estimate_voices_bic(resid, max_K=3, H=8, debug=debug)
        
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
    detected_bpm = tempo_info['bpm']
    
    # Add tempo info to results
    results["analysis_summary"]["detected_bpm"] = detected_bpm
    results["analysis_summary"]["tempo_confidence"] = tempo_info['confidence']
    results["analysis_summary"]["beat_interval"] = tempo_info['beat_interval']
    
    # Add note values based on duration (using detected BPM)
    for note in results["notes"]:
        note_val = duration_to_note_value(note.get("duration_seconds", 0.5), bpm=detected_bpm)
        note["note_value"] = note_val["type"]
        note["note_divisions"] = note_val["divisions"]
        note["dotted"] = note_val["dotted"]
    
    for chord in results["chords"]:
        note_val = duration_to_note_value(chord.get("duration_seconds", 0.5), bpm=detected_bpm)
        chord["note_value"] = note_val["type"]
        chord["note_divisions"] = note_val["divisions"]
        chord["dotted"] = note_val["dotted"]
    
    # Detect triplets (must be after regular note values are assigned)
    # Sort by time for triplet detection
    results["notes"] = sorted(results["notes"], key=lambda x: x.get("time_seconds", 0))
    results["chords"] = sorted(results["chords"], key=lambda x: x.get("time_seconds", 0))
    
    # Apply triplet detection (modifies notes/chords in place) - use detected BPM
    detect_triplets(results["notes"], bpm=detected_bpm, tolerance=0.20)
    detect_triplets_in_chords(results["chords"], bpm=detected_bpm, tolerance=0.20)
    
    # Update summary
    results["analysis_summary"].update({
        "total_notes": len(results["notes"]),
        "total_chords": len(results["chords"])
    })
    
    return results


#* ─── Independent Two-Hands Analysis ─────────────────────────────────────────
def analyze_audio_independent_hands(wav_path_or_array, debug=False, split_midi=60):
    """
    Analyze audio with INDEPENDENT onset detection for bass and treble hands.
    
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
    
    # Bass band: Focus on low-frequency rumble
    bass_audio, bass_nr_db = multiband_spectral_gate(
        bass_audio, sr=SAMPLE_RATE, n_fft=FFT_SIZE, hop_length=HOP_SIZE,
        noise_estimation_seconds=0.15,
        gate_threshold_db=-8,  # Less aggressive - bass notes are sustained
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
        gate_threshold_db=-10,  # Slightly more aggressive for hiss
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
    bass_frames = frame_audio(bass_audio)
    bass_mags = np.array([compute_magnitude(f) for f in bass_frames])
    bass_flux = normalize(compute_flux(bass_mags))
    # Use slope validation for bass - helps filter noise-induced false onsets
    # min_slope_ratio=0.3 means the onset must rise to at least 30% above baseline
    bass_onsets = find_onsets_with_slope_validation(
        bass_flux, K=2.5, min_slope_ratio=0.3, slope_window=3, debug=debug
    )
    print(f"[Bass] Found {len(bass_onsets)} validated onsets")
    
    print(f"\n[Treble] Detecting onsets in treble band...")
    treble_frames = frame_audio(treble_audio)
    treble_mags = np.array([compute_magnitude(f) for f in treble_frames])
    treble_flux = normalize(compute_flux(treble_mags))
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
        if 1 <= onset_frame < magnitude.shape[1] - 1:
            mag_window = np.mean(magnitude[:, onset_frame-1:onset_frame+2], axis=1)
        else:
            mag_window = magnitude[:, min(onset_frame, magnitude.shape[1]-1)]
        
        # Ringing cancellation + BIC voice estimation
        resid, _ = cancel_ringing(mag_window, freqs)
        bic_est = estimate_voices_bic(resid, max_K=6, H=8, debug=debug)  # Allow more voices for chords
        
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
        note_val = duration_to_note_value(dur)
        
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
    
    # Add note values to chords (using detected BPM)
    for chord in results["chords"]:
        if "note_value" not in chord:
            note_val = duration_to_note_value(chord.get("duration_seconds", 0.5), bpm=detected_bpm)
            chord["note_value"] = note_val["type"]
            chord["note_divisions"] = note_val["divisions"]
            chord["dotted"] = note_val["dotted"]
    
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


def analyze_audio_split_ranges(wav_path_or_array, debug=False, split_midi=60):
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
    
    Returns:
        Results with notes categorized by bass/treble range
    """
    # 1) Analyze full audio with harmonic subtraction
    print(f"[Split Analysis] Analyzing full audio with harmonic subtraction...")
    results = analyze_audio_optimized(wav_path_or_array, debug=debug)
    
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

def analyze_audio(wav_path_or_array, debug=False, use_split=True, independent_hands=True):
    """
    Main audio analysis function.
    
    Args:
        wav_path_or_array: Audio file path or numpy array
        debug: Enable debug output
        use_split: If True, use frequency range splitting to separate left/right hand (default: True)
        independent_hands: If True and use_split is True, detect bass and treble rhythms 
                          independently (enables held bass chord + moving treble melody).
                          If False, uses shared onset detection (default: True)
    
    For the legacy frame-by-frame pipeline, use analyze_audio_legacy().
    """
    if use_split:
        if independent_hands:
            return analyze_audio_independent_hands(wav_path_or_array, debug=debug)
        else:
            return analyze_audio_split_ranges(wav_path_or_array, debug=debug)
    else:
        return analyze_audio_optimized(wav_path_or_array, debug=debug)

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
    C_full = np.abs(librosa.cqt(
        y=audio, 
        sr=SAMPLE_RATE,
        hop_length=HOP_SIZE,
        n_bins=CQT_BINS,
        bins_per_octave=12,
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
        fft_mag_center = compute_magnitude(frames[idx])
        if 1 <= idx < len(frames)-1:
            fft_mag_prev   = compute_magnitude(frames[idx-1])
            fft_mag_next   = compute_magnitude(frames[idx+1])
            mag_window = (fft_mag_prev + fft_mag_center + fft_mag_next) / 3.0
        else:
            mag_window = fft_mag_center

        # ringing cancellation
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0/SAMPLE_RATE)
        resid, updated = cancel_ringing(mag_window, freqs)
        # 2) Explain it with K harmonic sources chosen by BIC
        bic_est = estimate_voices_bic(resid, max_K=3, H=8)
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
def analyze_audio_cmdline(wav_path_or_array, use_legacy=False, use_split=True, split_midi=60, independent_hands=True):
    """
    Command-line focused audio analysis with both single note and chord detection.
    Includes detailed console logging of the analysis process and thresholds.
    
    Args:
        wav_path_or_array: Audio file path or numpy array
        use_legacy: Use old frame-by-frame pipeline (default: False)
        use_split: Use frequency range splitting to separate left/right hand (default: True)
        split_midi: MIDI note to split at when use_split=True (default: 60 = middle C)
        independent_hands: If True, detect bass/treble rhythms independently (default: True)
    """
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
    C_full = np.abs(librosa.cqt(
        y=audio, 
        sr=SAMPLE_RATE,
        hop_length=HOP_SIZE,
        n_bins=CQT_BINS,
        bins_per_octave=12,
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
        print(f"  🔬 Building onset-centered spectrum...")
        fft_mag_center = compute_magnitude(frames[idx])
        if 1 <= idx < len(frames)-1:
            fft_mag_prev   = compute_magnitude(frames[idx-1])
            fft_mag_next   = compute_magnitude(frames[idx+1])
            mag_window = (fft_mag_prev + fft_mag_center + fft_mag_next) / 3.0
            print(f"     Using 3-frame average (frames {idx-1}-{idx+1}) for stability")
        else:
            mag_window = fft_mag_center
            print(f"     Using single frame {idx} (edge case)")
        
        spectrum_energy = np.sum(mag_window)
        max_magnitude = np.max(mag_window)
        print(f"     Spectrum energy: {spectrum_energy:.2f}, max magnitude: {max_magnitude:.4f}")

        # ringing cancellation
        freqs = np.fft.rfftfreq(FFT_SIZE, 1.0/SAMPLE_RATE)
        resid, updated = cancel_ringing(mag_window, freqs)
        # 2) Explain it with K harmonic sources chosen by BIC
        print(f"  🎼 Performing BIC harmonic mixture analysis...")
        bic_est = estimate_voices_bic(resid, max_K=3, H=8)
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

#* ─── Main Pipeline ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Use absolute path to audio file
    wav_path = os.path.join(os.path.dirname(__file__), 'audio', test_benchmark)
    print(f"🎹 Piano Note Detection - Command Line")
    print(f"Reading audio from: {wav_path}")
    try:
        audio = read_wav(wav_path)
    except Exception as e:
        print(f"Failed to open audio file: {e}")
        exit()
    
    results = analyze_audio_cmdline(audio)

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
                print(f"  {chord['time_seconds']:6.2f}s -> {off_str} (dur {dur_str}): {chord['label']:>8} octave {chord['octave']} ({chord['inversion']} inversion) - confidence: {chord['confidence']:.3f}")
        
        if not results["notes"] and not results["chords"]:
            print("  No notes or chords detected")
        
        if not results["notes"] and not results["chords"]:
            print("  No notes or chords detected")
            print("  No notes or chords detected")
