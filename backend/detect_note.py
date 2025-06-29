import os

import librosa
import numpy as np
import soundfile as sf
from hmmlearn import hmm
from numba import njit
from scipy.signal import get_window, medfilt

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

# Precompute MIDI → Hz for CQT bins 21…108
bin_freq = np.array([440.0 * 2**((m - 69)/12) for m in np.arange(21, 21 + CQT_BINS)])

#* ─── Read + High-Pass Filter ────────────────────────────────────────────────
def read_wav(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    if sr != SAMPLE_RATE:
        raise ValueError(f"Expected {SAMPLE_RATE} Hz, got {sr}")
    # simple one‐pole HPF: y[n] = x[n] - x[n-1] + alpha y[n-1]
    alpha = 0.95
    y = np.empty_like(audio)
    prev_x, prev_y = audio[0], audio[0]
    y[0] = prev_y
    for i in range(1, len(audio)):
        y[i] = audio[i] - prev_x + alpha * prev_y
        prev_x, prev_y = audio[i], y[i]
    return y

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

def find_onsets(flux, window=50, K=1.5):
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

#* ─── Chord Detection ────────────────────────────────────────────────────────
def make_templates():
    """Build normalized pitch-class templates for common triads."""
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
NOTE_LABELS    = ROOTS.copy()

def extract_chroma(audio, sr, hop_length=512):
    C = librosa.feature.chroma_cqt(y=audio, sr=sr, hop_length=hop_length)
    return C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-6)

def match_chords(chroma: np.ndarray,
                 templates: np.ndarray,
                 labels: list[str]):
    """
    Frame-wise template matching.

    Args:
      chroma:    shape (12, T) matrix of normalized chroma vectors
      templates: shape (N, 12) array of chord templates
      labels:    length-N list of chord names matching templates rows

    Returns:
      roots:    length-T list of chord labels (e.g. "C:maj")
      roots_pc: length-T list of semitone classes (0=C,1=C#, …)
      scores:   shape (N, T) similarity scores for each chord/template
    """
    # 1) Compute similarity between each template and each chroma frame
    scores = templates.dot(chroma)         # (N, T)

    # 2) Pick best-matching template per frame
    best_idx = np.argmax(scores, axis=0)   # (T,)
    roots    = [labels[i] for i in best_idx]

    # 3) Map root names to pitch classes
    roots_pc = []
    for lbl in roots:
        root_name = lbl.split(':')[0]  # e.g. "C" from "C:maj"
        roots_pc.append(ROOTS.index(root_name))

    return roots, roots_pc, scores

def smooth_with_hmm(emission_probs: np.ndarray,
                    labels: list[str],
                    stay_prob: float = 0.9) -> list[str]:
    """
    emission_probs: shape (n_frames, n_states) of P(observed | state)
    labels:        list of length n_states mapping state idx → label
    """
    n_states = emission_probs.shape[1]
    # build HMM
    model = hmm.MultinomialHMM(n_components=n_states, init_params="")
    # uniform start
    model.startprob_ = np.ones(n_states) / n_states
    # high self-transition
    tm = np.full((n_states, n_states), (1 - stay_prob)/(n_states-1))
    np.fill_diagonal(tm, stay_prob)
    model.transmat_ = tm
    # emission probabilities (rows = states, cols = symbols)
    # but MultinomialHMM expects shape (n_states, n_symbols),
    # so we treat each frame as drawing one “symbol” (the best chord idx)
    # instead, we’ll do custom decode: use log(emission_probs) as log-likelihoods.

    # Run Viterbi
    logp = np.log(emission_probs + 1e-8)  # avoid log(0)
    _, state_seq = model.decode(logp, algorithm="viterbi")
    return [labels[s] for s in state_seq]

def detect_true_bass_pc(mag_frame, floor_frac=0.1):
    """Find the semitone class of the lowest active bin in a magnitude spectrum."""
    thresh = mag_frame.max() * floor_frac
    active_bins = np.where(mag_frame >= thresh)[0]
    if active_bins.size == 0:
        return None
    # lowest active bin index modulo 12 gives the pitch class
    return int(active_bins.min().item() % 12)

#* ─── TWM Pitch Picker ──────────────────────────────────────────────────────
class Peak:
    def __init__(self,f,m,p,b): self.freq, self.mag, self.phase, self.bin = f,m,p,b

def parabolic_interp(y1,y2,y3,k):
    a = (y1 - 2*y2 + y3)/2
    b = (y3 - y1)/2
    if a==0: return k*SAMPLE_RATE/FFT_SIZE
    xp = -b/(2*a)
    return (k+xp)*SAMPLE_RATE/FFT_SIZE

def find_spectral_peaks(mag, fft_complex, thresh=0.01, max_peaks=50):
    M = np.max(mag)
    min_t = thresh*M
    peaks = []
    for k in range(1, MAG_SIZE-1):
        if mag[k]>mag[k-1] and mag[k]>mag[k+1] and mag[k]>min_t:
            f = parabolic_interp(mag[k-1],mag[k],mag[k+1],k)
            ph = np.angle(fft_complex[k])
            peaks.append(Peak(f, mag[k], ph, k))
            if len(peaks)>=max_peaks: break
    peaks.sort(key=lambda pk: pk.mag, reverse=True)
    return peaks

def compute_TWM_error(peaks, f0, max_h=10):
    if not peaks: return 1e6
    err, nm = 0,0
    # forward
    for h in range(1, max_h+1):
        tgt = f0*h
        if tgt>SAMPLE_RATE/2: break
        d = min(abs(pk.freq - tgt) for pk in peaks)
        err += d; nm+=1
    # backward
    for pk in peaks:
        if pk.freq < 0.8*f0: continue
        h = int(round(pk.freq/f0))
        if h<1 or h>max_h: continue
        err += abs(pk.freq - f0*h)*pk.mag
    return err/(nm+1)

def detect_f0_TWM(peaks, lo=80, hi=800):
    if not peaks: return 0
    best, be = 0,1e6
    for f0 in np.arange(lo, hi+1, 2):
        e = compute_TWM_error(peaks,f0)
        if e<be: best,be = f0,e
    # refine
    for f0 in np.arange(max(lo,best-5), min(hi,best+5)+1, 0.2):
        e = compute_TWM_error(peaks,f0)
        if e<be: best,be = f0,e
    return best if be<50 else 0

def pick_pitches_TWM(cqt_mag, mag, fft_complex, max_voices=4):
    peaks = find_spectral_peaks(mag, fft_complex)
    f0 = detect_f0_TWM(peaks, 80, 800)
    notes = []
    if f0>0:
        m = int(round(69 + 12*np.log2(f0/440)))
        if 21<=m<=108: notes.append(m)
    
    # More conservative polyphonic detection
    residual = cqt_mag.copy()
    if notes:
        # Remove harmonics of the fundamental more aggressively
        b0 = notes[0]-21
        fundamental_mag = cqt_mag[b0] if 0 <= b0 < CQT_BINS else 0
        
        for h in range(1, 8):  # Check more harmonics
            b = int(round(b0 + 12*np.log2(h)))
            if 0<=b<CQT_BINS:
                # More aggressive harmonic subtraction
                residual[b] = max(0, residual[b] - cqt_mag[b] * 0.8)
    
    # Much higher threshold for additional notes to avoid harmonic confusion
    mres = np.max(residual)
    threshold = 0.8 * mres  # Increased from 0.5 to 0.8
    
    # Only add additional notes if they're significantly strong
    additional_notes = []
    for b in range(CQT_BINS):
        if residual[b] > threshold and len(notes) + len(additional_notes) < max_voices:
            midi = 21+b
            if midi not in notes:
                # Extra check: make sure this isn't close to a harmonic of existing notes
                is_harmonic = False
                for existing_note in notes:
                    freq_existing = 440.0 * 2**((existing_note - 69)/12)
                    freq_candidate = 440.0 * 2**((midi - 69)/12)
                    ratio = freq_candidate / freq_existing
                    # Check if it's close to a harmonic ratio (2, 3, 4, 5, 6)
                    for h in [2, 3, 4, 5, 6]:
                        if abs(ratio - h) < 0.1:  # Within 10% of harmonic ratio
                            is_harmonic = True
                            break
                if not is_harmonic:
                    additional_notes.append(midi)
    
    notes.extend(additional_notes)
    return sorted(notes)

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
                if harmonic_details:
                    for h, mag, weight in harmonic_details:
                        print(f"      Harmonic {h}: mag={mag:.3f}, weight={weight:.3f}")
    
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
        
    return note_info

def detect_chord_frame(chroma, mag, frame_idx, debug=False):
    c_frame = chroma[:, frame_idx]

    score = 0

    # 2) Chroma-peak ratio check
    sorted_bins = np.sort(c_frame)[::-1]
    if sorted_bins[1] >= 0.5 * sorted_bins[0]:
        score += 1

    # 3) Template vs note score
    note_score = NOTE_TEMPLATES.dot(c_frame).max()
    chord_scores = CH_TEMPLATES.dot(c_frame)
    best_chord_score = chord_scores.max()
    if best_chord_score > note_score:
        score += 1

    if debug:
        print(f"[ChordGate] frame={frame_idx}, pts={score}/2, "
              f"ratio={sorted_bins[1]/sorted_bins[0]:.2f}, "
              f"note>chord? {note_score:.2f}>{best_chord_score:.2f}")

    if score < 1:
        return None

    # Chord result
    ci = int(np.argmax(chord_scores).item())
    chord_label = CH_LABELS[ci]
    root_pc = ROOTS.index(chord_label.split(':')[0])
    bass_pc = detect_true_bass_pc(mag[:, frame_idx])
    if bass_pc is None or bass_pc == root_pc:
        inv = 'root'
    elif bass_pc in {(root_pc+3)%12, (root_pc+4)%12}:
        inv = 'first'
    elif bass_pc == (root_pc+7)%12:
        inv = 'second'
    else:
        inv = 'slash'

    return {
        "type": "chord",
        "label": chord_label,
        "inversion": inv,
        "confidence": best_chord_score,
        "note_score": note_score
    }

def analyze_audio(wav_path_or_array, debug=False):
    # 1) Load
    if isinstance(wav_path_or_array, str):
        audio = read_wav(wav_path_or_array)
    else:
        audio = wav_path_or_array
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

    # 2) Onsets
    frames = frame_audio(audio)
    mags = np.array([compute_magnitude(f) for f in frames])
    flux = normalize(compute_flux(mags))
    onsets = find_onsets(flux)

    # 3) Precompute chroma & full-range CQT
    chroma = extract_chroma(audio, SAMPLE_RATE, hop_length=HOP_SIZE)
    C_full = np.abs(librosa.cqt(
        y=audio, 
        sr=SAMPLE_RATE,
        hop_length=HOP_SIZE,
        n_bins=CQT_BINS,
        bins_per_octave=12,
        fmin=librosa.note_to_hz('C1')
    ))

    # 4) Compute gating features
    T = chroma.shape[1]
    peak_ratios = np.zeros(T)
    score_ratios = np.zeros(T)
    spec_entropy = np.zeros(T)
    spec_flatness = np.zeros(T)
    for i in range(T):
        bins = np.sort(chroma[:, i])[::-1]
        peak_ratios[i] = bins[1] / (bins[0] + 1e-9)
        note_sc = NOTE_TEMPLATES.dot(chroma[:,i]).max()
        chord_sc = CH_TEMPLATES.dot(chroma[:,i]).max()
        score_ratios[i] = chord_sc / (note_sc + 1e-9)
        mf = C_full[:, i]
        p = mf / (mf.sum() + 1e-9)
        spec_entropy[i] = -np.sum(p * np.log(p + 1e-9))
        geo = np.exp(np.mean(np.log(mf + 1e-9)))
        arith = np.mean(mf + 1e-9)
        spec_flatness[i] = geo / arith

    # 4.1) Dynamic thresholds
    thr_peak       = np.median(peak_ratios) + np.std(peak_ratios)
    thr_score      = np.median(score_ratios) + 1.5*np.std(score_ratios)
    thr_entropy    = np.median(spec_entropy) + 0.5*np.std(spec_entropy)
    thr_flatness   = np.median(spec_flatness) + 0.5*np.std(spec_flatness)

    # 4.2) Smooth gate across time
    gate_raw = ((peak_ratios>thr_peak).astype(int)
                + (score_ratios>thr_score).astype(int)
                + (spec_entropy>thr_entropy).astype(int)
                + (spec_flatness>thr_flatness).astype(int)) >= 3
    gate_smooth = medfilt(gate_raw.astype(int), kernel_size=5).astype(bool)

    results = {"onsets": [], "notes": [], "chords": []}
    # 4) Process each onset frame
    for i, onset in enumerate(onsets):
        idx = min(onset + 3, len(frames)-1)
        frame = frames[idx]
        
        # Convert frame index to time
        time_seconds = onset * HOP_SIZE / SAMPLE_RATE

        if debug:
            print(f"\n=== ONSET {i+1} at frame {onset} ({time_seconds:.2f}s) ===")

        if gate_smooth[onset]:
            # chord detection on harmonic
            res = detect_chord_frame(chroma, C_full, onset, debug)
            if res:
                res.update({"time_seconds":round(time_seconds,3), "frame_index":int(onset)})
                results["chords"].append(res)
        else:
            # note detection on original
            res = detect_single_note_frame(frame, debug)
            if res["midi_note"]:
                res.update({"time_seconds":round(time_seconds,3), "frame_index":int(onset)})
                results["notes"].append(res)

    # Add analysis summary with proper Python types
    results["analysis_summary"] = {
        "total_onsets": len(onsets),
        "total_notes": len(results["notes"]),
        "total_chords": len(results["chords"]),
        "duration_seconds": float(len(audio) / SAMPLE_RATE),
        "sample_rate": int(SAMPLE_RATE)
    }

    return results

#* ─── Command-line Analysis Function ───────────────────────────────────────────
def analyze_audio_cmdline(wav_path_or_array):
    """
    Command-line focused audio analysis that only does single note detection.
    Based on the original implementation before chord detection was added.
    """
    try:
        # Read audio
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
    
    def midi_to_name(m):
        names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
        return f"{names[m%12]}{(m//12)-1}"
    
    print(f"Audio duration: {len(audio) / SAMPLE_RATE:.2f}s")
    
    # Traditional onset detection
    print("🔍 Detecting onsets...")
    frames = frame_audio(audio)
    mags = np.array([compute_magnitude(f) for f in frames])
    flux = normalize(compute_flux(mags))
    onsets = find_onsets(flux)
    
    print(f"✓ Found {len(onsets)} onsets at frames: {onsets}")
    
    # Results structure
    results = {
        "onsets": [],
        "notes": [],
        "analysis_summary": {
            "total_onsets": len(onsets),
            "duration_seconds": float(len(audio) / SAMPLE_RATE),
            "sample_rate": int(SAMPLE_RATE)
        }
    }
    
    # Process each onset
    print("🎵 Analyzing notes at each onset...")
    for i, onset in enumerate(onsets):
        idx = min(onset + 3, len(frames)-1)
        frame = frames[idx]
        
        # Convert frame index to time
        time_seconds = float(onset * HOP_SIZE / SAMPLE_RATE)
        
        print(f"\n=== ONSET {i+1}/{len(onsets)} at frame {onset} ({time_seconds:.2f}s) ===")
        
        # Method 1: FFT-based detection
        fft_note, freqs, fft_mag = detect_fundamental_from_fft(frame)
        if fft_note:
            print(f"  FFT method: {midi_to_name(fft_note)} (MIDI {fft_note})")
        
        # Method 2: CQT-based detection
        cqt_mag = compute_cqt(frame)
        
        # Method 3: Simple CQT method
        simple_note = detect_fundamental_simple(cqt_mag)
        if simple_note:
            print(f"  Simple method: {midi_to_name(simple_note)}")
        
        # Method 4: HPS method
        hps_notes = pick_pitches_HPS(cqt_mag, max_voices=1)
        if hps_notes:
            print(f"  HPS method: {midi_to_name(hps_notes[0])}")
        
        # Method 5: Robust detection with octave correction
        robust_note, robust_method = detect_pitch_robust(frame, cqt_mag, fft_mag, freqs, debug=False)
        if robust_note:
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
        
        # Add onset information
        onset_info = {
            "time_seconds": round(time_seconds, 3),
            "frame_index": int(onset)
        }
        results["onsets"].append(onset_info)
        
        # Add note information if detected
        if final_note:
            note_info = {
                "time_seconds": round(time_seconds, 3),
                "frame_index": int(onset),
                "midi_note": int(final_note),
                "note_name": midi_to_name(final_note),
                "frequency_hz": round(440.0 * 2**((final_note - 69)/12), 2),
                "method": method_used,
                "confidence": confidence
            }
            results["notes"].append(note_info)
            
            print(f"  ➤ DETECTED: {midi_to_name(final_note)} (method: {method_used}, confidence: {confidence:.1f})")
        else:
            print(f"  ➤ No note detected")
    
    print(f"\n🎼 Analysis complete!")
    print(f"   Total onsets: {len(results['onsets'])}")
    print(f"   Notes detected: {len(results['notes'])}")
    print(f"   Detection rate: {len(results['notes'])/len(results['onsets'])*100:.1f}%")
    
    return results

#* ─── Main Pipeline ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Use absolute path to audio file
    wav_path = os.path.join(os.path.dirname(__file__), 'audio', 'test_chromatic.wav')
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
        for note in results["notes"]:
            print(f"{note['time_seconds']:6.2f}s: {note['note_name']:>4} ({note['frequency_hz']:6.1f}Hz) - {note['method']}")
    else:
        print(f"Analysis failed: {results['error']}")