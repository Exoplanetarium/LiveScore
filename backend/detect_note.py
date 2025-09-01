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
import numpy as np
import soundfile as sf
from numba import njit
from scipy.optimize import nnls
from scipy.signal import get_window, medfilt, resample_poly


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
    """Cheap salience: score each MIDI by aligning harmonics on FFT (no CQT)."""
    # restrict to piano range
    midi_lo, midi_hi = 24, 108
    scores = []
    for m in range(midi_lo, midi_hi+1):
        t = get_template(m)
        # dot with log-magnitude to reduce dominance of a few bins
        s = float((t * np.log1p(mag)).sum())
        scores.append((s, m))
    scores.sort(reverse=True)
    return [m for s,m in scores[:top]]

def estimate_voices_bic(mag_window, max_K=3, H=8):
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
    cand_midis = _salience_candidates_from_fft(x, top=8, H=H)

    best = {'K': 0, 'midis': [], 'gains': np.array([]), 'bic': _bic(np.sum(x*x), B, 0), 'err': float(np.sum(x*x))}
    # Try K=1..max_K by taking top-K candidates; refine by pruning tiny gains
    for K in range(1, max_K+1):
        midis = cand_midis[:K]
        gains, err = _fit_nonneg_mixture(x, midis, iters=6)
        # prune near-zero components and recompute (optional)
        keep = gains > (0.02 * gains.max())
        if keep.any() and keep.sum() < K:
            midis = [m for m, k in zip(midis, keep) if k]
            gains, err = _fit_nonneg_mixture(x, midis, iters=4)
            K_eff = len(midis)
        else:
            K_eff = K
        bic = _bic(err, B, dof=K_eff*1.8)  # mild penalty; tweak 1.3–2.0 if needed
        if bic < best['bic']:
            best = {'K': K_eff, 'midis': midis, 'gains': gains[:K_eff], 'bic': bic, 'err': float(err)}
    return best

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

def estimate_bass_bin(mag, t0, *, bpo=12, fmin=librosa.note_to_hz('C1'), frames_ahead=3, lowpass_hz=220.0,
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

def bin_to_octave(bin_idx, *, bpo=12, fmin=librosa.note_to_hz('C1'), a4=440.0):
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

def _bin_to_pc(bin_idx, *, bpo=12, fmin=librosa.note_to_hz('C1'), a4=440.0):
    """
    Map a CQT bin index to a pitch-class (0..11) via frequency -> MIDI -> PC.
    """
    freq = fmin * (2.0 ** (bin_idx / float(bpo)))
    midi = 69.0 + 12.0 * np.log2(freq / float(a4))
    return int(round(midi)) % 12

def detect_bass_pc_conf(mag, t0, *, bpo=12, fmin=librosa.note_to_hz('C1'), frames_ahead=2, lowpass_hz=220.0):
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
                      t0, *, bpo=12, fmin=librosa.note_to_hz('C1'),
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
            fmin=librosa.note_to_hz('C1')
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

def analyze_audio(wav_path_or_array, debug=False):
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
        fmin=librosa.note_to_hz('C1')
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

        is_chord_final = (K >= 2)

        if is_chord_final:            
            # Use existing chord detection for labeling and inversion analysis
            res = detect_chord_multiframe(chroma, C_full, onset, num_frames=1, debug=True)
            if res is not None:
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
def analyze_audio_cmdline(wav_path_or_array):
    """
    Command-line focused audio analysis with both single note and chord detection.
    Includes detailed console logging of the analysis process and thresholds.
    """
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
        fmin=librosa.note_to_hz('C1')
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
