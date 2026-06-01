"""
GPU-accelerated operations for LiveScore audio processing.

Provides CUDA-accelerated replacements for CPU-bound operations:
- STFT / ISTFT (replaces scipy + manual numpy FFT loops)
- Fused noise reduction pipeline (persistent tones + multiband gate in 1 STFT)
- CQT computation (replaces librosa.cqt)
- Rhythm MLP inference (batched PyTorch instead of numpy one-at-a-time)
- Onset flux computation (replaces frame_audio + compute_magnitude loop)

All functions gracefully fall back to CPU if CUDA is not available.
"""

import math
import os
import sys
from functools import lru_cache
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add rhythm_training to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'rhythm_training'))

if TYPE_CHECKING:
    from rhythm_training.train_ensemble import (MultiResFeatureExtractor,
                                                _build_model_from_config,
                                                decode_note_events)
    from rhythm_training.train_mel_baseline import (MelBaselineTranscriber,
                                                    MelFeatureExtractor)
    from rhythm_training.train_mel_baseline import \
        _build_model_from_config as _build_mel_model
    from rhythm_training.train_transcription import (PianoTranscriptionModel,
                                                     transcribe_audio)

# ─── Device Management ──────────────────────────────────────────────────────

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_GPU = DEVICE.type == 'cuda'

def get_device():
    return DEVICE

def print_gpu_info():
    if USE_GPU:
        print(f"[GPU] Using {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory
        print(f"[GPU] VRAM: {mem / 1e9:.1f} GB")
    else:
        print("[GPU] CUDA not available, using CPU fallback")


# ─── Cached Window Functions ────────────────────────────────────────────────

_window_cache: Dict[Tuple[int, str], torch.Tensor] = {}

def _get_window(n_fft: int, window_type: str = 'hann') -> torch.Tensor:
    key = (n_fft, window_type, str(DEVICE))
    if key not in _window_cache:
        if window_type == 'hann':
            win = torch.hann_window(n_fft, device=DEVICE)
        elif window_type == 'hamming':
            win = torch.hamming_window(n_fft, device=DEVICE)
        else:
            win = torch.hann_window(n_fft, device=DEVICE)
        _window_cache[key] = win
    return _window_cache[key]


# ─── GPU STFT / ISTFT ──────────────────────────────────────────────────────

def gpu_stft(audio: np.ndarray, n_fft: int = 2048, hop_length: int = 512,
             window_type: str = 'hann', center: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    GPU-accelerated STFT.

    Returns:
        (stft_complex, magnitude, phase) as GPU tensors.
        stft_complex: shape (n_fft//2+1, n_frames), complex64
        magnitude: shape (n_fft//2+1, n_frames), float32
        phase: shape (n_fft//2+1, n_frames), float32
    """
    audio_t = torch.from_numpy(audio).float().to(DEVICE)
    win = _get_window(n_fft, window_type)

    stft_complex = torch.stft(
        audio_t, n_fft, hop_length=hop_length,
        window=win, return_complex=True, center=center
    )
    magnitude = torch.abs(stft_complex)
    phase = torch.angle(stft_complex)
    return stft_complex, magnitude, phase


def gpu_istft(stft_complex: torch.Tensor, n_fft: int = 2048,
              hop_length: int = 512, length: Optional[int] = None,
              window_type: str = 'hann', center: bool = True) -> np.ndarray:
    """
    GPU-accelerated inverse STFT. Returns numpy array.
    """
    win = _get_window(n_fft, window_type)
    audio = torch.istft(
        stft_complex, n_fft, hop_length=hop_length,
        window=win, length=length, center=center
    )
    return audio.cpu().numpy().astype(np.float32)


def batch_stft(audios: List[np.ndarray], n_fft: int = 2048,
               hop_length: int = 512) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Batched GPU STFT for processing multiple audio signals simultaneously.
    All audio signals must have the same length.

    Returns: (stft_complex, magnitude, phase) with batch dimension 0.
    """
    # Pad to same length
    max_len = max(len(a) for a in audios)
    batch = np.zeros((len(audios), max_len), dtype=np.float32)
    for i, a in enumerate(audios):
        batch[i, :len(a)] = a

    batch_t = torch.from_numpy(batch).to(DEVICE)
    win = _get_window(n_fft)

    stft_complex = torch.stft(
        batch_t, n_fft, hop_length=hop_length,
        window=win, return_complex=True, center=True
    )
    magnitude = torch.abs(stft_complex)
    phase = torch.angle(stft_complex)
    return stft_complex, magnitude, phase


def batch_istft(stft_complex: torch.Tensor, n_fft: int = 2048,
                hop_length: int = 512, lengths: Optional[List[int]] = None) -> List[np.ndarray]:
    """Batched inverse STFT. Returns list of numpy arrays."""
    win = _get_window(n_fft)
    results = []
    for i in range(stft_complex.shape[0]):
        length = lengths[i] if lengths else None
        audio = torch.istft(
            stft_complex[i], n_fft, hop_length=hop_length,
            window=win, length=length, center=True
        )
        results.append(audio.cpu().numpy().astype(np.float32))
    return results


# ─── GPU STFT for compute_stft_once() Replacement ──────────────────────────

def gpu_compute_stft_once(audio: np.ndarray, sr: int = 44100,
                          n_fft: int = 2048, hop_length: int = 512,
                          window_type: str = 'hann'):
    """
    GPU replacement for compute_stft_once(). Returns numpy arrays
    matching the expected API: (stft_data, magnitude, phase, freqs).

    Uses float64 output to match the existing pipeline's precision.
    """
    stft_complex, magnitude, phase = gpu_stft(audio, n_fft, hop_length, window_type, center=False)

    # Convert to numpy, matching original float64 precision
    stft_np = stft_complex.cpu().numpy().astype(np.complex128)
    mag_np = magnitude.cpu().double().cpu().numpy()
    phase_np = phase.cpu().double().cpu().numpy()
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)

    return stft_np, mag_np, phase_np, freqs


# ─── Fused Noise Reduction Pipeline ────────────────────────────────────────

def fused_noise_reduce(audio: np.ndarray, sr: int = 44100,
                       n_fft: int = 2048, hop_length: int = 512,
                       # persistent tone params
                       persistence_percentile: float = 10,
                       subtraction_strength: float = 0.8,
                       persistent_min_freq: float = 30,
                       persistent_max_freq: float = 4000,
                       # multiband gate params
                       noise_estimation_seconds: float = 0.15,
                       gate_threshold_db: float = -10,
                       min_gate_threshold_db: float = -50
                       ) -> Tuple[np.ndarray, float, float]:
    """
    Fused noise reduction: persistent tone removal + multiband spectral gate
    in a SINGLE GPU STFT/ISTFT pair (replaces 2 separate CPU STFT/ISTFT ops).

    Returns:
        (filtered_audio, persistent_db_removed, gate_db_removed)
    """
    orig_len = len(audio)
    audio_t = torch.from_numpy(audio).float().to(DEVICE)
    win = _get_window(n_fft)

    # ── Single GPU STFT ──
    stft = torch.stft(audio_t, n_fft, hop_length=hop_length,
                      window=win, return_complex=True, center=True)
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    n_bins, n_frames = magnitude.shape

    freqs_np = np.fft.rfftfreq(n_fft, 1.0/sr)
    freqs = torch.from_numpy(freqs_np).float().to(DEVICE)

    original_power = torch.sum(magnitude ** 2).item()

    # ─── STEP 1: Persistent Tone Removal (on GPU) ───
    persistent_floor = torch.quantile(
        magnitude, persistence_percentile / 100.0, dim=1, keepdim=True
    )
    median_magnitude = torch.median(magnitude, dim=1, keepdim=True).values
    overall_median = torch.median(magnitude).item()

    # Persistence score
    persistence_score = persistent_floor / (median_magnitude + 1e-10)

    # Frequency mask
    freq_mask = (freqs >= persistent_min_freq) & (freqs <= persistent_max_freq)

    # Energy threshold
    min_energy_threshold = 0.05 * overall_median
    has_energy = persistent_floor[:, 0] > min_energy_threshold

    # Combined mask
    high_persistence = (persistence_score[:, 0] > 0.2) & freq_mask & has_energy

    # Build subtraction mask
    subtraction_mask = torch.zeros_like(persistent_floor)
    subtraction_mask[high_persistence, :] = (
        subtraction_strength * persistence_score[high_persistence, :]
    )

    # Apply persistent tone subtraction
    magnitude = torch.maximum(
        magnitude - subtraction_mask * persistent_floor,
        magnitude * 0.01
    )

    after_persistent_power = torch.sum(magnitude ** 2).item()
    persistent_db = 10 * math.log10(original_power / (after_persistent_power + 1e-10))

    # ─── STEP 2: Multiband Spectral Gate (on GPU) ───
    # Find quietest frames for noise estimation
    frame_rms = torch.sqrt(torch.mean(magnitude ** 2, dim=0))
    n_noise_frames = max(5, int(noise_estimation_seconds * sr / hop_length))
    quietest_indices = torch.argsort(frame_rms)[:n_noise_frames]

    noise_floor = torch.mean(magnitude[:, quietest_indices], dim=1, keepdim=True)
    noise_floor = torch.maximum(noise_floor, torch.tensor(1e-10, device=DEVICE))

    # Multiband thresholds
    band_thresholds = torch.ones(n_bins, device=DEVICE) * gate_threshold_db
    for i, f in enumerate(freqs_np):
        if f < 40:
            band_thresholds[i] = gate_threshold_db - 10
        elif f < 200:
            band_thresholds[i] = gate_threshold_db - 3
        elif f < 2000:
            band_thresholds[i] = gate_threshold_db
        elif f < 8000:
            band_thresholds[i] = gate_threshold_db - 3
        else:
            band_thresholds[i] = gate_threshold_db - 8

    band_thresholds = band_thresholds.unsqueeze(1)  # (n_bins, 1)

    threshold_linear = torch.pow(10.0, band_thresholds / 20.0)
    min_threshold = 10.0 ** (min_gate_threshold_db / 20.0)
    threshold = torch.maximum(noise_floor * threshold_linear,
                              torch.tensor(min_threshold, device=DEVICE))

    # SNR and soft gate
    snr = magnitude / threshold

    # Transient detection
    frame_energy = torch.sum(magnitude ** 2, dim=0)
    energy_diff = torch.diff(frame_energy, prepend=frame_energy[:1])
    transient_threshold = torch.quantile(energy_diff, 0.9)
    transient_mask = energy_diff > transient_threshold

    # Soft gate with sigmoid
    gate_mask = 1.0 / (1.0 + torch.exp(-4.0 * (snr - 1.0)))

    # Preserve transients
    gate_mask[:, transient_mask] = gate_mask[:, transient_mask] * 0.5 + 0.5

    # Apply gate
    magnitude_filtered = magnitude * gate_mask

    after_gate_power = torch.sum(magnitude_filtered ** 2).item()
    gate_db = 10 * math.log10(after_persistent_power / (after_gate_power + 1e-10))

    # ── Single GPU ISTFT ──
    filtered_stft = magnitude_filtered * torch.exp(1j * phase)
    filtered_audio = torch.istft(
        filtered_stft, n_fft, hop_length=hop_length,
        window=win, length=orig_len, center=True
    )

    result = filtered_audio.cpu().numpy().astype(np.float32)

    print(f"[GPU Noise] Fused pipeline: persistent={persistent_db:.2f}dB, gate={gate_db:.2f}dB removed")
    return result, float(persistent_db), float(gate_db)


def gpu_multiband_spectral_gate(audio: np.ndarray, sr: int = 44100,
                                n_fft: int = 2048, hop_length: int = 512,
                                noise_estimation_seconds: float = 0.15,
                                gate_threshold_db: float = -10,
                                min_gate_threshold_db: float = -50
                                ) -> Tuple[np.ndarray, float]:
    """
    GPU-accelerated multiband spectral gate (standalone, for per-band use).
    Same API as multiband_spectral_gate() in detect_note.py.
    """
    orig_len = len(audio)
    audio_t = torch.from_numpy(audio).float().to(DEVICE)
    win = _get_window(n_fft)

    stft = torch.stft(audio_t, n_fft, hop_length=hop_length,
                      window=win, return_complex=True, center=True)
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    n_bins, n_frames = magnitude.shape

    freqs_np = np.fft.rfftfreq(n_fft, 1.0/sr)

    # Noise estimation from quietest frames
    frame_rms = torch.sqrt(torch.mean(magnitude ** 2, dim=0))
    n_noise_frames = max(5, int(noise_estimation_seconds * sr / hop_length))
    quietest_indices = torch.argsort(frame_rms)[:n_noise_frames]
    noise_floor = torch.mean(magnitude[:, quietest_indices], dim=1, keepdim=True)
    noise_floor = torch.maximum(noise_floor, torch.tensor(1e-10, device=DEVICE))

    # Multiband thresholds
    band_thresholds = torch.ones(n_bins, device=DEVICE) * gate_threshold_db
    for i, f in enumerate(freqs_np):
        if f < 40:
            band_thresholds[i] = gate_threshold_db - 10
        elif f < 200:
            band_thresholds[i] = gate_threshold_db - 3
        elif f < 2000:
            band_thresholds[i] = gate_threshold_db
        elif f < 8000:
            band_thresholds[i] = gate_threshold_db - 3
        else:
            band_thresholds[i] = gate_threshold_db - 8

    band_thresholds = band_thresholds.unsqueeze(1)
    threshold_linear = torch.pow(10.0, band_thresholds / 20.0)
    min_threshold = 10.0 ** (min_gate_threshold_db / 20.0)
    threshold = torch.maximum(noise_floor * threshold_linear,
                              torch.tensor(min_threshold, device=DEVICE))

    snr = magnitude / threshold

    # Transients
    frame_energy = torch.sum(magnitude ** 2, dim=0)
    energy_diff = torch.diff(frame_energy, prepend=frame_energy[:1])
    transient_threshold = torch.quantile(energy_diff, 0.9)
    transient_mask = energy_diff > transient_threshold

    gate_mask = 1.0 / (1.0 + torch.exp(-4.0 * (snr - 1.0)))
    gate_mask[:, transient_mask] = gate_mask[:, transient_mask] * 0.5 + 0.5

    magnitude_filtered = magnitude * gate_mask

    original_power = torch.sum(magnitude ** 2).item()
    filtered_power = torch.sum(magnitude_filtered ** 2).item()
    noise_removed_db = 10 * math.log10(original_power / (filtered_power + 1e-10))

    filtered_stft = magnitude_filtered * torch.exp(1j * phase)
    filtered_audio = torch.istft(
        filtered_stft, n_fft, hop_length=hop_length,
        window=win, length=orig_len, center=True
    )

    return filtered_audio.cpu().numpy().astype(np.float32), float(noise_removed_db)


def gpu_batch_multiband_gate(audios: List[np.ndarray], sr: int = 44100,
                             n_fft: int = 2048, hop_length: int = 512,
                             noise_estimation_seconds: float = 0.15,
                             gate_threshold_db: float = -10,
                             min_gate_threshold_db: float = -50
                             ) -> List[Tuple[np.ndarray, float]]:
    """
    Batched GPU multiband spectral gate for processing bass + treble
    simultaneously in a single GPU kernel launch.
    """
    max_len = max(len(a) for a in audios)
    orig_lens = [len(a) for a in audios]

    batch = np.zeros((len(audios), max_len), dtype=np.float32)
    for i, a in enumerate(audios):
        batch[i, :len(a)] = a

    batch_t = torch.from_numpy(batch).to(DEVICE)
    win = _get_window(n_fft)

    # Batched STFT: (batch, n_fft//2+1, n_frames)
    stft = torch.stft(batch_t, n_fft, hop_length=hop_length,
                      window=win, return_complex=True, center=True)
    magnitude = torch.abs(stft)
    phase = torch.angle(stft)
    n_bins = magnitude.shape[1]

    freqs_np = np.fft.rfftfreq(n_fft, 1.0/sr)

    # Noise estimation per batch item
    frame_rms = torch.sqrt(torch.mean(magnitude ** 2, dim=1))  # (batch, n_frames)
    n_noise_frames = max(5, int(noise_estimation_seconds * sr / hop_length))

    results = []
    for b in range(len(audios)):
        quietest_indices = torch.argsort(frame_rms[b])[:n_noise_frames]
        noise_floor = torch.mean(magnitude[b, :, quietest_indices], dim=1, keepdim=True)
        noise_floor = torch.maximum(noise_floor, torch.tensor(1e-10, device=DEVICE))

        # Build threshold
        band_thresholds = torch.ones(n_bins, device=DEVICE) * gate_threshold_db
        for i, f in enumerate(freqs_np):
            if f < 40:
                band_thresholds[i] = gate_threshold_db - 10
            elif f < 200:
                band_thresholds[i] = gate_threshold_db - 3
            elif f < 2000:
                pass  # keep default
            elif f < 8000:
                band_thresholds[i] = gate_threshold_db - 3
            else:
                band_thresholds[i] = gate_threshold_db - 8

        band_thresholds = band_thresholds.unsqueeze(1)
        threshold_linear = torch.pow(10.0, band_thresholds / 20.0)
        min_threshold = 10.0 ** (min_gate_threshold_db / 20.0)
        threshold = torch.maximum(noise_floor * threshold_linear,
                                  torch.tensor(min_threshold, device=DEVICE))

        snr = magnitude[b] / threshold

        frame_energy = torch.sum(magnitude[b] ** 2, dim=0)
        energy_diff = torch.diff(frame_energy, prepend=frame_energy[:1])
        transient_threshold = torch.quantile(energy_diff, 0.9)
        transient_mask = energy_diff > transient_threshold

        gate_mask = 1.0 / (1.0 + torch.exp(-4.0 * (snr - 1.0)))
        gate_mask[:, transient_mask] = gate_mask[:, transient_mask] * 0.5 + 0.5

        mag_filtered = magnitude[b] * gate_mask

        orig_power = torch.sum(magnitude[b] ** 2).item()
        filt_power = torch.sum(mag_filtered ** 2).item()
        db_removed = 10 * math.log10(orig_power / (filt_power + 1e-10))

        filtered_stft = mag_filtered * torch.exp(1j * phase[b])
        audio_out = torch.istft(
            filtered_stft, n_fft, hop_length=hop_length,
            window=win, length=orig_lens[b], center=True
        )
        results.append((audio_out.cpu().numpy().astype(np.float32), float(db_removed)))

    return results


# ─── GPU CQT ───────────────────────────────────────────────────────────────

_cqt_filterbank_cache: Dict[Tuple, torch.Tensor] = {}

def _build_cqt_filterbank(n_fft_cqt: int, sr: int, n_bins: int,
                          bins_per_octave: int, fmin: float) -> torch.Tensor:
    """
    Build a filterbank matrix mapping large-FFT bins to CQT bins.
    Shape: (n_bins, n_fft_cqt//2+1)
    """
    Q = 1.0 / (2.0 ** (1.0 / bins_per_octave) - 1.0)
    freqs = np.fft.rfftfreq(n_fft_cqt, 1.0 / sr)

    filterbank = np.zeros((n_bins, len(freqs)), dtype=np.float32)

    for k in range(n_bins):
        f_center = fmin * (2.0 ** (k / bins_per_octave))
        # Bandwidth for constant-Q
        bw = f_center / Q
        f_low = f_center - bw / 2
        f_high = f_center + bw / 2

        # Gaussian window centered on f_center with sigma = bw/4
        sigma = bw / 4.0
        if sigma < 1e-6:
            continue
        weights = np.exp(-0.5 * ((freqs - f_center) / sigma) ** 2)
        # Zero out bins far from center (3 sigma)
        weights[freqs < f_center - 3 * sigma] = 0
        weights[freqs > f_center + 3 * sigma] = 0

        # Normalize so energy is preserved
        w_sum = np.sum(weights)
        if w_sum > 0:
            filterbank[k] = weights / w_sum

    return torch.from_numpy(filterbank).to(DEVICE)


def _get_cqt_filterbank(n_fft_cqt: int, sr: int, n_bins: int,
                        bins_per_octave: int, fmin: float) -> torch.Tensor:
    key = (n_fft_cqt, sr, n_bins, bins_per_octave, fmin)
    if key not in _cqt_filterbank_cache:
        _cqt_filterbank_cache[key] = _build_cqt_filterbank(
            n_fft_cqt, sr, n_bins, bins_per_octave, fmin
        )
    return _cqt_filterbank_cache[key]


def gpu_cqt(audio: np.ndarray, sr: int = 44100, n_bins: int = 88,
            bins_per_octave: int = 12, fmin: float = 27.5,
            hop_length: int = 512) -> np.ndarray:
    """
    GPU-accelerated CQT using large STFT + precomputed filterbank.

    Uses n_fft=32768 for sufficient low-frequency resolution (1.35 Hz bins),
    which resolves the lowest piano notes (A0=27.5 Hz, A#0=29.14 Hz).

    Returns:
        CQT magnitude array, shape (n_bins, n_frames), compatible with
        np.abs(librosa.cqt(...)) output.
    """
    n_fft_cqt = 32768  # Large FFT for low-frequency resolution

    audio_t = torch.from_numpy(audio).float().to(DEVICE)
    win = _get_window(n_fft_cqt)
    pad_mode = 'reflect'

    # torch.stft(center=True) reflect-pads by n_fft/2 on each side; for live
    # chunks shorter than that, fall back to constant padding instead of raising.
    if audio_t.numel() <= n_fft_cqt // 2:
        pad_mode = 'constant'

    # GPU STFT with large window
    stft = torch.stft(
        audio_t, n_fft_cqt, hop_length=hop_length,
        window=win, return_complex=True, center=True, pad_mode=pad_mode
    )
    magnitude = torch.abs(stft)  # (n_fft_cqt//2+1, n_frames)

    # Apply filterbank: (n_bins, n_fft//2+1) @ (n_fft//2+1, n_frames) = (n_bins, n_frames)
    filterbank = _get_cqt_filterbank(n_fft_cqt, sr, n_bins, bins_per_octave, fmin)
    cqt_mag = filterbank @ magnitude

    return cqt_mag.cpu().numpy()


# ─── GPU Onset Flux ─────────────────────────────────────────────────────────

def gpu_magnitude_and_flux(audio: np.ndarray, n_fft: int = 2048,
                           hop_length: int = 512,
                           window_type: str = 'hann') -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated computation of STFT magnitude and spectral flux.
    Replaces: frame_audio() + [compute_magnitude(f) for f in frames] + compute_flux().

    Returns:
        (magnitude, flux) as numpy arrays.
        magnitude: shape (n_fft//2+1, n_frames) - transposed from frame-based layout
        flux: shape (n_frames,) - normalized spectral flux
    """
    audio_t = torch.from_numpy(audio).float().to(DEVICE)
    win = _get_window(n_fft, window_type)

    stft = torch.stft(audio_t, n_fft, hop_length=hop_length,
                      window=win, return_complex=True, center=True)
    magnitude = torch.abs(stft)  # (n_fft//2+1, n_frames)

    # Spectral flux: sum of squared positive differences between consecutive frames
    # magnitude shape is (freq_bins, frames), diff along frames axis
    diffs = torch.diff(magnitude, dim=1)
    diffs_positive = torch.clamp(diffs, min=0)
    flux = torch.sum(diffs_positive ** 2, dim=0)
    flux = torch.cat([torch.zeros(1, device=DEVICE), flux])

    # Normalize
    mx = torch.max(flux)
    if mx > 0:
        flux = flux / mx

    # For magnitude, we need (freq_bins, n_frames) to match compute_stft_once output
    # BUT frame_audio + compute_magnitude gives (n_frames, freq_bins) as mags array
    # The calling code uses mags in different shapes depending on context.
    # Return both orientations for compatibility.
    return magnitude.cpu().numpy(), flux.cpu().numpy()


# ─── GPU Rhythm MLP ─────────────────────────────────────────────────────────

class GpuRhythmMLP(nn.Module):
    """
    PyTorch version of RhythmQuantizerMLP for batched GPU inference.
    Loads weights from the numpy .npz format used by the CPU model.
    """

    NOTE_TYPES = ['whole', 'half', 'quarter', 'eighth', '16th', '32nd']
    NOTE_TYPE_BEATS = {
        'whole': 4.0, 'half': 2.0, 'quarter': 1.0,
        'eighth': 0.5, '16th': 0.25, '32nd': 0.125
    }

    def __init__(self):
        super().__init__()
        self.fc1 = None
        self.fc2 = None
        self.fc3 = None  # optional third layer
        self.type_head = None
        self.dotted_head = None
        self.triplet_head = None
        self.has_third_layer = False
        self.initialized = False

    def load_from_npz(self, path: str):
        """Load weights from numpy .npz file (exported by CPU model)."""
        data = np.load(path)

        # Layer 1
        W1, b1 = data['W1'], data['b1']
        self.fc1 = nn.Linear(W1.shape[0], W1.shape[1])
        self.fc1.weight.data = torch.from_numpy(W1.T).float()
        self.fc1.bias.data = torch.from_numpy(b1).float()

        # Layer 2
        W2, b2 = data['W2'], data['b2']
        self.fc2 = nn.Linear(W2.shape[0], W2.shape[1])
        self.fc2.weight.data = torch.from_numpy(W2.T).float()
        self.fc2.bias.data = torch.from_numpy(b2).float()

        # Optional layer 3
        if 'W3' in data:
            W3, b3 = data['W3'], data['b3']
            self.fc3 = nn.Linear(W3.shape[0], W3.shape[1])
            self.fc3.weight.data = torch.from_numpy(W3.T).float()
            self.fc3.bias.data = torch.from_numpy(b3).float()
            self.has_third_layer = True

        # Output heads
        Wt, bt = data['W_type'], data['b_type']
        self.type_head = nn.Linear(Wt.shape[0], Wt.shape[1])
        self.type_head.weight.data = torch.from_numpy(Wt.T).float()
        self.type_head.bias.data = torch.from_numpy(bt).float()

        Wd, bd = data['W_dotted'], data['b_dotted']
        self.dotted_head = nn.Linear(Wd.shape[0], Wd.shape[1])
        self.dotted_head.weight.data = torch.from_numpy(Wd.T).float()
        self.dotted_head.bias.data = torch.from_numpy(bd).float()

        Wtr, btr = data['W_triplet'], data['b_triplet']
        self.triplet_head = nn.Linear(Wtr.shape[0], Wtr.shape[1])
        self.triplet_head.weight.data = torch.from_numpy(Wtr.T).float()
        self.triplet_head.bias.data = torch.from_numpy(btr).float()

        self.to(DEVICE)
        self.eval()
        self.initialized = True
        print(f"[GPU Rhythm] Loaded model from {path} (3-layer: {self.has_third_layer})")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        if self.has_third_layer:
            h = F.relu(self.fc3(h))

        type_logits = self.type_head(h)
        dotted_logits = self.dotted_head(h)
        triplet_logits = self.triplet_head(h)

        return {
            'type_probs': F.softmax(type_logits, dim=-1),
            'dotted_probs': F.softmax(dotted_logits, dim=-1),
            'triplet_probs': F.softmax(triplet_logits, dim=-1),
        }

    @torch.no_grad()
    def predict_batch(self, features: np.ndarray) -> List[Dict]:
        """
        Predict rhythm values for ALL notes at once on GPU.

        Args:
            features: (n_notes, 8) numpy array of features

        Returns:
            List of prediction dicts, one per note
        """
        if len(features) == 0:
            return []

        x = torch.from_numpy(features).float().to(DEVICE)
        out = self.forward(x)

        type_idx = torch.argmax(out['type_probs'], dim=-1).cpu().numpy()
        dotted = (torch.argmax(out['dotted_probs'], dim=-1) == 1).cpu().numpy()
        triplet = (torch.argmax(out['triplet_probs'], dim=-1) == 1).cpu().numpy()

        type_conf = torch.max(out['type_probs'], dim=-1).values.cpu().numpy()
        dotted_conf = torch.max(out['dotted_probs'], dim=-1).values.cpu().numpy()
        triplet_conf = torch.max(out['triplet_probs'], dim=-1).values.cpu().numpy()

        results = []
        for i in range(len(features)):
            t = int(type_idx[i])
            d = bool(dotted[i])
            tr = bool(triplet[i])
            note_type = self.NOTE_TYPES[t]

            base_beats = self.NOTE_TYPE_BEATS[note_type]
            if d:
                base_beats *= 1.5
            if tr:
                base_beats *= 2.0 / 3.0

            results.append({
                'note_type': note_type,
                'dotted': d,
                'is_triplet': tr,
                'confidence': float(type_conf[i] * dotted_conf[i] * triplet_conf[i]),
                'beats': base_beats,
            })

        return results


# ─── GPU Rhythm Model Singleton ─────────────────────────────────────────────

_gpu_rhythm_model: Optional[GpuRhythmMLP] = None
_gpu_rhythm_model_loaded = False

def get_gpu_rhythm_model() -> Optional[GpuRhythmMLP]:
    """Lazy-load the GPU-accelerated rhythm model (singleton)."""
    global _gpu_rhythm_model, _gpu_rhythm_model_loaded

    if _gpu_rhythm_model_loaded:
        return _gpu_rhythm_model

    model_paths = [
        os.path.join(os.path.dirname(__file__), 'rhythm_training', 'rhythm_model.npz'),
        os.path.join(os.path.dirname(__file__), 'rhythm_model.npz'),
        # Modal container path
        '/root/rhythm_training/rhythm_model.npz',
    ]

    for path in model_paths:
        if os.path.exists(path):
            try:
                _gpu_rhythm_model = GpuRhythmMLP()
                _gpu_rhythm_model.load_from_npz(path)
                break
            except Exception as e:
                print(f"[GPU Rhythm] Failed to load from {path}: {e}")
                _gpu_rhythm_model = None

    if _gpu_rhythm_model is None:
        print("[GPU Rhythm] Model not found, will use CPU fallback")

    _gpu_rhythm_model_loaded = True
    return _gpu_rhythm_model


def gpu_extract_features(notes: List[Dict], bpm: float,
                         use_ioi_as_duration: bool = True) -> np.ndarray:
    """
    Vectorized feature extraction for rhythm model (matches extract_features_for_ml).
    Uses numpy vectorization instead of per-note Python loop.
    """
    n = len(notes)
    if n == 0:
        return np.zeros((0, 8), dtype=np.float32)

    beat_duration = 60.0 / bpm
    measure_duration = beat_duration * 4

    # Extract arrays
    onsets = np.array([note.get('time_seconds', 0) for note in notes], dtype=np.float32)
    durations = np.array([note.get('duration_seconds', 0.5) for note in notes], dtype=np.float32)
    pitches = np.array([note.get('midi_note', 60) for note in notes], dtype=np.float32)

    # IOIs (vectorized)
    iois = np.zeros(n, dtype=np.float32)
    iois[:-1] = np.diff(onsets)
    iois[-1] = durations[-1]  # last note uses duration

    # Primary / secondary duration
    if use_ioi_as_duration:
        primary_dur = iois
        secondary_dur = durations
    else:
        primary_dur = durations
        secondary_dur = iois

    dur_beats = primary_dur / beat_duration
    ioi_beats = secondary_dur / beat_duration
    beat_pos = (onsets % beat_duration) / beat_duration
    measure_pos = (onsets % measure_duration) / beat_duration / 4.0

    # Previous IOI
    prev_ioi_beats = np.ones(n, dtype=np.float32)
    prev_ioi_beats[1:] = np.diff(onsets) / beat_duration

    # Duration/IOI ratio
    dur_ioi_ratio = durations / np.maximum(iois, 0.01)

    # Normalized pitch
    norm_pitch = (pitches - 60) / 40.0

    # Fixed tempo column
    norm_tempo = np.ones(n, dtype=np.float32)

    features = np.stack([
        dur_beats, ioi_beats, beat_pos, measure_pos,
        prev_ioi_beats, dur_ioi_ratio, norm_pitch, norm_tempo
    ], axis=1)

    return features


# ─── GPU Rhythm Transformer (Sequence Model) ────────────────────────────────

class GpuRhythmTransformer(nn.Module):
    """
    GPU-accelerated Transformer for sequence-aware rhythm + rest prediction.

    Unlike the MLP which treats each note independently, this model sees the
    full note sequence and can predict:
      - note_type, dotted, triplet (same as MLP)
      - has_rest_after (NEW: learned from MAESTRO phrase structure)

    Loads weights from a .pt checkpoint saved by train_transformer.py.
    """

    NOTE_TYPES = ['whole', 'half', 'quarter', 'eighth', '16th', '32nd']
    NOTE_TYPE_BEATS = {
        'whole': 4.0, 'half': 2.0, 'quarter': 1.0,
        'eighth': 0.5, '16th': 0.25, '32nd': 0.125
    }

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.config = None

    def _build(self, config: dict):
        """Build layers from config dict."""
        d_model = config.get('d_model', 64)
        n_heads = config.get('n_heads', 4)
        n_layers = config.get('n_layers', 4)
        d_ff = config.get('d_ff', 256)
        input_dim = config.get('input_dim', 10)

        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)

        # Sinusoidal positional encoding
        max_len = 512
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=0.0,  # no dropout at inference
            batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.head_type = nn.Linear(d_model, 6)
        self.head_dotted = nn.Linear(d_model, 2)
        self.head_triplet = nn.Linear(d_model, 2)
        self.head_rest = nn.Linear(d_model, 2)

        self.config = config

    def load_from_pt(self, path: str):
        """Load weights from .pt checkpoint (saved by train_transformer.py)."""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        config = checkpoint.get('config', {})
        self._build(config)
        
        # Handle key remapping: training uses pos_enc.pe, inference uses pe
        state_dict = checkpoint['model_state_dict']
        if 'pos_enc.pe' in state_dict and 'pe' not in state_dict:
            state_dict['pe'] = state_dict.pop('pos_enc.pe')
        
        self.load_state_dict(state_dict)
        self.to(DEVICE)
        self.eval()
        self.initialized = True
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[GPU Transformer] Loaded from {path} ({n_params:,} params, "
              f"d_model={config.get('d_model')}, layers={config.get('n_layers')})")

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: (batch, seq_len, input_dim)
        Returns dict of softmax probabilities.
        """
        h = self.input_proj(x)
        h = h + self.pe[:, :h.size(1), :]
        h = self.encoder(h)

        return {
            'type_probs': F.softmax(self.head_type(h), dim=-1),
            'dotted_probs': F.softmax(self.head_dotted(h), dim=-1),
            'triplet_probs': F.softmax(self.head_triplet(h), dim=-1),
            'rest_probs': F.softmax(self.head_rest(h), dim=-1),
        }

    @torch.no_grad()
    def predict_batch(self, features: np.ndarray) -> List[Dict]:
        """
        Predict rhythm values + rest placement for a sequence of notes.

        Args:
            features: (n_notes, 10) numpy array of features.
                      If only 8 features provided, pads with zeros.

        Returns:
            List of prediction dicts, one per note, including 'has_rest'.
        """
        if len(features) == 0:
            return []

        # Pad to 10 features if needed (backward compat with 8-feature extraction)
        if features.shape[1] < 10:
            pad = np.zeros((features.shape[0], 10 - features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, pad], axis=1)

        # Add batch dimension: (1, seq_len, 10)
        x = torch.from_numpy(features).float().unsqueeze(0).to(DEVICE)
        out = self.forward(x)

        # Remove batch dimension
        type_idx = torch.argmax(out['type_probs'][0], dim=-1).cpu().numpy()
        dotted = (torch.argmax(out['dotted_probs'][0], dim=-1) == 1).cpu().numpy()
        triplet = (torch.argmax(out['triplet_probs'][0], dim=-1) == 1).cpu().numpy()
        has_rest = (torch.argmax(out['rest_probs'][0], dim=-1) == 1).cpu().numpy()

        type_conf = torch.max(out['type_probs'][0], dim=-1).values.cpu().numpy()
        rest_conf = torch.max(out['rest_probs'][0], dim=-1).values.cpu().numpy()

        results = []
        for i in range(len(features)):
            t = int(type_idx[i])
            d = bool(dotted[i])
            tr = bool(triplet[i])
            note_type = self.NOTE_TYPES[t]

            base_beats = self.NOTE_TYPE_BEATS[note_type]
            if d:
                base_beats *= 1.5
            if tr:
                base_beats *= 2.0 / 3.0

            results.append({
                'note_type': note_type,
                'dotted': d,
                'is_triplet': tr,
                'has_rest': bool(has_rest[i]),
                'confidence': float(type_conf[i]),
                'rest_confidence': float(rest_conf[i]),
                'beats': base_beats,
            })

        return results


# ─── GPU Transformer Model Singleton ─────────────────────────────────────────

_gpu_transformer_model: Optional[GpuRhythmTransformer] = None
_gpu_transformer_model_loaded = False


def get_gpu_transformer_model() -> Optional[GpuRhythmTransformer]:
    """Lazy-load the GPU Transformer rhythm model (singleton)."""
    global _gpu_transformer_model, _gpu_transformer_model_loaded

    if _gpu_transformer_model_loaded:
        return _gpu_transformer_model

    model_paths = [
        os.path.join(os.path.dirname(__file__), 'rhythm_training', 'rhythm_transformer.pt'),
        os.path.join(os.path.dirname(__file__), 'rhythm_transformer.pt'),
        '/root/rhythm_training/rhythm_transformer.pt',
    ]

    for path in model_paths:
        if os.path.exists(path):
            try:
                _gpu_transformer_model = GpuRhythmTransformer()
                _gpu_transformer_model.load_from_pt(path)
                break
            except Exception as e:
                print(f"[GPU Transformer] Failed to load from {path}: {e}")
                _gpu_transformer_model = None

    if _gpu_transformer_model is None:
        print("[GPU Transformer] Model not found, will use MLP/heuristic fallback")

    _gpu_transformer_model_loaded = True
    return _gpu_transformer_model


def gpu_extract_features_v2(notes: List[Dict], bpm: float,
                            use_ioi_as_duration: bool = True) -> np.ndarray:
    """
    Vectorized feature extraction for the Transformer model.
    Returns (n_notes, 10) — same 8 features as MLP plus:
      - feature 8: rest_gap_beats (observed gap to next note, from timing)
      - feature 9: next_ioi_beats (forward IOI context)
    """
    n = len(notes)
    if n == 0:
        return np.zeros((0, 10), dtype=np.float32)

    beat_duration = 60.0 / bpm
    measure_duration = beat_duration * 4

    onsets = np.array([note.get('time_seconds', 0) for note in notes], dtype=np.float32)
    durations = np.array([note.get('duration_seconds', 0.5) for note in notes], dtype=np.float32)
    pitches = np.array([note.get('midi_note', 60) for note in notes], dtype=np.float32)

    # IOIs
    iois = np.zeros(n, dtype=np.float32)
    iois[:-1] = np.diff(onsets)
    iois[-1] = durations[-1]

    if use_ioi_as_duration:
        primary_dur = iois
    else:
        primary_dur = durations

    dur_beats = primary_dur / beat_duration
    ioi_beats = iois / beat_duration
    beat_pos = (onsets % beat_duration) / beat_duration
    measure_pos = (onsets % measure_duration) / measure_duration

    prev_ioi_beats = np.ones(n, dtype=np.float32)
    prev_ioi_beats[1:] = np.diff(onsets) / beat_duration

    dur_ioi_ratio = durations / np.maximum(iois, 0.01)
    norm_pitch = (pitches - 60) / 40.0
    norm_tempo = np.full(n, bpm / 120.0, dtype=np.float32)

    # NEW features for Transformer
    # Rest gap: offset to next onset (may be negative if overlap)
    offsets = onsets + durations
    rest_gap = np.zeros(n, dtype=np.float32)
    rest_gap[:-1] = np.maximum(onsets[1:] - offsets[:-1], 0)
    rest_gap_beats = rest_gap / beat_duration

    # Next IOI (forward context)
    next_ioi_beats = np.zeros(n, dtype=np.float32)
    next_ioi_beats[:-1] = ioi_beats[1:]
    next_ioi_beats[-1] = ioi_beats[-1]

    features = np.stack([
        dur_beats, ioi_beats, beat_pos, measure_pos,
        prev_ioi_beats, dur_ioi_ratio, norm_pitch, norm_tempo,
        rest_gap_beats, next_ioi_beats,
    ], axis=1)

    return features


# ─── GPU Custom Piano Transcription Model ────────────────────────────────────

class GpuPianoTranscriber(nn.Module):
    """
    GPU inference wrapper for custom piano transcription model.

    Loads a trained PianoTranscriptionModel checkpoint and provides
    a transcribe() method matching the ByteDance interface so it can
    drop in as a replacement.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.config = None
        self.model = None

    def load_from_pt(self, path: str):
        """Load a trained transcription model checkpoint."""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        config = checkpoint.get('config', {})
        self.config = config

        # Import and build the model architecture
        from rhythm_training.train_transcription import PianoTranscriptionModel  # type: ignore

        self.model = PianoTranscriptionModel(
            n_mels=config.get('n_mels', 229),
            conv_channels=config.get('conv_channels', [32, 64, 128]),
            d_model=config.get('d_model', 256),
            n_heads=config.get('n_heads', 8),
            n_layers=config.get('n_layers', 4),
            d_ff=config.get('d_ff', 1024),
            n_keys=config.get('n_keys', 88),
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(DEVICE)
        self.model.eval()

        self.initialized = True
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[GPU Transcriber] Loaded custom model from {path} "
              f"({n_params:,} params, d_model={config.get('d_model')})")

    @torch.no_grad()
    def transcribe(
        self,
        audio: np.ndarray,
        onset_threshold: float = 0.4,
        frame_threshold: float = 0.3,
        min_note_duration: float = 0.05,
    ) -> Dict:
        """
        Transcribe audio to note events.

        Returns dict matching ByteDance format:
            {'est_note_events': [{'onset_time', 'offset_time', 'midi_note', 'velocity'}, ...]}
        """
        if not self.initialized or self.model is None:
            return {'est_note_events': []}

        from rhythm_training.train_transcription import transcribe_audio  # type: ignore
        events = transcribe_audio(
            audio, self.model, DEVICE,
            sr=self.config.get('sample_rate', 16000),
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            min_note_duration=min_note_duration,
        )
        return {'est_note_events': events}


# ─── GPU Transcription Model Singleton ────────────────────────────────────────

_gpu_transcriber: Optional[GpuPianoTranscriber] = None
_gpu_transcriber_loaded = False
_gpu_transcriber_status: Dict[str, object] = {
    'model': 'custom_transcriber',
    'attempted': False,
    'initialized': False,
    'use_gpu': USE_GPU,
    'searched_paths': [],
    'selected_path': None,
    'reason': 'not_attempted',
    'last_error': None,
}


def get_gpu_transcriber_status() -> Dict[str, object]:
    """Return the current custom transcriber loader status."""
    return dict(_gpu_transcriber_status)


def get_gpu_transcriber() -> Optional[GpuPianoTranscriber]:
    """Lazy-load the custom GPU piano transcription model (singleton)."""
    global _gpu_transcriber, _gpu_transcriber_loaded, _gpu_transcriber_status

    if _gpu_transcriber_loaded:
        return _gpu_transcriber

    model_paths = [
        os.path.join(os.path.dirname(__file__), 'rhythm_training', 'piano_transcription.pt'),
        os.path.join(os.path.dirname(__file__), 'piano_transcription.pt'),
        '/root/rhythm_training/piano_transcription.pt',
    ]

    _gpu_transcriber_status.update({
        'attempted': True,
        'initialized': False,
        'use_gpu': USE_GPU,
        'searched_paths': list(model_paths),
        'selected_path': None,
        'reason': 'loading',
        'last_error': None,
    })

    if not USE_GPU:
        _gpu_transcriber_status.update({
            'reason': 'cuda_unavailable',
            'last_error': 'CUDA not available',
        })
        _gpu_transcriber_loaded = True
        return None

    found_checkpoint = False

    for path in model_paths:
        if os.path.exists(path):
            found_checkpoint = True
            _gpu_transcriber_status['selected_path'] = path
            try:
                _gpu_transcriber = GpuPianoTranscriber()
                _gpu_transcriber.load_from_pt(path)
                _gpu_transcriber_status.update({
                    'initialized': True,
                    'reason': 'initialized',
                    'last_error': None,
                })
                break
            except Exception as e:
                print(f"[GPU Transcriber] Failed to load from {path}: {e}")
                _gpu_transcriber = None
                _gpu_transcriber_status.update({
                    'reason': 'load_failed',
                    'last_error': f"{type(e).__name__}: {e}",
                })

    # Note: Custom model (piano_transcription.pt) is optional - ensemble model is primary
    # No warning needed since ensemble_transcription.pt is the main transcriber
    if _gpu_transcriber is None and _gpu_transcriber_status.get('reason') != 'load_failed':
        _gpu_transcriber_status.update({
            'reason': 'checkpoint_missing' if not found_checkpoint else 'not_initialized',
            'last_error': None,
        })

    _gpu_transcriber_loaded = True
    return _gpu_transcriber


# ─── Parallel Phrasing Utilities ────────────────────────────────────────────

def parallel_process_hands(bass_func, treble_func, bass_args, treble_args):
    """
    Run bass and treble processing in parallel using ThreadPoolExecutor.
    Returns (bass_result, treble_result).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=2) as executor:
        bass_future = executor.submit(bass_func, *bass_args)
        treble_future = executor.submit(treble_func, *treble_args)
        bass_result = bass_future.result()
        treble_result = treble_future.result()

    return bass_result, treble_result


# ─── GPU Multi-Resolution Ensemble Transcriber ────────────────────────────────

class GpuEnsembleTranscriber(nn.Module):
    """
    GPU inference wrapper for multi-resolution ensemble transcription model.

    Combines:
      - MultiResFeatureExtractor (549 features from 3 STFTs + CQT + chroma + onsets + HPSS)
      - PitchAwareTranscriber (pitch-aligned per-key processing, ~50K params)

    Drop-in replacement for ByteDance PianoTranscription interface.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.config = None
        self.extractor = None
        self.model = None

    def load_from_pt(self, path: str):
        """Load a trained ensemble model checkpoint."""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        config = checkpoint.get('config', {})
        self.config = config

        # Build feature extractor
        from rhythm_training.train_ensemble import MultiResFeatureExtractor  # type: ignore
        from rhythm_training.train_ensemble import _build_model_from_config  # type: ignore

        self.extractor = MultiResFeatureExtractor(
            sr=config.get('sample_rate', 16000),
            hop_length=config.get('hop_length', 512),
            device=DEVICE,
            hop_lengths=config.get('hop_lengths', None),
        )

        # Build and load model
        self.model = _build_model_from_config(config)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(DEVICE)
        self.model.eval()

        self.initialized = True
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[GPU Ensemble] Loaded from {path} ({n_params:,} params)")

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray, onset_threshold: float = 0.7,
                   frame_threshold: float = 0.5,
                   min_note_duration: float = 0.05,
                   min_velocity: int = 15,
                   filter_harmonics: bool = True) -> Dict:
        """
        Transcribe audio to note events.

        Returns dict matching ByteDance format:
            {'est_note_events': [{'onset_time', 'offset_time', 'midi_note', 'velocity'}, ...]}
        """
        import time
        timings = {}
        t_total = time.perf_counter()
        
        if not self.initialized:
            return {'est_note_events': [], '_inference_timing_ms': {'error': 'not_initialized'}}

        sr = self.config.get('sample_rate', 16000)
        hop = self.config.get('hop_length', 512)

        # Audio → GPU tensor
        t0 = time.perf_counter()
        audio_t = torch.from_numpy(audio).float().to(DEVICE)
        timings['audio_to_gpu'] = (time.perf_counter() - t0) * 1000

        # Extract multi-resolution features
        t0 = time.perf_counter()
        features = self.extractor.extract(audio_t)  # (1, T, 373)
        torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
        timings['feature_extraction'] = (time.perf_counter() - t0) * 1000

        # Process in overlapping chunks for long audio
        t0 = time.perf_counter()
        n_frames = features.size(1)
        chunk_frames = int(10.0 * sr / hop)  # 10 second chunks
        overlap = chunk_frames // 4
        step = chunk_frames - overlap

        n_keys = self.config.get('n_keys', 88)
        # Get note value classes from config or model output (supports 6 or 10 classes)
        n_note_value_classes = self.config.get('n_note_value_classes', 10)
        all_onset = np.zeros((n_frames, n_keys), dtype=np.float32)
        all_frame = np.zeros((n_frames, n_keys), dtype=np.float32)
        all_vel = np.zeros((n_frames, n_keys), dtype=np.float32)
        all_note_value = np.zeros((n_frames, n_keys, n_note_value_classes), dtype=np.float32)
        counts = np.zeros(n_frames, dtype=np.float32)

        n_chunks = 0
        for start in range(0, n_frames, step):
            end = min(start + chunk_frames, n_frames)
            chunk = features[:, start:end, :]  # (1, chunk_len, 373)

            out = self.model(chunk)

            onset_p = torch.sigmoid(out['onset_logits'][0]).cpu().numpy()
            frame_p = torch.sigmoid(out['frame_logits'][0]).cpu().numpy()
            vel = out['velocity'][0].cpu().numpy()
            # note_value_logits: (1, chunk_len, 88, N) -> softmax -> (chunk_len, 88, N)
            nv_probs = F.softmax(out['note_value_logits'][0], dim=-1).cpu().numpy()
            # Handle potential mismatch if old model has 6 classes but config says 10
            actual_nv_classes = nv_probs.shape[-1]
            if actual_nv_classes != n_note_value_classes:
                # Pad or truncate to match expected size
                if actual_nv_classes < n_note_value_classes:
                    pad = np.zeros((*nv_probs.shape[:-1], n_note_value_classes - actual_nv_classes), dtype=np.float32)
                    nv_probs = np.concatenate([nv_probs, pad], axis=-1)
                else:
                    nv_probs = nv_probs[..., :n_note_value_classes]

            actual_len = end - start
            all_onset[start:end] += onset_p[:actual_len]
            all_frame[start:end] += frame_p[:actual_len]
            all_vel[start:end] += vel[:actual_len]
            all_note_value[start:end] += nv_probs[:actual_len]
            counts[start:end] += 1.0
            n_chunks += 1
        
        torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
        timings['model_inference'] = (time.perf_counter() - t0) * 1000

        # Average overlapping regions
        t0 = time.perf_counter()
        counts = np.maximum(counts, 1.0)
        all_onset /= counts[:, None]
        all_frame /= counts[:, None]
        all_vel /= counts[:, None]
        all_note_value /= counts[:, None, None]

        # Decode note events with post-processing (including note values)
        from rhythm_training.train_ensemble import decode_note_events  # type: ignore
        events = decode_note_events(
            all_onset, all_frame, all_vel,
            note_value_probs=all_note_value,
            sr=sr, hop=hop,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            min_note_duration=min_note_duration,
            min_velocity=min_velocity,
            use_peak_picking=True,
            filter_harmonics=filter_harmonics,
        )
        timings['decode_notes'] = (time.perf_counter() - t0) * 1000
        
        timings['total'] = (time.perf_counter() - t_total) * 1000
        timings['audio_duration_ms'] = len(audio) / sr * 1000
        timings['n_frames'] = n_frames
        timings['n_chunks'] = n_chunks
        timings['real_time_factor'] = timings['total'] / timings['audio_duration_ms'] if timings['audio_duration_ms'] > 0 else 0
        
        print(f"[TIMING] Ensemble.transcribe: audio_to_gpu={timings['audio_to_gpu']:.1f}ms, features={timings['feature_extraction']:.1f}ms, model={timings['model_inference']:.1f}ms, decode={timings['decode_notes']:.1f}ms | TOTAL={timings['total']:.1f}ms for {timings['audio_duration_ms']:.0f}ms audio (RTF={timings['real_time_factor']:.2f}x)")

        return {'est_note_events': events, '_inference_timing_ms': timings}


# ─── Ensemble Transcriber Singleton ───────────────────────────────────────────

_gpu_ensemble_transcriber: Optional[GpuEnsembleTranscriber] = None
_gpu_ensemble_transcriber_loaded = False


def get_gpu_ensemble_transcriber() -> Optional[GpuEnsembleTranscriber]:
    """Lazy-load the GPU ensemble transcriber (singleton)."""
    global _gpu_ensemble_transcriber, _gpu_ensemble_transcriber_loaded

    if _gpu_ensemble_transcriber_loaded:
        return _gpu_ensemble_transcriber

    model_paths = [
        os.path.join(os.path.dirname(__file__), 'rhythm_training', 'ensemble_transcription.pt'),
        os.path.join(os.path.dirname(__file__), 'ensemble_transcription.pt'),
        '/root/rhythm_training/ensemble_transcription.pt',
    ]

    for path in model_paths:
        if os.path.exists(path):
            try:
                _gpu_ensemble_transcriber = GpuEnsembleTranscriber()
                _gpu_ensemble_transcriber.load_from_pt(path)
                break
            except Exception as e:
                print(f"[GPU Ensemble] Failed to load from {path}: {e}")
                _gpu_ensemble_transcriber = None

    if _gpu_ensemble_transcriber is None:
        print("[GPU Ensemble] Model not found, will use fallback transcriber")

    _gpu_ensemble_transcriber_loaded = True
    return _gpu_ensemble_transcriber


# ─── GPU Mel Baseline Transcriber ──────────────────────────────────────────

class GpuMelBaselineTranscriber(nn.Module):
    """
    GPU inference wrapper for mel-only baseline transcription model.

    Uses a single log-mel spectrogram (229 bins) as input instead of the
    full 1098-feature multi-resolution stack. Paired with a larger
    ConvStack + Conformer model that learns its own representations.

    Drop-in replacement for GpuEnsembleTranscriber interface.
    """

    def __init__(self):
        super().__init__()
        self.initialized = False
        self.config = None
        self.extractor = None
        self.model = None

    def load_from_pt(self, path: str):
        """Load a trained mel baseline model checkpoint."""
        checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
        config = checkpoint.get('config', {})
        self.config = config

        from rhythm_training.train_mel_baseline import MelFeatureExtractor  # type: ignore
        from rhythm_training.train_mel_baseline import _build_model_from_config  # type: ignore

        self.extractor = MelFeatureExtractor(
            sr=config.get('sample_rate', 16000),
            hop_length=config.get('hop_length', 256),
            n_fft=config.get('n_fft', 2048),
            n_mels=config.get('n_mels', 229),
            device=DEVICE,
        )

        self.model = _build_model_from_config(config)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.to(DEVICE)
        self.model.eval()

        self.initialized = True
        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[GPU MelBaseline] Loaded from {path} ({n_params:,} params)")

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray, onset_threshold: float = 0.4,
                   frame_threshold: float = 0.5,
                   min_note_duration: float = 0.05,
                   min_velocity: int = 15,
                   filter_harmonics: bool = True) -> Dict:
        """
        Transcribe audio to note events.

        Returns dict matching ByteDance/Ensemble format:
            {'est_note_events': [...], '_inference_timing_ms': {...}}
        """
        import time
        timings = {}
        t_total = time.perf_counter()

        if not self.initialized:
            return {'est_note_events': [], '_inference_timing_ms': {'error': 'not_initialized'}}

        sr = self.config.get('sample_rate', 16000)
        hop = self.config.get('hop_length', 256)

        t0 = time.perf_counter()
        audio_t = torch.from_numpy(audio).float().to(DEVICE)
        timings['audio_to_gpu'] = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        features = self.extractor.extract(audio_t)  # (1, T, 229)
        torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
        timings['feature_extraction'] = (time.perf_counter() - t0) * 1000

        # Process in overlapping chunks for long audio
        t0 = time.perf_counter()
        n_frames = features.size(1)
        chunk_frames = int(10.0 * sr / hop)
        overlap = chunk_frames // 4
        step = chunk_frames - overlap

        n_keys = self.config.get('n_keys', 88)
        n_note_value_classes = self.config.get('n_note_value_classes', 10)
        all_onset = np.zeros((n_frames, n_keys), dtype=np.float32)
        all_frame = np.zeros((n_frames, n_keys), dtype=np.float32)
        all_vel = np.zeros((n_frames, n_keys), dtype=np.float32)
        all_note_value = np.zeros((n_frames, n_keys, n_note_value_classes), dtype=np.float32)
        counts = np.zeros(n_frames, dtype=np.float32)

        n_chunks = 0
        for start in range(0, n_frames, step):
            end = min(start + chunk_frames, n_frames)
            chunk = features[:, start:end, :]

            out = self.model(chunk)

            onset_p = torch.sigmoid(out['onset_logits'][0]).cpu().numpy()
            frame_p = torch.sigmoid(out['frame_logits'][0]).cpu().numpy()
            vel = out['velocity'][0].cpu().numpy()
            nv_probs = F.softmax(out['note_value_logits'][0], dim=-1).cpu().numpy()

            actual_nv_classes = nv_probs.shape[-1]
            if actual_nv_classes != n_note_value_classes:
                if actual_nv_classes < n_note_value_classes:
                    pad = np.zeros((*nv_probs.shape[:-1], n_note_value_classes - actual_nv_classes), dtype=np.float32)
                    nv_probs = np.concatenate([nv_probs, pad], axis=-1)
                else:
                    nv_probs = nv_probs[..., :n_note_value_classes]

            actual_len = end - start
            all_onset[start:end] += onset_p[:actual_len]
            all_frame[start:end] += frame_p[:actual_len]
            all_vel[start:end] += vel[:actual_len]
            all_note_value[start:end] += nv_probs[:actual_len]
            counts[start:end] += 1.0
            n_chunks += 1

        torch.cuda.synchronize() if DEVICE.type == 'cuda' else None
        timings['model_inference'] = (time.perf_counter() - t0) * 1000

        # Average overlapping regions
        t0 = time.perf_counter()
        counts = np.maximum(counts, 1.0)
        all_onset /= counts[:, None]
        all_frame /= counts[:, None]
        all_vel /= counts[:, None]
        all_note_value /= counts[:, None, None]

        from rhythm_training.train_ensemble import decode_note_events  # type: ignore
        events = decode_note_events(
            all_onset, all_frame, all_vel,
            note_value_probs=all_note_value,
            sr=sr, hop=hop,
            onset_threshold=onset_threshold,
            frame_threshold=frame_threshold,
            min_note_duration=min_note_duration,
            min_velocity=min_velocity,
            use_peak_picking=True,
            filter_harmonics=filter_harmonics,
        )
        timings['decode_notes'] = (time.perf_counter() - t0) * 1000

        timings['total'] = (time.perf_counter() - t_total) * 1000
        timings['audio_duration_ms'] = len(audio) / sr * 1000
        timings['n_frames'] = n_frames
        timings['n_chunks'] = n_chunks
        timings['real_time_factor'] = timings['total'] / timings['audio_duration_ms'] if timings['audio_duration_ms'] > 0 else 0

        print(f"[TIMING] MelBaseline.transcribe: audio_to_gpu={timings['audio_to_gpu']:.1f}ms, features={timings['feature_extraction']:.1f}ms, model={timings['model_inference']:.1f}ms, decode={timings['decode_notes']:.1f}ms | TOTAL={timings['total']:.1f}ms for {timings['audio_duration_ms']:.0f}ms audio (RTF={timings['real_time_factor']:.2f}x)")

        return {'est_note_events': events, '_inference_timing_ms': timings}


# ─── Mel Baseline Transcriber Singleton ─────────────────────────────────────

_gpu_mel_baseline_transcriber: Optional[GpuMelBaselineTranscriber] = None
_gpu_mel_baseline_transcriber_loaded = False
_gpu_mel_baseline_transcriber_status: Dict[str, object] = {
    'model': 'mel_baseline',
    'attempted': False,
    'initialized': False,
    'use_gpu': USE_GPU,
    'searched_paths': [],
    'selected_path': None,
    'reason': 'not_attempted',
    'last_error': None,
}


def get_gpu_mel_baseline_transcriber_status() -> Dict[str, object]:
    """Return the current mel baseline loader status."""
    return dict(_gpu_mel_baseline_transcriber_status)


def get_gpu_mel_baseline_transcriber() -> Optional[GpuMelBaselineTranscriber]:
    """Lazy-load the GPU mel baseline transcriber (singleton)."""
    global _gpu_mel_baseline_transcriber, _gpu_mel_baseline_transcriber_loaded
    global _gpu_mel_baseline_transcriber_status

    if _gpu_mel_baseline_transcriber_loaded:
        return _gpu_mel_baseline_transcriber

    override_path = os.environ.get('LIVE_MEL_BASELINE_MODEL_PATH') or os.environ.get('MEL_BASELINE_MODEL_PATH')
    model_paths = []
    if override_path:
        model_paths.append(override_path)
    model_paths.extend([
        os.path.join(os.path.dirname(__file__), 'rhythm_training', 'mel_baseline_transcription.pt'),
        os.path.join(os.path.dirname(__file__), 'mel_baseline_transcription.pt'),
        '/root/rhythm_training/mel_baseline_transcription.pt',
    ])

    _gpu_mel_baseline_transcriber_status.update({
        'attempted': True,
        'initialized': False,
        'use_gpu': USE_GPU,
        'searched_paths': list(model_paths),
        'selected_path': None,
        'reason': 'loading',
        'last_error': None,
    })

    if not USE_GPU:
        _gpu_mel_baseline_transcriber_status.update({
            'reason': 'cuda_unavailable',
            'last_error': 'CUDA not available',
        })
        _gpu_mel_baseline_transcriber_loaded = True
        return None

    found_checkpoint = False

    for path in model_paths:
        if os.path.exists(path):
            found_checkpoint = True
            _gpu_mel_baseline_transcriber_status['selected_path'] = path
            try:
                _gpu_mel_baseline_transcriber = GpuMelBaselineTranscriber()
                _gpu_mel_baseline_transcriber.load_from_pt(path)
                _gpu_mel_baseline_transcriber_status.update({
                    'initialized': True,
                    'reason': 'initialized',
                    'last_error': None,
                })
                break
            except Exception as e:
                print(f"[GPU MelBaseline] Failed to load from {path}: {e}")
                _gpu_mel_baseline_transcriber = None
                _gpu_mel_baseline_transcriber_status.update({
                    'reason': 'load_failed',
                    'last_error': f"{type(e).__name__}: {e}",
                })

    if _gpu_mel_baseline_transcriber is None:
        print("[GPU MelBaseline] Model not found (not trained yet?)")
        if _gpu_mel_baseline_transcriber_status.get('reason') != 'load_failed':
            _gpu_mel_baseline_transcriber_status.update({
                'reason': 'checkpoint_missing' if not found_checkpoint else 'not_initialized',
                'last_error': None,
            })

    _gpu_mel_baseline_transcriber_loaded = True
    return _gpu_mel_baseline_transcriber
