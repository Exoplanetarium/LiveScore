"""
Multi-resolution ensemble transcriber for LiveScore.

Architecture:
  1. Multi-resolution feature extraction (GPU-parallel):
     - 3 mel spectrograms at different STFT window sizes (1024/2048/4096)
     - CQT via filterbank on 4096-STFT
     - Chromagram (folded CQT)
     - 9 onset detection functions (flux/energy/HFC x 3 resolutions)
     - HPSS harmonic + percussive mel
     Total: 549 features per hop × 2 hops = 1098 features per frame

  2. PitchAwareTranscriber (~50K params):
     - Reshapes 1098 features into per-key (88, 12) + global (42)
     - Per-key shared MLP for spectral encoding
     - Key-axis Conv1d for harmonic/octave patterns
     - Per-key TCN (dilated causal convolutions) for temporal context
     - Per-key output heads: onset, frame, velocity, note_value

Key design: pitch-aligned feature processing — each key gets its own
spectral views instead of mixing all 1098 features through a flat Conv1d.

Training data: MAESTRO v3.0.0 (aligned audio + MIDI)
Features are computed on-the-fly during training (no preprocessed dataset).

Usage:
    # 1. Prepare segment index from MAESTRO
    python train_ensemble.py --prepare

    # 2. (Optional) Precompute features for 5-10x faster training
    python train_ensemble.py --precompute

    # 3. Train
    python train_ensemble.py --train --epochs 50 --batch-size 8

    # 4. Benchmark against ByteDance
    python train_ensemble.py --benchmark
"""

import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ─── Constants ───────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000
HOP_LENGTH = 512
HOP_LENGTH_FINE = 256
HOP_LENGTHS = [HOP_LENGTH, HOP_LENGTH_FINE]  # coarse + fine
N_MELS = 88               # per resolution (piano-key aligned)
PIANO_KEYS = 88
MIDI_OFFSET = 21           # A0
N_FFTS = [1024, 2048, 4096]
CQT_BINS = 88
CHROMA_BINS = 12
ONSET_FEATURES = 9         # 3 functions x 3 resolutions
HPSS_FEATURES = N_MELS * 2 # harmonic + percussive mel (from largest STFT)
N_FEATURES = N_MELS * len(N_FFTS) + CQT_BINS + CHROMA_BINS + ONSET_FEATURES + HPSS_FEATURES  # 549
N_FEATURES_MULTI_HOP = N_FEATURES * len(HOP_LENGTHS)  # 1098

SEGMENT_SECONDS = 10.0
SEGMENT_FRAMES = int(SEGMENT_SECONDS * SAMPLE_RATE / HOP_LENGTH)
SEGMENT_FRAMES_FINE = int(SEGMENT_SECONDS * SAMPLE_RATE / HOP_LENGTH_FINE)

# Note-value classes for rhythm prediction head (expanded to include dotted)
# 10 classes: base values + dotted variants for common note values
NOTE_VALUE_CLASSES = 10
NOTE_VALUE_BEATS = [
    0.125,   # 32nd
    0.25,    # 16th
    0.5,     # eighth
    0.75,    # dotted eighth
    1.0,     # quarter
    1.5,     # dotted quarter
    2.0,     # half
    3.0,     # dotted half
    4.0,     # whole
    6.0,     # dotted whole
]
NOTE_VALUE_NAMES = [
    '32nd', '16th', 'eighth', 'dotted_eighth',
    'quarter', 'dotted_quarter', 'half', 'dotted_half',
    'whole', 'dotted_whole'
]

MAESTRO_DIR = Path(__file__).parent / "maestro_midi"
MAESTRO_CSV = MAESTRO_DIR / "maestro-v3.0.0.csv"
INDEX_DIR = Path(__file__).parent / "ensemble_index"
FEATURES_DIR = Path(__file__).parent / "precomputed_features"
MODEL_PATH = Path(__file__).parent / "ensemble_transcription.pt"


# ─── Multi-Resolution Feature Extractor ─────────────────────────────────────

class MultiResFeatureExtractor:
    """
    GPU-accelerated multi-resolution feature extraction.

    Computes 549 features per frame from audio per hop length:
      - 3 x 88 mel spectrograms (n_fft = 1024, 2048, 4096)
      - 88 CQT bins (filterbank on 4096-STFT)
      - 12 chroma bins (folded from CQT)
      - 9 onset functions (spectral flux, RMS energy, HFC x 3 resolutions)
      - 2 x 88 HPSS mel spectrograms (harmonic + percussive from 4096-STFT)

    Multi-hop mode: extracts features at multiple hop lengths (e.g. 512 + 256),
    upsamples coarser features to match the finest resolution, and concatenates
    along the feature dimension (e.g. 549 * 2 = 1098 features).

    All filterbanks are precomputed and cached on GPU.
    """

    def __init__(self, sr: int = SAMPLE_RATE, hop_length: int = HOP_LENGTH,
                 n_mels: int = N_MELS, n_ffts: List[int] = None,
                 device: torch.device = None,
                 hop_lengths: List[int] = None):
        self.sr = sr
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_ffts = n_ffts or N_FFTS
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.hop_lengths = hop_lengths or [hop_length]
        self.multi_hop = len(self.hop_lengths) > 1

        # Precompute STFT windows
        self.windows = {}
        for n_fft in self.n_ffts:
            self.windows[n_fft] = torch.hann_window(n_fft, device=self.device)

        # Precompute mel filterbanks
        self.mel_fbs = {}
        for n_fft in self.n_ffts:
            self.mel_fbs[n_fft] = self._build_mel_filterbank(n_fft)

        # Precompute CQT filterbank (applied to largest STFT)
        self.cqt_fb = self._build_cqt_filterbank(max(self.n_ffts))

    def _build_mel_filterbank(self, n_fft: int) -> torch.Tensor:
        """Build mel filterbank: (n_mels, n_fft//2+1) on GPU."""
        import librosa
        fb = librosa.filters.mel(
            sr=self.sr, n_fft=n_fft, n_mels=self.n_mels,
            fmin=27.5, fmax=self.sr // 2,
        )
        return torch.from_numpy(fb.astype(np.float32)).to(self.device)

    def _build_cqt_filterbank(self, n_fft: int) -> torch.Tensor:
        """Build CQT filterbank: (88, n_fft//2+1) on GPU."""
        bins_per_octave = 12
        n_bins = CQT_BINS
        fmin = 27.5
        Q = 1.0 / (2.0 ** (1.0 / bins_per_octave) - 1.0)
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sr)

        filterbank = np.zeros((n_bins, len(freqs)), dtype=np.float32)
        for k in range(n_bins):
            f_center = fmin * (2.0 ** (k / bins_per_octave))
            bw = f_center / Q
            sigma = bw / 4.0
            if sigma < 1e-6:
                continue
            weights = np.exp(-0.5 * ((freqs - f_center) / sigma) ** 2)
            weights[freqs < f_center - 3 * sigma] = 0
            weights[freqs > f_center + 3 * sigma] = 0
            w_sum = np.sum(weights)
            if w_sum > 0:
                filterbank[k] = weights / w_sum

        return torch.from_numpy(filterbank).to(self.device)

    def _extract_single_hop(self, audio: torch.Tensor, hop_length: int) -> torch.Tensor:
        """
        Extract 549 features from audio at a specific hop length.

        Args:
            audio: (batch, samples) tensor on self.device
            hop_length: hop size in samples

        Returns:
            features: (batch, n_frames, 549) tensor on self.device
        """
        # Compute 3 STFTs (same hop, different window sizes)
        magnitudes = {}
        for n_fft in self.n_ffts:
            stft = torch.stft(
                audio, n_fft, hop_length=hop_length,
                window=self.windows[n_fft], return_complex=True, center=True,
            )  # (batch, n_fft//2+1, n_frames)
            magnitudes[n_fft] = torch.abs(stft)

        # Align frame counts (center=True with same hop -> same count, but guard)
        n_frames = min(m.size(-1) for m in magnitudes.values())
        for n_fft in self.n_ffts:
            magnitudes[n_fft] = magnitudes[n_fft][:, :, :n_frames]

        parts = []

        # ── 1. Multi-resolution mel spectrograms (3 x 88 = 264) ──
        for n_fft in self.n_ffts:
            mel = torch.matmul(
                self.mel_fbs[n_fft].unsqueeze(0),
                magnitudes[n_fft],
            )  # (batch, n_mels, T)
            mel = torch.log(mel + 1e-6)
            parts.append(mel)

        # ── 2. CQT (88) ──
        largest_mag = magnitudes[max(self.n_ffts)]
        cqt = torch.matmul(
            self.cqt_fb.unsqueeze(0), largest_mag,
        )  # (batch, 88, T)
        cqt_log = torch.log(cqt + 1e-6)
        parts.append(cqt_log)

        # ── 3. Chromagram (12) ──
        batch_size = cqt.size(0)
        cqt_padded = F.pad(cqt, (0, 0, 0, 96 - CQT_BINS))  # (batch, 96, T)
        chroma = cqt_padded.view(batch_size, 8, 12, n_frames).sum(dim=1)
        chroma = chroma / (chroma.sum(dim=1, keepdim=True) + 1e-8)
        parts.append(chroma)

        # ── 4. Onset detection functions (9) ──
        for n_fft in self.n_ffts:
            mag = magnitudes[n_fft]

            # Spectral flux (half-wave rectified difference)
            diff = torch.diff(mag, dim=-1)
            diff = torch.clamp(diff, min=0)
            flux = diff.sum(dim=1)
            flux = F.pad(flux, (1, 0))
            flux = flux / (flux.max(dim=-1, keepdim=True).values + 1e-8)
            parts.append(flux.unsqueeze(1))

            # RMS energy
            rms = torch.sqrt(torch.mean(mag ** 2, dim=1))
            rms = rms / (rms.max(dim=-1, keepdim=True).values + 1e-8)
            parts.append(rms.unsqueeze(1))

            # High-frequency content (frequency-weighted energy)
            freq_weights = torch.linspace(0, 1, mag.size(1), device=self.device)
            hfc = (mag ** 2 * freq_weights.view(1, -1, 1)).sum(dim=1)
            hfc = torch.sqrt(hfc)
            hfc = hfc / (hfc.max(dim=-1, keepdim=True).values + 1e-8)
            parts.append(hfc.unsqueeze(1))

        # ── 5. HPSS: harmonic + percussive mel spectrograms (2 x 88 = 176) ──
        # Use the largest STFT magnitude for best frequency resolution
        largest_mag = magnitudes[max(self.n_ffts)]  # (batch, freq, T)
        # Median-filter based HPSS on magnitude spectrogram
        # Harmonic: median filter along time axis (captures sustained tones)
        # Percussive: median filter along frequency axis (captures transients)
        hpss_kernel = 17  # median filter length (odd, moderate for GPU efficiency)
        # Pad for median filtering
        # Time median (harmonic) - filter along dim=-1
        mag_pad_t = F.pad(largest_mag, (hpss_kernel // 2, hpss_kernel // 2), mode='reflect')
        harmonic_mag = mag_pad_t.unfold(-1, hpss_kernel, 1).median(dim=-1).values
        # Frequency median (percussive) - filter along dim=1
        mag_pad_f = F.pad(largest_mag, (0, 0, hpss_kernel // 2, hpss_kernel // 2), mode='reflect')
        percussive_mag = mag_pad_f.unfold(1, hpss_kernel, 1).median(dim=-1).values
        # Soft masks (Wiener-style)
        harmonic_mask = harmonic_mag ** 2 / (harmonic_mag ** 2 + percussive_mag ** 2 + 1e-10)
        percussive_mask = percussive_mag ** 2 / (harmonic_mag ** 2 + percussive_mag ** 2 + 1e-10)
        harmonic_spec = largest_mag * harmonic_mask
        percussive_spec = largest_mag * percussive_mask
        # Apply mel filterbank to each
        mel_fb_largest = self.mel_fbs[max(self.n_ffts)]
        harmonic_mel = torch.log(torch.matmul(mel_fb_largest.unsqueeze(0), harmonic_spec) + 1e-6)
        percussive_mel = torch.log(torch.matmul(mel_fb_largest.unsqueeze(0), percussive_spec) + 1e-6)
        parts.append(harmonic_mel)    # (batch, 88, T)
        parts.append(percussive_mel)  # (batch, 88, T)

        # Stack all features: (batch, N_FEATURES, T)
        all_features = torch.cat(parts, dim=1)

        # Transpose to model convention: (batch, T, N_FEATURES)
        return all_features.permute(0, 2, 1)

    def extract(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract features from audio, optionally at multiple hop lengths.

        Args:
            audio: (batch, samples) or (samples,) tensor, can be on any device.

        Returns:
            features: (batch, n_frames, n_features) tensor on self.device
                      In single-hop mode: n_features=549, n_frames depends on hop_length
                      In multi-hop mode: n_features=1098, n_frames at finest hop resolution
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        if not self.multi_hop:
            return self._extract_single_hop(audio, self.hop_lengths[0])

        # Multi-hop: extract at each hop length, upsample coarser to finest
        finest_hop = min(self.hop_lengths)
        features_by_hop = []
        target_T = None

        for hop in self.hop_lengths:
            feats = self._extract_single_hop(audio, hop)  # (batch, T_hop, 373)
            if hop == finest_hop:
                target_T = feats.size(1)
            features_by_hop.append((hop, feats))

        # Upsample coarser features to match finest resolution
        aligned = []
        for hop, feats in features_by_hop:
            if feats.size(1) == target_T:
                aligned.append(feats)
            else:
                # (batch, T_coarse, 373) -> interpolate -> (batch, target_T, 373)
                feats_t = feats.permute(0, 2, 1)  # (batch, 373, T_coarse)
                feats_up = F.interpolate(feats_t, size=target_T, mode='linear',
                                         align_corners=False)
                aligned.append(feats_up.permute(0, 2, 1))

        # Concatenate along feature dimension
        return torch.cat(aligned, dim=-1)  # (batch, target_T, 746)

    @property
    def n_features(self) -> int:
        """Total features per frame."""
        base = N_MELS * len(self.n_ffts) + CQT_BINS + CHROMA_BINS + ONSET_FEATURES + HPSS_FEATURES
        return base * len(self.hop_lengths)


# ─── Meta-Learner Model ─────────────────────────────────────────────────────

class EnsembleMetaLearner(nn.Module):
    """
    Conv1d + BiGRU meta-learner for multi-resolution feature fusion.

    ~3M params (configurable):
      - 3x Conv1d: local time-frequency pattern extraction
      - 3-layer BiGRU: temporal context for onset/frame prediction
      - 4 output heads: onset, frame, velocity, note_value (per 88 piano keys)
    """

    def __init__(self, n_features: int = N_FEATURES,
                 conv_channels: List[int] = None,
                 gru_hidden: int = 64, gru_layers: int = 2,
                 n_keys: int = PIANO_KEYS, dropout: float = 0.1):
        super().__init__()
        if conv_channels is None:
            conv_channels = [512, 256, 128]

        self.n_features = n_features
        self.n_keys = n_keys

        # Conv1d stack (operates on feature dim as channels, time as length)
        self.conv1 = nn.Conv1d(n_features, conv_channels[0], kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(conv_channels[0])
        self.conv2 = nn.Conv1d(conv_channels[0], conv_channels[1], kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_channels[1])
        self.conv3 = nn.Conv1d(conv_channels[1], conv_channels[2], kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(conv_channels[2])

        # BiGRU
        self.gru = nn.GRU(
            conv_channels[2], gru_hidden, num_layers=gru_layers,
            batch_first=True, bidirectional=True, dropout=dropout if gru_layers > 1 else 0,
        )

        gru_out_dim = gru_hidden * 2  # bidirectional

        # Output heads
        self.onset_head = nn.Sequential(
            nn.Linear(gru_out_dim, gru_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, n_keys),
        )
        # Frame head receives GRU output + onset logits (onset-frame coupling)
        self.frame_head = nn.Sequential(
            nn.Linear(gru_out_dim + n_keys, gru_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, n_keys),
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(gru_out_dim, gru_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, n_keys),
            nn.Sigmoid(),
        )
        # Note-value head: predicts rhythmic value class at each onset
        # Output shape: (batch, T, n_keys * NOTE_VALUE_CLASSES)
        self.note_value_head = nn.Sequential(
            nn.Linear(gru_out_dim, gru_out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gru_out_dim, n_keys * NOTE_VALUE_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, T, n_features) multi-resolution features

        Returns:
            dict with onset_logits, frame_logits, velocity (batch, T, 88)
                  and note_value_logits (batch, T, 88, NOTE_VALUE_CLASSES)
        """
        # Conv1d expects (batch, channels, time)
        h = x.permute(0, 2, 1)  # (batch, n_features, T)

        h = F.gelu(self.bn1(self.conv1(h)))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = F.gelu(self.bn3(self.conv3(h)))

        # BiGRU expects (batch, time, features)
        h = h.permute(0, 2, 1)  # (batch, T, conv_channels[-1])

        h, _ = self.gru(h)  # (batch, T, gru_hidden*2)

        B, T, _ = h.shape
        onset_logits = self.onset_head(h)  # (B, T, 88)

        # Onset-frame coupling: feed onset logits into frame head
        frame_input = torch.cat([h, onset_logits.detach()], dim=-1)  # (B, T, gru_out + 88)
        frame_logits = self.frame_head(frame_input)  # (B, T, 88)

        nv = self.note_value_head(h)  # (B, T, 88*10)
        nv = nv.view(B, T, self.n_keys, NOTE_VALUE_CLASSES)  # (B, T, 88, 10)

        return {
            'onset_logits': onset_logits,
            'frame_logits': frame_logits,
            'velocity': self.velocity_head(h),
            'note_value_logits': nv,
        }


# ─── Pitch-Aware Model ──────────────────────────────────────────────────────

# Indices for extracting per-key features from the flat 1098 vector.
# Per hop (549 features): 6 groups of 88 pitch-aligned bins.
_PITCH_GROUPS = [0, 88, 176, 264, 373, 461]  # mel×3, CQT, HPSS_h, HPSS_p
_HOP_OFFSETS = [0, N_FEATURES]  # 0, 549

# Precompute gather indices (done once at import time)
_KEY_INDICES = []
for _hop in _HOP_OFFSETS:
    for _start in _PITCH_GROUPS:
        _KEY_INDICES.extend(range(_hop + _start, _hop + _start + PIANO_KEYS))
# len = 2 hops × 6 groups × 88 keys = 1056

_GLOBAL_INDICES = []
for _hop in _HOP_OFFSETS:
    _GLOBAL_INDICES.extend(range(_hop + 352, _hop + 364))  # chroma (12)
    _GLOBAL_INDICES.extend(range(_hop + 364, _hop + 373))  # onset (9)
# len = 2 × (12 + 9) = 42

N_KEY_FEATURES = 12   # 6 spectral views × 2 hops
N_GLOBAL_FEATURES = 42


# ─── Conformer Building Blocks ─────────────────────────────────────────────

class ConformerFeedForward(nn.Module):
    """Macaron-style feed-forward module with expansion factor."""

    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        d_inner = d_model * expansion
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_inner),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerConvModule(nn.Module):
    """Conformer convolution module: pointwise → GLU → depthwise → BN → SiLU → pointwise."""

    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        # Pointwise expansion (2x for GLU)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        # Depthwise conv
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        # Pointwise projection
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, d_model)
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)  # (batch, d_model, T)
        x = self.pointwise1(x)  # (batch, 2*d_model, T)
        x = self.glu(x)         # (batch, d_model, T)
        x = self.depthwise(x)   # (batch, d_model, T)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise2(x)  # (batch, d_model, T)
        x = self.dropout(x)
        return x.permute(0, 2, 1)  # (batch, T, d_model)


class ConformerBlock(nn.Module):
    """
    Single Conformer block (macaron structure) with optional gradient checkpointing:
        x = x + 0.5 * FFN(x)
        x = x + MHSA(x)
        x = x + Conv(x)
        x = x + 0.5 * FFN(x)
        x = LayerNorm(x)
    """

    def __init__(self, d_model: int, n_heads: int = 4, ff_expansion: int = 4,
                 conv_kernel: int = 31, dropout: float = 0.1, use_checkpoint: bool = False):
        super().__init__()
        self.ff1 = ConformerFeedForward(d_model, ff_expansion, dropout)
        self.ff2 = ConformerFeedForward(d_model, ff_expansion, dropout)

        self.attn_norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.conv = ConformerConvModule(d_model, conv_kernel, dropout)
        self.final_norm = nn.LayerNorm(d_model)
        self.use_checkpoint = use_checkpoint

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """Internal forward: used for gradient checkpointing."""
        # 1. First half-step FFN
        x = x + 0.5 * self.ff1(x)
        # 2. Multi-head self-attention
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + self.attn_dropout(attn_out)
        # 3. Convolution module
        x = x + self.conv(x)
        # 4. Second half-step FFN
        x = x + 0.5 * self.ff2(x)
        # 5. Final layer norm
        return self.final_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al.)."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, T, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class PitchAwareTranscriber(nn.Module):
    """
    Pitch-aware Conformer transcriber.

    Architecture:
    1. Reshapes 1098 features into per-key (88, 12) + global (42)
    2. Per-key shared MLP: processes each key's 12 spectral views
    3. Key-axis Conv1d: captures harmonic/octave patterns across 88 keys
    4. Conformer on pooled temporal representation (B, T, D) — NOT per-key
    5. Broadcasts temporal context back to keys and combines with per-key features
    6. Per-key output heads: predict per key
    """

    def __init__(self, key_hidden=32, temporal_hidden=32, temporal_layers=4,
                 n_key_conv_layers=2, dropout=0.0,
                 n_heads=2, ff_expansion=4, conv_kernel=31, use_checkpoint=False):
        super().__init__()
        self.n_keys = PIANO_KEYS
        self.n_nv = NOTE_VALUE_CLASSES

        # Register index tensors as buffers (move to device with model)
        self.register_buffer('key_idx', torch.tensor(_KEY_INDICES, dtype=torch.long))
        self.register_buffer('global_idx', torch.tensor(_GLOBAL_INDICES, dtype=torch.long))

        # Per-key feature encoder (shared across all 88 keys)
        self.key_encoder = nn.Sequential(
            nn.Linear(N_KEY_FEATURES, key_hidden),
            nn.GELU(),
            nn.Linear(key_hidden, key_hidden),
            nn.GELU(),
        )

        # Global feature encoder
        self.global_encoder = nn.Sequential(
            nn.Linear(N_GLOBAL_FEATURES, key_hidden),
            nn.GELU(),
        )

        # Key-axis convolution (captures harmonic/octave relationships)
        key_conv_layers = []
        for _ in range(n_key_conv_layers):
            key_conv_layers.extend([
                nn.Conv1d(key_hidden, key_hidden, kernel_size=5, padding=2),
                nn.BatchNorm1d(key_hidden),
                nn.GELU(),
            ])
        self.key_conv = nn.Sequential(*key_conv_layers)

        # Project pooled key + global features to Conformer d_model
        # Input: key_summary (key_hidden) + global (key_hidden) = 2*key_hidden
        combined_dim = key_hidden * 2
        self.temporal_proj = nn.Linear(combined_dim, temporal_hidden)

        # Positional encoding for Conformer input
        self.pos_enc = SinusoidalPositionalEncoding(
            temporal_hidden, max_len=2048, dropout=dropout,
        )

        # Conformer stack — runs on (B, T, D), NOT (B*88, T, D)
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=temporal_hidden,
                n_heads=n_heads,
                ff_expansion=ff_expansion,
                conv_kernel=conv_kernel,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
            )
            for _ in range(temporal_layers)
        ])

        # Output projection: combine per-key features + global + temporal context
        # Input: key_h (key_hidden) + global_h (key_hidden) + conformer_out (temporal_hidden)
        t_out = temporal_hidden
        self.output_proj = nn.Sequential(
            nn.Linear(key_hidden * 2 + temporal_hidden, t_out),
            nn.GELU(),
        )

        # Per-key temporal conv: gives each key temporal context for duration estimation
        # Runs along time axis per key (shared weights across keys)
        # Receptive field: 3 layers × kernel 7 with dilation 1,2,4 = ~25 frames ≈ 0.4s
        self.key_temporal = nn.Sequential(
            nn.Conv1d(t_out, t_out, kernel_size=7, padding=3, groups=1),
            nn.GELU(),
            nn.Conv1d(t_out, t_out, kernel_size=7, padding=6, dilation=2, groups=1),
            nn.GELU(),
            nn.Conv1d(t_out, t_out, kernel_size=7, padding=12, dilation=4, groups=1),
            nn.GELU(),
        )
        # Zero-init last conv so residual starts as identity (no disruption on resume)
        nn.init.zeros_(self.key_temporal[-2].weight)
        nn.init.zeros_(self.key_temporal[-2].bias)

        # Per-key output heads (predict 1 value per key, shared weights)
        self.onset_head = nn.Sequential(
            nn.Linear(t_out, t_out), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(t_out, 1),
        )
        self.frame_head = nn.Sequential(
            nn.Linear(t_out + 1, t_out), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(t_out, 1),
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(t_out, t_out), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(t_out, 1), nn.Sigmoid(),
        )
        self.note_value_head = nn.Sequential(
            nn.Linear(t_out, t_out), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(t_out, NOTE_VALUE_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape

        # 1. Reshape: extract per-key and global features
        per_key = x[:, :, self.key_idx]  # (B, T, 1056)
        per_key = per_key.reshape(B, T, N_KEY_FEATURES, self.n_keys)  # (B, T, 12, 88)
        per_key = per_key.permute(0, 1, 3, 2)  # (B, T, 88, 12)

        global_f = x[:, :, self.global_idx]  # (B, T, 42)

        # 2. Per-key encoding (shared MLP applied to each key)
        key_h = self.key_encoder(per_key)  # (B, T, 88, key_hidden)

        # 3. Key-axis conv (capture harmonic patterns across keys)
        BT = B * T
        kh = key_h.reshape(BT, self.n_keys, -1).permute(0, 2, 1)  # (BT, key_hidden, 88)
        kh = self.key_conv(kh)  # (BT, key_hidden, 88)
        key_h = kh.permute(0, 2, 1).reshape(B, T, self.n_keys, -1)  # (B, T, 88, key_hidden)

        # 4. Global features
        global_h = self.global_encoder(global_f)  # (B, T, key_hidden)

        # 5. Conformer on pooled temporal representation
        # Pool per-key features across keys for temporal backbone
        key_summary = key_h.mean(dim=2)  # (B, T, key_hidden)
        temporal_in = torch.cat([key_summary, global_h], dim=-1)  # (B, T, 2*key_hidden)
        h = self.temporal_proj(temporal_in)  # (B, T, temporal_hidden)
        h = self.pos_enc(h)
        for block in self.conformer_blocks:
            h = block(h)  # (B, T, temporal_hidden)

        # 6. Broadcast temporal context to all keys, combine with per-key features
        h_keys = h.unsqueeze(2).expand(-1, -1, self.n_keys, -1)  # (B, T, 88, temporal_hidden)
        global_keys = global_h.unsqueeze(2).expand(-1, -1, self.n_keys, -1)  # (B, T, 88, key_hidden)
        h = self.output_proj(
            torch.cat([key_h, global_keys, h_keys], dim=-1)
        )  # (B, T, 88, t_out)

        # 7. Per-key temporal conv: each key sees its own time-axis context
        # (B, T, 88, D) → (B*88, D, T) → conv → (B*88, D, T) → (B, T, 88, D)
        h_t = h.permute(0, 2, 3, 1).reshape(B * self.n_keys, -1, T)  # (B*88, D, T)
        h_t = self.key_temporal(h_t)  # (B*88, D, T)
        h = h + h_t.reshape(B, self.n_keys, -1, T).permute(0, 3, 1, 2)  # residual

        # 8. Per-key output heads
        onset_logits = self.onset_head(h).squeeze(-1)  # (B, T, 88)

        frame_in = torch.cat([h, onset_logits.unsqueeze(-1).detach()], dim=-1)
        frame_logits = self.frame_head(frame_in).squeeze(-1)  # (B, T, 88)

        velocity = self.velocity_head(h).squeeze(-1)  # (B, T, 88)

        nv_logits = self.note_value_head(h)  # (B, T, 88, 10)

        return {
            'onset_logits': onset_logits,
            'frame_logits': frame_logits,
            'velocity': velocity,
            'note_value_logits': nv_logits,
        }


# ─── Loss Function ──────────────────────────────────────────────────────────

class EnsembleLoss(nn.Module):
    """
    Velocity-weighted loss for onset/frame/velocity prediction.

    Same soft-note emphasis as VelocityWeightedLoss in train_transcription.py:
      weight = 1 + alpha * (1 - velocity)
      pp note (vel=30): weight ~2.5x
      ff note (vel=100): weight ~1.4x
    """

    def __init__(self, alpha: float = 2.0, pos_weight: float = 5.0,
                 onset_weight: float = 1.0, frame_weight: float = 1.0,
                 velocity_weight: float = 0.5, note_value_weight: float = 1.0,
                 focal_gamma: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.onset_w = onset_weight
        self.frame_w = frame_weight
        self.vel_w = velocity_weight
        self.nv_w = note_value_weight
        self.focal_gamma = focal_gamma

    def forward(self, onset_logits, frame_logits, velocity_pred,
                onset_gt, frame_gt, velocity_gt,
                note_value_logits=None, note_value_gt=None) -> Dict[str, torch.Tensor]:

        # Velocity-based per-sample weighting
        vel_weight = torch.ones_like(velocity_gt)
        active = frame_gt > 0.5
        if active.any():
            vel_weight[active] = 1.0 + self.alpha * (1.0 - velocity_gt[active])

        # Onset focal loss (weighted)
        onset_probs = torch.sigmoid(onset_logits)
        onset_p_t = torch.where(onset_gt > 0.5, onset_probs, 1.0 - onset_probs)
        onset_focal_w = (1.0 - onset_p_t.detach()) ** self.focal_gamma
        onset_bce = F.binary_cross_entropy_with_logits(
            onset_logits, onset_gt, reduction='none',
        )
        onset_sample_w = torch.where(
            onset_gt > 0.5,
            vel_weight * self.pos_weight,
            torch.ones_like(vel_weight),
        )
        onset_loss = (onset_focal_w * onset_bce * onset_sample_w).mean()

        # Frame focal loss (weighted)
        frame_probs = torch.sigmoid(frame_logits)
        frame_p_t = torch.where(frame_gt > 0.5, frame_probs, 1.0 - frame_probs)
        frame_focal_w = (1.0 - frame_p_t.detach()) ** self.focal_gamma
        frame_bce = F.binary_cross_entropy_with_logits(
            frame_logits, frame_gt, reduction='none',
        )
        frame_sample_w = torch.where(
            frame_gt > 0.5,
            vel_weight * self.pos_weight,
            torch.ones_like(vel_weight),
        )
        frame_loss = (frame_focal_w * frame_bce * frame_sample_w).mean()

        # Velocity MSE (only for active frames)
        if active.any():
            velocity_loss = F.mse_loss(velocity_pred[active], velocity_gt[active])
        else:
            velocity_loss = torch.tensor(0.0, device=onset_logits.device)

        total = (
            self.onset_w * onset_loss
            + self.frame_w * frame_loss
            + self.vel_w * velocity_loss
        )

        result = {
            'total': total,
            'onset': onset_loss,
            'frame': frame_loss,
            'velocity': velocity_loss,
        }

        # Note-value focal loss (at ALL active frames for temporal pooling benefit)
        # Training at all frames lets the head learn duration-aware predictions,
        # so pooling across onset→offset at inference gives better estimates.
        if note_value_logits is not None and note_value_gt is not None:
            active_mask = frame_gt > 0.5  # (B, T, 88) — all frames where a note is active
            if active_mask.any():
                # note_value_logits: (B, T, 88, 10), note_value_gt: (B, T, 88) int64
                nv_logits_flat = note_value_logits[active_mask]  # (N_active, 10)
                nv_gt_flat = note_value_gt[active_mask]          # (N_active,)

                # Focal loss implementation
                gamma = self.focal_gamma
                probs = F.softmax(nv_logits_flat, dim=-1)  # (N_active, 10)
                # Get probability of true class
                p_t = probs.gather(1, nv_gt_flat.unsqueeze(1)).squeeze(1)  # (N_active,)
                # Focal weight: (1 - p_t)^gamma
                focal_weight = (1 - p_t.detach()) ** gamma
                # Cross-entropy per sample
                ce = F.cross_entropy(nv_logits_flat, nv_gt_flat, reduction='none')
                nv_loss = (focal_weight * ce).mean()
            else:
                nv_loss = torch.tensor(0.0, device=onset_logits.device)
            total = total + self.nv_w * nv_loss
            result['total'] = total
            result['note_value'] = nv_loss

        return result


# ─── Dataset ────────────────────────────────────────────────────────────────

class EnsembleTranscriptionDataset(Dataset):
    """
    On-the-fly feature computation from MAESTRO audio+MIDI pairs.

    Instead of storing preprocessed features (~150GB), loads audio segments
    and computes multi-resolution features on GPU during training.
    The 3 STFTs + filterbank multiplies take <10ms per segment on GPU.
    """

    def __init__(self, index_path: str, sr: int = SAMPLE_RATE,
                 hop_length: int = HOP_LENGTH, augment: bool = False,
                 label_hop_length: int = None):
        self.sr = sr
        self.hop_length = hop_length
        # Label resolution: use fine hop if provided, else same as hop_length
        self.label_hop_length = label_hop_length or hop_length
        self.segment_frames = int(SEGMENT_SECONDS * sr / self.label_hop_length)
        self.segment_samples = int(SEGMENT_SECONDS * sr)
        self.augment = augment

        with open(index_path) as f:
            self.index = json.load(f)

        self.segments = self.index['segments']
        self.pieces = self.index['pieces']
        aug_str = " (with augmentation)" if augment else ""
        print(f"[Dataset] {len(self.segments)} segments from {len(self.pieces)} pieces{aug_str}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        piece = self.pieces[seg['piece_idx']]

        import librosa

        # Load audio segment
        start_sec = seg['start_sec']
        duration_sec = SEGMENT_SECONDS
        audio, _ = librosa.load(
            piece['audio'], sr=self.sr, mono=True,
            offset=start_sec, duration=duration_sec,
        )

        # Pad if needed
        if len(audio) < self.segment_samples:
            audio = np.pad(audio, (0, self.segment_samples - len(audio)))
        audio = audio[:self.segment_samples]
        
        # Augmentation (training only)
        if self.augment:
            # Random gain (0.7 - 1.3x)
            gain = np.random.uniform(0.7, 1.3)
            audio = audio * gain
            # Random noise (very small)
            noise_level = np.random.uniform(0, 0.005)
            audio = audio + np.random.randn(len(audio)).astype(np.float32) * noise_level

        # Create frame-level labels from MIDI
        onset, frame, velocity, note_value, bpm = self._create_labels(piece['midi'], start_sec)

        return {
            'audio': torch.from_numpy(audio).float(),
            'onset': onset,
            'frame': frame,
            'velocity': velocity,
            'note_value': note_value,
            'bpm': torch.tensor(bpm, dtype=torch.float32),
        }

    def _create_labels(self, midi_path: str, start_sec: float):
        """Create onset/frame/velocity/note_value labels from MIDI for this segment."""
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(midi_path)

        onset = np.zeros((self.segment_frames, PIANO_KEYS), dtype=np.float32)
        frame = np.zeros((self.segment_frames, PIANO_KEYS), dtype=np.float32)
        velocity = np.zeros((self.segment_frames, PIANO_KEYS), dtype=np.float32)
        note_value = np.zeros((self.segment_frames, PIANO_KEYS), dtype=np.int64)  # class index

        end_sec = start_sec + SEGMENT_SECONDS
        frame_time = self.label_hop_length / self.sr

        # Get BPM from MIDI
        tempo_times, tempos = midi.get_tempo_changes()
        if len(tempos) > 0:
            bpm = float(tempos[0])
        else:
            bpm = 120.0
        beat_duration = 60.0 / bpm

        # Pre-compute log note-value beats for quantization
        log_nv = np.log2(NOTE_VALUE_BEATS)  # [-3, -2, -1, 0, 1, 2]

        # Collect ALL notes (not just in segment) for per-hand IOI computation
        all_notes = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                all_notes.append(note)
        all_notes.sort(key=lambda n: n.start)

        # Per-hand IOI map: (pitch, start_time) -> IOI seconds
        hand_notes = {'bass': [], 'treble': []}
        for note in all_notes:
            hand = 'bass' if note.pitch < 60 else 'treble'
            hand_notes[hand].append(note)

        ioi_map = {}
        for hand, notes_list in hand_notes.items():
            for j in range(len(notes_list) - 1):
                curr = notes_list[j]
                nxt = notes_list[j + 1]
                ioi_map[(curr.pitch, curr.start)] = nxt.start - curr.start

        # Fill targets
        for note in all_notes:
            if note.end < start_sec or note.start > end_sec:
                continue

            key = note.pitch - MIDI_OFFSET
            if key < 0 or key >= PIANO_KEYS:
                continue

            onset_f = int((note.start - start_sec) / frame_time)
            offset_f = int((note.end - start_sec) / frame_time)
            onset_f = max(0, min(onset_f, self.segment_frames - 1))
            offset_f = max(0, min(offset_f, self.segment_frames))

            vel_norm = note.velocity / 127.0

            # Onset (2 frames tolerance)
            for f in range(onset_f, min(onset_f + 2, self.segment_frames)):
                onset[f, key] = 1.0

            # Frame + velocity
            for f in range(onset_f, offset_f):
                frame[f, key] = 1.0
                velocity[f, key] = vel_norm

            # Note-value class from per-hand IOI
            ioi = ioi_map.get((note.pitch, note.start))
            if ioi is None or ioi < 0.03:
                # Last note in hand or near-simultaneous — use MIDI duration
                ioi = note.end - note.start
            ioi_beats = max(0.0625, min(8.0, ioi / beat_duration))
            class_idx = int(np.argmin(np.abs(log_nv - np.log2(ioi_beats))))
            # Set at ALL active frames (not just onset) so the head learns
            # duration-aware predictions that can be pooled at inference
            for f in range(onset_f, offset_f):
                if f < self.segment_frames:
                    note_value[f, key] = class_idx

        return (
            torch.from_numpy(onset),
            torch.from_numpy(frame),
            torch.from_numpy(velocity),
            torch.from_numpy(note_value),
            bpm,
        )


# ─── Precomputed Dataset ────────────────────────────────────────────────────

class PrecomputedDataset(Dataset):
    """
    Dataset that loads precomputed features from disk.
    
    Much faster than on-the-fly computation since it skips:
    - Audio decoding (librosa.load)
    - MIDI parsing (pretty_midi)
    - GPU feature extraction
    """
    
    def __init__(self, split: str = 'train', augment: bool = False,
                 mixup_alpha: float = 0.0):
        self.split_dir = FEATURES_DIR / split
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        if not self.split_dir.exists():
            raise RuntimeError(
                f"Precomputed features not found at {self.split_dir}\n"
                f"Run: python train_ensemble.py --precompute"
            )

        self.files = sorted(self.split_dir.glob("*.pt"))
        aug_str = " (with augmentation)" if augment else ""
        mixup_str = f", mixup={mixup_alpha}" if mixup_alpha > 0 else ""
        print(f"[PrecomputedDataset] {len(self.files)} segments from {self.split_dir}{aug_str}{mixup_str}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)

        # Mixup augmentation (training only)
        if self.augment and self.mixup_alpha > 0 and np.random.random() < 0.5:
            idx_b = np.random.randint(0, len(self.files))
            data_b = torch.load(self.files[idx_b], weights_only=True)

            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = max(lam, 1.0 - lam)  # primary segment dominates

            T = min(data['features'].size(0), data_b['features'].size(0))
            data['features'] = lam * data['features'][:T] + (1.0 - lam) * data_b['features'][:T]

            for key in ['onset', 'frame', 'velocity']:
                data[key] = lam * data[key][:T] + (1.0 - lam) * data_b[key][:T]

            # note_value is integer class — keep primary segment's labels
            data['note_value'] = data['note_value'][:T]

        # BPM: fallback to 120 for old precomputed files that lack it
        if 'bpm' not in data:
            data['bpm'] = torch.tensor(120.0, dtype=torch.float32)

        # SpecAugment-style feature augmentation (training only)
        if self.augment and 'features' in data:
            features = data['features']  # (T, 373)
            
            # Random time masking (mask 1-5 consecutive frames with zeros)
            if np.random.random() < 0.5:
                t_mask_len = np.random.randint(1, 6)
                t_start = np.random.randint(0, max(1, features.size(0) - t_mask_len))
                features[t_start:t_start + t_mask_len, :] = 0
            
            # Random frequency masking (mask 5-20 feature channels)
            if np.random.random() < 0.5:
                f_mask_len = np.random.randint(5, 21)
                f_start = np.random.randint(0, max(1, features.size(1) - f_mask_len))
                features[:, f_start:f_start + f_mask_len] = 0
            
            # Random gain on features
            gain = np.random.uniform(0.9, 1.1)
            data['features'] = features * gain

        # Pitch shift augmentation (training only): roll pitch-related features
        # along the frequency bin axis and shift labels along the key axis
        if self.augment and np.random.random() < 0.2:
            shift = np.random.choice([-1, 1])
            data['features'] = _shift_features_pitch(data['features'], shift)
            data['onset'] = _shift_pitch_labels(data['onset'], shift)
            data['frame'] = _shift_pitch_labels(data['frame'], shift)
            data['velocity'] = _shift_pitch_labels(data['velocity'], shift)
            data['note_value'] = _shift_pitch_labels(data['note_value'], shift)
        
        return data


def _shift_pitch_labels(labels: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift frame-level labels along the piano key axis by `shift` semitones.

    Args:
        labels: (T, 88) tensor (float or int64)
        shift: number of semitones to shift (positive = up, negative = down)

    Returns:
        shifted: (T, 88) tensor with out-of-range keys zeroed
    """
    if shift == 0:
        return labels
    shifted = torch.zeros_like(labels)
    if shift > 0:
        shifted[:, shift:] = labels[:, :-shift]
    else:
        shifted[:, :shift] = labels[:, -shift:]
    return shifted


def _shift_features_pitch(features: torch.Tensor, shift: int) -> torch.Tensor:
    """Shift pitch-related feature bins by `shift` semitones in precomputed features.

    Feature layout per hop (549 features):
      [0:88]     mel_1024    — pitch-related, roll by shift
      [88:176]   mel_2048    — pitch-related, roll by shift
      [176:264]  mel_4096    — pitch-related, roll by shift
      [264:352]  CQT         — pitch-related, roll by shift
      [352:364]  chroma      — pitch-related, roll by shift (mod 12)
      [364:373]  onset (9)   — frequency-summed, don't shift
      [373:461]  HPSS harm   — pitch-related, roll by shift
      [461:549]  HPSS perc   — pitch-related, roll by shift

    For multi-hop, the pattern repeats at offset 549.

    Args:
        features: (T, n_features) tensor
        shift: semitones to shift (positive = up)

    Returns:
        shifted features tensor
    """
    if shift == 0:
        return features
    out = features.clone()
    hop_size = N_FEATURES  # 549 per hop

    for hop_start in range(0, features.size(1), hop_size):
        # 88-bin blocks: mel x3, CQT, HPSS harmonic, HPSS percussive
        for block_offset in [0, 88, 176, 264, 373, 461]:
            start = hop_start + block_offset
            end = start + 88
            if end > features.size(1):
                break
            block = features[:, start:end]
            shifted_block = torch.zeros_like(block)
            if shift > 0:
                shifted_block[:, shift:] = block[:, :-shift]
            else:
                shifted_block[:, :shift] = block[:, -shift:]
            out[:, start:end] = shifted_block

        # Chroma (12 bins): circular roll
        chroma_start = hop_start + 352
        chroma_end = chroma_start + 12
        if chroma_end <= features.size(1):
            out[:, chroma_start:chroma_end] = torch.roll(
                features[:, chroma_start:chroma_end], shifts=shift, dims=1)

    return out


def precompute_features(args):
    """
    Precompute features and labels for all segments.
    
    Processes audio on GPU in batches, saves to disk for fast training.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    
    print(f"Precomputing features on {device}")
    print(f"Output directory: {FEATURES_DIR}")
    
    # Feature extractor on GPU (multi-hop)
    extractor = MultiResFeatureExtractor(
        sr=SAMPLE_RATE, hop_length=HOP_LENGTH, device=device,
        hop_lengths=HOP_LENGTHS,
    )
    
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    for split in ['train', 'validation', 'test']:
        index_path = INDEX_DIR / f"{split}_index.json"
        if not index_path.exists():
            print(f"Index not found: {index_path}, skipping {split}")
            continue
        
        split_dir = FEATURES_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)
        
        # Load index
        with open(index_path) as f:
            index = json.load(f)
        
        segments = index['segments']
        pieces = index['pieces']
        
        # Check what's already done
        existing = set(f.stem for f in split_dir.glob("*.pt"))
        to_process = [(i, seg) for i, seg in enumerate(segments)
                      if f"seg_{i:06d}" not in existing]

        if not to_process:
            print(f"  {split}: already complete ({len(segments)} segments)")
            continue

        print(f"  {split}: {len(to_process)}/{len(segments)} segments to process")
        
        # Create a temporary dataset for label generation (at fine hop resolution)
        temp_dataset = EnsembleTranscriptionDataset(
            str(index_path), label_hop_length=HOP_LENGTH_FINE)
        
        # Process in batches for GPU efficiency
        batch_size = args.precompute_batch
        processed = 0
        start_time = time.time()
        
        for batch_start in range(0, len(to_process), batch_size):
            batch_items = to_process[batch_start:batch_start + batch_size]

            # Load audio for batch (CPU, parallelized)
            import librosa
            audios = []
            labels_list = []
            indices = []

            for seg_idx, seg in batch_items:
                piece = pieces[seg['piece_idx']]
                start_sec = seg['start_sec']

                try:
                    # Load audio segment
                    audio, _ = librosa.load(
                        piece['audio'], sr=SAMPLE_RATE, mono=True,
                        offset=start_sec, duration=SEGMENT_SECONDS,
                    )

                    # Pad if needed
                    segment_samples = int(SEGMENT_SECONDS * SAMPLE_RATE)
                    if len(audio) < segment_samples:
                        audio = np.pad(audio, (0, segment_samples - len(audio)))
                    audio = audio[:segment_samples]

                    # Get labels (now includes bpm)
                    onset, frame, velocity, note_value, bpm = temp_dataset._create_labels(
                        piece['midi'], start_sec
                    )

                    audios.append(torch.from_numpy(audio).float())
                    labels_list.append((onset, frame, velocity, note_value, bpm))
                    indices.append(seg_idx)

                except Exception as e:
                    print(f"    Error processing segment {seg_idx}: {e}")
                    continue
            
            if not audios:
                continue
            
            # Stack and extract features on GPU
            audio_batch = torch.stack(audios).to(device)
            with torch.no_grad():
                features_batch = extractor.extract(audio_batch)  # (B, T, 373)
            
            # Save each segment
            for i, seg_idx in enumerate(indices):
                onset, frame, velocity, note_value, bpm = labels_list[i]

                data = {
                    'features': features_batch[i].cpu(),
                    'onset': onset,
                    'frame': frame,
                    'velocity': velocity,
                    'note_value': note_value,
                    'bpm': torch.tensor(bpm, dtype=torch.float32),
                }

                out_path = split_dir / f"seg_{seg_idx:06d}.pt"
                torch.save(data, out_path)

            processed += len(indices)
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(to_process) - processed) / rate if rate > 0 else 0
            
            print(f"    {processed}/{len(to_process)} ({rate:.1f} seg/s, ETA: {eta/60:.1f} min)", 
                  end='\r')
        
        print(f"\n  {split}: done ({processed} segments in {(time.time()-start_time)/60:.1f} min)")
    
    print(f"\nPrecomputation complete! Features saved to {FEATURES_DIR}")
    
    # Print size
    total_size = sum(f.stat().st_size for f in FEATURES_DIR.rglob("*.pt"))
    print(f"Total size: {total_size / 1e9:.2f} GB")


# ─── Note Event Decoding ────────────────────────────────────────────────────

def decode_note_events(
    onset_probs: np.ndarray,
    frame_probs: np.ndarray,
    velocity: np.ndarray,
    note_value_probs: np.ndarray = None,
    sr: int = SAMPLE_RATE,
    hop: int = HOP_LENGTH,
    onset_threshold: float = 0.75,
    frame_threshold: float = 0.75,
    min_note_duration: float = 0.05,
    min_velocity: int = 15,  # Filter very soft false positives
    dedup_window: float = 0.05,  # Duplicate detection window (seconds)
    use_peak_picking: bool = True,  # Only keep onset peaks
    filter_harmonics: bool = True,  # Remove likely harmonic false positives
    extend_to_next_onset: bool = False,  # Extend notes to next onset of same pitch
) -> List[Dict]:
    """
    Decode frame-level onset/frame/velocity predictions into note events.

    Args:
        onset_probs: (n_frames, 88) onset probabilities
        frame_probs: (n_frames, 88) frame probabilities
        velocity: (n_frames, 88) velocity predictions [0, 1]
        note_value_probs: (n_frames, 88, 6) note value class probabilities (optional)
        sr, hop: for time conversion
        onset_threshold, frame_threshold: detection thresholds
        min_note_duration: minimum note length in seconds
        min_velocity: minimum velocity to keep (0-127)
        dedup_window: window for duplicate detection (seconds)
        use_peak_picking: only keep local maxima in onset probabilities
        filter_harmonics: remove notes that are likely harmonics of louder notes
        extend_to_next_onset: extend notes until next onset of same pitch (better for piano)

    Returns:
        List of dicts: {'onset_time', 'offset_time', 'midi_note', 'velocity', 'note_value_class', ...}
    """
    frame_time = hop / sr
    n_frames = onset_probs.shape[0]
    min_frames = int(min_note_duration / frame_time)

    # Peak picking: only consider frames that are local maxima
    if use_peak_picking:
        onset_peaks = np.zeros_like(onset_probs, dtype=bool)
        for key in range(PIANO_KEYS):
            probs = onset_probs[:, key]
            # A frame is a peak if it's higher than neighbors and above threshold
            for f in range(1, n_frames - 1):
                if (probs[f] > onset_threshold and 
                    probs[f] >= probs[f-1] and 
                    probs[f] >= probs[f+1]):
                    onset_peaks[f, key] = True
            # Handle edges
            if probs[0] > onset_threshold and probs[0] >= probs[1]:
                onset_peaks[0, key] = True
            if probs[-1] > onset_threshold and probs[-1] >= probs[-2]:
                onset_peaks[-1, key] = True
    else:
        onset_peaks = onset_probs > onset_threshold

    note_events = []

    for key in range(PIANO_KEYS):
        frame_mask = frame_probs[:, key] > frame_threshold
        onset_frames = np.where(onset_peaks[:, key])[0]

        for onset_f in onset_frames:
            # Extend while frame is active
            offset_f = onset_f + 1
            while offset_f < n_frames and frame_mask[offset_f]:
                offset_f += 1

            # Enforce minimum duration
            if offset_f - onset_f < min_frames:
                offset_f = min(onset_f + min_frames, n_frames)

            # Velocity (average over active frames)
            vel_avg = velocity[onset_f:offset_f, key].mean()
            vel_int = int(np.clip(vel_avg * 127, 1, 127))
            
            # Skip very quiet notes (likely false positives)
            if vel_int < min_velocity:
                continue

            event = {
                'onset_time': float(onset_f * frame_time),
                'offset_time': float(offset_f * frame_time),
                'midi_note': int(key + MIDI_OFFSET),
                'velocity': vel_int,
                'onset_prob': float(onset_probs[onset_f, key]),  # Keep for filtering
            }
            
            # Add note value prediction if available
            if note_value_probs is not None:
                # Pool note_value_probs across the note's duration (onset→offset)
                # instead of reading a single onset frame — gives more robust estimates
                pooled_probs = note_value_probs[onset_f:offset_f, key, :].mean(axis=0)
                nv_class = int(np.argmax(pooled_probs))
                nv_conf = float(pooled_probs[nv_class])
                event['note_value_class'] = nv_class
                event['note_value_confidence'] = nv_conf
                event['note_value_name'] = NOTE_VALUE_NAMES[nv_class]
            
            note_events.append(event)

    # Sort by onset time
    note_events.sort(key=lambda e: (e['onset_time'], e['midi_note']))

    # Remove duplicate detections (same pitch within window)
    filtered = []
    for event in note_events:
        is_dup = False
        for prev in filtered[-15:]:
            if (abs(event['onset_time'] - prev['onset_time']) < dedup_window
                    and event['midi_note'] == prev['midi_note']):
                # Keep the one with higher onset probability
                if event['onset_prob'] > prev['onset_prob']:
                    filtered.remove(prev)
                else:
                    is_dup = True
                break
        if not is_dup:
            filtered.append(event)

    # Filter harmonics: if a note is +12/+19/+24 semitones above a louder note
    # at the same time, it's likely a harmonic false positive
    if filter_harmonics:
        harmonic_intervals = [12, 19, 24, 28, 31]  # octave, 5th+oct, 2 oct, etc.
        cleaned = []
        for event in filtered:
            is_harmonic = False
            # Check if this note is a harmonic of a concurrent louder note
            for other in filtered:
                if other is event:
                    continue
                # Must be at same time (within 30ms)
                if abs(event['onset_time'] - other['onset_time']) > 0.03:
                    continue
                # Check harmonic relationship
                interval = event['midi_note'] - other['midi_note']
                if interval in harmonic_intervals:
                    # This note is a harmonic of other - keep only if louder
                    if event['velocity'] < other['velocity'] * 0.7:
                        is_harmonic = True
                        break
            if not is_harmonic:
                cleaned.append(event)
        filtered = cleaned

    # Extend notes to next onset of same pitch (IOI-based duration)
    # This better matches piano notation where notes sustain until the next note
    if extend_to_next_onset:
        # Group by pitch
        by_pitch = {}
        for event in filtered:
            pitch = event['midi_note']
            if pitch not in by_pitch:
                by_pitch[pitch] = []
            by_pitch[pitch].append(event)
        
        # Sort each pitch group by onset time and extend to next onset
        for pitch, events in by_pitch.items():
            events.sort(key=lambda e: e['onset_time'])
            for i, event in enumerate(events):
                if i < len(events) - 1:
                    next_onset = events[i + 1]['onset_time']
                    # Extend to just before next onset (small gap for clarity)
                    new_offset = next_onset - 0.01
                    if new_offset > event['offset_time']:
                        event['offset_time'] = new_offset
                else:
                    # Last note of this pitch - extend to max 2 seconds or end of audio
                    max_duration = 2.0
                    max_offset = event['onset_time'] + max_duration
                    audio_end = n_frames * frame_time
                    event['offset_time'] = min(max_offset, audio_end)

    # Remove internal fields before returning
    for event in filtered:
        event.pop('onset_prob', None)
        # Keep note_value_class, note_value_confidence, note_value_name for downstream use

    return filtered


# ─── Data Preparation ───────────────────────────────────────────────────────

def prepare_segment_index():
    """
    Build segment index from MAESTRO audio+MIDI pairs.

    Creates a lightweight JSON index of (piece, start_time) pairs.
    No audio processing here — features are computed on-the-fly during training.
    """
    import librosa

    if not MAESTRO_CSV.exists():
        print(f"MAESTRO CSV not found at {MAESTRO_CSV}")
        print("Run: python prepare_training_data.py --download")
        return

    # Read CSV
    all_pieces = []
    with open(MAESTRO_CSV, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_pieces.append(row)

    print(f"Found {len(all_pieces)} pieces in MAESTRO CSV")

    # Check audio availability
    pieces_with_audio = []
    for piece in all_pieces:
        audio_path = MAESTRO_DIR / piece['audio_filename']
        midi_path = MAESTRO_DIR / piece['midi_filename']
        if audio_path.exists() and midi_path.exists():
            pieces_with_audio.append(piece)

    n_audio = len(pieces_with_audio)
    if n_audio == 0:
        print("\nNo audio files found! Download the full MAESTRO dataset:")
        print("  python train_transcription.py --download-audio")
        return

    print(f"Found audio for {n_audio}/{len(all_pieces)} pieces")

    # Build index per split
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'validation', 'test']:
        split_pieces = [p for p in pieces_with_audio if p['split'] == split]
        pieces_list = []
        segments = []

        for piece_idx, piece in enumerate(split_pieces):
            audio_path = str(MAESTRO_DIR / piece['audio_filename'])
            midi_path = str(MAESTRO_DIR / piece['midi_filename'])

            try:
                duration = librosa.get_duration(path=audio_path)
            except Exception as e:
                print(f"  Skipping {audio_path}: {e}")
                continue

            stored_idx = len(pieces_list)
            pieces_list.append({
                'audio': audio_path,
                'midi': midi_path,
                'composer': piece.get('canonical_composer', ''),
                'title': piece.get('canonical_title', ''),
                'duration': duration,
            })

            # Create segment entries
            for start in np.arange(0, duration - SEGMENT_SECONDS / 2, SEGMENT_SECONDS):
                segments.append({
                    'piece_idx': stored_idx,
                    'start_sec': float(start),
                })

        index = {
            'pieces': pieces_list,
            'segments': segments,
            'sr': SAMPLE_RATE,
            'hop_length': HOP_LENGTH,
            'segment_seconds': SEGMENT_SECONDS,
        }

        index_path = INDEX_DIR / f"{split}_index.json"
        with open(index_path, 'w') as f:
            json.dump(index, f)

        print(f"  {split}: {len(pieces_list)} pieces, {len(segments)} segments -> {index_path}")

    print(f"\nIndex preparation complete! Saved to {INDEX_DIR}")


# ─── Model Builder ──────────────────────────────────────────────────────────

def _build_model_from_config(config: dict) -> nn.Module:
    """Build the right model from a checkpoint config dict.

    Supports both PitchAwareTranscriber (new) and EnsembleMetaLearner (legacy).
    """
    model_type = config.get('model_type', 'EnsembleMetaLearner')

    if model_type == 'PitchAwareTranscriber':
        return PitchAwareTranscriber(
            key_hidden=config.get('key_hidden', 128),
            temporal_hidden=config.get('temporal_hidden', 128),
            temporal_layers=config.get('temporal_layers', 4),
            n_key_conv_layers=config.get('n_key_conv_layers', 2),
            dropout=config.get('dropout', 0.1),
            n_heads=config.get('n_heads', 2),
            ff_expansion=config.get('ff_expansion', 4),
            conv_kernel=config.get('conv_kernel', 31),
            use_checkpoint=config.get('use_checkpoint', False),
        )

    # Legacy: EnsembleMetaLearner
    return EnsembleMetaLearner(
        n_features=config.get('n_features', N_FEATURES_MULTI_HOP),
        conv_channels=config.get('conv_channels', [256, 256, 128]),
        gru_hidden=config.get('gru_hidden', 64),
        gru_layers=config.get('gru_layers', 2),
        dropout=config.get('dropout', 0.1),
    )


# ─── Training ───────────────────────────────────────────────────────────────

def train(args):
    """Main training loop."""
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Check for precomputed features
    use_precomputed = (FEATURES_DIR / 'train').exists() and len(list((FEATURES_DIR / 'train').glob("*.pt"))) > 0
    
    if use_precomputed:
        print(f"Using precomputed features from {FEATURES_DIR}")
        train_dataset = PrecomputedDataset(
            'train', augment=True, mixup_alpha=args.mixup_alpha)
        val_dataset = PrecomputedDataset('validation', augment=False)
        extractor = None  # Not needed
    else:
        # Feature extractor (lives on GPU, shared across batches)
        extractor = MultiResFeatureExtractor(
            sr=SAMPLE_RATE, hop_length=HOP_LENGTH, device=device,
            hop_lengths=HOP_LENGTHS,
        )
        print(f"Feature extractor: {extractor.n_features} features per frame "
              f"(multi-hop: {extractor.multi_hop})")

        # Datasets
        train_index = INDEX_DIR / "train_index.json"
        val_index = INDEX_DIR / "validation_index.json"
        if not train_index.exists():
            print(f"Index not found at {train_index}")
            print("Run: python train_ensemble.py --prepare")
            return

        train_dataset = EnsembleTranscriptionDataset(
            str(train_index), augment=True, label_hop_length=HOP_LENGTH_FINE)
        val_dataset = EnsembleTranscriptionDataset(
            str(val_index), augment=False, label_hop_length=HOP_LENGTH_FINE)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = PitchAwareTranscriber(
        key_hidden=args.key_hidden,
        temporal_hidden=args.temporal_hidden,
        temporal_layers=args.temporal_layers,
        n_key_conv_layers=args.n_key_conv_layers,
        dropout=args.dropout,
        n_heads=args.n_heads,
        ff_expansion=args.ff_expansion,
        conv_kernel=args.conv_kernel,
        use_checkpoint=args.use_checkpoint,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = EnsembleLoss(
        alpha=args.vel_alpha, pos_weight=args.pos_weight,
        note_value_weight=args.nv_weight,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # Noam / Transformer LR schedule (Vaswani et al. "Attention is All You Need")
    # lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})
    # Linear warmup for warmup_steps, then inverse sqrt decay.
    # This is the standard, most battle-tested schedule for attention-based models.
    warmup_steps = args.warmup_steps
    d_model = args.temporal_hidden

    def noam_lambda(step):
        step = max(step, 1)  # avoid division by zero
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)

    best_val_loss = float('inf')
    best_onset_f1 = 0.0
    start_epoch = 0

    # Resume from checkpoint if requested
    if args.resume and MODEL_PATH.exists():
        print(f"Resuming from checkpoint: {MODEL_PATH}")
        checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"  Could not load optimizer state (architecture changed?): {e}")
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"  Restored scheduler state (lr={optimizer.param_groups[0]['lr']:.2e})")
            except Exception as e:
                print(f"  Could not load scheduler state: {e}")
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_onset_f1 = checkpoint.get('onset_f1', 0.0)
        # Reset best_val_loss if checkpoint was from old model without note_value
        # (old loss doesn't include note_value component, so not comparable)
        if 'note_value_acc' not in checkpoint:
            print("  Old checkpoint without note_value metrics - resetting best_val_loss")
            best_val_loss = float('inf')
        else:
            best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"  Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
    elif args.resume:
        print(f"No checkpoint found at {MODEL_PATH}, starting fresh")

    # AMP: mixed precision for speedup on GPUs with tensor cores
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # ── ETA estimation: time a few warmup batches before real training ──
    if start_epoch == 0:
        import time as _time
        _warmup_batches = min(5, len(train_loader))
        model.train()
        print(f"Timing {_warmup_batches} warmup batches for ETA estimation...")
        _t0 = _time.perf_counter()
        for _wb, batch in enumerate(train_loader):
            if _wb >= _warmup_batches:
                break
            onset_gt = batch['onset'].to(device)
            frame_gt = batch['frame'].to(device)
            vel_gt = batch['velocity'].to(device)
            nv_gt = batch['note_value'].to(device)
            if use_precomputed:
                features = batch['features'].to(device)
            else:
                audio = batch['audio'].to(device)
                with torch.no_grad():
                    features = extractor.extract(audio)
            T = min(features.size(1), onset_gt.size(1))
            features, onset_gt = features[:, :T, :], onset_gt[:, :T, :]
            frame_gt, vel_gt, nv_gt = frame_gt[:, :T, :], vel_gt[:, :T, :], nv_gt[:, :T, :]
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(features)
                losses = criterion(
                    out['onset_logits'], out['frame_logits'], out['velocity'],
                    onset_gt, frame_gt, vel_gt,
                    note_value_logits=out['note_value_logits'], note_value_gt=nv_gt,
                )
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        _t1 = _time.perf_counter()
        _sec_per_batch = (_t1 - _t0) / _warmup_batches
        _total_batches = len(train_loader) * args.epochs
        _eta_sec = _sec_per_batch * _total_batches
        _eta_h = int(_eta_sec // 3600)
        _eta_m = int((_eta_sec % 3600) // 60)
        print(f"  Avg batch time: {_sec_per_batch:.2f}s")
        print(f"  Total batches: {_total_batches} ({len(train_loader)}/epoch × {args.epochs} epochs)")
        print(f"  *** Estimated training time: {_eta_h}h {_eta_m}m ***")
        # Reset model/optimizer for clean start (warmup batches were real steps)
        # The scheduler already advanced by _warmup_batches steps, which is fine —
        # OneCycleLR will still follow its schedule from this point

    for epoch in range(start_epoch, args.epochs):
        # ── Train ──
        model.train()
        train_losses = defaultdict(float)
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            onset_gt = batch['onset'].to(device)
            frame_gt = batch['frame'].to(device)
            vel_gt = batch['velocity'].to(device)
            nv_gt = batch['note_value'].to(device)

            # Get features (precomputed or on-the-fly)
            if use_precomputed:
                features = batch['features'].to(device)
            else:
                audio = batch['audio'].to(device)
                with torch.no_grad():
                    features = extractor.extract(audio)  # (B, T, 373)

            # Trim features and labels to the shorter of the two
            T = min(features.size(1), onset_gt.size(1))
            features = features[:, :T, :]
            onset_gt = onset_gt[:, :T, :]
            frame_gt = frame_gt[:, :T, :]
            vel_gt = vel_gt[:, :T, :]
            nv_gt = nv_gt[:, :T, :]

            optimizer.zero_grad()

            # AMP: forward pass in float16
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(features)

                losses = criterion(
                    out['onset_logits'], out['frame_logits'], out['velocity'],
                    onset_gt, frame_gt, vel_gt,
                    note_value_logits=out['note_value_logits'],
                    note_value_gt=nv_gt,
                )

            # AMP: scaled backward + optimizer step
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Noam schedule: step per batch

            for k, v in losses.items():
                train_losses[k] += v.item()
            n_batches += 1

            if batch_idx % 100 == 0:
                avg = train_losses['total'] / n_batches
                print(f"  Epoch {epoch+1} batch {batch_idx}/{len(train_loader)}: loss={avg:.4f}")

        # ── Validate ──
        model.eval()
        val_losses = defaultdict(float)
        n_val = 0
        onset_tp, onset_fp, onset_fn = 0, 0, 0
        frame_tp, frame_fp, frame_fn = 0, 0, 0
        nv_correct, nv_total = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                onset_gt = batch['onset'].to(device)
                frame_gt = batch['frame'].to(device)
                vel_gt = batch['velocity'].to(device)
                nv_gt = batch['note_value'].to(device)

                # Get features (precomputed or on-the-fly)
                if use_precomputed:
                    features = batch['features'].to(device)
                else:
                    audio = batch['audio'].to(device)
                    features = extractor.extract(audio)

                T = min(features.size(1), onset_gt.size(1))
                features = features[:, :T, :]
                onset_gt = onset_gt[:, :T, :]
                frame_gt = frame_gt[:, :T, :]
                vel_gt = vel_gt[:, :T, :]
                nv_gt = nv_gt[:, :T, :]

                # AMP: validation forward in float16
                with torch.amp.autocast('cuda', enabled=use_amp):
                    out = model(features)
                    losses = criterion(
                        out['onset_logits'], out['frame_logits'], out['velocity'],
                        onset_gt, frame_gt, vel_gt,
                        note_value_logits=out['note_value_logits'],
                        note_value_gt=nv_gt,
                    )
                for k, v in losses.items():
                    val_losses[k] += v.item()
                n_val += 1

                # F1 metrics
                onset_pred = (torch.sigmoid(out['onset_logits']) > 0.5).float()
                frame_pred = (torch.sigmoid(out['frame_logits']) > 0.5).float()

                onset_tp += ((onset_pred == 1) & (onset_gt == 1)).sum().item()
                onset_fp += ((onset_pred == 1) & (onset_gt == 0)).sum().item()
                onset_fn += ((onset_pred == 0) & (onset_gt == 1)).sum().item()

                # Note-value accuracy at onset frames
                onset_mask_val = onset_gt > 0.5
                if onset_mask_val.any():
                    nv_pred_class = out['note_value_logits'][onset_mask_val].argmax(dim=-1)
                    nv_gt_class = nv_gt[onset_mask_val]
                    nv_correct += (nv_pred_class == nv_gt_class).sum().item()
                    nv_total += nv_gt_class.numel()

                frame_tp += ((frame_pred == 1) & (frame_gt == 1)).sum().item()
                frame_fp += ((frame_pred == 1) & (frame_gt == 0)).sum().item()
                frame_fn += ((frame_pred == 0) & (frame_gt == 1)).sum().item()

        # Compute F1
        onset_p = onset_tp / max(onset_tp + onset_fp, 1)
        onset_r = onset_tp / max(onset_tp + onset_fn, 1)
        onset_f1 = 2 * onset_p * onset_r / max(onset_p + onset_r, 1e-8)

        frame_p = frame_tp / max(frame_tp + frame_fp, 1)
        frame_r = frame_tp / max(frame_tp + frame_fn, 1)
        frame_f1 = 2 * frame_p * frame_r / max(frame_p + frame_r, 1e-8)

        nv_acc = nv_correct / max(nv_total, 1)

        avg_train = {k: v / max(n_batches, 1) for k, v in train_losses.items()}
        avg_val = {k: v / max(n_val, 1) for k, v in val_losses.items()}

        print(f"\nEpoch {epoch+1}/{args.epochs} (lr={optimizer.param_groups[0]['lr']:.2e})")
        nv_train_str = f", nv={avg_train.get('note_value', 0):.4f}" if 'note_value' in avg_train else ''
        print(f"  Train loss: {avg_train['total']:.4f} "
              f"(onset={avg_train['onset']:.4f}, frame={avg_train['frame']:.4f}, "
              f"vel={avg_train['velocity']:.4f}{nv_train_str})")
        print(f"  Val loss:   {avg_val['total']:.4f}")
        print(f"  Onset  P={onset_p:.3f} R={onset_r:.3f} F1={onset_f1:.3f}")
        print(f"  Frame  P={frame_p:.3f} R={frame_r:.3f} F1={frame_f1:.3f}")
        print(f"  NoteVal acc={nv_acc:.3f} ({nv_correct}/{nv_total})")

        # (Noam schedule steps per batch, not per epoch)

        # Save best model (by onset F1, not val_loss)
        if onset_f1 > best_onset_f1:
            best_onset_f1 = onset_f1
            best_val_loss = avg_val['total']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': {
                    'model_type': 'PitchAwareTranscriber',
                    'key_hidden': args.key_hidden,
                    'temporal_hidden': args.temporal_hidden,
                    'temporal_layers': args.temporal_layers,
                    'n_key_conv_layers': args.n_key_conv_layers,
                    'dropout': args.dropout,
                    'n_heads': args.n_heads,
                    'ff_expansion': args.ff_expansion,
                    'conv_kernel': args.conv_kernel,
                    'use_checkpoint': args.use_checkpoint,
                    'n_keys': PIANO_KEYS,
                    'n_note_value_classes': NOTE_VALUE_CLASSES,
                    'sample_rate': SAMPLE_RATE,
                    'hop_length': HOP_LENGTH_FINE,
                    'hop_lengths': HOP_LENGTHS,
                    'has_tempo_head': False,
                    'nv_pooled': True,
                },
                'epoch': epoch,
                'val_loss': best_val_loss,
                'onset_f1': onset_f1,
                'frame_f1': frame_f1,
                'note_value_acc': nv_acc,
            }, str(MODEL_PATH))
            print(f"  Saved best model! (onset_f1={best_onset_f1:.3f}, val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best onset F1: {best_onset_f1:.3f}")
    print(f"  Model saved to: {MODEL_PATH}")

    # ── Threshold sweep: find optimal onset/frame thresholds for F1 ──
    print(f"\n{'='*60}")
    print("Threshold sweep on validation set (using best saved model)...")
    print(f"{'='*60}")
    checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    onset_counts = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in thresholds}
    frame_counts = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in thresholds}

    with torch.no_grad():
        for batch in val_loader:
            onset_gt = batch['onset'].to(device)
            frame_gt = batch['frame'].to(device)
            if use_precomputed:
                features = batch['features'].to(device)
            else:
                audio = batch['audio'].to(device)
                features = extractor.extract(audio)
            T = min(features.size(1), onset_gt.size(1))
            features, onset_gt = features[:, :T, :], onset_gt[:, :T, :]
            frame_gt = frame_gt[:, :T, :]

            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(features)

            onset_probs = torch.sigmoid(out['onset_logits'])
            frame_probs = torch.sigmoid(out['frame_logits'])

            for t in thresholds:
                o_pred = (onset_probs > t).float()
                onset_counts[t]['tp'] += ((o_pred == 1) & (onset_gt == 1)).sum().item()
                onset_counts[t]['fp'] += ((o_pred == 1) & (onset_gt == 0)).sum().item()
                onset_counts[t]['fn'] += ((o_pred == 0) & (onset_gt == 1)).sum().item()

                f_pred = (frame_probs > t).float()
                frame_counts[t]['tp'] += ((f_pred == 1) & (frame_gt == 1)).sum().item()
                frame_counts[t]['fp'] += ((f_pred == 1) & (frame_gt == 0)).sum().item()
                frame_counts[t]['fn'] += ((f_pred == 0) & (frame_gt == 1)).sum().item()

    print(f"\n{'Thresh':>6}  {'Onset P':>8} {'Onset R':>8} {'Onset F1':>8}  │  {'Frame P':>8} {'Frame R':>8} {'Frame F1':>8}")
    print("─" * 80)
    best_onset_t, best_onset_f1_sweep = 0.5, 0.0
    best_frame_t, best_frame_f1_sweep = 0.5, 0.0
    for t in thresholds:
        oc = onset_counts[t]
        op = oc['tp'] / max(oc['tp'] + oc['fp'], 1)
        orc = oc['tp'] / max(oc['tp'] + oc['fn'], 1)
        of1 = 2 * op * orc / max(op + orc, 1e-8)
        if of1 > best_onset_f1_sweep:
            best_onset_f1_sweep = of1
            best_onset_t = t

        fc = frame_counts[t]
        fp = fc['tp'] / max(fc['tp'] + fc['fp'], 1)
        frc = fc['tp'] / max(fc['tp'] + fc['fn'], 1)
        ff1 = 2 * fp * frc / max(fp + frc, 1e-8)
        if ff1 > best_frame_f1_sweep:
            best_frame_f1_sweep = ff1
            best_frame_t = t

        marker_o = " ◄" if t == best_onset_t and of1 == best_onset_f1_sweep else ""
        marker_f = " ◄" if t == best_frame_t and ff1 == best_frame_f1_sweep else ""
        print(f"  {t:.2f}   {op:>8.3f} {orc:>8.3f} {of1:>8.3f}{marker_o:3s} │  {fp:>8.3f} {frc:>8.3f} {ff1:>8.3f}{marker_f}")

    print(f"\n  Best onset threshold: {best_onset_t:.2f} → F1={best_onset_f1_sweep:.3f}")
    print(f"  Best frame threshold: {best_frame_t:.2f} → F1={best_frame_f1_sweep:.3f}")
    print(f"  (Default was 0.5 → onset F1={best_onset_f1:.3f})")


# ─── Benchmark ──────────────────────────────────────────────────────────────

def benchmark(args):
    """Benchmark ensemble inference speed vs ByteDance."""
    device = torch.device(args.device)

    # Generate test audio (10 seconds)
    test_audio = np.random.randn(SAMPLE_RATE * 10).astype(np.float32) * 0.1
    n_runs = 10

    # ── Ensemble ──
    print("=" * 60)
    print("BENCHMARK: Ensemble vs ByteDance transcription speed")
    print("=" * 60)

    if MODEL_PATH.exists():
        checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        hop_lengths = config.get('hop_lengths', None)
        extractor = MultiResFeatureExtractor(
            sr=SAMPLE_RATE, device=device, hop_lengths=hop_lengths)
        model = _build_model_from_config(config)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model.to(device)
        model.eval()

        audio_t = torch.from_numpy(test_audio).float().to(device)

        # Warmup
        with torch.no_grad():
            features = extractor.extract(audio_t)
            _ = model(features)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        # Timed runs
        start = time.perf_counter()
        for _ in range(n_runs):
            with torch.no_grad():
                features = extractor.extract(audio_t)
                _ = model(features)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        ensemble_ms = (time.perf_counter() - start) / n_runs * 1000
        print(f"\nEnsemble:  {ensemble_ms:.1f} ms / 10s audio")
    else:
        print(f"\nEnsemble model not found at {MODEL_PATH}, skipping")
        ensemble_ms = None

    # ── ByteDance ──
    try:
        import librosa
        from piano_transcription_inference import (PianoTranscription,
                                                   sample_rate)

        bd_audio, _ = librosa.load(
            None, sr=sample_rate, mono=True,
        ) if False else (test_audio, SAMPLE_RATE)
        # ByteDance expects 16kHz, reuse test_audio
        bd = PianoTranscription(device=str(device))

        # Warmup
        bd.transcribe(test_audio, midi_path=None)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(n_runs):
            bd.transcribe(test_audio, midi_path=None)
            if device.type == 'cuda':
                torch.cuda.synchronize()
        bd_ms = (time.perf_counter() - start) / n_runs * 1000
        print(f"ByteDance: {bd_ms:.1f} ms / 10s audio")

        if ensemble_ms:
            print(f"\nSpeedup: {bd_ms / ensemble_ms:.1f}x")

    except ImportError:
        print("\nByteDance piano_transcription_inference not installed, skipping")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Train multi-resolution ensemble transcriber for LiveScore')

    # Actions
    parser.add_argument('--prepare', action='store_true',
                        help='Prepare segment index from MAESTRO')
    parser.add_argument('--precompute', action='store_true',
                        help='Precompute features for faster training')
    parser.add_argument('--train', action='store_true',
                        help='Train the ensemble meta-learner')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark inference speed vs ByteDance')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size (16 uses ~8GB with checkpoint on 12GB GPU)')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='Base LR for Noam schedule (scaled by d_model^-0.5)')
    parser.add_argument('--key-hidden', type=int, default=128,
                        help='PitchAwareTranscriber per-key hidden dim')
    parser.add_argument('--temporal-hidden', type=int, default=64,
                        help='Conformer d_model dimension')
    parser.add_argument('--temporal-layers', type=int, default=4,
                        help='Number of Conformer blocks')
    parser.add_argument('--n-key-conv-layers', type=int, default=2,
                        help='PitchAwareTranscriber key-axis conv layers')
    parser.add_argument('--n-heads', type=int, default=2,
                        help='Conformer attention heads')
    parser.add_argument('--ff-expansion', type=int, default=4,
                        help='Conformer feed-forward expansion factor')
    parser.add_argument('--conv-kernel', type=int, default=31,
                        help='Conformer depthwise conv kernel size')
    parser.add_argument('--warmup-steps', type=int, default=4000,
                        help='Noam schedule warmup steps')
    parser.add_argument('--use-checkpoint', action='store_true', default=True,
                        help='Gradient checkpointing (trades ~10%% compute for ~40%% memory savings)')
    parser.add_argument('--no-checkpoint', action='store_false', dest='use_checkpoint',
                        help='Disable gradient checkpointing')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--vel-alpha', type=float, default=2.0,
                        help='Velocity weighting (higher = more soft note emphasis)')
    parser.add_argument('--pos-weight', type=float, default=5.0,
                        help='Positive class weight for onset/frame BCE')
    parser.add_argument('--nv-weight', type=float, default=1.0,
                        help='Note-value loss weight (reduce to let onset/frame dominate)')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--precompute-batch', type=int, default=16,
                        help='Batch size for feature precomputation')
    parser.add_argument('--mixup-alpha', type=float, default=0.0,
                        help='Mixup alpha (0 to disable)')

    args = parser.parse_args()

    if args.prepare:
        prepare_segment_index()
    elif args.precompute:
        precompute_features(args)
    elif args.train:
        train(args)
    elif args.benchmark:
        benchmark(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
