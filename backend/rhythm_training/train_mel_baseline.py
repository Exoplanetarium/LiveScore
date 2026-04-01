"""
Reduced-feature baseline transcriber for LiveScore.

Experiment: use ONLY a single log-mel spectrogram (229 bins) instead of
the full 1098-feature multi-resolution stack. Let the model learn its
own internal representations from the raw spectral data.

Architecture:
  1. Simple feature extraction:
     - Single STFT (n_fft=2048, hop=256) -> 229-bin log-mel spectrogram
     - That's it. No CQT, no chroma, no onset functions, no HPSS.

  2. ConvStack + Conformer model (~2M params):
     - 2D ConvStack: extracts local time-frequency patterns from mel "image"
     - Conformer temporal backbone: larger than ensemble (128d, 6 layers)
     - Per-key output heads: onset, frame, velocity, note_value

Training data: same MAESTRO v3.0.0 segments as train_ensemble.py.
Reuses the same segment index and label generation.

Usage:
    # Uses same segment index as train_ensemble.py
    # If not prepared yet:  python train_ensemble.py --prepare

    # Train
    python train_mel_baseline.py --train --epochs 50 --batch-size 8

    # Resume training
    python train_mel_baseline.py --train --epochs 50 --batch-size 8 --resume

    # Benchmark
    python train_mel_baseline.py --benchmark
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
HOP_LENGTH = 256          # single fine hop (matches ensemble's fine hop)
N_FFT = 2048
N_MELS = 229              # standard for SOTA models (Onsets & Frames, Kong et al.)
PIANO_KEYS = 88
MIDI_OFFSET = 21           # A0

SEGMENT_SECONDS = 10.0
SEGMENT_FRAMES = int(SEGMENT_SECONDS * SAMPLE_RATE / HOP_LENGTH)  # 625

# Regression onset target: tent-shaped window around true onset (Kong et al. 2020)
# Peak=1.0 at true onset, linear decay to 0.0 at ±ONSET_TENT_SEC seconds
ONSET_TENT_SEC = 0.05  # ±50ms tent window

# Note-value classes (same as train_ensemble.py)
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

# Paths (reuse ensemble's index, separate model checkpoint)
MAESTRO_DIR = Path(__file__).parent / "maestro_midi"
MAESTRO_CSV = MAESTRO_DIR / "maestro-v3.0.0.csv"
INDEX_DIR = Path(__file__).parent / "ensemble_index"
FEATURES_DIR = Path(__file__).parent / "precomputed_features_mel"
MODEL_PATH = Path(__file__).parent / "mel_baseline_transcription.pt"


# ─── Mel Feature Extractor ──────────────────────────────────────────────────

class MelFeatureExtractor:
    """
    Simple log-mel spectrogram feature extractor.

    Single STFT (n_fft=2048) -> 229-bin mel filterbank -> log scale.
    Output: (batch, n_frames, 229) per frame.
    """

    def __init__(self, sr: int = SAMPLE_RATE, hop_length: int = HOP_LENGTH,
                 n_fft: int = N_FFT, n_mels: int = N_MELS,
                 device: torch.device = None):
        self.sr = sr
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Precompute STFT window
        self.window = torch.hann_window(n_fft, device=self.device)

        # Precompute mel filterbank
        self.mel_fb = self._build_mel_filterbank()

    def _build_mel_filterbank(self) -> torch.Tensor:
        """Build mel filterbank: (n_mels, n_fft//2+1) on GPU."""
        import librosa
        fb = librosa.filters.mel(
            sr=self.sr, n_fft=self.n_fft, n_mels=self.n_mels,
            fmin=30.0, fmax=self.sr // 2,
        )
        return torch.from_numpy(fb.astype(np.float32)).to(self.device)

    def extract(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract log-mel spectrogram from audio.

        Args:
            audio: (batch, samples) or (samples,) tensor, can be on any device.

        Returns:
            features: (batch, n_frames, n_mels) tensor on self.device
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        # STFT
        stft = torch.stft(
            audio, self.n_fft, hop_length=self.hop_length,
            window=self.window, return_complex=True, center=True,
        )  # (batch, n_fft//2+1, n_frames)
        magnitude = torch.abs(stft)

        # Mel filterbank
        mel = torch.matmul(
            self.mel_fb.unsqueeze(0), magnitude,
        )  # (batch, n_mels, T)

        # Log scale
        mel = torch.log(mel + 1e-6)

        # Transpose to (batch, T, n_mels)
        return mel.permute(0, 2, 1)

    @property
    def n_features(self) -> int:
        return self.n_mels


# ─── ConvStack: 2D local pattern extraction ──────────────────────────────────

class ConvStack(nn.Module):
    """
    2D convolutional stack operating on mel spectrogram as an image.

    Input:  (B, T, N_MELS) -> reshape to (B, 1, T, N_MELS)
    Output: (B, T, out_channels) after pooling over frequency axis.

    Inspired by Onsets & Frames (Hawthorne et al. 2018) ConvStack.
    """

    def __init__(self, n_mels: int = N_MELS, out_channels: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        # 3 conv blocks with batch norm, increasing channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # pool frequency only
            nn.Dropout2d(dropout),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=(1, 2)),  # pool frequency only
            nn.Dropout2d(dropout),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Dropout2d(dropout),
        )

        # After 2x frequency pooling: n_mels // 4
        freq_out = n_mels // 4
        self.fc = nn.Linear(128 * freq_out, out_channels)
        self.fc_norm = nn.LayerNorm(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, N_MELS)
        Returns:
            (B, T, out_channels)
        """
        B, T, F = x.shape
        # Reshape to image: (B, 1, T, F)
        h = x.unsqueeze(1)

        h = self.conv1(h)  # (B, 32, T, F//2)
        h = self.conv2(h)  # (B, 64, T, F//4)
        h = self.conv3(h)  # (B, 128, T, F//4)

        # Flatten frequency axis: (B, 128, T, F//4) -> (B, T, 128 * F//4)
        h = h.permute(0, 2, 1, 3)  # (B, T, 128, F//4)
        h = h.reshape(B, T, -1)

        h = self.fc(h)      # (B, T, out_channels)
        h = self.fc_norm(h)
        return h


# ─── Conformer Building Blocks ─────────────────────────────────────────────
# (Same as train_ensemble.py)

class ConformerFeedForward(nn.Module):
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
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.glu = nn.GLU(dim=1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.activation = nn.SiLU()
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = x.permute(0, 2, 1)
        x = self.pointwise1(x)
        x = self.glu(x)
        x = self.depthwise(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x.permute(0, 2, 1)


class ConformerBlock(nn.Module):
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
        x = x + 0.5 * self.ff1(x)
        residual = x
        x_norm = self.attn_norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + self.attn_dropout(attn_out)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.final_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, x, use_reentrant=False
            )
        return self._forward_impl(x)


class SinusoidalPositionalEncoding(nn.Module):
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
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ─── Mel Baseline Transcriber ───────────────────────────────────────────────

class MelBaselineTranscriber(nn.Module):
    """
    ConvStack + Conformer transcriber using only log-mel input.

    Architecture:
    1. ConvStack: 2D convolutions extract local patterns from mel spectrogram
    2. Conformer temporal backbone: captures long-range temporal context
    3. Per-key projection + output heads
    4. Velocity-gated onset refinement (Kong et al. 2020): second-stage BiGRU
       that uses velocity predictions to suppress false positive onsets

    ~7M+ params with default settings.
    """

    def __init__(self, n_mels: int = N_MELS,
                 conv_out: int = 128,
                 d_model: int = 192,
                 n_layers: int = 6,
                 n_heads: int = 4,
                 ff_expansion: int = 4,
                 conv_kernel: int = 31,
                 dropout: float = 0.1,
                 use_checkpoint: bool = False):
        super().__init__()
        self.n_keys = PIANO_KEYS
        self.n_nv = NOTE_VALUE_CLASSES

        # 1. ConvStack: mel -> per-frame features
        self.conv_stack = ConvStack(n_mels, conv_out, dropout)

        # 2. Project to Conformer d_model
        self.proj = nn.Linear(conv_out, d_model)

        # 3. Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=2048, dropout=dropout)

        # 4. Conformer stack
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ff_expansion=ff_expansion,
                conv_kernel=conv_kernel,
                dropout=dropout,
                use_checkpoint=use_checkpoint,
            )
            for _ in range(n_layers)
        ])

        # 5. Per-key projection: (B, T, d_model) -> (B, T, 88, key_dim)
        key_dim = d_model // 4
        self.key_proj = nn.Sequential(
            nn.Linear(d_model, PIANO_KEYS * key_dim),
            nn.GELU(),
        )
        self.key_dim = key_dim

        # 6. Per-key temporal conv (dilated causal, shared across keys)
        self.key_temporal = nn.Sequential(
            nn.Conv1d(key_dim, key_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(key_dim, key_dim, kernel_size=7, padding=6, dilation=2),
            nn.GELU(),
            nn.Conv1d(key_dim, key_dim, kernel_size=7, padding=12, dilation=4),
            nn.GELU(),
        )
        # Zero-init last conv so residual starts as identity
        nn.init.zeros_(self.key_temporal[-2].weight)
        nn.init.zeros_(self.key_temporal[-2].bias)

        # 7. Raw output heads (per key, shared weights)
        self.onset_head_raw = nn.Sequential(
            nn.Linear(key_dim, key_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(key_dim, 1),
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(key_dim, key_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(key_dim, 1), nn.Sigmoid(),
        )

        # 8. Velocity-gated onset refinement (Kong et al. 2020)
        # Input: concat(raw_onset, sqrt(raw_onset) * velocity) per key = 2 dims
        # BiGRU refines onset predictions using velocity context
        self.onset_refine_gru = nn.GRU(
            input_size=PIANO_KEYS * 2,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )
        self.onset_refine_fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, PIANO_KEYS),  # 128*2 bidirectional -> 88 keys
        )

        # 9. Frame head (conditioned on refined onset)
        self.frame_head = nn.Sequential(
            nn.Linear(key_dim + 1, key_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(key_dim, 1),
        )
        self.note_value_head = nn.Sequential(
            nn.Linear(key_dim, key_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(key_dim, NOTE_VALUE_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, N_MELS) log-mel spectrogram

        Returns:
            dict with onset_logits, frame_logits, velocity (B, T, 88)
                  and note_value_logits (B, T, 88, 10)
        """
        B, T, _ = x.shape

        # 1. ConvStack
        h = self.conv_stack(x)  # (B, T, conv_out)

        # 2. Project + positional encoding
        h = self.proj(h)  # (B, T, d_model)
        h = self.pos_enc(h)

        # 3. Conformer
        for block in self.conformer_blocks:
            h = block(h)  # (B, T, d_model)

        # 4. Per-key projection
        key_h = self.key_proj(h)  # (B, T, 88 * key_dim)
        key_h = key_h.reshape(B, T, self.n_keys, self.key_dim)  # (B, T, 88, key_dim)

        # 5. Per-key temporal conv
        # (B, T, 88, D) -> (B*88, D, T) -> conv -> (B*88, D, T) -> (B, T, 88, D)
        h_t = key_h.permute(0, 2, 3, 1).reshape(B * self.n_keys, self.key_dim, T)
        h_t = self.key_temporal(h_t)
        key_h = key_h + h_t.reshape(B, self.n_keys, self.key_dim, T).permute(0, 3, 1, 2)

        # 6. Raw onset and velocity predictions
        raw_onset_logits = self.onset_head_raw(key_h).squeeze(-1)  # (B, T, 88)
        velocity = self.velocity_head(key_h).squeeze(-1)  # (B, T, 88)

        # 7. Velocity-gated onset refinement
        # Combine raw onset activation with velocity-modulated onset
        # Compute sigmoid in float32 BEFORE sqrt — under AMP, float16 sigmoid
        # underflows to exact 0 for negative logits, and sqrt backward at 0
        # computes grad/(2*0) = NaN. By casting logits to float32 first,
        # sigmoid preserves tiny values (e.g. 1e-20) that sqrt can handle.
        raw_onset_prob = torch.sigmoid(raw_onset_logits.float())
        vel_gated = raw_onset_prob.sqrt() * velocity.float().detach()  # (B, T, 88)
        refine_input = torch.cat([raw_onset_prob, vel_gated], dim=-1)  # (B, T, 176)
        refine_out, _ = self.onset_refine_gru(refine_input)  # (B, T, 256)
        onset_logits = self.onset_refine_fc(refine_out)  # (B, T, 88)

        # 8. Frame head conditioned on refined onset
        frame_in = torch.cat([key_h, onset_logits.unsqueeze(-1).detach()], dim=-1)
        frame_logits = self.frame_head(frame_in).squeeze(-1)  # (B, T, 88)

        nv_logits = self.note_value_head(key_h)  # (B, T, 88, 10)

        return {
            'onset_logits': onset_logits,
            'raw_onset_logits': raw_onset_logits,
            'frame_logits': frame_logits,
            'velocity': velocity,
            'note_value_logits': nv_logits,
        }


# ─── Loss Function ──────────────────────────────────────────────────────────
# (Same as train_ensemble.py)

class EnsembleLoss(nn.Module):
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
                note_value_logits=None, note_value_gt=None,
                raw_onset_logits=None) -> Dict[str, torch.Tensor]:
        vel_weight = torch.ones_like(velocity_gt)
        active = frame_gt > 0.5
        if active.any():
            vel_weight[active] = 1.0 + self.alpha * (1.0 - velocity_gt[active])

        # Onset loss — supports continuous regression targets (tent-shaped)
        # With regression targets, onset_gt is continuous [0,1] not binary.
        # pos_weight is applied proportionally to the target value.
        onset_bce = F.binary_cross_entropy_with_logits(onset_logits, onset_gt, reduction='none')
        # Smooth weighting: scale pos_weight by onset_gt value (0=background, 1=peak)
        onset_sample_w = 1.0 + (self.pos_weight - 1.0) * onset_gt
        onset_loss = (onset_bce * onset_sample_w).mean()

        # Auxiliary raw onset loss (0.5x weight) — gives onset_head_raw direct
        # gradient signal so the BiGRU gets meaningful input from epoch 1
        if raw_onset_logits is not None:
            raw_onset_bce = F.binary_cross_entropy_with_logits(
                raw_onset_logits, onset_gt, reduction='none')
            raw_onset_loss = (raw_onset_bce * onset_sample_w).mean()
        else:
            raw_onset_loss = torch.tensor(0.0, device=onset_logits.device)

        # Frame focal loss
        frame_probs = torch.sigmoid(frame_logits)
        frame_p_t = torch.where(frame_gt > 0.5, frame_probs, 1.0 - frame_probs)
        frame_focal_w = (1.0 - frame_p_t.detach()) ** self.focal_gamma
        frame_bce = F.binary_cross_entropy_with_logits(frame_logits, frame_gt, reduction='none')
        frame_sample_w = torch.where(
            frame_gt > 0.5, vel_weight * self.pos_weight, torch.ones_like(vel_weight))
        frame_loss = (frame_focal_w * frame_bce * frame_sample_w).mean()

        # Velocity MSE
        if active.any():
            velocity_loss = F.mse_loss(velocity_pred[active], velocity_gt[active])
        else:
            velocity_loss = torch.tensor(0.0, device=onset_logits.device)

        total = (self.onset_w * onset_loss + 0.5 * raw_onset_loss
                 + self.frame_w * frame_loss + self.vel_w * velocity_loss)

        result = {
            'total': total, 'onset': onset_loss, 'raw_onset': raw_onset_loss,
            'frame': frame_loss, 'velocity': velocity_loss,
        }

        # Note-value loss (onset frames only — note value is a note-level property)
        if note_value_logits is not None and note_value_gt is not None:
            onset_mask = onset_gt > 0.5
            if onset_mask.any():
                nv_logits_flat = note_value_logits[onset_mask]
                nv_gt_flat = note_value_gt[onset_mask]
                gamma = self.focal_gamma
                probs = F.softmax(nv_logits_flat, dim=-1)
                p_t = probs.gather(1, nv_gt_flat.unsqueeze(1)).squeeze(1)
                focal_weight = (1 - p_t.detach()) ** gamma
                ce = F.cross_entropy(nv_logits_flat, nv_gt_flat, reduction='none')
                nv_loss = (focal_weight * ce).mean()
            else:
                nv_loss = torch.tensor(0.0, device=onset_logits.device)
            total = total + self.nv_w * nv_loss
            result['total'] = total
            result['note_value'] = nv_loss

        return result


# ─── Dataset ────────────────────────────────────────────────────────────────
# Directly reuses the MAESTRO index from train_ensemble.py.

class MelTranscriptionDataset(Dataset):
    """
    On-the-fly mel extraction from MAESTRO audio+MIDI pairs.
    Reuses the same segment index format as train_ensemble.py.
    """

    def __init__(self, index_path: str, sr: int = SAMPLE_RATE,
                 hop_length: int = HOP_LENGTH, augment: bool = False):
        self.sr = sr
        self.hop_length = hop_length
        self.segment_frames = int(SEGMENT_SECONDS * sr / hop_length)
        self.segment_samples = int(SEGMENT_SECONDS * sr)
        self.augment = augment

        with open(index_path) as f:
            self.index = json.load(f)

        self.segments = self.index['segments']
        self.pieces = self.index['pieces']
        aug_str = " (with augmentation)" if augment else ""
        print(f"[MelDataset] {len(self.segments)} segments from {len(self.pieces)} pieces{aug_str}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        piece = self.pieces[seg['piece_idx']]

        import librosa

        start_sec = seg['start_sec']
        audio, _ = librosa.load(
            piece['audio'], sr=self.sr, mono=True,
            offset=start_sec, duration=SEGMENT_SECONDS,
        )

        if len(audio) < self.segment_samples:
            audio = np.pad(audio, (0, self.segment_samples - len(audio)))
        audio = audio[:self.segment_samples]

        if self.augment:
            gain = np.random.uniform(0.7, 1.3)
            audio = audio * gain
            noise_level = np.random.uniform(0, 0.005)
            audio = audio + np.random.randn(len(audio)).astype(np.float32) * noise_level

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
        """Create onset/frame/velocity/note_value labels from MIDI.

        Onset targets use regression-style tent-shaped targets (Kong et al. 2020):
        peak=1.0 at the true onset, linear decay to 0.0 at ±ONSET_TENT_SEC.
        This teaches the model to output peaky activations instead of broad
        binary blobs, significantly reducing false positives.
        """
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(midi_path)

        onset = np.zeros((self.segment_frames, PIANO_KEYS), dtype=np.float32)
        frame = np.zeros((self.segment_frames, PIANO_KEYS), dtype=np.float32)
        velocity = np.zeros((self.segment_frames, PIANO_KEYS), dtype=np.float32)
        note_value = np.zeros((self.segment_frames, PIANO_KEYS), dtype=np.int64)

        end_sec = start_sec + SEGMENT_SECONDS
        frame_time = self.hop_length / self.sr

        tempo_times, tempos = midi.get_tempo_changes()
        bpm = float(tempos[0]) if len(tempos) > 0 else 120.0
        beat_duration = 60.0 / bpm

        log_nv = np.log2(NOTE_VALUE_BEATS)

        # Tent window radius in frames
        tent_frames = int(ONSET_TENT_SEC / frame_time)  # ~3 frames at hop=256

        all_notes = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                all_notes.append(note)
        all_notes.sort(key=lambda n: n.start)

        # Per-hand IOI map
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

        for note in all_notes:
            if note.end < start_sec or note.start > end_sec:
                continue

            key = note.pitch - MIDI_OFFSET
            if key < 0 or key >= PIANO_KEYS:
                continue

            # Precise onset time relative to segment start
            onset_time_rel = note.start - start_sec
            onset_f = int(onset_time_rel / frame_time)
            offset_f = int((note.end - start_sec) / frame_time)
            onset_f = max(0, min(onset_f, self.segment_frames - 1))
            offset_f = max(0, min(offset_f, self.segment_frames))

            vel_norm = note.velocity / 127.0

            # Regression onset target: tent-shaped around true onset
            # For each frame in the tent window, compute distance to true
            # onset in seconds, then linear ramp from 1.0 to 0.0
            for f in range(max(0, onset_f - tent_frames),
                           min(self.segment_frames, onset_f + tent_frames + 1)):
                f_time = f * frame_time
                dist_sec = abs(f_time - onset_time_rel)
                tent_val = max(0.0, 1.0 - dist_sec / ONSET_TENT_SEC)
                # Use max to handle overlapping onsets from different notes
                onset[f, key] = max(onset[f, key], tent_val)

            for f in range(onset_f, offset_f):
                frame[f, key] = 1.0
                velocity[f, key] = vel_norm

            ioi = ioi_map.get((note.pitch, note.start))
            if ioi is None or ioi < 0.03:
                ioi = note.end - note.start
            ioi_beats = max(0.0625, min(8.0, ioi / beat_duration))
            class_idx = int(np.argmin(np.abs(log_nv - np.log2(ioi_beats))))
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

class PrecomputedMelDataset(Dataset):
    """
    Dataset that loads precomputed mel features from disk.
    """

    def __init__(self, split: str = 'train', augment: bool = False,
                 mixup_alpha: float = 0.0):
        self.split_dir = FEATURES_DIR / split
        self.augment = augment
        self.mixup_alpha = mixup_alpha
        if not self.split_dir.exists():
            raise RuntimeError(
                f"Precomputed mel features not found at {self.split_dir}\n"
                f"Run: python train_mel_baseline.py --precompute"
            )

        self.files = sorted(self.split_dir.glob("*.pt"))
        aug_str = " (with augmentation)" if augment else ""
        print(f"[PrecomputedMelDataset] {len(self.files)} segments from {self.split_dir}{aug_str}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], weights_only=True)

        # Mixup
        if self.augment and self.mixup_alpha > 0 and np.random.random() < 0.5:
            idx_b = np.random.randint(0, len(self.files))
            data_b = torch.load(self.files[idx_b], weights_only=True)
            lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            lam = max(lam, 1.0 - lam)
            T = min(data['features'].size(0), data_b['features'].size(0))
            data['features'] = lam * data['features'][:T] + (1.0 - lam) * data_b['features'][:T]
            for key in ['onset', 'frame', 'velocity']:
                data[key] = lam * data[key][:T] + (1.0 - lam) * data_b[key][:T]
            data['note_value'] = data['note_value'][:T]

        if 'bpm' not in data:
            data['bpm'] = torch.tensor(120.0, dtype=torch.float32)

        # SpecAugment
        if self.augment and 'features' in data:
            features = data['features']  # (T, N_MELS)

            if np.random.random() < 0.5:
                t_mask_len = np.random.randint(1, 6)
                t_start = np.random.randint(0, max(1, features.size(0) - t_mask_len))
                features[t_start:t_start + t_mask_len, :] = 0

            if np.random.random() < 0.5:
                f_mask_len = np.random.randint(5, 30)
                f_start = np.random.randint(0, max(1, features.size(1) - f_mask_len))
                features[:, f_start:f_start + f_mask_len] = 0

            gain = np.random.uniform(0.9, 1.1)
            data['features'] = features * gain

        # Pitch shift augmentation
        if self.augment and np.random.random() < 0.2:
            shift = np.random.choice([-1, 1])
            # Shift mel bins (approximate pitch shift on mel spectrogram)
            features = data['features']
            shifted_features = torch.zeros_like(features)
            # Shift along frequency axis (mel bins roughly pitch-aligned)
            mel_shift = shift * 3  # ~3 mel bins per semitone at 229 bins
            if mel_shift > 0:
                shifted_features[:, mel_shift:] = features[:, :-mel_shift]
            else:
                shifted_features[:, :mel_shift] = features[:, -mel_shift:]
            data['features'] = shifted_features
            # Shift labels along key axis
            for key in ['onset', 'frame', 'velocity', 'note_value']:
                labels = data[key]
                shifted_labels = torch.zeros_like(labels)
                if shift > 0:
                    shifted_labels[:, shift:] = labels[:, :-shift]
                else:
                    shifted_labels[:, :shift] = labels[:, -shift:]
                data[key] = shifted_labels

        return data


# ─── Note Event Decoding ────────────────────────────────────────────────────
# Reuse from train_ensemble (imported at runtime to avoid circular deps)

def _get_decode_note_events():
    """Import decode_note_events from train_ensemble."""
    from train_ensemble import decode_note_events
    return decode_note_events


# ─── Model Builder ──────────────────────────────────────────────────────────

def _build_model_from_config(config: dict) -> nn.Module:
    """Build MelBaselineTranscriber from checkpoint config."""
    return MelBaselineTranscriber(
        n_mels=config.get('n_mels', N_MELS),
        conv_out=config.get('conv_out', 128),
        d_model=config.get('d_model', 192),
        n_layers=config.get('n_layers', 6),
        n_heads=config.get('n_heads', 4),
        ff_expansion=config.get('ff_expansion', 4),
        conv_kernel=config.get('conv_kernel', 31),
        dropout=config.get('dropout', 0.1),
        use_checkpoint=config.get('use_checkpoint', False),
    )


# ─── Feature Precomputation ─────────────────────────────────────────────────

def precompute_features(args):
    """Precompute mel features for all segments."""
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')

    print(f"Precomputing mel features on {device}")
    print(f"Output directory: {FEATURES_DIR}")

    extractor = MelFeatureExtractor(
        sr=SAMPLE_RATE, hop_length=HOP_LENGTH, device=device,
    )

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    for split in ['train', 'validation', 'test']:
        index_path = INDEX_DIR / f"{split}_index.json"
        if not index_path.exists():
            print(f"Index not found: {index_path}, skipping {split}")
            continue

        split_dir = FEATURES_DIR / split
        split_dir.mkdir(parents=True, exist_ok=True)

        with open(index_path) as f:
            index = json.load(f)

        segments = index['segments']
        pieces = index['pieces']

        existing = set(f.stem for f in split_dir.glob("*.pt"))
        to_process = [(i, seg) for i, seg in enumerate(segments)
                      if f"seg_{i:06d}" not in existing]

        if not to_process:
            print(f"  {split}: already complete ({len(segments)} segments)")
            continue

        print(f"  {split}: {len(to_process)}/{len(segments)} segments to process")

        temp_dataset = MelTranscriptionDataset(str(index_path))

        batch_size = args.precompute_batch
        processed = 0
        start_time = time.time()

        for batch_start in range(0, len(to_process), batch_size):
            batch_items = to_process[batch_start:batch_start + batch_size]

            import librosa
            audios = []
            labels_list = []
            indices = []

            for seg_idx, seg in batch_items:
                piece = pieces[seg['piece_idx']]
                start_sec = seg['start_sec']

                try:
                    audio, _ = librosa.load(
                        piece['audio'], sr=SAMPLE_RATE, mono=True,
                        offset=start_sec, duration=SEGMENT_SECONDS,
                    )
                    segment_samples = int(SEGMENT_SECONDS * SAMPLE_RATE)
                    if len(audio) < segment_samples:
                        audio = np.pad(audio, (0, segment_samples - len(audio)))
                    audio = audio[:segment_samples]

                    onset, frame, velocity, note_value, bpm = temp_dataset._create_labels(
                        piece['midi'], start_sec)

                    audios.append(torch.from_numpy(audio).float())
                    labels_list.append((onset, frame, velocity, note_value, bpm))
                    indices.append(seg_idx)
                except Exception as e:
                    print(f"    Error processing segment {seg_idx}: {e}")
                    continue

            if not audios:
                continue

            audio_batch = torch.stack(audios).to(device)
            with torch.no_grad():
                features_batch = extractor.extract(audio_batch)

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
    total_size = sum(f.stat().st_size for f in FEATURES_DIR.rglob("*.pt"))
    print(f"Total size: {total_size / 1e9:.2f} GB")


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
        print(f"Using precomputed mel features from {FEATURES_DIR}")
        train_dataset = PrecomputedMelDataset(
            'train', augment=True, mixup_alpha=args.mixup_alpha)
        val_dataset = PrecomputedMelDataset('validation', augment=False)
        extractor = None
    else:
        extractor = MelFeatureExtractor(
            sr=SAMPLE_RATE, hop_length=HOP_LENGTH, device=device,
        )
        print(f"Feature extractor: {extractor.n_features} mel bins per frame")

        train_index = INDEX_DIR / "train_index.json"
        val_index = INDEX_DIR / "validation_index.json"
        if not train_index.exists():
            print(f"Index not found at {train_index}")
            print("Run: python train_ensemble.py --prepare")
            return

        train_dataset = MelTranscriptionDataset(
            str(train_index), augment=True)
        val_dataset = MelTranscriptionDataset(
            str(val_index), augment=False)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = MelBaselineTranscriber(
        n_mels=N_MELS,
        conv_out=args.conv_out,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        ff_expansion=args.ff_expansion,
        conv_kernel=args.conv_kernel,
        dropout=args.dropout,
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

    warmup_steps = args.warmup_steps
    d_model = args.d_model

    def noam_lambda(step):
        step = max(step, 1)
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)

    best_val_loss = float('inf')
    best_onset_f1 = 0.0
    start_epoch = 0

    # Resume from checkpoint
    if args.resume and MODEL_PATH.exists():
        print(f"Resuming from checkpoint: {MODEL_PATH}")
        checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"  Could not load optimizer state: {e}")
        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print(f"  Restored scheduler state (lr={optimizer.param_groups[0]['lr']:.2e})")
            except Exception as e:
                print(f"  Could not load scheduler state: {e}")
        start_epoch = checkpoint.get('epoch', 0) + 1
        best_onset_f1 = checkpoint.get('onset_f1', 0.0)
        if 'note_value_acc' not in checkpoint:
            print("  Old checkpoint without note_value metrics - resetting best_val_loss")
            best_val_loss = float('inf')
        else:
            best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"  Resuming from epoch {start_epoch}, best_onset_f1={best_onset_f1:.3f}")
    elif args.resume:
        print(f"No checkpoint found at {MODEL_PATH}, starting fresh")

    # AMP
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)

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

            if use_precomputed:
                features = batch['features'].to(device)
            else:
                audio = batch['audio'].to(device)
                with torch.no_grad():
                    features = extractor.extract(audio)

            T = min(features.size(1), onset_gt.size(1))
            features = features[:, :T, :]
            onset_gt = onset_gt[:, :T, :]
            frame_gt = frame_gt[:, :T, :]
            vel_gt = vel_gt[:, :T, :]
            nv_gt = nv_gt[:, :T, :]

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(features)
                losses = criterion(
                    out['onset_logits'], out['frame_logits'], out['velocity'],
                    onset_gt, frame_gt, vel_gt,
                    note_value_logits=out['note_value_logits'],
                    note_value_gt=nv_gt,
                    raw_onset_logits=out['raw_onset_logits'],
                )

            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

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

                with torch.amp.autocast('cuda', enabled=use_amp):
                    out = model(features)
                    losses = criterion(
                        out['onset_logits'], out['frame_logits'], out['velocity'],
                        onset_gt, frame_gt, vel_gt,
                        note_value_logits=out['note_value_logits'],
                        note_value_gt=nv_gt,
                        raw_onset_logits=out['raw_onset_logits'],
                    )
                for k, v in losses.items():
                    val_losses[k] += v.item()
                n_val += 1

                onset_probs_val = torch.sigmoid(out['onset_logits'])
                onset_pred = (onset_probs_val > 0.5).float()
                frame_pred = (torch.sigmoid(out['frame_logits']) > 0.5).float()

                # Binarize regression onset GT at 0.5 for F1 computation
                # (tent peak is 1.0, frames at ~half the tent are 0.5)
                onset_gt_bin = (onset_gt > 0.5).float()
                onset_tp += ((onset_pred == 1) & (onset_gt_bin == 1)).sum().item()
                onset_fp += ((onset_pred == 1) & (onset_gt_bin == 0)).sum().item()
                onset_fn += ((onset_pred == 0) & (onset_gt_bin == 1)).sum().item()

                onset_mask_val = onset_gt > 0.5
                if onset_mask_val.any():
                    nv_pred_class = out['note_value_logits'][onset_mask_val].argmax(dim=-1)
                    nv_gt_class = nv_gt[onset_mask_val]
                    nv_correct += (nv_pred_class == nv_gt_class).sum().item()
                    nv_total += nv_gt_class.numel()

                frame_tp += ((frame_pred == 1) & (frame_gt == 1)).sum().item()
                frame_fp += ((frame_pred == 1) & (frame_gt == 0)).sum().item()
                frame_fn += ((frame_pred == 0) & (frame_gt == 1)).sum().item()

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

        # Save best model (by onset F1)
        if onset_f1 > best_onset_f1:
            best_onset_f1 = onset_f1
            best_val_loss = avg_val['total']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': {
                    'model_type': 'MelBaselineTranscriber',
                    'n_mels': N_MELS,
                    'conv_out': args.conv_out,
                    'd_model': args.d_model,
                    'n_layers': args.n_layers,
                    'n_heads': args.n_heads,
                    'ff_expansion': args.ff_expansion,
                    'conv_kernel': args.conv_kernel,
                    'dropout': args.dropout,
                    'use_checkpoint': args.use_checkpoint,
                    'n_keys': PIANO_KEYS,
                    'n_note_value_classes': NOTE_VALUE_CLASSES,
                    'sample_rate': SAMPLE_RATE,
                    'hop_length': HOP_LENGTH,
                    'n_fft': N_FFT,
                    'nv_pooled': True,
                    'regression_onset': True,
                    'velocity_gated_onset': True,
                    'onset_tent_sec': ONSET_TENT_SEC,
                    'pos_weight': args.pos_weight,
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

    # ── Threshold sweep ──
    print(f"\n{'='*60}")
    print("Threshold sweep on validation set (using best saved model)...")
    print(f"{'='*60}")
    checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    thresholds = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
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

            # Binarize regression onset GT for threshold sweep
            onset_gt_bin = (onset_gt > 0.5).float()

            for t in thresholds:
                o_pred = (onset_probs > t).float()
                onset_counts[t]['tp'] += ((o_pred == 1) & (onset_gt_bin == 1)).sum().item()
                onset_counts[t]['fp'] += ((o_pred == 1) & (onset_gt_bin == 0)).sum().item()
                onset_counts[t]['fn'] += ((o_pred == 0) & (onset_gt_bin == 1)).sum().item()

                f_pred = (frame_probs > t).float()
                frame_counts[t]['tp'] += ((f_pred == 1) & (frame_gt == 1)).sum().item()
                frame_counts[t]['fp'] += ((f_pred == 1) & (frame_gt == 0)).sum().item()
                frame_counts[t]['fn'] += ((f_pred == 0) & (frame_gt == 1)).sum().item()

    print(f"\n{'Thresh':>6}  {'Onset P':>8} {'Onset R':>8} {'Onset F1':>8}  |  {'Frame P':>8} {'Frame R':>8} {'Frame F1':>8}")
    print("-" * 80)
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

        print(f"  {t:.2f}   {op:>8.3f} {orc:>8.3f} {of1:>8.3f}    |  {fp:>8.3f} {frc:>8.3f} {ff1:>8.3f}")

    print(f"\n  Best onset threshold: {best_onset_t:.2f} -> F1={best_onset_f1_sweep:.3f}")
    print(f"  Best frame threshold: {best_frame_t:.2f} -> F1={best_frame_f1_sweep:.3f}")


# ─── Benchmark ──────────────────────────────────────────────────────────────

def benchmark(args):
    """Benchmark mel baseline inference speed."""
    device = torch.device(args.device)

    test_audio = np.random.randn(SAMPLE_RATE * 10).astype(np.float32) * 0.1
    n_runs = 10

    print("=" * 60)
    print("BENCHMARK: Mel Baseline transcription speed")
    print("=" * 60)

    if MODEL_PATH.exists():
        checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
        config = checkpoint.get('config', {})
        extractor = MelFeatureExtractor(sr=SAMPLE_RATE, device=device)
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
        mel_ms = (time.perf_counter() - start) / n_runs * 1000
        print(f"\nMel Baseline: {mel_ms:.1f} ms / 10s audio")
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model params: {n_params:,}")
    else:
        print(f"\nModel not found at {MODEL_PATH}, skipping")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Train mel-only baseline transcriber for LiveScore')

    # Actions
    parser.add_argument('--precompute', action='store_true',
                        help='Precompute mel features for faster training')
    parser.add_argument('--train', action='store_true',
                        help='Train the mel baseline model')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark inference speed')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1.0,
                        help='Base LR for Noam schedule')
    parser.add_argument('--conv-out', type=int, default=128,
                        help='ConvStack output channels')
    parser.add_argument('--d-model', type=int, default=192,
                        help='Conformer d_model dimension')
    parser.add_argument('--n-layers', type=int, default=6,
                        help='Number of Conformer blocks')
    parser.add_argument('--n-heads', type=int, default=4,
                        help='Conformer attention heads')
    parser.add_argument('--ff-expansion', type=int, default=4,
                        help='Conformer feed-forward expansion factor')
    parser.add_argument('--conv-kernel', type=int, default=31,
                        help='Conformer depthwise conv kernel size')
    parser.add_argument('--warmup-steps', type=int, default=4000,
                        help='Noam schedule warmup steps')
    parser.add_argument('--use-checkpoint', action='store_true', default=True,
                        help='Gradient checkpointing')
    parser.add_argument('--no-checkpoint', action='store_false', dest='use_checkpoint')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--vel-alpha', type=float, default=2.0)
    parser.add_argument('--pos-weight', type=float, default=2.0)
    parser.add_argument('--nv-weight', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from last checkpoint')
    parser.add_argument('--precompute-batch', type=int, default=16)
    parser.add_argument('--mixup-alpha', type=float, default=0.0)

    args = parser.parse_args()

    if args.precompute:
        precompute_features(args)
    elif args.train:
        train(args)
    elif args.benchmark:
        benchmark(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
