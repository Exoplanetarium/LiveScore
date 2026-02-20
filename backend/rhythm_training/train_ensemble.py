"""
Multi-resolution ensemble transcriber for LiveScore.

Architecture:
  1. Multi-resolution feature extraction (GPU-parallel):
     - 3 mel spectrograms at different STFT window sizes (1024/2048/4096)
     - CQT via filterbank on 4096-STFT
     - Chromagram (folded CQT)
     - 9 onset detection functions (flux/energy/HFC x 3 resolutions)
     Total: 373 features per frame

  2. Meta-learner (Conv1d + BiGRU, ~770K params):
     - 3x Conv1d for local pattern extraction
     - 2-layer BiGRU for temporal context
     - 3 heads: onset (88 keys), frame (88 keys), velocity (88 keys)

Key design: GPU-parallel signal processing provides rich, diverse features.
A small trained model learns which features to trust for each situation.
Result: 10-15x faster than ByteDance with competitive accuracy.

Training data: MAESTRO v3.0.0 (aligned audio + MIDI)
Features are computed on-the-fly during training (no preprocessed dataset).

Usage:
    # 1. Prepare segment index from MAESTRO
    python train_ensemble.py --prepare

    # 2. Train
    python train_ensemble.py --train --epochs 50 --batch-size 8

    # 3. Benchmark against ByteDance
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
N_MELS = 88               # per resolution (piano-key aligned)
PIANO_KEYS = 88
MIDI_OFFSET = 21           # A0
N_FFTS = [1024, 2048, 4096]
CQT_BINS = 88
CHROMA_BINS = 12
ONSET_FEATURES = 9         # 3 functions x 3 resolutions
N_FEATURES = N_MELS * len(N_FFTS) + CQT_BINS + CHROMA_BINS + ONSET_FEATURES  # 373

SEGMENT_SECONDS = 10.0
SEGMENT_FRAMES = int(SEGMENT_SECONDS * SAMPLE_RATE / HOP_LENGTH)

MAESTRO_DIR = Path(__file__).parent / "maestro_midi"
MAESTRO_CSV = MAESTRO_DIR / "maestro-v3.0.0.csv"
INDEX_DIR = Path(__file__).parent / "ensemble_index"
MODEL_PATH = Path(__file__).parent / "ensemble_transcription.pt"


# ─── Multi-Resolution Feature Extractor ─────────────────────────────────────

class MultiResFeatureExtractor:
    """
    GPU-accelerated multi-resolution feature extraction.

    Computes 373 features per frame from audio in a single GPU pass:
      - 3 x 88 mel spectrograms (n_fft = 1024, 2048, 4096)
      - 88 CQT bins (filterbank on 4096-STFT)
      - 12 chroma bins (folded from CQT)
      - 9 onset functions (spectral flux, RMS energy, HFC x 3 resolutions)

    All filterbanks are precomputed and cached on GPU.
    """

    def __init__(self, sr: int = SAMPLE_RATE, hop_length: int = HOP_LENGTH,
                 n_mels: int = N_MELS, n_ffts: List[int] = None,
                 device: torch.device = None):
        self.sr = sr
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_ffts = n_ffts or N_FFTS
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    def extract(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Extract all 373 features from audio.

        Args:
            audio: (batch, samples) or (samples,) tensor, can be on any device.

        Returns:
            features: (batch, n_frames, 373) tensor on self.device
        """
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        audio = audio.to(self.device)

        # Compute 3 STFTs (same hop, different window sizes)
        magnitudes = {}
        for n_fft in self.n_ffts:
            stft = torch.stft(
                audio, n_fft, hop_length=self.hop_length,
                window=self.windows[n_fft], return_complex=True, center=True,
            )  # (batch, n_fft//2+1, n_frames)
            magnitudes[n_fft] = torch.abs(stft)

        # Align frame counts (center=True with same hop → same count, but guard)
        n_frames = min(m.size(-1) for m in magnitudes.values())
        for n_fft in self.n_ffts:
            magnitudes[n_fft] = magnitudes[n_fft][:, :, :n_frames]

        parts = []

        # ── 1. Multi-resolution mel spectrograms (3 x 88 = 264) ──
        for n_fft in self.n_ffts:
            # mel_fb: (n_mels, n_fft//2+1) @ mag: (batch, n_fft//2+1, T)
            mel = torch.matmul(
                self.mel_fbs[n_fft].unsqueeze(0),  # (1, n_mels, freq)
                magnitudes[n_fft],                  # (batch, freq, T)
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
        # Fold 88 CQT bins into 12 pitch classes (7 full octaves + 4 extra keys)
        batch_size = cqt.size(0)
        # Pad CQT to 96 bins (8 octaves) for clean folding, then take 12
        cqt_padded = F.pad(cqt, (0, 0, 0, 96 - CQT_BINS))  # (batch, 96, T)
        chroma = cqt_padded.view(batch_size, 8, 12, n_frames).sum(dim=1)  # (batch, 12, T)
        chroma = chroma / (chroma.sum(dim=1, keepdim=True) + 1e-8)
        parts.append(chroma)

        # ── 4. Onset detection functions (9) ──
        for n_fft in self.n_ffts:
            mag = magnitudes[n_fft]  # (batch, freq, T)

            # Spectral flux (half-wave rectified difference)
            diff = torch.diff(mag, dim=-1)  # (batch, freq, T-1)
            diff = torch.clamp(diff, min=0)
            flux = diff.sum(dim=1)  # (batch, T-1)
            flux = F.pad(flux, (1, 0))  # (batch, T)
            flux = flux / (flux.max(dim=-1, keepdim=True).values + 1e-8)
            parts.append(flux.unsqueeze(1))  # (batch, 1, T)

            # RMS energy
            rms = torch.sqrt(torch.mean(mag ** 2, dim=1))  # (batch, T)
            rms = rms / (rms.max(dim=-1, keepdim=True).values + 1e-8)
            parts.append(rms.unsqueeze(1))

            # High-frequency content (frequency-weighted energy)
            freq_weights = torch.linspace(0, 1, mag.size(1), device=self.device)
            hfc = (mag ** 2 * freq_weights.view(1, -1, 1)).sum(dim=1)
            hfc = torch.sqrt(hfc)
            hfc = hfc / (hfc.max(dim=-1, keepdim=True).values + 1e-8)
            parts.append(hfc.unsqueeze(1))

        # Stack all features: (batch, 373, T)
        all_features = torch.cat(parts, dim=1)

        # Transpose to model convention: (batch, T, 373)
        return all_features.permute(0, 2, 1)

    @property
    def n_features(self) -> int:
        """Total features per frame."""
        return N_MELS * len(self.n_ffts) + CQT_BINS + CHROMA_BINS + ONSET_FEATURES


# ─── Meta-Learner Model ─────────────────────────────────────────────────────

class EnsembleMetaLearner(nn.Module):
    """
    Conv1d + BiGRU meta-learner for multi-resolution feature fusion.

    ~770K params:
      - 3x Conv1d: local time-frequency pattern extraction
      - 2-layer BiGRU: temporal context for onset/frame prediction
      - 3 output heads: onset, frame, velocity (per 88 piano keys)
    """

    def __init__(self, n_features: int = N_FEATURES,
                 conv_channels: List[int] = None,
                 gru_hidden: int = 64, gru_layers: int = 2,
                 n_keys: int = PIANO_KEYS, dropout: float = 0.1):
        super().__init__()
        if conv_channels is None:
            conv_channels = [256, 256, 128]

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
        self.frame_head = nn.Sequential(
            nn.Linear(gru_out_dim, gru_out_dim),
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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, T, n_features) multi-resolution features

        Returns:
            dict with onset_logits, frame_logits, velocity — all (batch, T, 88)
        """
        # Conv1d expects (batch, channels, time)
        h = x.permute(0, 2, 1)  # (batch, n_features, T)

        h = F.gelu(self.bn1(self.conv1(h)))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = F.gelu(self.bn3(self.conv3(h)))

        # BiGRU expects (batch, time, features)
        h = h.permute(0, 2, 1)  # (batch, T, conv_channels[-1])

        h, _ = self.gru(h)  # (batch, T, gru_hidden*2)

        return {
            'onset_logits': self.onset_head(h),
            'frame_logits': self.frame_head(h),
            'velocity': self.velocity_head(h),
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
                 velocity_weight: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.pos_weight = pos_weight
        self.onset_w = onset_weight
        self.frame_w = frame_weight
        self.vel_w = velocity_weight

    def forward(self, onset_logits, frame_logits, velocity_pred,
                onset_gt, frame_gt, velocity_gt) -> Dict[str, torch.Tensor]:

        # Velocity-based per-sample weighting
        vel_weight = torch.ones_like(velocity_gt)
        active = frame_gt > 0.5
        if active.any():
            vel_weight[active] = 1.0 + self.alpha * (1.0 - velocity_gt[active])

        # Onset BCE (weighted)
        onset_bce = F.binary_cross_entropy_with_logits(
            onset_logits, onset_gt, reduction='none',
        )
        onset_sample_w = torch.where(
            onset_gt > 0.5,
            vel_weight * self.pos_weight,
            torch.ones_like(vel_weight),
        )
        onset_loss = (onset_bce * onset_sample_w).mean()

        # Frame BCE (weighted)
        frame_bce = F.binary_cross_entropy_with_logits(
            frame_logits, frame_gt, reduction='none',
        )
        frame_sample_w = torch.where(
            frame_gt > 0.5,
            vel_weight * self.pos_weight,
            torch.ones_like(vel_weight),
        )
        frame_loss = (frame_bce * frame_sample_w).mean()

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

        return {
            'total': total,
            'onset': onset_loss,
            'frame': frame_loss,
            'velocity': velocity_loss,
        }


# ─── Dataset ────────────────────────────────────────────────────────────────

class EnsembleTranscriptionDataset(Dataset):
    """
    On-the-fly feature computation from MAESTRO audio+MIDI pairs.

    Instead of storing preprocessed features (~150GB), loads audio segments
    and computes multi-resolution features on GPU during training.
    The 3 STFTs + filterbank multiplies take <10ms per segment on GPU.
    """

    def __init__(self, index_path: str, sr: int = SAMPLE_RATE,
                 hop_length: int = HOP_LENGTH):
        self.sr = sr
        self.hop_length = hop_length
        self.segment_frames = SEGMENT_FRAMES
        self.segment_samples = self.segment_frames * hop_length

        with open(index_path) as f:
            self.index = json.load(f)

        self.segments = self.index['segments']
        self.pieces = self.index['pieces']
        print(f"[Dataset] {len(self.segments)} segments from {len(self.pieces)} pieces")

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

        # Create frame-level labels from MIDI
        onset, frame, velocity = self._create_labels(piece['midi'], start_sec)

        return {
            'audio': torch.from_numpy(audio).float(),
            'onset': onset,
            'frame': frame,
            'velocity': velocity,
        }

    def _create_labels(self, midi_path: str, start_sec: float):
        """Create onset/frame/velocity labels from MIDI for this segment."""
        import pretty_midi
        midi = pretty_midi.PrettyMIDI(midi_path)

        onset = np.zeros((self.segment_frames, PIANO_KEYS), dtype=np.float32)
        frame = np.zeros((self.segment_frames, PIANO_KEYS), dtype=np.float32)
        velocity = np.zeros((self.segment_frames, PIANO_KEYS), dtype=np.float32)

        end_sec = start_sec + SEGMENT_SECONDS
        frame_time = self.hop_length / self.sr

        for instrument in midi.instruments:
            if instrument.is_drum:
                continue
            for note in instrument.notes:
                if note.end < start_sec or note.start > end_sec:
                    continue

                key = note.pitch - MIDI_OFFSET
                if key < 0 or key >= PIANO_KEYS:
                    continue

                # Frames relative to segment start
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

        return (
            torch.from_numpy(onset),
            torch.from_numpy(frame),
            torch.from_numpy(velocity),
        )


# ─── Note Event Decoding ────────────────────────────────────────────────────

def decode_note_events(
    onset_probs: np.ndarray,
    frame_probs: np.ndarray,
    velocity: np.ndarray,
    sr: int = SAMPLE_RATE,
    hop: int = HOP_LENGTH,
    onset_threshold: float = 0.4,
    frame_threshold: float = 0.3,
    min_note_duration: float = 0.05,
    min_velocity: int = 15,  # Filter very soft false positives
    dedup_window: float = 0.05,  # Duplicate detection window (seconds)
    use_peak_picking: bool = True,  # Only keep onset peaks
    filter_harmonics: bool = True,  # Remove likely harmonic false positives
) -> List[Dict]:
    """
    Decode frame-level onset/frame/velocity predictions into note events.

    Args:
        onset_probs: (n_frames, 88) onset probabilities
        frame_probs: (n_frames, 88) frame probabilities
        velocity: (n_frames, 88) velocity predictions [0, 1]
        sr, hop: for time conversion
        onset_threshold, frame_threshold: detection thresholds
        min_note_duration: minimum note length in seconds
        min_velocity: minimum velocity to keep (0-127)
        dedup_window: window for duplicate detection (seconds)
        use_peak_picking: only keep local maxima in onset probabilities
        filter_harmonics: remove notes that are likely harmonics of louder notes

    Returns:
        List of dicts: {'onset_time', 'offset_time', 'midi_note', 'velocity'}
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

            note_events.append({
                'onset_time': float(onset_f * frame_time),
                'offset_time': float(offset_f * frame_time),
                'midi_note': int(key + MIDI_OFFSET),
                'velocity': vel_int,
                'onset_prob': float(onset_probs[onset_f, key]),  # Keep for filtering
            })

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

    # Remove internal fields before returning
    for event in filtered:
        event.pop('onset_prob', None)

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


# ─── Training ───────────────────────────────────────────────────────────────

def train(args):
    """Main training loop."""
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Feature extractor (lives on GPU, shared across batches)
    extractor = MultiResFeatureExtractor(
        sr=SAMPLE_RATE, hop_length=HOP_LENGTH, device=device,
    )
    print(f"Feature extractor: {extractor.n_features} features per frame")

    # Datasets
    train_index = INDEX_DIR / "train_index.json"
    val_index = INDEX_DIR / "validation_index.json"
    if not train_index.exists():
        print(f"Index not found at {train_index}")
        print("Run: python train_ensemble.py --prepare")
        return

    train_dataset = EnsembleTranscriptionDataset(str(train_index))
    val_dataset = EnsembleTranscriptionDataset(str(val_index))

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    conv_channels = [256, 256, 128]
    model = EnsembleMetaLearner(
        n_features=extractor.n_features,
        conv_channels=conv_channels,
        gru_hidden=args.gru_hidden,
        gru_layers=args.gru_layers,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = EnsembleLoss(
        alpha=args.vel_alpha, pos_weight=args.pos_weight,
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')
    best_onset_f1 = 0.0

    # AMP: mixed precision for speedup on GPUs with tensor cores
    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        train_losses = defaultdict(float)
        n_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            audio = batch['audio'].to(device)
            onset_gt = batch['onset'].to(device)
            frame_gt = batch['frame'].to(device)
            vel_gt = batch['velocity'].to(device)

            # Extract multi-resolution features on GPU
            with torch.no_grad():
                features = extractor.extract(audio)  # (B, T, 373)

            # Trim features and labels to the shorter of the two
            T = min(features.size(1), onset_gt.size(1))
            features = features[:, :T, :]
            onset_gt = onset_gt[:, :T, :]
            frame_gt = frame_gt[:, :T, :]
            vel_gt = vel_gt[:, :T, :]

            optimizer.zero_grad()

            # AMP: forward pass in float16
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(features)

                losses = criterion(
                    out['onset_logits'], out['frame_logits'], out['velocity'],
                    onset_gt, frame_gt, vel_gt,
                )

            # AMP: scaled backward + optimizer step
            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            for k, v in losses.items():
                train_losses[k] += v.item()
            n_batches += 1

            if batch_idx % 100 == 0 and batch_idx > 0:
                avg = train_losses['total'] / n_batches
                print(f"  Epoch {epoch+1} batch {batch_idx}: loss={avg:.4f}")

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_losses = defaultdict(float)
        n_val = 0
        onset_tp, onset_fp, onset_fn = 0, 0, 0
        frame_tp, frame_fp, frame_fn = 0, 0, 0

        with torch.no_grad():
            for batch in val_loader:
                audio = batch['audio'].to(device)
                onset_gt = batch['onset'].to(device)
                frame_gt = batch['frame'].to(device)
                vel_gt = batch['velocity'].to(device)

                features = extractor.extract(audio)
                T = min(features.size(1), onset_gt.size(1))
                features = features[:, :T, :]
                onset_gt = onset_gt[:, :T, :]
                frame_gt = frame_gt[:, :T, :]
                vel_gt = vel_gt[:, :T, :]

                # AMP: validation forward in float16
                with torch.amp.autocast('cuda', enabled=use_amp):
                    out = model(features)
                    losses = criterion(
                        out['onset_logits'], out['frame_logits'], out['velocity'],
                        onset_gt, frame_gt, vel_gt,
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

        avg_train = {k: v / max(n_batches, 1) for k, v in train_losses.items()}
        avg_val = {k: v / max(n_val, 1) for k, v in val_losses.items()}

        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"  Train loss: {avg_train['total']:.4f} "
              f"(onset={avg_train['onset']:.4f}, frame={avg_train['frame']:.4f}, "
              f"vel={avg_train['velocity']:.4f})")
        print(f"  Val loss:   {avg_val['total']:.4f}")
        print(f"  Onset  P={onset_p:.3f} R={onset_r:.3f} F1={onset_f1:.3f}")
        print(f"  Frame  P={frame_p:.3f} R={frame_r:.3f} F1={frame_f1:.3f}")

        # Save best model
        if avg_val['total'] < best_val_loss:
            best_val_loss = avg_val['total']
            best_onset_f1 = onset_f1
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'n_features': extractor.n_features,
                    'conv_channels': conv_channels,
                    'gru_hidden': args.gru_hidden,
                    'gru_layers': args.gru_layers,
                    'n_keys': PIANO_KEYS,
                    'sample_rate': SAMPLE_RATE,
                    'hop_length': HOP_LENGTH,
                },
                'epoch': epoch,
                'val_loss': best_val_loss,
                'onset_f1': onset_f1,
                'frame_f1': frame_f1,
            }, str(MODEL_PATH))
            print(f"  Saved best model! (val_loss={best_val_loss:.4f}, onset_f1={onset_f1:.3f})")

    print(f"\nTraining complete!")
    print(f"  Best val loss: {best_val_loss:.4f}")
    print(f"  Best onset F1: {best_onset_f1:.3f}")
    print(f"  Model saved to: {MODEL_PATH}")


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
        extractor = MultiResFeatureExtractor(sr=SAMPLE_RATE, device=device)
        model = EnsembleMetaLearner(n_features=extractor.n_features)

        checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
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
    parser.add_argument('--train', action='store_true',
                        help='Train the ensemble meta-learner')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark inference speed vs ByteDance')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--gru-hidden', type=int, default=64)
    parser.add_argument('--gru-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--vel-alpha', type=float, default=2.0,
                        help='Velocity weighting (higher = more soft note emphasis)')
    parser.add_argument('--pos-weight', type=float, default=5.0,
                        help='Positive class weight for onset/frame BCE')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)

    args = parser.parse_args()

    if args.prepare:
        prepare_segment_index()
    elif args.train:
        train(args)
    elif args.benchmark:
        benchmark(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
