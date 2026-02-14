"""
Custom piano transcription model for LiveScore.

Architecture: Onsets-and-Frames variant with velocity-weighted loss.
  - Input: Log-mel spectrogram (229 mel bins)
  - Encoder: ConvStack (3 layers) → Bidirectional Transformer (4 layers)
  - Output heads (per frame, per 88 piano keys):
      1. Onset detection (binary)
      2. Frame activation (binary)
      3. Velocity estimation (continuous 0-1)

Key design choice for soft accompaniment detection:
  The loss function weights each note inversely by velocity, so missing a
  pianissimo note (velocity ~30) incurs 2-3x more penalty than missing a
  fortissimo note (velocity ~100). This forces the model to be sensitive
  to soft accompaniment patterns.

Training data: MAESTRO v3.0.0 (aligned audio + MIDI)
  - Download full dataset: https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip
  - Or audio only: the WAV files referenced in maestro-v3.0.0.csv

Usage:
    # 1. Download MAESTRO audio (if you only have MIDI)
    python train_transcription.py --download-audio

    # 2. Prepare training data (mel spectrograms + frame labels)
    python train_transcription.py --prepare

    # 3. Train
    python train_transcription.py --train --epochs 50 --batch-size 8

    # 4. Export for inference
    python train_transcription.py --export
"""

import csv
import json
import math
import os
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ─── Constants ───────────────────────────────────────────────────────────────

SAMPLE_RATE = 16000       # Standard for transcription models (matches ByteDance)
HOP_LENGTH = 512          # ~32ms per frame at 16kHz
N_FFT = 2048
N_MELS = 229              # Covers full piano range with good resolution
PIANO_KEYS = 88           # A0 (MIDI 21) to C8 (MIDI 108)
MIDI_OFFSET = 21          # MIDI number of lowest piano key (A0)

# Segment size for training (seconds → frames)
SEGMENT_SECONDS = 10.0
SEGMENT_FRAMES = int(SEGMENT_SECONDS * SAMPLE_RATE / HOP_LENGTH)

MAESTRO_DIR = Path(__file__).parent / "maestro_midi"
MAESTRO_CSV = MAESTRO_DIR / "maestro-v3.0.0.csv"
PREPARED_DIR = Path(__file__).parent / "transcription_data"
MODEL_PATH = Path(__file__).parent / "piano_transcription.pt"

MAESTRO_AUDIO_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip"


# ─── Mel Spectrogram ────────────────────────────────────────────────────────

def compute_mel_spectrogram(audio: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Compute log-mel spectrogram from audio.

    Returns: (n_frames, N_MELS) float32 array
    """
    try:
        import librosa
    except ImportError:
        raise ImportError("librosa is required: pip install librosa")

    mel = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=30.0, fmax=sr // 2,
    )
    # Log scale with floor to avoid log(0)
    log_mel = np.log(mel + 1e-6).T  # (n_frames, N_MELS)
    return log_mel.astype(np.float32)


# ─── Model Architecture ─────────────────────────────────────────────────────

class ConvStack(nn.Module):
    """CNN encoder for mel spectrogram features."""

    def __init__(self, n_mels: int = N_MELS, channels: List[int] = [32, 64, 128]):
        super().__init__()
        layers = []
        in_ch = 1  # single-channel mel spectrogram
        for out_ch in channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.MaxPool2d(kernel_size=(1, 2)),  # pool only along frequency axis
            ])
            in_ch = out_ch
            n_mels //= 2  # frequency dimension halves each pool

        self.conv = nn.Sequential(*layers)
        self.output_dim = channels[-1] * n_mels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, time, n_mels) → (batch, time, output_dim)
        """
        # Reshape for conv2d: (B, 1, T, F)
        x = x.unsqueeze(1)
        x = self.conv(x)  # (B, C, T, F')
        B, C, T, F = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F)  # (B, T, C*F)
        return x


class PianoTranscriptionModel(nn.Module):
    """
    Onsets-and-Frames style model for piano transcription.

    Architecture:
        Mel spectrogram → ConvStack → Transformer encoder → per-key heads

    Outputs per frame (for each of 88 piano keys):
        - onset: probability of note onset at this frame
        - frame: probability of note active at this frame
        - velocity: estimated velocity (0-1) if note is present
    """

    def __init__(
        self,
        n_mels: int = N_MELS,
        conv_channels: List[int] = [32, 64, 128],
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        n_keys: int = PIANO_KEYS,
    ):
        super().__init__()
        self.n_keys = n_keys
        self.d_model = d_model

        # CNN encoder
        self.conv_stack = ConvStack(n_mels, conv_channels)

        # Project CNN output to model dimension
        self.input_proj = nn.Linear(self.conv_stack.output_dim, d_model)

        # Positional encoding (sinusoidal)
        max_len = 2000  # ~64 seconds at 32ms per frame
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=d_ff, dropout=dropout,
            batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Output heads — per-key predictions
        self.onset_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_keys),
        )
        self.frame_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_keys),
        )
        self.velocity_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_keys),
            nn.Sigmoid(),  # velocity in [0, 1]
        )

    def forward(self, mel: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        mel: (batch, n_frames, n_mels)

        Returns dict of:
            onset_logits: (batch, n_frames, 88)
            frame_logits: (batch, n_frames, 88)
            velocity: (batch, n_frames, 88)  — values in [0, 1]
        """
        h = self.conv_stack(mel)        # (B, T, conv_dim)
        h = self.input_proj(h)          # (B, T, d_model)
        h = h + self.pe[:, :h.size(1)]  # add positional encoding
        h = self.encoder(h)             # (B, T, d_model)

        return {
            'onset_logits': self.onset_head(h),
            'frame_logits': self.frame_head(h),
            'velocity': self.velocity_head(h),
        }


# ─── Velocity-Weighted Loss ─────────────────────────────────────────────────

class VelocityWeightedLoss(nn.Module):
    """
    Loss function that penalizes missing soft notes more than loud notes.

    For onset and frame heads:
      - Standard BCE loss for negative samples (no note)
      - Weighted BCE for positive samples: weight = 1 + alpha * (1 - velocity/127)
        So pianissimo notes (vel~30) get weight ~2.5 while fortissimo (vel~100) gets ~1.4

    This forces the model to be especially sensitive to soft accompaniment
    while still accurately detecting loud melody notes.
    """

    def __init__(self, alpha: float = 2.0, pos_weight: float = 5.0):
        """
        Args:
            alpha: Velocity weighting strength. Higher = more emphasis on soft notes.
            pos_weight: Base positive class weight (notes are sparse, need upweighting).
        """
        super().__init__()
        self.alpha = alpha
        self.pos_weight = pos_weight

    def forward(
        self,
        onset_logits: torch.Tensor,    # (B, T, 88)
        frame_logits: torch.Tensor,    # (B, T, 88)
        velocity_pred: torch.Tensor,   # (B, T, 88)
        onset_labels: torch.Tensor,    # (B, T, 88) binary
        frame_labels: torch.Tensor,    # (B, T, 88) binary
        velocity_labels: torch.Tensor, # (B, T, 88) 0-1 continuous
    ) -> Dict[str, torch.Tensor]:

        # ── Velocity-based sample weights ──
        # Higher weight for softer notes (lower velocity)
        # velocity_labels is 0-1 (0 = no note, fraction = velocity/127)
        # Only weight positive samples (where frame_labels == 1)
        vel_weight = torch.ones_like(velocity_labels)
        active = frame_labels > 0.5
        if active.any():
            # For active notes: weight = 1 + alpha * (1 - velocity)
            # vel~0.24 (pp, vel=30): weight = 1 + 2*(1-0.24) = 2.52
            # vel~0.79 (ff, vel=100): weight = 1 + 2*(1-0.79) = 1.42
            vel_weight[active] = 1.0 + self.alpha * (1.0 - velocity_labels[active])

        # ── Onset loss (velocity-weighted BCE) ──
        # pos_weight compensates for class imbalance (onsets are very sparse)
        onset_bce = F.binary_cross_entropy_with_logits(
            onset_logits, onset_labels, reduction='none'
        )
        # Apply velocity weighting + pos_weight for positive class
        onset_sample_weight = torch.where(
            onset_labels > 0.5,
            vel_weight * self.pos_weight,  # positive: velocity-weighted + upsampled
            torch.ones_like(vel_weight),   # negative: standard weight
        )
        onset_loss = (onset_bce * onset_sample_weight).mean()

        # ── Frame loss (velocity-weighted BCE) ──
        frame_bce = F.binary_cross_entropy_with_logits(
            frame_logits, frame_labels, reduction='none'
        )
        frame_sample_weight = torch.where(
            frame_labels > 0.5,
            vel_weight * self.pos_weight,
            torch.ones_like(vel_weight),
        )
        frame_loss = (frame_bce * frame_sample_weight).mean()

        # ── Velocity loss (MSE, only for active frames) ──
        if active.any():
            velocity_loss = F.mse_loss(velocity_pred[active], velocity_labels[active])
        else:
            velocity_loss = torch.tensor(0.0, device=onset_logits.device)

        # ── Combined loss ──
        total = onset_loss * 1.0 + frame_loss * 1.0 + velocity_loss * 0.5

        return {
            'total': total,
            'onset': onset_loss,
            'frame': frame_loss,
            'velocity': velocity_loss,
        }


# ─── Dataset ────────────────────────────────────────────────────────────────

class PianoTranscriptionDataset(Dataset):
    """
    Dataset of prepared mel spectrogram segments with frame-level labels.

    Each segment is SEGMENT_SECONDS long and stored as a .npz file:
        - mel: (n_frames, N_MELS) float32
        - onset: (n_frames, 88) float32  — 1.0 at onset frames
        - frame: (n_frames, 88) float32  — 1.0 while note is active
        - velocity: (n_frames, 88) float32 — velocity/127 while active, 0 otherwise
    """

    def __init__(self, data_dir: str, split: str = 'train'):
        self.data_dir = Path(data_dir)
        self.split = split

        # Load manifest
        manifest_path = self.data_dir / f"{split}_manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                f"Run: python train_transcription.py --prepare"
            )

        with open(manifest_path) as f:
            self.segments = json.load(f)

        print(f"[Dataset] Loaded {len(self.segments)} {split} segments")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        seg = self.segments[idx]
        data = np.load(seg['path'])

        mel = torch.from_numpy(data['mel']).float()
        onset = torch.from_numpy(data['onset']).float()
        frame = torch.from_numpy(data['frame']).float()
        velocity = torch.from_numpy(data['velocity']).float()

        return mel, onset, frame, velocity


# ─── Data Preparation ───────────────────────────────────────────────────────

def download_maestro_audio():
    """Download full MAESTRO dataset (audio + MIDI, ~120GB)."""
    zip_path = MAESTRO_DIR.parent / "maestro-full.zip"

    # Check if audio already exists
    test_wav = list(MAESTRO_DIR.glob("**/*.wav"))
    if len(test_wav) > 10:
        print(f"MAESTRO audio already present ({len(test_wav)} WAV files)")
        return

    print("=" * 60)
    print("MAESTRO v3.0.0 FULL DATASET DOWNLOAD")
    print("This includes audio files (~120GB). Make sure you have space.")
    print("=" * 60)
    print(f"\nURL: {MAESTRO_AUDIO_URL}")

    MAESTRO_DIR.parent.mkdir(parents=True, exist_ok=True)

    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size) if total_size > 0 else 0
        gb_down = downloaded / 1e9
        gb_total = total_size / 1e9
        print(f"\r  [{percent:3d}%] {gb_down:.1f}/{gb_total:.1f} GB", end="", flush=True)

    urllib.request.urlretrieve(MAESTRO_AUDIO_URL, str(zip_path), show_progress)
    print("\nDownload complete!")

    print("Extracting (this will take a while)...")
    with zipfile.ZipFile(str(zip_path), 'r') as z:
        z.extractall(str(MAESTRO_DIR.parent))

    # Move extracted files if needed
    extracted = MAESTRO_DIR.parent / "maestro-v3.0.0"
    if extracted.exists() and extracted != MAESTRO_DIR:
        # Merge into existing maestro_midi dir
        import shutil
        for item in extracted.iterdir():
            dest = MAESTRO_DIR / item.name
            if item.is_dir():
                if dest.exists():
                    # Merge directory contents
                    for f in item.iterdir():
                        shutil.move(str(f), str(dest / f.name))
                else:
                    shutil.move(str(item), str(dest))
            else:
                shutil.move(str(item), str(dest))
        shutil.rmtree(str(extracted), ignore_errors=True)

    zip_path.unlink(missing_ok=True)
    print(f"Extracted to {MAESTRO_DIR}")


def prepare_training_data():
    """
    Convert MAESTRO audio+MIDI pairs into training segments.

    For each piece:
        1. Load audio → compute mel spectrogram
        2. Load MIDI → create frame-level onset/frame/velocity labels
        3. Split into fixed-length segments
        4. Save as .npz files
    """
    try:
        import librosa
        import pretty_midi
    except ImportError:
        print("Required: pip install librosa pretty_midi")
        return

    if not MAESTRO_CSV.exists():
        print(f"MAESTRO CSV not found at {MAESTRO_CSV}")
        print("Run: python prepare_training_data.py --download")
        return

    # Read CSV
    pieces = []
    with open(MAESTRO_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pieces.append(row)

    print(f"Found {len(pieces)} pieces in MAESTRO CSV")

    # Check for audio files
    n_audio = sum(1 for p in pieces if (MAESTRO_DIR / p['audio_filename']).exists())
    if n_audio == 0:
        print("\nNo audio files found! You need to download the full MAESTRO dataset.")
        print("Run: python train_transcription.py --download-audio")
        print("Or download manually from:")
        print(f"  {MAESTRO_AUDIO_URL}")
        return

    print(f"Found audio for {n_audio}/{len(pieces)} pieces")

    # Prepare output directory
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    # Process by split
    manifests = {'train': [], 'validation': [], 'test': []}
    total_segments = 0
    errors = 0

    for idx, piece in enumerate(pieces):
        split = piece['split']
        audio_path = MAESTRO_DIR / piece['audio_filename']
        midi_path = MAESTRO_DIR / piece['midi_filename']

        if not audio_path.exists():
            continue
        if not midi_path.exists():
            continue

        if idx % 20 == 0:
            print(f"Processing {idx}/{len(pieces)} ({total_segments} segments so far)...")

        try:
            # Load audio
            audio, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)

            # Compute mel spectrogram
            mel = compute_mel_spectrogram(audio, sr=SAMPLE_RATE)
            n_frames = mel.shape[0]

            # Load MIDI and create frame-level labels
            midi = pretty_midi.PrettyMIDI(str(midi_path))

            onset_labels = np.zeros((n_frames, PIANO_KEYS), dtype=np.float32)
            frame_labels = np.zeros((n_frames, PIANO_KEYS), dtype=np.float32)
            velocity_labels = np.zeros((n_frames, PIANO_KEYS), dtype=np.float32)

            for instrument in midi.instruments:
                if instrument.is_drum:
                    continue
                for note in instrument.notes:
                    key = note.pitch - MIDI_OFFSET
                    if key < 0 or key >= PIANO_KEYS:
                        continue

                    # Convert times to frames
                    onset_frame = int(note.start * SAMPLE_RATE / HOP_LENGTH)
                    offset_frame = int(note.end * SAMPLE_RATE / HOP_LENGTH)

                    if onset_frame >= n_frames:
                        continue

                    offset_frame = min(offset_frame, n_frames)
                    vel_normalized = note.velocity / 127.0

                    # Onset: mark onset frame (and +/- 1 frame for tolerance)
                    for f in range(max(0, onset_frame), min(onset_frame + 2, n_frames)):
                        onset_labels[f, key] = 1.0

                    # Frame: mark all active frames
                    for f in range(onset_frame, offset_frame):
                        frame_labels[f, key] = 1.0
                        velocity_labels[f, key] = vel_normalized

            # Split into segments
            split_dir = PREPARED_DIR / split
            split_dir.mkdir(parents=True, exist_ok=True)

            for seg_start in range(0, n_frames - SEGMENT_FRAMES // 2, SEGMENT_FRAMES):
                seg_end = min(seg_start + SEGMENT_FRAMES, n_frames)

                seg_mel = mel[seg_start:seg_end]
                seg_onset = onset_labels[seg_start:seg_end]
                seg_frame = frame_labels[seg_start:seg_end]
                seg_vel = velocity_labels[seg_start:seg_end]

                # Pad if shorter than SEGMENT_FRAMES
                if seg_mel.shape[0] < SEGMENT_FRAMES:
                    pad_len = SEGMENT_FRAMES - seg_mel.shape[0]
                    seg_mel = np.pad(seg_mel, ((0, pad_len), (0, 0)))
                    seg_onset = np.pad(seg_onset, ((0, pad_len), (0, 0)))
                    seg_frame = np.pad(seg_frame, ((0, pad_len), (0, 0)))
                    seg_vel = np.pad(seg_vel, ((0, pad_len), (0, 0)))

                # Save
                seg_name = f"seg_{idx:04d}_{seg_start:06d}.npz"
                seg_path = str(split_dir / seg_name)
                np.savez_compressed(
                    seg_path,
                    mel=seg_mel, onset=seg_onset,
                    frame=seg_frame, velocity=seg_vel,
                )

                manifests[split].append({
                    'path': seg_path,
                    'piece_idx': idx,
                    'start_frame': seg_start,
                    'composer': piece['canonical_composer'],
                })
                total_segments += 1

        except Exception as e:
            errors += 1
            if errors < 10:
                print(f"  Error processing {audio_path.name}: {e}")

    # Save manifests
    for split_name, segments in manifests.items():
        manifest_path = PREPARED_DIR / f"{split_name}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(segments, f)
        print(f"  {split_name}: {len(segments)} segments")

    print(f"\nData preparation complete!")
    print(f"  Total segments: {total_segments}")
    print(f"  Errors: {errors}")
    print(f"  Saved to: {PREPARED_DIR}")


# ─── Training ───────────────────────────────────────────────────────────────

def train(args):
    """Main training loop."""
    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    print(f"Using device: {device}")

    # Dataset
    train_dataset = PianoTranscriptionDataset(str(PREPARED_DIR), split='train')
    val_dataset = PianoTranscriptionDataset(str(PREPARED_DIR), split='validation')

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Model
    model = PianoTranscriptionModel(
        n_mels=N_MELS,
        conv_channels=[32, 64, 128],
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = VelocityWeightedLoss(alpha=args.vel_alpha, pos_weight=args.pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # ── Train ──
        model.train()
        train_losses = {'total': 0, 'onset': 0, 'frame': 0, 'velocity': 0}
        n_batches = 0

        for batch_idx, (mel, onset_gt, frame_gt, vel_gt) in enumerate(train_loader):
            mel = mel.to(device)
            onset_gt = onset_gt.to(device)
            frame_gt = frame_gt.to(device)
            vel_gt = vel_gt.to(device)

            optimizer.zero_grad()
            out = model(mel)

            losses = criterion(
                out['onset_logits'], out['frame_logits'], out['velocity'],
                onset_gt, frame_gt, vel_gt,
            )

            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k in train_losses:
                train_losses[k] += losses[k].item()
            n_batches += 1

            if batch_idx % 100 == 0 and batch_idx > 0:
                avg = train_losses['total'] / n_batches
                print(f"  Epoch {epoch+1} batch {batch_idx}: loss={avg:.4f}")

        scheduler.step()

        # ── Validate ──
        model.eval()
        val_losses = {'total': 0, 'onset': 0, 'frame': 0, 'velocity': 0}
        n_val = 0

        # Track onset/frame metrics
        onset_tp, onset_fp, onset_fn = 0, 0, 0
        frame_tp, frame_fp, frame_fn = 0, 0, 0

        with torch.no_grad():
            for mel, onset_gt, frame_gt, vel_gt in val_loader:
                mel = mel.to(device)
                onset_gt = onset_gt.to(device)
                frame_gt = frame_gt.to(device)
                vel_gt = vel_gt.to(device)

                out = model(mel)
                losses = criterion(
                    out['onset_logits'], out['frame_logits'], out['velocity'],
                    onset_gt, frame_gt, vel_gt,
                )

                for k in val_losses:
                    val_losses[k] += losses[k].item()
                n_val += 1

                # Compute precision/recall for onset and frame
                onset_pred = (torch.sigmoid(out['onset_logits']) > 0.5).float()
                frame_pred = (torch.sigmoid(out['frame_logits']) > 0.5).float()

                onset_tp += ((onset_pred == 1) & (onset_gt == 1)).sum().item()
                onset_fp += ((onset_pred == 1) & (onset_gt == 0)).sum().item()
                onset_fn += ((onset_pred == 0) & (onset_gt == 1)).sum().item()

                frame_tp += ((frame_pred == 1) & (frame_gt == 1)).sum().item()
                frame_fp += ((frame_pred == 1) & (frame_gt == 0)).sum().item()
                frame_fn += ((frame_pred == 0) & (frame_gt == 1)).sum().item()

        # Compute F1 scores
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': {
                    'n_mels': N_MELS,
                    'conv_channels': [32, 64, 128],
                    'd_model': args.d_model,
                    'n_heads': args.n_heads,
                    'n_layers': args.n_layers,
                    'd_ff': args.d_ff,
                    'n_keys': PIANO_KEYS,
                    'sample_rate': SAMPLE_RATE,
                    'hop_length': HOP_LENGTH,
                    'n_fft': N_FFT,
                },
                'epoch': epoch,
                'val_loss': best_val_loss,
                'onset_f1': onset_f1,
                'frame_f1': frame_f1,
            }, str(MODEL_PATH))
            print(f"  Saved best model! (val_loss={best_val_loss:.4f})")

    print(f"\nTraining complete! Best val loss: {best_val_loss:.4f}")
    print(f"Model saved to: {MODEL_PATH}")


# ─── Inference ───────────────────────────────────────────────────────────────

def transcribe_audio(
    audio: np.ndarray,
    model: PianoTranscriptionModel,
    device: torch.device,
    sr: int = SAMPLE_RATE,
    onset_threshold: float = 0.4,
    frame_threshold: float = 0.3,
    min_note_duration: float = 0.05,
) -> List[Dict]:
    """
    Transcribe audio using the trained model.

    Args:
        audio: mono audio at SAMPLE_RATE
        model: trained PianoTranscriptionModel
        device: torch device
        onset_threshold: lower = more sensitive to soft onsets
        frame_threshold: lower = more sensitive to soft sustained notes
        min_note_duration: minimum note duration in seconds

    Returns:
        List of note event dicts matching ByteDance format:
        [{'onset_time': float, 'offset_time': float, 'midi_note': int, 'velocity': int}, ...]
    """
    model.eval()

    # Compute mel spectrogram
    mel = compute_mel_spectrogram(audio, sr=sr)
    n_frames = mel.shape[0]

    # Process in overlapping chunks to avoid memory issues
    chunk_frames = SEGMENT_FRAMES
    overlap = chunk_frames // 4
    step = chunk_frames - overlap

    all_onset_probs = np.zeros((n_frames, PIANO_KEYS), dtype=np.float32)
    all_frame_probs = np.zeros((n_frames, PIANO_KEYS), dtype=np.float32)
    all_velocity = np.zeros((n_frames, PIANO_KEYS), dtype=np.float32)
    counts = np.zeros(n_frames, dtype=np.float32)

    with torch.no_grad():
        for start in range(0, n_frames, step):
            end = min(start + chunk_frames, n_frames)
            chunk_mel = mel[start:end]

            # Pad if needed
            if chunk_mel.shape[0] < chunk_frames:
                pad_len = chunk_frames - chunk_mel.shape[0]
                chunk_mel = np.pad(chunk_mel, ((0, pad_len), (0, 0)))

            # Inference
            x = torch.from_numpy(chunk_mel).float().unsqueeze(0).to(device)
            out = model(x)

            onset_p = torch.sigmoid(out['onset_logits'][0]).cpu().numpy()
            frame_p = torch.sigmoid(out['frame_logits'][0]).cpu().numpy()
            vel = out['velocity'][0].cpu().numpy()

            # Accumulate (handle overlap by averaging)
            actual_len = end - start
            all_onset_probs[start:end] += onset_p[:actual_len]
            all_frame_probs[start:end] += frame_p[:actual_len]
            all_velocity[start:end] += vel[:actual_len]
            counts[start:end] += 1.0

    # Average overlapping regions
    counts = np.maximum(counts, 1.0)
    all_onset_probs /= counts[:, None]
    all_frame_probs /= counts[:, None]
    all_velocity /= counts[:, None]

    # ── Decode note events from frame-level predictions ──
    frame_time = HOP_LENGTH / sr  # seconds per frame
    min_frames = int(min_note_duration / frame_time)

    note_events = []

    for key in range(PIANO_KEYS):
        onset_mask = all_onset_probs[:, key] > onset_threshold
        frame_mask = all_frame_probs[:, key] > frame_threshold

        # Find onset positions
        onset_frames = np.where(onset_mask)[0]

        for onset_f in onset_frames:
            # Find offset: first frame after onset where frame drops below threshold
            offset_f = onset_f + 1
            while offset_f < n_frames and frame_mask[offset_f]:
                offset_f += 1

            # Enforce minimum duration
            if offset_f - onset_f < min_frames:
                offset_f = min(onset_f + min_frames, n_frames)

            # Get velocity (average over active frames)
            vel_avg = all_velocity[onset_f:offset_f, key].mean()
            velocity = int(np.clip(vel_avg * 127, 1, 127))

            note_events.append({
                'onset_time': onset_f * frame_time,
                'offset_time': offset_f * frame_time,
                'midi_note': key + MIDI_OFFSET,
                'velocity': velocity,
            })

    # Sort by onset time
    note_events.sort(key=lambda e: (e['onset_time'], e['midi_note']))

    # Remove duplicate detections (same note within 50ms)
    filtered = []
    for event in note_events:
        is_dup = False
        for prev in filtered[-10:]:  # only check recent
            if (abs(event['onset_time'] - prev['onset_time']) < 0.05 and
                    event['midi_note'] == prev['midi_note']):
                is_dup = True
                break
        if not is_dup:
            filtered.append(event)

    return filtered


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Train custom piano transcription model for LiveScore')

    # Actions
    parser.add_argument('--download-audio', action='store_true',
                        help='Download full MAESTRO dataset with audio (~120GB)')
    parser.add_argument('--prepare', action='store_true',
                        help='Prepare training data (mel spectrograms + labels)')
    parser.add_argument('--train', action='store_true',
                        help='Train the model')
    parser.add_argument('--export', action='store_true',
                        help='Export model for inference')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d-model', type=int, default=256)
    parser.add_argument('--n-heads', type=int, default=8)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--d-ff', type=int, default=1024)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--vel-alpha', type=float, default=2.0,
                        help='Velocity weighting strength (higher = more emphasis on soft notes)')
    parser.add_argument('--pos-weight', type=float, default=5.0,
                        help='Positive class weight for onset/frame BCE')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--num-workers', type=int, default=4)

    args = parser.parse_args()

    if args.download_audio:
        download_maestro_audio()

    if args.prepare:
        prepare_training_data()

    if args.train:
        train(args)

    if args.export:
        print(f"Model checkpoint at: {MODEL_PATH}")
        print("The GPU inference class in gpu_ops.py loads this directly.")

    if not any([args.download_audio, args.prepare, args.train, args.export]):
        parser.print_help()
        print("\n\nQuick start:")
        print("  1. Download MAESTRO audio:  python train_transcription.py --download-audio")
        print("  2. Prepare training data:   python train_transcription.py --prepare")
        print("  3. Train:                   python train_transcription.py --train --epochs 50")


if __name__ == '__main__':
    main()
