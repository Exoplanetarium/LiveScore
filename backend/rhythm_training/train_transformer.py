"""
Transformer-based sequence model for rhythm quantization + rest prediction.

Unlike the MLP which treats each note independently, this model sees the full
sequence of notes and can learn:
  - Phrase structure (where rests naturally belong)
  - Rhythmic patterns and consistency within passages
  - Metric context (how note values relate to beat/measure position)
  - Phrasing gestures (held notes followed by rests at phrase boundaries)

Architecture:
  - Input: per-note features (10-dim) including rest ground truth
  - Positional encoding: learned (sequence position) + sinusoidal (beat position)
  - Encoder: 4 Transformer layers, 64-dim, 4 heads, bidirectional attention
  - Output heads: note_type (6), dotted (2), triplet (2), rest (2)

Training data:
  - MAESTRO v3.0.0 MIDI files (professional piano with aligned MIDI)
  - Rest labels derived from MIDI: gap between note offset and next onset
  - Sequences grouped by piece (not shuffled individual notes)

Usage:
    python train_transformer.py --data rhythm_seq_data.jsonl --epochs 10
    python train_transformer.py --prepare   # Generate sequence training data first
"""

import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ─── Constants ───────────────────────────────────────────────────────────────

NOTE_TYPES = ['whole', 'half', 'quarter', 'eighth', '16th', '32nd']
NOTE_TYPE_TO_IDX = {nt: i for i, nt in enumerate(NOTE_TYPES)}
IDX_TO_NOTE_TYPE = {i: nt for i, nt in enumerate(NOTE_TYPES)}
NOTE_TYPE_BEATS = {
    'whole': 4.0, 'half': 2.0, 'quarter': 1.0,
    'eighth': 0.5, '16th': 0.25, '32nd': 0.125
}

SEQ_DATA_PATH = Path(__file__).parent / "rhythm_seq_data.jsonl"
MODEL_PATH = Path(__file__).parent / "rhythm_transformer.pt"
NPZ_PATH = Path(__file__).parent / "rhythm_transformer.npz"
MAESTRO_DIR = Path(__file__).parent / "maestro_midi"

INPUT_DIM = 10   # 8 original features + rest_gap_beats + next_ioi_beats
MAX_SEQ_LEN = 512


# ─── Positional Encoding ────────────────────────────────────────────────────

class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = MAX_SEQ_LEN):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, seq_len, d_model)"""
        return x + self.pe[:, :x.size(1), :]


# ─── Transformer Model ──────────────────────────────────────────────────────

class RhythmTransformer(nn.Module):
    """
    Sequence-to-sequence Transformer for joint rhythm + rest prediction.

    Each note in the sequence produces 4 predictions:
      1. note_type (6 classes)
      2. dotted (binary)
      3. triplet (binary)
      4. has_rest_after (binary)  <-- NEW: learned rest placement
    """

    def __init__(
        self,
        input_dim: int = INPUT_DIM,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = MAX_SEQ_LEN,
    ):
        super().__init__()
        self.d_model = d_model

        # Project input features to model dimension
        self.input_proj = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout)

        # Transformer encoder (bidirectional — we have the full sequence)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Task-specific output heads
        self.head_type = nn.Linear(d_model, 6)      # note type
        self.head_dotted = nn.Linear(d_model, 2)     # dotted
        self.head_triplet = nn.Linear(d_model, 2)    # triplet
        self.head_rest = nn.Linear(d_model, 2)       # rest after note

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, seq_len, input_dim) note features
            padding_mask: (batch, seq_len) True where padded

        Returns:
            Dict of logits, each (batch, seq_len, n_classes)
        """
        h = self.input_proj(x)                 # (B, S, d_model)
        h = self.pos_enc(h)                    # add positional info
        h = self.dropout(h)

        # Transformer encoder (bidirectional self-attention)
        h = self.encoder(h, src_key_padding_mask=padding_mask)  # (B, S, d_model)

        return {
            'type': self.head_type(h),
            'dotted': self.head_dotted(h),
            'triplet': self.head_triplet(h),
            'rest': self.head_rest(h),
        }


# ─── Dataset ────────────────────────────────────────────────────────────────

class RhythmSeqDataset(Dataset):
    """
    Dataset of sequences (one per piece / segment).

    Each item is a dict with:
      - 'features': (seq_len, 10) float32
      - 'labels_type': (seq_len,) int64
      - 'labels_dotted': (seq_len,) int64
      - 'labels_triplet': (seq_len,) int64
      - 'labels_rest': (seq_len,) int64
    """

    def __init__(self, path: str, max_seq_len: int = MAX_SEQ_LEN):
        self.sequences = []
        self.max_seq_len = max_seq_len

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    seq = json.loads(line.strip())
                    if len(seq['features']) < 4:
                        continue
                    self.sequences.append(seq)
                except json.JSONDecodeError:
                    continue

        print(f"[SeqDataset] Loaded {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        feats = np.array(seq['features'], dtype=np.float32)[:self.max_seq_len]
        labels = seq['labels']
        n = len(feats)

        return {
            'features': torch.from_numpy(feats),
            'labels_type': torch.tensor(labels['type'][:n], dtype=torch.long),
            'labels_dotted': torch.tensor(labels['dotted'][:n], dtype=torch.long),
            'labels_triplet': torch.tensor(labels['triplet'][:n], dtype=torch.long),
            'labels_rest': torch.tensor(labels['rest'][:n], dtype=torch.long),
            'length': n,
        }


def collate_sequences(batch):
    """Pad sequences to same length within batch."""
    max_len = max(item['length'] for item in batch)
    B = len(batch)
    input_dim = batch[0]['features'].shape[1]

    features = torch.zeros(B, max_len, input_dim)
    labels_type = torch.zeros(B, max_len, dtype=torch.long)
    labels_dotted = torch.zeros(B, max_len, dtype=torch.long)
    labels_triplet = torch.zeros(B, max_len, dtype=torch.long)
    labels_rest = torch.zeros(B, max_len, dtype=torch.long)
    padding_mask = torch.ones(B, max_len, dtype=torch.bool)  # True = padded

    for i, item in enumerate(batch):
        n = item['length']
        features[i, :n] = item['features']
        labels_type[i, :n] = item['labels_type']
        labels_dotted[i, :n] = item['labels_dotted']
        labels_triplet[i, :n] = item['labels_triplet']
        labels_rest[i, :n] = item['labels_rest']
        padding_mask[i, :n] = False

    return {
        'features': features,
        'labels_type': labels_type,
        'labels_dotted': labels_dotted,
        'labels_triplet': labels_triplet,
        'labels_rest': labels_rest,
        'padding_mask': padding_mask,
    }


# ─── Data Preparation ───────────────────────────────────────────────────────

def prepare_sequence_data():
    """
    Process MAESTRO MIDI files into sequence training data with rest labels.

    For each piece:
      1. Load MIDI notes (onset, offset, pitch)
      2. Compute IOIs, durations, beat/measure positions
      3. Derive ground truth rest labels from gaps:
         - gap = next_onset - current_offset
         - has_rest = 1 if gap >= 0.5 * beat_duration else 0
      4. Save as a sequence (not individual notes)
    """
    sys.path.insert(0, str(Path(__file__).parent))
    from evaluate_rhythm import load_midi_ground_truth, quantize_ground_truth

    midi_files = (
        list(MAESTRO_DIR.glob("**/*.midi")) +
        list(MAESTRO_DIR.glob("**/*.mid"))
    )
    print(f"Found {len(midi_files)} MIDI files")

    if len(midi_files) == 0:
        print("No MIDI files found. Run prepare_training_data.py --download first.")
        return

    note_type_to_idx = {nt: i for i, nt in enumerate(NOTE_TYPES)}
    sequences = []
    errors = 0
    total_notes = 0

    for file_idx, midi_path in enumerate(midi_files):
        if file_idx % 50 == 0:
            print(f"Processing {file_idx}/{len(midi_files)} "
                  f"({total_notes} notes so far)...")
        try:
            notes, tempo_changes = load_midi_ground_truth(str(midi_path))
            if len(notes) < 10:
                continue

            # Get tempo
            bpm = tempo_changes[1][0] if len(tempo_changes[1]) > 0 else 120.0
            beat_duration = 60.0 / bpm
            measure_duration = beat_duration * 4

            # ── Build per-note features and labels ──
            features = []
            labels_type = []
            labels_dotted = []
            labels_triplet = []
            labels_rest = []

            for i, note in enumerate(notes):
                onset = note['onset']
                offset = note['offset']
                duration = note['duration']
                pitch = note['pitch']

                # IOI to next note
                if i < len(notes) - 1:
                    next_onset = notes[i + 1]['onset']
                    ioi = next_onset - onset
                else:
                    ioi = duration

                # Previous IOI
                if i > 0:
                    prev_ioi = notes[i]['onset'] - notes[i - 1]['onset']
                else:
                    prev_ioi = ioi

                # Next IOI (look-ahead feature — Transformer is bidirectional)
                if i < len(notes) - 2:
                    next_ioi = notes[i + 2]['onset'] - notes[i + 1]['onset']
                else:
                    next_ioi = ioi

                # Rest gap: time between this note's offset and next onset
                if i < len(notes) - 1:
                    gap = notes[i + 1]['onset'] - offset
                else:
                    gap = 0.0

                # ── 10 input features ──
                dur_beats = duration / beat_duration
                ioi_beats = ioi / beat_duration
                beat_pos = (onset % beat_duration) / beat_duration
                measure_pos = (onset % measure_duration) / measure_duration
                prev_ioi_beats = prev_ioi / beat_duration
                dur_ioi_ratio = duration / max(ioi, 0.01)
                norm_pitch = (pitch - 60) / 40.0
                norm_tempo = bpm / 120.0
                rest_gap_beats = max(gap, 0) / beat_duration
                next_ioi_beats = next_ioi / beat_duration

                features.append([
                    dur_beats,         # 0
                    ioi_beats,         # 1
                    beat_pos,          # 2
                    measure_pos,       # 3
                    prev_ioi_beats,    # 4
                    dur_ioi_ratio,     # 5
                    norm_pitch,        # 6
                    norm_tempo,        # 7
                    rest_gap_beats,    # 8  (NEW: observed rest gap)
                    next_ioi_beats,    # 9  (NEW: forward context)
                ])

                # ── Ground truth note type ──
                note_values = [
                    ('whole', 4.0, False, False),
                    ('whole', 6.0, True, False),
                    ('half', 2.0, False, False),
                    ('half', 3.0, True, False),
                    ('quarter', 1.0, False, False),
                    ('quarter', 1.5, True, False),
                    ('eighth', 0.5, False, False),
                    ('eighth', 0.75, True, False),
                    ('16th', 0.25, False, False),
                    ('16th', 0.375, True, False),
                    ('32nd', 0.125, False, False),
                    ('quarter', 2/3, False, True),
                    ('eighth', 1/3, False, True),
                    ('16th', 1/6, False, True),
                ]
                best_type, best_dotted, best_triplet = 'quarter', False, False
                best_dist = float('inf')
                for nv in note_values:
                    nt, beats, dotted, is_triplet = nv
                    if ioi_beats > 0.01:
                        dist = abs(np.log2(ioi_beats / beats))
                    else:
                        dist = abs(ioi_beats - beats)
                    if dist < best_dist:
                        best_dist = dist
                        best_type = nt
                        best_dotted = dotted
                        best_triplet = is_triplet

                labels_type.append(note_type_to_idx.get(best_type, 2))
                labels_dotted.append(1 if best_dotted else 0)
                labels_triplet.append(1 if best_triplet else 0)

                # ── Rest label: is there a meaningful gap before next note? ──
                # Ground truth from MIDI: gap >= 0.5 beats is a rest
                has_rest = 1 if rest_gap_beats >= 0.5 else 0
                labels_rest.append(has_rest)

            # Split long pieces into chunks of MAX_SEQ_LEN
            n = len(features)
            for start in range(0, n, MAX_SEQ_LEN):
                end = min(start + MAX_SEQ_LEN, n)
                if end - start < 4:
                    continue
                sequences.append({
                    'features': features[start:end],
                    'labels': {
                        'type': labels_type[start:end],
                        'dotted': labels_dotted[start:end],
                        'triplet': labels_triplet[start:end],
                        'rest': labels_rest[start:end],
                    }
                })
                total_notes += end - start

        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"  Error: {midi_path.name}: {e}")

    print(f"\nProcessed {len(midi_files)} files ({errors} errors)")
    print(f"Generated {len(sequences)} sequences, {total_notes} total notes")

    # Rest distribution
    all_rest = []
    for s in sequences:
        all_rest.extend(s['labels']['rest'])
    rest_pos = sum(all_rest)
    print(f"Rest distribution: {rest_pos} rests ({rest_pos/max(len(all_rest),1)*100:.1f}%) "
          f"/ {len(all_rest)-rest_pos} continuations ({(len(all_rest)-rest_pos)/max(len(all_rest),1)*100:.1f}%)")

    # Save
    print(f"Saving to {SEQ_DATA_PATH}...")
    with open(SEQ_DATA_PATH, 'w', encoding='utf-8') as f:
        for seq in sequences:
            f.write(json.dumps(seq) + '\n')

    print("Done!")


# ─── Training ───────────────────────────────────────────────────────────────

def train_epoch(model, dataloader, optimizer, criterion, device, rest_weight=2.0):
    """Train for one epoch with multi-task loss."""
    model.train()

    total_loss = 0
    correct = {'type': 0, 'dotted': 0, 'triplet': 0, 'rest': 0}
    total = 0

    for batch_idx, batch in enumerate(dataloader):
        features = batch['features'].to(device)
        padding_mask = batch['padding_mask'].to(device)
        labels = {
            k: batch[f'labels_{k}'].to(device)
            for k in ['type', 'dotted', 'triplet', 'rest']
        }

        optimizer.zero_grad()
        out = model(features, padding_mask)

        # Compute loss only on non-padded positions
        active = ~padding_mask  # (B, S), True = real note
        active_flat = active.reshape(-1)  # (B*S,)

        loss = torch.tensor(0.0, device=device)
        for key in ['type', 'dotted', 'triplet', 'rest']:
            logits = out[key].reshape(-1, out[key].size(-1))  # (B*S, C)
            target = labels[key].reshape(-1)                   # (B*S,)

            # Mask to active positions
            logits_a = logits[active_flat]
            target_a = target[active_flat]

            if logits_a.numel() == 0:
                continue

            w = rest_weight if key == 'rest' else (2.0 if key == 'type' else 1.0)
            loss += w * criterion(logits_a, target_a)

            # Accuracy
            preds = logits_a.argmax(dim=-1)
            correct[key] += (preds == target_a).sum().item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        n_active = active_flat.sum().item()
        total_loss += loss.item() * n_active
        total += n_active

        if batch_idx % 200 == 0 and batch_idx > 0:
            avg_loss = total_loss / total
            type_acc = correct['type'] / total
            rest_acc = correct['rest'] / total
            print(f"  Batch {batch_idx}: loss={avg_loss:.4f}, "
                  f"type_acc={type_acc:.3f}, rest_acc={rest_acc:.3f}")

    return {
        'loss': total_loss / max(total, 1),
        'type_acc': correct['type'] / max(total, 1),
        'dotted_acc': correct['dotted'] / max(total, 1),
        'triplet_acc': correct['triplet'] / max(total, 1),
        'rest_acc': correct['rest'] / max(total, 1),
    }


def export_transformer(model, save_path):
    """Export Transformer weights so inference can load them."""
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': {
            'input_dim': INPUT_DIM,
            'd_model': model.d_model,
            'n_heads': model.encoder.layers[0].self_attn.num_heads,
            'n_layers': len(model.encoder.layers),
            'd_ff': model.encoder.layers[0].linear1.out_features,
        },
    }, save_path)
    print(f"Exported model to {save_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Train Transformer rhythm model with rest prediction')
    parser.add_argument('--prepare', action='store_true',
                        help='Prepare sequence training data from MAESTRO')
    parser.add_argument('--data', default=str(SEQ_DATA_PATH),
                        help='Path to sequence training data JSONL')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--d-ff', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--rest-weight', type=float, default=2.0,
                        help='Loss weight for rest prediction head')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', default=str(MODEL_PATH))
    args = parser.parse_args()

    if args.prepare:
        prepare_sequence_data()
        return

    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Dataset
    print(f"Loading data from {args.data}...")
    dataset = RhythmSeqDataset(args.data, max_seq_len=MAX_SEQ_LEN)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_sequences, num_workers=0,
    )

    # Model
    model = RhythmTransformer(
        input_dim=INPUT_DIM,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()

    # Training
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        metrics = train_epoch(
            model, dataloader, optimizer, criterion, device,
            rest_weight=args.rest_weight,
        )
        scheduler.step()

        print(f"Epoch {epoch + 1} results:")
        print(f"  Loss:         {metrics['loss']:.4f}")
        print(f"  Type Acc:     {metrics['type_acc']:.3%}")
        print(f"  Dotted Acc:   {metrics['dotted_acc']:.3%}")
        print(f"  Triplet Acc:  {metrics['triplet_acc']:.3%}")
        print(f"  Rest Acc:     {metrics['rest_acc']:.3%}")

        combined = (metrics['type_acc'] + metrics['rest_acc']) / 2
        if combined > best_acc:
            best_acc = combined
            export_transformer(model, args.output)
            print(f"  New best model saved! (combined={combined:.3%})")

    print(f"\nTraining complete! Best combined accuracy: {best_acc:.3%}")
    print(f"Model saved to: {args.output}")


if __name__ == '__main__':
    main()
