"""
Mini hyperparameter experiment for LiveScore ensemble transcriber.

Tests a pitch-aware architecture that exploits the structure of the 1098
input features instead of treating them as a flat vector.

The 1098 features per frame are structured:
  Per key (88 keys × 6 spectral views × 2 hops = 12 per key):
    mel_1024[k], mel_2048[k], mel_4096[k], CQT[k], HPSS_h[k], HPSS_p[k]
  Global (chroma 12 + onset 9) × 2 hops = 42 features

Usage:
    python mini_experiment.py
    python mini_experiment.py --max-note-density 0.1
    python mini_experiment.py --epochs 5 --configs base focal2
"""

import math
import os
import sys
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from rhythm_training.train_ensemble import (FEATURES_DIR, N_FEATURES,
                                            N_FEATURES_MULTI_HOP,
                                            NOTE_VALUE_CLASSES, PIANO_KEYS,
                                            EnsembleLoss, EnsembleMetaLearner,
                                            PitchAwareTranscriber,
                                            PrecomputedDataset)

# ─── Configs ─────────────────────────────────────────────────────────────────

CONFIGS = [
    # Baseline Conformer (memory efficient): ~100K params
    {
        "name": "base",
        "key_hidden": 32,
        "temporal_hidden": 32,
        "temporal_layers": 4,
        "n_key_conv_layers": 2,
        "n_heads": 1,
        "ff_expansion": 4,
        "conv_kernel": 31,
        "lr": 1.0,
        "warmup_steps": 500,
        "dropout": 0.1,
        "focal_gamma": 0.0,
        "pos_weight": 5.0,
        "weight_decay": 0.01,
        "use_checkpoint": False,
    },
    # Wider temporal dimension
    {
        "name": "wide_temporal",
        "key_hidden": 32,
        "temporal_hidden": 48,
        "temporal_layers": 4,
        "n_key_conv_layers": 2,
        "n_heads": 1,
        "ff_expansion": 4,
        "conv_kernel": 31,
        "lr": 1.0,
        "warmup_steps": 500,
        "dropout": 0.1,
        "focal_gamma": 0.0,
        "pos_weight": 5.0,
        "weight_decay": 0.01,
        "use_checkpoint": False,
    },
    # More attention heads
    {
        "name": "multi_head",
        "key_hidden": 32,
        "temporal_hidden": 32,
        "temporal_layers": 4,
        "n_key_conv_layers": 2,
        "n_heads": 2,
        "ff_expansion": 4,
        "conv_kernel": 31,
        "lr": 1.0,
        "warmup_steps": 500,
        "dropout": 0.1,
        "focal_gamma": 0.0,
        "pos_weight": 5.0,
        "weight_decay": 0.01,
        "use_checkpoint": True,  # Use checkpointing with multi-head
    },
    # Focal loss variant
    {
        "name": "focal2",
        "key_hidden": 32,
        "temporal_hidden": 32,
        "temporal_layers": 4,
        "n_key_conv_layers": 2,
        "n_heads": 1,
        "ff_expansion": 4,
        "conv_kernel": 31,
        "lr": 1.0,
        "warmup_steps": 500,
        "dropout": 0.1,
        "focal_gamma": 2.0,
        "pos_weight": 5.0,
        "weight_decay": 0.01,
        "use_checkpoint": False,
    },
    # Wider key features
    {
        "name": "wide_key",
        "key_hidden": 48,
        "temporal_hidden": 32,
        "temporal_layers": 4,
        "n_key_conv_layers": 2,
        "n_heads": 1,
        "ff_expansion": 4,
        "conv_kernel": 31,
        "lr": 1.0,
        "warmup_steps": 500,
        "dropout": 0.1,
        "focal_gamma": 0.0,
        "pos_weight": 5.0,
        "weight_decay": 0.01,
        "use_checkpoint": False,
    },
]


# ─── Subset Dataset ──────────────────────────────────────────────────────────

class SubsetPrecomputedDataset(PrecomputedDataset):
    """PrecomputedDataset limited to first N segments, optionally filtered by note density."""

    def __init__(self, split='train', max_segments=300, max_note_density=None,
                 augment=False, mixup_alpha=0.0):
        super().__init__(split=split, augment=augment, mixup_alpha=mixup_alpha)

        if max_note_density is not None:
            filtered = []
            scan_limit = min(len(self.files), max_segments * 3)
            print(f"  Filtering for note density <= {max_note_density} "
                  f"(scanning up to {scan_limit} segments)...")
            for f in self.files[:scan_limit]:
                data = torch.load(f, weights_only=True)
                T = data['onset'].shape[0]
                density = data['onset'].sum().item() / max(T, 1)
                if density <= max_note_density:
                    filtered.append(f)
                if len(filtered) >= max_segments:
                    break
            self.files = filtered
        else:
            self.files = self.files[:max_segments]

        print(f"  [Subset] Using {len(self.files)} segments from {split}")


# ─── Training ────────────────────────────────────────────────────────────────

def build_model(config, device):
    """Build model from config dict."""
    if config.get('model') == 'EnsembleMetaLearner':
        return EnsembleMetaLearner(
            n_features=N_FEATURES_MULTI_HOP,
            conv_channels=config['conv_channels'],
            gru_hidden=config['gru_hidden'],
            gru_layers=config['gru_layers'],
            dropout=config['dropout'],
        ).to(device)

    return PitchAwareTranscriber(
        key_hidden=config['key_hidden'],
        temporal_hidden=config['temporal_hidden'],
        temporal_layers=config['temporal_layers'],
        n_key_conv_layers=config['n_key_conv_layers'],
        dropout=config['dropout'],
        n_heads=config.get('n_heads', 2),
        ff_expansion=config.get('ff_expansion', 4),
        conv_kernel=config.get('conv_kernel', 31),
        use_checkpoint=config.get('use_checkpoint', False),
    ).to(device)


def train_config(config, train_dataset, val_dataset, device, epochs=30):
    """Train one config, return best metrics."""
    model = build_model(config, device)
    n_params = sum(p.numel() for p in model.parameters())

    criterion = EnsembleLoss(
        pos_weight=config['pos_weight'],
        focal_gamma=config.get('focal_gamma', 0.0),
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config['lr'],
        weight_decay=config.get('weight_decay', 0.0),
    )

    # Noam / Transformer LR schedule (per-step)
    warmup_steps = config.get('warmup_steps', 500)
    d_model = config.get('temporal_hidden', 64)

    def noam_lambda(step):
        step = max(step, 1)
        return d_model ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=noam_lambda)

    train_loader = DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)

    use_amp = device.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_loss = float('inf')
    best_metrics = {}

    for epoch in range(epochs):
        # ── Train ──
        model.train()
        train_loss_sum = 0.0
        n_batches = 0

        for batch in train_loader:
            features = batch['features'].to(device)
            onset_gt = batch['onset'].to(device)
            frame_gt = batch['frame'].to(device)
            vel_gt = batch['velocity'].to(device)
            nv_gt = batch['note_value'].to(device)

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
                )

            scaler.scale(losses['total']).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Noam schedule: step per batch

            train_loss_sum += losses['total'].item()
            n_batches += 1

        # ── Validate ──
        model.eval()
        val_loss_sum = 0.0
        n_val = 0
        onset_tp, onset_fp, onset_fn = 0, 0, 0
        frame_tp, frame_fp, frame_fn = 0, 0, 0
        nv_correct, nv_total = 0, 0

        with torch.no_grad():
            for batch in val_loader:
                features = batch['features'].to(device)
                onset_gt = batch['onset'].to(device)
                frame_gt = batch['frame'].to(device)
                vel_gt = batch['velocity'].to(device)
                nv_gt = batch['note_value'].to(device)

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
                    )

                val_loss_sum += losses['total'].item()
                n_val += 1

                onset_pred = (torch.sigmoid(out['onset_logits']) > 0.5).float()
                frame_pred = (torch.sigmoid(out['frame_logits']) > 0.5).float()

                onset_tp += ((onset_pred == 1) & (onset_gt == 1)).sum().item()
                onset_fp += ((onset_pred == 1) & (onset_gt == 0)).sum().item()
                onset_fn += ((onset_pred == 0) & (onset_gt == 1)).sum().item()

                frame_tp += ((frame_pred == 1) & (frame_gt == 1)).sum().item()
                frame_fp += ((frame_pred == 1) & (frame_gt == 0)).sum().item()
                frame_fn += ((frame_pred == 0) & (frame_gt == 1)).sum().item()

                onset_mask = onset_gt > 0.5
                if onset_mask.any():
                    nv_pred_class = out['note_value_logits'][onset_mask].argmax(dim=-1)
                    nv_gt_class = nv_gt[onset_mask]
                    nv_correct += (nv_pred_class == nv_gt_class).sum().item()
                    nv_total += nv_gt_class.numel()

        # Metrics
        avg_train = train_loss_sum / max(n_batches, 1)
        avg_val = val_loss_sum / max(n_val, 1)
        onset_p = onset_tp / max(onset_tp + onset_fp, 1)
        onset_r = onset_tp / max(onset_tp + onset_fn, 1)
        onset_f1 = 2 * onset_p * onset_r / max(onset_p + onset_r, 1e-8)
        frame_p = frame_tp / max(frame_tp + frame_fp, 1)
        frame_r = frame_tp / max(frame_tp + frame_fn, 1)
        frame_f1 = 2 * frame_p * frame_r / max(frame_p + frame_r, 1e-8)
        nv_acc = nv_correct / max(nv_total, 1)

        lr = optimizer.param_groups[0]['lr']
        print(f"    Epoch {epoch+1:3d}/{epochs}  lr={lr:.2e}  "
              f"train={avg_train:.4f}  val={avg_val:.4f}  "
              f"onset_f1={onset_f1:.3f}  frame_f1={frame_f1:.3f}  "
              f"nv_acc={nv_acc:.3f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_metrics = {
                'onset_f1': onset_f1,
                'frame_f1': frame_f1,
                'nv_acc': nv_acc,
                'val_loss': avg_val,
                'train_loss': avg_train,
                'best_epoch': epoch + 1,
            }

    return {
        'name': config['name'],
        'n_params': n_params,
        **best_metrics,
    }


# ─── Results ─────────────────────────────────────────────────────────────────

def print_results_table(results, args):
    print(f"\n{'='*90}")
    print(f"MINI EXPERIMENT RESULTS "
          f"({args.epochs} epochs, {args.train_segments} train / "
          f"{args.val_segments} val segments)")
    if args.max_note_density is not None:
        print(f"  Note density filter: <= {args.max_note_density}")
    print(f"{'='*90}\n")

    header = (f"{'Config':<20s} {'Params':>10s} {'Best Ep':>8s} "
              f"{'Val Loss':>9s} {'Onset F1':>9s} {'Frame F1':>9s} "
              f"{'NV Acc':>8s}")
    print(header)
    print('-' * len(header))

    for r in results:
        print(f"{r['name']:<20s} {r['n_params']:>10,d} "
              f"{r.get('best_epoch', '-'):>8} "
              f"{r.get('val_loss', float('inf')):>9.4f} "
              f"{r.get('onset_f1', 0):>9.3f} "
              f"{r.get('frame_f1', 0):>9.3f} "
              f"{r.get('nv_acc', 0):>8.3f}")

    print('-' * len(header))

    if results:
        best_onset = max(results, key=lambda r: r.get('onset_f1', 0))
        best_frame = max(results, key=lambda r: r.get('frame_f1', 0))
        best_nv = max(results, key=lambda r: r.get('nv_acc', 0))
        best_loss = min(results, key=lambda r: r.get('val_loss', float('inf')))
        print(f"\nBest onset F1:    {best_onset['name']} ({best_onset.get('onset_f1', 0):.3f})")
        print(f"Best frame F1:    {best_frame['name']} ({best_frame.get('frame_f1', 0):.3f})")
        print(f"Best NV accuracy: {best_nv['name']} ({best_nv.get('nv_acc', 0):.3f})")
        print(f"Best val loss:    {best_loss['name']} ({best_loss.get('val_loss', float('inf')):.4f})")


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Mini hyperparameter experiment')
    parser.add_argument('--train-segments', type=int, default=300,
                        help='Number of training segments to use')
    parser.add_argument('--val-segments', type=int, default=75,
                        help='Number of validation segments to use')
    parser.add_argument('--max-note-density', type=float, default=None,
                        help='Filter for simpler segments (e.g. 0.1 for monophonic)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Epochs per config')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--configs', type=str, nargs='*', default=None,
                        help='Run only specific configs by name (default: all)')
    args = parser.parse_args()

    device = torch.device(args.device)
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = torch.device('cpu')
    print(f"Device: {device}")

    if not (FEATURES_DIR / 'train').exists():
        print(f"Precomputed features not found at {FEATURES_DIR}")
        print("Run: python train_ensemble.py --precompute")
        return

    print("\nLoading training subset...")
    train_dataset = SubsetPrecomputedDataset(
        'train', max_segments=args.train_segments,
        max_note_density=args.max_note_density,
        augment=False, mixup_alpha=0.0,
    )
    print("Loading validation subset...")
    val_dataset = SubsetPrecomputedDataset(
        'validation', max_segments=args.val_segments,
        max_note_density=args.max_note_density,
        augment=False, mixup_alpha=0.0,
    )

    configs = CONFIGS
    if args.configs:
        config_names = set(args.configs)
        configs = [c for c in CONFIGS if c['name'] in config_names]
        unknown = config_names - {c['name'] for c in configs}
        if unknown:
            print(f"Warning: unknown configs: {unknown}")
            print(f"Available: {[c['name'] for c in CONFIGS]}")

    print(f"\nRunning {len(configs)} configs x {args.epochs} epochs each")
    print(f"Train: {len(train_dataset)} segments, Val: {len(val_dataset)} segments\n")

    results = []
    for i, config in enumerate(configs):
        print(f"\n{'─'*70}")
        print(f"[{i+1}/{len(configs)}] Config: {config['name']}")

        if config.get('model') == 'EnsembleMetaLearner':
            print(f"  [OLD ARCH] conv={config['conv_channels']} "
                  f"gru={config['gru_hidden']}x{config['gru_layers']}")
        else:
            print(f"  [Conformer] key_h={config['key_hidden']} "
                  f"d_model={config['temporal_hidden']} layers={config['temporal_layers']} "
                  f"heads={config.get('n_heads', 2)} conv_k={config.get('conv_kernel', 31)}")
        print(f"  lr={config['lr']} focal={config.get('focal_gamma', 0)} "
              f"pos_w={config['pos_weight']} warmup={config.get('warmup_steps', 500)}")

        t0 = time.perf_counter()
        result = train_config(config, train_dataset, val_dataset, device, args.epochs)
        elapsed = time.perf_counter() - t0
        result['time_sec'] = elapsed
        results.append(result)

        print(f"  Done in {elapsed:.1f}s — best onset_f1={result.get('onset_f1', 0):.3f} "
              f"at epoch {result.get('best_epoch', '?')}")

    print_results_table(results, args)


if __name__ == '__main__':
    main()
