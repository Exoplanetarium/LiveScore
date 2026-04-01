#!/usr/bin/env python3
"""Quick memory usage test for Conformer with gradient checkpointing."""

import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

from rhythm_training.train_ensemble import (N_FEATURES_MULTI_HOP,
                                            PitchAwareTranscriber)


def test_memory(temporal_hidden=32, n_heads=1, use_checkpoint=False, batch_size=2, seq_len=625):
    """Test forward + backward memory usage."""
    print(f"\n{'='*70}")
    print(f"Testing: temporal_hidden={temporal_hidden}, n_heads={n_heads}, "
          f"use_checkpoint={use_checkpoint}")
    print(f"Batch size: {batch_size}, Sequence length: {seq_len}")
    print(f"{'='*70}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()

    # Create model
    model = PitchAwareTranscriber(
        key_hidden=32,
        temporal_hidden=temporal_hidden,
        temporal_layers=4,
        n_key_conv_layers=2,
        dropout=0.1,
        n_heads=n_heads,
        ff_expansion=4,
        conv_kernel=31,
        use_checkpoint=use_checkpoint,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Create dummy input
    x = torch.randn(batch_size, seq_len, N_FEATURES_MULTI_HOP, device=device)
    target_onset = torch.randint(0, 2, (batch_size, seq_len, 88), device=device).float()

    if device.type == 'cuda':
        allocated_before = torch.cuda.memory_allocated() / 1e9
        print(f"Memory before forward: {allocated_before:.2f} GB")

    # Forward pass
    model.train()
    offset_logits = model(x)['onset_logits']

    if device.type == 'cuda':
        allocated_after_fwd = torch.cuda.memory_allocated() / 1e9
        print(f"Memory after forward: {allocated_after_fwd:.2f} GB")

    # Backward pass
    loss = torch.nn.functional.binary_cross_entropy_with_logits(offset_logits, target_onset)
    loss.backward()

    if device.type == 'cuda':
        allocated_after_bwd = torch.cuda.memory_allocated() / 1e9
        peak_memory = torch.cuda.max_memory_allocated() / 1e9
        print(f"Memory after backward: {allocated_after_bwd:.2f} GB")
        print(f"Peak memory used: {peak_memory:.2f} GB")

    print("✓ Test completed successfully!")
    return True

if __name__ == '__main__':
    # Test configurations in order of increasing memory usage
    configs = [
        # (temporal_hidden, n_heads, use_checkpoint, batch_size)
        (32, 1, True, 1),      # Minimal: should work
        (32, 1, True, 2),      # Low: recommended starting point
        (32, 1, False, 2),     # Without checkpointing
        (64, 1, True, 2),      # Moderate: with checkpointing
        (64, 1, True, 4),      # Higher batch: with checkpointing
        (64, 2, True, 4),      # Original config (might still OOM)
    ]

    for temporal_hidden, n_heads, use_checkpoint, batch_size in configs:
        try:
            test_memory(
                temporal_hidden=temporal_hidden,
                n_heads=n_heads,
                use_checkpoint=use_checkpoint,
                batch_size=batch_size,
            )
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                print(f"✗ OOM Error: {e}")
                break