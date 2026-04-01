# CUDA OOM Diagnosis & Memory Fix

## Problem Summary

**Previous OOM Error**: `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.03 GiB. GPU 0 has a total capacity of 11.94 GiB of which 0 bytes is free. Of the allocated memory 26.50 GiB is allocated by PyTorch`

**Root Cause**: The per-key Conformer architecture creates O(T²) attention matrices across multiple heads and batch dimensions:

- Effective batch size: `B*88` (88 piano keys distributed across batch)
- Sequence length: T=312 frames (~10 seconds)
- Each attention head creates: `(B*88, T, T)` tensors for attention weights
- With temporal_hidden=64, n_heads=2, batch_size=4, this generated massive intermediate tensors

**Memory Formula**:

```
Per Attention Head: (B*88, T, T) × 4 bytes ≈ (352, 312, 312) × 4 ≈ 54 MB
Total with all heads + gradients + intermediate activations:
  - 4 Conformer blocks × 2 heads × backward pass overhead ≈ 2-3 GB
  - Per-key reshape duplications + attention context = additional explosion
```

## Fixes Implemented

### 1. **Gradient Checkpointing** (trades compute for memory)

- Added `use_checkpoint` parameter to `ConformerBlock`
- Implements `torch.utils.checkpoint.checkpoint()` with `use_reentrant=False`
- Activations recomputed during backward instead of stored
- **Expected Savings**: ~35-40% memory reduction

### 2. **Reduced Default Dimensions**

- `batch_size`: 8 → **2** (default, can be increased with memory sufficient GPUs)
- `temporal_hidden`: 64 → **32** (d_model dimension for attention)
- `n_heads`: 2 → **1** (reduces per-head overhead)
- **Expected Savings**: ~60-75% with dimension reduction alone

### 3. **New Command-Line Options**

```bash
# Training with memory-efficient defaults
python train_ensemble.py --train --epochs 50

# With gradient checkpointing (slower but more memory-efficient)
python train_ensemble.py --train --epochs 50 --use-checkpoint

# For GPUs with >20GB VRAM, scale back up
python train_ensemble.py --train --epochs 50 \
  --batch-size 8 --temporal-hidden 64 --n-heads 2

# For memory-constrained GPUs
python train_ensemble.py --train --epochs 50 \
  --batch-size 1 --temporal-hidden 16 --n-heads 1 --use-checkpoint
```

## Changed Files

### `train_ensemble.py`

1. **ConformerBlock** (lines 530-579):
   - Added `use_checkpoint` parameter
   - Split forward into `_forward_impl()` and `forward()`
   - Uses `torch.utils.checkpoint.checkpoint()` when enabled

2. **PitchAwareTranscriber** (line 610):
   - Added `use_checkpoint` parameter to `__init__`
   - Passes to all ConformerBlock instances

3. **Argparse** (lines 2079-2111):
   - `--batch-size`: 8 → 2
   - `--temporal-hidden`: 64 → 32
   - `--n-heads`: 2 → 1
   - Added `--use-checkpoint` flag
   - Updated help text with memory notes

4. **Model Construction** (lines 1611, 1683):
   - Updated both instantiation points to pass `use_checkpoint`

5. **Config Saving** (line 1963):
   - Added `use_checkpoint` to saved checkpoint config

## Recommended Next Steps

### For Your Config (RTX 5070 Ti, 11.94 GB VRAM):

```bash
# Initial safe test (should use ~3-4 GB)
python train_ensemble.py --precompute
python train_ensemble.py --train --epochs 10 --batch-size 2 \
  --temporal-hidden 32 --n-heads 1 --use-checkpoint

# If that works, try without checkpointing
python train_ensemble.py --train --epochs 50 --batch-size 2 \
  --temporal-hidden 32 --n-heads 1

# Then gradually increase
python train_ensemble.py --train --epochs 50 --batch-size 4 \
  --temporal-hidden 32 --n-heads 1

# And finally try full config
python train_ensemble.py --train --epochs 50 --batch-size 4 \
  --temporal-hidden 48 --n-heads 2
```

## Expected Memory Usage (Approximate)

| Config | Batch | d_model | Heads | Checkpoint | Est. Peak VRAM |
| ------ | ----- | ------- | ----- | ---------- | -------------- |
| Tiny   | 1     | 16      | 1     | Yes        | ~1.5 GB        |
| Small  | 2     | 32      | 1     | Yes        | ~2.5 GB        |
| Medium | 2     | 32      | 1     | No         | ~3.5 GB        |
| Large  | 4     | 48      | 2     | No         | ~6-7 GB        |
| XL     | 8     | 64      | 2     | No         | ~12+ GB (OOM)  |

## Test Script Location

`backend/test_memory.py` - Use this to profile memory before training

```bash
python backend/test_memory.py
```

Tests configs from most conservative to most aggressive, stopping at first OOM.
