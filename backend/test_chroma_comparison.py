"""
Compare chroma and CQT between regular and optimized pipelines
"""
import sys

import librosa
import numpy as np

# Add backend to path
sys.path.insert(0, r'C:\Users\ToniH\Documents\GitHub\LiveScore\backend')

from detect_note import (CQT_BINS, HOP_SIZE, SAMPLE_RATE, extract_chroma,
                         read_wav)

# You need to provide a test file path
if len(sys.argv) < 2:
    print("Usage: python test_chroma_comparison.py <path_to_wav_file>")
    sys.exit(1)

test_file = sys.argv[1]

print("=" * 80)
print("CHROMA & CQT COMPARISON TEST")
print("=" * 80)

# Load audio
audio = read_wav(test_file)
print(f"\nAudio loaded: {len(audio)} samples")

# ===== REGULAR PIPELINE =====
print("\n" + "=" * 80)
print("REGULAR PIPELINE")
print("=" * 80)

# Extract chroma using extract_chroma (what regular pipeline uses)
chroma_regular = extract_chroma(audio, SAMPLE_RATE, hop_length=HOP_SIZE)
print(f"Chroma (extract_chroma): shape={chroma_regular.shape}")
print(f"  First frame: {chroma_regular[:, 0]}")
print(f"  Stats: min={chroma_regular.min():.6f}, max={chroma_regular.max():.6f}, mean={chroma_regular.mean():.6f}")

# Compute CQT
C_full_regular = np.abs(librosa.cqt(
    y=audio, 
    sr=SAMPLE_RATE,
    hop_length=HOP_SIZE,
    n_bins=CQT_BINS,
    bins_per_octave=12,
    fmin=librosa.note_to_hz('C1')
))
print(f"\nCQT (C_full): shape={C_full_regular.shape}")
print(f"  First frame stats: min={C_full_regular[:, 0].min():.6f}, max={C_full_regular[:, 0].max():.6f}")

# ===== OPTIMIZED PIPELINE =====
print("\n" + "=" * 80)
print("OPTIMIZED PIPELINE")
print("=" * 80)

# Compute CQT
C_full_optimized = np.abs(librosa.cqt(
    y=audio, sr=SAMPLE_RATE, hop_length=HOP_SIZE,
    n_bins=CQT_BINS, bins_per_octave=12,
    fmin=librosa.note_to_hz('C1')
))
print(f"CQT (C_full): shape={C_full_optimized.shape}")
print(f"  First frame stats: min={C_full_optimized[:, 0].min():.6f}, max={C_full_optimized[:, 0].max():.6f}")

# Derive chroma from CQT (what optimized pipeline does)
chroma_optimized = librosa.feature.chroma_cqt(C=C_full_optimized, sr=SAMPLE_RATE, hop_length=HOP_SIZE)
chroma_optimized = chroma_optimized / (np.linalg.norm(chroma_optimized, axis=0, keepdims=True) + 1e-6)
print(f"\nChroma (from CQT): shape={chroma_optimized.shape}")
print(f"  First frame: {chroma_optimized[:, 0]}")
print(f"  Stats: min={chroma_optimized.min():.6f}, max={chroma_optimized.max():.6f}, mean={chroma_optimized.mean():.6f}")

# ===== COMPARISON =====
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

# Compare CQT
print(f"\nCQT Comparison:")
print(f"  Shapes match: {C_full_regular.shape == C_full_optimized.shape}")
print(f"  Are they close: {np.allclose(C_full_regular, C_full_optimized, rtol=1e-5, atol=1e-8)}")
if not np.allclose(C_full_regular, C_full_optimized):
    print(f"  Max absolute diff: {np.max(np.abs(C_full_regular - C_full_optimized))}")
    print(f"  Mean absolute diff: {np.mean(np.abs(C_full_regular - C_full_optimized))}")

# Compare Chroma
print(f"\nChroma Comparison:")
print(f"  Shapes match: {chroma_regular.shape == chroma_optimized.shape}")
print(f"  Are they close: {np.allclose(chroma_regular, chroma_optimized, rtol=1e-5, atol=1e-8)}")
if not np.allclose(chroma_regular, chroma_optimized):
    print(f"  Max absolute diff: {np.max(np.abs(chroma_regular - chroma_optimized))}")
    print(f"  Mean absolute diff: {np.mean(np.abs(chroma_regular - chroma_optimized))}")
    print(f"\n  First 5 frames comparison:")
    for i in range(min(5, chroma_regular.shape[1])):
        print(f"    Frame {i}:")
        print(f"      Regular:   {chroma_regular[:, i]}")
        print(f"      Optimized: {chroma_optimized[:, i]}")
        print(f"      Diff: {np.abs(chroma_regular[:, i] - chroma_optimized[:, i])}")

print("\n" + "=" * 80)
print("INVESTIGATING extract_chroma() internals")
print("=" * 80)

# Check what extract_chroma actually does NOW (after fix)
print("\nAfter fixing extract_chroma() to use same CQT parameters:")

# Test with explicit parameters matching what we use in C_full
chroma_with_params = librosa.feature.chroma_cqt(
    y=audio, sr=SAMPLE_RATE, hop_length=HOP_SIZE,
    n_chroma=12, fmin=librosa.note_to_hz('C1'),
    bins_per_octave=12
)
chroma_with_params_norm = chroma_with_params / (np.linalg.norm(chroma_with_params, axis=0, keepdims=True) + 1e-6)

print(f"\nchroma_cqt with explicit params (matching extract_chroma):")
print(f"  Shape: {chroma_with_params_norm.shape}")
print(f"  Matches regular chroma: {np.allclose(chroma_with_params_norm, chroma_regular)}")
if not np.allclose(chroma_with_params_norm, chroma_regular):
    print(f"  Max diff: {np.max(np.abs(chroma_with_params_norm - chroma_regular))}")

# Now test deriving from explicit CQT with C parameter
print(f"\nDeriving chroma from explicit C_full (optimized pipeline approach):")
chroma_from_cqt = librosa.feature.chroma_cqt(C=C_full_regular, sr=SAMPLE_RATE, hop_length=HOP_SIZE)
chroma_from_cqt_norm = chroma_from_cqt / (np.linalg.norm(chroma_from_cqt, axis=0, keepdims=True) + 1e-6)

print(f"  Shape: {chroma_from_cqt_norm.shape}")
print(f"  Matches regular chroma: {np.allclose(chroma_from_cqt_norm, chroma_regular)}")
print(f"  Matches chroma_with_params: {np.allclose(chroma_from_cqt_norm, chroma_with_params_norm)}")
if not np.allclose(chroma_from_cqt_norm, chroma_with_params_norm):
    print(f"  Max diff from chroma_with_params: {np.max(np.abs(chroma_from_cqt_norm - chroma_with_params_norm))}")
    print(f"\n  This means even with same parameters, chroma_cqt(y=audio) != chroma_cqt(C=C_full)")
    print(f"  Reason: When y= is passed, librosa computes its own internal CQT")
    print(f"  When C= is passed, librosa uses the provided CQT")

# Check if the issue is n_bins parameter
print(f"\n" + "=" * 80)
print(f"TESTING: Does librosa.cqt default n_bins differ from our CQT_BINS?")
print(f"=" * 80)

# What does librosa.cqt use as default n_bins?
# From librosa docs: n_bins defaults to 7 * bins_per_octave (84 bins for bins_per_octave=12)
print(f"\nOur explicit CQT:")
print(f"  n_bins={CQT_BINS} (88 bins)")
print(f"  bins_per_octave=12")
print(f"  fmin=librosa.note_to_hz('C1') = {librosa.note_to_hz('C1'):.2f} Hz")

print(f"\nlibrosa.feature.chroma_cqt default CQT (when y= is used):")
print(f"  n_bins defaults to 7 * bins_per_octave = 84 bins")
print(f"  This is DIFFERENT from our CQT_BINS=88!")

# The solution: pass n_bins explicitly to chroma_cqt
print(f"\n" + "=" * 80)
print(f"SOLUTION TEST: Pass n_bins explicitly")
print(f"=" * 80)

# Note: librosa.feature.chroma_cqt doesn't have n_bins parameter directly
# It has n_octaves parameter. Let's calculate what n_octaves gives us 88 bins:
# n_bins = n_octaves * bins_per_octave
# 88 = n_octaves * 12
# n_octaves = 88 / 12 = 7.333...

n_octaves_for_88_bins = CQT_BINS / 12.0
print(f"\nTo get {CQT_BINS} bins with bins_per_octave=12:")

chroma_with_correct_bins = librosa.feature.chroma_cqt(
    y=audio, sr=SAMPLE_RATE, hop_length=HOP_SIZE,
    n_chroma=12,
    fmin=librosa.note_to_hz('C1'),
    bins_per_octave=12
)
chroma_with_correct_bins_norm = chroma_with_correct_bins / (np.linalg.norm(chroma_with_correct_bins, axis=0, keepdims=True) + 1e-6)

print(f"\nchroma_cqt with n_octaves={n_octaves_for_88_bins:.4f}:")
print(f"  Shape: {chroma_with_correct_bins_norm.shape}")
print(f"  Matches optimized chroma: {np.allclose(chroma_with_correct_bins_norm, chroma_optimized)}")
print(f"  Matches chroma from C_full: {np.allclose(chroma_with_correct_bins_norm, chroma_from_cqt_norm)}")

if not np.allclose(chroma_with_correct_bins_norm, chroma_from_cqt_norm):
    print(f"  Max diff: {np.max(np.abs(chroma_with_correct_bins_norm - chroma_from_cqt_norm))}")
else:
    print(f"  ✓ SUCCESS! Using n_octaves={n_octaves_for_88_bins:.4f} makes chroma_cqt(y=...) match chroma_cqt(C=...)")


print("\n" + "=" * 80)
