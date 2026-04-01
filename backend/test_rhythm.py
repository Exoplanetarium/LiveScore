"""Quick test of rhythm quantization to diagnose issues."""

import numpy as np
from detect_note import detect_tempo_from_onsets, duration_to_note_value

print("=" * 60)
print("RHYTHM QUANTIZATION DIAGNOSTIC")
print("=" * 60)

# Test 1: Duration to note value mapping
print("\n1. DURATION -> NOTE VALUE MAPPING (120 BPM)")
print("-" * 40)
bpm = 120
beat_dur = 60.0 / bpm  # 0.5s per beat

test_durations = [0.45, 0.50, 0.55, 0.48, 0.52, 0.40, 0.60, 0.75, 1.0, 0.25, 0.30, 0.35]
for dur in test_durations:
    result = duration_to_note_value(dur, bpm=bpm)
    raw_beats = dur / beat_dur
    print(f"  {dur*1000:5.0f}ms ({raw_beats:.2f}b) -> {result['type']:8s} "
          f"({result['beats']:.2f}b) err={result['quantization_error']:.2f}"
          f" {'TRIPLET' if result.get('is_triplet') else ''}")

# Test 2: BPM detection
print("\n2. BPM DETECTION")
print("-" * 40)

# Simulate quarter notes at 100 BPM
onsets_100bpm = np.arange(0, 5, 0.6)  # 100 BPM = 0.6s per beat
result = detect_tempo_from_onsets(onsets_100bpm)
print(f"  Quarter notes at 100 BPM (0.6s intervals):")
print(f"    Detected: {result['bpm']:.1f} BPM (confidence: {result['confidence']:.2f})")

# Simulate eighth notes at 120 BPM  
onsets_120bpm_eighths = np.arange(0, 5, 0.25)  # 0.25s = eighth at 120
result = detect_tempo_from_onsets(onsets_120bpm_eighths)
print(f"  Eighth notes at 120 BPM (0.25s intervals):")
print(f"    Detected: {result['bpm']:.1f} BPM (confidence: {result['confidence']:.2f})")

# Simulate mixed rhythms (realistic piano)
onsets_mixed = [0, 0.5, 0.75, 1.0, 1.5, 2.0, 2.25, 2.5, 3.0]
result = detect_tempo_from_onsets(onsets_mixed)
print(f"  Mixed rhythm (quarters + eighths at 120 BPM):")
print(f"    Detected: {result['bpm']:.1f} BPM (confidence: {result['confidence']:.2f})")

# Test 3: Effect of wrong BPM
print("\n3. EFFECT OF WRONG BPM")
print("-" * 40)
actual_dur = 0.5  # Should be quarter note at 120 BPM

for test_bpm in [80, 100, 120, 140, 160]:
    result = duration_to_note_value(actual_dur, bpm=test_bpm)
    print(f"  500ms @ {test_bpm:3d} BPM -> {result['type']:8s} "
          f"(err={result['quantization_error']:.2f})")

# Test 4: Boundary cases - where does it flip?
print("\n4. QUANTIZATION BOUNDARIES (120 BPM)")
print("-" * 40)
print("  Looking for flip points between note values...")
prev_type = None
for ms in range(200, 800, 10):
    dur = ms / 1000.0
    result = duration_to_note_value(dur, bpm=120)
    if result['type'] != prev_type:
        if prev_type is not None:
            print(f"    {ms}ms: FLIP from {prev_type} to {result['type']}")
        prev_type = result['type']

print("\nDone.")
