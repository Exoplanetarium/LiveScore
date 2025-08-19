#!/usr/bin/env python3
"""
Test script for the improved chord vs note detection logic
Focuses on analyzing just the first few onsets of the chromatic run
"""

import os
import sys

sys.path.append(os.path.dirname(__file__))

from detect_note import analyze_audio_cmdline, read_wav


def main():
    # Test on chromatic run
    wav_path = os.path.join(os.path.dirname(__file__), 'audio', 'test_chromatic.wav')
    print(f"🧪 Testing improved detection on: {wav_path}")
    
    try:
        audio = read_wav(wav_path)
        print(f"✓ Loaded audio: {len(audio)} samples, {len(audio)/22050:.2f}s duration")
    except Exception as e:
        print(f"✗ Failed to load audio: {e}")
        return
    
    # Run analysis
    results = analyze_audio_cmdline(audio)
    
    if "error" in results:
        print(f"✗ Analysis failed: {results['error']}")
        return
    
    print(f"\n" + "="*60)
    print(f"IMPROVED DETECTION RESULTS SUMMARY")
    print(f"="*60)
    print(f"Total onsets analyzed: {len(results['onsets'])}")
    print(f"Notes detected: {len(results['notes'])}")
    print(f"Chords detected: {len(results['chords'])}")
    print(f"Note/Chord ratio: {len(results['notes'])}/{len(results['chords'])}")
    
    if results['chords']:
        print(f"\n⚠️  Chords detected (potential false positives):")
        for chord in results['chords']:
            print(f"  {chord['time_seconds']:6.2f}s: {chord['label']} (confidence: {chord['confidence']:.3f})")
    
    if results['notes']:
        print(f"\n✓ Notes detected:")
        for note in results['notes'][:10]:  # Show first 10
            print(f"  {note['time_seconds']:6.2f}s: {note['note_name']}")
        if len(results['notes']) > 10:
            print(f"  ... and {len(results['notes']) - 10} more notes")

if __name__ == "__main__":
    main()
