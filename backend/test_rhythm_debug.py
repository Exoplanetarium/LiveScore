"""
Test script for duration/rhythm debugging.

Run this with: python test_rhythm_debug.py [audio_file.wav]

This will analyze the audio and print detailed information about:
1. Tempo detection
2. Inter-onset intervals (IOIs)
3. Quantization decisions (IOI vs duration-based)
4. Quantization errors
"""

import os
import sys

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from backend.detect_note import (HOP_SIZE, SAMPLE_RATE, analyze_audio,
                         duration_to_note_value)


def print_rhythm_analysis(results):
    """Print detailed rhythm analysis from detection results."""
    
    summary = results.get('analysis_summary', {})
    bpm = summary.get('detected_bpm', 120)
    confidence = summary.get('tempo_confidence', 0)
    beat_duration = 60.0 / bpm
    
    print(f"\n{'='*70}")
    print(f"RHYTHM ANALYSIS RESULTS")
    print(f"{'='*70}")
    
    print(f"\n📊 TEMPO:")
    print(f"   Detected BPM: {bpm}")
    print(f"   Confidence: {confidence:.2%}")
    print(f"   Beat duration: {beat_duration*1000:.1f}ms")
    
    # Analyze notes
    notes = results.get('notes', [])
    chords = results.get('chords', [])
    
    if notes:
        print(f"\n🎵 NOTES ({len(notes)} total):")
        print(f"{'#':>3} {'Time(s)':>8} {'Name':>6} {'Dur(ms)':>8} {'IOI(ms)':>8} {'Ratio':>6} {'Value':>12} {'Method':>20} {'Err':>5}")
        print(f"{'-'*3} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*6} {'-'*12} {'-'*20} {'-'*5}")
        
        times = [n.get('time_seconds', 0) for n in notes]
        iois = np.diff(times) * 1000 if len(times) > 1 else []
        
        high_error_count = 0
        rest_count = 0
        for i, note in enumerate(notes):
            t = note.get('time_seconds', 0)
            name = note.get('note_name', '?')
            dur = note.get('duration_seconds', 0) * 1000
            
            note_type = note.get('note_value', '?')
            if note.get('dotted', False):
                note_type = f"dot.{note_type}"
            
            method = note.get('quantization_method', 'unknown')[:20]
            error = note.get('quantization_error', 0)
            
            # Calculate ratio
            ioi_str = "-"
            ratio_str = "-"
            if i < len(iois):
                ioi_str = f"{iois[i]:.0f}"
                ratio = iois[i] / dur if dur > 0 else 0
                ratio_str = f"{ratio:.2f}"
            
            error_str = f"{error*100:.0f}%"
            
            # Flag high error or rest
            flag = ""
            if error > 0.15:
                flag = " ⚠️"
                high_error_count += 1
            if note.get('has_rest_after'):
                flag += " 🔇"
                rest_count += 1
            
            print(f"{i:>3} {t:>8.3f} {name:>6} {dur:>8.0f} {ioi_str:>8} {ratio_str:>6} {note_type:>12} {method:>20} {error_str:>5}{flag}")
        
        print(f"\n   Legend: ⚠️ = >15% error, 🔇 = rest detected after note")
        if high_error_count > 0:
            print(f"   ⚠️  {high_error_count} notes have >15% quantization error")
        if rest_count > 0:
            print(f"   🔇 {rest_count} notes have rests following them")
    
    if chords:
        print(f"\n🎹 CHORDS ({len(chords)} total):")
        print(f"{'#':>3} {'Time(s)':>8} {'Label':>12} {'Dur(ms)':>8} {'Value':>12} {'Method':>12}")
        print(f"{'-'*3} {'-'*8} {'-'*12} {'-'*8} {'-'*12} {'-'*12}")
        
        for i, chord in enumerate(chords):
            t = chord.get('time_seconds', 0)
            label = chord.get('label', '?')[:12]
            dur = chord.get('duration_seconds', 0) * 1000
            
            note_type = chord.get('note_value', '?')
            if chord.get('dotted', False):
                note_type = f"dot.{note_type}"
            
            method = chord.get('quantization_method', 'unknown')
            
            print(f"{i:>3} {t:>8.3f} {label:>12} {dur:>8.1f} {note_type:>12} {method:>12}")
    
    # Summary statistics
    if notes:
        errors = [n.get('quantization_error', 0) for n in notes]
        methods = [n.get('quantization_method', 'unknown') for n in notes]
        
        # Count IOI-based vs duration-based (check if method contains 'ioi')
        ioi_count = sum(1 for m in methods if 'ioi' in m.lower())
        dur_count = len(methods) - ioi_count
        rest_count = sum(1 for n in notes if n.get('has_rest_after'))
        
        print(f"\n📈 STATISTICS:")
        print(f"   Mean quantization error: {np.mean(errors)*100:.1f}%")
        print(f"   Max quantization error: {np.max(errors)*100:.1f}%")
        print(f"   IOI-based: {ioi_count}/{len(methods)} ({ioi_count/len(methods)*100:.0f}%)")
        print(f"   Duration-based: {dur_count}/{len(methods)} ({dur_count/len(methods)*100:.0f}%)")
        print(f"   Rests detected: {rest_count}")
        
        # Suggest tempo adjustment if many errors
        if np.mean(errors) > 0.10:
            print(f"\n💡 SUGGESTION: High average error ({np.mean(errors)*100:.1f}%) suggests tempo may be wrong.")
            print(f"   Try adjusting tempo with 0.5x or 2x multiplier in the app.")


def compare_quantization_methods(notes, bpm):
    """Compare IOI vs duration-based quantization for each note."""
    
    beat_duration = 60.0 / bpm
    
    print(f"\n{'='*70}")
    print(f"COMPARISON: IOI vs DURATION QUANTIZATION")
    print(f"{'='*70}\n")
    
    times = [n.get('time_seconds', 0) for n in notes]
    iois = np.diff(times)
    
    print(f"{'#':>3} {'IOI(ms)':>8} {'IOI_val':>10} {'Dur(ms)':>8} {'Dur_val':>10} {'Match?':>8}")
    print(f"{'-'*3} {'-'*8} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")
    
    mismatches = 0
    for i, note in enumerate(notes[:-1]):  # Skip last (no IOI)
        ioi = iois[i]
        dur = note.get('duration_seconds', 0)
        
        ioi_val = duration_to_note_value(ioi, bpm=bpm)
        dur_val = duration_to_note_value(dur, bpm=bpm)
        
        ioi_name = ('dot.' if ioi_val['dotted'] else '') + ioi_val['type']
        dur_name = ('dot.' if dur_val['dotted'] else '') + dur_val['type']
        
        match = "✓" if ioi_name == dur_name else "✗"
        if ioi_name != dur_name:
            mismatches += 1
        
        print(f"{i:>3} {ioi*1000:>8.1f} {ioi_name:>10} {dur*1000:>8.1f} {dur_name:>10} {match:>8}")
    
    print(f"\n   Mismatches: {mismatches}/{len(notes)-1}")
    if mismatches > 0:
        print(f"   In these cases, IOI-based quantization is usually more accurate")


def test_alternate_tempos(notes, detected_bpm):
    """Test how quantization errors change at different tempos."""
    
    print(f"\n{'='*70}")
    print(f"TEMPO SENSITIVITY ANALYSIS")
    print(f"{'='*70}\n")
    
    # Test tempos: 0.5x, 0.75x, 1x, 1.25x, 1.5x, 2x
    multipliers = [0.5, 0.67, 0.75, 0.83, 1.0, 1.2, 1.33, 1.5, 2.0]
    
    results = []
    
    for mult in multipliers:
        test_bpm = detected_bpm * mult
        
        # Calculate errors at this tempo
        errors = []
        times = [n.get('time_seconds', 0) for n in notes]
        iois = np.diff(times)
        
        for i, note in enumerate(notes):
            dur = note.get('duration_seconds', 0.5)
            
            # Use similar logic to quantize_rhythm_from_ioi
            if i < len(iois):
                ioi = iois[i]
                ratio = ioi / dur if dur > 0.01 else 10.0
                
                if 0.7 <= ratio <= 1.4:
                    val = duration_to_note_value(ioi, bpm=test_bpm)
                else:
                    val = duration_to_note_value(dur, bpm=test_bpm)
            else:
                val = duration_to_note_value(dur, bpm=test_bpm)
            
            errors.append(val.get('quantization_error', 0))
        
        mean_err = np.mean(errors) * 100
        max_err = np.max(errors) * 100
        high_err_count = sum(1 for e in errors if e > 0.15)
        
        results.append((test_bpm, mult, mean_err, max_err, high_err_count))
    
    print(f"{'BPM':>8} {'Mult':>6} {'Mean Err':>10} {'Max Err':>10} {'High Err':>10}")
    print(f"{'-'*8} {'-'*6} {'-'*10} {'-'*10} {'-'*10}")
    
    best = min(results, key=lambda x: x[2])  # Lowest mean error
    
    for bpm, mult, mean_err, max_err, high_count in results:
        marker = " ← BEST" if bpm == best[0] else ""
        print(f"{bpm:>8.0f} {mult:>6.2f} {mean_err:>9.1f}% {max_err:>9.1f}% {high_count:>10}{marker}")
    
    if best[0] != detected_bpm:
        print(f"\n💡 SUGGESTION: Try tempo {best[0]:.0f} BPM (multiplier {best[1]:.2f}x)")
        print(f"   This reduces mean error from {results[4][2]:.1f}% to {best[2]:.1f}%")


if __name__ == "__main__":
    
    audio_file = os.path.join(os.path.dirname(__file__), 'audio', "test_fugue1_cmajor.wav")
    
    print(f"Analyzing: {audio_file}")
    print("Please wait...")
    
    # Run analysis with debug=True to see IOI quantization details
    results = analyze_audio(audio_file, debug=True)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        sys.exit(1)
    
    # Print detailed analysis
    print_rhythm_analysis(results)
    
    # Compare methods
    notes = results.get('notes', [])
    bpm = results.get('analysis_summary', {}).get('detected_bpm', 120)
    if len(notes) > 1:
        compare_quantization_methods(notes, bpm)
        test_alternate_tempos(notes, bpm)
