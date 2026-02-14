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
from detect_note import (HOP_SIZE, SAMPLE_RATE, analyze_audio,
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


def print_visual_timeline(results, max_width=100):
    """
    Print a horizontal timeline showing notes/chords with proportional duration bars.
    
    Output looks like:
    C4════E4══G4════════C5══[Cmaj]════════
    
    Where the bar length represents duration proportionally.
    """
    notes = results.get('notes', [])
    chords = results.get('chords', [])
    summary = results.get('analysis_summary', {})
    bpm = summary.get('detected_bpm', 120)
    beat_duration = 60.0 / bpm
    
    print(f"\n{'='*max_width}")
    print(f"🎼 VISUAL TRANSCRIPTION TIMELINE (BPM: {bpm})")
    print(f"{'='*max_width}")
    print(f"   Bar length ∝ duration | ─ = 1/16 beat | ░ = rest")
    print(f"{'─'*max_width}\n")
    
    # Combine notes and chords into unified events
    events = []
    
    for note in notes:
        events.append({
            'time': note.get('time_seconds', 0),
            'label': note.get('note_name', '?'),
            'duration': note.get('duration_seconds', 0.25),
            'note_value': note.get('note_value', 'quarter'),
            'dotted': note.get('dotted', False),
            'is_triplet': note.get('is_triplet', False),
            'type': 'note',
            'has_rest': note.get('has_rest_after', False),
            'rest_dur': note.get('rest_duration', 0)
        })
    
    for chord in chords:
        # Use chord label or construct from notes
        label = chord.get('label', None)
        if not label:
            midi_notes = chord.get('midi_notes', [])
            if midi_notes:
                from detect_note import note_to_name
                names = [note_to_name(m) for m in midi_notes[:2]]  # First 2 notes
                label = '+'.join(names)
                if len(midi_notes) > 2:
                    label += f"+{len(midi_notes)-2}"
            else:
                label = "chd"
        
        events.append({
            'time': chord.get('time_seconds', 0),
            'label': f"[{label}]",
            'duration': chord.get('duration_seconds', 0.5),
            'note_value': chord.get('note_value', 'quarter'),
            'dotted': chord.get('dotted', False),
            'is_triplet': chord.get('is_triplet', False),
            'type': 'chord',
            'has_rest': chord.get('has_rest_after', False),
            'rest_dur': chord.get('rest_duration', 0)
        })
    
    if not events:
        print("   No notes or chords detected.")
        return
    
    # Sort by time
    events.sort(key=lambda e: e['time'])
    
    # Calculate scale: how many chars per beat?
    # We want a reasonable density - about 4 chars per 16th note = 16 chars per beat
    chars_per_beat = 12
    
    # Build the continuous timeline string
    timeline = ""
    prev_end_time = 0
    line_count = 1
    
    for event in events:
        t = event['time']
        label = event['label']
        dur = event['duration']
        nv = event['note_value']
        dotted = event['dotted']
        is_triplet = event['is_triplet']
        
        # Add rest if there's a gap
        gap = t - prev_end_time
        if gap > beat_duration * 0.15:  # More than ~1/8 beat gap
            gap_beats = gap / beat_duration
            rest_chars = max(1, int(gap_beats * chars_per_beat))
            timeline += "░" * rest_chars
        
        # Add note/chord label
        # Add markers for dotted/triplet
        prefix = ""
        if is_triplet:
            prefix = "³"
        if dotted:
            prefix += "•"
        
        timeline += prefix + label
        
        # Add duration bar
        dur_beats = dur / beat_duration
        bar_chars = max(1, int(dur_beats * chars_per_beat))
        
        # Use different bar chars for notes vs chords
        if event['type'] == 'chord':
            bar_char = "═"
        else:
            bar_char = "─"
        
        timeline += bar_char * bar_chars
        
        prev_end_time = t + dur
    
    # Print with line wrapping
    print("   ", end="")
    line_len = 3
    wrap_width = max_width - 6
    
    for char in timeline:
        print(char, end="")
        line_len += 1
        if line_len >= wrap_width and char in "─═░":
            print("\n   ", end="")
            line_len = 3
            line_count += 1
    
    print(f"\n\n   ({len(events)} events over {line_count} lines)")
    print(f"   Legend: ─ = note duration | ═ = chord duration | ░ = rest")
    print(f"           • = dotted | ³ = triplet | [X] = chord")


def print_beat_aligned_timeline(results, max_width=100):
    """
    Print timeline with beat markers for easier reading.
    Each line is one measure (4 beats by default).
    """
    notes = results.get('notes', [])
    chords = results.get('chords', [])
    summary = results.get('analysis_summary', {})
    bpm = summary.get('detected_bpm', 120)
    beat_duration = 60.0 / bpm
    beats_per_measure = 4
    
    print(f"\n{'='*max_width}")
    print(f"🎵 BEAT-ALIGNED TIMELINE (BPM: {bpm}, {beats_per_measure}/4 time)")
    print(f"{'='*max_width}\n")
    
    # Combine events
    events = []
    for note in notes:
        events.append({
            'time': note.get('time_seconds', 0),
            'label': note.get('note_name', '?'),
            'duration': note.get('duration_seconds', 0.25),
            'type': 'note'
        })
    for chord in chords:
        label = chord.get('label', 'chd')[:6]
        events.append({
            'time': chord.get('time_seconds', 0),
            'label': f"[{label}]",
            'duration': chord.get('duration_seconds', 0.5),
            'type': 'chord'
        })
    
    if not events:
        print("   No events.")
        return
    
    events.sort(key=lambda e: e['time'])
    
    # Calculate total measures
    total_time = events[-1]['time'] + events[-1]['duration']
    total_beats = total_time / beat_duration
    total_measures = int(np.ceil(total_beats / beats_per_measure))
    
    chars_per_beat = 16
    chars_per_measure = chars_per_beat * beats_per_measure
    
    # Build each measure
    for measure in range(total_measures):
        measure_start = measure * beats_per_measure * beat_duration
        measure_end = (measure + 1) * beats_per_measure * beat_duration
        
        # Beat markers
        header = f"M{measure+1:02d}│"
        for b in range(beats_per_measure):
            header += f"{b+1}" + "·" * (chars_per_beat - 1)
        print(f"   {header}")
        
        # Build measure content
        line = "   " + " " * 4  # Indent for measure number
        pos = 0  # Current position in chars
        
        # Get events in this measure
        measure_events = [e for e in events 
                         if e['time'] >= measure_start and e['time'] < measure_end]
        
        for event in measure_events:
            # Position within measure
            event_beat = (event['time'] - measure_start) / beat_duration
            event_pos = int(event_beat * chars_per_beat)
            
            # Fill gap with spaces/rests
            if event_pos > pos:
                gap = event_pos - pos
                line += "·" * gap
                pos = event_pos
            
            # Add label
            label = event['label']
            line += label
            pos += len(label)
            
            # Add duration bar
            dur_beats = event['duration'] / beat_duration
            bar_chars = max(0, int(dur_beats * chars_per_beat) - len(label))
            bar_char = "═" if event['type'] == 'chord' else "─"
            line += bar_char * bar_chars
            pos += bar_chars
        
        # Fill rest of measure
        remaining = chars_per_measure - (pos - 4)
        if remaining > 0:
            line += "·" * remaining
        
        print(line)
        print()


def print_compact_score(results, beats_per_line=4):
    """
    Print a compact score-like view with multiple beats per line.
    Shows the rhythm pattern more clearly.
    """
    notes = results.get('notes', [])
    chords = results.get('chords', [])
    summary = results.get('analysis_summary', {})
    bpm = summary.get('detected_bpm', 120)
    beat_duration = 60.0 / bpm
    
    print(f"\n{'='*70}")
    print(f"🎵 COMPACT SCORE VIEW ({beats_per_line} beats per line)")
    print(f"{'='*70}\n")
    
    # Combine and sort events
    events = []
    for note in notes:
        events.append({
            'time': note.get('time_seconds', 0),
            'label': note.get('note_name', '?'),
            'duration': note.get('duration_seconds', 0.25),
            'note_value': note.get('note_value', 'quarter'),
            'dotted': note.get('dotted', False),
            'type': 'note'
        })
    for chord in chords:
        label = chord.get('label', 'chd')
        events.append({
            'time': chord.get('time_seconds', 0),
            'label': f"[{label[:6]}]",
            'duration': chord.get('duration_seconds', 0.5),
            'note_value': chord.get('note_value', 'quarter'),
            'dotted': chord.get('dotted', False),
            'type': 'chord'
        })
    
    if not events:
        print("   No events to display.")
        return
    
    events.sort(key=lambda e: e['time'])
    
    # Note value symbols (similar to music notation)
    note_symbols = {
        'whole': '𝅝', 'half': '𝅗𝅥', 'quarter': '♩', 'eighth': '♪',
        '16th': '𝅘𝅥𝅯', '32nd': '𝅘𝅥𝅰', 'grace': '⁽♪⁾'
    }
    
    # ASCII fallback symbols
    ascii_symbols = {
        'whole': 'W', 'half': 'H', 'quarter': 'Q', 'eighth': 'E',
        '16th': 'S', '32nd': 'T', 'grace': 'g'
    }
    
    # Determine total duration
    if events:
        total_time = events[-1]['time'] + events[-1]['duration']
        total_beats = total_time / beat_duration
    else:
        total_beats = 0
    
    # Group events by beat
    current_beat = 0
    line_events = []
    lines = []
    
    for event in events:
        event_beat = event['time'] / beat_duration
        
        # Check if we need a new line
        while event_beat >= current_beat + beats_per_line:
            if line_events:
                lines.append((current_beat, line_events))
            line_events = []
            current_beat += beats_per_line
        
        line_events.append(event)
    
    # Don't forget the last line
    if line_events:
        lines.append((current_beat, line_events))
    
    # Print each line
    for start_beat, line_events in lines:
        # Header showing beat numbers
        beat_header = f"   Beat {int(start_beat)+1}-{int(start_beat)+beats_per_line}: "
        print(beat_header)
        
        # Print events on this line
        line_str = "   "
        for event in line_events:
            label = event['label'][:6]
            nv = event['note_value']
            sym = ascii_symbols.get(nv, '?')
            dot = '.' if event['dotted'] else ''
            
            line_str += f"{label}({sym}{dot}) "
        
        print(line_str)
        print()


def test_alternate_tempos(notes, detected_bpm):
    """Test how quantization errors change at different tempos."""
    
    print(f"\n{'='*70}")
    print(f"TEMPO SENSITIVITY ANALYSIS")
    print(f"{'='*70}\n")
    
    # Test tempos: 0.33x to 2x (covers typical range 40-240 BPM)
    multipliers = [0.33, 0.4, 0.5, 0.67, 0.75, 0.83, 1.0, 1.2, 1.33, 1.5, 2.0]
    
    results = []
    
    for mult in multipliers:
        test_bpm = detected_bpm * mult
        
        # Calculate errors at this tempo using IOI (more reliable than duration)
        times = [n.get('time_seconds', 0) for n in notes]
        iois = np.diff(times)
        
        errors = []
        for ioi in iois:
            val = duration_to_note_value(ioi, bpm=test_bpm)
            errors.append(val.get('quantization_error', 0))
        
        if not errors:
            continue
            
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
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug rhythm detection')
    parser.add_argument('audio_file', nargs='?', 
                        default=os.path.join(os.path.dirname(__file__), 'audio', "test_fugue1_cmajor.wav"),
                        help='Path to audio file')
    parser.add_argument('--neural', '-n', action='store_true',
                        help='Use neural network model (more accurate, requires GPU)')
    parser.add_argument('--device', '-d', default='cuda',
                        help='Device for neural model: cuda or cpu (default: cuda)')
    parser.add_argument('--width', '-w', type=int, default=120,
                        help='Max width for timeline display (default: 120)')
    args = parser.parse_args()
    
    audio_file = args.audio_file
    
    print(f"Analyzing: {audio_file}")
    if args.neural:
        print(f"Using NEURAL model on {args.device}...")
    else:
        print("Using signal processing pipeline...")
    print("Please wait...")
    
    # Run analysis - use neural if requested
    results = analyze_audio(audio_file, debug=True, use_neural=args.neural, device=args.device)
    
    if 'error' in results:
        print(f"Error: {results['error']}")
        sys.exit(1)
    
    # Print detailed analysis
    print_rhythm_analysis(results)
    
    # Print horizontal visual timeline (notes in sequence with duration bars)
    print_visual_timeline(results, max_width=args.width)
    
    # Print beat-aligned view (measure by measure)
    print_beat_aligned_timeline(results, max_width=args.width)
    
    # Compare methods
    notes = results.get('notes', [])
    bpm = results.get('analysis_summary', {}).get('detected_bpm', 120)
    if len(notes) > 1:
        compare_quantization_methods(notes, bpm)
        test_alternate_tempos(notes, bpm)
