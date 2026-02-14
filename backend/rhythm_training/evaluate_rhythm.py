"""
Rhythm Evaluation and Training Framework

This module:
1. Compares transcription output to MIDI ground truth
2. Measures rhythm quantization accuracy
3. Provides training data for ML-based rhythm quantization

Usage:
    python evaluate_rhythm.py --midi ground_truth.mid --audio recording.wav
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_midi_ground_truth(midi_path):
    """
    Load MIDI file and extract note events with timing.
    
    Returns list of:
        {'onset': float (seconds), 'offset': float, 'pitch': int, 'velocity': int}
    """
    try:
        import pretty_midi
    except ImportError:
        print("Installing pretty_midi...")
        os.system("pip install pretty_midi")
        import pretty_midi
    
    midi = pretty_midi.PrettyMIDI(midi_path)
    
    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            notes.append({
                'onset': note.start,
                'offset': note.end,
                'duration': note.end - note.start,
                'pitch': note.pitch,
                'velocity': note.velocity
            })
    
    # Sort by onset time
    notes.sort(key=lambda n: (n['onset'], n['pitch']))
    
    return notes, midi.get_tempo_changes()


def extract_rhythm_features(notes, tempo_changes):
    """
    Extract rhythm features from note list.
    
    For each note, compute:
    - IOI (inter-onset interval) to next note
    - Duration
    - Position within beat
    - Position within measure
    """
    if len(notes) == 0:
        return []
    
    # Get tempo (use first tempo or default 120)
    if len(tempo_changes[1]) > 0:
        bpm = tempo_changes[1][0]
    else:
        bpm = 120.0
    
    beat_duration = 60.0 / bpm
    measure_duration = beat_duration * 4  # Assume 4/4
    
    features = []
    
    for i, note in enumerate(notes):
        onset = note['onset']
        duration = note['duration']
        
        # IOI to next note (0 for last note)
        if i < len(notes) - 1:
            ioi = notes[i + 1]['onset'] - onset
        else:
            ioi = duration  # Use duration for last note
        
        # Position within beat (0.0 to 1.0)
        beat_pos = (onset % beat_duration) / beat_duration
        
        # Position within measure (0.0 to 4.0 for 4/4)
        measure_pos = (onset % measure_duration) / beat_duration
        
        # Duration in beats
        dur_beats = duration / beat_duration
        
        # IOI in beats
        ioi_beats = ioi / beat_duration
        
        features.append({
            'onset': onset,
            'duration': duration,
            'pitch': note['pitch'],
            'ioi': ioi,
            'beat_pos': beat_pos,
            'measure_pos': measure_pos,
            'dur_beats': dur_beats,
            'ioi_beats': ioi_beats,
            'bpm': bpm
        })
    
    return features


def quantize_ground_truth(features):
    """
    Quantize ground truth features to note values.
    
    This gives us the "correct" answer for training.
    MIDI files from scores are already quantized, so we just
    need to map durations to note values.
    """
    # Standard note values in beats
    note_values = [
        ('whole', 4.0, False),
        ('whole', 6.0, True),   # dotted whole
        ('half', 2.0, False),
        ('half', 3.0, True),    # dotted half
        ('quarter', 1.0, False),
        ('quarter', 1.5, True), # dotted quarter
        ('eighth', 0.5, False),
        ('eighth', 0.75, True), # dotted eighth
        ('16th', 0.25, False),
        ('16th', 0.375, True),  # dotted 16th
        ('32nd', 0.125, False),
        # Triplets
        ('quarter', 2/3, False, True),   # quarter triplet
        ('eighth', 1/3, False, True),    # eighth triplet
        ('16th', 1/6, False, True),      # 16th triplet
    ]
    
    quantized = []
    
    for feat in features:
        # Use IOI as primary duration (more reliable than note-off)
        dur_beats = feat['ioi_beats']
        
        # Find closest note value
        best_match = None
        best_dist = float('inf')
        
        for nv in note_values:
            if len(nv) == 4:
                note_type, beats, dotted, is_triplet = nv
            else:
                note_type, beats, dotted = nv
                is_triplet = False
            
            # Log distance (ratio-based)
            if dur_beats > 0.01 and beats > 0:
                dist = abs(np.log2(dur_beats / beats))
            else:
                dist = abs(dur_beats - beats)
            
            if dist < best_dist:
                best_dist = dist
                best_match = {
                    'note_type': note_type,
                    'beats': beats,
                    'dotted': dotted,
                    'is_triplet': is_triplet,
                    'quantization_error': dist
                }
        
        quantized.append({
            **feat,
            **best_match
        })
    
    return quantized


def compare_transcription_to_ground_truth(transcribed_notes, ground_truth_notes, 
                                          time_tolerance=0.05, pitch_tolerance=0):
    """
    Compare transcribed notes to ground truth.
    
    Returns:
        - matched pairs (transcribed, ground_truth)
        - false positives (transcribed notes not in ground truth)
        - false negatives (ground truth notes not transcribed)
    """
    matched = []
    false_positives = []
    false_negatives = list(ground_truth_notes)  # Copy
    
    for t_note in transcribed_notes:
        t_onset = t_note.get('time_seconds', t_note.get('onset', 0))
        t_pitch = t_note.get('midi_note', t_note.get('pitch', 0))
        
        # Find matching ground truth note
        best_match = None
        best_idx = -1
        best_dist = float('inf')
        
        for i, gt_note in enumerate(false_negatives):
            gt_onset = gt_note.get('onset', 0)
            gt_pitch = gt_note.get('pitch', 0)
            
            time_dist = abs(t_onset - gt_onset)
            pitch_dist = abs(t_pitch - gt_pitch)
            
            if time_dist <= time_tolerance and pitch_dist <= pitch_tolerance:
                if time_dist < best_dist:
                    best_dist = time_dist
                    best_match = gt_note
                    best_idx = i
        
        if best_match:
            matched.append((t_note, best_match))
            false_negatives.pop(best_idx)
        else:
            false_positives.append(t_note)
    
    return matched, false_positives, false_negatives


def evaluate_rhythm_accuracy(matched_pairs, bpm):
    """
    For matched note pairs, compare rhythm quantization.
    """
    beat_duration = 60.0 / bpm
    
    results = {
        'total_notes': len(matched_pairs),
        'correct_note_value': 0,
        'correct_dotted': 0,
        'correct_triplet': 0,
        'errors': []
    }
    
    for t_note, gt_note in matched_pairs:
        # Get transcribed values
        t_value = t_note.get('note_value', 'quarter')
        t_dotted = t_note.get('dotted', False)
        t_triplet = t_note.get('is_triplet', False)
        
        # Compute ground truth from MIDI
        gt_duration = gt_note.get('duration', 0.5)
        gt_dur_beats = gt_duration / beat_duration
        
        # Quantize ground truth
        gt_quantized = quantize_single_duration(gt_dur_beats)
        
        # Compare
        value_match = t_value == gt_quantized['note_type']
        dotted_match = t_dotted == gt_quantized['dotted']
        triplet_match = t_triplet == gt_quantized['is_triplet']
        
        if value_match:
            results['correct_note_value'] += 1
        if dotted_match:
            results['correct_dotted'] += 1
        if triplet_match:
            results['correct_triplet'] += 1
        
        if not (value_match and dotted_match and triplet_match):
            results['errors'].append({
                'onset': t_note.get('time_seconds', 0),
                'pitch': t_note.get('midi_note', 0),
                'transcribed': {'value': t_value, 'dotted': t_dotted, 'triplet': t_triplet},
                'ground_truth': gt_quantized,
                'raw_dur_beats': gt_dur_beats
            })
    
    # Compute percentages
    n = max(1, results['total_notes'])
    results['note_value_accuracy'] = results['correct_note_value'] / n
    results['dotted_accuracy'] = results['correct_dotted'] / n
    results['triplet_accuracy'] = results['correct_triplet'] / n
    
    return results


def quantize_single_duration(dur_beats):
    """Quantize a single duration in beats to note value."""
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
    
    best_match = {'note_type': 'quarter', 'beats': 1.0, 'dotted': False, 'is_triplet': False}
    best_dist = float('inf')
    
    for note_type, beats, dotted, is_triplet in note_values:
        if dur_beats > 0.01:
            dist = abs(np.log2(dur_beats / beats))
        else:
            dist = abs(dur_beats - beats)
        
        if dist < best_dist:
            best_dist = dist
            best_match = {
                'note_type': note_type,
                'beats': beats,
                'dotted': dotted,
                'is_triplet': is_triplet
            }
    
    return best_match


def create_training_data(midi_files_dir, output_path):
    """
    Create training data from a directory of MIDI files.
    
    Output format (JSON lines):
        {"input": [onset, duration, ioi, beat_pos, pitch], "output": [note_type_idx, dotted, triplet]}
    """
    import glob
    
    midi_files = glob.glob(os.path.join(midi_files_dir, "**/*.mid"), recursive=True)
    midi_files += glob.glob(os.path.join(midi_files_dir, "**/*.midi"), recursive=True)
    
    print(f"Found {len(midi_files)} MIDI files")
    
    # Note type to index mapping
    note_type_to_idx = {
        'whole': 0, 'half': 1, 'quarter': 2, 'eighth': 3, 
        '16th': 4, '32nd': 5
    }
    
    training_data = []
    
    for midi_path in midi_files:
        try:
            notes, tempo_changes = load_midi_ground_truth(midi_path)
            features = extract_rhythm_features(notes, tempo_changes)
            quantized = quantize_ground_truth(features)
            
            for q in quantized:
                # Input features (normalized)
                input_feat = [
                    q['dur_beats'],           # Duration in beats
                    q['ioi_beats'],           # IOI in beats
                    q['beat_pos'],            # Position in beat (0-1)
                    q['measure_pos'] / 4.0,   # Position in measure (0-1)
                    (q['pitch'] - 60) / 40.0, # Normalized pitch
                ]
                
                # Output labels
                output_label = [
                    note_type_to_idx.get(q['note_type'], 2),
                    1 if q['dotted'] else 0,
                    1 if q['is_triplet'] else 0
                ]
                
                training_data.append({
                    'input': input_feat,
                    'output': output_label,
                    'raw': {
                        'dur_beats': q['dur_beats'],
                        'ioi_beats': q['ioi_beats'],
                        'note_type': q['note_type']
                    }
                })
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            continue
    
    print(f"Created {len(training_data)} training examples")
    
    # Save
    with open(output_path, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Saved to {output_path}")
    return training_data


def print_evaluation_report(results):
    """Print a formatted evaluation report."""
    print(f"\n{'='*60}")
    print("RHYTHM QUANTIZATION EVALUATION")
    print(f"{'='*60}\n")
    
    print(f"Total matched notes: {results['total_notes']}")
    print(f"\nAccuracy:")
    print(f"  Note value: {results['note_value_accuracy']*100:.1f}%")
    print(f"  Dotted:     {results['dotted_accuracy']*100:.1f}%")
    print(f"  Triplet:    {results['triplet_accuracy']*100:.1f}%")
    
    if results['errors']:
        print(f"\nFirst 10 errors:")
        for err in results['errors'][:10]:
            t = err['transcribed']
            gt = err['ground_truth']
            print(f"  @{err['onset']:.2f}s: transcribed={t['value']}"
                  f"{'.' if t['dotted'] else ''}"
                  f"{'(3)' if t['triplet'] else ''}"
                  f" vs ground_truth={gt['note_type']}"
                  f"{'.' if gt['dotted'] else ''}"
                  f"{'(3)' if gt['is_triplet'] else ''}"
                  f" (raw={err['raw_dur_beats']:.3f} beats)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate rhythm quantization')
    parser.add_argument('--midi', '-m', help='MIDI ground truth file')
    parser.add_argument('--audio', '-a', help='Audio file to transcribe')
    parser.add_argument('--create-training-data', '-c', 
                        help='Create training data from MIDI directory')
    parser.add_argument('--output', '-o', default='rhythm_training_data.jsonl',
                        help='Output path for training data')
    
    args = parser.parse_args()
    
    if args.create_training_data:
        create_training_data(args.create_training_data, args.output)
    
    elif args.midi and args.audio:
        from detect_note import analyze_audio
        
        print(f"Loading ground truth: {args.midi}")
        gt_notes, tempo_changes = load_midi_ground_truth(args.midi)
        bpm = tempo_changes[1][0] if len(tempo_changes[1]) > 0 else 120
        print(f"Ground truth: {len(gt_notes)} notes at {bpm} BPM")
        
        print(f"\nTranscribing: {args.audio}")
        results = analyze_audio(args.audio, debug=False, use_neural=True)
        
        if 'error' in results:
            print(f"Error: {results['error']}")
            sys.exit(1)
        
        transcribed = results.get('notes', [])
        print(f"Transcribed: {len(transcribed)} notes")
        
        # Compare
        matched, fp, fn = compare_transcription_to_ground_truth(transcribed, gt_notes)
        print(f"\nMatched: {len(matched)}, False positives: {len(fp)}, False negatives: {len(fn)}")
        
        # Evaluate rhythm
        eval_results = evaluate_rhythm_accuracy(matched, bpm)
        print_evaluation_report(eval_results)
    
    else:
        parser.print_help()
