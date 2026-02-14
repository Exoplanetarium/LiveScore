"""
Download and prepare training data from MAESTRO dataset.

MAESTRO contains ~200 hours of piano recordings with aligned MIDI.
We use the MIDI files to create ground truth for rhythm quantization.

Usage:
    python prepare_training_data.py --download   # Download MAESTRO
    python prepare_training_data.py --process    # Generate training data
"""

import json
import os
import sys
import urllib.request
import zipfile
from pathlib import Path

# MAESTRO v3.0.0 - just MIDI files (small download)
MAESTRO_MIDI_URL = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"
MAESTRO_DIR = Path(__file__).parent / "maestro_midi"
TRAINING_DATA_PATH = Path(__file__).parent / "rhythm_training_data.jsonl"


def download_maestro():
    """Download MAESTRO MIDI files."""
    zip_path = MAESTRO_DIR.parent / "maestro-midi.zip"
    
    if MAESTRO_DIR.exists() and any(MAESTRO_DIR.glob("**/*.midi")):
        print(f"MAESTRO already exists at {MAESTRO_DIR}")
        return
    
    print(f"Downloading MAESTRO MIDI files...")
    print(f"URL: {MAESTRO_MIDI_URL}")
    print("This is ~57MB, may take a minute...")
    
    MAESTRO_DIR.parent.mkdir(parents=True, exist_ok=True)
    
    # Download with progress
    def show_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100 // total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\r  [{percent:3d}%] {mb_down:.1f}/{mb_total:.1f} MB", end="", flush=True)
    
    urllib.request.urlretrieve(MAESTRO_MIDI_URL, zip_path, show_progress)
    print("\nDownload complete!")
    
    # Extract
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(MAESTRO_DIR.parent)
    
    # Rename extracted folder
    extracted = MAESTRO_DIR.parent / "maestro-v3.0.0"
    if extracted.exists():
        if MAESTRO_DIR.exists():
            import shutil
            shutil.rmtree(MAESTRO_DIR)
        extracted.rename(MAESTRO_DIR)
    
    # Clean up zip
    zip_path.unlink()
    
    print(f"Extracted to {MAESTRO_DIR}")


def process_maestro():
    """Process MAESTRO MIDI files into training data."""
    from evaluate_rhythm import (extract_rhythm_features,
                                 load_midi_ground_truth, quantize_ground_truth)
    
    midi_files = list(MAESTRO_DIR.glob("**/*.midi")) + list(MAESTRO_DIR.glob("**/*.mid"))
    print(f"Found {len(midi_files)} MIDI files")
    
    if len(midi_files) == 0:
        print("No MIDI files found. Run with --download first.")
        return
    
    # Note type to index
    note_type_to_idx = {
        'whole': 0, 'half': 1, 'quarter': 2, 'eighth': 3, 
        '16th': 4, '32nd': 5
    }
    
    training_data = []
    errors = 0
    
    for i, midi_path in enumerate(midi_files):
        if i % 50 == 0:
            print(f"Processing {i}/{len(midi_files)}...")
        
        try:
            notes, tempo_changes = load_midi_ground_truth(str(midi_path))
            
            if len(notes) < 10:
                continue
            
            features = extract_rhythm_features(notes, tempo_changes)
            quantized = quantize_ground_truth(features)
            
            # Get BPM
            if len(tempo_changes[1]) > 0:
                bpm = tempo_changes[1][0]
            else:
                bpm = 120.0
            
            for j, q in enumerate(quantized):
                # Compute additional features for context
                if j > 0:
                    prev_ioi = quantized[j-1]['ioi_beats']
                else:
                    prev_ioi = 1.0
                
                if j < len(quantized) - 1:
                    next_ioi = quantized[j+1]['ioi_beats'] if j+1 < len(quantized) else q['ioi_beats']
                else:
                    next_ioi = q['ioi_beats']
                
                # Ratio of duration to IOI
                dur_ioi_ratio = q['dur_beats'] / max(q['ioi_beats'], 0.01)
                
                # Input features
                input_feat = [
                    q['dur_beats'],           # 0: Duration in beats
                    q['ioi_beats'],           # 1: IOI in beats  
                    q['beat_pos'],            # 2: Position in beat
                    q['measure_pos'] / 4.0,   # 3: Position in measure
                    prev_ioi,                 # 4: Previous IOI
                    dur_ioi_ratio,            # 5: Dur/IOI ratio
                    (q['pitch'] - 60) / 40.0, # 6: Normalized pitch
                    bpm / 120.0,              # 7: Normalized tempo
                ]
                
                # Output labels
                note_type = q.get('note_type', 'quarter')
                if note_type not in note_type_to_idx:
                    note_type = 'quarter'
                
                output_label = [
                    note_type_to_idx[note_type],
                    1 if q.get('dotted', False) else 0,
                    1 if q.get('is_triplet', False) else 0
                ]
                
                training_data.append({
                    'input': input_feat,
                    'output': output_label
                })
        
        except Exception as e:
            errors += 1
            if errors < 5:
                print(f"  Error processing {midi_path.name}: {e}")
            continue
    
    print(f"\nProcessed {len(midi_files)} files with {errors} errors")
    print(f"Generated {len(training_data)} training examples")
    
    # Analyze distribution
    from collections import Counter
    type_counts = Counter(d['output'][0] for d in training_data)
    dotted_counts = Counter(d['output'][1] for d in training_data)
    triplet_counts = Counter(d['output'][2] for d in training_data)
    
    print(f"\nNote type distribution:")
    for idx, name in enumerate(['whole', 'half', 'quarter', 'eighth', '16th', '32nd']):
        print(f"  {name}: {type_counts.get(idx, 0)} ({type_counts.get(idx, 0)/len(training_data)*100:.1f}%)")
    
    print(f"\nDotted: {dotted_counts.get(1, 0)} ({dotted_counts.get(1, 0)/len(training_data)*100:.1f}%)")
    print(f"Triplet: {triplet_counts.get(1, 0)} ({triplet_counts.get(1, 0)/len(training_data)*100:.1f}%)")
    
    # Save
    print(f"\nSaving to {TRAINING_DATA_PATH}...")
    with open(TRAINING_DATA_PATH, 'w') as f:
        for item in training_data:
            f.write(json.dumps(item) + '\n')
    
    print("Done!")
    return training_data


def train_model():
    """Train the rhythm model on prepared data."""
    from rhythm_model import train_rhythm_model
    
    if not TRAINING_DATA_PATH.exists():
        print(f"Training data not found at {TRAINING_DATA_PATH}")
        print("Run with --process first.")
        return
    
    model = train_rhythm_model(str(TRAINING_DATA_PATH), epochs=50)
    
    output_path = Path(__file__).parent / "rhythm_model.npz"
    model.save(str(output_path))
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare rhythm training data')
    parser.add_argument('--download', '-d', action='store_true',
                        help='Download MAESTRO MIDI dataset')
    parser.add_argument('--process', '-p', action='store_true',
                        help='Process MIDI files into training data')
    parser.add_argument('--train', '-t', action='store_true',
                        help='Train the rhythm model')
    parser.add_argument('--all', '-a', action='store_true',
                        help='Do all steps: download, process, train')
    
    args = parser.parse_args()
    
    if args.all or args.download:
        download_maestro()
    
    if args.all or args.process:
        process_maestro()
    
    if args.all or args.train:
        train_model()
    
    if not any([args.download, args.process, args.train, args.all]):
        parser.print_help()
