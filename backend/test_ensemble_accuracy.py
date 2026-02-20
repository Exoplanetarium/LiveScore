"""Quick accuracy test for the trained ensemble model."""

import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from rhythm_training.train_ensemble import (HOP_LENGTH, MIDI_OFFSET,
                                            MODEL_PATH, PIANO_KEYS,
                                            SAMPLE_RATE, EnsembleMetaLearner,
                                            MultiResFeatureExtractor,
                                            decode_note_events)


def load_midi_notes(midi_path):
    """Load note events from MIDI file."""
    import pretty_midi
    midi = pretty_midi.PrettyMIDI(midi_path)
    notes = []
    for inst in midi.instruments:
        if inst.is_drum:
            continue
        for note in inst.notes:
            notes.append({
                'onset_time': note.start,
                'offset_time': note.end,
                'midi_note': note.pitch,
                'velocity': note.velocity,
            })
    notes.sort(key=lambda n: (n['onset_time'], n['midi_note']))
    return notes


def compute_note_metrics(pred_notes, gt_notes, onset_tol=0.05):
    """
    Compute precision, recall, F1 for note detection.
    A note is matched if onset is within tolerance and pitch matches.
    """
    matched = 0
    gt_matched = set()
    
    for pred in pred_notes:
        for i, gt in enumerate(gt_notes):
            if i in gt_matched:
                continue
            if (abs(pred['onset_time'] - gt['onset_time']) <= onset_tol
                    and pred['midi_note'] == gt['midi_note']):
                matched += 1
                gt_matched.add(i)
                break
    
    precision = matched / len(pred_notes) if pred_notes else 0
    recall = matched / len(gt_notes) if gt_notes else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'matched': matched,
        'predicted': len(pred_notes),
        'ground_truth': len(gt_notes),
    }


def test_on_sample():
    """Test ensemble model on a sample from MAESTRO test set."""
    import librosa

    # Load test index
    index_path = os.path.join(os.path.dirname(__file__), 
                              'rhythm_training', 'ensemble_index', 'test_index.json')
    
    if not os.path.exists(index_path):
        print("Test index not found. Run: python train_ensemble.py --prepare")
        return
    
    with open(index_path) as f:
        test_idx = json.load(f)
    
    pieces = test_idx['pieces']
    if not pieces:
        print("No test pieces found!")
        return
    
    print(f"Found {len(pieces)} test pieces")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}")
        return
    
    checkpoint = torch.load(str(MODEL_PATH), map_location=device, weights_only=False)
    config = checkpoint.get('config', {})
    
    print(f"Model info:")
    print(f"  - Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  - Val loss: {checkpoint.get('val_loss', '?'):.4f}")
    print(f"  - Onset F1 (validation): {checkpoint.get('onset_f1', '?'):.3f}")
    print(f"  - Frame F1 (validation): {checkpoint.get('frame_f1', '?'):.3f}")
    
    extractor = MultiResFeatureExtractor(
        sr=config.get('sample_rate', SAMPLE_RATE),
        hop_length=config.get('hop_length', HOP_LENGTH),
        device=device,
    )
    
    model = EnsembleMetaLearner(
        n_features=config.get('n_features', 373),
        conv_channels=config.get('conv_channels', [256, 256, 128]),
        gru_hidden=config.get('gru_hidden', 64),
        gru_layers=config.get('gru_layers', 2),
        n_keys=config.get('n_keys', PIANO_KEYS),
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Test on first 3 pieces (first 30 seconds each)
    n_test = min(3, len(pieces))
    all_metrics = []
    
    for i in range(n_test):
        piece = pieces[i]
        audio_path = piece['audio']
        midi_path = piece['midi']
        
        if not os.path.exists(audio_path):
            print(f"Audio not found: {audio_path}")
            continue
        
        print(f"\n[{i+1}/{n_test}] {piece.get('title', 'Unknown')[:50]}...")
        
        # Load 30 seconds of audio
        test_duration = 30.0
        sr = config.get('sample_rate', SAMPLE_RATE)
        
        audio, _ = librosa.load(audio_path, sr=sr, mono=True, duration=test_duration)
        print(f"  Audio: {len(audio)/sr:.1f}s @ {sr}Hz")
        
        # Load ground truth MIDI (filter to same time range)
        gt_notes_all = load_midi_notes(midi_path)
        gt_notes = [n for n in gt_notes_all if n['onset_time'] < test_duration]
        print(f"  Ground truth: {len(gt_notes)} notes")
        
        # Run inference
        audio_t = torch.from_numpy(audio).float().to(device)
        
        with torch.no_grad():
            features = extractor.extract(audio_t)  # (1, T, 373)
            out = model(features)
            
            onset_p = torch.sigmoid(out['onset_logits'][0]).cpu().numpy()
            frame_p = torch.sigmoid(out['frame_logits'][0]).cpu().numpy()
            velocity = out['velocity'][0].cpu().numpy()
        
        # Decode notes - test with improved post-processing
        for onset_th, frame_th in [(0.6, 0.5), (0.7, 0.5)]:
            pred_notes = decode_note_events(
                onset_p, frame_p, velocity,
                sr=sr, hop=config.get('hop_length', HOP_LENGTH),
                onset_threshold=onset_th,
                frame_threshold=frame_th,
                min_note_duration=0.05,
                min_velocity=15,
                use_peak_picking=True,
                filter_harmonics=True,
            )
            m = compute_note_metrics(pred_notes, gt_notes, onset_tol=0.05)
            print(f"  thresh={onset_th}/{frame_th}: P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} ({len(pred_notes)} notes)")
        
        # Use best for reporting
        pred_notes = decode_note_events(
            onset_p, frame_p, velocity,
            sr=sr, hop=config.get('hop_length', HOP_LENGTH),
            onset_threshold=0.7,
            frame_threshold=0.5,
            min_note_duration=0.05,
            min_velocity=15,
            use_peak_picking=True,
            filter_harmonics=True,
        )
        print(f"  Predicted: {len(pred_notes)} notes")
        
        # Compute metrics
        metrics = compute_note_metrics(pred_notes, gt_notes, onset_tol=0.05)
        all_metrics.append(metrics)
        
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall:    {metrics['recall']:.3f}")
        print(f"  F1:        {metrics['f1']:.3f}")
    
    # Average metrics
    if all_metrics:
        print("\n" + "=" * 50)
        print("OVERALL RESULTS")
        print("=" * 50)
        avg_p = np.mean([m['precision'] for m in all_metrics])
        avg_r = np.mean([m['recall'] for m in all_metrics])
        avg_f1 = np.mean([m['f1'] for m in all_metrics])
        print(f"Avg Precision: {avg_p:.3f}")
        print(f"Avg Recall:    {avg_r:.3f}")
        print(f"Avg F1:        {avg_f1:.3f}")


if __name__ == '__main__':
    test_on_sample()
