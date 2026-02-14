"""
ML-based Rhythm Quantization

This replaces the heuristic-based quantization with a learned model.
The model takes raw timing features and predicts quantized note values.

Architecture: Simple 2-layer MLP (can run on CPU, fast inference)
    Input: [dur_beats, ioi_beats, beat_pos, measure_pos, prev_note_type, next_ioi]
    Output: [note_type (6 classes), dotted (2), triplet (2)]

Training: Use MIDI files where rhythm is already quantized as ground truth.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

# Note type mappings
NOTE_TYPES = ['whole', 'half', 'quarter', 'eighth', '16th', '32nd']
NOTE_TYPE_TO_IDX = {nt: i for i, nt in enumerate(NOTE_TYPES)}
IDX_TO_NOTE_TYPE = {i: nt for i, nt in enumerate(NOTE_TYPES)}

# Note type to beats
NOTE_TYPE_BEATS = {
    'whole': 4.0, 'half': 2.0, 'quarter': 1.0, 
    'eighth': 0.5, '16th': 0.25, '32nd': 0.125
}


class RhythmQuantizerMLP:
    """
    Simple MLP for rhythm quantization.
    Uses numpy only (no PyTorch dependency for inference).
    Supports both 2-layer and 3-layer architectures.
    """
    
    def __init__(self, hidden_size=128):
        self.hidden_size = hidden_size
        self.input_size = 8   # Features per note
        self.output_size = 10  # 6 note types + 2 dotted + 2 triplet
        
        # Initialize weights (will be loaded from trained model)
        self.weights = None
        self.initialized = False
        self.has_third_layer = False  # Will be set on load
    
    def _init_weights(self):
        """Initialize random weights (for training)."""
        np.random.seed(42)
        scale1 = np.sqrt(2.0 / self.input_size)
        scale2 = np.sqrt(2.0 / self.hidden_size)
        
        self.weights = {
            'W1': np.random.randn(self.input_size, self.hidden_size) * scale1,
            'b1': np.zeros(self.hidden_size),
            'W2': np.random.randn(self.hidden_size, self.hidden_size) * scale2,
            'b2': np.zeros(self.hidden_size),
            'W_type': np.random.randn(self.hidden_size, 6) * scale2,
            'b_type': np.zeros(6),
            'W_dotted': np.random.randn(self.hidden_size, 2) * scale2,
            'b_dotted': np.zeros(2),
            'W_triplet': np.random.randn(self.hidden_size, 2) * scale2,
            'b_triplet': np.zeros(2),
        }
        self.initialized = True
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (batch_size, input_size) or (input_size,)
        
        Returns:
            dict with 'note_type_probs', 'dotted_probs', 'triplet_probs'
        """
        if not self.initialized:
            self._init_weights()
        
        # Ensure 2D
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        # Layer 1
        h1 = self._relu(x @ self.weights['W1'] + self.weights['b1'])
        
        # Layer 2
        h2 = self._relu(h1 @ self.weights['W2'] + self.weights['b2'])
        
        # Layer 3 (if exists - from PyTorch trained model)
        if self.has_third_layer and 'W3' in self.weights:
            h2 = self._relu(h2 @ self.weights['W3'] + self.weights['b3'])
        
        # Output heads
        type_logits = h2 @ self.weights['W_type'] + self.weights['b_type']
        dotted_logits = h2 @ self.weights['W_dotted'] + self.weights['b_dotted']
        triplet_logits = h2 @ self.weights['W_triplet'] + self.weights['b_triplet']
        
        return {
            'note_type_probs': self._softmax(type_logits),
            'dotted_probs': self._softmax(dotted_logits),
            'triplet_probs': self._softmax(triplet_logits),
            'note_type_logits': type_logits,
            'dotted_logits': dotted_logits,
            'triplet_logits': triplet_logits,
        }
    
    def predict(self, x):
        """
        Predict quantized values.
        
        Returns:
            dict with 'note_type', 'dotted', 'is_triplet', 'confidence'
        """
        out = self.forward(x)
        
        type_idx = np.argmax(out['note_type_probs'], axis=-1)
        dotted = np.argmax(out['dotted_probs'], axis=-1) == 1
        triplet = np.argmax(out['triplet_probs'], axis=-1) == 1
        
        # Confidence = product of max probs
        type_conf = np.max(out['note_type_probs'], axis=-1)
        dotted_conf = np.max(out['dotted_probs'], axis=-1)
        triplet_conf = np.max(out['triplet_probs'], axis=-1)
        
        if np.isscalar(type_idx) or type_idx.ndim == 0:
            return {
                'note_type': IDX_TO_NOTE_TYPE[int(type_idx)],
                'dotted': bool(dotted),
                'is_triplet': bool(triplet),
                'confidence': float(type_conf * dotted_conf * triplet_conf),
                'beats': self._get_beats(int(type_idx), bool(dotted), bool(triplet))
            }
        else:
            return [{
                'note_type': IDX_TO_NOTE_TYPE[int(t)],
                'dotted': bool(d),
                'is_triplet': bool(tr),
                'confidence': float(tc * dc * trc),
                'beats': self._get_beats(int(t), bool(d), bool(tr))
            } for t, d, tr, tc, dc, trc in zip(
                type_idx, dotted, triplet, type_conf, dotted_conf, triplet_conf
            )]
    
    def _get_beats(self, type_idx, dotted, triplet):
        """Calculate beats for a note value."""
        base_beats = NOTE_TYPE_BEATS[IDX_TO_NOTE_TYPE[type_idx]]
        if dotted:
            base_beats *= 1.5
        if triplet:
            base_beats *= 2/3
        return base_beats
    
    def save(self, path):
        """Save model weights."""
        np.savez(path, **self.weights)
        print(f"Saved model to {path}")
    
    def load(self, path):
        """Load model weights."""
        data = np.load(path)
        self.weights = {key: data[key] for key in data.files}
        self.has_third_layer = 'W3' in self.weights
        self.initialized = True
        print(f"Loaded model from {path} (3-layer: {self.has_third_layer})")


def extract_features_for_ml(notes: List[Dict], bpm: float, use_ioi_as_duration: bool = True) -> np.ndarray:
    """
    Extract features for ML model from a list of notes.
    
    Args:
        notes: List of note dicts with 'time_seconds', 'duration_seconds', 'midi_note'
        bpm: Tempo in BPM
        use_ioi_as_duration: If True, use IOI instead of duration as primary signal.
                             This is better for audio where durations are unreliable.
    
    Returns:
        (n_notes, 8) array of features
    """
    if len(notes) == 0:
        return np.zeros((0, 8))
    
    beat_duration = 60.0 / bpm
    measure_duration = beat_duration * 4  # Assume 4/4
    
    features = []
    
    for i, note in enumerate(notes):
        onset = note.get('time_seconds', 0)
        duration = note.get('duration_seconds', 0.5)
        pitch = note.get('midi_note', 60)
        
        # IOI to next note
        if i < len(notes) - 1:
            ioi = notes[i + 1].get('time_seconds', onset) - onset
        else:
            ioi = duration
        
        # For audio inference, IOI is more reliable than duration
        # (audio durations are affected by reverb, pedal, overlapping notes)
        if use_ioi_as_duration:
            # Use IOI as the primary "duration" signal, keep actual duration as secondary
            primary_dur = ioi
            secondary_dur = duration
        else:
            # Original: use duration as primary (for MIDI training)
            primary_dur = duration
            secondary_dur = ioi
        
        # Duration in beats (primary signal)
        dur_beats = primary_dur / beat_duration
        
        # IOI in beats (secondary signal)
        ioi_beats = secondary_dur / beat_duration
        
        # Position in beat (0-1)
        beat_pos = (onset % beat_duration) / beat_duration
        
        # Position in measure (0-4 for 4/4)
        measure_pos = (onset % measure_duration) / beat_duration
        
        # Previous note IOI (context)
        if i > 0:
            prev_ioi = onset - notes[i - 1].get('time_seconds', 0)
            prev_ioi_beats = prev_ioi / beat_duration
        else:
            prev_ioi_beats = 1.0  # Default to quarter
        
        # Ratio of duration to IOI (key indicator!)
        dur_ioi_ratio = duration / max(ioi, 0.01)
        
        # Normalized pitch (relative to middle C)
        norm_pitch = (pitch - 60) / 40.0
        
        features.append([
            dur_beats,           # 0: Duration in beats (IOI when use_ioi_as_duration=True)
            ioi_beats,           # 1: Secondary duration signal
            beat_pos,            # 2: Position in beat
            measure_pos / 4.0,   # 3: Position in measure (normalized)
            prev_ioi_beats,      # 4: Previous IOI
            dur_ioi_ratio,       # 5: Duration/IOI ratio
            norm_pitch,          # 6: Normalized pitch
            1.0,                 # 7: Normalized tempo (fixed to 1.0 - model was trained only at 120 BPM)
        ])
    
    return np.array(features, dtype=np.float32)


def train_rhythm_model(training_data_path: str, 
                       epochs: int = 100,
                       learning_rate: float = 0.01,
                       batch_size: int = 32) -> RhythmQuantizerMLP:
    """
    Train the rhythm quantization model.
    
    Args:
        training_data_path: Path to JSONL training data
        epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
    
    Returns:
        Trained model
    """
    print(f"Loading training data from {training_data_path}...")
    
    # Load data
    X = []
    Y_type = []
    Y_dotted = []
    Y_triplet = []
    
    with open(training_data_path, 'r') as f:
        for line in f:
            item = json.loads(line.strip())
            
            # Pad input to 8 features if needed
            inp = item['input']
            while len(inp) < 8:
                inp.append(0.0)
            
            X.append(inp[:8])
            Y_type.append(item['output'][0])
            Y_dotted.append(item['output'][1])
            Y_triplet.append(item['output'][2])
    
    X = np.array(X, dtype=np.float32)
    Y_type = np.array(Y_type, dtype=np.int32)
    Y_dotted = np.array(Y_dotted, dtype=np.int32)
    Y_triplet = np.array(Y_triplet, dtype=np.int32)
    
    print(f"Loaded {len(X)} examples")
    print(f"Note type distribution: {np.bincount(Y_type, minlength=6)}")
    
    # Create model
    model = RhythmQuantizerMLP(hidden_size=64)
    model._init_weights()
    
    # Training loop (simple SGD)
    n_samples = len(X)
    
    for epoch in range(epochs):
        # Shuffle
        perm = np.random.permutation(n_samples)
        X_shuffled = X[perm]
        Y_type_shuffled = Y_type[perm]
        Y_dotted_shuffled = Y_dotted[perm]
        Y_triplet_shuffled = Y_triplet[perm]
        
        total_loss = 0
        n_batches = 0
        
        for i in range(0, n_samples, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            Y_type_batch = Y_type_shuffled[i:i+batch_size]
            Y_dotted_batch = Y_dotted_shuffled[i:i+batch_size]
            Y_triplet_batch = Y_triplet_shuffled[i:i+batch_size]
            
            # Forward pass
            out = model.forward(X_batch)
            
            # Compute loss (cross entropy)
            # For simplicity, just compute accuracy and update with gradient approx
            type_pred = np.argmax(out['note_type_probs'], axis=1)
            type_acc = np.mean(type_pred == Y_type_batch)
            
            # Simple gradient descent (numerical gradient approximation)
            # This is slow but works without autograd
            eps = 1e-4
            
            for key in model.weights:
                grad = np.zeros_like(model.weights[key])
                
                # Only update a subset of weights per batch (stochastic)
                if np.random.rand() > 0.5:
                    continue
                
                # Sample random indices to update
                flat_weights = model.weights[key].flatten()
                n_update = min(10, len(flat_weights))
                indices = np.random.choice(len(flat_weights), n_update, replace=False)
                
                for idx in indices:
                    # Compute numerical gradient
                    old_val = flat_weights[idx]
                    
                    flat_weights[idx] = old_val + eps
                    model.weights[key] = flat_weights.reshape(model.weights[key].shape)
                    out_plus = model.forward(X_batch)
                    loss_plus = -np.mean(np.log(out_plus['note_type_probs'][
                        np.arange(len(Y_type_batch)), Y_type_batch] + 1e-10))
                    
                    flat_weights[idx] = old_val - eps
                    model.weights[key] = flat_weights.reshape(model.weights[key].shape)
                    out_minus = model.forward(X_batch)
                    loss_minus = -np.mean(np.log(out_minus['note_type_probs'][
                        np.arange(len(Y_type_batch)), Y_type_batch] + 1e-10))
                    
                    grad_val = (loss_plus - loss_minus) / (2 * eps)
                    flat_weights[idx] = old_val - learning_rate * grad_val
                
                model.weights[key] = flat_weights.reshape(model.weights[key].shape)
            
            total_loss += (1 - type_acc)
            n_batches += 1
        
        if epoch % 10 == 0:
            # Evaluate on all data
            out = model.forward(X)
            type_pred = np.argmax(out['note_type_probs'], axis=1)
            dotted_pred = np.argmax(out['dotted_probs'], axis=1)
            triplet_pred = np.argmax(out['triplet_probs'], axis=1)
            
            type_acc = np.mean(type_pred == Y_type)
            dotted_acc = np.mean(dotted_pred == Y_dotted)
            triplet_acc = np.mean(triplet_pred == Y_triplet)
            
            print(f"Epoch {epoch}: type_acc={type_acc:.3f}, "
                  f"dotted_acc={dotted_acc:.3f}, triplet_acc={triplet_acc:.3f}")
    
    return model


# ============================================================================
# Integration with existing pipeline
# ============================================================================

_loaded_model: Optional[RhythmQuantizerMLP] = None

def load_rhythm_model(model_path: str = None) -> RhythmQuantizerMLP:
    """Load the rhythm model (singleton)."""
    global _loaded_model
    
    if _loaded_model is not None:
        return _loaded_model
    
    if model_path is None:
        model_path = os.path.join(os.path.dirname(__file__), 'rhythm_model.npz')
    
    if os.path.exists(model_path):
        model = RhythmQuantizerMLP()
        model.load(model_path)
        _loaded_model = model
        return model
    else:
        print(f"No trained model found at {model_path}")
        print("Using heuristic quantization as fallback")
        return None


def quantize_with_ml(notes: List[Dict], bpm: float, 
                     model: RhythmQuantizerMLP = None) -> List[Dict]:
    """
    Quantize notes using the ML model.
    
    Args:
        notes: List of note dicts
        bpm: Tempo
        model: Optional pre-loaded model
    
    Returns:
        Notes with updated 'note_value', 'dotted', 'is_triplet' fields
    """
    if model is None:
        model = load_rhythm_model()
    
    if model is None:
        # Fallback to heuristic
        return notes
    
    # Extract features
    features = extract_features_for_ml(notes, bpm)
    
    if len(features) == 0:
        return notes
    
    # Predict
    predictions = model.predict(features)
    
    # Update notes
    for i, (note, pred) in enumerate(zip(notes, predictions)):
        note['note_value'] = pred['note_type']
        note['dotted'] = pred['dotted']
        note['is_triplet'] = pred['is_triplet']
        note['note_divisions'] = pred['beats']
        note['quantization_method'] = 'ml'
        note['quantization_confidence'] = pred['confidence']
    
    return notes


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Rhythm quantization model')
    parser.add_argument('--train', '-t', help='Path to training data (JSONL)')
    parser.add_argument('--output', '-o', default='rhythm_model.npz',
                        help='Output path for trained model')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--test', help='Test model on a note sequence')
    
    args = parser.parse_args()
    
    if args.train:
        model = train_rhythm_model(args.train, epochs=args.epochs)
        model.save(args.output)
    
    elif args.test:
        model = load_rhythm_model(args.test)
        if model:
            # Test with some example inputs
            test_features = np.array([
                [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],  # Quarter note on beat
                [0.5, 0.5, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],  # Eighth note
                [0.33, 0.33, 0.0, 0.0, 0.33, 1.0, 0.0, 1.0],  # Triplet eighth
            ])
            
            for i, feat in enumerate(test_features):
                pred = model.predict(feat)
                print(f"Test {i+1}: {pred}")
    
    else:
        parser.print_help()
