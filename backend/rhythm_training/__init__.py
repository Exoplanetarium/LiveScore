"""
Rhythm Training Module

This module provides ML-based rhythm quantization to replace heuristic approaches.

Quick Start:
    1. Download training data:
       python prepare_training_data.py --download --process
    
    2. Train the model:
       python prepare_training_data.py --train
    
    3. Use in your code:
       from rhythm_training import quantize_with_ml
       notes = quantize_with_ml(notes, bpm)
"""

from .evaluate_rhythm import (compare_transcription_to_ground_truth,
                              evaluate_rhythm_accuracy, load_midi_ground_truth)
from .rhythm_model import (RhythmQuantizerMLP, extract_features_for_ml,
                           load_rhythm_model, quantize_with_ml)

__all__ = [
    'RhythmQuantizerMLP',
    'load_rhythm_model', 
    'quantize_with_ml',
    'extract_features_for_ml',
    'load_midi_ground_truth',
    'compare_transcription_to_ground_truth',
    'evaluate_rhythm_accuracy',
]
