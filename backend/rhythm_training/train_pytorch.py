"""
Fast PyTorch training for rhythm quantization model.
Uses GPU acceleration for training on large datasets.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, IterableDataset

# Note type mappings
NOTE_TYPES = ['whole', 'half', 'quarter', 'eighth', '16th', '32nd']
NOTE_TYPE_TO_IDX = {nt: i for i, nt in enumerate(NOTE_TYPES)}
IDX_TO_NOTE_TYPE = {i: nt for i, nt in enumerate(NOTE_TYPES)}


class RhythmDataset(IterableDataset):
    """
    Streaming dataset for large JSONL files.
    Doesn't load everything into memory.
    """
    
    def __init__(self, path: str, shuffle_buffer: int = 10000):
        self.path = path
        self.shuffle_buffer = shuffle_buffer
        
        # Count lines for length estimation
        self._len = None
    
    def __iter__(self):
        buffer = []
        
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    
                    # Pad input to 8 features
                    inp = item['input']
                    while len(inp) < 8:
                        inp.append(0.0)
                    
                    x = torch.tensor(inp[:8], dtype=torch.float32)
                    y_type = torch.tensor(item['output'][0], dtype=torch.long)
                    y_dotted = torch.tensor(item['output'][1], dtype=torch.long)
                    y_triplet = torch.tensor(item['output'][2], dtype=torch.long)
                    
                    buffer.append((x, y_type, y_dotted, y_triplet))
                    
                    # Shuffle buffer when full
                    if len(buffer) >= self.shuffle_buffer:
                        np.random.shuffle(buffer)
                        for item in buffer:
                            yield item
                        buffer = []
                        
                except json.JSONDecodeError:
                    continue
        
        # Yield remaining items
        if buffer:
            np.random.shuffle(buffer)
            for item in buffer:
                yield item


class RhythmNet(nn.Module):
    """
    Neural network for rhythm quantization.
    Simple but effective architecture.
    """
    
    def __init__(self, hidden_size=128, dropout=0.2):
        super().__init__()
        
        self.input_size = 8
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(self.input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
        )
        
        # Output heads
        self.head_type = nn.Linear(hidden_size // 2, 6)    # 6 note types
        self.head_dotted = nn.Linear(hidden_size // 2, 2)  # Binary
        self.head_triplet = nn.Linear(hidden_size // 2, 2)  # Binary
    
    def forward(self, x):
        h = self.shared(x)
        return {
            'type': self.head_type(h),
            'dotted': self.head_dotted(h),
            'triplet': self.head_triplet(h),
        }


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_type_correct = 0
    total_dotted_correct = 0
    total_triplet_correct = 0
    total_samples = 0
    
    for batch_idx, (x, y_type, y_dotted, y_triplet) in enumerate(dataloader):
        x = x.to(device)
        y_type = y_type.to(device)
        y_dotted = y_dotted.to(device)
        y_triplet = y_triplet.to(device)
        
        optimizer.zero_grad()
        
        out = model(x)
        
        # Multi-task loss
        loss_type = criterion(out['type'], y_type)
        loss_dotted = criterion(out['dotted'], y_dotted)
        loss_triplet = criterion(out['triplet'], y_triplet)
        
        # Weight note type more heavily (most important)
        loss = loss_type * 2.0 + loss_dotted + loss_triplet
        
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * x.size(0)
        total_type_correct += (out['type'].argmax(1) == y_type).sum().item()
        total_dotted_correct += (out['dotted'].argmax(1) == y_dotted).sum().item()
        total_triplet_correct += (out['triplet'].argmax(1) == y_triplet).sum().item()
        total_samples += x.size(0)
        
        if batch_idx % 1000 == 0 and batch_idx > 0:
            print(f"  Batch {batch_idx}: loss={total_loss/total_samples:.4f}, "
                  f"type_acc={total_type_correct/total_samples:.3f}, "
                  f"dotted_acc={total_dotted_correct/total_samples:.3f}")
    
    return {
        'loss': total_loss / total_samples,
        'type_acc': total_type_correct / total_samples,
        'dotted_acc': total_dotted_correct / total_samples,
        'triplet_acc': total_triplet_correct / total_samples,
    }


def export_to_numpy(model, save_path):
    """Export PyTorch model to numpy format for inference without PyTorch."""
    state = model.state_dict()
    
    # Map to the format expected by RhythmQuantizerMLP
    numpy_weights = {
        'W1': state['shared.0.weight'].cpu().numpy().T,
        'b1': state['shared.0.bias'].cpu().numpy(),
        'W2': state['shared.3.weight'].cpu().numpy().T,
        'b2': state['shared.3.bias'].cpu().numpy(),
        'W3': state['shared.6.weight'].cpu().numpy().T,  # Extra layer
        'b3': state['shared.6.bias'].cpu().numpy(),
        'W_type': state['head_type.weight'].cpu().numpy().T,
        'b_type': state['head_type.bias'].cpu().numpy(),
        'W_dotted': state['head_dotted.weight'].cpu().numpy().T,
        'b_dotted': state['head_dotted.bias'].cpu().numpy(),
        'W_triplet': state['head_triplet.weight'].cpu().numpy().T,
        'b_triplet': state['head_triplet.bias'].cpu().numpy(),
    }
    
    np.savez(save_path, **numpy_weights)
    print(f"Exported to {save_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='rhythm_training_data.jsonl',
                        help='Path to training data JSONL')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--hidden', type=int, default=128,
                        help='Hidden layer size')
    parser.add_argument('--device', default='cuda',
                        help='Device (cuda or cpu)')
    parser.add_argument('--output', default='rhythm_model.pt',
                        help='Output model path')
    args = parser.parse_args()
    
    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create dataset and dataloader
    print(f"Loading data from {args.data}...")
    dataset = RhythmDataset(args.data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    
    # Create model
    model = RhythmNet(hidden_size=args.hidden).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 40)
        
        metrics = train_epoch(model, dataloader, optimizer, criterion, device)
        
        print(f"Epoch {epoch + 1} complete:")
        print(f"  Loss: {metrics['loss']:.4f}")
        print(f"  Type Accuracy: {metrics['type_acc']:.3%}")
        print(f"  Dotted Accuracy: {metrics['dotted_acc']:.3%}")
        print(f"  Triplet Accuracy: {metrics['triplet_acc']:.3%}")
        
        # Save if best
        if metrics['type_acc'] > best_acc:
            best_acc = metrics['type_acc']
            torch.save(model.state_dict(), args.output)
            print(f"  Saved new best model!")
    
    # Export to numpy format
    numpy_path = args.output.replace('.pt', '.npz')
    export_to_numpy(model, numpy_path)
    
    print(f"\nTraining complete!")
    print(f"Best type accuracy: {best_acc:.3%}")
    print(f"Models saved to: {args.output} and {numpy_path}")


if __name__ == '__main__':
    main()
