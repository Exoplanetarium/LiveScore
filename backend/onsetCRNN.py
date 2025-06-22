import glob
import itertools
import json
import os

import librosa
import numpy as np
import pretty_midi
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset


# finding all .mid / .wav pairs
class MAPSDataset(Dataset):
    """
    PyTorch Dataset for MAPS audio+MIDI onset labels.
    Expects structure:
      MAPS_ROOT/ENSTDkCl/ISOL/...
        *.wav
        corresponding *.mid
    """
    def __init__(self, maps_root, split='train', sr=44100, hop_length=512, n_mels=128):
        self.sr = sr
        self.hop_length = hop_length
        self.n_mels = n_mels
        
        print(f"Looking for files in: {maps_root}")
        print(f"Absolute path: {os.path.abspath(maps_root)}")
        
        # Check if the directory exists
        if not os.path.exists(maps_root):
            print(f"ERROR: Directory {maps_root} does not exist!")
            return
        
        # Use os.walk to properly traverse all subdirectories
        self.pairs = []
        wav_count = 0
        for root, dirs, files in os.walk(maps_root):
            print(f"Searching in: {root}")
            print(f"  Subdirs: {dirs[:5]}...")  # Show first 5 subdirs
            print(f"  Files: {len(files)} total")
            
            wav_files = [f for f in files if f.endswith('.wav')]
            mid_files = [f for f in files if f.endswith('.mid')]
            print(f"  .wav files: {len(wav_files)}")
            print(f"  .mid files: {len(mid_files)}")
            
            for file in files:
                if file.endswith('.wav'):
                    wav_count += 1
                    wav_path = os.path.join(root, file)
                    mid_path = wav_path[:-4] + '.mid'
                    if os.path.exists(mid_path):
                        self.pairs.append((wav_path, mid_path))
                        print(f"  ✓ Found pair: {file}")
                    else:
                        print(f"  ✗ Missing .mid for: {file}")
        
        print(f"Total .wav files found: {wav_count}")
        print(f"Found {len(self.pairs)} valid wav/mid pairs")
        
        if len(self.pairs) == 0:
            print("ERROR: No valid pairs found!")
            print(f"Check that {maps_root} contains .wav and .mid files")
            return

        cutoff = int(0.8 * len(self.pairs))
        if split == 'train':
            self.pairs = self.pairs[:cutoff]
        else:
            self.pairs = self.pairs[cutoff:]
        
        print(f"After split '{split}': {len(self.pairs)} pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        wav_path, midi_path = self.pairs[idx]
        audio, _ = librosa.load(wav_path, sr=self.sr)
        # compute mel-spectrogram
        mel = librosa.feature.melspectrogram(y=audio, sr=self.sr,
                                             n_fft=2048,
                                             hop_length=self.hop_length,
                                             n_mels=self.n_mels)
        mel_db = librosa.power_to_db(S=mel, ref=np.max)
        # normalize
        mel_norm = (mel_db + 80) / 80
        # onset detection
        pm = pretty_midi.PrettyMIDI(midi_path)
        onset_frames = []
        for inst in pm.instruments:
            for note in inst.notes:
                t = note.start
                f = int(np.round(t * self.sr / self.hop_length))
                onset_frames.append(f)
        # create label vector
        T = mel_norm.shape[1]
        label = np.zeros(T, dtype=np.float32)
        for f in onset_frames:
            if 0 <= f < T:
                label[f] = 1.0        # to torch tensors
        x = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)  # (1, n_mels, T)
        y = torch.tensor(label, dtype=torch.float32)
        return x, y

# pad spectrogram outputs to same elapsed time
def collate_fn(batch):
    x_batch, y_batch = zip(*batch)
    
    # Find max time dimension
    max_time = max(x.shape[2] for x in x_batch)
    
    # Pad mel-spectrograms
    x_padded = []
    y_padded = []
    
    for x, y in zip(x_batch, y_batch):
        # Pad mel-spectrogram: (1, n_mels, T) -> (1, n_mels, max_time)
        pad_amount = max_time - x.shape[2]
        if pad_amount > 0:
            x_pad = torch.nn.functional.pad(x, (0, pad_amount), value=0.0)
            y_pad = torch.nn.functional.pad(y, (0, pad_amount), value=0.0)
        else:
            x_pad = x
            y_pad = y
        
        x_padded.append(x_pad)
        y_padded.append(y_pad)
    
    # Stack into batch
    x_batch = torch.stack(x_padded)
    y_batch = torch.stack(y_padded)
    
    return x_batch, y_batch

# onset detection using CRNN (convolutional recurrent neural network)
class CRNNOnsetModel(nn.Module):
    def __init__(self, n_mels=128, hidden_dim=64):
        super().__init__()
        self.n_mels = n_mels
        self.hidden_dim = hidden_dim
        
        self.conv1 = nn.Conv2d(1, 16, (3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, (3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        # Calculate the actual input size after convolutions
        # After conv layers: 32 channels, n_mels height (unchanged due to padding=1)
        gru_input_size = 32 * n_mels
        
        self.gru = nn.GRU(input_size=gru_input_size,
                          hidden_size=hidden_dim,
                          batch_first=True,
                          bidirectional=True)
        self.fc = nn.Linear(hidden_dim*2, 1)

    def forward(self, x):
        # x: (batch, 1, n_mels, T)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        b, c, m, t = x.size()
        
        # Reshape for GRU: (batch, time, features)
        x = x.permute(0, 3, 1, 2).reshape(b, t, -1)
        
        out, _ = self.gru(x)
        out = self.fc(out).squeeze(-1)
        return torch.sigmoid(out)  # (batch, T)

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(x)
        loss = F.binary_cross_entropy(preds, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def eval_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = F.binary_cross_entropy(preds, y)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader, device, threshold=0.5):
    """Evaluate model with onset detection metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            
            # Apply threshold and flatten
            pred_binary = (preds > threshold).float()
            
            # Flatten and convert to numpy
            pred_flat = pred_binary.cpu().numpy().flatten()
            label_flat = y.cpu().numpy().flatten()
            
            all_preds.extend(pred_flat)
            all_labels.extend(label_flat)
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary', zero_division=0
    )
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Count onset statistics
    true_onsets = np.sum(all_labels)
    pred_onsets = np.sum(all_preds)
    correct_onsets = np.sum((all_labels == 1) & (all_preds == 1))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_onsets': int(true_onsets),
        'pred_onsets': int(pred_onsets),
        'correct_onsets': int(correct_onsets)
    }

def test_model(model_path, test_loader, device):
    model = CRNNOnsetModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    print("Testing trained model...")
    metrics = evaluate_model(model, test_loader, device)
    
    print(f"\n--- TEST RESULTS ---")
    print(f"Accuracy:  {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1-Score:  {metrics['f1']:.3f}")
    print(f"\nOnset Statistics:")
    print(f"True onsets:    {metrics['true_onsets']}")
    print(f"Predicted:      {metrics['pred_onsets']}")
    print(f"Correct:        {metrics['correct_onsets']}")
    
    return metrics

def train_model_with_params(maps_root, device, params, epochs=10):
    """Train a model with specific hyperparameters"""
    # Create datasets with the specific n_mels parameter
    train_ds = MAPSDataset(maps_root, split='train', n_mels=params['n_mels'])
    val_ds = MAPSDataset(maps_root, split='val', n_mels=params['n_mels'])
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=8, collate_fn=collate_fn)
    
    model = CRNNOnsetModel(
        n_mels=params['n_mels'], 
        hidden_dim=params['hidden_dim']
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    
    best_f1 = 0.0
    best_epoch = 0
    
    print(f"  Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        print(f"    Epoch {epoch}/{epochs} - Training...", end="", flush=True)
        train_loss = train_epoch(model, train_loader, optimizer, device)
        print(f" Validating...", end="", flush=True)
        val_loss = eval_epoch(model, val_loader, device)
        print(f" Done. Loss: {train_loss:.4f}/{val_loss:.4f}")
        
        # Evaluate every 3 epochs or at the end
        if epoch % 3 == 0 or epoch == epochs:
            print(f"    Evaluating metrics...", end="", flush=True)
            val_metrics = evaluate_model(model, val_loader, device, threshold=params['threshold'])
            print(f" F1: {val_metrics['f1']:.3f}")
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_epoch = epoch
                # Save this configuration's best model
                torch.save(model.state_dict(), f'temp_model_{hash(str(params))}.pth')
                print(f"    ★ New best F1 for this config: {best_f1:.3f}")
    
    return best_f1, best_epoch

def grid_search(maps_root, device, param_grid, epochs=10):
    """Perform grid search over hyperparameters"""
    print("Starting hyperparameter grid search...")
    print(f"Total combinations: {len(list(itertools.product(*param_grid.values())))}")
    
    results = []
    best_score = 0.0
    best_params = None
    
    # Generate all combinations
    param_names = list(param_grid.keys())
    param_combinations = itertools.product(*param_grid.values())
    
    for i, param_values in enumerate(param_combinations):
        params = dict(zip(param_names, param_values))
        
        print(f"\n--- Configuration {i+1} ---")
        print(f"Parameters: {params}")
        
        try:
            f1_score, best_epoch = train_model_with_params(
                maps_root, device, params, epochs
            )
            
            result = {
                'params': params,
                'f1_score': f1_score,
                'best_epoch': best_epoch
            }
            results.append(result)
            
            print(f"Best F1: {f1_score:.3f} (epoch {best_epoch})")
            
            # Track overall best
            if f1_score > best_score:
                best_score = f1_score
                best_params = params.copy()
                # Save the best model with a permanent name
                torch.save(torch.load(f'temp_model_{hash(str(params))}.pth'), 
                          'best_gridsearch_model.pth')
                print(f"★ NEW BEST OVERALL: {best_score:.3f}")
            
        except Exception as e:
            print(f"Error with configuration {params}: {e}")
            result = {
                'params': params,
                'f1_score': 0.0,
                'best_epoch': 0,
                'error': str(e)
            }
            results.append(result)
    
    # Clean up temporary files
    import glob
    temp_files = glob.glob('temp_model_*.pth')
    for f in temp_files:
        try:
            os.remove(f)
        except:
            pass
    
    # Save results
    with open('grid_search_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print("GRID SEARCH SUMMARY")
    print(f"{'='*50}")
    
    # Sort results by F1 score
    results_sorted = sorted(results, key=lambda x: x['f1_score'], reverse=True)
    print(f"\nTop 5 configurations:")
    for i, result in enumerate(results_sorted[:5]):
        print(f"{i+1}. F1: {result['f1_score']:.3f} | {result['params']}")
    
    print(f"\nBest configuration:")
    print(f"F1 Score: {best_score:.3f}")
    print(f"Parameters: {best_params}")
    print(f"Model saved as: 'best_gridsearch_model.pth'")
    print(f"Results saved as: 'grid_search_results.json'")
    
    return best_params, best_score, results

def inspect_saved_model(model_path):
    """Inspect a saved model to understand its architecture and parameters"""
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found!")
        return
    
    # Load the state dict to examine the model structure
    state_dict = torch.load(model_path, map_location='cpu')
    
    print(f"Model file: {model_path}")
    print(f"File size: {os.path.getsize(model_path) / 1024:.1f} KB")
    print("\nModel layers and parameter shapes:")
    
    total_params = 0
    for name, tensor in state_dict.items():
        params = tensor.numel()
        total_params += params
        print(f"  {name}: {tuple(tensor.shape)} ({params:,} parameters)")
    
    print(f"\nTotal parameters: {total_params:,}")
    
    # Try to infer n_mels from GRU input size
    if 'gru.weight_ih_l0' in state_dict:
        gru_input_size = state_dict['gru.weight_ih_l0'].shape[1]
        inferred_n_mels = gru_input_size // 32  # Since we use 32 channels
        print(f"\nInferred model architecture:")
        print(f"  n_mels: {inferred_n_mels}")
        print(f"  GRU input size: {gru_input_size}")
    
    if 'gru.weight_hh_l0' in state_dict:
        hidden_dim = state_dict['gru.weight_hh_l0'].shape[1] // 3  # GRU has 3 gates
        print(f"  hidden_dim: {hidden_dim}")
    
    # Check if we can load the actual results file
    results_file = 'grid_search_results.json'
    if os.path.exists(results_file):
        print(f"\nFound {results_file}. Loading best configuration...")
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Find the best result
        best_result = max(results, key=lambda x: x.get('f1_score', 0))
        print(f"Best configuration from grid search:")
        print(f"  F1 Score: {best_result['f1_score']:.3f}")
        print(f"  Parameters: {best_result['params']}")
        print(f"  Best epoch: {best_result['best_epoch']}")
    else:
        print(f"\n{results_file} not found. Cannot show exact hyperparameters used.")

def main():
    maps_root = os.path.join(os.path.dirname(__file__), 'dataset', 'AkPnBcht', 'ISOL')  # Path to specific MAPS subset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    # prepare data
    train_ds = MAPSDataset(maps_root, split='train')
    val_ds   = MAPSDataset(maps_root, split='val')
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=8, collate_fn=collate_fn)

    # model, optimizer
    model = CRNNOnsetModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)    # training loop
    best_f1 = 0.0
    for epoch in range(1, 11):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_loss   = eval_epoch(model, val_loader, device)
        
        # Evaluate with metrics every few epochs
        if epoch % 3 == 0 or epoch == 10:
            val_metrics = evaluate_model(model, val_loader, device)
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.3f}, Precision: {val_metrics['precision']:.3f}, Recall: {val_metrics['recall']:.3f}")
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                torch.save(model.state_dict(), 'best_crnn_onset.pth')
                print(f"  New best F1: {best_f1:.3f} - model saved!")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

    # Final evaluation on validation set
    print(f"\n--- FINAL VALIDATION RESULTS ---")
    final_metrics = evaluate_model(model, val_loader, device)
    for key, value in final_metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.3f}")
        else:
            print(f"{key}: {value}")

    # save final model
    torch.save(model.state_dict(), 'crnn_onset_final.pth')
    print(f"\nFinal model saved as 'crnn_onset_final.pth'")
    print(f"Best model saved as 'best_crnn_onset.pth' (F1: {best_f1:.3f})")

if __name__ == '__main__':
    import sys
    maps_root = os.path.join(os.path.dirname(__file__), 'dataset', 'AkPnBcht', 'RAND')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        # Test mode - load saved model and evaluate
        print("Test mode activated")
        
        # Create test dataset (use validation split)
        test_ds = MAPSDataset(maps_root, split='val')
        test_loader = DataLoader(test_ds, batch_size=8, collate_fn=collate_fn)
          # Test the best model
        if os.path.exists('best_crnn_onset.pth'):
            test_model('best_crnn_onset.pth', test_loader, device)
        else:
            print("No saved model found! Train the model first.")
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'inspect':
        # Inspect saved model
        model_file = 'best_gridsearch_model.pth' if len(sys.argv) <= 2 else sys.argv[2]
        inspect_saved_model(model_file)
        
    elif len(sys.argv) > 1 and sys.argv[1] == 'gridsearch':
        # Grid search mode
        print("Grid search mode activated")
        
        # Define hyperparameter grid
        param_grid = {
            'n_mels': [64, 128],
            'hidden_dim': [32, 64, 128],
            'lr': [1e-4, 1e-3, 5e-3],
            'threshold': [0.3, 0.5, 0.7]
        }
        
        print("Hyperparameter Grid:")
        for key, values in param_grid.items():
            print(f"  {key}: {values}")
        
        # Run grid search
        best_params, best_score, all_results = grid_search(
            maps_root, device, param_grid, epochs=12
        )
        
    else:
        # Training mode
        main()