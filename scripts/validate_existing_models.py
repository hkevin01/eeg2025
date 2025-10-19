"""
Validate Existing Models - Quick Performance Check
Test Challenge 1 and Challenge 2 models on validation data
"""
import sys
from pathlib import Path
sys.path.append('src')

import torch
import torch.nn as nn
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

print("=" * 80)
print("🔍 VALIDATING EXISTING MODELS")
print("=" * 80)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

# ============================================================================
# Challenge 1: Response Time Prediction
# ============================================================================

class Challenge1ValidationDataset(Dataset):
    """Load Challenge 1 validation data from HDF5"""
    
    def __init__(self, hdf5_files, train_ratio=0.8):
        self.samples = []
        
        for hdf5_path in hdf5_files:
            with h5py.File(hdf5_path, 'r') as f:
                eeg = f['eeg'][:]
                labels = f['labels'][:]
                
                # Use last 20% as validation
                n_samples = len(eeg)
                val_start = int(n_samples * train_ratio)
                
                for i in range(val_start, n_samples):
                    self.samples.append((eeg[i], labels[i]))
        
        print(f"Validation samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        eeg, label = self.samples[idx]
        return torch.FloatTensor(eeg), torch.FloatTensor([label])


def calculate_nrmse(predictions, targets):
    """Calculate Normalized RMSE"""
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    target_range = targets.max() - targets.min()
    nrmse = rmse / target_range if target_range > 0 else rmse
    return nrmse, rmse


def validate_challenge1_model(model_path, dataset):
    """Validate Challenge 1 model"""
    print(f"\n{'=' * 80}")
    print(f"Testing: {model_path.name}")
    print(f"{'=' * 80}")
    
    try:
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        
        # Check if it's a full checkpoint or just state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            print(f"Checkpoint info:")
            if 'epoch' in checkpoint:
                print(f"  Epoch: {checkpoint['epoch']}")
            if 'val_nrmse' in checkpoint:
                print(f"  Saved val NRMSE: {checkpoint['val_nrmse']:.4f}")
            if 'train_nrmse' in checkpoint:
                print(f"  Saved train NRMSE: {checkpoint['train_nrmse']:.4f}")
            
            # Try to load model - we need to know architecture
            print("\n⚠️  Model architecture not stored, cannot validate")
            print("   (TCN checkpoint format doesn't include model class)")
            return None
        
        else:
            print("⚠️  State dict only, need model architecture")
            return None
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def test_challenge1_weights():
    """Test the weights_challenge_1_multi_release.pt model"""
    print(f"\n{'=' * 80}")
    print("Testing: weights_challenge_1_multi_release.pt")
    print(f"{'=' * 80}")
    
    model_path = Path('weights_challenge_1_multi_release.pt')
    
    if not model_path.exists():
        print("❌ Model not found")
        return None
    
    try:
        weights = torch.load(model_path, map_location=device)
        
        print(f"\nModel info:")
        print(f"  Type: {type(weights)}")
        if isinstance(weights, dict):
            print(f"  Keys: {list(weights.keys())[:10]}")
            
            # Try to infer architecture from keys
            if any('conv' in k for k in weights.keys()):
                print("  Architecture: Likely CNN-based")
            if any('tcn' in k.lower() for k in weights.keys()):
                print("  Architecture: Likely TCN-based")
        
        print("\n⚠️  Need to match exact architecture to test")
        return None
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# ============================================================================
# Challenge 2: Externalizing Behavior
# ============================================================================

def test_challenge2_model():
    """Test Challenge 2 model"""
    print(f"\n{'=' * 80}")
    print("CHALLENGE 2: Externalizing Behavior")
    print(f"{'=' * 80}")
    
    model_path = Path('weights_challenge_2_multi_release.pt')
    
    if not model_path.exists():
        print("❌ Model not found")
        return None
    
    try:
        weights = torch.load(model_path, map_location=device)
        
        print(f"\nModel info:")
        print(f"  Size: {model_path.stat().st_size / 1024:.1f} KB")
        print(f"  Type: {type(weights)}")
        
        if isinstance(weights, dict):
            if 'model_state_dict' in weights:
                print(f"  Contains: model_state_dict")
                if 'metadata' in weights:
                    print(f"  Metadata: {weights['metadata']}")
            else:
                print(f"  Keys: {list(weights.keys())[:10]}")
        
        print("\n✅ Model file valid")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None


# ============================================================================
# Main Validation
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("CHALLENGE 1: Response Time Prediction")
    print("=" * 80)
    
    # Load validation data
    print("\n📂 Loading validation data...")
    hdf5_files = [
        Path('data/cached/challenge1_R1_windows.h5'),
        Path('data/cached/challenge1_R2_windows.h5'),
        Path('data/cached/challenge1_R3_windows.h5'),
    ]
    
    existing_files = [f for f in hdf5_files if f.exists()]
    
    if not existing_files:
        print("❌ No HDF5 files found!")
        return
    
    print(f"Found {len(existing_files)} files")
    dataset = Challenge1ValidationDataset(existing_files)
    
    # Test model 1: TCN competition best
    model1_path = Path('checkpoints/challenge1_tcn_competition_best.pth')
    if model1_path.exists():
        validate_challenge1_model(model1_path, dataset)
    else:
        print(f"\n❌ {model1_path} not found")
    
    # Test model 2: Multi-release weights
    test_challenge1_weights()
    
    # Test Challenge 2
    test_challenge2_model()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 VALIDATION SUMMARY")
    print("=" * 80)
    print("\n⚠️  ISSUE: Cannot fully validate without model architecture")
    print("\nTo properly test, we need:")
    print("  1. Model class definition (TCN architecture)")
    print("  2. Exact hyperparameters used during training")
    print("  3. Input preprocessing details")
    print("\nRECOMMENDATION:")
    print("  • Check training scripts for model architecture")
    print("  • Look for saved model configs")
    print("  • Or: Fix & train new models with full validation")


if __name__ == "__main__":
    main()
