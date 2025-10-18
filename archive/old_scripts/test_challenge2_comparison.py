#!/usr/bin/env python3
"""
Compare CompactExternalizingCNN vs TCN for Challenge 2
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Import models
import sys
sys.path.append('src')
sys.path.append('improvements')

from all_improvements import TCN_EEG

# CompactExternalizingCNN definition
class CompactExternalizingCNN(nn.Module):
    """Compact CNN for externalizing prediction"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(96),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.regressor = nn.Sequential(
            nn.Linear(96, 48),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(48, 24),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Linear(24, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.regressor(features)
        return output

print("="*80)
print("🔬 Challenge 2 Model Comparison")
print("="*80)
print()

# 1. Load CompactCNN
print("📦 Model 1: CompactExternalizingCNN")
compact_cnn = CompactExternalizingCNN()
try:
    compact_cnn.load_state_dict(
        torch.load('weights_challenge_2_multi_release.pt', 
                   map_location='cpu', weights_only=False)
    )
    print("   ✅ Loaded weights: weights_challenge_2_multi_release.pt")
    compact_params = sum(p.numel() for p in compact_cnn.parameters())
    print(f"   Parameters: {compact_params:,}")
    print(f"   Architecture: 4-layer CNN with BatchNorm")
    print(f"   Reported Val NRMSE: 0.2917")
except Exception as e:
    print(f"   ❌ Error loading: {e}")
    compact_cnn = None

print()

# 2. Load TCN
print("📦 Model 2: TCN_EEG")
tcn_model = TCN_EEG(
    num_channels=129,
    num_outputs=1,
    num_filters=48,
    kernel_size=7,
    dropout=0.3,
    num_levels=5
)
try:
    checkpoint = torch.load('checkpoints/challenge2_tcn_competition_best.pth',
                           map_location='cpu', weights_only=False)
    if 'model_state_dict' in checkpoint:
        tcn_model.load_state_dict(checkpoint['model_state_dict'])
        print("   ✅ Loaded checkpoint: challenge2_tcn_competition_best.pth")
        print(f"   Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"   Val Loss: {checkpoint.get('val_loss', 'N/A'):.6f}")
    else:
        tcn_model.load_state_dict(checkpoint)
        print("   ✅ Loaded weights")
    tcn_params = sum(p.numel() for p in tcn_model.parameters())
    print(f"   Parameters: {tcn_params:,}")
    print(f"   Architecture: 5-layer TCN with dilated convolutions")
except Exception as e:
    print(f"   ❌ Error loading: {e}")
    tcn_model = None

print()
print("="*80)
print("🧪 Testing with Dummy Data")
print("="*80)
print()

# Test with dummy data
batch_size = 4
dummy_eeg = torch.randn(batch_size, 129, 200)
print(f"Input shape: {dummy_eeg.shape} (batch, channels, time)")
print()

# Test CompactCNN
if compact_cnn is not None:
    compact_cnn.eval()
    with torch.no_grad():
        output = compact_cnn(dummy_eeg)
    print(f"CompactCNN output shape: {output.shape}")
    print(f"CompactCNN predictions: {output.squeeze().numpy()}")
    print(f"CompactCNN range: [{output.min():.3f}, {output.max():.3f}]")
    print()

# Test TCN
if tcn_model is not None:
    tcn_model.eval()
    with torch.no_grad():
        output = tcn_model(dummy_eeg)
    print(f"TCN output shape: {output.shape}")
    print(f"TCN predictions: {output.squeeze().numpy()}")
    print(f"TCN range: [{output.min():.3f}, {output.max():.3f}]")
    print()

print("="*80)
print("📊 Comparison Summary")
print("="*80)
print()

if compact_cnn and tcn_model:
    print("✅ Both models loaded successfully!")
    print()
    print("Model Comparison:")
    print(f"  CompactCNN: {compact_params:,} params, Val NRMSE 0.2917")
    print(f"  TCN:        {tcn_params:,} params, Val Loss 0.667792")
    print()
    print("Recommendation:")
    print("  • Use CompactCNN for submission v6 (proven baseline)")
    print("  • TCN needs proper validation on actual test set")
    print("  • Compare on leaderboard to determine best model")
else:
    print("⚠️  Could not load both models for comparison")

print()
print("="*80)
