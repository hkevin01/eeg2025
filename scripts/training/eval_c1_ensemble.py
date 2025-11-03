#!/usr/bin/env python3
"""
Evaluate C1 Ensemble - 5 EMA Models
Target: < 0.95 NRMSE
"""
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

print("=" * 60)
print("🔮 C1 Ensemble Evaluation")
print("=" * 60)

# Load V10 architecture
class CompactResponseTimeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(129, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        features = self.features(x)
        output = self.regressor(features)
        return output.squeeze(-1)

def nrmse(pred, target):
    return np.sqrt(np.mean((pred - target) ** 2)) / (np.std(target) + 1e-8)

# Load validation data
print("\n📥 Loading validation data...")
files = [f'data/cached/challenge1_R{i}_windows.h5' for i in range(1, 5)]
all_data, all_labels = [], []

for path in files:
    with h5py.File(path, 'r') as f:
        all_data.append(f['eeg'][:].astype(np.float32))
        all_labels.append(f['labels'][:].astype(np.float32))

all_data = np.concatenate(all_data, axis=0)
all_labels = np.concatenate(all_labels, axis=0)

# Normalize
for ch in range(129):
    m, s = all_data[:, ch].mean(), all_data[:, ch].std()
    if s > 0:
        all_data[:, ch] = (all_data[:, ch] - m) / s

# Split (same as training)
np.random.seed(42)
indices = np.arange(len(all_data))
np.random.shuffle(indices)
val_idx = indices[int(0.8 * len(indices)):]

X_val = torch.from_numpy(all_data[val_idx])
y_val = all_labels[val_idx]

print(f"✅ Loaded {len(X_val):,} validation samples")

# Load ensemble models
model_paths = [
    'checkpoints/c1_phase1_seed42_ema_best.pt',
    'checkpoints/c1_phase1_seed123_ema_best.pt',
    'checkpoints/c1_phase1_seed456_ema_best.pt',
    'checkpoints/c1_phase1_seed789_ema_best.pt',
    'checkpoints/c1_phase1_seed1337_ema_best.pt',
]

# Check which models exist
existing_models = [p for p in model_paths if Path(p).exists()]
print(f"\n🔍 Found {len(existing_models)} ensemble models")

if len(existing_models) == 0:
    print("❌ No ensemble models found! Trying single model...")
    existing_models = ['checkpoints/c1_v10_improved_best.pt']

models = []
for path in existing_models:
    model = CompactResponseTimeCNN()
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model.eval()
    models.append(model)
    print(f"  ✅ Loaded: {path}")

# Evaluate ensemble
print(f"\n🧠 Evaluating {len(models)} model(s)...")
all_preds = []

with torch.no_grad():
    batch_size = 128
    for i in range(0, len(X_val), batch_size):
        batch = X_val[i:i+batch_size]
        
        # Get predictions from all models
        batch_preds = []
        for model in models:
            pred = model(batch).numpy()
            batch_preds.append(pred)
        
        # Average predictions
        ensemble_pred = np.mean(batch_preds, axis=0)
        all_preds.append(ensemble_pred)

predictions = np.concatenate(all_preds)

# Calculate metrics
val_nrmse = nrmse(predictions, y_val)
baseline = 1.00019
target = 0.95

print("\n" + "=" * 60)
print("📊 RESULTS")
print("=" * 60)
print(f"🎯 Ensemble Val NRMSE: {val_nrmse:.6f}")
print(f"📌 Baseline: {baseline}")
print(f"🎯 Target: < {target}")

if val_nrmse < target:
    improvement = (baseline - val_nrmse) / baseline * 100
    print(f"\n✅ TARGET ACHIEVED! ({improvement:.1f}% improvement)")
    print("🚀 Ready to submit!")
elif val_nrmse < baseline:
    improvement = (baseline - val_nrmse) / baseline * 100
    gap = (val_nrmse / target - 1) * 100
    print(f"\n🎉 IMPROVED by {improvement:.1f}%!")
    print(f"⚠️  Need {gap:.1f}% more for target")
else:
    gap = (val_nrmse - baseline) / baseline * 100
    print(f"\n⚠️  Worse by {gap:.1f}%")

print("=" * 60)

# Show prediction stats
print(f"\n📈 Prediction Statistics:")
print(f"  Mean: {predictions.mean():.3f}")
print(f"  Std: {predictions.std():.3f}")
print(f"  Min: {predictions.min():.3f}")
print(f"  Max: {predictions.max():.3f}")
print(f"\n📈 Target Statistics:")
print(f"  Mean: {y_val.mean():.3f}")
print(f"  Std: {y_val.std():.3f}")
print(f"  Min: {y_val.min():.3f}")
print(f"  Max: {y_val.max():.3f}")
