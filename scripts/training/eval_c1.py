#!/usr/bin/env python3
"""
Evaluate C1 models on validation set
"""
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(42)
np.random.seed(42)

# Dataset
class H5Dataset(Dataset):
    def __init__(self, h5_paths):
        data_list, label_list = [], []
        for path in h5_paths:
            with h5py.File(path, 'r') as f:
                data_list.append(f['eeg'][:])
                label_list.append(f['labels'][:])
        
        self.data = np.concatenate(data_list).astype(np.float32)
        self.labels = np.concatenate(label_list).astype(np.float32)
        
        # Normalize
        for ch in range(self.data.shape[1]):
            m, s = self.data[:, ch].mean(), self.data[:, ch].std()
            if s > 0:
                self.data[:, ch] = (self.data[:, ch] - m) / s
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return torch.from_numpy(self.data[idx]), torch.tensor(self.labels[idx])

# Model definition (CompactCNN)
class CompactCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(129, 64, 5, padding=2)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(128, 128, 3, padding=1)
        self.pool3 = nn.MaxPool1d(2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 1)
        self.drop = nn.Dropout(0.4)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.drop(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.drop(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x).squeeze(-1)

def nrmse(pred, target):
    mse = ((pred - target) ** 2).mean()
    std = target.std()
    return np.sqrt(mse) / (std + 1e-8)

# Load data
print("="*60)
print("🔍 Evaluating C1 Models")
print("="*60)

files = [f'data/cached/challenge1_R{i}_windows.h5' for i in range(1, 5)]
dataset = H5Dataset(files)

# Same 80/20 split as training
train_size = int(0.8 * len(dataset))
_, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, len(dataset) - train_size],
    generator=torch.Generator().manual_seed(42)
)

val_loader = DataLoader(val_dataset, batch_size=64, num_workers=0)
print(f"\n📊 Validation samples: {len(val_dataset):,}")

# Load and evaluate model
model = CompactCNN()
model.load_state_dict(torch.load('checkpoints/c1_efficient_best.pt', map_location='cpu'))
model.eval()

print(f"\n🧠 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

all_preds = []
all_targets = []

with torch.no_grad():
    for X, y in val_loader:
        pred = model(X)
        all_preds.append(pred.numpy())
        all_targets.append(y.numpy())

all_preds = np.concatenate(all_preds)
all_targets = np.concatenate(all_targets)

val_nrmse = nrmse(all_preds, all_targets)

print("\n" + "="*60)
print("📈 RESULTS")
print("="*60)
print(f"Validation NRMSE: {val_nrmse:.6f}")
print(f"Baseline (C1 test): 1.00019")
print(f"Target: < 0.95")

print("\n" + "="*60)
if val_nrmse < 0.95:
    print("🎉 TARGET ACHIEVED!")
    print(f"   Improvement: {(1.00019 - val_nrmse) / 1.00019 * 100:.1f}%")
elif val_nrmse < 1.00019:
    print("✅ IMPROVED over baseline!")
    print(f"   Improvement: {(1.00019 - val_nrmse) / 1.00019 * 100:.1f}%")
else:
    print("⚠️  Not improved yet")
    print(f"   Gap: {(val_nrmse - 1.00019) / 1.00019 * 100:.1f}% worse")
print("="*60)
