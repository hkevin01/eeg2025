#!/usr/bin/env python3
"""
Fast GPU Training - Skips slow preprocessing
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import mne

# ROCm environment variables
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HSA_FORCE_FINE_GRAIN_PCIE'] = '1'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:512'

sys.path.insert(0, str(Path(__file__).parent.parent))

print("🚀 Fast GPU Training (No Preprocessing)")
print("="*60)

# GPU Setup
def setup_device():
    if not torch.cuda.is_available():
        print("⚠️  No GPU, using CPU")
        return torch.device("cpu"), {}
    
    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    
    print(f"\n✅ GPU: {gpu_name}")
    print(f"   Memory: {props.total_memory / (1024**3):.2f} GB")
    
    # Limit memory usage
    torch.cuda.set_per_process_memory_fraction(0.7, device=device)
    
    return device, {'gpu_name': gpu_name}

# Fast Dataset - NO PREPROCESSING
class FastEEGDataset(Dataset):
    def __init__(self, data_dir, max_subjects=None, max_files_per_subject=2):
        self.data_dir = Path(data_dir)
        self.windows = []
        self.labels = []
        
        print(f"\n📂 Loading from: {data_dir}")
        print(f"   (Skipping preprocessing for speed)")
        
        subjects = sorted(list(self.data_dir.glob("sub-NDAR*")))
        if max_subjects:
            subjects = subjects[:max_subjects]
        
        print(f"   Subjects: {len(subjects)}")
        
        total_files = 0
        for subj_dir in tqdm(subjects, desc="Loading"):
            eeg_dir = subj_dir / "eeg"
            if not eeg_dir.exists():
                continue
            
            # Limit files per subject for speed
            set_files = list(eeg_dir.glob("*.set"))[:max_files_per_subject]
            
            for set_file in set_files:
                try:
                    # Load WITHOUT preprocessing
                    raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
                    
                    # Get data directly (no filtering!)
                    data = raw.get_data()
                    sfreq = raw.info['sfreq']
                    window_samples = int(2.0 * sfreq)
                    
                    # Create fewer windows for speed
                    step = window_samples  # No overlap
                    for start in range(0, data.shape[1] - window_samples, step):
                        window = data[:, start:start + window_samples]
                        self.windows.append(window.astype(np.float32))  # float32 for GPU
                        self.labels.append(np.random.randint(0, 2))
                    
                    total_files += 1
                    
                except Exception as e:
                    continue
        
        print(f"\n✅ Loaded {len(self.windows)} windows from {total_files} files")
        if len(self.windows) > 0:
            print(f"   Shape: {self.windows[0].shape}")
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.windows[idx]), torch.LongTensor([self.labels[idx]])[0]

# Tiny model for testing
class TinyTransformer(nn.Module):
    def __init__(self, n_channels=129, seq_len=1000):
        super().__init__()
        hidden = 64
        self.proj = nn.Linear(n_channels, hidden)
        self.pos = nn.Parameter(torch.randn(1, seq_len, hidden) * 0.01)
        
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden*2,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(hidden, 2)
    
    def forward(self, x):
        x = x.transpose(1, 2)  # (B, T, C)
        x = self.proj(x)
        x = x + self.pos[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.head(x)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for data, target in tqdm(loader, desc="Training"):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, pred = output.max(1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return total_loss / len(loader), 100. * correct / total

def main():
    device, config = setup_device()
    
    # Load data FAST
    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
    dataset = FastEEGDataset(data_dir, max_subjects=3, max_files_per_subject=1)
    
    if len(dataset) == 0:
        print("❌ No data!")
        return
    
    batch_size = 4 if device.type == 'cuda' else 8
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    print(f"\n📦 Dataloader: {len(dataset)} windows, batch_size={batch_size}")
    
    # Tiny model
    sample, _ = dataset[0]
    model = TinyTransformer(n_channels=sample.shape[0], seq_len=sample.shape[1]).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: {n_params:,} parameters")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Train
    print("\n🏋️  Training...")
    print("="*60)
    
    for epoch in range(3):
        print(f"\nEpoch {epoch+1}/3")
        loss, acc = train_epoch(model, loader, criterion, optimizer, device)
        print(f"Loss: {loss:.4f}, Acc: {acc:.2f}%")
        
        if device.type == 'cuda':
            mem = torch.cuda.memory_allocated() / (1024**3)
            print(f"GPU Memory: {mem:.2f}GB")
    
    print("\n✅ Training complete!")
    
    # Save
    out_dir = Path(__file__).parent.parent / "outputs" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "tiny_gpu.pth")
    print(f"💾 Saved to: {out_dir / 'tiny_gpu.pth'}")

if __name__ == "__main__":
    main()
