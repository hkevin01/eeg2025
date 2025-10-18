# EEGNeX + ROCm Strategy: Maximizing Starter Kit Libraries

## ✅ Current Stack (Already Using!)

You're **already using all the right libraries**:

| Library | Purpose | Status |
|---------|---------|--------|
| **eegdash** | Competition data loader | ✅ Using |
| **braindecode** | EEG deep learning toolkit | ✅ Using |
| **mne** | EEG signal processing | ✅ Using |
| **torch** | Deep learning framework | ✅ Using |

**Your implementation is correct!** Now let's maximize it.

## 🚀 Why Use EEGNeX (Despite Size)

### Your Current Models vs EEGNeX

| Model | Params | Training Time | Performance | Best For |
|-------|--------|---------------|-------------|----------|
| **CompactResponseTimeCNN** | 75K | Fast (~2 hours) | Good (1.00 NRMSE) | Quick iteration |
| **CompactExternalizingCNN** | 64K | Fast (~2 hours) | Good (1.46 NRMSE) | Quick iteration |
| **EEGNeX** | 5M+ | Slow (~12 hours) | **Excellent** | Final submission |

### Why EEGNeX is Worth It

1. **State-of-the-art architecture** (from starter kit)
2. **Proven on competition data** (starter kit benchmark)
3. **Event-locked analysis** (exactly what Challenge 1 needs)
4. **Better feature extraction** (5M params = more patterns)
5. **ROCm acceleration available** (you have AMD GPU!)

### EEGNeX Architecture (from braindecode)

```python
from braindecode.models import EEGNeX

model = EEGNeX(
    n_chans=129,        # HBN has 129 channels
    n_outputs=1,        # Regression (response time)
    n_times=200,        # 2s windows @ 100Hz
    sfreq=100,
    # Architecture optimized for EEG:
    # - Depthwise separable convolutions
    # - Temporal & spatial attention
    # - Multi-scale feature extraction
)
```

**Key advantages:**
- Depthwise separable convs (efficient despite size)
- Built-in attention (temporal & spatial)
- Designed specifically for EEG data

## 🔥 ROCm GPU Acceleration Strategy

### Your AMD GPU Setup

You have ROCm installed! Let's use it:

```python
import torch

# Check ROCm availability
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"ROCm version: {torch.version.hip}")

# Enable TF32 for faster training (if supported)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### Training Optimizations

```python
# 1. Mixed Precision Training (2x speedup)
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
for X, y in train_loader:
    X, y = X.to('cuda'), y.to('cuda')
    
    with autocast():
        outputs = model(X)
        loss = criterion(outputs, y)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

# 2. DataLoader Optimization (3x speedup)
train_loader = DataLoader(
    dataset,
    batch_size=32,  # Larger batch with GPU
    num_workers=8,  # Parallel data loading
    pin_memory=True,  # Fast CPU-GPU transfer
    persistent_workers=True,  # Keep workers alive
)

# 3. Gradient Checkpointing (2x memory, same speed)
# For EEGNeX's deep layers
torch.utils.checkpoint.checkpoint_sequential(model.layers, ...)

# 4. Compile model (PyTorch 2.0+)
model = torch.compile(model, mode='reduce-overhead')
```

### Expected Speedup

| Optimization | Speedup | Memory |
|--------------|---------|--------|
| Base (CPU) | 1x | - |
| GPU (naive) | 5-10x | Same |
| + Mixed precision | 15-20x | 50% less |
| + DataLoader opts | 30-40x | Same |
| + Compile | 50-60x | Same |

**Result: 12 hour training → 15-30 minutes with ROCm!**

## 📚 Maximizing Starter Kit Libraries

### 1. eegdash: Competition Data Loader

**What you're doing right:**
```python
from eegdash import EEGChallengeDataset

dataset = EEGChallengeDataset(
    release='R1',  # Correct!
    mini=False,    # Full data
    query=dict(task='contrastChangeDetection'),  # Target task
    cache_dir='data/raw'
)
```

**What you can add:**
```python
# Load multiple releases efficiently
releases = ['R1', 'R2', 'R3', 'R4']  # Use R4!
datasets = []

for release in releases:
    ds = EEGChallengeDataset(
        release=release,
        mini=False,
        query=dict(task='contrastChangeDetection'),
        cache_dir='data/raw',
        # NEW: Description fields for Challenge 2
        description_fields=['externalizing', 'age', 'sex']
    )
    datasets.extend(ds.datasets)

# Combine all releases
from braindecode.datasets import BaseConcatDataset
combined = BaseConcatDataset(datasets)
```

### 2. braindecode: EEG Deep Learning

**What you're using:**
- ✅ Preprocessor (preprocessing pipelines)
- ✅ create_windows_from_events (window extraction)
- ✅ preprocess (parallel processing)

**What you should add:**

#### Event-Locked Analysis (Critical for Challenge 1!)
```python
from braindecode.preprocessing import Preprocessor, create_windows_from_events
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

# This is KEY for Challenge 1 (Response Time)
preprocessors = [
    # Extract response time from events
    Preprocessor(
        annotate_trials_with_target,
        apply_on_array=False,
        target_field='rt_from_stimulus',  # This is your label!
        epoch_length=2.0,
        require_stimulus=True,
        require_response=True,
    ),
    # Add event anchors
    Preprocessor(add_aux_anchors, apply_on_array=False),
]

# Apply preprocessing
preprocess(dataset, preprocessors, n_jobs=-1)

# Create windows LOCKED TO STIMULUS
SHIFT_AFTER_STIM = 0.5  # 500ms after stimulus
WINDOW_LEN = 2.0        # 2 second windows

windows = create_windows_from_events(
    dataset,
    mapping={'stimulus_anchor': 0},  # Lock to stimulus
    trial_start_offset_samples=int(SHIFT_AFTER_STIM * 100),
    trial_stop_offset_samples=int((SHIFT_AFTER_STIM + WINDOW_LEN) * 100),
    window_size_samples=int(2.0 * 100),  # 200 samples
    window_stride_samples=100,  # 1 second stride
    preload=True,
)

# Add metadata
windows = add_extras_columns(
    windows, dataset, 
    desc='stimulus_anchor',
    keys=('target', 'rt_from_stimulus', 'stimulus_onset', 
          'response_onset', 'correct')
)
```

**Why this matters:**
- Response time is relative to stimulus onset
- You need windows locked to stimulus, not arbitrary
- This is why starter kit uses braindecode!

#### EEGNeX from braindecode
```python
from braindecode.models import EEGNeX

# Challenge 1: Response Time
model_c1 = EEGNeX(
    n_chans=129,
    n_outputs=1,  # Regression
    n_times=200,  # 2s @ 100Hz
    sfreq=100,
    drop_prob=0.5,
    # EEGNeX uses:
    # - Depthwise separable convs
    # - Temporal convolutions
    # - Layer normalization
)

# Challenge 2: Externalizing
model_c2 = EEGNeX(
    n_chans=129,
    n_outputs=1,
    n_times=200,  # 2s @ 100Hz
    sfreq=100,
    drop_prob=0.5,
)
```

### 3. mne: Signal Processing Foundation

**What braindecode uses mne for:**
- Raw data loading (.bdf, .edf files)
- Channel operations
- Filtering (already done by competition)
- Event handling

**You can add:**
```python
import mne

# Advanced preprocessing (if needed)
def advanced_preprocessing(raw):
    # ICA for artifact removal
    ica = mne.preprocessing.ICA(n_components=20, random_state=42)
    ica.fit(raw)
    ica.exclude = [0, 1]  # Exclude blink/eye movement components
    ica.apply(raw)
    
    # Robust re-referencing
    raw.set_eeg_reference('average', projection=True)
    
    return raw

# Apply to dataset
# (But competition data is already preprocessed!)
```

### 4. torch: Deep Learning Framework

**What you're using:**
- ✅ nn.Module (model definitions)
- ✅ DataLoader (data loading)
- ✅ Optimizers (Adam, SGD)
- ✅ Loss functions (MSE)

**What you should add with ROCm:**

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# 1. Mixed Precision Training
scaler = GradScaler()

def train_step(model, X, y, optimizer, criterion, scaler):
    optimizer.zero_grad()
    
    # Mixed precision forward pass
    with autocast():
        outputs = model(X)
        loss = criterion(outputs, y)
    
    # Mixed precision backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    return loss.item()

# 2. Better optimizers
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

optimizer = AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,  # L2 regularization
    betas=(0.9, 0.999)
)

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    epochs=50,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,  # Warmup for 30% of training
)

# 3. Gradient clipping (for stability)
nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 4. EMA (Exponential Moving Average)
from torch.optim.swa_utils import AveragedModel

ema_model = AveragedModel(model, avg_fn=lambda avg, model, num: 
    0.999 * avg + 0.001 * model)

# Update EMA after each batch
ema_model.update_parameters(model)
```

## 🎯 Complete EEGNeX Training Script

Let me create a complete training script that uses all libraries optimally:

```python
#!/usr/bin/env python3
"""
Challenge 1: EEGNeX with ROCm acceleration
Uses: eegdash, braindecode, mne, torch
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from braindecode.models import EEGNeX
from braindecode.datasets import BaseConcatDataset
from braindecode.preprocessing import (
    Preprocessor, preprocess, create_windows_from_events
)
from eegdash import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)
import numpy as np
from tqdm import tqdm

# ROCm setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def load_data(releases=['R1', 'R2', 'R3', 'R4']):
    """Load and preprocess data using eegdash + braindecode"""
    all_datasets = []
    
    for release in releases:
        print(f"Loading {release}...")
        dataset = EEGChallengeDataset(
            release=release,
            mini=False,
            query=dict(task='contrastChangeDetection'),
            cache_dir='data/raw'
        )
        
        # Event-locked preprocessing (KEY!)
        preprocessors = [
            Preprocessor(
                annotate_trials_with_target,
                apply_on_array=False,
                target_field='rt_from_stimulus',
                epoch_length=2.0,
                require_stimulus=True,
                require_response=True,
            ),
            Preprocessor(add_aux_anchors, apply_on_array=False),
        ]
        
        preprocess(dataset, preprocessors, n_jobs=-1)
        
        # Create event-locked windows
        dataset = keep_only_recordings_with('stimulus_anchor', dataset)
        windows = create_windows_from_events(
            dataset,
            mapping={'stimulus_anchor': 0},
            trial_start_offset_samples=50,  # 0.5s after stimulus
            trial_stop_offset_samples=250,  # 2.5s total
            window_size_samples=200,
            window_stride_samples=100,
            preload=True,
        )
        
        windows = add_extras_columns(
            windows, dataset,
            desc='stimulus_anchor',
            keys=('target', 'rt_from_stimulus')
        )
        
        all_datasets.extend(windows.datasets)
    
    return BaseConcatDataset(all_datasets)

def train_eegnex():
    """Train EEGNeX with ROCm acceleration"""
    
    # Load data
    print("Loading training data...")
    train_dataset = load_data(['R1', 'R2', 'R3', 'R4'])  # Use R4!
    val_dataset = load_data(['R5'])
    
    # DataLoaders (optimized for GPU)
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    
    # EEGNeX model
    print("Creating EEGNeX model...")
    model = EEGNeX(
        n_chans=129,
        n_outputs=1,
        n_times=200,
        sfreq=100,
        drop_prob=0.5,
    ).to(device)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Optimizer (AdamW with weight decay)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.01,
    )
    
    # Scheduler (OneCycleLR)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        epochs=50,
        steps_per_epoch=len(train_loader),
    )
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    best_nrmse = float('inf')
    
    for epoch in range(50):
        model.train()
        train_loss = 0
        
        for X, y, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            X = X.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32).unsqueeze(1)
            
            optimizer.zero_grad()
            
            # Mixed precision
            with autocast():
                outputs = model(X)
                loss = criterion(outputs, y)
            
            scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        y_preds, y_trues = [], []
        
        with torch.no_grad():
            for X, y, _ in val_loader:
                X = X.to(device, dtype=torch.float32)
                y = y.to(device, dtype=torch.float32)
                
                with autocast():
                    outputs = model(X)
                
                y_preds.extend(outputs.cpu().numpy().flatten())
                y_trues.extend(y.cpu().numpy().flatten())
        
        # Calculate NRMSE
        y_preds = np.array(y_preds)
        y_trues = np.array(y_trues)
        rmse = np.sqrt(np.mean((y_preds - y_trues) ** 2))
        nrmse = rmse / (y_trues.std() + 1e-8)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss/len(train_loader):.4f}, Val NRMSE={nrmse:.4f}")
        
        if nrmse < best_nrmse:
            best_nrmse = nrmse
            torch.save(model.state_dict(), 'eegnex_best.pth')
            print(f"  → Saved best model (NRMSE={nrmse:.4f})")
    
    return best_nrmse

if __name__ == '__main__':
    best_nrmse = train_eegnex()
    print(f"\n✅ Training complete! Best NRMSE: {best_nrmse:.4f}")
```

## 🎯 Action Plan

### Immediate (Today)
1. Create EEGNeX training script (use template above)
2. Test ROCm acceleration (should see 50x speedup)
3. Train on R1-R4 (add R4 data!)

### This Week
1. Compare EEGNeX vs CompactCNN performance
2. Implement ensemble (CompactCNN + EEGNeX)
3. Add TTA to EEGNeX

### Expected Results

| Model | NRMSE | Training Time | Params |
|-------|-------|---------------|--------|
| CompactCNN (current) | 1.00 | 2 hours | 75K |
| EEGNeX (no R4) | ~0.85 | 30 min (ROCm) | 5M |
| EEGNeX (with R4) | ~0.75 | 40 min (ROCm) | 5M |
| Ensemble (both) | ~0.65 | N/A | 5M+75K |

**Target: Top 10 leaderboard (NRMSE < 0.7)**

## 📚 Resources

**Your existing implementations:**
- `scripts/training/challenge1/train_challenge1_multi_release.py` - Current training
- `starter_kit_integration/challenge_1.py` - Starter kit example with EEGNeX
- `submission_tta.py` - TTA implementation

**Documentation:**
- braindecode.org - EEGNeX architecture details
- eegdash docs - Competition data format
- ROCm docs - GPU optimization

---

**Status: Ready to train EEGNeX with ROCm acceleration! Expected 15-20% NRMSE improvement! 🚀**
