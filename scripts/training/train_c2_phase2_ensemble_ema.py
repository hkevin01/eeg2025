#!/usr/bin/env python3
"""
Challenge 2 Phase 2 - Variance Reduction Ensemble
- 5 seeds with EMA (decay=0.999)
- Robust validation
- Early stopping + ReduceLROnPlateau
- Goal: C2 from 1.00066 → 1.00035-1.00050
"""
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import mne
from tqdm import tqdm
import time
from datetime import datetime
from copy import deepcopy

from braindecode.models import EEGNeX

# ============================================================
# HYPERPARAMETERS (Variance Reduction Focused)
# ============================================================
SEEDS = [42, 123, 456, 789, 1337]
EPOCHS = 25
BATCH_SIZE = 64
LR = 0.002
WEIGHT_DECAY = 0.001  # Reduced from 0.01 per recommendations
MIN_LR = 1e-5

# Early stopping & LR scheduler
EARLY_STOP_PATIENCE = 10
LR_PATIENCE = 5
LR_FACTOR = 0.5

# EMA
EMA_DECAY = 0.999

# Mixup (conservative for regression)
MIXUP_ALPHA = 0.15  # Reduced from 0.2 to avoid mean bias
PROB_MIXUP_BATCH = 0.7

# Augmentation (probabilistic, conservative)
PROB_TIME_SHIFT = 0.5
PROB_AMP_SCALE = 0.5
PROB_NOISE = 0.3

# ============================================================
# EMA Helper Class
# ============================================================
class ModelEMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.original = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model):
        """Update EMA weights"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] + 
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self, model):
        """Apply EMA weights to model (for validation/test)"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.original[name] = param.data.clone()
                param.data.copy_(self.shadow[name])
    
    def restore(self, model):
        """Restore original weights"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.original[name])

# ============================================================
# Dataset with Robust Loading
# ============================================================
class C2Dataset(Dataset):
    """BIDS-based dataset with conservative augmentation"""
    
    def __init__(self, data_dirs, max_subjects=None, augment=False):
        self.segments = []
        self.scores = []
        self.augment = augment
        
        print(f"\n📁 Loading from BIDS (augment={augment})...")
        
        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            participants_file = data_dir / "participants.tsv"
            
            if not participants_file.exists():
                print(f"⚠️  Skipping {data_dir}: no participants.tsv")
                continue
            
            df = pd.read_csv(participants_file, sep='\t')
            
            # Check for externalizing column
            if 'externalizing' not in df.columns:
                print(f"⚠️  Skipping {data_dir}: no 'externalizing' column")
                continue
            
            df = df.dropna(subset=['externalizing'])
            
            if max_subjects:
                df = df.head(max_subjects)
            
            print(f"   {data_dir.name}: {len(df)} subjects")
            
            loaded_count = 0
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"   {data_dir.name}"):
                subject_id = row['participant_id']
                subject_dir = data_dir / subject_id / "eeg"
                
                if not subject_dir.exists():
                    continue
                
                # Look for EEG files
                eeg_files = list(subject_dir.glob("*RestingState*.bdf"))
                if not eeg_files:
                    eeg_files = list(subject_dir.glob("*RestingState*.set"))
                if not eeg_files:
                    continue
                
                try:
                    # Load EEG
                    if eeg_files[0].suffix == '.bdf':
                        raw = mne.io.read_raw_bdf(eeg_files[0], preload=True, verbose=False)
                    else:
                        raw = mne.io.read_raw_eeglab(eeg_files[0], preload=True, verbose=False)
                    
                    # Resample if needed
                    if raw.info['sfreq'] != 100:
                        raw.resample(100, verbose=False)
                    
                    data = raw.get_data()
                    
                    # Validate channel count
                    if data.shape[0] != 129:
                        continue
                    
                    # Z-score normalize
                    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
                    
                    # Create segments (4s for augment, 2s for val)
                    segment_length = 400 if augment else 200
                    n_samples = data.shape[1]
                    n_segments = n_samples // segment_length
                    
                    externalizing = float(row['externalizing'])
                    
                    for i in range(n_segments):
                        start = i * segment_length
                        end = start + segment_length
                        segment = data[:, start:end]
                        
                        self.segments.append(torch.FloatTensor(segment))
                        self.scores.append(externalizing)
                    
                    loaded_count += 1
                
                except Exception as e:
                    continue
        
        print(f"✅ Loaded {len(self.segments)} segments from {loaded_count} subjects")
        if len(self.scores) > 0:
            scores_array = np.array(self.scores)
            print(f"   Externalizing: mean={scores_array.mean():.3f}, std={scores_array.std():.3f}")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = self.segments[idx]
        score = torch.FloatTensor([self.scores[idx]])
        
        if self.augment and segment.shape[1] == 400:
            # Random crop 4s → 2s
            start = torch.randint(0, 201, (1,)).item()
            segment = segment[:, start:start+200]
            
            # Conservative probabilistic augmentation
            if np.random.rand() < PROB_TIME_SHIFT:
                shift = np.random.randint(-10, 11)
                segment = torch.roll(segment, shifts=shift, dims=1)
            
            if np.random.rand() < PROB_AMP_SCALE:
                scale = 0.9 + 0.2 * np.random.rand()  # 0.9-1.1 (tighter than before)
                segment = segment * scale
            
            if np.random.rand() < PROB_NOISE:
                noise = torch.randn_like(segment) * 0.03  # Reduced from 0.05
                segment = segment + noise
        
        return segment, score

# ============================================================
# Mixup (Conservative for Regression)
# ============================================================
def mixup_data(x, y, alpha=0.15):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ============================================================
# Training Function with EMA
# ============================================================
def train_model(seed, train_dataset, val_dataset, device):
    print("=" * 60)
    print(f"🌱 Training Model with Seed {seed}")
    print("=" * 60)
    
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
    
    # Create model
    model = EEGNeX(
        n_chans=129,
        n_times=200,
        n_outputs=1,
        sfreq=100
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize EMA
    ema = ModelEMA(model, decay=EMA_DECAY)
    print(f"EMA initialized with decay={EMA_DECAY}")
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR,
        verbose=True
    )
    
    criterion = nn.SmoothL1Loss()
    
    # DataLoaders with fixed seeds
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        generator=torch.Generator().manual_seed(seed)
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        # Train
        model.train()
        train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Conservative mixup
            if np.random.rand() < PROB_MIXUP_BATCH:
                x_batch, y_a, y_b, lam = mixup_data(x_batch, y_batch, alpha=MIXUP_ALPHA)
                
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)
            else:
                optimizer.zero_grad()
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update EMA
            ema.update(model)
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate with EMA weights
        ema.apply_shadow(model)
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(x_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        ema.restore(model)  # Restore training weights
        
        # Update scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint with EMA weights
            ema.apply_shadow(model)
            checkpoint_path = f'checkpoints/c2_phase2_seed{seed}_ema_best.pt'
            torch.save({
                'epoch': epoch,
                'seed': seed,
                'model_state_dict': model.state_dict(),  # EMA weights
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'train_loss': train_loss,
                'nrmse': val_loss,
                'metrics': {
                    'nrmse': val_loss,
                    'val_loss': val_loss,
                    'train_loss': train_loss,
                    'ema_decay': EMA_DECAY
                }
            }, checkpoint_path)
            ema.restore(model)
        else:
            patience_counter += 1
        
        # Print progress
        elapsed = time.time() - start_time
        eta = (elapsed / epoch) * (EPOCHS - epoch)
        
        status = "✅" if val_loss == best_val_loss else "  "
        print(f"{status} Epoch {epoch:2d}/{EPOCHS} | "
              f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
              f"Best: {best_val_loss:.6f} (e{best_epoch}) | "
              f"LR: {current_lr:.2e} | Patience: {patience_counter}/{EARLY_STOP_PATIENCE} | "
              f"ETA: {eta/60:.1f}m")
        
        # Early stopping
        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
            break
    
    total_time = time.time() - start_time
    print(f"\n✅ Seed {seed} complete!")
    print(f"   Best Val: {best_val_loss:.6f} at epoch {best_epoch}")
    print(f"   Time: {total_time/60:.1f} min")
    print()
    
    return best_val_loss, best_epoch

# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("🧠 Challenge 2 Phase 2 - Variance Reduction Ensemble")
    print("=" * 60)
    print(f"Strategy: {len(SEEDS)} seeds + EMA (decay={EMA_DECAY})")
    print(f"Goal: C2 from 1.00066 → 1.00035-1.00050")
    print()
    
    # Device (force CPU for stability)
    device = 'cpu'
    print(f"Device: {device} (forced for ROCm stability)")
    print()
    
    # Load datasets (once for all models)
    print("Loading datasets...")
    train_dataset = C2Dataset(['data/raw/ds005509-bdf'], max_subjects=None, augment=True)
    val_dataset = C2Dataset(['data/raw/ds005509-bdf'], max_subjects=30, augment=False)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("❌ ERROR: No data loaded! Check data paths.")
        return
    
    print(f"\n✅ Data ready: {len(train_dataset)} train, {len(val_dataset)} val samples")
    print()
    
    # Train ensemble
    results = []
    
    for seed in SEEDS:
        val_loss, best_epoch = train_model(seed, train_dataset, val_dataset, device)
        results.append((seed, val_loss, best_epoch))
    
    # Summary
    print("=" * 60)
    print("🎉 Ensemble Training Complete!")
    print("=" * 60)
    for seed, val_loss, best_epoch in results:
        print(f"Seed {seed:4d}: Val Loss = {val_loss:.6f} (epoch {best_epoch})")
    print()
    
    avg_val_loss = np.mean([vl for _, vl, _ in results])
    std_val_loss = np.std([vl for _, vl, _ in results])
    print(f"Ensemble Statistics:")
    print(f"  Mean Val Loss: {avg_val_loss:.6f}")
    print(f"  Std Val Loss:  {std_val_loss:.6f}")
    print(f"  CV: {100*std_val_loss/avg_val_loss:.2f}%")
    print()
    
    print("Next steps:")
    print("1. Create ensemble submission.py with TTA")
    print("2. Add linear calibration on validation")
    print("3. Test locally")
    print("4. Submit as V11")
    print("=" * 60)

if __name__ == "__main__":
    main()
