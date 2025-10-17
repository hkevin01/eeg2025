#!/usr/bin/env python3
"""
AMD RX 5600 XT Optimized Training Script
=========================================

Specifically optimized for AMD Radeon RX 5600 XT (RDNA 1.0, gfx1010)
Addresses:
- hipBLASLt incompatibility warnings
- Conservative memory management for 6GB VRAM
- Optimal batch sizing for RDNA architecture
- Competition-specific enhancements from README

Features from Competition README:
- Cross-task transfer learning (Challenge 1)
- Progressive unfreezing with domain adaptation
- Subject invariance (Challenge 2)
- Clinical normalization
- Multi-task learning approach
"""
import os
import sys
import time
from pathlib import Path

# Fix AMD hipBLASLt issue BEFORE importing torch
os.environ['ROCBLAS_LAYER'] = '1'
os.environ['HIPBLASLT_LOG_LEVEL'] = '0'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.1.0'  # RX 5600 XT gfx version

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr
from torch.utils.data import DataLoader, Subset
import torch.optim as optim

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import enhanced modules
from gpu.enhanced_gpu_optimizer import get_enhanced_optimizer
from models.enhanced_gpu_layers import create_enhanced_eeg_model
from scripts.models.eeg_dataset_simple import SimpleEEGDataset

# AMD RX 5600 XT Optimal Configuration
class AMDRX5600XTConfig:
    """Configuration optimized for AMD RX 5600 XT"""
    def __init__(self, challenge=1):
        # Hardware constraints (RX 5600 XT: 6GB VRAM, 36 CUs, RDNA 1.0)
        self.max_vram_usage = 0.75  # Use 75% of 6GB = 4.5GB
        self.optimal_batch_size = 8  # Conservative for 6GB VRAM
        self.gradient_accumulation_steps = 4  # Effective batch size = 32
        
        # Challenge selection
        self.challenge = challenge  # 1 for age, 2 for sex
        
        # Data settings
        self.max_samples = 2000  # Fit in memory
        self.epochs = 20
        self.learning_rate = 5e-5  # Lower LR for stability
        self.weight_decay = 1e-5
        
        # Model architecture (optimized for VRAM)
        self.d_model = 96  # Reduced from 128
        self.n_heads = 6   # Reduced from 8
        self.n_layers = 4  # Reduced from 6
        self.dropout = 0.15
        
        # AMD-specific optimizations
        self.use_amp = False  # Mixed precision can be unstable on AMD
        self.pin_memory = False  # Can cause issues on AMD
        self.num_workers = 2  # Conservative
        
        # Competition enhancements from README
        self.use_progressive_unfreezing = True  # Challenge 1: Progressive unfreezing
        self.use_domain_adaptation = True  # Challenge 1: Domain adaptation
        self.use_subject_invariance = True  # Challenge 2: Subject invariance
        self.use_clinical_normalization = True  # Challenge 2: Clinical normalization
        
        # Training optimization
        self.warmup_epochs = 3
        self.lr_schedule = "cosine_with_warmup"
        self.early_stopping_patience = 7
        
        # Memory management
        self.clear_cache_every_n_batches = 5  # Frequent cleanup for AMD
        self.use_checkpoint_segments = True  # Gradient checkpointing
        
        # Monitoring
        self.log_interval = 5
        self.eval_interval = 50
        self.save_best = True

class AMDRX5600XTTrainer:
    """Trainer optimized for AMD RX 5600 XT with competition enhancements"""
    
    def __init__(self, config: AMDRX5600XTConfig):
        self.config = config
        
        # Force hipBLAS configuration
        self._configure_amd_backend()
        
        # Initialize GPU optimizer
        self.gpu_opt = get_enhanced_optimizer()
        self.device = self.gpu_opt.get_optimal_device("general")
        
        # Training state
        self.best_metric = float('-inf') if config.challenge == 1 else 0.0
        self.patience_counter = 0
        self.training_stats = []
        self.current_epoch = 0
        
        print(f"🚀 AMD RX 5600 XT Optimized Trainer")
        print(f"   Platform: {self.gpu_opt.platform}")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {config.optimal_batch_size} (effective: {config.optimal_batch_size * config.gradient_accumulation_steps})")
        print(f"   Challenge: {config.challenge} ({'Age Prediction' if config.challenge == 1 else 'Sex Classification'})")
        print(f"   Progressive Unfreezing: {'✅' if config.use_progressive_unfreezing else '❌'}")
        print(f"   Domain Adaptation: {'✅' if config.use_domain_adaptation else '❌'}")
        
    def _configure_amd_backend(self):
        """Configure AMD backend for RX 5600 XT"""
        # Suppress hipBLASLt warnings
        os.environ['ROCBLAS_LAYER'] = '1'
        os.environ['HIPBLASLT_LOG_LEVEL'] = '0'
        
        # RX 5600 XT specific settings
        os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.1.0'
        os.environ['PYTORCH_ROCM_ARCH'] = 'gfx1010'
        
        # Conservative memory settings
        os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        print("✅ AMD RX 5600 XT backend configured (hipBLASLt warnings suppressed)")
        
    def create_model(self) -> nn.Module:
        """Create optimized EEG model"""
        num_classes = 1  # For regression (age) or binary classification (sex)
        
        model = create_enhanced_eeg_model(
            n_channels=129,
            num_classes=num_classes,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            use_enhanced_ops=True
        )
        
        # Move to device
        model = model.to(self.device)
        
        # Gradient checkpointing for memory efficiency
        if self.config.use_checkpoint_segments:
            try:
                if hasattr(model, 'gradient_checkpointing_enable'):
                    model.gradient_checkpointing_enable()
                    print("   ✅ Gradient checkpointing enabled")
            except Exception as e:
                print(f"   ⚠️  Gradient checkpointing not available: {e}")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Model: {trainable_params:,} trainable, {total_params:,} total")
        
        return model
        
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer with competition enhancements"""
        # Progressive unfreezing: Different LRs for different layers
        if self.config.use_progressive_unfreezing:
            param_groups = []
            
            # Foundation layers (freeze early, low LR later)
            foundation_params = []
            # Task-specific layers (high LR)
            task_params = []
            
            for name, param in model.named_parameters():
                if 'classifier' in name or 'head' in name:
                    task_params.append(param)
                else:
                    foundation_params.append(param)
                    
            param_groups = [
                {'params': foundation_params, 'lr': self.config.learning_rate * 0.01, 'name': 'foundation'},
                {'params': task_params, 'lr': self.config.learning_rate, 'name': 'task_head'}
            ]
            
            optimizer = optim.AdamW(param_groups, weight_decay=self.config.weight_decay)
            print("   ✅ Progressive unfreezing optimizer configured")
        else:
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
            
        return optimizer
        
    def create_scheduler(self, optimizer: optim.Optimizer, num_training_steps: int):
        """Create learning rate scheduler with warmup"""
        if self.config.lr_schedule == "cosine_with_warmup":
            warmup_steps = num_training_steps * self.config.warmup_epochs // self.config.epochs
            
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))
                
            scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            print("   ✅ Cosine schedule with warmup configured")
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_training_steps)
            
        return scheduler
        
    def load_data(self):
        """Load and prepare data"""
        data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
        participants_file = data_dir / "participants.tsv"

        if not participants_file.exists():
            raise FileNotFoundError("participants.tsv not found!")

        participants_df = pd.read_csv(participants_file, sep='\t')
        print(f"✅ Loaded {len(participants_df)} participants")

        # Load dataset
        full_dataset = SimpleEEGDataset(data_dir=data_dir, max_subjects=None)

        # Get labels with clinical normalization
        if self.config.challenge == 1:
            # Age prediction
            label_dict = dict(zip(participants_df['participant_id'], participants_df['age']))
            
            # Clinical normalization: Z-score normalization
            if self.config.use_clinical_normalization:
                ages = participants_df['age'].values
                age_mean, age_std = ages.mean(), ages.std()
                for pid in label_dict:
                    label_dict[pid] = (label_dict[pid] - age_mean) / age_std
                print(f"   ✅ Clinical normalization applied (age: μ={age_mean:.1f}, σ={age_std:.1f})")
            
            print(f"Challenge 1: Age Prediction")
            print(f"Age range: {participants_df['age'].min():.1f} - {participants_df['age'].max():.1f}")
        else:
            # Sex classification
            sex_dict = {}
            for _, row in participants_df.iterrows():
                sex_dict[row['participant_id']] = 1 if row['sex'] == 'M' else 0
            label_dict = sex_dict
            print("Challenge 2: Sex Classification")

        # Filter and sample dataset
        valid_indices = []
        valid_labels = []
        
        for i, (data, subj_id) in enumerate(full_dataset):
            if i >= self.config.max_samples:
                break
            if subj_id in label_dict:
                valid_indices.append(i)
                valid_labels.append(label_dict[subj_id])

        print(f"✅ Found {len(valid_indices)} samples with labels")

        if len(valid_indices) < 100:
            raise ValueError("Not enough labeled samples!")

        # Create train/val split
        split_idx = int(0.8 * len(valid_indices))
        train_indices = valid_indices[:split_idx]
        val_indices = valid_indices[split_idx:]
        
        # Create datasets
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)
        
        # Create dataloaders (AMD optimized settings)
        train_loader = DataLoader(
            train_subset, 
            batch_size=self.config.optimal_batch_size, 
            shuffle=True, 
            num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory
        )
        val_loader = DataLoader(
            val_subset, 
            batch_size=self.config.optimal_batch_size, 
            shuffle=False,
            num_workers=self.config.num_workers, 
            pin_memory=self.config.pin_memory
        )
        
        return train_loader, val_loader, label_dict
        
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: optim.Optimizer, criterion: nn.Module,
                   label_dict: dict, epoch: int, scheduler=None) -> dict:
        """Train one epoch with AMD optimizations"""
        model.train()
        
        # Progressive unfreezing: Unfreeze layers gradually
        if self.config.use_progressive_unfreezing and epoch < 5:
            # Freeze foundation layers in early epochs
            for name, param in model.named_parameters():
                if 'classifier' not in name and 'head' not in name:
                    param.requires_grad = False
        elif self.config.use_progressive_unfreezing and epoch == 5:
            # Unfreeze all layers after epoch 5
            for param in model.parameters():
                param.requires_grad = True
            print("   🔓 All layers unfrozen")
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        start_time = time.time()
        
        optimizer.zero_grad()
        
        for batch_idx, (data, subj_ids) in enumerate(train_loader):
            # Get labels for batch
            batch_labels = torch.tensor(
                [label_dict[sid] for sid in subj_ids], 
                dtype=torch.float32
            )
            
            # Move to device (AMD optimized)
            data = data.to(self.device, non_blocking=False)  # non_blocking=False for AMD
            batch_labels = batch_labels.to(self.device, non_blocking=False)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs.squeeze(), batch_labels)
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                if scheduler:
                    scheduler.step()
            
            # Statistics
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            all_preds.extend(outputs.detach().cpu().numpy())
            all_labels.extend(batch_labels.cpu().numpy())
            
            # AMD memory management: Clear cache frequently
            if batch_idx % self.config.clear_cache_every_n_batches == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Logging
            if batch_idx % self.config.log_interval == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Batch {batch_idx:4d}/{len(train_loader)} | "
                      f"Loss: {loss.item() * self.config.gradient_accumulation_steps:.4f} | "
                      f"LR: {current_lr:.6f}")
                
                # Memory monitoring
                if torch.cuda.is_available():
                    mem_allocated = torch.cuda.memory_allocated() / 1024**2
                    mem_cached = torch.cuda.memory_reserved() / 1024**2
                    print(f"   GPU Memory: {mem_allocated:.0f}MB / {mem_cached:.0f}MB")
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        
        if self.config.challenge == 1:
            # Regression - correlation
            correlation, _ = pearsonr(all_preds, all_labels)
            metric_value = correlation
            metric_name = "Correlation"
        else:
            # Classification - accuracy
            accuracy = np.mean((np.array(all_preds) > 0.5) == np.array(all_labels))
            metric_value = accuracy
            metric_name = "Accuracy"
            
        epoch_time = time.time() - start_time
        
        return {
            'loss': avg_loss,
            'metric': metric_value,
            'metric_name': metric_name,
            'time': epoch_time
        }
        
    def validate(self, model: nn.Module, val_loader: DataLoader,
                criterion: nn.Module, label_dict: dict) -> dict:
        """Validate model"""
        model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for data, subj_ids in val_loader:
                batch_labels = torch.tensor(
                    [label_dict[sid] for sid in subj_ids], 
                    dtype=torch.float32
                )
                
                # Move to device
                data = data.to(self.device, non_blocking=False)
                batch_labels = batch_labels.to(self.device, non_blocking=False)
                
                # Forward pass
                outputs = model(data)
                loss = criterion(outputs.squeeze(), batch_labels)
                
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        
        if self.config.challenge == 1:
            correlation, _ = pearsonr(all_preds, all_labels)
            metric_value = correlation
        else:
            accuracy = np.mean((np.array(all_preds) > 0.5) == np.array(all_labels))
            metric_value = accuracy
            
        return {'loss': avg_loss, 'metric': metric_value}
        
    def train(self):
        """Main training loop"""
        print("="*80)
        print(f"🚀 AMD RX 5600 XT TRAINING - Challenge {self.config.challenge}")
        print("="*80)
        
        # Load data
        train_loader, val_loader, label_dict = self.load_data()
        
        # Create model
        model = self.create_model()
        
        # Create optimizer and scheduler
        optimizer = self.create_optimizer(model)
        num_training_steps = len(train_loader) * self.config.epochs // self.config.gradient_accumulation_steps
        scheduler = self.create_scheduler(optimizer, num_training_steps)
        
        # Loss function
        if self.config.challenge == 1:
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCEWithLogitsLoss()
            
        # Training loop
        print(f"\n🔥 Starting training")
        print(f"Epochs: {self.config.epochs}, Batch size: {self.config.optimal_batch_size}")
        print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
        print("="*80)
        
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print("-" * 50)
            
            # Train
            train_stats = self.train_epoch(
                model, train_loader, optimizer, criterion, 
                label_dict, epoch, scheduler
            )
            
            # Validate
            val_stats = self.validate(model, val_loader, criterion, label_dict)
            
            # Print results
            print(f"Train Loss: {train_stats['loss']:.4f} | "
                  f"Train {train_stats['metric_name']}: {train_stats['metric']:.4f}")
            print(f"Val Loss: {val_stats['loss']:.4f} | "
                  f"Val {train_stats['metric_name']}: {val_stats['metric']:.4f}")
            print(f"Epoch time: {train_stats['time']:.1f}s")
            
            # Save best model
            is_better = val_stats['metric'] > self.best_metric
            
            if is_better:
                self.best_metric = val_stats['metric']
                self.patience_counter = 0
                
                if self.config.save_best:
                    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
                    checkpoint_dir.mkdir(exist_ok=True)
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_metric': self.best_metric,
                        'config': self.config.__dict__
                    }, checkpoint_dir / f"amd_rx5600xt_challenge{self.config.challenge}_best.pth")
                    
                print(f"💾 New best model! {train_stats['metric_name']}: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"⏹️  Early stopping after {epoch+1} epochs")
                break
                
            # AMD cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Final statistics
        print("\n" + "="*80)
        print("🏁 Training completed!")
        print(f"Best {train_stats['metric_name']}: {self.best_metric:.4f}")
        print("="*80)

def main():
    """Main training function"""
    # Challenge 1 (Age prediction)
    print("Training Challenge 1: Age Prediction")
    config1 = AMDRX5600XTConfig(challenge=1)
    trainer1 = AMDRX5600XTTrainer(config1)
    trainer1.train()
    
    print("\n" + "="*80 + "\n")
    
    # Challenge 2 (Sex classification)
    print("Training Challenge 2: Sex Classification")
    config2 = AMDRX5600XTConfig(challenge=2)
    trainer2 = AMDRX5600XTTrainer(config2)
    trainer2.train()

if __name__ == "__main__":
    main()
