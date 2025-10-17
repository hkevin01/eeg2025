#!/usr/bin/env python3
"""
Enhanced GPU Training Script
===========================

Advanced training script with enhanced GPU optimization for both NVIDIA and AMD.
Features:
- Intelligent platform-specific optimizations
- Dynamic batch size optimization
- Performance profiling and monitoring
- Advanced memory management
- Automated hyperparameter tuning
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
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

# Enhanced Configuration
class TrainingConfig:
    def __init__(self):
        # Basic settings
        self.challenge = 1  # 1 for age, 2 for sex
        self.max_samples = 3000  # Increased for better performance
        self.epochs = 15
        self.base_batch_size = 16
        self.learning_rate = 1e-4
        self.weight_decay = 1e-5
        
        # Enhanced GPU settings
        self.use_enhanced_ops = True
        self.enable_profiling = True
        self.dynamic_batch_sizing = True
        self.memory_optimization = True
        
        # Model architecture
        self.d_model = 128
        self.n_heads = 8
        self.n_layers = 6
        self.dropout = 0.1
        
        # Training optimization
        self.gradient_accumulation_steps = 1
        self.warmup_epochs = 2
        self.lr_schedule = "cosine"
        self.early_stopping_patience = 5
        
        # Monitoring
        self.log_interval = 10
        self.eval_interval = 100
        self.save_best = True

class AdvancedTrainer:
    """Advanced trainer with enhanced GPU optimization"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.gpu_opt = get_enhanced_optimizer()
        
        # Training state
        self.best_metric = float('-inf') if config.challenge == 1 else 0.0
        self.patience_counter = 0
        self.training_stats = []
        
        # Setup device and batch size
        self.device = self.gpu_opt.get_optimal_device("transformer")
        if config.dynamic_batch_sizing:
            self.batch_size = self.gpu_opt.optimize_batch_size(config.base_batch_size)
        else:
            self.batch_size = config.base_batch_size
            
        print(f"🚀 Advanced Trainer initialized")
        print(f"   Platform: {self.gpu_opt.platform}")
        print(f"   Device: {self.device}")
        print(f"   Batch size: {self.batch_size}")
        
    def create_model(self) -> nn.Module:
        """Create enhanced EEG model"""
        num_classes = 1  # For regression (age) or binary classification (sex)
        
        model = create_enhanced_eeg_model(
            n_channels=129,
            num_classes=num_classes,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            use_enhanced_ops=self.config.use_enhanced_ops
        )
        
        # Move to optimal device
        model = model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Model parameters: {trainable_params:,} trainable, {total_params:,} total")
        
        return model
        
    def create_optimizer(self, model: nn.Module) -> optim.Optimizer:
        """Create optimizer with enhanced settings"""
        # Separate parameters for different learning rates
        backbone_params = []
        head_params = []
        
        for name, param in model.named_parameters():
            if 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
                
        # Different learning rates for backbone and head
        optimizer = optim.AdamW([
            {'params': backbone_params, 'lr': self.config.learning_rate * 0.1},
            {'params': head_params, 'lr': self.config.learning_rate}
        ], weight_decay=self.config.weight_decay)
        
        return optimizer
        
    def create_scheduler(self, optimizer: optim.Optimizer, num_training_steps: int):
        """Create learning rate scheduler"""
        if self.config.lr_schedule == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=num_training_steps
            )
        elif self.config.lr_schedule == "linear":
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1.0, end_factor=0.1, 
                total_iters=num_training_steps
            )
        else:
            scheduler = None
            
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

        # Get labels
        if self.config.challenge == 1:
            label_dict = dict(zip(participants_df['participant_id'], participants_df['age']))
            print("Challenge 1: Age Prediction")
            print(f"Age range: {participants_df['age'].min():.1f} - {participants_df['age'].max():.1f}")
        else:
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
        
        # Create dataloaders
        train_loader = DataLoader(
            train_subset, batch_size=self.batch_size, shuffle=True, 
            num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=self.batch_size, shuffle=False,
            num_workers=2, pin_memory=True
        )
        
        return train_loader, val_loader, label_dict
        
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: optim.Optimizer, criterion: nn.Module,
                   label_dict: dict, epoch: int, scheduler=None) -> dict:
        """Train one epoch with enhanced optimization"""
        model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        # Progress tracking
        start_time = time.time()
        
        for batch_idx, (data, subj_ids) in enumerate(train_loader):
            # Get labels for batch
            batch_labels = torch.tensor(
                [label_dict[sid] for sid in subj_ids], 
                dtype=torch.float32
            )
            
            # Move to optimal devices using enhanced optimizer
            with self.gpu_opt.memory_management("training"):
                data = self.gpu_opt.optimize_tensor_for_operation(data, "transformer")
                batch_labels = self.gpu_opt.optimize_tensor_for_operation(batch_labels, "general")
                
                # Forward pass
                if self.config.enable_profiling:
                    outputs = self.gpu_opt.profiler.profile_operation(
                        "forward_pass", model, data
                    )
                else:
                    outputs = model(data)
                    
                # Compute loss
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
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
                total_loss += loss.item()
                all_preds.extend(outputs.detach().cpu().numpy())
                all_labels.extend(batch_labels.cpu().numpy())
                
                # Logging
                if batch_idx % self.config.log_interval == 0:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"   Batch {batch_idx:4d}/{len(train_loader)} | "
                          f"Loss: {loss.item():.4f} | LR: {current_lr:.6f}")
                    
                    # Memory monitoring
                    if torch.cuda.is_available():
                        mem_allocated = torch.cuda.memory_allocated() / 1024**2
                        mem_cached = torch.cuda.memory_reserved() / 1024**2
                        print(f"   GPU Memory: {mem_allocated:.0f}MB allocated, {mem_cached:.0f}MB cached")
        
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
                
                # Move to devices
                data = self.gpu_opt.optimize_tensor_for_operation(data, "transformer")
                batch_labels = self.gpu_opt.optimize_tensor_for_operation(batch_labels, "general")
                
                # Forward pass
                outputs = model(data)
                loss = criterion(outputs, batch_labels)
                
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
        print(f"🚀 ENHANCED GPU TRAINING - Challenge {self.config.challenge}")
        print("="*80)
        
        # Load data
        train_loader, val_loader, label_dict = self.load_data()
        
        # Create model
        model = self.create_model()
        
        # Create optimizer and scheduler
        optimizer = self.create_optimizer(model)
        num_training_steps = len(train_loader) * self.config.epochs
        scheduler = self.create_scheduler(optimizer, num_training_steps)
        
        # Loss function
        if self.config.challenge == 1:
            criterion = nn.MSELoss()
        else:
            criterion = nn.BCELoss()
            
        # Training loop
        print(f"\n🔥 Starting training")
        print(f"Epochs: {self.config.epochs}, Batch size: {self.batch_size}")
        print(f"Train samples: {len(train_loader.dataset)}, Val samples: {len(val_loader.dataset)}")
        print("="*80)
        
        for epoch in range(self.config.epochs):
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
            is_better = (val_stats['metric'] > self.best_metric if self.config.challenge == 1 
                        else val_stats['metric'] > self.best_metric)
            
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
                    }, checkpoint_dir / "enhanced_best.pth")
                    
                print(f"💾 New best model saved! {train_stats['metric_name']}: {self.best_metric:.4f}")
            else:
                self.patience_counter += 1
                
            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping after {epoch+1} epochs")
                break
                
            # GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
        # Final statistics
        print("\n" + "="*80)
        print("🏁 Training completed!")
        print(f"Best {train_stats['metric_name']}: {self.best_metric:.4f}")
        
        # Performance statistics
        if self.config.enable_profiling:
            stats = self.gpu_opt.get_performance_stats()
            print(f"\n📊 Performance Statistics:")
            for key, value in stats.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for k, v in value.items():
                        print(f"     {k}: {v:.4f}s" if isinstance(v, float) else f"     {k}: {v}")
                else:
                    print(f"   {key}: {value}")

def main():
    """Main training function"""
    config = TrainingConfig()
    trainer = AdvancedTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()
