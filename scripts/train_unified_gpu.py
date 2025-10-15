#!/usr/bin/env python3
"""
Unified GPU Training Script
===========================

Uses the unified GPU optimization module for both NVIDIA and AMD GPUs.
Automatically detects platform and applies appropriate optimizations.
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import unified GPU module
from gpu.unified_gpu_optimized import (
    GPUPlatformDetector,
    UnifiedFFTOptimizer,
    UnifiedBLASOptimizer,
    UnifiedLinearLayer
)
from scripts.models.eeg_dataset_simple import SimpleEEGDataset

# Configuration
CONFIG = {
    'challenge': 1,  # 1 for age, 2 for sex
    'max_samples': 3000,  # Conservative for GPU safety
    'epochs': 8,
    'batch_size': 16,  # Smaller batch for GPU stability
    'learning_rate': 1e-4,
    'sleep_between_epochs': 2,
    'check_resources_every': 10,
    'use_gpu': True,  # Enable unified GPU
}

class ResourceMonitor:
    """Monitor system resources and GPU memory"""
    def __init__(self, cpu_threshold=80, mem_threshold=85, gpu_mem_threshold=80):
        self.cpu_threshold = cpu_threshold
        self.mem_threshold = mem_threshold
        self.gpu_mem_threshold = gpu_mem_threshold
        self.pid = os.getpid()

    def check_and_sleep(self):
        """Check resources and sleep if needed"""
        cpu = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory().percent
        
        high_usage = False
        
        if cpu > self.cpu_threshold or mem > self.mem_threshold:
            print(f"  ⚠️  High usage: CPU={cpu:.1f}%, MEM={mem:.1f}% - sleeping 5s...")
            high_usage = True
        
        # Check GPU memory if available
        if torch.cuda.is_available():
            try:
                gpu_mem_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
                if gpu_mem_used > self.gpu_mem_threshold:
                    print(f"  ⚠️  High GPU memory: {gpu_mem_used:.1f}% - cleaning up...")
                    torch.cuda.empty_cache()
                    high_usage = True
            except:
                pass
        
        if high_usage:
            time.sleep(5)
            return True
        return False

class GPUOptimizedFoundationModel(nn.Module):
    """Foundation model with unified GPU optimizations"""
    def __init__(self, n_channels=129, seq_len=1000, hidden_dim=64,
                 n_heads=4, n_layers=2, dropout=0.1, device="cuda"):
        super().__init__()
        
        # Use unified linear layers for GPU optimization
        self.input_proj = UnifiedLinearLayer(n_channels, hidden_dim, device=device)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Initialize FFT optimizer for spectral features (optional)
        self.fft_opt = UnifiedFFTOptimizer(device=device)

    def forward(self, x):
        # Optional: Add spectral features using unified FFT
        # spectral_features = self.extract_spectral_features(x)
        
        x = x.transpose(1, 2)  # (batch, time, channels)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return x.mean(dim=1)
    
    def extract_spectral_features(self, x):
        """Extract spectral features using unified FFT (optional)"""
        # x shape: (batch, channels, time)
        with torch.no_grad():
            X = self.fft_opt.rfft_batch(x, dim=-1)
            power = torch.abs(X) ** 2
            # Use log power in specific frequency bands
            return torch.log(power.mean(dim=-1) + 1e-8)

class GPUOptimizedPredictionModel(nn.Module):
    """Prediction model with unified GPU optimizations"""
    def __init__(self, backbone, hidden_dim=64, output_dim=1, use_sigmoid=False, device="cuda"):
        super().__init__()
        self.backbone = backbone
        
        # Use unified linear layers
        self.head = nn.Sequential(
            UnifiedLinearLayer(hidden_dim, hidden_dim, device=device),
            nn.ReLU(),
            nn.Dropout(0.1),
            UnifiedLinearLayer(hidden_dim, output_dim, device=device)
        )
        
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        features = self.backbone(x)
        out = self.head(features).squeeze(-1)
        if self.use_sigmoid:
            out = self.sigmoid(out)
        return out

def train_with_unified_gpu():
    print("="*80)
    print(f"🚀 UNIFIED GPU TRAINING - Challenge {CONFIG['challenge']}")
    print("="*80)
    
    # Initialize GPU platform detection
    gpu_detector = GPUPlatformDetector()
    gpu_detector.print_info()
    
    # Select device
    if CONFIG['use_gpu'] and gpu_detector.gpu_available:
        device = torch.device('cuda')
        print(f"✅ Using {gpu_detector.platform} for GPU acceleration")
    else:
        device = torch.device('cpu')
        print("⚠️  Using CPU (GPU disabled or unavailable)")
    
    monitor = ResourceMonitor()

    # Load foundation model
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "minimal_best.pth"
    if not checkpoint_path.exists():
        print("❌ Foundation model not found!")
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Create GPU-optimized backbone
        backbone = GPUOptimizedFoundationModel(
            hidden_dim=64, n_heads=4, n_layers=2, device=str(device)
        ).to(device)
        
        # Load weights (skip GPU-specific layers)
        backbone_dict = {}
        for k, v in checkpoint['model_state_dict'].items():
            if not k.startswith('classifier') and 'input_proj' in k:
                # Map old linear layer to new unified linear layer
                if k.endswith('.weight'):
                    backbone_dict[k] = v
                elif k.endswith('.bias'):
                    backbone_dict[k] = v
            elif not k.startswith('classifier') and 'input_proj' not in k:
                backbone_dict[k] = v
        
        backbone.load_state_dict(backbone_dict, strict=False)

        # Create GPU-optimized prediction model
        is_classification = (CONFIG['challenge'] == 2)
        model = GPUOptimizedPredictionModel(
            backbone, hidden_dim=64, output_dim=1,
            use_sigmoid=is_classification, device=str(device)
        ).to(device)

        # Freeze backbone
        for param in model.backbone.parameters():
            param.requires_grad = False

        print("✅ GPU-optimized model loaded, backbone frozen")

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        print("Falling back to standard model...")
        
        # Fallback to CPU model
        from scripts.train_with_monitoring import FoundationModel, PredictionModel
        
        device = torch.device('cpu')
        backbone = FoundationModel(hidden_dim=64, n_heads=4, n_layers=2).to(device)
        backbone_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                        if not k.startswith('classifier')}
        backbone.load_state_dict(backbone_dict, strict=False)
        
        is_classification = (CONFIG['challenge'] == 2)
        model = PredictionModel(backbone, hidden_dim=64, output_dim=1,
                              use_sigmoid=is_classification).to(device)
        
        for param in model.backbone.parameters():
            param.requires_grad = False

    # Load data
    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
    participants_file = data_dir / "participants.tsv"

    if not participants_file.exists():
        print("❌ participants.tsv not found!")
        return

    participants_df = pd.read_csv(participants_file, sep='\t')
    print(f"✅ Loaded {len(participants_df)} participants")

    # Load dataset
    full_dataset = SimpleEEGDataset(data_dir=data_dir, max_subjects=None)

    # Get labels
    if CONFIG['challenge'] == 1:
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
        if i >= CONFIG['max_samples']:
            break
        if subj_id in label_dict:
            valid_indices.append(i)
            valid_labels.append(label_dict[subj_id])

    print(f"✅ Found {len(valid_indices)} samples with labels")

    if len(valid_indices) < 100:
        print("❌ Not enough labeled samples!")
        return

    # Create subset and dataloader
    subset = Subset(full_dataset, valid_indices)
    train_loader = DataLoader(subset, batch_size=CONFIG['batch_size'], shuffle=True)

    # Setup training
    if CONFIG['challenge'] == 1:
        criterion = nn.MSELoss()
        metric_name = "Correlation"
    else:
        criterion = nn.BCELoss()
        metric_name = "Accuracy"

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate']
    )

    print(f"\n🔥 Training with {gpu_detector.platform if device.type == 'cuda' else 'CPU'}")
    print(f"Samples: {len(valid_indices)}, Batch size: {CONFIG['batch_size']}")
    print("="*80)

    # Training loop
    model.train()
    for epoch in range(CONFIG['epochs']):
        epoch_loss = 0
        epoch_preds = []
        epoch_labels = []

        start_time = time.time()

        for batch_idx, (data, subj_ids) in enumerate(train_loader):
            # Get labels for this batch
            batch_labels = torch.tensor([label_dict[sid] for sid in subj_ids], dtype=torch.float32)
            
            data = data.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            
            # Forward pass with GPU optimization
            try:
                outputs = model(data)
                loss = criterion(outputs, batch_labels)
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_preds.extend(outputs.detach().cpu().numpy())
                epoch_labels.extend(batch_labels.cpu().numpy())
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"  ⚠️  GPU memory error - clearing cache...")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e

            # Resource monitoring
            if batch_idx % CONFIG['check_resources_every'] == 0:
                monitor.check_and_sleep()

        # Calculate metrics
        epoch_preds = np.array(epoch_preds)
        epoch_labels = np.array(epoch_labels)

        if CONFIG['challenge'] == 1:
            correlation, _ = pearsonr(epoch_preds, epoch_labels)
            metric_value = correlation
        else:
            accuracy = np.mean((epoch_preds > 0.5) == epoch_labels)
            metric_value = accuracy

        epoch_time = time.time() - start_time

        print(f"Epoch {epoch+1:2d}/{CONFIG['epochs']} | "
              f"Loss: {epoch_loss/len(train_loader):.4f} | "
              f"{metric_name}: {metric_value:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # Sleep between epochs
        if epoch < CONFIG['epochs'] - 1:
            time.sleep(CONFIG['sleep_between_epochs'])
            
        # GPU memory cleanup
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    print("\n✅ Training completed!")
    
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"Final GPU memory - Allocated: {allocated:.1f}MB, Cached: {cached:.1f}MB")
        torch.cuda.empty_cache()

if __name__ == "__main__":
    train_with_unified_gpu()
