#!/usr/bin/env python3
"""
Challenge 1: Cross-Task Transfer Learning (SuS → CCD)
=====================================================

Transfer learning from foundation model to CCD prediction tasks:
- Response Time prediction (Pearson r > 0.3 target)
- Success Rate classification (AUROC > 0.7 target)

Official Metrics:
- Response Time: Pearson correlation
- Success: AUROC, Balanced Accuracy
- Combined Score: (correlation + AUROC) / 2
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import time
import json
from datetime import datetime
from tqdm import tqdm

print("🎯 Challenge 1: Cross-Task Transfer Learning")
print("=" * 70)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configuration
CONFIG = {
    'data_dir': project_root / "data" / "raw" / "hbn",
    'checkpoint_path': project_root / "checkpoints" / "foundation_best.pth",
    'output_dir': project_root / "outputs" / "challenge1",
    'checkpoint_dir': project_root / "checkpoints" / "challenge1",
    'log_dir': project_root / "logs",
    
    # Model
    'hidden_dim': 128,
    'n_heads': 8,
    'n_layers': 4,
    'dropout': 0.1,
    
    # Training
    'batch_size': 32,
    'epochs': 30,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'patience': 10,
    'min_delta': 0.001,
    
    # Transfer learning
    'freeze_encoder': True,  # Freeze foundation encoder initially
    'unfreeze_after_epoch': 5,  # Progressive unfreezing
}

# Create directories
for dir_path in [CONFIG['output_dir'], CONFIG['checkpoint_dir'], CONFIG['log_dir']]:
    dir_path.mkdir(parents=True, exist_ok=True)

class FoundationTransformer(nn.Module):
    """Foundation model (same as training)"""
    def __init__(self, n_channels=129, seq_len=1000, 
                 hidden_dim=128, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        
        self.input_proj = nn.Linear(n_channels, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
    
    def forward(self, x):
        """Extract features without classification head"""
        x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return x

class Challenge1Model(nn.Module):
    """Transfer learning model for Challenge 1"""
    def __init__(self, encoder, hidden_dim=128, dropout=0.1):
        super().__init__()
        
        self.encoder = encoder
        
        # Task-specific heads
        self.rt_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Response time (continuous)
        )
        
        self.success_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # Success (binary classification)
        )
    
    def forward(self, x):
        features = self.encoder(x)
        rt_pred = self.rt_head(features).squeeze(-1)
        success_logits = self.success_head(features)
        return rt_pred, success_logits

# For now, use simple EEG dataset (we'll need CCD-specific data later)
sys.path.insert(0, str(project_root / "scripts"))
from models.eeg_dataset_simple import SimpleEEGDataset

def compute_challenge1_metrics(rt_pred, rt_true, success_pred, success_true):
    """Compute official Challenge 1 metrics"""
    metrics = {}
    
    # Response Time: Pearson correlation
    rt_pred_np = rt_pred.detach().cpu().numpy()
    rt_true_np = rt_true.detach().cpu().numpy()
    
    valid_mask = ~(np.isnan(rt_pred_np) | np.isnan(rt_true_np) | np.isinf(rt_pred_np) | np.isinf(rt_true_np))
    if valid_mask.sum() > 10:
        correlation, p_value = pearsonr(rt_pred_np[valid_mask], rt_true_np[valid_mask])
        metrics['rt_correlation'] = correlation
        metrics['rt_p_value'] = p_value
    else:
        metrics['rt_correlation'] = 0.0
        metrics['rt_p_value'] = 1.0
    
    # Success: AUROC and Balanced Accuracy
    success_prob = F.softmax(success_pred, dim=1)[:, 1].detach().cpu().numpy()
    success_true_np = success_true.detach().cpu().numpy()
    
    if len(np.unique(success_true_np)) > 1:
        metrics['success_auroc'] = roc_auc_score(success_true_np, success_prob)
        success_pred_class = (success_prob > 0.5).astype(int)
        metrics['success_balanced_acc'] = balanced_accuracy_score(success_true_np, success_pred_class)
    else:
        metrics['success_auroc'] = 0.5
        metrics['success_balanced_acc'] = 0.5
    
    # Combined score (official)
    metrics['combined_score'] = (metrics['rt_correlation'] + metrics['success_auroc']) / 2
    
    return metrics

def train_epoch(model, loader, rt_criterion, success_criterion, optimizer, device, epoch):
    """Train one epoch"""
    model.train()
    
    total_rt_loss = 0
    total_success_loss = 0
    all_rt_pred = []
    all_rt_true = []
    all_success_pred = []
    all_success_true = []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch+1} Training")
    for batch_idx, (data, target) in enumerate(pbar):
        data = data.to(device)
        # For now, we're using sex as proxy target (0/1)
        # In real challenge, we'd have actual RT and success labels
        target = target.to(device)
        
        # Create synthetic RT labels (in real challenge, these come from data)
        rt_target = torch.randn_like(target.float()) * 100 + 500  # Synthetic RT: mean=500ms, std=100ms
        
        optimizer.zero_grad()
        
        rt_pred, success_logits = model(data)
        
        # Compute losses
        rt_loss = rt_criterion(rt_pred, rt_target)
        success_loss = success_criterion(success_logits, target)
        
        # Combined loss
        loss = rt_loss + success_loss
        loss.backward()
        optimizer.step()
        
        total_rt_loss += rt_loss.item()
        total_success_loss += success_loss.item()
        
        # Collect predictions
        all_rt_pred.append(rt_pred.detach())
        all_rt_true.append(rt_target.detach())
        all_success_pred.append(success_logits.detach())
        all_success_true.append(target.detach())
        
        if (batch_idx + 1) % 10 == 0:
            pbar.set_postfix({
                'rt_loss': f'{rt_loss.item():.4f}',
                'success_loss': f'{success_loss.item():.4f}'
            })
    
    # Compute metrics
    rt_pred_all = torch.cat(all_rt_pred)
    rt_true_all = torch.cat(all_rt_true)
    success_pred_all = torch.cat(all_success_pred)
    success_true_all = torch.cat(all_success_true)
    
    metrics = compute_challenge1_metrics(rt_pred_all, rt_true_all, success_pred_all, success_true_all)
    
    return total_rt_loss / len(loader), total_success_loss / len(loader), metrics

def validate(model, loader, rt_criterion, success_criterion, device, epoch):
    """Validate"""
    model.eval()
    
    total_rt_loss = 0
    total_success_loss = 0
    all_rt_pred = []
    all_rt_true = []
    all_success_pred = []
    all_success_true = []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch+1} Validation")
        for data, target in pbar:
            data = data.to(device)
            target = target.to(device)
            
            # Synthetic RT target
            rt_target = torch.randn_like(target.float()) * 100 + 500
            
            rt_pred, success_logits = model(data)
            
            rt_loss = rt_criterion(rt_pred, rt_target)
            success_loss = success_criterion(success_logits, target)
            
            total_rt_loss += rt_loss.item()
            total_success_loss += success_loss.item()
            
            all_rt_pred.append(rt_pred)
            all_rt_true.append(rt_target)
            all_success_pred.append(success_logits)
            all_success_true.append(target)
    
    rt_pred_all = torch.cat(all_rt_pred)
    rt_true_all = torch.cat(all_rt_true)
    success_pred_all = torch.cat(all_success_pred)
    success_true_all = torch.cat(all_success_true)
    
    metrics = compute_challenge1_metrics(rt_pred_all, rt_true_all, success_pred_all, success_true_all)
    
    return total_rt_loss / len(loader), total_success_loss / len(loader), metrics

def main():
    start_time = time.time()
    
    device = torch.device('cpu')  # Safe CPU training
    print(f"\n📱 Device: {device}")
    print("=" * 70)
    
    # Load dataset
    print("\n📂 Loading Dataset...")
    dataset = SimpleEEGDataset(data_dir=CONFIG['data_dir'], max_subjects=None)
    print(f"   Total: {len(dataset)} windows")
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=4)
    
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create foundation encoder
    print("\n🧠 Creating Model...")
    sample_data, _ = dataset[0]
    n_channels, seq_len = sample_data.shape
    
    encoder = FoundationTransformer(
        n_channels=n_channels,
        seq_len=seq_len,
        hidden_dim=CONFIG['hidden_dim'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        dropout=CONFIG['dropout']
    )
    
    # Load pretrained weights if available
    if CONFIG['checkpoint_path'].exists():
        print(f"   Loading pretrained weights from {CONFIG['checkpoint_path']}")
        checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=device)
        # Load only encoder weights (foundation model has classifier, we don't need it)
        encoder_state = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items() 
                        if not k.startswith('classifier')}
        encoder.load_state_dict(encoder_state, strict=False)
        print("   ✅ Pretrained weights loaded")
    else:
        print("   ⚠️  No pretrained checkpoint found, training from scratch")
    
    # Create Challenge 1 model
    model = Challenge1Model(encoder, hidden_dim=CONFIG['hidden_dim'], dropout=CONFIG['dropout']).to(device)
    
    # Freeze encoder initially
    if CONFIG['freeze_encoder']:
        for param in model.encoder.parameters():
            param.requires_grad = False
        print("   🔒 Encoder frozen (will unfreeze later)")
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Trainable parameters: {n_params:,}")
    
    # Loss functions
    rt_criterion = nn.MSELoss()
    success_criterion = nn.CrossEntropyLoss()
    
    # Optimizer (only task-specific heads initially)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    # Training loop
    print("\n" + "=" * 70)
    print("🏋️  Training Started")
    print("=" * 70)
    
    best_combined_score = -float('inf')
    patience_counter = 0
    history = {
        'train_rt_loss': [],
        'train_success_loss': [],
        'train_metrics': [],
        'val_rt_loss': [],
        'val_success_loss': [],
        'val_metrics': []
    }
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"{'='*70}")
        
        # Unfreeze encoder after specified epoch
        if epoch == CONFIG['unfreeze_after_epoch'] and CONFIG['freeze_encoder']:
            print("\n🔓 Unfreezing encoder for fine-tuning")
            for param in model.encoder.parameters():
                param.requires_grad = True
            # Recreate optimizer with all parameters
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=CONFIG['learning_rate'] * 0.1,  # Lower LR for fine-tuning
                weight_decay=CONFIG['weight_decay']
            )
        
        # Train & validate
        train_rt_loss, train_success_loss, train_metrics = train_epoch(
            model, train_loader, rt_criterion, success_criterion, optimizer, device, epoch
        )
        val_rt_loss, val_success_loss, val_metrics = validate(
            model, val_loader, rt_criterion, success_criterion, device, epoch
        )
        
        # Record history
        history['train_rt_loss'].append(train_rt_loss)
        history['train_success_loss'].append(train_success_loss)
        history['train_metrics'].append(train_metrics)
        history['val_rt_loss'].append(val_rt_loss)
        history['val_success_loss'].append(val_success_loss)
        history['val_metrics'].append(val_metrics)
        
        # Print metrics
        print(f"\n📊 Epoch {epoch+1} Results:")
        print(f"   Train RT Loss: {train_rt_loss:.4f}, Success Loss: {train_success_loss:.4f}")
        print(f"   Val RT Loss: {val_rt_loss:.4f}, Success Loss: {val_success_loss:.4f}")
        print(f"\n   📈 Train Metrics:")
        print(f"      RT Correlation: {train_metrics['rt_correlation']:.4f}")
        print(f"      Success AUROC: {train_metrics['success_auroc']:.4f}")
        print(f"      Combined Score: {train_metrics['combined_score']:.4f}")
        print(f"\n   📈 Val Metrics:")
        print(f"      RT Correlation: {val_metrics['rt_correlation']:.4f} (target: >0.30)")
        print(f"      Success AUROC: {val_metrics['success_auroc']:.4f} (target: >0.70)")
        print(f"      Combined Score: {val_metrics['combined_score']:.4f}")
        
        # Save checkpoint
        combined_score = val_metrics['combined_score']
        if combined_score > best_combined_score + CONFIG['min_delta']:
            best_combined_score = combined_score
            patience_counter = 0
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_combined_score': best_combined_score,
                'val_metrics': val_metrics,
                'config': CONFIG
            }
            
            best_path = CONFIG['checkpoint_dir'] / "challenge1_best.pth"
            torch.save(checkpoint, best_path)
            print(f"\n   ⭐ New best model! Combined score: {best_combined_score:.4f}")
        else:
            patience_counter += 1
            print(f"\n   Patience: {patience_counter}/{CONFIG['patience']}")
        
        # Early stopping
        if patience_counter >= CONFIG['patience']:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save final history
    history_path = CONFIG['log_dir'] / f"challenge1_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    # Convert to JSON-serializable format
    history_json = {}
    for key, value in history.items():
        if isinstance(value[0], dict):
            history_json[key] = value
        else:
            history_json[key] = [float(v) if isinstance(v, (np.floating, torch.Tensor)) else v for v in value]
    
    with open(history_path, 'w') as f:
        json.dump(history_json, f, indent=2)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("✅ Training Complete!")
    print(f"{'='*70}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Best combined score: {best_combined_score:.4f}")
    print(f"   Best RT correlation: {history['val_metrics'][0]['rt_correlation']:.4f}")
    print(f"   Best Success AUROC: {history['val_metrics'][0]['success_auroc']:.4f}")
    print(f"   Checkpoints: {CONFIG['checkpoint_dir']}")
    print(f"   History: {history_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
