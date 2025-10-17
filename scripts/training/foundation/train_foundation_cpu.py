#!/usr/bin/env python3
"""
Foundation Model Training - CPU Version
Simple, reliable, no GPU complications
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import json
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.models.eeg_dataset_simple import SimpleEEGDataset

print("🚀 Foundation Model Training (CPU)")
print("=" * 70)

# Configuration
CONFIG = {
    'data_dir': Path(__file__).parent.parent / "data" / "raw" / "hbn",
    'output_dir': Path(__file__).parent.parent / "outputs",
    'checkpoint_dir': Path(__file__).parent.parent / "checkpoints",
    'log_dir': Path(__file__).parent.parent / "logs",
    
    # Data
    'max_subjects': None,  # ALL
    'train_split': 0.8,
    
    # Model
    'hidden_dim': 128,
    'n_heads': 8,
    'n_layers': 4,
    'dropout': 0.1,
    
    # Training
    'batch_size': 16,
    'epochs': 20,
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'gradient_clip': 1.0,
    'save_every': 2,
}

# Create directories
for dir_path in [CONFIG['output_dir'], CONFIG['checkpoint_dir'], CONFIG['log_dir']]:
    dir_path.mkdir(exist_ok=True)

# Model
class FoundationTransformer(nn.Module):
    def __init__(self, n_channels=129, seq_len=1000, 
                 hidden_dim=128, n_heads=8, n_layers=4, 
                 n_classes=2, dropout=0.1):
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
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.classifier(x)
        return x

# Training
def train_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    print(f"\n📚 Epoch {epoch+1} - Training")
    for batch_idx, (data, target) in enumerate(loader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['gradient_clip'])
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"   [{batch_idx+1}/{len(loader)}] loss={loss.item():.4f}, acc={100.*correct/total:.1f}%")
    
    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, epoch):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    print(f"\n🔍 Epoch {epoch+1} - Validation")
    with torch.no_grad():
        for data, target in loader:
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    print(f"   Val: loss={total_loss/len(loader):.4f}, acc={100.*correct/total:.1f}%")
    return total_loss / len(loader), 100. * correct / total

def main():
    start_time = time.time()
    
    print(f"📱 Device: CPU")
    print("=" * 70)
    
    # Load dataset
    print("\n📂 Loading Dataset...")
    dataset = SimpleEEGDataset(data_dir=CONFIG['data_dir'], max_subjects=CONFIG['max_subjects'])
    print(f"   Total: {len(dataset)} windows")
    
    # Split
    train_size = int(CONFIG['train_split'] * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=4)
    
    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    print("\n🧠 Creating Model...")
    sample_data, _ = dataset[0]
    n_channels, seq_len = sample_data.shape
    
    model = FoundationTransformer(
        n_channels=n_channels,
        seq_len=seq_len,
        hidden_dim=CONFIG['hidden_dim'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        dropout=CONFIG['dropout']
    )
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {n_params:,} (~{n_params*4/(1024**2):.1f} MB)")
    
    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), 
                                  lr=CONFIG['learning_rate'],
                                  weight_decay=CONFIG['weight_decay'])
    
    # Training
    print("\n" + "=" * 70)
    print("🏋️  Training Started")
    print("=" * 70)
    
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(CONFIG['epochs']):
        epoch_start = time.time()
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"{'='*70}")
        
        # Train & validate
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, epoch)
        
        epoch_time = time.time() - epoch_start
        
        # Record
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train: loss={train_loss:.4f}, acc={train_acc:.1f}%")
        print(f"   Val:   loss={val_loss:.4f}, acc={val_acc:.1f}%")
        print(f"   Time:  {epoch_time:.1f}s")
        
        # Save
        if (epoch + 1) % CONFIG['save_every'] == 0 or val_loss < best_val_loss:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': CONFIG
            }
            
            path = CONFIG['checkpoint_dir'] / f"foundation_epoch_{epoch+1}.pth"
            torch.save(checkpoint, path)
            print(f"   💾 Checkpoint: {path.name}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = CONFIG['checkpoint_dir'] / "foundation_best.pth"
                torch.save(checkpoint, best_path)
                print(f"   ⭐ New best model!")
    
    # Finish
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("✅ Training Complete!")
    print(f"{'='*70}")
    print(f"   Total time: {total_time/60:.1f} minutes ({total_time/3600:.2f} hours)")
    print(f"   Best val loss: {best_val_loss:.4f}")
    print(f"   Checkpoints: {CONFIG['checkpoint_dir']}")
    
    # Save history
    history_path = CONFIG['log_dir'] / f"foundation_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"   History: {history_path}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
