#!/usr/bin/env python3
"""
Minimal training script - designed to actually complete
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import sys
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.models.eeg_dataset_simple import SimpleEEGDataset

print("=" * 80)
print("🚀 MINIMAL FOUNDATION TRAINING")
print("=" * 80)

# Simple config - small enough to complete quickly
CONFIG = {
    'hidden_dim': 64,       # Smaller model
    'n_heads': 4,
    'n_layers': 2,
    'dropout': 0.1,
    'batch_size': 32,       # Larger batches
    'epochs': 5,            # Just 5 epochs
    'learning_rate': 1e-3,
    'max_samples': 5000,    # Limit dataset size
}

# Simple model
class SimpleModel(nn.Module):
    def __init__(self, n_channels=129, seq_len=1000, hidden_dim=64, 
                 n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, hidden_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, hidden_dim) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.classifier(x)

def main():
    device = torch.device('cpu')
    print(f"Device: {device}")
    
    # Load dataset
    print("\n📂 Loading dataset...")
    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
    full_dataset = SimpleEEGDataset(data_dir=data_dir, max_subjects=None)
    
    # Limit to max_samples
    if len(full_dataset) > CONFIG['max_samples']:
        indices = torch.randperm(len(full_dataset))[:CONFIG['max_samples']]
        dataset = Subset(full_dataset, indices)
        print(f"Using {len(dataset)} samples (limited from {len(full_dataset)})")
    else:
        dataset = full_dataset
        print(f"Using all {len(dataset)} samples")
    
    # Split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], 
                           shuffle=False, num_workers=2)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Model
    print("\n🧠 Creating model...")
    model = SimpleModel(
        hidden_dim=CONFIG['hidden_dim'],
        n_heads=CONFIG['n_heads'],
        n_layers=CONFIG['n_layers'],
        dropout=CONFIG['dropout']
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    # Training
    print("\n" + "=" * 80)
    print("🏋️  TRAINING")
    print("=" * 80)
    
    history = []
    best_val_loss = float('inf')
    
    for epoch in range(CONFIG['epochs']):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}")
        print(f"{'='*80}")
        
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            if (batch_idx + 1) % 20 == 0:
                print(f"  [{batch_idx+1}/{len(train_loader)}] "
                      f"loss={loss.item():.4f}, "
                      f"acc={100.*train_correct/train_total:.1f}%")
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validate
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_loss /= len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"  Train: loss={train_loss:.4f}, acc={train_acc:.1f}%")
        print(f"  Val:   loss={val_loss:.4f}, acc={val_acc:.1f}%")
        
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        })
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
            checkpoint_dir.mkdir(exist_ok=True)
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': CONFIG
            }, checkpoint_dir / "minimal_best.pth")
            print(f"  💾 Saved checkpoint (best val loss: {val_loss:.4f})")
    
    # Save history
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    with open(log_dir / "minimal_history.json", 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoint: checkpoints/minimal_best.pth")
    print(f"History: logs/minimal_history.json")

if __name__ == "__main__":
    main()
