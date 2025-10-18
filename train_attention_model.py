"""
Training script for Attention-Enhanced CNN on Challenge 1
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append('src')

from models_with_attention import AttentionCNN_ResponseTime, LightweightAttentionCNN

print("=" * 70)
print("ATTENTION-ENHANCED CNN TRAINING")
print("=" * 70)

# Configuration
CONFIG = {
    'model_type': 'lightweight',  # 'full' or 'lightweight'
    'num_heads': 4,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'num_epochs': 50,
    'patience': 10,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

print(f"\nDevice: {CONFIG['device']}")
print(f"Model: {CONFIG['model_type']}")
print(f"Attention heads: {CONFIG['num_heads']}")
print(f"Learning rate: {CONFIG['learning_rate']}")

# Create model
print("\n" + "=" * 70)
print("CREATING MODEL")
print("=" * 70)

if CONFIG['model_type'] == 'full':
    model = AttentionCNN_ResponseTime(num_heads=CONFIG['num_heads'])
else:
    model = LightweightAttentionCNN(num_heads=CONFIG['num_heads'])

model = model.to(CONFIG['device'])

params = sum(p.numel() for p in model.parameters())
print(f"Model: {model.__class__.__name__}")
print(f"Parameters: {params:,}")

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.AdamW(
    model.parameters(),
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay']
)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=5,
    verbose=True
)

print("\n" + "=" * 70)
print("TRAINING SETUP COMPLETE")
print("=" * 70)
print("\nOptimizer: AdamW")
print(f"Loss: MSE")
print(f"Scheduler: ReduceLROnPlateau")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Epochs: {CONFIG['num_epochs']}")
print(f"Early stopping patience: {CONFIG['patience']}")

# Save configuration
print("\n" + "=" * 70)
print("READY TO TRAIN")
print("=" * 70)
print("\nTo train this model, you need to:")
print("1. Load your EEG dataset (Challenge 1)")
print("2. Create DataLoaders for train/val")
print("3. Run the training loop")
print("\nExample training loop:")
print("""
# Training loop
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(CONFIG['num_epochs']):
    # Training
    model.train()
    train_loss = 0.0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(CONFIG['device'])
        batch_y = batch_y.to(CONFIG['device'])
        
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(CONFIG['device'])
            batch_y = batch_y.to(CONFIG['device'])
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    
    # Learning rate scheduling
    scheduler.step(val_loss)
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': CONFIG
        }, 'attention_cnn_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= CONFIG['patience']:
            print(f"Early stopping at epoch {epoch}")
            break
    
    print(f"Epoch {epoch+1}/{CONFIG['num_epochs']} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
""")

