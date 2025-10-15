#!/usr/bin/env python3
"""
Train EEG Transformer Model
Self-supervised learning on HBN EEG data
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models.eeg_dataset import EEGDataset
from models.transformer import EEGTransformer
import argparse
from tqdm import tqdm
import json

def contrastive_loss(embeddings, temperature=0.07):
    """Simple contrastive loss for self-supervised learning"""
    # Normalize embeddings
    embeddings = nn.functional.normalize(embeddings, dim=1)
    
    # Compute similarity matrix
    similarity = torch.matmul(embeddings, embeddings.T) / temperature
    
    # Create labels (diagonal elements are positive pairs)
    batch_size = embeddings.shape[0]
    labels = torch.arange(batch_size, device=embeddings.device)
    
    # Cross entropy loss
    loss = nn.functional.cross_entropy(similarity, labels)
    return loss


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        eeg = batch['eeg'].to(device)
        
        # Forward pass
        embeddings = model(eeg)
        
        # Contrastive loss
        loss = contrastive_loss(embeddings)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(dataloader)


def validate(model, dataloader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            eeg = batch['eeg'].to(device)
            embeddings = model(eeg)
            loss = contrastive_loss(embeddings)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description="Train EEG Transformer")
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--d-model', type=int, default=256, help='Model dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--output-dim', type=int, default=128, help='Output embedding dimension')
    parser.add_argument('--max-files', type=int, default=None, help='Max files per subject')
    parser.add_argument('--output-dir', type=str, default='outputs/transformer', help='Output directory')
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EEG Transformer Training")
    print("=" * 60)
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print()
    
    # Load dataset
    print("Loading dataset...")
    dataset = EEGDataset(max_files_per_subject=args.max_files)
    print()
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print()
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("Creating model...")
    model = EEGTransformer(
        n_channels=129,
        seq_len=1000,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.d_model * 4,
        dropout=0.1,
        output_dim=args.output_dim
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print()
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Training loop
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    print("Starting training...")
    print()
    
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Log
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_dir / 'best_model.pt')
            print(f"  ✅ Saved best model (val_loss: {val_loss:.4f})")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, output_dir / f'checkpoint_epoch_{epoch}.pt')
        
        print()
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("=" * 60)
    print("Training complete!")
    print(f"Best val loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
