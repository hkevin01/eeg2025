#!/usr/bin/env python3
"""
Multi-process CPU training with PyTorch - dramatically faster than single process!
Uses torch.multiprocessing to parallelize batch processing across CPU cores.
"""
import os
os.environ['OMP_NUM_THREADS'] = '4'  # Optimize OpenMP threads per process

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from pathlib import Path
import time
import sys
import logging

# Import the production dataset
sys.path.append(str(Path(__file__).parent))
from models.eeg_dataset_production import ProductionEEGDataset

# Configuration
CONFIG = {
    'data_dir': '/home/kevin/Projects/eeg2025/data/raw/hbn',
    'cache_dir': '/home/kevin/Projects/eeg2025/data/cache',
    'checkpoint_dir': '/home/kevin/Projects/eeg2025/checkpoints',
    'log_file': f'/home/kevin/Projects/eeg2025/logs/training_multiprocess_{time.strftime("%Y%m%d_%H%M%S")}.log',

    # Training params - optimized for speed
    'num_workers': 8,  # Number of parallel processes
    'batch_size': 32,  # Batch size PER WORKER (total = 32 * 8 = 256)
    'epochs': 5,  # Fewer epochs for faster iteration
    'learning_rate': 2e-4,  # Slightly higher LR for larger effective batch size
    'max_subjects': 10,

    # Model params
    'n_channels': 129,
    'sequence_length': 1000,
    'hidden_size': 256,
    'num_heads': 8,
    'num_layers': 4,
    'dropout': 0.1,
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(CONFIG['log_file']),
        logging.StreamHandler()
    ]
)


class TransformerEEGModel(nn.Module):
    """Same model as before"""
    def __init__(self, n_channels, sequence_length, hidden_size, num_heads, num_layers, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(n_channels, hidden_size)
        self.pos_encoding = nn.Parameter(torch.randn(1, sequence_length, hidden_size) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, channels, time) -> (batch, time, channels)
        x = self.input_proj(x)
        x = x + self.pos_encoding
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.classifier(x)
        return x.squeeze(-1)


def setup_distributed(rank, world_size):
    """Setup distributed training environment"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)


def cleanup_distributed():
    """Cleanup distributed training"""
    dist.destroy_process_group()


def train_worker(rank, world_size, config):
    """Training function for each worker process"""

    # Setup distributed
    setup_distributed(rank, world_size)

    # Only log from rank 0
    if rank == 0:
        logging.info(f"Starting multi-process training with {world_size} workers")
        logging.info(f"Effective batch size: {config['batch_size']} x {world_size} = {config['batch_size'] * world_size}")

    # Create dataset
    dataset = ProductionEEGDataset(
        data_dir=config['data_dir'],
        cache_dir=config['cache_dir'],
        max_subjects=config['max_subjects'],
        window_size=2.0,
        overlap=0.5,
        verbose=(rank == 0)  # Only print from main process
    )

    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=2,  # 2 data loading threads per worker
        pin_memory=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        sampler=val_sampler,
        num_workers=2,
        pin_memory=False
    )

    # Create model
    model = TransformerEEGModel(
        n_channels=config['n_channels'],
        sequence_length=config['sequence_length'],
        hidden_size=config['hidden_size'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )

    # Wrap with DDP
    model = DDP(model)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=1e-5)

    # Training loop
    for epoch in range(config['epochs']):
        train_sampler.set_epoch(epoch)  # Shuffle differently each epoch

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        start_time = time.time()

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).long()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            if rank == 0 and batch_idx % 50 == 0:
                logging.info(f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] - Loss: {loss.item():.4f}")

        # Gather metrics from all workers
        train_loss_tensor = torch.tensor([train_loss]).float()
        train_correct_tensor = torch.tensor([train_correct]).float()
        train_total_tensor = torch.tensor([train_total]).float()

        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(train_total_tensor, op=dist.ReduceOp.SUM)

        avg_train_loss = train_loss_tensor.item() / (len(train_loader) * world_size)
        avg_train_acc = 100.0 * train_correct_tensor.item() / train_total_tensor.item()

        epoch_time = time.time() - start_time

        if rank == 0:
            logging.info(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.2f}%, Time: {epoch_time:.1f}s")

            # Save checkpoint
            if (epoch + 1) % 2 == 0:
                checkpoint_path = Path(config['checkpoint_dir']) / f'multiprocess_epoch_{epoch+1}.pth'
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'train_acc': avg_train_acc,
                }, checkpoint_path)
                logging.info(f"Saved checkpoint: {checkpoint_path}")

    cleanup_distributed()


def main():
    """Main entry point"""
    # Check number of available CPUs
    num_cpus = os.cpu_count()
    world_size = min(CONFIG['num_workers'], num_cpus)

    logging.info(f"System has {num_cpus} CPUs")
    logging.info(f"Will use {world_size} parallel workers")

    # Spawn worker processes
    mp.spawn(
        train_worker,
        args=(world_size, CONFIG),
        nprocs=world_size,
        join=True
    )

    logging.info("Training complete!")


if __name__ == '__main__':
    main()
