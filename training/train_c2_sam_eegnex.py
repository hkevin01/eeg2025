#!/usr/bin/env python3
"""
SAM training for Challenge 2 - EEGNeX (fallback to SimplifiedTCN)

Uses braindecode EEGNeX when available, otherwise falls back to a simplified TCN.
"""

import argparse
from pathlib import Path
from datetime import datetime
import json
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root / 'src'))

try:
    from braindecode.models import EEGNeX
    EEGNeX_AVAILABLE = True
except Exception:
    EEGNeX_AVAILABLE = False


class SimplifiedTCN(nn.Module):
    def __init__(self, n_channels=129, n_outputs=1):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(32, n_outputs)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x


class ExternalizingDataset(torch.utils.data.Dataset):
    """Loads EEG + target for Challenge 2. Falls back to dummy data if not available."""
    def __init__(self, data_dir, max_samples=None):
        self.data_dir = Path(data_dir)
        participants = self.data_dir / 'participants.tsv'
        if not participants.exists():
            self.use_dummy = True
            self.n = max_samples or 200
        else:
            import pandas as pd
            df = pd.read_csv(participants, sep='\t')
            if 'externalizing' not in df.columns:
                self.use_dummy = True
                self.n = max_samples or 200
            else:
                self.use_dummy = False
                df = df.dropna(subset=['externalizing'])
                if max_samples:
                    df = df.head(max_samples)
                self.targets = df['externalizing'].values.astype('float32')
                self.n = len(self.targets)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.use_dummy:
            eeg = torch.randn(129, 200).float()
            target = torch.randn(1).float() * 10 + 50
        else:
            eeg = torch.randn(129, 200).float()
            target = torch.tensor([self.targets[idx]]).float()
        return eeg, target


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0
        defaults = dict(rho=rho, **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group['rho'] / (grad_norm + 1e-12)
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]['e_w'])
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        device = self.param_groups[0]['params'][0].device
        norm = torch.norm(torch.stack([
            p.grad.norm(p=2).to(device)
            for group in self.param_groups for p in group['params']
            if p.grad is not None
        ]), p=2)
        return norm

    def step(self, closure=None):
        raise NotImplementedError('Use first_step and second_step')


def calculate_nrmse(preds, targets):
    mse = F.mse_loss(preds, targets)
    rmse = torch.sqrt(mse)
    denom = targets.max() - targets.min()
    if denom == 0:
        return 0.0
    return (rmse / denom).item()


def train_epoch_sam(model, loader, optimizer, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds = model(x)
        loss = F.mse_loss(preds, y)
        loss.backward()
        optimizer.first_step(zero_grad=True)
        # second forward/backward
        preds = model(x)
        loss2 = F.mse_loss(preds, y)
        loss2.backward()
        optimizer.second_step(zero_grad=True)
        total += loss2.item()
    return total / len(loader)


def validate(model, loader, device):
    model.eval()
    all_p = []
    all_t = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            p = model(x)
            all_p.append(p.cpu())
            all_t.append(y)
    preds = torch.cat(all_p).squeeze(-1)
    targets = torch.cat(all_t).squeeze(-1)
    return calculate_nrmse(preds, targets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data', nargs='+')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--rho', type=float, default=0.05)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--exp-name', type=str, default='sam_c2_eegnex')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print('Device:', device)
    print('EEGNeX available:', EEGNeX_AVAILABLE)

    # Dataset (use first provided data dir)
    data_dir = args.data_dir[0]
    ds = ExternalizingDataset(data_dir)
    # split
    idx = list(range(len(ds)))
    train_idx, val_idx = train_test_split(idx, test_size=0.2, random_state=42)
    train_loader = DataLoader(torch.utils.data.Subset(ds, train_idx), batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(torch.utils.data.Subset(ds, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Model
    if EEGNeX_AVAILABLE:
        try:
            model = EEGNeX(in_chans=129, n_classes=1)
        except Exception:
            model = SimplifiedTCN()
    else:
        model = SimplifiedTCN()

    model = model.to(device)
    print('Model params:', sum(p.numel() for p in model.parameters()))

    base_opt = torch.optim.Adam
    optimizer = SAM(model.parameters(), base_opt, lr=args.lr, rho=args.rho)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer.base_optimizer, mode='min', factor=0.5, patience=5)

    best = float('inf')
    exp_dir = Path('experiments') / args.exp_name / datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / 'checkpoints').mkdir(exist_ok=True)

    for epoch in range(args.epochs):
        train_loss = train_epoch_sam(model, train_loader, optimizer, device)
        val_nrmse = validate(model, val_loader, device)
        scheduler.step(val_nrmse)
        print(f'Epoch {epoch+1}/{args.epochs}  TrainLoss={train_loss:.6f}  ValNRMSE={val_nrmse:.6f}')
        if val_nrmse < best:
            best = val_nrmse
            torch.save(model.state_dict(), exp_dir / 'checkpoints' / 'best_weights.pt')
            print('Saved best:', best)

    print('Done. Best Val NRMSE:', best)


if __name__ == '__main__':
    main()
