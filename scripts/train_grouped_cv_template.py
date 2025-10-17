#!/usr/bin/env python3
"""
Release-Grouped Cross-Validation Training Template

Implements:
- 3-fold grouped CV (by release)
- Robust loss (Huber)
- Optional CORAL domain alignment
- Per-sample residual reweighting
- Ensemble-ready (saves OOF predictions)

Competition-compliant:
- No test-time leakage
- No external data
- Physiologically valid augmentations only
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import mean_squared_error

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# =============================================================================
# Configuration
# =============================================================================

class Config:
    """Training configuration"""
    
    # Seed & Device
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Data
    batch_size: int = 32
    num_workers: int = 4
    n_folds: int = 3  # R1+R2→R3, R1+R3→R2, R2+R3→R1
    
    # Loss (huber more robust to outliers than MSE)
    loss_type: str = "huber"  # 'mse' or 'huber'
    huber_delta: float = 1.0
    
    # Regularization
    weight_decay: float = 1e-4
    dropout: float = 0.5
    
    # Training
    epochs: int = 50
    lr: float = 1e-3
    scheduler: str = "cosine"  # 'cosine', 'plateau', or 'none'
    patience: int = 10  # early stopping
    gradient_clip_norm: Optional[float] = 1.0
    
    # CORAL domain alignment (optional)
    use_coral: bool = False  # Set True for Phase 2
    coral_lambda: float = 1e-3  # 1e-3 to 1e-2
    
    # Robust reweighting (downweight large residuals after warmup)
    use_residual_reweight: bool = True
    reweight_warmup_epochs: int = 5
    reweight_clip: float = 3.0  # clip residuals at 3 std
    
    # Ensemble
    ensemble_agg: str = "median"  # 'mean', 'median', 'weighted'
    
    # Paths
    weights_dir: Path = PROJECT_ROOT / "weights"
    logs_dir: Path = PROJECT_ROOT / "logs"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"


CFG = Config()


# =============================================================================
# Utilities
# =============================================================================

def set_seed(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def nrmse(y_true, y_pred):
    """Normalized RMSE (divided by std of targets)"""
    eps = 1e-8
    denom = np.std(y_true) + eps
    return rmse(y_true, y_pred) / denom


def huber_loss_torch(pred, target, delta=1.0):
    """Huber loss (robust to outliers)"""
    err = pred - target
    abs_err = err.abs()
    quad = torch.clamp(abs_err, max=delta)
    lin = abs_err - quad
    return (0.5 * quad**2 + delta * lin).mean()


def coral_loss(h_s: torch.Tensor, h_t: torch.Tensor) -> torch.Tensor:
    """CORAL: Correlation Alignment between source and target features
    
    Args:
        h_s: Source features [B_s, F]
        h_t: Target features [B_t, F]
    
    Returns:
        CORAL loss (squared Frobenius norm of covariance difference)
    """
    if h_s.numel() == 0 or h_t.numel() == 0:
        return torch.tensor(0.0, device=h_s.device)
    
    bs = h_s.size(0)
    bt = h_t.size(0)
    
    # Center features
    hs = h_s - h_s.mean(dim=0, keepdim=True)
    ht = h_t - h_t.mean(dim=0, keepdim=True)
    
    # Compute covariances
    cs = (hs.T @ hs) / (bs - 1 + 1e-8)
    ct = (ht.T @ ht) / (bt - 1 + 1e-8)
    
    # Frobenius norm of difference
    return ((cs - ct) ** 2).sum()


# =============================================================================
# Training & Validation
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    device: str,
    epoch: int,
    total_epochs: int,
    coral_lambda: float = 0.0,
) -> float:
    """Train for one epoch"""
    model.train()
    
    total_loss = 0.0
    n_samples = 0
    
    for batch_idx, batch in enumerate(train_loader):
        x, y, meta = batch
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float().view(-1)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass
        if CFG.use_coral and hasattr(model, 'forward_with_features'):
            pred, feat = model.forward_with_features(x)
            pred = pred.view(-1)
        else:
            pred = model(x).view(-1)
        
        # Base loss
        if CFG.loss_type == "mse":
            base_loss = nn.MSELoss()(pred, y)
        elif CFG.loss_type == "huber":
            base_loss = huber_loss_torch(pred, y, delta=CFG.huber_delta)
        else:
            raise ValueError(f"Unknown loss_type: {CFG.loss_type}")
        
        loss = base_loss
        
        # CORAL alignment across releases (if enabled)
        if CFG.use_coral and hasattr(model, 'forward_with_features'):
            releases = meta["release_id"]
            unique_rel = torch.unique(releases)
            
            coral_terms = []
            for i in range(len(unique_rel)):
                for j in range(i+1, len(unique_rel)):
                    ri = unique_rel[i]
                    rj = unique_rel[j]
                    hi = feat[releases == ri]
                    hj = feat[releases == rj]
                    if hi.size(0) > 1 and hj.size(0) > 1:
                        coral_terms.append(coral_loss(hi, hj))
            
            if len(coral_terms) > 0:
                loss = loss + coral_lambda * torch.stack(coral_terms).mean()
        
        # Residual-based reweighting (after warmup)
        if CFG.use_residual_reweight and epoch >= CFG.reweight_warmup_epochs:
            with torch.no_grad():
                res = (pred - y)
                std = res.std().clamp(min=1e-6)
                z = (res / std).abs()
                w = torch.clamp(CFG.reweight_clip / (z + 1e-6), max=1.0)
                w = w.detach()
            
            # Re-compute weighted loss
            if CFG.loss_type == "mse":
                loss = ((pred - y) ** 2 * w).mean()
            elif CFG.loss_type == "huber":
                err = pred - y
                abs_err = err.abs()
                delta = CFG.huber_delta
                quad = torch.clamp(abs_err, max=delta)
                lin = abs_err - quad
                loss = (w * (0.5 * quad**2 + delta * lin)).mean()
        
        # Backward pass
        loss.backward()
        
        if CFG.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.gradient_clip_norm)
        
        optimizer.step()
        
        bs = y.size(0)
        total_loss += loss.item() * bs
        n_samples += bs
    
    return total_loss / max(1, n_samples)


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: str,
    target_std: Optional[float] = None,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Validate model"""
    model.eval()
    
    y_true = []
    y_pred = []
    
    for batch in val_loader:
        x, y, meta = batch
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).float().view(-1)
        
        pred = model(x).view(-1)
        
        y_true.append(y.cpu().numpy())
        y_pred.append(pred.cpu().numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    
    val_rmse = rmse(y_true, y_pred)
    
    # NRMSE
    denom = target_std if target_std is not None else (np.std(y_true) + 1e-8)
    val_nrmse = val_rmse / denom
    
    return val_rmse, val_nrmse, y_true, y_pred


def get_scheduler(optimizer: optim.Optimizer, total_epochs: int):
    """Get learning rate scheduler"""
    if CFG.scheduler == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)
    elif CFG.scheduler == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, verbose=True
        )
    else:
        return None


# =============================================================================
# Release-Grouped CV
# =============================================================================

def get_release_grouped_folds(dataset, releases=['R1', 'R2', 'R3']):
    """Create 3 folds by holding out each release
    
    Args:
        dataset: Dataset with release_id in metadata
        releases: List of release names
    
    Returns:
        List of (train_indices, val_indices) tuples
    """
    print(f"\n📦 Creating release-grouped folds for: {releases}")
    
    # Build index mapping by release
    release_indices = {r: [] for r in releases}
    
    for idx in range(len(dataset)):
        _, _, meta = dataset[idx]
        rel = meta['release_id']
        if rel in release_indices:
            release_indices[rel].append(idx)
    
    # Print statistics
    for rel in releases:
        print(f"  {rel}: {len(release_indices[rel])} samples")
    
    # Create folds: each holds out one release
    folds = []
    for val_release in releases:
        train_idx = []
        for rel in releases:
            if rel != val_release:
                train_idx.extend(release_indices[rel])
        
        val_idx = release_indices[val_release]
        
        print(f"\nFold {len(folds)}: Train on {[r for r in releases if r != val_release]}, Val on {val_release}")
        print(f"  Train: {len(train_idx)} samples")
        print(f"  Val: {len(val_idx)} samples")
        
        folds.append((train_idx, val_idx))
    
    return folds


def compute_target_std(loader: DataLoader) -> float:
    """Compute std of targets for NRMSE normalization"""
    ys = []
    for _, y, _ in loader:
        ys.append(y.view(-1).numpy())
    y = np.concatenate(ys)
    return float(np.std(y) + 1e-8)


# =============================================================================
# Main Training Loop
# =============================================================================

def run_fold(
    fold_id: int,
    train_idx: List[int],
    val_idx: List[int],
    dataset,
    model_factory,
    challenge_name: str,
) -> Dict:
    """Train one fold"""
    
    print(f"\n{'='*80}")
    print(f"🚀 TRAINING FOLD {fold_id} - {challenge_name}")
    print(f"{'='*80}\n")
    
    device = CFG.device
    
    # Create dataloaders
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(val_idx)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    # Model
    model = model_factory().to(device)
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = get_scheduler(optimizer, CFG.epochs)
    
    # Compute target std for NRMSE (from training data only!)
    target_std = compute_target_std(train_loader)
    print(f"Target std (for NRMSE): {target_std:.4f}")
    
    # Training loop
    best_score = float('inf')
    best_state = None
    best_epoch = -1
    wait = 0
    history = []
    
    for epoch in range(CFG.epochs):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device,
            epoch, CFG.epochs, coral_lambda=CFG.coral_lambda if CFG.use_coral else 0.0
        )
        
        # Validate
        val_rmse, val_nrmse, y_true, y_pred = validate(
            model, val_loader, device, target_std=target_std
        )
        
        score = val_nrmse  # Use NRMSE for early stopping
        
        # Scheduler step
        if CFG.scheduler == "plateau" and scheduler is not None:
            scheduler.step(score)
        elif CFG.scheduler == "cosine" and scheduler is not None:
            scheduler.step()
        
        # Log
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_rmse": float(val_rmse),
            "val_nrmse": float(val_nrmse),
            "lr": float(optimizer.param_groups[0]["lr"]),
        })
        
        # Check improvement
        improved = score < best_score
        if improved:
            best_score = score
            best_state = {
                "model": model.state_dict(),
                "epoch": epoch,
                "val_rmse": float(val_rmse),
                "val_nrmse": float(val_nrmse),
            }
            best_epoch = epoch
            wait = 0
        else:
            wait += 1
        
        # Print progress
        marker = "✓" if improved else ""
        print(f"[Fold {fold_id}] Epoch {epoch:03d} | "
              f"train_loss={train_loss:.4f} | "
              f"val_rmse={val_rmse:.4f} | "
              f"val_nrmse={val_nrmse:.4f} | "
              f"best@{best_epoch}={best_score:.4f} {marker}")
        
        # Early stopping
        if wait >= CFG.patience:
            print(f"\n⚠️  Early stopping at epoch {epoch} (patience={CFG.patience})")
            break
    
    # Load best model
    model.load_state_dict(best_state["model"])
    
    # Get OOF predictions
    _, _, oof_true, oof_pred = validate(model, val_loader, device, target_std=target_std)
    
    # Save model weights
    CFG.weights_dir.mkdir(exist_ok=True)
    weight_path = CFG.weights_dir / f"{challenge_name}_fold_{fold_id}.pt"
    torch.save(model.state_dict(), weight_path)
    print(f"\n💾 Saved weights: {weight_path}")
    
    # Return artifacts
    return {
        "fold_id": fold_id,
        "best_epoch": best_epoch,
        "best_val_rmse": best_state["val_rmse"],
        "best_val_nrmse": best_state["val_nrmse"],
        "oof_true": oof_true.tolist(),
        "oof_pred": oof_pred.tolist(),
        "history": history,
        "weight_path": str(weight_path),
    }


# =============================================================================
# Main Orchestration
# =============================================================================

def main(args):
    """Main training orchestration"""
    
    print(f"\n{'='*80}")
    print(f"🏁 RELEASE-GROUPED CROSS-VALIDATION TRAINING")
    print(f"{'='*80}\n")
    
    print(f"Challenge: {args.challenge}")
    print(f"Releases: {args.releases}")
    print(f"Folds: {CFG.n_folds}")
    print(f"Device: {CFG.device}")
    print(f"Loss: {CFG.loss_type}")
    print(f"CORAL: {CFG.use_coral}")
    print(f"Residual Reweight: {CFG.use_residual_reweight}")
    
    set_seed(CFG.seed)
    
    # TODO: Load your dataset here
    # dataset = YourDataset(releases=args.releases)
    # model_factory = lambda: YourModel()
    
    print("\n⚠️  TODO: Implement dataset loading and model factory")
    print("This is a template - integrate with your existing code!")
    
    # TODO: Uncomment when integrated
    # folds = get_release_grouped_folds(dataset, releases=args.releases)
    #
    # all_artifacts = []
    # all_oof_true = []
    # all_oof_pred = []
    #
    # for fold_id, (train_idx, val_idx) in enumerate(folds):
    #     artifact = run_fold(
    #         fold_id, train_idx, val_idx, dataset,
    #         model_factory, args.challenge
    #     )
    #     all_artifacts.append(artifact)
    #     all_oof_true.append(np.array(artifact["oof_true"]))
    #     all_oof_pred.append(np.array(artifact["oof_pred"]))
    #
    # # Compute overall OOF metrics
    # oof_true = np.concatenate(all_oof_true)
    # oof_pred = np.concatenate(all_oof_pred)
    # oof_rmse = rmse(oof_true, oof_pred)
    # oof_nrmse = nrmse(oof_true, oof_pred)
    #
    # print(f"\n{'='*80}")
    # print(f"📊 OVERALL OOF METRICS")
    # print(f"{'='*80}\n")
    # print(f"RMSE:  {oof_rmse:.4f}")
    # print(f"NRMSE: {oof_nrmse:.4f}")
    #
    # # Save artifacts
    # CFG.artifacts_dir.mkdir(exist_ok=True)
    # with open(CFG.artifacts_dir / f"{args.challenge}_artifacts.json", "w") as f:
    #     json.dump(all_artifacts, f, indent=2)
    #
    # print(f"\n💾 Saved artifacts: {CFG.artifacts_dir / f'{args.challenge}_artifacts.json'}")
    # print(f"\n✅ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge", type=str, required=True,
                        choices=["challenge1", "challenge2"],
                        help="Which challenge to train")
    parser.add_argument("--releases", type=str, nargs="+", default=["R1", "R2", "R3"],
                        help="Releases to use for training")
    parser.add_argument("--use-coral", action="store_true",
                        help="Enable CORAL domain alignment")
    parser.add_argument("--coral-lambda", type=float, default=1e-3,
                        help="CORAL loss weight")
    
    args = parser.parse_args()
    
    # Update config from args
    CFG.use_coral = args.use_coral
    CFG.coral_lambda = args.coral_lambda
    
    main(args)
