"""
Challenge 1 Trainer: Enhanced Cross-Task Transfer Learning
==========================================================

Implements advanced transfer learning from SuS to CCD tasks with:
- Multi-stage transfer learning pipeline
- Progressive unfreezing strategies
- Advanced domain alignment techniques
- Official challenge metric optimization
"""

import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import wandb
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from scipy.stats import pearsonr, spearmanr

from ..models.advanced_foundation_model import AdvancedEEGFoundationModel
from ..dataio.hbn_dataset import HBNDataset, create_hbn_datasets
from ..utils.domain_adaptation import DomainAdapter
from ..models.heads.regression import TemporalRegressionHead
from ..models.heads.classification import CalibratedClassificationHead

logger = logging.getLogger(__name__)


@dataclass
class Challenge1Config:
    """Configuration for Challenge 1 training."""

    # Model architecture
    backbone_type: str = "transformer"
    hidden_dim: int = 768
    num_layers: int = 12

    # Transfer learning stages
    ssl_checkpoint: str = "runs/ssl_pretrain/best.ckpt"
    sus_checkpoint: Optional[str] = None
    progressive_unfreezing: bool = True
    unfreeze_schedule: List[float] = None  # [0.25, 0.5, 0.75, 1.0]

    # Task-specific heads
    rt_head_config: Dict[str, Any] = None
    success_head_config: Dict[str, Any] = None

    # Domain adaptation
    use_subject_adaptation: bool = True
    use_site_adaptation: bool = True
    domain_adaptation_weight: float = 0.1

    # Loss configuration
    rt_loss_weight: float = 1.0
    success_loss_weight: float = 1.0
    domain_loss_weight: float = 0.1

    # Training configuration
    batch_size: int = 32
    max_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 5

    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4
    monitor_metric: str = "val_combined_score"

    # Official metrics
    rt_correlation_weight: float = 0.5
    success_accuracy_weight: float = 0.5

    def __post_init__(self):
        if self.unfreeze_schedule is None:
            self.unfreeze_schedule = [0.25, 0.5, 0.75, 1.0]

        if self.rt_head_config is None:
            self.rt_head_config = {
                "hidden_dims": [512, 256, 128],
                "dropout": 0.3,
                "use_temporal_attention": True,
                "temporal_window": 5
            }

        if self.success_head_config is None:
            self.success_head_config = {
                "hidden_dims": [512, 256],
                "dropout": 0.3,
                "use_calibration": True,
                "calibration_bins": 15
            }


class Challenge1Model(nn.Module):
    """Enhanced model for Challenge 1 with specialized heads."""

    def __init__(self, config: Challenge1Config, n_channels: int = 128):
        super().__init__()
        self.config = config
        self.n_channels = n_channels

        # Create foundation model backbone
        from ..models.advanced_foundation_model import FoundationModelConfig
        foundation_config = FoundationModelConfig(
            backbone_type=config.backbone_type,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            use_domain_adaptation=True,
            use_compression_ssl=False,
            use_gpu_optimization=True
        )

        self.foundation_model = AdvancedEEGFoundationModel(foundation_config)

        # Specialized task heads
        self.rt_head = TemporalRegressionHead(
            input_dim=config.hidden_dim,
            **config.rt_head_config
        )

        self.success_head = CalibratedClassificationHead(
            input_dim=config.hidden_dim,
            num_classes=2,
            **config.success_head_config
        )

        # Domain adapters
        if config.use_subject_adaptation:
            self.subject_adapter = DomainAdapter(
                feature_dim=config.hidden_dim,
                num_tasks=2,  # SuS, CCD
                adapter_dim=128
            )

        # Track unfreezing progress
        self.current_unfreeze_ratio = 0.0

    def progressive_unfreeze(self, epoch: int, total_epochs: int):
        """Apply progressive unfreezing based on training progress."""
        if not self.config.progressive_unfreezing:
            return

        progress = epoch / total_epochs
        target_ratio = 0.0

        for ratio in self.config.unfreeze_schedule:
            if progress >= ratio / len(self.config.unfreeze_schedule):
                target_ratio = ratio

        if target_ratio > self.current_unfreeze_ratio:
            self._unfreeze_backbone_layers(target_ratio)
            self.current_unfreeze_ratio = target_ratio

    def _unfreeze_backbone_layers(self, ratio: float):
        """Unfreeze backbone layers up to specified ratio."""
        all_params = list(self.foundation_model.task_backbone.parameters())
        n_params_to_unfreeze = int(len(all_params) * ratio)

        # Freeze all first
        for param in all_params:
            param.requires_grad = False

        # Unfreeze from the end (top layers first)
        for param in all_params[-n_params_to_unfreeze:]:
            param.requires_grad = True

        logger.info(f"Unfroze {n_params_to_unfreeze}/{len(all_params)} backbone parameters")

    def forward(
        self,
        x: torch.Tensor,
        task_ids: torch.Tensor,
        subject_ids: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with task-specific processing."""

        # Extract features from foundation model
        foundation_outputs = self.foundation_model(
            x=x,
            task_ids=task_ids,
            domain_ids={"subject": subject_ids} if subject_ids is not None else None,
            mode="training"
        )

        features = foundation_outputs["features"]

        # Apply subject adaptation if enabled
        if hasattr(self, 'subject_adapter') and subject_ids is not None:
            adapted_features = self.subject_adapter(features, task_ids)
        else:
            adapted_features = features

        # Task-specific predictions
        rt_prediction = self.rt_head(adapted_features)
        success_prediction = self.success_head(adapted_features)

        outputs = {
            "rt_prediction": rt_prediction,
            "success_prediction": success_prediction,
            "task_embeddings": foundation_outputs.get("task_embeddings", None)
        }

        # Add domain losses if available
        if "domain_loss_subject" in foundation_outputs:
            outputs["domain_loss_subject"] = foundation_outputs["domain_loss_subject"]

        if return_features:
            outputs["features"] = features
            outputs["adapted_features"] = adapted_features

        return outputs


class Challenge1Trainer:
    """Enhanced trainer for Challenge 1 with official metrics."""

    def __init__(
        self,
        config: Challenge1Config,
        model: Challenge1Model,
        device: torch.device,
        log_dir: Optional[str] = None
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=config.max_epochs // 4,
            T_mult=2,
            eta_min=config.learning_rate * 0.01
        )

        # Loss functions
        self.rt_loss_fn = nn.MSELoss()
        self.success_loss_fn = nn.CrossEntropyLoss(weight=self._compute_class_weights())

        # Metrics tracking
        self.best_metric = float('-inf')
        self.patience_counter = 0

        # Logging
        if log_dir:
            self.writer = SummaryWriter(log_dir)
            self.log_dir = Path(log_dir)
        else:
            self.writer = None
            self.log_dir = None

    def _compute_class_weights(self) -> torch.Tensor:
        """Compute class weights for balanced training."""
        # These should be computed from actual data
        # Placeholder for now
        return torch.tensor([1.0, 1.0], device=self.device)

    def load_ssl_checkpoint(self, checkpoint_path: str):
        """Load SSL pretrained weights."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Extract SSL weights and load into foundation model
            ssl_state_dict = checkpoint.get('model_state_dict', checkpoint)

            # Filter weights for foundation model
            foundation_dict = {}
            for key, value in ssl_state_dict.items():
                if key.startswith('backbone.') or key.startswith('task_backbone.'):
                    foundation_dict[key] = value

            # Load with strict=False to allow missing keys
            missing_keys, unexpected_keys = self.model.foundation_model.load_state_dict(
                foundation_dict, strict=False
            )

            logger.info(f"Loaded SSL checkpoint from {checkpoint_path}")
            logger.info(f"Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")

        except Exception as e:
            logger.error(f"Failed to load SSL checkpoint: {e}")

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(
            x=batch["eeg"],
            task_ids=batch.get("task_label", torch.zeros(batch["eeg"].size(0), device=self.device)),
            subject_ids=batch.get("subject_id", None)
        )

        # Compute losses
        losses = {}
        total_loss = 0.0

        # Response time loss (only for CCD samples)
        if "mean_rt" in batch:
            rt_mask = ~torch.isnan(batch["mean_rt"])
            if rt_mask.any():
                rt_loss = self.rt_loss_fn(
                    outputs["rt_prediction"][rt_mask],
                    batch["mean_rt"][rt_mask].unsqueeze(-1)
                )
                losses["rt_loss"] = rt_loss
                total_loss += self.config.rt_loss_weight * rt_loss

        # Success classification loss
        if "success_label" in batch:
            success_loss = self.success_loss_fn(
                outputs["success_prediction"],
                batch["success_label"]
            )
            losses["success_loss"] = success_loss
            total_loss += self.config.success_loss_weight * success_loss

        # Domain adaptation losses
        if "domain_loss_subject" in outputs:
            domain_loss = outputs["domain_loss_subject"]
            losses["domain_loss"] = domain_loss
            total_loss += self.config.domain_loss_weight * domain_loss

        losses["total_loss"] = total_loss

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Convert to float for logging
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}

    def validate_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Single validation step."""
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(
                x=batch["eeg"],
                task_ids=batch.get("task_label", torch.zeros(batch["eeg"].size(0), device=self.device)),
                subject_ids=batch.get("subject_id", None)
            )

        # Collect predictions and targets
        predictions = {
            "rt_pred": outputs["rt_prediction"].cpu().numpy(),
            "success_pred": torch.softmax(outputs["success_prediction"], dim=-1).cpu().numpy()
        }

        targets = {}
        if "mean_rt" in batch:
            targets["rt_true"] = batch["mean_rt"].cpu().numpy()
        if "success_label" in batch:
            targets["success_true"] = batch["success_label"].cpu().numpy()

        return predictions, targets

    def compute_official_metrics(
        self,
        all_predictions: List[Dict[str, np.ndarray]],
        all_targets: List[Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """Compute official Challenge 1 metrics."""

        # Concatenate all predictions and targets
        rt_preds = np.concatenate([p["rt_pred"] for p in all_predictions if "rt_pred" in p])
        success_preds = np.concatenate([p["success_pred"] for p in all_predictions if "success_pred" in p])

        rt_targets = np.concatenate([t["rt_true"] for t in all_targets if "rt_true" in t])
        success_targets = np.concatenate([t["success_true"] for t in all_targets if "success_true" in t])

        metrics = {}

        # Response time correlation (official metric)
        if len(rt_preds) > 0 and len(rt_targets) > 0:
            # Remove NaN values
            valid_mask = ~(np.isnan(rt_preds.flatten()) | np.isnan(rt_targets))
            if valid_mask.sum() > 10:  # Need sufficient samples
                rt_corr, rt_p_value = pearsonr(
                    rt_preds.flatten()[valid_mask],
                    rt_targets[valid_mask]
                )
                metrics["rt_correlation"] = rt_corr
                metrics["rt_p_value"] = rt_p_value
            else:
                metrics["rt_correlation"] = 0.0
                metrics["rt_p_value"] = 1.0

        # Success classification accuracy (official metric)
        if len(success_preds) > 0 and len(success_targets) > 0:
            success_pred_labels = np.argmax(success_preds, axis=1)
            balanced_acc = balanced_accuracy_score(success_targets, success_pred_labels)
            metrics["success_balanced_accuracy"] = balanced_acc

            # Additional metrics
            accuracy = np.mean(success_pred_labels == success_targets)
            metrics["success_accuracy"] = accuracy

            if len(np.unique(success_targets)) > 1:
                auc = roc_auc_score(success_targets, success_preds[:, 1])
                metrics["success_auc"] = auc

        # Combined official score
        rt_score = metrics.get("rt_correlation", 0.0)
        success_score = metrics.get("success_balanced_accuracy", 0.0)

        combined_score = (
            self.config.rt_correlation_weight * rt_score +
            self.config.success_accuracy_weight * success_score
        )
        metrics["combined_score"] = combined_score

        return metrics

    def train_epoch(self, train_loader, val_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""

        # Apply progressive unfreezing
        self.model.progressive_unfreeze(epoch, self.config.max_epochs)

        # Training phase
        train_losses = []
        self.model.train()

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(self.device)

            # Training step
            losses = self.train_step(batch)
            train_losses.append(losses)

            # Log batch metrics
            if self.writer and batch_idx % 50 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                for key, value in losses.items():
                    self.writer.add_scalar(f"train_batch/{key}", value, global_step)

        # Validation phase
        val_predictions = []
        val_targets = []

        self.model.eval()
        for batch in val_loader:
            # Move batch to device
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(self.device)

            predictions, targets = self.validate_step(batch)
            val_predictions.append(predictions)
            val_targets.append(targets)

        # Compute epoch metrics
        train_metrics = {}
        if train_losses:
            for key in train_losses[0].keys():
                train_metrics[f"train_{key}"] = np.mean([loss[key] for loss in train_losses])

        val_metrics = self.compute_official_metrics(val_predictions, val_targets)
        val_metrics = {f"val_{k}": v for k, v in val_metrics.items()}

        # Combine metrics
        epoch_metrics = {**train_metrics, **val_metrics}

        # Learning rate step
        self.scheduler.step()

        # Logging
        if self.writer:
            for key, value in epoch_metrics.items():
                self.writer.add_scalar(f"epoch/{key}", value, epoch)

        return epoch_metrics

    def fit(self, train_loader, val_loader) -> Dict[str, List[float]]:
        """Full training loop."""

        # Load SSL checkpoint if provided
        if self.config.ssl_checkpoint:
            self.load_ssl_checkpoint(self.config.ssl_checkpoint)

        history = {"train_loss": [], "val_combined_score": []}

        logger.info(f"Starting Challenge 1 training for {self.config.max_epochs} epochs")

        for epoch in range(self.config.max_epochs):
            start_time = time.time()

            # Train epoch
            metrics = self.train_epoch(train_loader, val_loader, epoch)

            # Track history
            history["train_loss"].append(metrics.get("train_total_loss", 0.0))
            history["val_combined_score"].append(metrics.get("val_combined_score", 0.0))

            # Early stopping check
            current_metric = metrics.get(f"val_{self.config.monitor_metric.replace('val_', '')}", 0.0)

            if current_metric > self.best_metric + self.config.min_delta:
                self.best_metric = current_metric
                self.patience_counter = 0

                # Save best model
                if self.log_dir:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'best_metric': self.best_metric,
                        'config': self.config
                    }, self.log_dir / "best_model.ckpt")

            else:
                self.patience_counter += 1

            # Log epoch summary
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{self.config.max_epochs} - "
                f"Time: {epoch_time:.2f}s - "
                f"Train Loss: {metrics.get('train_total_loss', 0.0):.4f} - "
                f"Val Combined: {metrics.get('val_combined_score', 0.0):.4f} - "
                f"Best: {self.best_metric:.4f}"
            )

            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        if self.writer:
            self.writer.close()

        return history
