"""
Challenge 2 Trainer: Psychopathology Factor Prediction
======================================================

Implements multi-output regression for CBCL psychopathology factors with:
- Subject-invariant feature learning
- Clinical score normalization
- Cross-task generalization
- Official correlation metrics
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from scipy.stats import pearsonr, spearmanr
from sklearn.preprocessing import RobustScaler, StandardScaler
from torch.utils.tensorboard import SummaryWriter

from ..dataio.hbn_dataset import HBNDataset, create_hbn_datasets
from ..models.advanced_foundation_model import AdvancedEEGFoundationModel
from ..models.heads.psychopathology import (
    ClinicalNormalizationLayer,
    PsychopathologyHead,
)
from ..utils.domain_adaptation import DomainAdapter, IRMPenalty

logger = logging.getLogger(__name__)


@dataclass
class Challenge2Config:
    """Configuration for Challenge 2 training."""

    # Model architecture
    backbone_type: str = "transformer"
    hidden_dim: int = 768
    num_layers: int = 12

    # Psychopathology factors
    target_factors: List[str] = (
        None  # ["p_factor", "internalizing", "externalizing", "attention"]
    )
    use_age_normalization: bool = True
    use_demographic_features: bool = True

    # Multi-task learning
    tasks: List[str] = None  # ["RS", "SuS", "MW", "CCD"]
    task_weights: Dict[str, float] = None

    # Subject invariance
    use_subject_invariance: bool = True
    use_irm_penalty: bool = True
    irm_weight: float = 0.1

    # Clinical data processing
    score_normalization: str = "robust"  # "standard", "robust", "quantile"
    missing_data_strategy: str = "median"  # "mean", "median", "zero"
    outlier_threshold: float = 3.0

    # Loss configuration
    correlation_loss_weight: float = 1.0
    mse_loss_weight: float = 0.5
    invariance_loss_weight: float = 0.1

    # Training configuration
    batch_size: int = 64
    max_epochs: int = 100
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_epochs: int = 10

    # Cross-validation
    use_cross_validation: bool = True
    cv_folds: int = 5
    stratify_by_age: bool = True

    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    monitor_metric: str = "val_avg_correlation"

    def __post_init__(self):
        if self.target_factors is None:
            self.target_factors = [
                "p_factor",
                "internalizing",
                "externalizing",
                "attention",
            ]

        if self.tasks is None:
            self.tasks = ["RS", "SuS", "MW", "CCD"]

        if self.task_weights is None:
            self.task_weights = {task: 1.0 for task in self.tasks}


class Challenge2Model(nn.Module):
    """Enhanced model for Challenge 2 with clinical factor prediction."""

    def __init__(self, config: Challenge2Config, n_channels: int = 128):
        super().__init__()
        self.config = config
        self.n_channels = n_channels
        self.num_factors = len(config.target_factors)

        # Create foundation model backbone
        from ..models.advanced_foundation_model import FoundationModelConfig

        foundation_config = FoundationModelConfig(
            backbone_type=config.backbone_type,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            use_domain_adaptation=True,
            use_compression_ssl=False,
            use_gpu_optimization=True,
        )

        self.foundation_model = AdvancedEEGFoundationModel(foundation_config)

        # Clinical normalization layer
        if config.use_age_normalization:
            self.clinical_normalizer = ClinicalNormalizationLayer(
                num_factors=self.num_factors, age_bins=10, use_gender=True
            )

        # Psychopathology prediction head
        self.psych_head = PsychopathologyHead(
            input_dim=config.hidden_dim,
            num_factors=self.num_factors,
            factor_names=config.target_factors,
            use_uncertainty=True,
            use_correlation_loss=True,
        )

        # Subject invariance components
        if config.use_subject_invariance:
            self.subject_adapter = DomainAdapter(
                feature_dim=config.hidden_dim,
                num_tasks=len(config.tasks),
                adapter_dim=256,
            )

        if config.use_irm_penalty:
            self.irm_penalty = IRMPenalty()

        # Task-specific feature extractors
        self.task_extractors = nn.ModuleDict(
            {
                task: nn.Sequential(
                    nn.Linear(config.hidden_dim, config.hidden_dim),
                    nn.LayerNorm(config.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                )
                for task in config.tasks
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        task_ids: torch.Tensor,
        subject_ids: Optional[torch.Tensor] = None,
        age: Optional[torch.Tensor] = None,
        gender: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass with clinical factor prediction."""

        # Extract features from foundation model
        foundation_outputs = self.foundation_model(
            x=x,
            task_ids=task_ids,
            domain_ids={"subject": subject_ids} if subject_ids is not None else None,
            mode="training",
        )

        features = foundation_outputs["features"]

        # Task-specific feature extraction
        task_features = []
        for i, task_id in enumerate(task_ids):
            task_name = self.config.tasks[task_id.item()]
            if task_name in self.task_extractors:
                task_feat = self.task_extractors[task_name](features[i : i + 1])
                task_features.append(task_feat)
            else:
                task_features.append(features[i : i + 1])

        adapted_features = torch.cat(task_features, dim=0)

        # Apply subject invariance if enabled
        if hasattr(self, "subject_adapter"):
            adapted_features = self.subject_adapter(adapted_features, task_ids)

        # Psychopathology predictions
        psych_outputs = self.psych_head(adapted_features)

        # Clinical normalization if enabled
        if hasattr(self, "clinical_normalizer") and age is not None:
            normalized_predictions = self.clinical_normalizer(
                psych_outputs["predictions"], age=age, gender=gender
            )
            psych_outputs["normalized_predictions"] = normalized_predictions

        outputs = {
            "predictions": psych_outputs["predictions"],
            "uncertainties": psych_outputs.get("uncertainties", None),
            "factor_correlations": psych_outputs.get("factor_correlations", None),
        }

        # Add normalized predictions if available
        if "normalized_predictions" in psych_outputs:
            outputs["normalized_predictions"] = psych_outputs["normalized_predictions"]

        # Add domain losses if available
        if "domain_loss_subject" in foundation_outputs:
            outputs["domain_loss_subject"] = foundation_outputs["domain_loss_subject"]

        if return_features:
            outputs["features"] = features
            outputs["adapted_features"] = adapted_features

        return outputs

    def compute_irm_penalty(
        self,
        x: torch.Tensor,
        task_ids: torch.Tensor,
        targets: torch.Tensor,
        subject_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute IRM penalty for subject invariance."""

        if not hasattr(self, "irm_penalty"):
            return torch.tensor(0.0, device=x.device)

        # Split by subjects for IRM
        unique_subjects = torch.unique(subject_ids)

        if len(unique_subjects) < 2:
            return torch.tensor(0.0, device=x.device)

        subject_losses = []

        for subject_id in unique_subjects:
            subject_mask = subject_ids == subject_id
            if subject_mask.sum() < 2:  # Need at least 2 samples
                continue

            subject_x = x[subject_mask]
            subject_task_ids = task_ids[subject_mask]
            subject_targets = targets[subject_mask]

            # Forward pass for this subject
            outputs = self.forward(subject_x, subject_task_ids, return_features=False)

            # Compute loss for this subject
            predictions = outputs["predictions"]
            loss = F.mse_loss(predictions, subject_targets)
            subject_losses.append(loss)

        if len(subject_losses) < 2:
            return torch.tensor(0.0, device=x.device)

        # Compute IRM penalty
        penalty = self.irm_penalty(subject_losses)

        return penalty


class Challenge2Trainer:
    """Enhanced trainer for Challenge 2 with clinical metrics."""

    def __init__(
        self,
        config: Challenge2Config,
        model: Challenge2Model,
        device: torch.device,
        log_dir: Optional[str] = None,
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate * 5,
            total_steps=config.max_epochs,
            pct_start=config.warmup_epochs / config.max_epochs,
            anneal_strategy="cos",
        )

        # Loss functions
        self.mse_loss_fn = nn.MSELoss()
        self.correlation_loss_fn = self._correlation_loss

        # Data scalers for clinical scores
        self.score_scalers = {}
        for factor in config.target_factors:
            if config.score_normalization == "robust":
                self.score_scalers[factor] = RobustScaler()
            else:
                self.score_scalers[factor] = StandardScaler()

        # Metrics tracking
        self.best_metric = float("-inf")
        self.patience_counter = 0

        # Logging
        if log_dir:
            self.writer = SummaryWriter(log_dir)
            self.log_dir = Path(log_dir)
        else:
            self.writer = None
            self.log_dir = None

    def _correlation_loss(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute negative correlation loss."""

        # Center the data
        pred_centered = predictions - predictions.mean(dim=0, keepdim=True)
        target_centered = targets - targets.mean(dim=0, keepdim=True)

        # Compute correlation coefficient
        numerator = (pred_centered * target_centered).sum(dim=0)

        pred_std = torch.sqrt((pred_centered**2).sum(dim=0) + 1e-8)
        target_std = torch.sqrt((target_centered**2).sum(dim=0) + 1e-8)

        correlation = numerator / (pred_std * target_std + 1e-8)

        # Return negative correlation (we want to maximize correlation)
        return -correlation.mean()

    def preprocess_clinical_scores(
        self, scores: Dict[str, np.ndarray], fit: bool = False
    ) -> torch.Tensor:
        """Preprocess clinical scores with normalization."""

        processed_scores = []

        for factor in self.config.target_factors:
            if factor not in scores:
                # Handle missing factors
                factor_scores = np.zeros(len(list(scores.values())[0]))
            else:
                factor_scores = scores[factor]

            # Handle missing values
            if self.config.missing_data_strategy == "median":
                factor_scores = np.where(
                    np.isnan(factor_scores), np.nanmedian(factor_scores), factor_scores
                )
            elif self.config.missing_data_strategy == "mean":
                factor_scores = np.where(
                    np.isnan(factor_scores), np.nanmean(factor_scores), factor_scores
                )
            else:  # zero
                factor_scores = np.where(np.isnan(factor_scores), 0.0, factor_scores)

            # Outlier removal
            if self.config.outlier_threshold > 0:
                mean_score = np.mean(factor_scores)
                std_score = np.std(factor_scores)
                outlier_mask = np.abs(factor_scores - mean_score) > (
                    self.config.outlier_threshold * std_score
                )
                factor_scores[outlier_mask] = np.clip(
                    factor_scores[outlier_mask],
                    mean_score - self.config.outlier_threshold * std_score,
                    mean_score + self.config.outlier_threshold * std_score,
                )

            # Normalization
            scaler = self.score_scalers[factor]

            if fit:
                factor_scores = scaler.fit_transform(
                    factor_scores.reshape(-1, 1)
                ).flatten()
            else:
                factor_scores = scaler.transform(factor_scores.reshape(-1, 1)).flatten()

            processed_scores.append(factor_scores)

        # Stack all factors
        processed_tensor = torch.FloatTensor(np.stack(processed_scores, axis=1))

        return processed_tensor

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.model(
            x=batch["eeg"],
            task_ids=batch.get(
                "task_label", torch.zeros(batch["eeg"].size(0), device=self.device)
            ),
            subject_ids=batch.get("subject_id", None),
            age=batch.get("age", None),
            gender=batch.get("gender", None),
        )

        # Get target clinical scores
        targets = batch["clinical_scores"]  # [batch_size, num_factors]
        predictions = outputs["predictions"]

        # Compute losses
        losses = {}
        total_loss = 0.0

        # MSE loss
        mse_loss = self.mse_loss_fn(predictions, targets)
        losses["mse_loss"] = mse_loss
        total_loss += self.config.mse_loss_weight * mse_loss

        # Correlation loss
        correlation_loss = self.correlation_loss_fn(predictions, targets)
        losses["correlation_loss"] = correlation_loss
        total_loss += self.config.correlation_loss_weight * correlation_loss

        # IRM penalty for subject invariance
        if self.config.use_irm_penalty and "subject_id" in batch:
            irm_penalty = self.model.compute_irm_penalty(
                batch["eeg"],
                batch.get(
                    "task_label", torch.zeros(batch["eeg"].size(0), device=self.device)
                ),
                targets,
                batch["subject_id"],
            )
            losses["irm_penalty"] = irm_penalty
            total_loss += self.config.irm_weight * irm_penalty

        # Domain adaptation losses
        if "domain_loss_subject" in outputs:
            domain_loss = outputs["domain_loss_subject"]
            losses["domain_loss"] = domain_loss
            total_loss += self.config.invariance_loss_weight * domain_loss

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
                task_ids=batch.get(
                    "task_label", torch.zeros(batch["eeg"].size(0), device=self.device)
                ),
                subject_ids=batch.get("subject_id", None),
                age=batch.get("age", None),
                gender=batch.get("gender", None),
            )

        # Collect predictions and targets
        predictions = outputs["predictions"].cpu().numpy()
        targets = batch["clinical_scores"].cpu().numpy()

        return predictions, targets

    def compute_official_metrics(
        self, all_predictions: List[np.ndarray], all_targets: List[np.ndarray]
    ) -> Dict[str, float]:
        """Compute official Challenge 2 metrics."""

        # Concatenate all predictions and targets
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        metrics = {}

        # Per-factor correlations (official metrics)
        factor_correlations = []

        for i, factor in enumerate(self.config.target_factors):
            if predictions.shape[1] > i and targets.shape[1] > i:
                pred_factor = predictions[:, i]
                target_factor = targets[:, i]

                # Remove invalid values
                valid_mask = ~(np.isnan(pred_factor) | np.isnan(target_factor))

                if valid_mask.sum() > 10:  # Need sufficient samples
                    corr, p_value = pearsonr(
                        pred_factor[valid_mask], target_factor[valid_mask]
                    )
                    metrics[f"{factor}_correlation"] = corr
                    metrics[f"{factor}_p_value"] = p_value
                    factor_correlations.append(corr)
                else:
                    metrics[f"{factor}_correlation"] = 0.0
                    metrics[f"{factor}_p_value"] = 1.0
                    factor_correlations.append(0.0)

        # Average correlation (primary metric)
        if factor_correlations:
            metrics["avg_correlation"] = np.mean(factor_correlations)
            metrics["min_correlation"] = np.min(factor_correlations)
            metrics["max_correlation"] = np.max(factor_correlations)

        # Additional metrics
        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))

        metrics["mse"] = mse
        metrics["mae"] = mae

        return metrics

    def train_epoch(self, train_loader, val_loader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""

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
                train_metrics[f"train_{key}"] = np.mean(
                    [loss[key] for loss in train_losses]
                )

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

        history = {"train_loss": [], "val_avg_correlation": []}

        logger.info(
            f"Starting Challenge 2 training for {self.config.max_epochs} epochs"
        )

        for epoch in range(self.config.max_epochs):
            start_time = time.time()

            # Train epoch
            metrics = self.train_epoch(train_loader, val_loader, epoch)

            # Track history
            history["train_loss"].append(metrics.get("train_total_loss", 0.0))
            history["val_avg_correlation"].append(
                metrics.get("val_avg_correlation", 0.0)
            )

            # Early stopping check
            current_metric = metrics.get(
                f"val_{self.config.monitor_metric.replace('val_', '')}", 0.0
            )

            if current_metric > self.best_metric + self.config.min_delta:
                self.best_metric = current_metric
                self.patience_counter = 0

                # Save best model
                if self.log_dir:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "score_scalers": self.score_scalers,
                            "best_metric": self.best_metric,
                            "config": self.config,
                        },
                        self.log_dir / "best_model.ckpt",
                    )

            else:
                self.patience_counter += 1

            # Log epoch summary
            epoch_time = time.time() - start_time
            logger.info(
                f"Epoch {epoch+1}/{self.config.max_epochs} - "
                f"Time: {epoch_time:.2f}s - "
                f"Train Loss: {metrics.get('train_total_loss', 0.0):.4f} - "
                f"Val Avg Corr: {metrics.get('val_avg_correlation', 0.0):.4f} - "
                f"Best: {self.best_metric:.4f}"
            )

            # Early stopping
            if self.patience_counter >= self.config.patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        if self.writer:
            self.writer.close()

        return history
