"""
Psychopathology regression training with domain invariance.

This module implements training for psychopathology regression (CBCL scores) with
domain adversarial training, IRM penalty, and uncertainty weighting for multi-task learning.
"""

import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import wandb
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import pearsonr

from ..models.invariance.dann import DANNModel, GRLScheduler, IRMPenalty, create_dann_model
from ..models.backbones.temporal_cnn import TemporalCNN
from ..models.heads import create_head

logger = logging.getLogger(__name__)


@dataclass
class PsychConfig:
    """Configuration for psychopathology regression training."""

    # Model architecture
    backbone: str = "temporal_cnn"
    backbone_config: Dict = None

    # Task configuration
    target_factors: List[str] = None  # ["p_factor", "internalizing", "externalizing", "attention"]
    normalize_targets: bool = True

    # Domain adversarial training
    use_dann: bool = True
    dann_config: Dict = None
    grl_schedule: Dict = None

    # Site/scanner adversary
    use_site_adversary: bool = False
    site_metadata_key: str = "site"

    # IRM penalty
    use_irm: bool = False
    irm_weight: float = 1.0
    irm_penalty_anneal_iters: int = 500

    # Multi-task uncertainty weighting
    use_uncertainty_weighting: bool = True
    learned_loss_weights: bool = True
    initial_log_vars: Dict[str, float] = None

    # Optimization
    lr: float = 1e-4
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 50

    # Early stopping
    early_stopping_patience: int = 15
    primary_metric: str = "mean_pearson_r"

    # Checkpointing
    ckpt_dir: str = "runs/psych"
    save_every: int = 5

    # Logging
    log_every: int = 20
    eval_every: int = 100

    def __post_init__(self):
        if self.target_factors is None:
            self.target_factors = ["p_factor", "internalizing", "externalizing", "attention"]

        if self.backbone_config is None:
            self.backbone_config = {
                "n_filters": 128,
                "kernel_size": 25,
                "n_blocks": 4
            }

        if self.dann_config is None:
            self.dann_config = {
                "hidden_dims": [128, 64],
                "dropout_rate": 0.3
            }

        if self.grl_schedule is None:
            self.grl_schedule = {
                "strategy": "linear_warmup",
                "initial_lambda": 0.0,
                "final_lambda": 0.2,
                "warmup_steps": 1000
            }

        if self.initial_log_vars is None:
            self.initial_log_vars = {factor: 0.0 for factor in self.target_factors}


class UncertaintyWeightedLoss(nn.Module):
    """
    Multi-task loss with learned uncertainty weighting.

    Implements the approach from "Multi-Task Learning Using Uncertainty to Weigh Losses
    for Scene Geometry and Semantics" (Kendall et al., 2018).
    """

    def __init__(
        self,
        task_names: List[str],
        initial_log_vars: Optional[Dict[str, float]] = None,
        learned_weights: bool = True
    ):
        """
        Initialize uncertainty-weighted loss.

        Args:
            task_names: List of task names
            initial_log_vars: Initial log variance values for each task
            learned_weights: Whether to learn the uncertainty weights
        """
        super().__init__()

        self.task_names = task_names
        self.learned_weights = learned_weights

        if learned_weights:
            # Learnable log variance parameters
            if initial_log_vars is None:
                initial_log_vars = {name: 0.0 for name in task_names}

            self.log_vars = nn.ParameterDict({
                name: nn.Parameter(torch.tensor(initial_log_vars.get(name, 0.0)))
                for name in task_names
            })
        else:
            # Fixed uniform weights
            self.register_buffer('log_vars', torch.zeros(len(task_names)))

    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        base_loss_fn: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute uncertainty-weighted multi-task loss.

        Args:
            predictions: Dictionary of task predictions
            targets: Dictionary of task targets
            base_loss_fn: Base loss function (default MSE)

        Returns:
            Total weighted loss and individual loss components
        """
        if base_loss_fn is None:
            base_loss_fn = nn.MSELoss()

        total_loss = 0.0
        loss_components = {}

        for task_name in self.task_names:
            if task_name not in predictions or task_name not in targets:
                continue

            # Compute base loss
            task_loss = base_loss_fn(predictions[task_name], targets[task_name])

            if self.learned_weights:
                # Get log variance for this task
                log_var = self.log_vars[task_name]

                # Uncertainty-weighted loss: L / (2*σ²) + log(σ²)/2
                precision = torch.exp(-log_var)
                weighted_loss = precision * task_loss + log_var / 2
            else:
                # Uniform weighting
                weighted_loss = task_loss

            total_loss += weighted_loss
            loss_components[f"{task_name}_loss"] = task_loss.item()

            if self.learned_weights:
                loss_components[f"{task_name}_log_var"] = log_var.item()
                loss_components[f"{task_name}_weight"] = precision.item()

        return total_loss, loss_components


class PsychometricMetrics:
    """
    Metrics for psychopathology regression evaluation.

    Computes Pearson correlation, RMSE, and other relevant metrics
    for each psychopathology factor.
    """

    @staticmethod
    def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Pearson correlation coefficient."""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() < 2:
            return 0.0

        corr, _ = pearsonr(y_true[mask], y_pred[mask])
        return corr if not np.isnan(corr) else 0.0

    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Root Mean Square Error."""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() == 0:
            return float('inf')

        return np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))

    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        if mask.sum() == 0:
            return float('inf')

        return np.mean(np.abs(y_true[mask] - y_pred[mask]))

    @classmethod
    def compute_all_metrics(
        cls,
        predictions: Dict[str, np.ndarray],
        targets: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """Compute all metrics for all factors."""
        metrics = {}
        pearson_values = []

        for factor in predictions.keys():
            if factor not in targets:
                continue

            y_true = targets[factor]
            y_pred = predictions[factor]

            # Individual factor metrics
            pearson_r = cls.pearson_correlation(y_true, y_pred)
            rmse_val = cls.rmse(y_true, y_pred)
            mae_val = cls.mae(y_true, y_pred)

            metrics[f"{factor}_pearson_r"] = pearson_r
            metrics[f"{factor}_rmse"] = rmse_val
            metrics[f"{factor}_mae"] = mae_val

            pearson_values.append(pearson_r)

        # Mean metrics across factors
        if pearson_values:
            metrics["mean_pearson_r"] = np.mean(pearson_values)
            metrics["min_pearson_r"] = np.min(pearson_values)
            metrics["max_pearson_r"] = np.max(pearson_values)

        return metrics


class PsychTrainer:
    """
    Trainer for psychopathology regression with domain invariance.

    Supports DANN, IRM, uncertainty weighting, and site adversarial training.
    """

    def __init__(
        self,
        config: PsychConfig,
        model: Union[nn.Module, DANNModel],
        device: torch.device,
        num_sites: Optional[int] = None
    ):
        self.config = config
        self.device = device
        self.num_sites = num_sites

        # Setup model
        if config.use_dann and not isinstance(model, DANNModel):
            # Wrap in DANN if requested
            self.model = self._create_dann_model(model)
        else:
            self.model = model

        self.model = self.model.to(device)

        # Initialize loss functions
        self.losses = self._init_losses()

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )

        # IRM penalty
        if config.use_irm:
            self.irm_penalty = IRMPenalty(penalty_weight=config.irm_weight)

        # Setup logging
        self.setup_logging()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = -float('inf')
        self.patience_counter = 0

        # Metrics tracker
        self.metrics = PsychometricMetrics()

        logger.info("Psychopathology trainer initialized")

    def _create_dann_model(self, base_model: nn.Module) -> DANNModel:
        """Create DANN model from base model."""
        # Create multi-task head
        task_head = create_head(
            "multitask",
            in_features=base_model.feature_dim if hasattr(base_model, 'feature_dim') else 128,
            num_outputs=len(self.config.target_factors),
            hidden_dim=128
        )

        # Determine number of domains
        num_domains = 2  # Default: control vs patient
        if self.config.use_site_adversary and self.num_sites:
            num_domains = self.num_sites

        # Create DANN model
        dann_model = create_dann_model(
            backbone=base_model,
            task_head=task_head,
            num_domains=num_domains,
            lambda_schedule_config=self.config.grl_schedule,
            domain_head_config=self.config.dann_config
        )

        return dann_model

    def _init_losses(self) -> Dict[str, nn.Module]:
        """Initialize loss functions."""
        losses = {}

        # Multi-task uncertainty weighted loss
        if self.config.use_uncertainty_weighting:
            losses["task_loss"] = UncertaintyWeightedLoss(
                task_names=self.config.target_factors,
                initial_log_vars=self.config.initial_log_vars,
                learned_weights=self.config.learned_loss_weights
            )
        else:
            losses["task_loss"] = nn.MSELoss()

        # Domain adversarial loss
        if self.config.use_dann:
            losses["domain_loss"] = nn.CrossEntropyLoss()

        return losses

    def setup_logging(self):
        """Setup logging infrastructure."""
        self.ckpt_dir = Path(self.config.ckpt_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard
        self.tb_writer = SummaryWriter(self.ckpt_dir / "logs")

        # Weights & Biases (optional)
        self.use_wandb = False
        try:
            if wandb.run is None:
                wandb.init(
                    project="eeg2025_psych",
                    config=self.config.__dict__,
                    dir=str(self.ckpt_dir)
                )
            self.use_wandb = True
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}")

    def training_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Single training step."""
        self.model.train()

        # Get batch data
        x = batch["eeg"].to(self.device)
        batch_size = x.shape[0]

        # Get targets
        targets = {}
        for factor in self.config.target_factors:
            if factor in batch:
                targets[factor] = batch[factor].to(self.device).float()

        # Get domain labels if available
        domain_labels = None
        if self.config.use_dann:
            if self.config.use_site_adversary and "site" in batch:
                domain_labels = batch["site"].to(self.device).long()
            elif "domain" in batch:
                domain_labels = batch["domain"].to(self.device).long()
            else:
                # Create dummy domain labels
                domain_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        losses = {}
        total_loss = 0

        # Forward pass
        if isinstance(self.model, DANNModel):
            outputs = self.model(x, return_features=True)
            task_predictions = outputs["task_output"]

            # Convert to factor predictions
            predictions = {}
            for i, factor in enumerate(self.config.target_factors):
                if task_predictions.dim() == 2 and task_predictions.shape[1] > i:
                    predictions[factor] = task_predictions[:, i]
                else:
                    predictions[factor] = task_predictions.squeeze()

            # Task loss
            if self.config.use_uncertainty_weighting:
                task_loss, loss_components = self.losses["task_loss"](predictions, targets)
                losses.update(loss_components)
            else:
                task_loss = 0
                for factor in self.config.target_factors:
                    if factor in predictions and factor in targets:
                        task_loss += self.losses["task_loss"](predictions[factor], targets[factor])
                losses["task_loss"] = task_loss.item()

            total_loss += task_loss

            # Domain adversarial loss
            if domain_labels is not None:
                domain_predictions = outputs["domain_output"]
                domain_loss = self.losses["domain_loss"](domain_predictions, domain_labels)
                losses["domain_loss"] = domain_loss.item()
                total_loss += domain_loss

                # Track domain accuracy
                domain_acc = (domain_predictions.argmax(dim=1) == domain_labels).float().mean().item()
                losses["domain_accuracy"] = domain_acc
                losses["grl_lambda"] = outputs["lambda"]

            # IRM penalty
            if self.config.use_irm and self.global_step > self.config.irm_penalty_anneal_iters:
                irm_loss = self.irm_penalty.compute_penalty(
                    outputs["features"],
                    targets[self.config.target_factors[0]],  # Use first factor as representative
                    domain_labels if domain_labels is not None else torch.zeros_like(targets[self.config.target_factors[0]]),
                    lambda x: predictions[self.config.target_factors[0]]
                )
                losses["irm_penalty"] = irm_loss.item()
                total_loss += irm_loss

        else:
            # Standard model without DANN
            outputs = self.model(x)

            # Convert to factor predictions
            predictions = {}
            for i, factor in enumerate(self.config.target_factors):
                if outputs.dim() == 2 and outputs.shape[1] > i:
                    predictions[factor] = outputs[:, i]
                else:
                    predictions[factor] = outputs.squeeze()

            # Task loss
            if self.config.use_uncertainty_weighting:
                task_loss, loss_components = self.losses["task_loss"](predictions, targets)
                losses.update(loss_components)
            else:
                task_loss = 0
                for factor in self.config.target_factors:
                    if factor in predictions and factor in targets:
                        task_loss += self.losses["task_loss"](predictions[factor], targets[factor])
                losses["task_loss"] = task_loss.item()

            total_loss += task_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        losses["total_loss"] = total_loss.item()
        losses["learning_rate"] = self.optimizer.param_groups[0]["lr"]

        return losses

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """Single validation step."""
        self.model.eval()

        with torch.no_grad():
            x = batch["eeg"].to(self.device)

            # Get targets
            targets = {}
            for factor in self.config.target_factors:
                if factor in batch:
                    targets[factor] = batch[factor].to(self.device).float()

            # Forward pass
            if isinstance(self.model, DANNModel):
                outputs = self.model(x, update_lambda=False)
                task_predictions = outputs["task_output"]
            else:
                task_predictions = self.model(x)

            # Convert to factor predictions
            predictions = {}
            for i, factor in enumerate(self.config.target_factors):
                if task_predictions.dim() == 2 and task_predictions.shape[1] > i:
                    predictions[factor] = task_predictions[:, i].cpu().numpy()
                else:
                    predictions[factor] = task_predictions.squeeze().cpu().numpy()

            # Convert targets to numpy
            targets_np = {}
            for factor in targets:
                targets_np[factor] = targets[factor].cpu().numpy()

            return {
                "predictions": predictions,
                "targets": targets_np
            }

    def train_epoch(self, train_loader, val_loader=None) -> Dict[str, float]:
        """Train for one epoch."""
        epoch_losses = []

        for batch_idx, batch in enumerate(train_loader):
            # Training step
            step_losses = self.training_step(batch)
            epoch_losses.append(step_losses)

            self.global_step += 1

            # Logging
            if batch_idx % self.config.log_every == 0:
                self.log_step(step_losses, batch_idx, len(train_loader))

            # Validation
            if (val_loader is not None and
                batch_idx % self.config.eval_every == 0 and
                batch_idx > 0):
                val_results = self.validate(val_loader)
                self.log_validation(val_results)

        # Aggregate epoch losses
        epoch_metrics = {}
        if epoch_losses:
            for key in epoch_losses[0].keys():
                values = [loss[key] for loss in epoch_losses if key in loss]
                epoch_metrics[f"epoch_{key}"] = np.mean(values) if values else 0.0

        return epoch_metrics

    def validate(self, val_loader) -> Dict[str, float]:
        """Run validation on entire validation set."""
        val_results = []

        for batch in val_loader:
            batch_results = self.validation_step(batch)
            val_results.append(batch_results)

        # Aggregate predictions and targets
        all_predictions = {}
        all_targets = {}

        for factor in self.config.target_factors:
            factor_preds = []
            factor_targets = []

            for result in val_results:
                if factor in result["predictions"]:
                    factor_preds.append(result["predictions"][factor])
                if factor in result["targets"]:
                    factor_targets.append(result["targets"][factor])

            if factor_preds and factor_targets:
                all_predictions[factor] = np.concatenate(factor_preds)
                all_targets[factor] = np.concatenate(factor_targets)

        # Compute metrics
        metrics = self.metrics.compute_all_metrics(all_predictions, all_targets)

        return metrics

    def log_step(self, losses: Dict[str, float], batch_idx: int, total_batches: int):
        """Log training step metrics."""
        log_str = f"Epoch {self.current_epoch} [{batch_idx}/{total_batches}] "
        log_str += f"Total: {losses['total_loss']:.4f}"

        if "domain_loss" in losses:
            log_str += f" | Domain: {losses['domain_loss']:.4f}"
            log_str += f" | λ: {losses.get('grl_lambda', 0):.3f}"

        logger.info(log_str)

        # TensorBoard logging
        for key, value in losses.items():
            self.tb_writer.add_scalar(f"train/{key}", value, self.global_step)

        # W&B logging
        if self.use_wandb:
            wandb.log({f"train/{k}": v for k, v in losses.items()}, step=self.global_step)

    def log_validation(self, metrics: Dict[str, float]):
        """Log validation metrics."""
        log_str = f"Validation - Mean Pearson r: {metrics.get('mean_pearson_r', 0):.4f}"

        for factor in self.config.target_factors:
            r_key = f"{factor}_pearson_r"
            if r_key in metrics:
                log_str += f" | {factor}: {metrics[r_key]:.3f}"

        logger.info(log_str)

        # TensorBoard logging
        for key, value in metrics.items():
            self.tb_writer.add_scalar(f"val/{key}", value, self.global_step)

        # W&B logging
        if self.use_wandb:
            wandb.log({f"val/{k}": v for k, v in metrics.items()}, step=self.global_step)

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config,
            "metrics": metrics,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
        }

        # Save regular checkpoint
        ckpt_path = self.ckpt_dir / f"checkpoint_epoch_{epoch}.ckpt"
        torch.save(checkpoint, ckpt_path)

        # Save best checkpoint
        if is_best:
            best_path = self.ckpt_dir / "best.ckpt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint at epoch {epoch} with {self.config.primary_metric}: {metrics[self.config.primary_metric]:.4f}")

        logger.info(f"Checkpoint saved: {ckpt_path}")

    def train(self, train_loader, val_loader=None) -> Dict[str, List[float]]:
        """Main training loop."""
        logger.info("Starting psychopathology training...")

        history = {"train_loss": [], "val_metrics": []}

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Training epoch
            epoch_metrics = self.train_epoch(train_loader, val_loader)

            # Learning rate scheduling
            self.scheduler.step()

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history["val_metrics"].append(val_metrics)

                # Check for best model
                current_metric = val_metrics.get(self.config.primary_metric, 0)
                is_best = current_metric > self.best_metric

                if is_best:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.patience_counter >= self.config.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch} (patience: {self.patience_counter})")
                    break
            else:
                is_best = False
                val_metrics = {}

            # Record training loss
            train_loss = epoch_metrics.get("epoch_total_loss", 0.0)
            history["train_loss"].append(train_loss)

            # Epoch logging
            epoch_time = time.time() - start_time
            log_str = f"Epoch {epoch}/{self.config.epochs} completed in {epoch_time:.2f}s"
            log_str += f" | Train Loss: {train_loss:.4f}"
            if val_metrics:
                log_str += f" | {self.config.primary_metric}: {val_metrics.get(self.config.primary_metric, 0):.4f}"

            logger.info(log_str)

            # Save checkpoint
            if epoch % self.config.save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)

        # Final checkpoint
        if val_loader is not None:
            final_metrics = self.validate(val_loader)
            self.save_checkpoint(self.config.epochs - 1, final_metrics, False)

        logger.info("Psychopathology training completed!")
        return history


def create_psych_model(config: PsychConfig, n_channels: int = 64) -> nn.Module:
    """
    Factory function to create psychopathology model.

    Args:
        config: Psychopathology configuration
        n_channels: Number of EEG channels

    Returns:
        Model instance
    """
    # Create backbone
    if config.backbone == "temporal_cnn":
        backbone = TemporalCNN(
            n_channels=n_channels,
            **config.backbone_config
        )
    else:
        raise ValueError(f"Unknown backbone: {config.backbone}")

    if config.use_dann:
        # DANN model will be created in trainer
        return backbone
    else:
        # Create standard multi-task model
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, n_channels, 1000)
            dummy_features = backbone(dummy_input)
            if isinstance(dummy_features, tuple):
                dummy_features = dummy_features[0]
            feature_dim = dummy_features.shape[-1]

        # Create multi-task head
        task_head = nn.Linear(feature_dim, len(config.target_factors))

        # Combine backbone and head
        class PsychModel(nn.Module):
            def __init__(self, backbone, head):
                super().__init__()
                self.backbone = backbone
                self.head = head

            def forward(self, x):
                features = self.backbone(x)
                if isinstance(features, tuple):
                    features = features[0]
                if features.dim() == 3:
                    features = features.mean(dim=1)
                return self.head(features)

        return PsychModel(backbone, task_head)
