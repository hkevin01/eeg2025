"""
Cross-task transfer training for EEG Foundation Challenge 2025.

This module implements transfer learning from self-supervised pretraining (SuS)
to downstream tasks (CCD) with official metrics and alignment techniques.
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
from scipy.stats import pearsonr

from ..models.backbones.temporal_cnn import TemporalCNN
from ..models.losses.corr_mse import CorrMSELoss
from ..models.heads import CCDRegressionHead, CCDClassificationHead
from ..utils.augmentations import SSLViewPipeline

logger = logging.getLogger(__name__)


@dataclass
class CrossTaskConfig:
    """Configuration for cross-task transfer training."""

    # Model architecture
    backbone: str = "temporal_cnn"
    freeze_ratio: float = 0.75
    ssl_checkpoint: str = "runs/pretrain/best.ckpt"

    # Task heads
    regression_head: str = "ccd_reg_head"
    classification_head: str = "ccd_clf_head"

    # Loss configuration
    reg_loss: str = "corr_mse"
    clf_loss: str = "focal"
    loss_weights: Dict[str, float] = None

    # Alignment techniques
    use_mmd_alignment: bool = False
    mmd_weight: float = 0.1
    use_film_adapter: bool = False
    task_token_dim: int = 32

    # Optimization
    lr: float = 2e-4
    weight_decay: float = 0.02
    batch_size: int = 64
    epochs: int = 30

    # Early stopping
    early_stopping_patience: int = 10
    primary_metric: str = "combined_score"  # Combination of RT correlation and success AUROC

    # Augmentation
    use_compression_views: bool = True
    wavelet_distortion_pct: float = 0.5
    quant_snr_db: float = 30.0

    # Checkpointing
    ckpt_dir: str = "runs/cross_task"
    save_every: int = 5

    # Logging
    log_every: int = 10
    eval_every: int = 50

    def __post_init__(self):
        if self.loss_weights is None:
            self.loss_weights = {"regression": 1.0, "classification": 1.0}


class FiLMAdapter(nn.Module):
    """
    Feature-wise Linear Modulation (FiLM) adapter for task conditioning.

    Applies task-specific feature modulation based on task tokens.
    """

    def __init__(self, feature_dim: int, task_token_dim: int, n_tasks: int = 2):
        super().__init__()

        self.feature_dim = feature_dim
        self.task_token_dim = task_token_dim
        self.n_tasks = n_tasks

        # Task token embeddings
        self.task_tokens = nn.Embedding(n_tasks, task_token_dim)

        # FiLM parameter generators
        self.gamma_generator = nn.Linear(task_token_dim, feature_dim)
        self.beta_generator = nn.Linear(task_token_dim, feature_dim)

        # Initialize to identity transformation
        self.gamma_generator.weight.data.zero_()
        self.gamma_generator.bias.data.fill_(1.0)
        self.beta_generator.weight.data.zero_()
        self.beta_generator.bias.data.zero_()

    def forward(self, features: torch.Tensor, task_id: torch.Tensor) -> torch.Tensor:
        """
        Apply FiLM conditioning to features.

        Args:
            features: Input features [batch_size, seq_len, feature_dim] or [batch_size, feature_dim]
            task_id: Task identifier [batch_size] (0=SuS, 1=CCD)

        Returns:
            Modulated features with same shape as input
        """
        # Get task tokens
        task_emb = self.task_tokens(task_id)  # [batch_size, task_token_dim]

        # Generate FiLM parameters
        gamma = self.gamma_generator(task_emb)  # [batch_size, feature_dim]
        beta = self.beta_generator(task_emb)    # [batch_size, feature_dim]

        # Apply FiLM conditioning
        if features.dim() == 3:  # [batch_size, seq_len, feature_dim]
            gamma = gamma.unsqueeze(1)  # [batch_size, 1, feature_dim]
            beta = beta.unsqueeze(1)    # [batch_size, 1, feature_dim]

        modulated_features = gamma * features + beta

        return modulated_features


class MMDAlignment(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) for domain alignment.

    Aligns SuS and CCD embeddings to reduce domain gap.
    """

    def __init__(self, kernel_type: str = "rbf", bandwidth: float = 1.0):
        super().__init__()
        self.kernel_type = kernel_type
        self.bandwidth = bandwidth

    def rbf_kernel(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix."""
        # x1: [n1, d], x2: [n2, d]
        # Returns: [n1, n2]
        pairwise_dist = torch.cdist(x1, x2, p=2) ** 2
        return torch.exp(-pairwise_dist / (2 * self.bandwidth ** 2))

    def forward(self, sus_embeddings: torch.Tensor, ccd_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD loss between SuS and CCD embeddings.

        Args:
            sus_embeddings: SuS domain embeddings [batch_size_sus, emb_dim]
            ccd_embeddings: CCD domain embeddings [batch_size_ccd, emb_dim]

        Returns:
            MMD loss scalar
        """
        if self.kernel_type == "rbf":
            # Compute kernel matrices
            k_sus_sus = self.rbf_kernel(sus_embeddings, sus_embeddings)
            k_ccd_ccd = self.rbf_kernel(ccd_embeddings, ccd_embeddings)
            k_sus_ccd = self.rbf_kernel(sus_embeddings, ccd_embeddings)

            # MMD^2 = E[k(x,x)] + E[k(y,y)] - 2*E[k(x,y)]
            mmd_loss = (
                k_sus_sus.mean() +
                k_ccd_ccd.mean() -
                2 * k_sus_ccd.mean()
            )

            return mmd_loss
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")


class CrossTaskModel(nn.Module):
    """
    Cross-task model for SuS → CCD transfer.

    Combines SSL pretrained backbone with task-specific heads and optional adapters.
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: CrossTaskConfig,
        n_channels: int = 64
    ):
        super().__init__()

        self.backbone = backbone
        self.config = config
        self.n_channels = n_channels

        # Get backbone feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, n_channels, 1000)
            dummy_features = backbone(dummy_input)
            if isinstance(dummy_features, tuple):
                dummy_features = dummy_features[0]
            self.feature_dim = dummy_features.shape[-1]

        # Freeze backbone parameters based on freeze_ratio
        self._freeze_backbone_parameters()

        # Task-specific heads
        self.regression_head = CCDRegressionHead(self.feature_dim)
        self.classification_head = CCDClassificationHead(self.feature_dim)

        # Optional adapters
        if config.use_film_adapter:
            self.film_adapter = FiLMAdapter(
                feature_dim=self.feature_dim,
                task_token_dim=config.task_token_dim,
                n_tasks=2  # SuS=0, CCD=1
            )

        if config.use_mmd_alignment:
            self.mmd_alignment = MMDAlignment()

    def _freeze_backbone_parameters(self):
        """Freeze a portion of backbone parameters."""
        all_params = list(self.backbone.parameters())
        n_params_to_freeze = int(len(all_params) * self.config.freeze_ratio)

        # Freeze early layers (first n_params_to_freeze parameters)
        for i, param in enumerate(all_params):
            if i < n_params_to_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True

        logger.info(f"Frozen {n_params_to_freeze}/{len(all_params)} backbone parameters")

    def forward(
        self,
        x: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with task-specific processing.

        Args:
            x: Input EEG signal [batch_size, n_channels, seq_len]
            task_id: Task identifier [batch_size] (0=SuS, 1=CCD)
            return_embeddings: Whether to return intermediate embeddings

        Returns:
            Dictionary of outputs
        """
        # Extract backbone features
        features = self.backbone(x)  # [batch_size, seq_len, feature_dim]

        # Global pooling for task heads
        pooled_features = features.mean(dim=1)  # [batch_size, feature_dim]

        outputs = {}

        # Apply FiLM adapter if enabled
        if hasattr(self, 'film_adapter') and task_id is not None:
            pooled_features = self.film_adapter(pooled_features, task_id)

        # Task-specific predictions
        rt_pred = self.regression_head(pooled_features)
        success_pred = self.classification_head(pooled_features)

        outputs["rt_prediction"] = rt_pred
        outputs["success_prediction"] = success_pred

        if return_embeddings:
            outputs["features"] = features
            outputs["pooled_features"] = pooled_features

        return outputs


class OfficialMetrics:
    """
    Official metrics for EEG Challenge 2025 cross-task evaluation.

    Implements Pearson correlation and RMSE for RT, AUROC/AUPRC/balanced accuracy for success.
    """

    @staticmethod
    def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Pearson correlation coefficient."""
        # Remove NaN values
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
    def auroc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Area Under ROC Curve."""
        try:
            return roc_auc_score(y_true, y_pred)
        except ValueError:
            return 0.5  # Random performance for edge cases

    @staticmethod
    def auprc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute Area Under Precision-Recall Curve."""
        try:
            return average_precision_score(y_true, y_pred)
        except ValueError:
            return np.mean(y_true)  # Random performance baseline

    @staticmethod
    def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute balanced accuracy."""
        try:
            y_pred_binary = (y_pred > 0.5).astype(int)
            return balanced_accuracy_score(y_true, y_pred_binary)
        except ValueError:
            return 0.5

    @classmethod
    def compute_all_metrics(
        cls,
        rt_true: np.ndarray,
        rt_pred: np.ndarray,
        success_true: np.ndarray,
        success_pred: np.ndarray
    ) -> Dict[str, float]:
        """Compute all official metrics."""
        metrics = {}

        # RT metrics
        metrics["rt_pearson"] = cls.pearson_correlation(rt_true, rt_pred)
        metrics["rt_rmse"] = cls.rmse(rt_true, rt_pred)

        # Success metrics
        metrics["success_auroc"] = cls.auroc(success_true, success_pred)
        metrics["success_auprc"] = cls.auprc(success_true, success_pred)
        metrics["success_balanced_acc"] = cls.balanced_accuracy(success_true, success_pred)

        # Combined metric (normalized)
        # Normalize RT correlation to [0, 1] and combine with AUROC
        normalized_rt_corr = (metrics["rt_pearson"] + 1) / 2  # [-1, 1] → [0, 1]
        metrics["combined_score"] = (normalized_rt_corr + metrics["success_auroc"]) / 2

        return metrics


class CrossTaskTrainer:
    """
    Cross-task trainer for SuS → CCD transfer learning.

    Implements transfer learning with official metrics and early stopping.
    """

    def __init__(
        self,
        config: CrossTaskConfig,
        model: CrossTaskModel,
        view_pipeline: Optional[SSLViewPipeline],
        device: torch.device
    ):
        self.config = config
        self.model = model.to(device)
        self.view_pipeline = view_pipeline
        self.device = device

        # Initialize losses
        self.losses = self._init_losses()

        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.epochs
        )

        # Logging setup
        self.setup_logging()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = -float('inf')
        self.patience_counter = 0

        # Metrics tracker
        self.metrics = OfficialMetrics()

        logger.info("Cross-task trainer initialized")

    def _init_losses(self) -> Dict[str, nn.Module]:
        """Initialize loss functions."""
        losses = {}

        # Regression loss
        if self.config.reg_loss == "corr_mse":
            losses["regression"] = CorrMSELoss()
        elif self.config.reg_loss == "mse":
            losses["regression"] = nn.MSELoss()
        elif self.config.reg_loss == "l1":
            losses["regression"] = nn.L1Loss()

        # Classification loss
        if self.config.clf_loss == "focal":
            losses["classification"] = FocalLoss(alpha=0.25, gamma=2.0)
        elif self.config.clf_loss == "bce":
            losses["classification"] = nn.BCEWithLogitsLoss()
        elif self.config.clf_loss == "cross_entropy":
            losses["classification"] = nn.CrossEntropyLoss()

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
                    project="eeg2025_cross_task",
                    config=self.config.__dict__,
                    dir=str(self.ckpt_dir)
                )
            self.use_wandb = True
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}")

    def load_ssl_checkpoint(self, checkpoint_path: str):
        """Load SSL pretrained weights."""
        if not Path(checkpoint_path).exists():
            logger.warning(f"SSL checkpoint not found: {checkpoint_path}")
            return

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            # Load backbone weights
            backbone_state = {}
            for key, value in checkpoint["model_state_dict"].items():
                if key.startswith("backbone."):
                    backbone_state[key[9:]] = value  # Remove "backbone." prefix

            self.model.backbone.load_state_dict(backbone_state, strict=False)
            logger.info(f"Loaded SSL checkpoint from {checkpoint_path}")

        except Exception as e:
            logger.error(f"Failed to load SSL checkpoint: {e}")

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step.

        Args:
            batch: Batch of CCD data with labels

        Returns:
            Dictionary of losses
        """
        self.model.train()

        # Get batch data
        x = batch["eeg"].to(self.device)  # [batch_size, n_channels, seq_len]
        rt_true = batch["rt"].to(self.device)  # [batch_size]
        success_true = batch["success"].to(self.device)  # [batch_size]

        batch_size = x.shape[0]

        # Create task IDs (all CCD for this training)
        task_id = torch.ones(batch_size, dtype=torch.long, device=self.device)

        # Apply augmentation if enabled
        if self.config.use_compression_views and self.view_pipeline is not None:
            x = self.view_pipeline(
                x,
                distortion_pct=self.config.wavelet_distortion_pct
            )

        # Forward pass
        outputs = self.model(x, task_id=task_id, return_embeddings=True)

        # Compute losses
        losses = {}
        total_loss = 0

        # Regression loss (RT prediction)
        rt_pred = outputs["rt_prediction"].squeeze()
        reg_loss = self.losses["regression"](rt_pred, rt_true)
        losses["regression_loss"] = reg_loss.item()
        total_loss += self.config.loss_weights["regression"] * reg_loss

        # Classification loss (success prediction)
        success_pred = outputs["success_prediction"].squeeze()
        clf_loss = self.losses["classification"](success_pred, success_true.float())
        losses["classification_loss"] = clf_loss.item()
        total_loss += self.config.loss_weights["classification"] * clf_loss

        # MMD alignment loss (if enabled and SuS data available)
        if hasattr(self.model, 'mmd_alignment') and "sus_embeddings" in batch:
            sus_embeddings = batch["sus_embeddings"].to(self.device)
            ccd_embeddings = outputs["pooled_features"]

            mmd_loss = self.model.mmd_alignment(sus_embeddings, ccd_embeddings)
            losses["mmd_loss"] = mmd_loss.item()
            total_loss += self.config.mmd_weight * mmd_loss

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
        """
        Single validation step.

        Args:
            batch: Batch of validation data

        Returns:
            Dictionary of predictions and targets
        """
        self.model.eval()

        with torch.no_grad():
            x = batch["eeg"].to(self.device)
            rt_true = batch["rt"].to(self.device)
            success_true = batch["success"].to(self.device)

            batch_size = x.shape[0]
            task_id = torch.ones(batch_size, dtype=torch.long, device=self.device)

            # Forward pass
            outputs = self.model(x, task_id=task_id)

            # Get predictions
            rt_pred = outputs["rt_prediction"].squeeze()
            success_pred = torch.sigmoid(outputs["success_prediction"].squeeze())

            # Compute validation losses
            reg_loss = self.losses["regression"](rt_pred, rt_true)
            clf_loss = self.losses["classification"](
                outputs["success_prediction"].squeeze(),
                success_true.float()
            )

            val_results = {
                "rt_true": rt_true.cpu().numpy(),
                "rt_pred": rt_pred.cpu().numpy(),
                "success_true": success_true.cpu().numpy(),
                "success_pred": success_pred.cpu().numpy(),
                "val_regression_loss": reg_loss.item(),
                "val_classification_loss": clf_loss.item(),
                "val_total_loss": (reg_loss + clf_loss).item()
            }

            return val_results

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
                values = [loss[key] for loss in epoch_losses]
                epoch_metrics[f"epoch_{key}"] = np.mean(values)

        return epoch_metrics

    def validate(self, val_loader) -> Dict[str, Any]:
        """Run validation on entire validation set."""
        val_results = []

        for batch in val_loader:
            batch_results = self.validation_step(batch)
            val_results.append(batch_results)

        # Aggregate predictions and targets
        all_rt_true = np.concatenate([r["rt_true"] for r in val_results])
        all_rt_pred = np.concatenate([r["rt_pred"] for r in val_results])
        all_success_true = np.concatenate([r["success_true"] for r in val_results])
        all_success_pred = np.concatenate([r["success_pred"] for r in val_results])

        # Compute official metrics
        metrics = self.metrics.compute_all_metrics(
            all_rt_true, all_rt_pred,
            all_success_true, all_success_pred
        )

        # Add validation losses
        for loss_key in ["val_regression_loss", "val_classification_loss", "val_total_loss"]:
            values = [r[loss_key] for r in val_results]
            metrics[loss_key] = np.mean(values)

        return metrics

    def log_step(self, losses: Dict[str, float], batch_idx: int, total_batches: int):
        """Log training step metrics."""
        log_str = f"Epoch {self.current_epoch} [{batch_idx}/{total_batches}] "
        log_str += f"Total: {losses['total_loss']:.4f} | "
        log_str += f"Reg: {losses['regression_loss']:.4f} | "
        log_str += f"Clf: {losses['classification_loss']:.4f}"

        if "mmd_loss" in losses:
            log_str += f" | MMD: {losses['mmd_loss']:.4f}"

        logger.info(log_str)

        # TensorBoard logging
        for key, value in losses.items():
            self.tb_writer.add_scalar(f"train/{key}", value, self.global_step)

        # W&B logging
        if self.use_wandb:
            wandb.log({f"train/{k}": v for k, v in losses.items()}, step=self.global_step)

    def log_validation(self, metrics: Dict[str, float]):
        """Log validation metrics."""
        log_str = f"Validation - Combined: {metrics['combined_score']:.4f} | "
        log_str += f"RT Corr: {metrics['rt_pearson']:.4f} | "
        log_str += f"RT RMSE: {metrics['rt_rmse']:.4f} | "
        log_str += f"Success AUROC: {metrics['success_auroc']:.4f} | "
        log_str += f"Success AUPRC: {metrics['success_auprc']:.4f} | "
        log_str += f"Success Bal.Acc: {metrics['success_balanced_acc']:.4f}"

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
        """
        Main training loop with early stopping.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader

        Returns:
            Training history
        """
        logger.info("Starting cross-task transfer training...")

        # Load SSL checkpoint
        if self.config.ssl_checkpoint:
            self.load_ssl_checkpoint(self.config.ssl_checkpoint)

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
                current_metric = val_metrics[self.config.primary_metric]
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
                log_str += f" | {self.config.primary_metric}: {val_metrics[self.config.primary_metric]:.4f}"

            logger.info(log_str)

            # Save checkpoint
            if epoch % self.config.save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_metrics, is_best)

        # Final checkpoint
        if val_loader is not None:
            final_metrics = self.validate(val_loader)
            self.save_checkpoint(self.config.epochs - 1, final_metrics, False)

        logger.info("Cross-task transfer training completed!")
        return history


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()


def create_cross_task_model(config: CrossTaskConfig, n_channels: int = 64) -> CrossTaskModel:
    """
    Factory function to create cross-task model.

    Args:
        config: Cross-task configuration
        n_channels: Number of EEG channels

    Returns:
        CrossTaskModel instance
    """
    # Create backbone (should match SSL pretraining)
    if config.backbone == "temporal_cnn":
        backbone = TemporalCNN(
            n_channels=n_channels,
            n_filters=128,  # Should match SSL config
            kernel_size=25,
            n_blocks=4
        )
    else:
        raise ValueError(f"Unknown backbone: {config.backbone}")

    # Create cross-task model
    model = CrossTaskModel(
        backbone=backbone,
        config=config,
        n_channels=n_channels
    )

    return model
