"""
Self-Supervised Learning Pretraining for EEG Foundation Challenge 2025.

This module implements SSL objectives including masked reconstruction, contrastive learning,
and view generation with various augmentation techniques for robust EEG representation learning.
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

from ..models.backbones.temporal_cnn import TemporalCNN
from ..models.losses.ssl_losses import (
    MaskedReconstructionLoss,
    ContrastiveLoss,
    PredictiveResidualLoss
)
from ..utils.schedulers import ParameterScheduler
from ..utils.augmentations import SSLViewPipeline

logger = logging.getLogger(__name__)


@dataclass
class SSLConfig:
    """Configuration for SSL pretraining."""

    # Model architecture
    backbone: str = "temporal_cnn"
    d_model: int = 128
    n_layers: int = 4
    emb_dim: int = 128

    # SSL objectives
    objectives: List[str] = None
    mask_ratio: float = 0.2
    temperature: float = 0.1

    # Augmentation views
    time_masking_ratio: float = 0.15
    channel_dropout: float = 0.1
    temporal_jitter_std: float = 0.02
    wavelet_distortion_pct: float = 1.0
    quant_snr_db: float = 25.0

    # Optimization
    lr: float = 3e-4
    weight_decay: float = 0.02
    batch_size: int = 64
    epochs: int = 50
    scheduler: str = "cosine"

    # Checkpointing
    ckpt_dir: str = "runs/pretrain"
    save_every: int = 5

    # Logging
    log_every: int = 10
    eval_every: int = 100

    def __post_init__(self):
        if self.objectives is None:
            self.objectives = ["masked", "contrastive", "predictive_residual"]


class MaskedTimeDecoder(nn.Module):
    """
    Masked time decoder head for reconstruction loss.

    Implements either linear or small Conv1D decoder for temporal reconstruction.
    """

    def __init__(
        self,
        d_model: int,
        n_channels: int,
        decoder_type: str = "linear",
        hidden_dim: Optional[int] = None
    ):
        super().__init__()

        self.decoder_type = decoder_type
        self.n_channels = n_channels

        if decoder_type == "linear":
            self.decoder = nn.Linear(d_model, n_channels)
        elif decoder_type == "conv1d":
            hidden_dim = hidden_dim or d_model // 2
            self.decoder = nn.Sequential(
                nn.Conv1d(d_model, hidden_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(hidden_dim, n_channels, kernel_size=1)
            )
        else:
            raise ValueError(f"Unknown decoder type: {decoder_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for masked reconstruction.

        Args:
            x: Encoded features [batch_size, seq_len, d_model]

        Returns:
            Reconstructed signal [batch_size, n_channels, seq_len]
        """
        if self.decoder_type == "linear":
            # x: [batch_size, seq_len, d_model] -> [batch_size, seq_len, n_channels]
            reconstructed = self.decoder(x)
            # Transpose to [batch_size, n_channels, seq_len]
            reconstructed = reconstructed.transpose(1, 2)
        else:  # conv1d
            # x: [batch_size, seq_len, d_model] -> [batch_size, d_model, seq_len]
            x = x.transpose(1, 2)
            reconstructed = self.decoder(x)

        return reconstructed


class SSLModel(nn.Module):
    """
    SSL model combining backbone with multiple SSL heads.
    """

    def __init__(
        self,
        backbone: nn.Module,
        config: SSLConfig,
        n_channels: int = 64
    ):
        super().__init__()

        self.backbone = backbone
        self.config = config
        self.n_channels = n_channels

        # SSL heads
        if "masked" in config.objectives:
            self.masked_decoder = MaskedTimeDecoder(
                d_model=config.d_model,
                n_channels=n_channels,
                decoder_type="linear"  # Start with linear, can be configured
            )

        if "contrastive" in config.objectives:
            self.projection_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model),
                nn.ReLU(),
                nn.Linear(config.d_model, config.emb_dim)
            )

        if "predictive_residual" in config.objectives:
            self.predictor_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model // 2),
                nn.ReLU(),
                nn.Linear(config.d_model // 2, config.d_model)
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple SSL objectives.

        Args:
            x: Input EEG signal [batch_size, n_channels, seq_len]
            mask: Optional mask for masked reconstruction [batch_size, seq_len]

        Returns:
            Dictionary of outputs for different SSL objectives
        """
        # Get backbone features
        features = self.backbone(x)  # [batch_size, seq_len, d_model]

        outputs = {"features": features}

        # Masked reconstruction
        if "masked" in self.config.objectives and hasattr(self, 'masked_decoder'):
            reconstructed = self.masked_decoder(features)
            outputs["reconstructed"] = reconstructed

        # Contrastive learning
        if "contrastive" in self.config.objectives and hasattr(self, 'projection_head'):
            # Global pooling for contrastive learning
            pooled_features = features.mean(dim=1)  # [batch_size, d_model]
            projections = self.projection_head(pooled_features)
            projections = F.normalize(projections, dim=-1)
            outputs["projections"] = projections

        # Predictive residual
        if "predictive_residual" in self.config.objectives and hasattr(self, 'predictor_head'):
            predicted = self.predictor_head(features)
            outputs["predicted"] = predicted

        return outputs


class SSLPretrainer:
    """
    SSL Pretrainer for EEG Foundation Challenge 2025.

    Implements multiple SSL objectives with robust training loop and schedulable parameters.
    """

    def __init__(
        self,
        config: SSLConfig,
        model: SSLModel,
        view_pipeline: SSLViewPipeline,
        device: torch.device
    ):
        self.config = config
        self.model = model.to(device)
        self.view_pipeline = view_pipeline
        self.device = device

        # Initialize losses
        self.losses = self._init_losses()

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )

        self.scheduler = self._init_scheduler()

        # Initialize parameter schedulers for temperature, mask ratio, etc.
        self.param_schedulers = self._init_param_schedulers()

        # Logging setup
        self.setup_logging()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')

        logger.info(f"SSL Pretrainer initialized with objectives: {config.objectives}")

    def _init_losses(self) -> Dict[str, nn.Module]:
        """Initialize SSL loss functions."""
        losses = {}

        if "masked" in self.config.objectives:
            losses["masked"] = MaskedReconstructionLoss()

        if "contrastive" in self.config.objectives:
            losses["contrastive"] = ContrastiveLoss(temperature=self.config.temperature)

        if "predictive_residual" in self.config.objectives:
            losses["predictive_residual"] = PredictiveResidualLoss()

        return losses

    def _init_scheduler(self):
        """Initialize learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.epochs
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.epochs // 3,
                gamma=0.1
            )
        else:
            return None

    def _init_param_schedulers(self) -> Dict[str, ParameterScheduler]:
        """Initialize parameter schedulers for SSL hyperparameters."""
        schedulers = {}

        # Temperature scheduler (warm up then cool down)
        schedulers["temperature"] = ParameterScheduler(
            param_name="temperature",
            schedule_type="cosine_warmup",
            start_value=0.05,
            end_value=self.config.temperature,
            warmup_epochs=5,
            total_epochs=self.config.epochs
        )

        # Mask ratio scheduler (start low, increase)
        schedulers["mask_ratio"] = ParameterScheduler(
            param_name="mask_ratio",
            schedule_type="linear",
            start_value=0.1,
            end_value=self.config.mask_ratio,
            total_epochs=self.config.epochs
        )

        # Distortion percentage scheduler
        schedulers["distortion_pct"] = ParameterScheduler(
            param_name="distortion_pct",
            schedule_type="cosine",
            start_value=0.5,
            end_value=self.config.wavelet_distortion_pct,
            total_epochs=self.config.epochs
        )

        return schedulers

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
                    project="eeg2025_ssl_pretrain",
                    config=self.config.__dict__,
                    dir=str(self.ckpt_dir)
                )
            self.use_wandb = True
        except Exception as e:
            logger.warning(f"W&B initialization failed: {e}")

    def generate_time_mask(self, seq_len: int, batch_size: int) -> torch.Tensor:
        """
        Generate random time masks for masked reconstruction.

        Args:
            seq_len: Sequence length
            batch_size: Batch size

        Returns:
            Boolean mask [batch_size, seq_len] where True = masked
        """
        current_mask_ratio = self.param_schedulers["mask_ratio"].get_value(self.current_epoch)

        masks = []
        for _ in range(batch_size):
            # Random contiguous masking
            mask_len = int(seq_len * current_mask_ratio)
            start_idx = np.random.randint(0, seq_len - mask_len + 1)

            mask = torch.zeros(seq_len, dtype=torch.bool)
            mask[start_idx:start_idx + mask_len] = True
            masks.append(mask)

        return torch.stack(masks).to(self.device)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single training step with multiple SSL objectives.

        Args:
            batch: Batch of EEG data

        Returns:
            Dictionary of losses
        """
        self.model.train()

        # Get original data
        x = batch["eeg"].to(self.device)  # [batch_size, n_channels, seq_len]
        batch_size, n_channels, seq_len = x.shape

        # Update schedulable parameters
        current_temp = self.param_schedulers["temperature"].get_value(self.current_epoch)
        current_distortion = self.param_schedulers["distortion_pct"].get_value(self.current_epoch)

        # Update loss temperature if contrastive
        if "contrastive" in self.losses:
            self.losses["contrastive"].temperature = current_temp

        # Generate augmented views
        view1 = self.view_pipeline(x, distortion_pct=current_distortion)
        view2 = self.view_pipeline(x, distortion_pct=current_distortion)

        # Generate masks for masked reconstruction
        mask = self.generate_time_mask(seq_len, batch_size)

        # Forward pass
        outputs = self.model(view1, mask=mask)
        outputs2 = self.model(view2)

        # Compute losses
        losses = {}
        total_loss = 0

        # Masked reconstruction loss
        if "masked" in self.config.objectives and "reconstructed" in outputs:
            masked_loss = self.losses["masked"](
                outputs["reconstructed"],
                x,
                mask
            )
            losses["masked_loss"] = masked_loss.item()
            total_loss += masked_loss

        # Contrastive loss
        if "contrastive" in self.config.objectives and "projections" in outputs:
            contrastive_loss = self.losses["contrastive"](
                outputs["projections"],
                outputs2["projections"]
            )
            losses["contrastive_loss"] = contrastive_loss.item()
            total_loss += contrastive_loss

        # Predictive residual loss
        if "predictive_residual" in self.config.objectives and "predicted" in outputs:
            residual_loss = self.losses["predictive_residual"](
                outputs["predicted"],
                outputs["features"]
            )
            losses["predictive_residual_loss"] = residual_loss.item()
            total_loss += residual_loss

        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        losses["total_loss"] = total_loss.item()
        losses["learning_rate"] = self.optimizer.param_groups[0]["lr"]
        losses["current_temperature"] = current_temp
        losses["current_mask_ratio"] = self.param_schedulers["mask_ratio"].get_value(self.current_epoch)
        losses["current_distortion_pct"] = current_distortion

        return losses

    def validation_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Single validation step.

        Args:
            batch: Batch of validation data

        Returns:
            Dictionary of validation losses
        """
        self.model.eval()

        with torch.no_grad():
            x = batch["eeg"].to(self.device)
            batch_size, n_channels, seq_len = x.shape

            # Generate views and masks
            view1 = self.view_pipeline(x, distortion_pct=0.5)  # Fixed for validation
            view2 = self.view_pipeline(x, distortion_pct=0.5)
            mask = self.generate_time_mask(seq_len, batch_size)

            # Forward pass
            outputs = self.model(view1, mask=mask)
            outputs2 = self.model(view2)

            # Compute validation losses
            val_losses = {}
            total_loss = 0

            if "masked" in self.config.objectives and "reconstructed" in outputs:
                masked_loss = self.losses["masked"](outputs["reconstructed"], x, mask)
                val_losses["val_masked_loss"] = masked_loss.item()
                total_loss += masked_loss

            if "contrastive" in self.config.objectives and "projections" in outputs:
                contrastive_loss = self.losses["contrastive"](
                    outputs["projections"], outputs2["projections"]
                )
                val_losses["val_contrastive_loss"] = contrastive_loss.item()
                total_loss += contrastive_loss

            if "predictive_residual" in self.config.objectives and "predicted" in outputs:
                residual_loss = self.losses["predictive_residual"](
                    outputs["predicted"], outputs["features"]
                )
                val_losses["val_predictive_residual_loss"] = residual_loss.item()
                total_loss += residual_loss

            val_losses["val_total_loss"] = total_loss.item()

            return val_losses

    def train_epoch(self, train_loader, val_loader=None) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader

        Returns:
            Dictionary of epoch metrics
        """
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
                val_losses = self.validate(val_loader)
                self.log_validation(val_losses)

        # Aggregate epoch losses
        epoch_metrics = {}
        if epoch_losses:
            for key in epoch_losses[0].keys():
                values = [loss[key] for loss in epoch_losses]
                epoch_metrics[f"epoch_{key}"] = np.mean(values)

        return epoch_metrics

    def validate(self, val_loader) -> Dict[str, float]:
        """
        Run validation on entire validation set.

        Args:
            val_loader: Validation data loader

        Returns:
            Aggregated validation metrics
        """
        val_losses = []

        for batch in val_loader:
            batch_losses = self.validation_step(batch)
            val_losses.append(batch_losses)

        # Aggregate validation losses
        val_metrics = {}
        if val_losses:
            for key in val_losses[0].keys():
                values = [loss[key] for loss in val_losses]
                val_metrics[key] = np.mean(values)

        return val_metrics

    def log_step(self, losses: Dict[str, float], batch_idx: int, total_batches: int):
        """Log training step metrics."""
        # Console logging
        log_str = f"Epoch {self.current_epoch} [{batch_idx}/{total_batches}] "
        log_str += f"Loss: {losses['total_loss']:.4f}"
        if "masked_loss" in losses:
            log_str += f" | Masked: {losses['masked_loss']:.4f}"
        if "contrastive_loss" in losses:
            log_str += f" | Contrastive: {losses['contrastive_loss']:.4f}"
        if "predictive_residual_loss" in losses:
            log_str += f" | Predictive: {losses['predictive_residual_loss']:.4f}"

        logger.info(log_str)

        # TensorBoard logging
        for key, value in losses.items():
            self.tb_writer.add_scalar(f"train/{key}", value, self.global_step)

        # W&B logging
        if self.use_wandb:
            wandb.log({f"train/{k}": v for k, v in losses.items()}, step=self.global_step)

    def log_validation(self, val_losses: Dict[str, float]):
        """Log validation metrics."""
        log_str = f"Validation - Total: {val_losses['val_total_loss']:.4f}"
        for key, value in val_losses.items():
            if key != "val_total_loss":
                log_str += f" | {key}: {value:.4f}"

        logger.info(log_str)

        # TensorBoard logging
        for key, value in val_losses.items():
            self.tb_writer.add_scalar(f"val/{key}", value, self.global_step)

        # W&B logging
        if self.use_wandb:
            wandb.log({f"val/{k}": v for k, v in val_losses.items()}, step=self.global_step)

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "config": self.config,
            "global_step": self.global_step,
            "best_loss": self.best_loss,
        }

        # Save regular checkpoint
        ckpt_path = self.ckpt_dir / f"checkpoint_epoch_{epoch}.ckpt"
        torch.save(checkpoint, ckpt_path)

        # Save best checkpoint
        if is_best:
            best_path = self.ckpt_dir / "best.ckpt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint at epoch {epoch}")

        logger.info(f"Checkpoint saved: {ckpt_path}")

    def train(self, train_loader, val_loader=None) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader

        Returns:
            Training history
        """
        logger.info("Starting SSL pretraining...")
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            start_time = time.time()

            # Update parameter schedulers
            for scheduler in self.param_schedulers.values():
                scheduler.step(epoch)

            # Training epoch
            epoch_metrics = self.train_epoch(train_loader, val_loader)

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Validation
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                current_val_loss = val_metrics.get("val_total_loss", float('inf'))
                history["val_loss"].append(current_val_loss)

                # Check for best model
                is_best = current_val_loss < self.best_loss
                if is_best:
                    self.best_loss = current_val_loss
            else:
                is_best = False
                current_val_loss = None

            # Record training loss
            train_loss = epoch_metrics.get("epoch_total_loss", 0.0)
            history["train_loss"].append(train_loss)

            # Epoch logging
            epoch_time = time.time() - start_time
            log_str = f"Epoch {epoch}/{self.config.epochs} completed in {epoch_time:.2f}s"
            log_str += f" | Train Loss: {train_loss:.4f}"
            if current_val_loss is not None:
                log_str += f" | Val Loss: {current_val_loss:.4f}"

            logger.info(log_str)

            # Save checkpoint
            if epoch % self.config.save_every == 0 or is_best:
                self.save_checkpoint(epoch, is_best)

        # Final checkpoint
        self.save_checkpoint(self.config.epochs - 1, False)

        logger.info("SSL pretraining completed!")
        return history


def create_ssl_model(config: SSLConfig, n_channels: int = 64) -> SSLModel:
    """
    Factory function to create SSL model.

    Args:
        config: SSL configuration
        n_channels: Number of EEG channels

    Returns:
        SSL model instance
    """
    # Create backbone
    if config.backbone == "temporal_cnn":
        backbone = TemporalCNN(
            n_channels=n_channels,
            n_filters=config.d_model,
            kernel_size=25,
            n_blocks=config.n_layers
        )
    else:
        raise ValueError(f"Unknown backbone: {config.backbone}")

    # Create SSL model
    model = SSLModel(
        backbone=backbone,
        config=config,
        n_channels=n_channels
    )

    return model
