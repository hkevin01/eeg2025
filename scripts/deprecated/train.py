#!/usr/bin/env python3
"""
Main training script for EEG Foundation Challenge 2025.
Supports pretraining, cross-task learning, and psychopathology prediction.

This script implements the scaffold approach for the competition with support
for different training modes and evaluation metrics.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import hydra
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataio.hbn_dataset import HBNDataset
from dataio.bids_loader import BidsWindowDataset
from models.backbones.temporal_cnn import TemporalCNN
from models.heads import ClassificationHead
from utils.compression import CompressionMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGLightningModule(LightningModule):
    """PyTorch Lightning module for EEG training."""

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))

        # Model architecture
        self.backbone = TemporalCNN(
            in_channels=config.model.in_channels,
            num_channels=config.model.num_channels,
            kernel_size=config.model.kernel_size,
            dropout=config.model.dropout
        )

        # Task-specific heads
        self.heads = nn.ModuleDict()
        if config.task.name in ["cross_task", "pretraining"]:
            # Multiple heads for different tasks
            self.heads["task_classification"] = ClassificationHead(
                in_features=config.model.num_channels[-1],
                num_classes=config.task.num_task_classes,
                dropout=config.model.dropout
            )
            self.heads["subject_classification"] = ClassificationHead(
                in_features=config.model.num_channels[-1],
                num_classes=config.task.num_subjects,
                dropout=config.model.dropout
            )

        if config.task.name in ["psychopathology", "cross_task"]:
            self.heads["psychopathology"] = ClassificationHead(
                in_features=config.model.num_channels[-1],
                num_classes=config.task.num_psych_classes,
                dropout=config.model.dropout
            )

        # Compression metrics
        self.compression_metrics = CompressionMetrics()

        # Loss weights
        self.loss_weights = config.training.get("loss_weights", {})

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through backbone and heads."""
        # Backbone feature extraction
        features = self.backbone(x)

        # Apply heads
        outputs = {}
        for head_name, head in self.heads.items():
            outputs[head_name] = head(features)

        return outputs

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step."""
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        return self._shared_step(batch, "val")

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        return self._shared_step(batch, "test")

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str) -> torch.Tensor:
        """Shared step for train/val/test."""
        x = batch["eeg"]
        batch_size = x.size(0)

        # Forward pass
        outputs = self(x)

        # Compute losses
        total_loss = 0.0
        losses = {}

        # Task classification loss
        if "task_classification" in outputs and "task_label" in batch:
            task_loss = F.cross_entropy(outputs["task_classification"], batch["task_label"])
            losses[f"{stage}_task_loss"] = task_loss
            total_loss += self.loss_weights.get("task", 1.0) * task_loss

        # Subject classification loss (for domain adaptation)
        if "subject_classification" in outputs and "subject_id" in batch:
            subject_loss = F.cross_entropy(outputs["subject_classification"], batch["subject_id"])
            losses[f"{stage}_subject_loss"] = subject_loss
            # Negative weight for adversarial training
            total_loss += self.loss_weights.get("subject", -0.1) * subject_loss

        # Psychopathology classification loss
        if "psychopathology" in outputs and "psych_label" in batch:
            psych_loss = F.cross_entropy(outputs["psychopathology"], batch["psych_label"])
            losses[f"{stage}_psych_loss"] = psych_loss
            total_loss += self.loss_weights.get("psych", 1.0) * psych_loss

        # Log losses
        for loss_name, loss_value in losses.items():
            self.log(loss_name, loss_value, prog_bar=True, batch_size=batch_size)

        self.log(f"{stage}_total_loss", total_loss, prog_bar=True, batch_size=batch_size)

        # Compute metrics
        if stage in ["val", "test"]:
            self._compute_metrics(outputs, batch, stage)

        return total_loss

    def _compute_metrics(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], stage: str):
        """Compute evaluation metrics."""
        batch_size = list(outputs.values())[0].size(0)

        # Task classification metrics
        if "task_classification" in outputs and "task_label" in batch:
            task_preds = torch.argmax(outputs["task_classification"], dim=1)
            task_acc = (task_preds == batch["task_label"]).float().mean()
            self.log(f"{stage}_task_acc", task_acc, batch_size=batch_size)

        # Psychopathology metrics
        if "psychopathology" in outputs and "psych_label" in batch:
            psych_preds = torch.argmax(outputs["psychopathology"], dim=1)
            psych_acc = (psych_preds == batch["psych_label"]).float().mean()
            self.log(f"{stage}_psych_acc", psych_acc, batch_size=batch_size)

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay
        )

        if self.config.training.get("use_scheduler", False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.max_epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )
            return [optimizer], [scheduler]

        return optimizer


def create_data_loaders(config: DictConfig) -> Dict[str, DataLoader]:
    """Create data loaders for training."""
    # Load datasets
    train_dataset = HBNDataset(
        bids_root=config.data.bids_root,
        split="train",
        window_length=config.data.window_length,
        overlap=config.data.overlap,
        sample_rate=config.data.sample_rate,
        task_type=config.task.name
    )

    val_dataset = HBNDataset(
        bids_root=config.data.bids_root,
        split="val",
        window_length=config.data.window_length,
        overlap=config.data.overlap,
        sample_rate=config.data.sample_rate,
        task_type=config.task.name
    )

    test_dataset = HBNDataset(
        bids_root=config.data.bids_root,
        split="test",
        window_length=config.data.window_length,
        overlap=config.data.overlap,
        sample_rate=config.data.sample_rate,
        task_type=config.task.name
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.training.num_workers,
        pin_memory=True
    )

    return {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }


def create_callbacks(config: DictConfig) -> list:
    """Create training callbacks."""
    callbacks = []

    # Model checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.paths.checkpoints,
        filename=f"{config.task.name}-{{epoch:02d}}-{{val_total_loss:.2f}}",
        monitor="val_total_loss",
        mode="min",
        save_top_k=3,
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    # Early stopping
    if config.training.get("early_stopping", False):
        early_stop_callback = EarlyStopping(
            monitor="val_total_loss",
            patience=config.training.early_stopping_patience,
            mode="min",
            verbose=True
        )
        callbacks.append(early_stop_callback)

    return callbacks


def create_loggers(config: DictConfig) -> list:
    """Create experiment loggers."""
    loggers = []

    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config.paths.logs,
        name=config.task.name
    )
    loggers.append(tb_logger)

    # Wandb logger (if enabled)
    if config.logging.get("use_wandb", False):
        wandb_logger = WandbLogger(
            project=config.logging.wandb_project,
            name=f"{config.task.name}_{config.run_id}",
            save_dir=config.paths.logs
        )
        loggers.append(wandb_logger)

    return loggers


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(config: DictConfig) -> None:
    """Main training function."""
    logger.info("Starting EEG training...")
    logger.info(f"Task: {config.task.name}")
    logger.info(f"Config:\n{OmegaConf.to_yaml(config)}")

    # Set random seeds
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)

    # Create data loaders
    logger.info("Creating data loaders...")
    data_loaders = create_data_loaders(config)
    logger.info(f"Train samples: {len(data_loaders['train'].dataset)}")
    logger.info(f"Val samples: {len(data_loaders['val'].dataset)}")
    logger.info(f"Test samples: {len(data_loaders['test'].dataset)}")

    # Create model
    logger.info("Creating model...")
    model = EEGLightningModule(config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create callbacks and loggers
    callbacks = create_callbacks(config)
    loggers = create_loggers(config)

    # Create trainer
    trainer = Trainer(
        max_epochs=config.training.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=config.training.precision,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=config.training.gradient_clip_val,
        accumulate_grad_batches=config.training.accumulate_grad_batches,
        check_val_every_n_epoch=config.training.check_val_every_n_epoch
    )

    # Training
    logger.info("Starting training...")
    trainer.fit(
        model,
        train_dataloaders=data_loaders["train"],
        val_dataloaders=data_loaders["val"]
    )

    # Testing
    if config.training.get("run_test", True):
        logger.info("Starting testing...")
        trainer.test(model, dataloaders=data_loaders["test"])

    logger.info("Training completed!")


if __name__ == "__main__":
    main()
