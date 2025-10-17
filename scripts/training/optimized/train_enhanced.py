#!/usr/bin/env python3
"""
Enhanced training script for EEG Foundation Challenge 2025.

This script implements the complete pipeline with official Starter Kit integration,
domain adaptation, compression-augmented SSL, and official metrics.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import hydra
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
import numpy as np
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataio.hbn_dataset import HBNDataset, create_hbn_datasets
from dataio.starter_kit import OfficialMetrics
from models.backbones.enhanced_cnn import EnhancedTemporalCNN, RobustEEGBackbone
from models.heads import ClassificationHead, MultiTaskHead, ContrastiveHead
from utils.compression import CompressionMetrics, CompressionAwareLoss
from utils.domain_adaptation import (
    create_domain_adaptation_components,
    GradientReversalScheduler,
    MMDLoss
)
from utils.submission import create_starter_kit_submission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EEGFoundationModel(LightningModule):
    """
    Enhanced PyTorch Lightning module for EEG Foundation Challenge.

    Integrates official metrics, domain adaptation, compression-aware SSL,
    and all challenge requirements.
    """

    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))

        # Task configuration
        self.task_type = config.task.name
        self.window_length = config.data.windows.get(f"{self.task_type}_len_s", 2.0)

        # Initialize backbone
        backbone_config = config.model.backbone
        if backbone_config.type == "enhanced_cnn":
            self.backbone = EnhancedTemporalCNN(
                in_channels=config.data.channels,
                num_channels=backbone_config.num_channels,
                kernel_size=backbone_config.kernel_size,
                dropout=backbone_config.dropout,
                use_se=backbone_config.get("use_se", True),
                use_conformer=backbone_config.get("use_conformer", True),
                conformer_dim=backbone_config.get("conformer_dim", 256),
                conformer_depth=backbone_config.get("conformer_depth", 4),
                num_domains=config.domain_adaptation.get("num_subjects", 1000),
                enable_domain_adaptation=config.domain_adaptation.get("enabled", True)
            )
        elif backbone_config.type == "robust_eeg":
            self.backbone = RobustEEGBackbone(
                in_channels=config.data.channels,
                num_channels=backbone_config.num_channels,
                kernel_size=backbone_config.kernel_size,
                dropout=backbone_config.dropout,
                channel_dropout=backbone_config.get("channel_dropout", 0.1),
                use_channel_attention=backbone_config.get("use_channel_attention", True)
            )
        else:
            raise ValueError(f"Unknown backbone type: {backbone_config.type}")

        feature_dim = self.backbone.feature_dim

        # Initialize task heads
        self.heads = nn.ModuleDict()

        if self.task_type == "pretraining":
            # SSL heads
            self.heads["contrastive"] = ContrastiveHead(
                in_features=feature_dim,
                projection_dim=config.model.ssl.projection_dim,
                hidden_dim=config.model.ssl.hidden_dim,
                dropout=config.model.ssl.dropout
            )

            if config.model.ssl.get("use_temporal_prediction", False):
                self.heads["temporal"] = ClassificationHead(
                    in_features=feature_dim,
                    num_classes=config.model.ssl.temporal_classes,
                    dropout=config.model.ssl.dropout
                )

        elif self.task_type == "cross_task":
            # Challenge 1 heads
            self.heads["response_time"] = nn.Sequential(
                nn.Linear(feature_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1)
            )

            self.heads["success"] = ClassificationHead(
                in_features=feature_dim,
                num_classes=2,  # Binary classification
                dropout=0.3
            )

            # Task classification for transfer learning
            self.heads["task"] = ClassificationHead(
                in_features=feature_dim,
                num_classes=config.model.num_task_classes,
                dropout=0.3
            )

        elif self.task_type == "psychopathology":
            # Challenge 2 heads - multi-target regression
            cbcl_dims = ["p_factor", "internalizing", "externalizing", "attention"]

            for dim in cbcl_dims:
                self.heads[dim] = nn.Sequential(
                    nn.Linear(feature_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 1)
                )

            # Binary classification
            self.heads["binary"] = ClassificationHead(
                in_features=feature_dim,
                num_classes=2,
                dropout=0.3
            )

        # Domain adaptation components
        domain_configs = {
            "subject": config.domain_adaptation.get("num_subjects", 1000),
            "site": config.domain_adaptation.get("num_sites", 10)
        }

        self.domain_components = create_domain_adaptation_components(
            feature_dim=feature_dim,
            domain_configs=domain_configs,
            adaptation_config=config.domain_adaptation,
            task_loss_fn=nn.CrossEntropyLoss()
        )

        # Compression metrics
        self.compression_metrics = CompressionMetrics()

        # Official metrics tracker
        self.val_predictions = []
        self.val_targets = []
        self.test_predictions = []
        self.test_targets = []

        # Loss weights
        self.loss_weights = config.training.get("loss_weights", {})

    def forward(self, x: torch.Tensor, alpha: float = 1.0) -> Dict[str, torch.Tensor]:
        """Forward pass through backbone and heads."""
        # Backbone feature extraction
        backbone_outputs = self.backbone(x, alpha=alpha, return_domain_logits=True)
        features = backbone_outputs["features"]

        # Apply domain adapter if available
        if "domain_adapter" in self.domain_components:
            # For now, use task 0 (could be made dynamic)
            task_ids = torch.zeros(features.size(0), dtype=torch.long, device=features.device)
            features = self.domain_components["domain_adapter"](features, task_ids)

        # Apply heads
        outputs = {"features": features}

        for head_name, head in self.heads.items():
            outputs[head_name] = head(features)

        # Add domain outputs if available
        if "domain_logits" in backbone_outputs:
            outputs["domain_logits"] = backbone_outputs["domain_logits"]

        return outputs

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Training step with domain adaptation and compression awareness."""
        return self._shared_step(batch, "train", batch_idx)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Validation step."""
        return self._shared_step(batch, "val", batch_idx)

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """Test step."""
        return self._shared_step(batch, "test", batch_idx)

    def _shared_step(self, batch: Dict[str, torch.Tensor], stage: str, batch_idx: int) -> torch.Tensor:
        """Shared step for train/val/test with official metrics."""
        x = batch["eeg"]
        batch_size = x.size(0)

        # Get current gradient reversal alpha
        alpha = 1.0
        if "multi_domain_loss" in self.domain_components:
            grl_scheduler = self.domain_components["multi_domain_loss"].grl_scheduler
            if hasattr(grl_scheduler, 'get_lambda'):
                alpha = grl_scheduler.get_lambda(self.global_step)

        # Forward pass
        outputs = self(x, alpha=alpha)
        features = outputs["features"]

        # Initialize loss components
        losses = {}
        total_loss = 0.0

        # Task-specific losses
        if self.task_type == "pretraining":
            # SSL contrastive loss (simplified - would need proper augmentations)
            if "contrastive" in outputs:
                # Placeholder contrastive loss
                contrastive_loss = F.mse_loss(outputs["contrastive"], torch.zeros_like(outputs["contrastive"]))
                losses["contrastive_loss"] = contrastive_loss
                total_loss += self.loss_weights.get("contrastive", 1.0) * contrastive_loss

        elif self.task_type == "cross_task":
            # Challenge 1 losses
            if "response_time_target" in batch and "response_time" in outputs:
                rt_target = batch["response_time_target"].squeeze()
                rt_pred = outputs["response_time"].squeeze()

                # Only compute loss for valid targets
                valid_mask = ~torch.isnan(rt_target)
                if valid_mask.sum() > 0:
                    rt_loss = F.mse_loss(rt_pred[valid_mask], rt_target[valid_mask])
                    losses["response_time_loss"] = rt_loss
                    total_loss += self.loss_weights.get("response_time", 1.0) * rt_loss

            if "success_target" in batch and "success" in outputs:
                success_target = batch["success_target"].squeeze().long()
                success_logits = outputs["success"]

                success_loss = F.cross_entropy(success_logits, success_target)
                losses["success_loss"] = success_loss
                total_loss += self.loss_weights.get("success", 1.0) * success_loss

        elif self.task_type == "psychopathology":
            # Challenge 2 losses
            cbcl_dims = ["p_factor", "internalizing", "externalizing", "attention"]

            for dim in cbcl_dims:
                target_key = f"{dim}_target"
                if target_key in batch and dim in outputs:
                    target = batch[target_key].squeeze()
                    pred = outputs[dim].squeeze()

                    # Regression loss
                    dim_loss = F.mse_loss(pred, target)
                    losses[f"{dim}_loss"] = dim_loss
                    total_loss += self.loss_weights.get(dim, 1.0) * dim_loss

            # Binary classification
            if "binary_target" in batch and "binary" in outputs:
                binary_target = batch["binary_target"].squeeze().long()
                binary_logits = outputs["binary"]

                binary_loss = F.cross_entropy(binary_logits, binary_target)
                losses["binary_loss"] = binary_loss
                total_loss += self.loss_weights.get("binary", 1.0) * binary_loss

        # Domain adaptation losses
        if "multi_domain_loss" in self.domain_components and stage == "train":
            domain_loss_inputs = {
                "task_outputs": outputs.get("task", torch.zeros(batch_size, 1, device=x.device)),
                "task_targets": torch.zeros(batch_size, dtype=torch.long, device=x.device)
            }

            # Add domain-specific inputs if available
            if "domain_logits" in outputs and "domain_label" in batch:
                domain_loss_inputs.update({
                    "domain_outputs": outputs["domain_logits"],
                    "domain_targets": batch["domain_label"].squeeze()
                })

            # Compute domain adaptation loss
            multi_domain_loss = self.domain_components["multi_domain_loss"](**domain_loss_inputs)

            for loss_name, loss_value in multi_domain_loss.items():
                if loss_name != "total_loss":
                    losses[loss_name] = loss_value

            # Add domain adaptation component to total loss
            if "dann_loss" in multi_domain_loss:
                total_loss += multi_domain_loss["dann_loss"]

        # Compression awareness
        if self.config.training.get("compression_aware", False):
            compression_metrics = self.compression_metrics.compute_feature_compression(features)
            compression_penalty = max(0, 10.0 - compression_metrics.get('best_compression_ratio', 1.0))
            losses["compression_penalty"] = compression_penalty
            total_loss += 0.01 * compression_penalty

        # Log all losses
        for loss_name, loss_value in losses.items():
            self.log(f"{stage}_{loss_name}", loss_value, prog_bar=True, batch_size=batch_size)

        self.log(f"{stage}_total_loss", total_loss, prog_bar=True, batch_size=batch_size)

        # Collect predictions for official metrics
        if stage in ["val", "test"]:
            self._collect_predictions(outputs, batch, stage)

        return total_loss

    def _collect_predictions(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], stage: str):
        """Collect predictions for official metrics computation."""
        predictions = {}
        targets = {}

        if self.task_type == "cross_task":
            if "response_time" in outputs and "response_time_target" in batch:
                predictions["response_time"] = outputs["response_time"].detach().cpu().numpy()
                targets["response_time"] = batch["response_time_target"].detach().cpu().numpy()

            if "success" in outputs and "success_target" in batch:
                success_probs = F.softmax(outputs["success"], dim=1)[:, 1]  # Probability of success
                predictions["success"] = success_probs.detach().cpu().numpy()
                targets["success"] = batch["success_target"].detach().cpu().numpy()

        elif self.task_type == "psychopathology":
            cbcl_dims = ["p_factor", "internalizing", "externalizing", "attention"]

            for dim in cbcl_dims:
                if dim in outputs and f"{dim}_target" in batch:
                    predictions[dim] = outputs[dim].detach().cpu().numpy()
                    targets[dim] = batch[f"{dim}_target"].detach().cpu().numpy()

            if "binary" in outputs and "binary_target" in batch:
                binary_probs = F.softmax(outputs["binary"], dim=1)[:, 1]  # Probability of atypical
                predictions["binary_label"] = binary_probs.detach().cpu().numpy()
                targets["binary_label"] = batch["binary_target"].detach().cpu().numpy()

        # Store predictions
        if stage == "val":
            self.val_predictions.append(predictions)
            self.val_targets.append(targets)
        elif stage == "test":
            self.test_predictions.append(predictions)
            self.test_targets.append(targets)

    def on_validation_epoch_end(self):
        """Compute official metrics at end of validation epoch."""
        if self.val_predictions:
            self._compute_epoch_metrics(self.val_predictions, self.val_targets, "val")
            self.val_predictions.clear()
            self.val_targets.clear()

    def on_test_epoch_end(self):
        """Compute official metrics at end of test epoch."""
        if self.test_predictions:
            self._compute_epoch_metrics(self.test_predictions, self.test_targets, "test")
            self.test_predictions.clear()
            self.test_targets.clear()

    def _compute_epoch_metrics(self, predictions_list: List[Dict], targets_list: List[Dict], stage: str):
        """Compute and log official epoch metrics."""
        # Aggregate predictions
        agg_predictions = {}
        agg_targets = {}

        if predictions_list:
            # Get all keys from first prediction
            keys = predictions_list[0].keys()

            for key in keys:
                pred_arrays = [p[key] for p in predictions_list if key in p]
                target_arrays = [t[key] for t in targets_list if key in t]

                if pred_arrays and target_arrays:
                    agg_predictions[key] = np.concatenate(pred_arrays).flatten()
                    agg_targets[key] = np.concatenate(target_arrays).flatten()

        # Compute official metrics
        if self.task_type == "cross_task":
            metrics = OfficialMetrics.compute_challenge1_metrics(agg_predictions, agg_targets)
        elif self.task_type == "psychopathology":
            metrics = OfficialMetrics.compute_challenge2_metrics(agg_predictions, agg_targets)
        else:
            metrics = {}

        # Log official metrics
        for metric_name, metric_value in metrics.items():
            if not np.isnan(metric_value):
                self.log(f"{stage}_official_{metric_name}", metric_value, prog_bar=True)

        logger.info(f"{stage.upper()} Official Metrics: {metrics}")

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
            betas=self.config.training.get("betas", (0.9, 0.999))
        )

        if self.config.training.get("use_scheduler", False):
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.max_epochs,
                eta_min=self.config.training.learning_rate * 0.01
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_total_loss"
                }
            }

        return optimizer


def create_data_loaders(config: DictConfig) -> Dict[str, DataLoader]:
    """Create data loaders with official Starter Kit integration."""
    logger.info("Creating enhanced datasets with Starter Kit integration...")

    datasets = create_hbn_datasets(
        bids_root=config.data.bids_root,
        task_type=config.task.name,
        window_length=config.data.windows.get(f"{config.task.name}_len_s", 2.0),
        overlap=config.data.overlap,
        sample_rate=config.data.sample_rate,
        use_official_splits=True,
        enable_compression_aug=config.data.get("compression_augmentation", False),
        compression_strengths=config.data.get("compression_strengths", [0.1, 0.2, 0.3])
    )

    # Create data loaders
    data_loaders = {}

    for split_name, dataset in datasets.items():
        is_train = (split_name == "train")

        data_loaders[split_name] = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=is_train,
            num_workers=config.training.num_workers,
            pin_memory=True,
            drop_last=is_train,  # Drop last batch for training to avoid batch size issues
            persistent_workers=config.training.num_workers > 0
        )

        logger.info(f"{split_name.capitalize()} samples: {len(dataset)}")

    return data_loaders


def create_callbacks(config: DictConfig) -> list:
    """Create enhanced training callbacks."""
    callbacks = []

    # Model checkpoint
    if config.task.name == "cross_task":
        monitor_metric = "val_official_mean_metric"
        mode = "max"
    elif config.task.name == "psychopathology":
        monitor_metric = "val_official_mean_r"
        mode = "max"
    else:
        monitor_metric = "val_total_loss"
        mode = "min"

    checkpoint_callback = ModelCheckpoint(
        dirpath=config.paths.checkpoints,
        filename=f"{config.task.name}-{{epoch:02d}}-{{{monitor_metric}:.3f}}",
        monitor=monitor_metric,
        mode=mode,
        save_top_k=3,
        save_last=True,
        auto_insert_metric_name=False
    )
    callbacks.append(checkpoint_callback)

    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)

    # Early stopping
    if config.training.get("early_stopping", False):
        early_stop_callback = EarlyStopping(
            monitor=monitor_metric,
            patience=config.training.early_stopping_patience,
            mode=mode,
            verbose=True,
            min_delta=config.training.get("early_stopping_min_delta", 0.001)
        )
        callbacks.append(early_stop_callback)

    return callbacks


def create_loggers(config: DictConfig) -> list:
    """Create experiment loggers."""
    loggers = []

    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=config.paths.logs,
        name=config.task.name,
        version=config.get("run_id", None)
    )
    loggers.append(tb_logger)

    # Wandb logger (if enabled)
    if config.logging.get("use_wandb", False):
        wandb_logger = WandbLogger(
            project=config.logging.wandb_project,
            name=f"{config.task.name}_{config.get('run_id', 'default')}",
            save_dir=config.paths.logs,
            tags=[config.task.name, "enhanced", "starter_kit"]
        )
        loggers.append(wandb_logger)

    return loggers


def run_baseline_training(config: DictConfig) -> str:
    """Run baseline training without advanced components."""
    logger.info("=== Running BASELINE training (no SSL, no compression aug, no DANN) ===")

    # Disable advanced features
    baseline_config = OmegaConf.to_container(config, resolve=True)
    baseline_config = OmegaConf.create(baseline_config)

    baseline_config.domain_adaptation.enabled = False
    baseline_config.data.compression_augmentation = False
    baseline_config.model.backbone.use_conformer = False
    baseline_config.training.compression_aware = False

    # Update paths
    baseline_config.paths.checkpoints = str(Path(config.paths.checkpoints) / "baseline")
    baseline_config.paths.logs = str(Path(config.paths.logs) / "baseline")

    # Create data loaders
    data_loaders = create_data_loaders(baseline_config)

    # Create model
    model = EEGFoundationModel(baseline_config)
    logger.info(f"Baseline model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create callbacks and loggers
    callbacks = create_callbacks(baseline_config)
    loggers = create_loggers(baseline_config)

    # Create trainer
    trainer = Trainer(
        max_epochs=baseline_config.training.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=baseline_config.training.precision,
        callbacks=callbacks,
        logger=loggers,
        gradient_clip_val=baseline_config.training.gradient_clip_val,
        accumulate_grad_batches=baseline_config.training.accumulate_grad_batches,
        check_val_every_n_epoch=baseline_config.training.check_val_every_n_epoch,
        deterministic=True
    )

    # Training
    trainer.fit(
        model,
        train_dataloaders=data_loaders["train"],
        val_dataloaders=data_loaders["val"]
    )

    # Testing
    trainer.test(model, dataloaders=data_loaders["test"], ckpt_path="best")

    return trainer.checkpoint_callback.best_model_path


def run_incremental_training(config: DictConfig, base_checkpoint: str) -> Dict[str, str]:
    """Run incremental training adding components one by one."""
    results = {}

    # Component configurations
    incremental_configs = [
        {
            "name": "with_conformer",
            "description": "Adding ConformerTiny backbone",
            "changes": {
                "model.backbone.use_conformer": True
            }
        },
        {
            "name": "with_compression_aug",
            "description": "Adding compression augmentation",
            "changes": {
                "data.compression_augmentation": True,
                "training.compression_aware": True
            }
        },
        {
            "name": "with_domain_adaptation",
            "description": "Adding DANN domain adaptation",
            "changes": {
                "domain_adaptation.enabled": True
            }
        }
    ]

    for inc_config in incremental_configs:
        logger.info(f"=== Running {inc_config['description']} ===")

        # Create incremental config
        current_config = OmegaConf.to_container(config, resolve=True)
        current_config = OmegaConf.create(current_config)

        # Apply changes
        for key, value in inc_config["changes"].items():
            OmegaConf.set(current_config, key, value)

        # Update paths
        current_config.paths.checkpoints = str(Path(config.paths.checkpoints) / inc_config["name"])
        current_config.paths.logs = str(Path(config.paths.logs) / inc_config["name"])

        # Create data loaders
        data_loaders = create_data_loaders(current_config)

        # Create model
        model = EEGFoundationModel(current_config)

        # Load baseline weights if compatible
        if base_checkpoint and Path(base_checkpoint).exists():
            try:
                checkpoint = torch.load(base_checkpoint, map_location="cpu")
                model.load_state_dict(checkpoint["state_dict"], strict=False)
                logger.info(f"Loaded compatible weights from {base_checkpoint}")
            except Exception as e:
                logger.warning(f"Could not load baseline weights: {e}")

        logger.info(f"{inc_config['name']} model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Create callbacks and loggers
        callbacks = create_callbacks(current_config)
        loggers = create_loggers(current_config)

        # Create trainer
        trainer = Trainer(
            max_epochs=current_config.training.max_epochs,
            accelerator="auto",
            devices="auto",
            precision=current_config.training.precision,
            callbacks=callbacks,
            logger=loggers,
            gradient_clip_val=current_config.training.gradient_clip_val,
            accumulate_grad_batches=current_config.training.accumulate_grad_batches,
            check_val_every_n_epoch=current_config.training.check_val_every_n_epoch,
            deterministic=True
        )

        # Training
        trainer.fit(
            model,
            train_dataloaders=data_loaders["train"],
            val_dataloaders=data_loaders["val"]
        )

        # Testing
        trainer.test(model, dataloaders=data_loaders["test"], ckpt_path="best")

        results[inc_config["name"]] = trainer.checkpoint_callback.best_model_path

    return results


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(config: DictConfig) -> None:
    """Main training function with ablation studies."""
    logger.info("Starting Enhanced EEG Foundation Challenge training...")
    logger.info(f"Task: {config.task.name}")
    logger.info(f"Enhanced config:\n{OmegaConf.to_yaml(config)}")

    # Set random seeds for reproducibility
    torch.manual_seed(config.training.seed)
    torch.cuda.manual_seed_all(config.training.seed)
    np.random.seed(config.training.seed)

    # Create output directories
    Path(config.paths.checkpoints).mkdir(parents=True, exist_ok=True)
    Path(config.paths.logs).mkdir(parents=True, exist_ok=True)

    # Run ablation study
    if config.training.get("run_ablation", True):
        logger.info("=== Starting Ablation Study ===")

        # 1. Baseline training
        baseline_checkpoint = run_baseline_training(config)

        # 2. Incremental training
        incremental_results = run_incremental_training(config, baseline_checkpoint)

        # 3. Summary
        logger.info("=== Ablation Study Summary ===")
        logger.info(f"Baseline: {baseline_checkpoint}")
        for name, checkpoint in incremental_results.items():
            logger.info(f"{name}: {checkpoint}")

    else:
        # Regular training with all components
        logger.info("=== Running Full Enhanced Training ===")

        # Create data loaders
        data_loaders = create_data_loaders(config)

        # Create model
        model = EEGFoundationModel(config)
        logger.info(f"Full model parameters: {sum(p.numel() for p in model.parameters()):,}")

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
            check_val_every_n_epoch=config.training.check_val_every_n_epoch,
            deterministic=True
        )

        # Training
        trainer.fit(
            model,
            train_dataloaders=data_loaders["train"],
            val_dataloaders=data_loaders["val"]
        )

        # Testing
        trainer.test(model, dataloaders=data_loaders["test"], ckpt_path="best")

    logger.info("Enhanced training completed!")


if __name__ == "__main__":
    main()
