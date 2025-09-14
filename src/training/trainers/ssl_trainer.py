"""
Self-Supervised Learning Trainer
=================================

Comprehensive trainer for EEG SSL with multi-task objectives.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader


class SSLTrainer:
    """
    Self-supervised learning trainer for EEG foundation models.

    Combines multiple SSL objectives:
    - Compression-augmented contrastive learning
    - Domain adversarial training
    - Task-aware adaptation
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 5e-5,
        weight_decay: float = 0.01,
        batch_size: int = 32,
        max_epochs: int = 100,
        warmup_epochs: int = 5,
        scheduler_type: str = "cosine_with_warmup",
        task_weights: Optional[Dict[str, float]] = None,
        device: str = "auto",
    ):
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs

        # Default task weights
        if task_weights is None:
            task_weights = {"ssl": 1.0, "domain_adaptation": 0.1, "main_task": 2.0}
        self.task_weights = task_weights

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )

        # Scheduler
        self.scheduler = self._create_scheduler(scheduler_type)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")

        # Logging
        self.logger = logging.getLogger(__name__)

    def _create_scheduler(self, scheduler_type: str):
        """Create learning rate scheduler."""
        total_steps = self.max_epochs * 1000  # Approximate
        warmup_steps = self.warmup_epochs * 1000

        if scheduler_type == "cosine_with_warmup":
            return optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_0=total_steps - warmup_steps, eta_min=1e-7
            )
        elif scheduler_type == "linear_warmup":
            return optim.lr_scheduler.LinearLR(
                self.optimizer, start_factor=0.1, total_iters=warmup_steps
            )
        else:
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.9)

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Single training step.

        Args:
            batch: Dictionary containing batch data
            batch_idx: Batch index

        Returns:
            Total loss
        """
        # Move batch to device
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Forward pass
        if hasattr(self.model, "training_step"):
            # Model has custom training step
            return self.model.training_step(batch, batch_idx)
        else:
            # Standard forward pass
            x = batch["eeg"]

            # Get model outputs
            if hasattr(self.model, "forward_ssl"):
                outputs = self.model.forward_ssl(x)
            else:
                outputs = self.model(x)

            # Compute loss (placeholder - should be implemented in model)
            if isinstance(outputs, dict):
                loss = outputs.get("loss", torch.tensor(0.0, device=self.device))
            else:
                # Simple MSE loss for demonstration
                target = batch.get("target", torch.zeros_like(outputs))
                loss = nn.functional.mse_loss(outputs, target)

            return loss

    def validation_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        """
        Single validation step.

        Args:
            batch: Dictionary containing batch data
            batch_idx: Batch index

        Returns:
            Dictionary of validation metrics
        """
        with torch.no_grad():
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            if hasattr(self.model, "validation_step"):
                return self.model.validation_step(batch, batch_idx)
            else:
                # Standard validation
                x = batch["eeg"]
                outputs = self.model(x)

                if isinstance(outputs, dict):
                    loss = outputs.get("loss", torch.tensor(0.0, device=self.device))
                else:
                    target = batch.get("target", torch.zeros_like(outputs))
                    loss = nn.functional.mse_loss(outputs, target)

                return {"val_loss": loss}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        log_every_n_steps: int = 50,
        val_every_n_epochs: int = 1,
    ) -> Dict[str, List[float]]:
        """
        Main training loop.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            log_every_n_steps: Logging frequency
            val_every_n_epochs: Validation frequency

        Returns:
            Training history
        """
        history = {"train_loss": [], "val_loss": []}

        self.logger.info(f"Starting training for {self.max_epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(
            f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}"
        )

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch

            # Training epoch
            train_loss = self._train_epoch(train_loader, log_every_n_steps)
            history["train_loss"].append(train_loss)

            # Validation epoch
            if val_loader is not None and epoch % val_every_n_epochs == 0:
                val_loss = self._val_epoch(val_loader)
                history["val_loss"].append(val_loss)

                # Check for best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self._save_checkpoint("best_model.pt")

            # Learning rate scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # Logging
            lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(
                f"Epoch {epoch:3d}/{self.max_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"LR: {lr:.2e}"
            )

            if val_loader is not None and len(history["val_loss"]) > 0:
                self.logger.info(f"Val Loss: {history['val_loss'][-1]:.4f}")

        self.logger.info("Training completed!")
        return history

    def _train_epoch(self, train_loader: DataLoader, log_every_n_steps: int) -> float:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(train_loader):
            self.global_step += 1

            # Forward pass
            loss = self.training_step(batch, batch_idx)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Optimizer step
            self.optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1

            # Logging
            if batch_idx % log_every_n_steps == 0:
                self.logger.debug(
                    f"Epoch {self.current_epoch} | "
                    f"Batch {batch_idx} | "
                    f"Loss: {loss.item():.4f}"
                )

        return total_loss / num_batches

    def _val_epoch(self, val_loader: DataLoader) -> float:
        """Run single validation epoch."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(val_loader):
            metrics = self.validation_step(batch, batch_idx)
            total_loss += metrics.get("val_loss", 0.0).item()
            num_batches += 1

        return total_loss / num_batches

    def _save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "best_val_loss": self.best_val_loss,
        }

        torch.save(checkpoint, filename)
        self.logger.info(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filename, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint["scheduler_state_dict"] and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        self.logger.info(f"Checkpoint loaded: {filename}")


class MultiTaskSSLTrainer(SSLTrainer):
    """
    Extended trainer for multi-task SSL with domain adaptation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Additional components for multi-task learning
        self.domain_schedulers = {}

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """
        Multi-task training step.

        Args:
            batch: Dictionary containing:
                - eeg: EEG data [batch_size, n_channels, seq_len]
                - task_ids: Task identifiers [batch_size]
                - subject_ids: Subject identifiers [batch_size]
                - site_ids: Site identifiers [batch_size]
                - targets: Target values (optional)

        Returns:
            Total weighted loss
        """
        x = batch["eeg"]

        # Extract features
        if hasattr(self.model, "feature_extractor"):
            features = self.model.feature_extractor(x)
        else:
            features = self.model(x)

        losses = {}
        total_loss = 0.0

        # SSL Loss (compression-augmented)
        if hasattr(self.model, "ssl_loss"):
            ssl_loss = self.model.ssl_loss(features)
            losses["ssl"] = ssl_loss
            total_loss += self.task_weights["ssl"] * ssl_loss

        # Domain adaptation loss
        if hasattr(self.model, "domain_adaptation") and "subject_ids" in batch:
            domain_labels = {
                "subject": batch.get("subject_ids"),
                "site": batch.get("site_ids"),
                "montage": batch.get("montage_ids"),
            }

            # Filter out None values
            domain_labels = {k: v for k, v in domain_labels.items() if v is not None}

            if domain_labels:
                domain_logits = self.model.domain_adaptation(features)
                domain_loss, _ = self.model.domain_adaptation.compute_domain_loss(
                    domain_logits, domain_labels
                )
                losses["domain"] = domain_loss
                total_loss += self.task_weights["domain_adaptation"] * domain_loss

        # Main task loss
        if "targets" in batch and hasattr(self.model, "task_head"):
            task_output = self.model.task_head(features)
            task_loss = nn.functional.mse_loss(task_output, batch["targets"])
            losses["main_task"] = task_loss
            total_loss += self.task_weights["main_task"] * task_loss

        # Store losses for logging
        if hasattr(self, "_current_losses"):
            self._current_losses.update(losses)
        else:
            self._current_losses = losses

        return total_loss


def create_ssl_trainer(config: Dict[str, Any], model: nn.Module) -> SSLTrainer:
    """
    Factory function to create SSL trainer from config.

    Args:
        config: Configuration dictionary
        model: Model to train

    Returns:
        SSL trainer instance
    """
    training_config = config.get("training", {})

    trainer_class = SSLTrainer
    if config.get("domain_adaptation", {}).get("enabled", False):
        trainer_class = MultiTaskSSLTrainer

    return trainer_class(
        model=model,
        learning_rate=training_config.get("lr", 5e-5),
        weight_decay=training_config.get("weight_decay", 0.01),
        batch_size=training_config.get("batch_size", 32),
        max_epochs=training_config.get("max_epochs", 100),
        warmup_epochs=training_config.get("warmup_epochs", 5),
        scheduler_type=training_config.get("scheduler", {}).get(
            "type", "cosine_with_warmup"
        ),
        task_weights=training_config.get("task_weights", None),
    )
