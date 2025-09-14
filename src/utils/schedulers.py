"""
Parameter schedulers for SSL training.

This module implements various parameter scheduling strategies for SSL hyperparameters
like temperature, mask ratio, and distortion percentages.
"""

import math
from typing import Callable, Union


class ParameterScheduler:
    """
    Generic parameter scheduler for SSL training.

    Supports various scheduling strategies for hyperparameters during training.
    """

    def __init__(
        self,
        param_name: str,
        schedule_type: str = "linear",
        start_value: float = 0.0,
        end_value: float = 1.0,
        total_epochs: int = 100,
        warmup_epochs: int = 0,
        schedule_fn: Union[Callable, None] = None,
    ):
        self.param_name = param_name
        self.schedule_type = schedule_type
        self.start_value = start_value
        self.end_value = end_value
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.schedule_fn = schedule_fn

        self.current_epoch = 0
        self.current_value = start_value

    def step(self, epoch: int):
        """Update scheduler for given epoch."""
        self.current_epoch = epoch
        self.current_value = self.get_value(epoch)

    def get_value(self, epoch: int) -> float:
        """Get parameter value for given epoch."""
        if self.schedule_fn is not None:
            return self.schedule_fn(epoch, self.total_epochs)

        # Handle warmup phase
        if epoch < self.warmup_epochs:
            if self.schedule_type == "cosine_warmup":
                # Linear warmup to start_value
                return self.start_value * (epoch / self.warmup_epochs)
            else:
                return self.start_value

        # Adjust epoch for post-warmup scheduling
        effective_epoch = epoch - self.warmup_epochs
        effective_total = self.total_epochs - self.warmup_epochs

        if effective_total <= 0:
            return self.end_value

        # Compute progress ratio
        progress = min(effective_epoch / effective_total, 1.0)

        if self.schedule_type == "linear":
            return self.start_value + (self.end_value - self.start_value) * progress

        elif self.schedule_type == "cosine":
            # Cosine annealing
            cosine_progress = 0.5 * (1 + math.cos(math.pi * progress))
            return (
                self.end_value + (self.start_value - self.end_value) * cosine_progress
            )

        elif self.schedule_type == "cosine_warmup":
            # Cosine annealing after warmup
            cosine_progress = 0.5 * (1 + math.cos(math.pi * progress))
            return (
                self.end_value + (self.start_value - self.end_value) * cosine_progress
            )

        elif self.schedule_type == "exponential":
            # Exponential decay/growth
            if self.end_value > self.start_value:
                # Exponential growth
                factor = (self.end_value / self.start_value) ** progress
                return self.start_value * factor
            else:
                # Exponential decay
                factor = (self.start_value / self.end_value) ** (1 - progress)
                return self.end_value * factor

        elif self.schedule_type == "step":
            # Step function
            if progress < 0.33:
                return self.start_value
            elif progress < 0.67:
                return (self.start_value + self.end_value) / 2
            else:
                return self.end_value

        elif self.schedule_type == "polynomial":
            # Polynomial decay (power=2)
            return self.start_value + (self.end_value - self.start_value) * (
                progress**2
            )

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")


class TemperatureScheduler(ParameterScheduler):
    """Specialized scheduler for contrastive learning temperature."""

    def __init__(
        self,
        start_temp: float = 0.05,
        end_temp: float = 0.1,
        total_epochs: int = 100,
        warmup_epochs: int = 5,
    ):
        super().__init__(
            param_name="temperature",
            schedule_type="cosine_warmup",
            start_value=start_temp,
            end_value=end_temp,
            total_epochs=total_epochs,
            warmup_epochs=warmup_epochs,
        )


class MaskRatioScheduler(ParameterScheduler):
    """Specialized scheduler for mask ratio in masked reconstruction."""

    def __init__(
        self,
        start_ratio: float = 0.1,
        end_ratio: float = 0.3,
        total_epochs: int = 100,
        schedule_type: str = "linear",
    ):
        super().__init__(
            param_name="mask_ratio",
            schedule_type=schedule_type,
            start_value=start_ratio,
            end_value=end_ratio,
            total_epochs=total_epochs,
        )


class DistortionScheduler(ParameterScheduler):
    """Specialized scheduler for augmentation distortion percentage."""

    def __init__(
        self,
        start_distortion: float = 0.5,
        end_distortion: float = 1.0,
        total_epochs: int = 100,
        schedule_type: str = "cosine",
    ):
        super().__init__(
            param_name="distortion_pct",
            schedule_type=schedule_type,
            start_value=start_distortion,
            end_value=end_distortion,
            total_epochs=total_epochs,
        )


class LearningRateScheduler:
    """
    Learning rate scheduler with various strategies.

    Can be used as a wrapper around PyTorch schedulers.
    """

    def __init__(
        self,
        optimizer,
        schedule_type: str = "cosine",
        total_epochs: int = 100,
        warmup_epochs: int = 0,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
    ):
        self.optimizer = optimizer
        self.schedule_type = schedule_type
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.current_epoch = 0
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: int):
        """Update learning rate for given epoch."""
        self.current_epoch = epoch

        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group["lr"] = self.get_lr(epoch, base_lr)

    def get_lr(self, epoch: int, base_lr: float) -> float:
        """Get learning rate for given epoch."""
        # Warmup phase
        if epoch < self.warmup_epochs:
            return base_lr * (epoch / self.warmup_epochs)

        # Post-warmup scheduling
        effective_epoch = epoch - self.warmup_epochs
        effective_total = self.total_epochs - self.warmup_epochs

        if effective_total <= 0:
            return self.min_lr

        progress = min(effective_epoch / effective_total, 1.0)

        if self.schedule_type == "cosine":
            cosine_progress = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (base_lr - self.min_lr) * cosine_progress

        elif self.schedule_type == "linear":
            return base_lr * (1 - progress) + self.min_lr * progress

        elif self.schedule_type == "exponential":
            return base_lr * (0.1**progress)

        else:
            return base_lr


def create_scheduler_from_config(config: dict) -> ParameterScheduler:
    """
    Factory function to create scheduler from configuration.

    Args:
        config: Scheduler configuration dictionary

    Returns:
        ParameterScheduler instance
    """
    return ParameterScheduler(
        param_name=config.get("param_name", "parameter"),
        schedule_type=config.get("schedule_type", "linear"),
        start_value=config.get("start_value", 0.0),
        end_value=config.get("end_value", 1.0),
        total_epochs=config.get("total_epochs", 100),
        warmup_epochs=config.get("warmup_epochs", 0),
    )
