"""
Domain adaptation utilities for EEG Foundation Challenge 2025.

This module provides domain adaptation techniques including DANN, MMD,
and IRM for cross-subject and cross-site generalization.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)


class GradientReversalScheduler:
    """
    Scheduler for gradient reversal lambda in DANN training.

    Implements curriculum learning for domain adaptation with
    warmup from 0 to target lambda.
    """

    def __init__(
        self,
        max_lambda: float = 1.0,
        warmup_steps: int = 1000,
        decay_steps: Optional[int] = None,
        schedule_type: str = "linear"
    ):
        """
        Initialize gradient reversal scheduler.

        Args:
            max_lambda: Maximum lambda value
            warmup_steps: Number of warmup steps
            decay_steps: Number of decay steps (optional)
            schedule_type: Schedule type ('linear', 'exponential', 'cosine')
        """
        self.max_lambda = max_lambda
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.schedule_type = schedule_type
        self.step_count = 0

    def step(self) -> float:
        """Get current lambda value and increment step count."""
        current_lambda = self.get_lambda(self.step_count)
        self.step_count += 1
        return current_lambda

    def get_lambda(self, step: int) -> float:
        """Get lambda value for given step."""
        if step < self.warmup_steps:
            # Warmup phase
            progress = step / self.warmup_steps

            if self.schedule_type == "linear":
                return progress * self.max_lambda
            elif self.schedule_type == "exponential":
                return self.max_lambda * (1 - math.exp(-5 * progress))
            elif self.schedule_type == "cosine":
                return self.max_lambda * 0.5 * (1 - math.cos(math.pi * progress))
            else:
                raise ValueError(f"Unknown schedule type: {self.schedule_type}")

        elif self.decay_steps is not None and step >= self.warmup_steps + self.decay_steps:
            # Decay phase
            decay_progress = (step - self.warmup_steps) / self.decay_steps
            decay_progress = min(decay_progress, 1.0)

            if self.schedule_type == "linear":
                return self.max_lambda * (1 - decay_progress)
            elif self.schedule_type == "exponential":
                return self.max_lambda * math.exp(-5 * decay_progress)
            elif self.schedule_type == "cosine":
                return self.max_lambda * 0.5 * (1 + math.cos(math.pi * decay_progress))

        else:
            # Stable phase
            return self.max_lambda

    def reset(self):
        """Reset step count."""
        self.step_count = 0


class MMDLoss(nn.Module):
    """
    Maximum Mean Discrepancy loss for domain adaptation.

    Measures the distance between feature distributions from
    different domains using kernel embeddings.
    """

    def __init__(
        self,
        kernel_type: str = "rbf",
        kernel_mul: float = 2.0,
        kernel_num: int = 5,
        fix_sigma: Optional[float] = None
    ):
        """
        Initialize MMD loss.

        Args:
            kernel_type: Type of kernel ('rbf', 'linear', 'poly')
            kernel_mul: Kernel multiplier for RBF
            kernel_num: Number of kernels for multi-scale RBF
            fix_sigma: Fixed sigma for RBF kernel (if None, computed dynamically)
        """
        super().__init__()
        self.kernel_type = kernel_type
        self.kernel_mul = kernel_mul
        self.kernel_num = kernel_num
        self.fix_sigma = fix_sigma

    def rbf_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute RBF kernel matrix."""
        n_X = X.size(0)
        n_Y = Y.size(0)

        # Compute pairwise distances
        X_norm = (X ** 2).sum(dim=1, keepdim=True)
        Y_norm = (Y ** 2).sum(dim=1, keepdim=True)

        distances = X_norm + Y_norm.T - 2 * X @ Y.T
        distances = torch.clamp(distances, min=0)

        # Compute kernel
        if self.fix_sigma is not None:
            sigma = self.fix_sigma
        else:
            # Dynamic sigma based on median distance
            median_dist = torch.median(distances[distances > 0])
            sigma = median_dist / math.sqrt(2)

        # Multi-scale RBF
        kernel_val = torch.zeros_like(distances)
        for i in range(self.kernel_num):
            bandwidth = sigma * (self.kernel_mul ** i)
            kernel_val += torch.exp(-distances / (2 * bandwidth ** 2))

        return kernel_val / self.kernel_num

    def linear_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """Compute linear kernel matrix."""
        return X @ Y.T

    def polynomial_kernel(self, X: torch.Tensor, Y: torch.Tensor, degree: int = 3) -> torch.Tensor:
        """Compute polynomial kernel matrix."""
        return (X @ Y.T + 1) ** degree

    def forward(self, source_features: torch.Tensor, target_features: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD loss between source and target features.

        Args:
            source_features: Features from source domain [batch_source, dim]
            target_features: Features from target domain [batch_target, dim]

        Returns:
            MMD loss value
        """
        if self.kernel_type == "rbf":
            XX = self.rbf_kernel(source_features, source_features)
            YY = self.rbf_kernel(target_features, target_features)
            XY = self.rbf_kernel(source_features, target_features)
        elif self.kernel_type == "linear":
            XX = self.linear_kernel(source_features, source_features)
            YY = self.linear_kernel(target_features, target_features)
            XY = self.linear_kernel(source_features, target_features)
        elif self.kernel_type == "poly":
            XX = self.polynomial_kernel(source_features, source_features)
            YY = self.polynomial_kernel(target_features, target_features)
            XY = self.polynomial_kernel(source_features, target_features)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")

        # Compute MMD^2
        mmd_loss = XX.mean() + YY.mean() - 2 * XY.mean()

        return torch.clamp(mmd_loss, min=0)


class IRMPenalty(nn.Module):
    """
    Invariant Risk Minimization penalty for domain generalization.

    Encourages the model to learn representations that perform
    equally well across different environments/domains.
    """

    def __init__(self, penalty_weight: float = 1.0):
        """
        Initialize IRM penalty.

        Args:
            penalty_weight: Weight for the IRM penalty term
        """
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(
        self,
        logits_list: List[torch.Tensor],
        targets_list: List[torch.Tensor],
        dummy_classifier: nn.Module
    ) -> torch.Tensor:
        """
        Compute IRM penalty across multiple environments.

        Args:
            logits_list: List of logits for each environment
            targets_list: List of targets for each environment
            dummy_classifier: Dummy classifier for gradient computation

        Returns:
            IRM penalty value
        """
        penalty = 0.0

        for logits, targets in zip(logits_list, targets_list):
            if len(logits) == 0:
                continue

            # Compute loss for this environment
            loss = F.cross_entropy(logits, targets)

            # Compute gradient penalty
            grad = torch.autograd.grad(
                outputs=loss,
                inputs=dummy_classifier.weight,
                create_graph=True,
                retain_graph=True,
                only_inputs=True
            )[0]

            penalty += (grad ** 2).sum()

        return self.penalty_weight * penalty


class MultiDomainLoss(nn.Module):
    """
    Multi-domain loss combining task loss with domain adaptation.

    Supports DANN, MMD, and IRM for robust cross-domain learning.
    """

    def __init__(
        self,
        task_loss_fn: nn.Module,
        adaptation_methods: List[str] = ["dann"],
        dann_weight: float = 0.1,
        mmd_weight: float = 0.1,
        irm_weight: float = 1.0,
        scheduler_config: Optional[Dict] = None
    ):
        """
        Initialize multi-domain loss.

        Args:
            task_loss_fn: Base task loss function
            adaptation_methods: List of adaptation methods to use
            dann_weight: Weight for DANN loss
            mmd_weight: Weight for MMD loss
            irm_weight: Weight for IRM penalty
            scheduler_config: Configuration for gradient reversal scheduler
        """
        super().__init__()

        self.task_loss_fn = task_loss_fn
        self.adaptation_methods = adaptation_methods
        self.dann_weight = dann_weight
        self.mmd_weight = mmd_weight
        self.irm_weight = irm_weight

        # Initialize components
        if "dann" in adaptation_methods:
            scheduler_config = scheduler_config or {}
            self.grl_scheduler = GradientReversalScheduler(**scheduler_config)

        if "mmd" in adaptation_methods:
            self.mmd_loss = MMDLoss()

        if "irm" in adaptation_methods:
            self.irm_penalty = IRMPenalty(irm_weight)

    def forward(
        self,
        task_outputs: torch.Tensor,
        task_targets: torch.Tensor,
        domain_outputs: Optional[torch.Tensor] = None,
        domain_targets: Optional[torch.Tensor] = None,
        source_features: Optional[torch.Tensor] = None,
        target_features: Optional[torch.Tensor] = None,
        env_logits: Optional[List[torch.Tensor]] = None,
        env_targets: Optional[List[torch.Tensor]] = None,
        dummy_classifier: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-domain loss.

        Args:
            task_outputs: Task predictions
            task_targets: Task ground truth
            domain_outputs: Domain predictions (for DANN)
            domain_targets: Domain ground truth (for DANN)
            source_features: Source domain features (for MMD)
            target_features: Target domain features (for MMD)
            env_logits: Environment-wise logits (for IRM)
            env_targets: Environment-wise targets (for IRM)
            dummy_classifier: Dummy classifier (for IRM)

        Returns:
            Dictionary with loss components
        """
        losses = {}

        # Task loss
        task_loss = self.task_loss_fn(task_outputs, task_targets)
        losses["task_loss"] = task_loss

        total_loss = task_loss

        # DANN loss
        if "dann" in self.adaptation_methods and domain_outputs is not None:
            dann_loss = F.cross_entropy(domain_outputs, domain_targets)
            losses["dann_loss"] = dann_loss

            # Apply scheduled weight
            current_lambda = self.grl_scheduler.step()
            weighted_dann_loss = self.dann_weight * current_lambda * dann_loss
            losses["dann_weight"] = current_lambda

            total_loss += weighted_dann_loss

        # MMD loss
        if ("mmd" in self.adaptation_methods and
            source_features is not None and
            target_features is not None):

            mmd_loss = self.mmd_loss(source_features, target_features)
            losses["mmd_loss"] = mmd_loss
            total_loss += self.mmd_weight * mmd_loss

        # IRM penalty
        if ("irm" in self.adaptation_methods and
            env_logits is not None and
            env_targets is not None and
            dummy_classifier is not None):

            irm_penalty = self.irm_penalty(env_logits, env_targets, dummy_classifier)
            losses["irm_penalty"] = irm_penalty
            total_loss += irm_penalty

        losses["total_loss"] = total_loss

        return losses


class DomainClassifier(nn.Module):
    """Domain classifier for DANN with multiple domain types."""

    def __init__(
        self,
        feature_dim: int,
        domain_configs: Dict[str, int],
        hidden_dim: int = 256,
        dropout: float = 0.5
    ):
        """
        Initialize domain classifier.

        Args:
            feature_dim: Input feature dimension
            domain_configs: Dictionary mapping domain type to number of classes
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
        """
        super().__init__()

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Domain-specific heads
        self.domain_heads = nn.ModuleDict()
        for domain_type, num_classes in domain_configs.items():
            self.domain_heads[domain_type] = nn.Linear(hidden_dim, num_classes)

    def forward(
        self,
        features: torch.Tensor,
        domain_type: str,
        alpha: float = 1.0
    ) -> torch.Tensor:
        """
        Forward pass for specific domain type.

        Args:
            features: Input features
            domain_type: Type of domain to classify
            alpha: Gradient reversal strength

        Returns:
            Domain classification logits
        """
        from .enhanced_cnn import GradientReversalLayer

        # Apply gradient reversal
        reversed_features = GradientReversalLayer.apply(features, alpha)

        # Shared processing
        hidden = self.shared_layers(reversed_features)

        # Domain-specific classification
        if domain_type in self.domain_heads:
            return self.domain_heads[domain_type](hidden)
        else:
            raise ValueError(f"Unknown domain type: {domain_type}")


class DomainAdapter(nn.Module):
    """
    Lightweight domain adapter using FiLM (Feature-wise Linear Modulation).

    Provides task-specific adaptation while preserving shared representations.
    """

    def __init__(
        self,
        feature_dim: int,
        num_tasks: int,
        adapter_dim: Optional[int] = None
    ):
        """
        Initialize domain adapter.

        Args:
            feature_dim: Input feature dimension
            num_tasks: Number of tasks/domains
            adapter_dim: Adapter dimension (if None, use feature_dim // 4)
        """
        super().__init__()

        if adapter_dim is None:
            adapter_dim = max(feature_dim // 4, 32)

        self.num_tasks = num_tasks

        # Task embedding
        self.task_embedding = nn.Embedding(num_tasks, adapter_dim)

        # FiLM parameters
        self.gamma_net = nn.Linear(adapter_dim, feature_dim)
        self.beta_net = nn.Linear(adapter_dim, feature_dim)

        # Initialize to identity transformation
        nn.init.zeros_(self.gamma_net.weight)
        nn.init.ones_(self.gamma_net.bias)
        nn.init.zeros_(self.beta_net.weight)
        nn.init.zeros_(self.beta_net.bias)

    def forward(self, features: torch.Tensor, task_ids: torch.Tensor) -> torch.Tensor:
        """
        Apply task-specific adaptation to features.

        Args:
            features: Input features [batch_size, feature_dim]
            task_ids: Task IDs [batch_size]

        Returns:
            Adapted features
        """
        # Get task embeddings
        task_emb = self.task_embedding(task_ids)  # [batch_size, adapter_dim]

        # Compute FiLM parameters
        gamma = self.gamma_net(task_emb)  # [batch_size, feature_dim]
        beta = self.beta_net(task_emb)   # [batch_size, feature_dim]

        # Apply FiLM transformation
        adapted_features = gamma * features + beta

        return adapted_features


def create_domain_adaptation_components(
    feature_dim: int,
    domain_configs: Dict[str, int],
    adaptation_config: Dict,
    task_loss_fn: nn.Module
) -> Dict[str, nn.Module]:
    """
    Factory function to create domain adaptation components.

    Args:
        feature_dim: Feature dimension
        domain_configs: Domain configuration
        adaptation_config: Adaptation configuration
        task_loss_fn: Task loss function

    Returns:
        Dictionary with domain adaptation components
    """
    components = {}

    # Domain classifier
    if "dann" in adaptation_config.get("methods", []):
        components["domain_classifier"] = DomainClassifier(
            feature_dim=feature_dim,
            domain_configs=domain_configs,
            hidden_dim=adaptation_config.get("hidden_dim", 256),
            dropout=adaptation_config.get("dropout", 0.5)
        )

    # Domain adapter
    if adaptation_config.get("use_adapter", False):
        components["domain_adapter"] = DomainAdapter(
            feature_dim=feature_dim,
            num_tasks=adaptation_config.get("num_tasks", 10),
            adapter_dim=adaptation_config.get("adapter_dim")
        )

    # Multi-domain loss
    components["multi_domain_loss"] = MultiDomainLoss(
        task_loss_fn=task_loss_fn,
        adaptation_methods=adaptation_config.get("methods", ["dann"]),
        dann_weight=adaptation_config.get("dann_weight", 0.1),
        mmd_weight=adaptation_config.get("mmd_weight", 0.1),
        irm_weight=adaptation_config.get("irm_weight", 1.0),
        scheduler_config=adaptation_config.get("scheduler", {})
    )

    return components
