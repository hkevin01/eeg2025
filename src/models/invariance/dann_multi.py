# -*- coding: utf-8 -*-
"""
Multi-Adversary DANN with Lambda Scheduler
==========================================

High-performance implementation of Domain Adversarial Neural Networks with:
- Multiple domain adversaries (subject, site, etc.)
- Flexible lambda scheduling (linear, cosine, step)
- Gradient reversal layer with runtime-configurable lambda
- Comprehensive loss weighting and normalization strategies

This implementation is optimized for the EEG Foundation Challenge 2025
and supports multi-site, multi-subject domain adaptation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal function for domain adversarial training.

    Forward pass: identity function
    Backward pass: multiply gradients by -lambda
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient reversal layer with runtime-configurable lambda.

    This layer applies the gradient reversal function during backpropagation
    while acting as an identity during forward pass.

    Args:
        lambda_val: Initial lambda value for gradient reversal

    Example:
        >>> grl = GradientReversalLayer(lambda_val=0.1)
        >>> x = torch.randn(32, 256, requires_grad=True)
        >>> y = grl(x, lambda_val=0.5)  # Override lambda at runtime
        >>> loss = y.sum()
        >>> loss.backward()
        >>> # x.grad will be -0.5 * ones_like(x)
    """

    def __init__(self, lambda_val: float = 0.0):
        super().__init__()
        self._lambda_val = float(lambda_val)

    @property
    def lambda_val(self) -> float:
        return self._lambda_val

    @lambda_val.setter
    def lambda_val(self, val: float):
        self._lambda_val = float(val)

    def forward(
        self, x: torch.Tensor, lambda_val: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional lambda override.

        Args:
            x: Input tensor
            lambda_val: Optional lambda value to override the instance value

        Returns:
            Output tensor (same as input during forward pass)
        """
        lmb = self._lambda_val if lambda_val is None else float(lambda_val)
        return GradientReversalFunction.apply(x, lmb)


@dataclass
class LambdaScheduleConfig:
    """
    Configuration for lambda scheduling in gradient reversal.

    Args:
        kind: Schedule type ("linear", "cosine", "step", "exponential")
        start: Starting lambda value
        end: Ending lambda value
        warmup_steps: Number of warmup steps where lambda = start * (step / warmup_steps)
        total_steps: Total number of training steps
        step_size: Step size for step schedule
        gamma: Decay factor for step schedule

    Example:
        >>> # Linear warmup from 0 to 0.25 over 10k steps
        >>> config = LambdaScheduleConfig(
        ...     kind="linear", start=0.0, end=0.25,
        ...     warmup_steps=1000, total_steps=10000
        ... )
    """

    kind: str = "linear"  # "linear", "cosine", "step", "exponential"
    start: float = 0.0
    end: float = 0.25
    warmup_steps: int = 0
    total_steps: int = 10000
    step_size: int = 1000  # For step schedule
    gamma: float = 0.5  # For step schedule


class LambdaScheduler:
    """
    Flexible lambda scheduler for gradient reversal layer.

    Supports multiple scheduling strategies:
    - Linear: Linear interpolation from start to end
    - Cosine: Cosine annealing schedule
    - Step: Step decay with gamma
    - Exponential: Exponential decay

    Args:
        config: LambdaScheduleConfig instance

    Example:
        >>> config = LambdaScheduleConfig(kind="cosine", start=0.0, end=0.2)
        >>> scheduler = LambdaScheduler(config)
        >>> for step in range(1000):
        ...     lambda_val = scheduler.step()
        ...     # Use lambda_val for GRL
    """

    def __init__(self, config: LambdaScheduleConfig):
        self.config = config
        self._step = 0

    def step(self) -> float:
        """Advance scheduler by one step and return lambda value."""
        self._step += 1
        return self.value()

    def value(self, step: Optional[int] = None) -> float:
        """Get lambda value for given step (or current step if None)."""
        s = self._step if step is None else step
        c = self.config

        # Warmup phase: linear ramp from 0 to start value
        if s < c.warmup_steps:
            if c.warmup_steps == 0:
                return c.start
            return c.start * (s / c.warmup_steps)

        # Main scheduling phase
        t = max(0, s - c.warmup_steps)
        T = max(1, c.total_steps - c.warmup_steps)
        progress = min(1.0, t / T)

        if c.kind == "linear":
            v = c.start + (c.end - c.start) * progress
        elif c.kind == "cosine":
            # Cosine annealing from start to end
            cos_term = 0.5 * (1 - math.cos(math.pi * progress))
            v = c.start + (c.end - c.start) * cos_term
        elif c.kind == "step":
            # Step decay
            n_steps = t // max(1, c.step_size)
            v = c.start * (c.gamma**n_steps)
            v = max(min(v, max(c.start, c.end)), min(c.start, c.end))
        elif c.kind == "exponential":
            # Exponential approach to end value
            exp_term = 1 - math.exp(-5 * progress)  # -5 for reasonable decay
            v = c.start + (c.end - c.start) * exp_term
        else:
            raise ValueError(f"Unknown schedule kind: {c.kind}")

        return float(v)

    def reset(self):
        """Reset scheduler to initial state."""
        self._step = 0


class DomainAdversary(nn.Module):
    """
    Domain adversary network for one specific domain (e.g., subject or site).

    A simple MLP classifier that tries to predict domain labels from
    gradient-reversed features. The gradient reversal encourages the
    feature extractor to learn domain-invariant representations.

    Args:
        emb_dim: Input feature dimension
        n_classes: Number of domain classes (e.g., number of subjects)
        hidden: Hidden layer size
        dropout: Dropout rate
        norm: Normalization type ("layernorm", "batchnorm", None)
        activation: Activation function ("relu", "gelu", "leaky_relu")

    Example:
        >>> # Subject adversary for 100 subjects
        >>> subject_adv = DomainAdversary(
        ...     emb_dim=256, n_classes=100, hidden=128
        ... )
        >>> z = torch.randn(32, 256)  # Batch of embeddings
        >>> logits = subject_adv(z)   # (32, 100)
    """

    def __init__(
        self,
        emb_dim: int,
        n_classes: int,
        hidden: int = 128,
        dropout: float = 0.1,
        norm: Optional[str] = "layernorm",
        activation: str = "relu",
    ):
        super().__init__()

        # Build normalization layer
        norm_layer = None
        if norm == "layernorm":
            norm_layer = nn.LayerNorm(emb_dim)
        elif norm == "batchnorm":
            norm_layer = nn.BatchNorm1d(emb_dim)

        # Build activation
        if activation == "relu":
            act_fn = nn.ReLU(inplace=True)
        elif activation == "gelu":
            act_fn = nn.GELU()
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build network
        layers: List[nn.Module] = []
        if norm_layer is not None:
            layers.append(norm_layer)

        layers.extend(
            [
                nn.Linear(emb_dim, hidden),
                act_fn,
                nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            ]
        )

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adversary network.

        Args:
            z: Input embeddings (B, emb_dim)

        Returns:
            Domain classification logits (B, n_classes)
        """
        return self.net(z)


class MultiAdversary(nn.Module):
    """
    Multi-adversary DANN with multiple domain classifiers sharing a GRL.

    This module handles multiple domain adaptation objectives simultaneously,
    such as invariance to subject identity and site/scanner effects.

    Args:
        emb_dim: Input embedding dimension
        domains: Dictionary mapping domain names to configuration
        grl_lambda_init: Initial lambda value for gradient reversal
        loss_weights: Per-domain loss weights
        norm: Normalization type for adversaries
        dropout: Dropout rate for adversaries
        temperature: Temperature for softmax (if used)

    Example:
        >>> domains = {
        ...     "subject": {"n_classes": 100, "hidden": 256},
        ...     "site": {"n_classes": 3, "hidden": 128},
        ... }
        >>> dann = MultiAdversary(
        ...     emb_dim=256, domains=domains,
        ...     loss_weights={"subject": 1.0, "site": 0.5}
        ... )
        >>> z = torch.randn(32, 256)
        >>> targets = {
        ...     "subject": torch.randint(0, 100, (32,)),
        ...     "site": torch.randint(0, 3, (32,))
        ... }
        >>> logits, loss = dann(z, lambda_val=0.1, targets=targets)
    """

    def __init__(
        self,
        emb_dim: int,
        domains: Dict[str, Dict],
        grl_lambda_init: float = 0.0,
        loss_weights: Optional[Dict[str, float]] = None,
        norm: Optional[str] = "layernorm",
        dropout: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()

        self.grl = GradientReversalLayer(grl_lambda_init)
        self.classifiers = nn.ModuleDict()
        self.loss_weights: Dict[str, float] = loss_weights or {}
        self.temperature = temperature
        self.domain_names = list(domains.keys())

        # Build domain classifiers
        for name, config in domains.items():
            n_classes = int(config["n_classes"])
            hidden = int(config.get("hidden", 128))
            activation = config.get("activation", "relu")

            self.classifiers[name] = DomainAdversary(
                emb_dim=emb_dim,
                n_classes=n_classes,
                hidden=hidden,
                dropout=dropout,
                norm=norm,
                activation=activation,
            )

    def forward(
        self,
        z: torch.Tensor,
        lambda_val: Optional[float] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
        reduction: str = "mean",
        return_probs: bool = False,
    ) -> Tuple[
        Dict[str, torch.Tensor],
        Optional[torch.Tensor],
        Optional[Dict[str, torch.Tensor]],
    ]:
        """
        Forward pass through GRL and all domain adversaries.

        Args:
            z: Input embeddings (B, emb_dim)
            lambda_val: Lambda value for gradient reversal (overrides instance value)
            targets: Optional dict of domain targets per domain
            reduction: Loss reduction ("mean", "sum", "none")
            return_probs: Whether to return probabilities along with logits

        Returns:
            logits: Dict of domain -> logits (B, n_classes)
            total_loss: Combined adversarial loss (None if no targets)
            probs: Dict of domain -> probabilities (only if return_probs=True)
        """
        # Apply gradient reversal
        z_rev = self.grl(z, lambda_val=lambda_val)

        logits: Dict[str, torch.Tensor] = {}
        probs: Dict[str, torch.Tensor] = {} if return_probs else {}
        losses: List[torch.Tensor] = []

        # Forward through each domain classifier
        for name, classifier in self.classifiers.items():
            domain_logits = classifier(z_rev)
            logits[name] = domain_logits / self.temperature

            if return_probs:
                probs[name] = F.softmax(logits[name], dim=-1)

            # Compute loss if targets provided
            if targets is not None and name in targets and targets[name] is not None:
                weight = float(self.loss_weights.get(name, 1.0))
                ce_loss = F.cross_entropy(
                    logits[name], targets[name], reduction=reduction
                )
                weighted_loss = weight * ce_loss
                losses.append(weighted_loss)

        # Combine losses
        total_loss = None
        if losses:
            if len(losses) == 1:
                total_loss = losses[0]
            else:
                total_loss = torch.stack(losses).sum()

        return logits, total_loss, probs if return_probs else None

    def get_domain_accuracy(
        self, logits: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute domain classification accuracy for monitoring.

        Args:
            logits: Domain logits from forward pass
            targets: True domain labels

        Returns:
            Dictionary of domain -> accuracy
        """
        accuracies = {}
        for name in self.domain_names:
            if name in logits and name in targets and targets[name] is not None:
                preds = torch.argmax(logits[name], dim=-1)
                correct = (preds == targets[name]).float()
                accuracies[name] = correct.mean().item()
        return accuracies

    def update_lambda(self, lambda_val: float):
        """Update the GRL lambda value."""
        self.grl.lambda_val = lambda_val


def create_lambda_scheduler_from_config(
    config: Dict, total_steps: int, warmup_ratio: float = 0.1
) -> LambdaScheduler:
    """
    Create lambda scheduler from configuration dictionary.

    Args:
        config: Configuration dictionary with schedule parameters
        total_steps: Total number of training steps
        warmup_ratio: Fraction of steps to use for warmup

    Returns:
        Configured LambdaScheduler instance

    Example:
        >>> config = {
        ...     "lambda_schedule": "cosine",
        ...     "lambda_start": 0.0,
        ...     "lambda_end": 0.25,
        ... }
        >>> scheduler = create_lambda_scheduler_from_config(config, 10000)
    """
    schedule_config = LambdaScheduleConfig(
        kind=config.get("lambda_schedule", "linear"),
        start=float(config.get("lambda_start", 0.0)),
        end=float(config.get("lambda_end", 0.25)),
        warmup_steps=int(
            config.get("lambda_warmup_steps", int(total_steps * warmup_ratio))
        ),
        total_steps=int(config.get("lambda_total_steps", total_steps)),
        step_size=int(config.get("lambda_step_size", max(1, total_steps // 10))),
        gamma=float(config.get("lambda_gamma", 0.5)),
    )
    return LambdaScheduler(schedule_config)


def create_multi_adversary_from_config(
    config: Dict, emb_dim: int, domain_info: Dict[str, int]
) -> MultiAdversary:
    """
    Create MultiAdversary from configuration and domain information.

    Args:
        config: Configuration dictionary
        emb_dim: Embedding dimension
        domain_info: Dict mapping domain names to number of classes

    Returns:
        Configured MultiAdversary instance

    Example:
        >>> config = {
        ...     "adversaries": {
        ...         "subject": {"weight": 1.0, "hidden": 256},
        ...         "site": {"weight": 0.5, "hidden": 128}
        ...     }
        ... }
        >>> domain_info = {"subject": 100, "site": 3}
        >>> dann = create_multi_adversary_from_config(config, 256, domain_info)
    """
    adversary_config = config.get("adversaries", {})

    # Build domains dictionary
    domains = {}
    loss_weights = {}

    for domain_name, n_classes in domain_info.items():
        if domain_name in adversary_config:
            domain_cfg = adversary_config[domain_name]
            domains[domain_name] = {
                "n_classes": n_classes,
                "hidden": domain_cfg.get("hidden", 128),
                "activation": domain_cfg.get("activation", "relu"),
            }
            loss_weights[domain_name] = float(domain_cfg.get("weight", 1.0))

    return MultiAdversary(
        emb_dim=emb_dim,
        domains=domains,
        grl_lambda_init=float(config.get("lambda_start", 0.0)),
        loss_weights=loss_weights,
        norm=config.get("adversary_norm", "layernorm"),
        dropout=float(config.get("adversary_dropout", 0.1)),
        temperature=float(config.get("adversary_temperature", 1.0)),
    )


# Example usage and testing functions
if __name__ == "__main__":
    # Test lambda scheduler
    config = LambdaScheduleConfig(
        kind="cosine", start=0.0, end=0.25, warmup_steps=100, total_steps=1000
    )
    scheduler = LambdaScheduler(config)

    print("Lambda schedule test:")
    for step in [0, 50, 100, 500, 1000]:
        lambda_val = scheduler.value(step)
        print(f"Step {step}: λ = {lambda_val:.4f}")

    # Test multi-adversary
    torch.manual_seed(42)
    batch_size, emb_dim = 16, 256

    domains = {
        "subject": {"n_classes": 50, "hidden": 128},
        "site": {"n_classes": 3, "hidden": 64},
    }

    dann = MultiAdversary(
        emb_dim=emb_dim, domains=domains, loss_weights={"subject": 1.0, "site": 0.5}
    )

    # Test forward pass
    z = torch.randn(batch_size, emb_dim)
    targets = {
        "subject": torch.randint(0, 50, (batch_size,)),
        "site": torch.randint(0, 3, (batch_size,)),
    }

    logits, loss, _ = dann(z, lambda_val=0.1, targets=targets)

    print(f"\nMulti-adversary test:")
    print(f"Subject logits shape: {logits['subject'].shape}")
    print(f"Site logits shape: {logits['site'].shape}")
    print(f"Total loss: {loss.item():.4f}")

    # Test accuracy computation
    accuracies = dann.get_domain_accuracy(logits, targets)
    print(f"Domain accuracies: {accuracies}")
