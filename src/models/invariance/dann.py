"""
Domain Adversarial Neural Networks (DANN) implementation for EEG domain adaptation.

This module implements the gradient reversal layer and domain adversarial training
components for learning domain-invariant representations across sites, scanners,
subjects, and other domain shifts in EEG data.

Enhanced with multi-adversary training and advanced scheduling strategies.

Reference: Ganin et al. "Domain-Adversarial Training of Neural Networks" (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Union, Tuple
import math
import numpy as np
from enum import Enum


class AdversaryType(Enum):
    """Types of adversarial discriminators."""
    SITE = "site"           # Site/scanner domain
    SUBJECT = "subject"     # Subject identity
    SESSION = "session"     # Recording session
    AGE_GROUP = "age_group" # Age group (pediatric/adult)
    TASK = "task"          # Task type


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer function.

    Forward pass is identity, backward pass reverses gradients with scaling factor lambda.
    """

    @staticmethod
    def forward(ctx, input_tensor: torch.Tensor, lambda_val: float) -> torch.Tensor:
        """Forward pass - identity function."""
        ctx.lambda_val = lambda_val
        return input_tensor.view_as(input_tensor)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple:
        """Backward pass - reverse gradients with lambda scaling."""
        return -ctx.lambda_val * grad_output, None


class GradientReversalLayer(nn.Module):
    """
    Gradient Reversal Layer for domain adversarial training.

    During forward pass, acts as identity. During backward pass, reverses gradients
    and scales them by lambda parameter.
    """

    def __init__(self, lambda_val: float = 1.0):
        """
        Initialize GRL.

        Args:
            lambda_val: Scaling factor for gradient reversal
        """
        super().__init__()
        self.lambda_val = lambda_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal."""
        return GradientReversalFunction.apply(x, self.lambda_val)

    def set_lambda(self, lambda_val: float):
        """Update lambda parameter."""
        self.lambda_val = lambda_val


class DomainAdversarialHead(nn.Module):
    """
    Domain adversarial head for distinguishing between domains.

    Predicts domain labels to encourage domain-invariant feature learning
    in the backbone network.
    """

    def __init__(
        self,
        input_dim: int,
        num_domains: int,
        hidden_dims: Optional[List[int]] = None,
        dropout_rate: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True
    ):
        """
        Initialize domain adversarial head.

        Args:
            input_dim: Dimension of input features
            num_domains: Number of domains to distinguish
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function name
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.input_dim = input_dim
        self.num_domains = num_domains

        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unknown activation: {activation}")

        # Build layers
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            # Linear layer
            layers.append(nn.Linear(prev_dim, hidden_dim))

            # Batch normalization
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation
            layers.append(self.activation)

            # Dropout
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        self.layers = nn.Sequential(*layers)

        # Output layer for domain classification
        self.domain_classifier = nn.Linear(prev_dim, num_domains)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for domain classification.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Domain logits [batch_size, num_domains]
        """
        features = self.layers(x)
        domain_logits = self.domain_classifier(features)
        return domain_logits


class GRLScheduler:
    """
    Scheduler for Gradient Reversal Layer lambda parameter.

    Implements various scheduling strategies for the GRL lambda parameter,
    including linear warmup, exponential, and adaptive schedules.
    """

    def __init__(
        self,
        strategy: str = "linear_warmup",
        initial_lambda: float = 0.0,
        final_lambda: float = 0.2,
        warmup_steps: int = 1000,
        total_steps: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize GRL scheduler.

        Args:
            strategy: Scheduling strategy ("linear_warmup", "exponential", "cosine", "adaptive")
            initial_lambda: Initial lambda value
            final_lambda: Final lambda value
            warmup_steps: Number of warmup steps
            total_steps: Total training steps
            **kwargs: Additional strategy-specific parameters
        """
        self.strategy = strategy
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps or warmup_steps * 10
        self.current_step = 0

        # Strategy-specific parameters
        self.decay_rate = kwargs.get("decay_rate", 0.95)
        self.adaptation_rate = kwargs.get("adaptation_rate", 0.1)
        self.domain_acc_history = []

    def step(self, domain_accuracy: Optional[float] = None) -> float:
        """
        Get the current lambda value and advance the scheduler.

        Args:
            domain_accuracy: Current domain classification accuracy (for adaptive scheduling)

        Returns:
            Current lambda value
        """
        if self.strategy == "linear_warmup":
            lambda_val = self._linear_warmup_schedule()
        elif self.strategy == "exponential":
            lambda_val = self._exponential_schedule()
        elif self.strategy == "cosine":
            lambda_val = self._cosine_schedule()
        elif self.strategy == "adaptive":
            lambda_val = self._adaptive_schedule(domain_accuracy)
        else:
            raise ValueError(f"Unknown scheduling strategy: {self.strategy}")

        self.current_step += 1
        return lambda_val

    def _linear_warmup_schedule(self) -> float:
        """Linear warmup from initial_lambda to final_lambda."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            progress = self.current_step / self.warmup_steps
            lambda_val = self.initial_lambda + progress * (self.final_lambda - self.initial_lambda)
        else:
            # Stay at final value
            lambda_val = self.final_lambda

        return lambda_val

    def _exponential_schedule(self) -> float:
        """Exponential schedule."""
        progress = min(self.current_step / self.total_steps, 1.0)
        lambda_val = self.initial_lambda + (self.final_lambda - self.initial_lambda) * (1 - math.exp(-5 * progress))
        return lambda_val

    def _cosine_schedule(self) -> float:
        """Cosine annealing schedule."""
        if self.current_step < self.warmup_steps:
            # Linear warmup
            progress = self.current_step / self.warmup_steps
            lambda_val = self.initial_lambda + progress * (self.final_lambda - self.initial_lambda)
        else:
            # Cosine annealing
            remaining_steps = self.current_step - self.warmup_steps
            total_remaining = self.total_steps - self.warmup_steps
            progress = remaining_steps / total_remaining
            lambda_val = self.final_lambda * (1 + math.cos(math.pi * progress)) / 2

        return lambda_val

    def _adaptive_schedule(self, domain_accuracy: Optional[float]) -> float:
        """Adaptive schedule based on domain classification accuracy."""
        if domain_accuracy is not None:
            self.domain_acc_history.append(domain_accuracy)

            # Keep only recent history
            if len(self.domain_acc_history) > 100:
                self.domain_acc_history = self.domain_acc_history[-100:]

            # If domain classifier is too good, increase lambda
            # If domain classifier is too bad, decrease lambda
            recent_acc = np.mean(self.domain_acc_history[-10:]) if len(self.domain_acc_history) >= 10 else domain_accuracy

            if recent_acc > 0.8:  # Domain classifier too good
                adjustment = self.adaptation_rate
            elif recent_acc < 0.6:  # Domain classifier too bad
                adjustment = -self.adaptation_rate
            else:
                adjustment = 0

            # Base schedule (linear warmup)
            base_lambda = self._linear_warmup_schedule()
            lambda_val = max(0, base_lambda + adjustment)
        else:
            # Fallback to linear warmup
            lambda_val = self._linear_warmup_schedule()

        return lambda_val


class MultiAdversaryDANN(nn.Module):
    """
    Multi-adversary DANN model for simultaneous invariance across multiple domains.

    Supports adversarial training across site, subject, session, age group, and task
    with individual lambda scheduling for each adversary.
    """

    def __init__(self,
                 backbone: nn.Module,
                 task_head: nn.Module,
                 adversary_configs: Dict[AdversaryType, Dict],
                 feature_dim: Optional[int] = None,
                 shared_grl: bool = False):
        """
        Initialize multi-adversary DANN.

        Args:
            backbone: Feature extraction backbone
            task_head: Task-specific prediction head
            adversary_configs: Configuration for each adversary type
            feature_dim: Dimension of backbone features
            shared_grl: Whether to use shared GRL for all adversaries
        """
        super().__init__()

        self.backbone = backbone
        self.task_head = task_head
        self.adversary_types = list(adversary_configs.keys())
        self.shared_grl = shared_grl

        # Auto-detect feature dimension
        if feature_dim is None:
            with torch.no_grad():
                dummy_input = torch.randn(1, 19, 1000)  # Typical EEG shape
                dummy_features = backbone(dummy_input)
                if isinstance(dummy_features, tuple):
                    dummy_features = dummy_features[0]
                feature_dim = dummy_features.shape[-1]

        self.feature_dim = feature_dim

        # Gradient reversal layers
        if shared_grl:
            self.grl = GradientReversalLayer(lambda_val=0.0)
            self.grls = {adv_type: self.grl for adv_type in self.adversary_types}
        else:
            self.grls = nn.ModuleDict({
                adv_type.value: GradientReversalLayer(lambda_val=0.0)
                for adv_type in self.adversary_types
            })

        # Domain adversarial heads
        self.domain_heads = nn.ModuleDict()
        self.schedulers = {}

        for adv_type, config in adversary_configs.items():
            # Create domain head
            head_config = config.get('head_config', {})
            head_config.setdefault('input_dim', feature_dim)
            head_config.setdefault('num_domains', config['num_domains'])

            self.domain_heads[adv_type.value] = DomainAdversarialHead(**head_config)

            # Create scheduler
            scheduler_config = config.get('scheduler_config', {})
            scheduler_config.setdefault('strategy', 'linear_warmup')
            scheduler_config.setdefault('initial_lambda', 0.0)
            scheduler_config.setdefault('final_lambda', 0.2)
            scheduler_config.setdefault('warmup_steps', 1000)

            self.schedulers[adv_type] = GRLScheduler(**scheduler_config)

    def forward(self,
                x: torch.Tensor,
                domain_labels: Optional[Dict[AdversaryType, torch.Tensor]] = None,
                return_features: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-adversary training.

        Args:
            x: Input data [batch_size, channels, time]
            domain_labels: Domain labels for each adversary type
            return_features: Whether to return backbone features

        Returns:
            Dictionary containing task predictions and domain predictions
        """
        # Extract features
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]

        # Task prediction
        task_pred = self.task_head(features)

        # Domain predictions
        domain_preds = {}
        for adv_type in self.adversary_types:
            # Apply gradient reversal
            if self.shared_grl:
                grl_features = self.grl(features)
            else:
                grl_features = self.grls[adv_type.value](features)

            # Domain prediction
            domain_pred = self.domain_heads[adv_type.value](grl_features)
            domain_preds[adv_type.value] = domain_pred

        # Prepare output
        output = {
            'task_pred': task_pred,
            'domain_preds': domain_preds
        }

        if return_features:
            output['features'] = features

        return output

    def update_lambda_values(self, domain_accuracies: Optional[Dict[AdversaryType, float]] = None):
        """
        Update lambda values for all adversaries.

        Args:
            domain_accuracies: Current domain accuracies for adaptive scheduling
        """
        for adv_type in self.adversary_types:
            # Get current lambda from scheduler
            domain_acc = domain_accuracies.get(adv_type) if domain_accuracies else None
            lambda_val = self.schedulers[adv_type].step(domain_acc)

            # Update GRL
            if self.shared_grl:
                # For shared GRL, use maximum lambda across all adversaries
                current_lambda = max(
                    self.schedulers[at].current_step / self.schedulers[at].warmup_steps * self.schedulers[at].final_lambda
                    for at in self.adversary_types
                )
                self.grl.set_lambda(current_lambda)
            else:
                self.grls[adv_type.value].set_lambda(lambda_val)

    def get_lambda_values(self) -> Dict[str, float]:
        """Get current lambda values for all adversaries."""
        if self.shared_grl:
            return {'shared': self.grl.lambda_val}
        else:
            return {
                adv_type.value: self.grls[adv_type.value].lambda_val
                for adv_type in self.adversary_types
            }

    def compute_adversarial_loss(self,
                                domain_preds: Dict[str, torch.Tensor],
                                domain_labels: Dict[AdversaryType, torch.Tensor],
                                weights: Optional[Dict[AdversaryType, float]] = None) -> torch.Tensor:
        """
        Compute weighted adversarial loss across all adversaries.

        Args:
            domain_preds: Domain predictions from forward pass
            domain_labels: True domain labels
            weights: Weights for each adversary type

        Returns:
            Combined adversarial loss
        """
        if weights is None:
            weights = {adv_type: 1.0 for adv_type in self.adversary_types}

        total_loss = 0.0
        total_weight = 0.0

        for adv_type in self.adversary_types:
            if adv_type in domain_labels:
                pred = domain_preds[adv_type.value]
                target = domain_labels[adv_type]
                weight = weights.get(adv_type, 1.0)

                # Compute cross-entropy loss
                loss = F.cross_entropy(pred, target, reduction='mean')
                total_loss += weight * loss
                total_weight += weight

        if total_weight > 0:
            return total_loss / total_weight
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)


class DANNModel(nn.Module):
    """
    Complete DANN model combining backbone, task head, and domain adversarial head.

    Implements domain adversarial neural network for learning domain-invariant
    representations while solving the main task.
    """

    def __init__(
        self,
        backbone: nn.Module,
        task_head: nn.Module,
        num_domains: int,
        feature_dim: Optional[int] = None,
        domain_head_config: Optional[Dict] = None,
        lambda_scheduler: Optional[GRLScheduler] = None
    ):
        """
        Initialize DANN model.

        Args:
            backbone: Feature extraction backbone
            task_head: Task-specific prediction head
            num_domains: Number of domains for adversarial training
            feature_dim: Dimension of backbone features (auto-detected if None)
            domain_head_config: Configuration for domain adversarial head
            lambda_scheduler: GRL lambda scheduler
        """
        super().__init__()

        self.backbone = backbone
        self.task_head = task_head
        self.num_domains = num_domains

        # Auto-detect feature dimension
        if feature_dim is None:
            with torch.no_grad():
                dummy_input = torch.randn(1, 19, 1000)  # Updated for EEG
                dummy_features = backbone(dummy_input)
                if isinstance(dummy_features, tuple):
                    dummy_features = dummy_features[0]
                feature_dim = dummy_features.shape[-1]

        self.feature_dim = feature_dim

        # Gradient reversal layer
        self.grl = GradientReversalLayer(lambda_val=0.0)

        # Domain adversarial head
        domain_head_config = domain_head_config or {}
        self.domain_head = DomainAdversarialHead(
            input_dim=feature_dim,
            num_domains=num_domains,
            **domain_head_config
        )

        # Lambda scheduler
        self.lambda_scheduler = lambda_scheduler

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
        update_lambda: bool = True,
        domain_accuracy: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with task and domain predictions.

        Args:
            x: Input tensor [batch_size, channels, time]
            return_features: Whether to return intermediate features
            update_lambda: Whether to update lambda scheduler
            domain_accuracy: Current domain accuracy for adaptive scheduling

        Returns:
            Dictionary with task predictions, domain predictions, and optionally features
        """
        # Extract features
        features = self.backbone(x)
        if isinstance(features, tuple):
            features = features[0]

        # Global pooling if needed
        if features.dim() == 3:  # [batch, time, features]
            pooled_features = features.mean(dim=1)  # [batch, features]
        else:
            pooled_features = features

        # Task prediction
        task_output = self.task_head(pooled_features)

        # Update lambda if scheduler is available
        if self.lambda_scheduler is not None and update_lambda:
            lambda_val = self.lambda_scheduler.step(domain_accuracy)
            self.grl.set_lambda(lambda_val)

        # Domain adversarial prediction
        reversed_features = self.grl(pooled_features)
        domain_output = self.domain_head(reversed_features)

        outputs = {
            "task_output": task_output,
            "domain_output": domain_output,
            "lambda": self.grl.lambda_val
        }

        if return_features:
            outputs["features"] = pooled_features
            outputs["raw_features"] = features

        return outputs

    def get_current_lambda(self) -> float:
        """Get current GRL lambda value."""
        return self.grl.lambda_val

    def set_lambda(self, lambda_val: float):
        """Manually set GRL lambda value."""
        self.grl.set_lambda(lambda_val)


class IRMPenalty:
    """
    Invariant Risk Minimization (IRM) penalty computation.

    Implements the IRM penalty for learning domain-invariant representations
    by encouraging the optimal classifier to be the same across domains.

    Reference: Arjovsky et al. "Invariant Risk Minimization" (2019)
    """

    def __init__(self, penalty_weight: float = 1.0):
        """
        Initialize IRM penalty.

        Args:
            penalty_weight: Weight for IRM penalty term
        """
        self.penalty_weight = penalty_weight

    def compute_penalty(
        self,
        features: torch.Tensor,
        targets: torch.Tensor,
        domain_ids: torch.Tensor,
        classifier: nn.Module
    ) -> torch.Tensor:
        """
        Compute IRM penalty.

        Args:
            features: Input features [batch_size, feature_dim]
            targets: Target values [batch_size]
            domain_ids: Domain identifiers [batch_size]
            classifier: Classifier module

        Returns:
            IRM penalty scalar
        """
        unique_domains = torch.unique(domain_ids)
        penalty = 0.0

        for domain in unique_domains:
            domain_mask = (domain_ids == domain)
            if domain_mask.sum() < 2:  # Need at least 2 samples
                continue

            domain_features = features[domain_mask]
            domain_targets = targets[domain_mask]

            # Create dummy classifier with gradient computation
            dummy_classifier = torch.ones_like(domain_features[:, :1], requires_grad=True)
            domain_predictions = (domain_features * dummy_classifier).sum(dim=1)

            # Compute loss for this domain
            domain_loss = F.mse_loss(domain_predictions, domain_targets)

            # Compute gradient of loss w.r.t. dummy classifier
            grad = torch.autograd.grad(domain_loss, dummy_classifier, create_graph=True)[0]

            # IRM penalty is the norm of the gradient
            penalty += grad.pow(2).sum()

        return self.penalty_weight * penalty


def create_dann_model(
    backbone: nn.Module,
    task_head: nn.Module,
    num_domains: int,
    lambda_schedule_config: Optional[Dict] = None,
    domain_head_config: Optional[Dict] = None
) -> DANNModel:
    """
    Factory function to create DANN model with default configurations.

    Args:
        backbone: Feature extraction backbone
        task_head: Task-specific head
        num_domains: Number of domains
        lambda_schedule_config: Configuration for lambda scheduler
        domain_head_config: Configuration for domain head

    Returns:
        Configured DANN model
    """
    # Default lambda scheduler config
    if lambda_schedule_config is None:
        lambda_schedule_config = {
            "strategy": "linear_warmup",
            "initial_lambda": 0.0,
            "final_lambda": 0.2,
            "warmup_steps": 1000
        }

    # Create lambda scheduler
    lambda_scheduler = GRLScheduler(**lambda_schedule_config)

    # Create DANN model
    model = DANNModel(
        backbone=backbone,
        task_head=task_head,
        num_domains=num_domains,
        domain_head_config=domain_head_config,
        lambda_scheduler=lambda_scheduler
    )

    return model
