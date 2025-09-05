"""
Classification heads for EEG Foundation Challenge models.

This module provides various classification head architectures for different
tasks in the EEG Foundation Challenge.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ClassificationHead(nn.Module):
    """
    Simple classification head with dropout and batch normalization.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.5,
        use_batch_norm: bool = True
    ):
        """
        Initialize classification head.

        Args:
            in_features: Number of input features
            num_classes: Number of output classes
            hidden_dim: Hidden dimension (if None, use single layer)
            dropout: Dropout probability
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        self.num_classes = num_classes

        if hidden_dim is None:
            # Single layer classifier
            layers = []
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(in_features))
            layers.extend([
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            ])
        else:
            # Two layer classifier
            layers = []
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(in_features))
            layers.extend([
                nn.Dropout(dropout),
                nn.Linear(in_features, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes)
            ])

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, in_features]

        Returns:
            Class logits [batch_size, num_classes]
        """
        return self.classifier(x)


class MultiTaskHead(nn.Module):
    """
    Multi-task classification head with shared and task-specific components.
    """

    def __init__(
        self,
        in_features: int,
        task_configs: dict,
        shared_dim: int = 256,
        dropout: float = 0.5
    ):
        """
        Initialize multi-task head.

        Args:
            in_features: Number of input features
            task_configs: Dictionary with task names and number of classes
            shared_dim: Dimension of shared representation
            dropout: Dropout probability
        """
        super().__init__()

        # Shared layers
        self.shared_layers = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(dropout),
            nn.Linear(in_features, shared_dim),
            nn.ReLU(),
            nn.BatchNorm1d(shared_dim),
            nn.Dropout(dropout)
        )

        # Task-specific heads
        self.task_heads = nn.ModuleDict()
        for task_name, num_classes in task_configs.items():
            self.task_heads[task_name] = nn.Linear(shared_dim, num_classes)

    def forward(self, x: torch.Tensor) -> dict:
        """
        Forward pass.

        Args:
            x: Input features [batch_size, in_features]

        Returns:
            Dictionary with task predictions
        """
        shared_features = self.shared_layers(x)

        outputs = {}
        for task_name, head in self.task_heads.items():
            outputs[task_name] = head(shared_features)

        return outputs


class AttentionHead(nn.Module):
    """
    Classification head with attention mechanism.
    """

    def __init__(
        self,
        in_features: int,
        num_classes: int,
        num_heads: int = 8,
        dropout: float = 0.5
    ):
        """
        Initialize attention head.

        Args:
            in_features: Number of input features
            num_classes: Number of output classes
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(in_features)
        self.classifier = ClassificationHead(
            in_features=in_features,
            num_classes=num_classes,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with self-attention.

        Args:
            x: Input features [batch_size, seq_len, in_features] or [batch_size, in_features]

        Returns:
            Class logits [batch_size, num_classes]
        """
        if x.dim() == 2:
            # Add sequence dimension
            x = x.unsqueeze(1)  # [batch_size, 1, in_features]

        # Self-attention
        attn_output, _ = self.attention(x, x, x)
        attn_output = self.norm(attn_output + x)

        # Global average pooling
        pooled = attn_output.mean(dim=1)  # [batch_size, in_features]

        # Classification
        return self.classifier(pooled)


class DomainAdversarialHead(nn.Module):
    """
    Domain adversarial head for domain adaptation.
    """

    def __init__(
        self,
        in_features: int,
        num_domains: int,
        hidden_dim: int = 256,
        dropout: float = 0.5,
        gradient_reversal_lambda: float = 1.0
    ):
        """
        Initialize domain adversarial head.

        Args:
            in_features: Number of input features
            num_domains: Number of domains (subjects/sites)
            hidden_dim: Hidden dimension
            dropout: Dropout probability
            gradient_reversal_lambda: Gradient reversal strength
        """
        super().__init__()

        self.gradient_reversal_lambda = gradient_reversal_lambda

        self.domain_classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with gradient reversal.

        Args:
            x: Input features [batch_size, in_features]

        Returns:
            Domain predictions [batch_size, num_domains]
        """
        # Apply gradient reversal
        reversed_features = GradientReversalFunction.apply(x, self.gradient_reversal_lambda)
        return self.domain_classifier(reversed_features)


class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal layer for domain adaptation.
    """

    @staticmethod
    def forward(ctx, x, lambda_param):
        ctx.lambda_param = lambda_param
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_param, None


class ContrastiveHead(nn.Module):
    """
    Contrastive learning head for self-supervised pretraining.
    """

    def __init__(
        self,
        in_features: int,
        projection_dim: int = 128,
        hidden_dim: int = 512,
        dropout: float = 0.1
    ):
        """
        Initialize contrastive head.

        Args:
            in_features: Number of input features
            projection_dim: Dimension of projection space
            hidden_dim: Hidden dimension
            dropout: Dropout probability
        """
        super().__init__()

        self.projector = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for contrastive learning.

        Args:
            x: Input features [batch_size, in_features]

        Returns:
            Projected features [batch_size, projection_dim]
        """
        projected = self.projector(x)
        # L2 normalize for contrastive learning
        return F.normalize(projected, dim=1)


def create_head(head_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create classification heads.

    Args:
        head_type: Type of head ('simple', 'multitask', 'attention', 'adversarial', 'contrastive')
        **kwargs: Head-specific arguments

    Returns:
        Classification head module
    """
    head_map = {
        'simple': ClassificationHead,
        'multitask': MultiTaskHead,
        'attention': AttentionHead,
        'adversarial': DomainAdversarialHead,
        'contrastive': ContrastiveHead
    }

    if head_type not in head_map:
        raise ValueError(f"Unknown head type: {head_type}. Available: {list(head_map.keys())}")

    return head_map[head_type](**kwargs)
