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


class CCDRegressionHead(nn.Module):
    """
    Regression head for CCD reaction time (RT) prediction.

    Predicts continuous reaction time values with proper normalization
    and uncertainty estimation capabilities.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[list] = None,
        dropout_rate: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True,
        predict_uncertainty: bool = False,
        output_activation: Optional[str] = None
    ):
        """
        Initialize CCD regression head.

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function name
            use_batch_norm: Whether to use batch normalization
            predict_uncertainty: Whether to predict uncertainty estimates
            output_activation: Output activation function (None, "sigmoid", "tanh")
        """
        super().__init__()

        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.predict_uncertainty = predict_uncertainty
        self.output_activation = output_activation

        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
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

        # Output layer(s)
        if predict_uncertainty:
            # Predict both mean and log variance
            self.mean_head = nn.Linear(prev_dim, 1)
            self.logvar_head = nn.Linear(prev_dim, 1)
        else:
            # Single output for RT prediction
            self.output_head = nn.Linear(prev_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for RT prediction.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            RT predictions [batch_size, 1] or [batch_size, 2] if uncertainty
        """
        # Pass through hidden layers
        features = self.layers(x)

        # Output prediction(s)
        if self.predict_uncertainty:
            # Predict mean and log variance
            mean = self.mean_head(features)
            logvar = self.logvar_head(features)

            # Apply output activation to mean if specified
            if self.output_activation == "sigmoid":
                mean = torch.sigmoid(mean)
            elif self.output_activation == "tanh":
                mean = torch.tanh(mean)

            # Combine mean and logvar
            output = torch.cat([mean, logvar], dim=-1)
        else:
            # Single RT prediction
            output = self.output_head(features)

            # Apply output activation if specified
            if self.output_activation == "sigmoid":
                output = torch.sigmoid(output)
            elif self.output_activation == "tanh":
                output = torch.tanh(output)

        return output


class CCDClassificationHead(nn.Module):
    """
    Classification head for CCD success prediction.

    Predicts binary success/failure with calibrated probabilities
    and optional class-wise confidence estimation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[list] = None,
        dropout_rate: float = 0.3,
        activation: str = "relu",
        use_batch_norm: bool = True,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Initialize CCD classification head.

        Args:
            input_dim: Dimension of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            activation: Activation function name
            use_batch_norm: Whether to use batch normalization
            class_weights: Weights for class balancing
        """
        super().__init__()

        self.input_dim = input_dim
        self.dropout_rate = dropout_rate
        self.class_weights = class_weights

        # Default hidden dimensions
        if hidden_dims is None:
            hidden_dims = [input_dim // 2, input_dim // 4]

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
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

        # Output layer (logits)
        self.output_head = nn.Linear(prev_dim, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize layer weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier/Glorot initialization
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for success prediction.

        Args:
            x: Input features [batch_size, input_dim]

        Returns:
            Success logits [batch_size, 1]
        """
        # Pass through hidden layers
        features = self.layers(x)

        # Output logits
        logits = self.output_head(features)

        return logits


def create_head(head_type: str, **kwargs) -> nn.Module:
    """
    Factory function to create classification heads.

    Args:
        head_type: Type of head ('simple', 'multitask', 'attention', 'adversarial', 'contrastive', 'ccd_regression', 'ccd_classification')
        **kwargs: Head-specific arguments

    Returns:
        Classification head module
    """
    head_map = {
        'simple': ClassificationHead,
        'multitask': MultiTaskHead,
        'attention': AttentionHead,
        'adversarial': DomainAdversarialHead,
        'contrastive': ContrastiveHead,
        'ccd_regression': CCDRegressionHead,
        'ccd_classification': CCDClassificationHead
    }

    if head_type not in head_map:
        raise ValueError(f"Unknown head type: {head_type}. Available: {list(head_map.keys())}")

    return head_map[head_type](**kwargs)
