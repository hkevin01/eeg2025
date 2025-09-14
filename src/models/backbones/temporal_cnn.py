"""
Temporal CNN backbone for EEG feature extraction.

This module implements an efficient temporal convolutional neural network
optimized for EEG signal processing with depthwise separable convolutions.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class DepthwiseSeparableConv1d(nn.Module):
    """
    Depthwise separable 1D convolution for efficiency.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        padding: Padding mode
        bias: Whether to use bias
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False,
    ):
        super().__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

        # Pointwise convolution
        self.pointwise = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class TemporalBlock(nn.Module):
    """
    Temporal convolutional block with residual connection.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        dilation: Dilation rate
        dropout: Dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation // 2

        self.conv1 = DepthwiseSeparableConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.conv2 = DepthwiseSeparableConv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual connection
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        # First conv block
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        out = self.dropout(out)

        # Second conv block
        out = self.conv2(out)
        out = self.norm2(out)

        # Add residual connection
        out = out + residual
        out = self.activation(out)

        return out


class TemporalCNN(nn.Module):
    """
    Temporal CNN backbone for EEG feature extraction.

    This architecture uses depthwise separable convolutions and dilated
    convolutions to efficiently capture temporal patterns in EEG data.

    Args:
        n_channels: Number of EEG channels
        n_classes: Number of output classes/features
        hidden_dims: List of hidden dimensions for each layer
        kernel_sizes: List of kernel sizes for each layer
        dilations: List of dilation rates for each layer
        dropout: Dropout probability
        pool_size: Global pooling size
    """

    def __init__(
        self,
        n_channels: int = 64,
        n_classes: int = 128,
        hidden_dims: Optional[Tuple[int, ...]] = None,
        kernel_sizes: Optional[Tuple[int, ...]] = None,
        dilations: Optional[Tuple[int, ...]] = None,
        dropout: float = 0.1,
        pool_size: int = 4,
    ):
        super().__init__()

        # Default architecture parameters
        if hidden_dims is None:
            hidden_dims = (32, 64, 128, 256)
        if kernel_sizes is None:
            kernel_sizes = (7, 7, 7, 7)
        if dilations is None:
            dilations = (1, 2, 4, 8)

        self.n_channels = n_channels
        self.n_classes = n_classes

        # Input projection
        self.input_proj = nn.Conv1d(
            in_channels=n_channels,
            out_channels=hidden_dims[0],
            kernel_size=1,
            bias=False,
        )
        self.input_norm = nn.BatchNorm1d(hidden_dims[0])

        # Temporal blocks
        layers = []
        in_dim = hidden_dims[0]

        for out_dim, kernel_size, dilation in zip(hidden_dims, kernel_sizes, dilations):
            layers.append(
                TemporalBlock(
                    in_channels=in_dim,
                    out_channels=out_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_dim = out_dim

        self.temporal_layers = nn.ModuleList(layers)

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(pool_size)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_dims[-1] * pool_size, hidden_dims[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], n_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, n_channels, sequence_length)

        Returns:
            Output features of shape (batch_size, n_classes)
        """
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = torch.relu(x)

        # Temporal processing
        for layer in self.temporal_layers:
            x = layer(x)

        # Global pooling and output projection
        x = self.global_pool(x)
        x = self.output_proj(x)

        return x

    def get_features(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract features from intermediate layer.

        Args:
            x: Input tensor
            layer_idx: Layer index to extract features from (-1 for last layer)

        Returns:
            Intermediate features
        """
        # Input projection
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = torch.relu(x)

        # Process through temporal layers
        for i, layer in enumerate(self.temporal_layers):
            x = layer(x)
            if i == layer_idx or (
                layer_idx == -1 and i == len(self.temporal_layers) - 1
            ):
                break

        return x


def create_temporal_cnn(
    n_channels: int = 64, model_size: str = "small", **kwargs
) -> TemporalCNN:
    """
    Factory function to create TemporalCNN with predefined configurations.

    Args:
        n_channels: Number of EEG channels
        model_size: Model size ('tiny', 'small', 'medium', 'large')
        **kwargs: Additional arguments passed to TemporalCNN

    Returns:
        TemporalCNN instance
    """
    configs = {
        "tiny": {
            "hidden_dims": (16, 32, 64),
            "kernel_sizes": (5, 5, 5),
            "dilations": (1, 2, 4),
            "n_classes": 64,
        },
        "small": {
            "hidden_dims": (32, 64, 128),
            "kernel_sizes": (7, 7, 7),
            "dilations": (1, 2, 4),
            "n_classes": 128,
        },
        "medium": {
            "hidden_dims": (64, 128, 256, 512),
            "kernel_sizes": (7, 7, 7, 7),
            "dilations": (1, 2, 4, 8),
            "n_classes": 256,
        },
        "large": {
            "hidden_dims": (128, 256, 512, 1024),
            "kernel_sizes": (9, 9, 9, 9),
            "dilations": (1, 2, 4, 8),
            "n_classes": 512,
        },
    }

    config = configs.get(model_size, configs["small"])
    config.update(kwargs)

    return TemporalCNN(n_channels=n_channels, **config)
