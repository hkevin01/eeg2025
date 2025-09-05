"""
Enhanced backbone architectures for EEG Foundation Challenge 2025.

This module provides improved backbone architectures including ConformerTiny,
Squeeze-and-Excitation blocks, and domain adaptation components.
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

logger = logging.getLogger(__name__)


class SqueezeExcitation1D(nn.Module):
    """1D Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.se(x)


class DepthwiseSeparableConv1d(nn.Module):
    """Depthwise separable convolution with squeeze-excitation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        use_se: bool = True,
        se_reduction: int = 16
    ):
        super().__init__()

        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=bias
        )

        # Pointwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=bias)

        # Squeeze-and-Excitation
        self.use_se = use_se
        if use_se:
            self.se = SqueezeExcitation1D(out_channels, se_reduction)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        if self.use_se:
            x = self.se(x)
        return x


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding for variable-length sequences."""

    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        # Cache for efficiency
        self.max_seq_len = max_seq_len
        self._cached_freqs = None
        self._cached_cos = None
        self._cached_sin = None

    def _update_cache(self, seq_len: int, device: torch.device):
        if (self._cached_freqs is None or
            seq_len > self._cached_freqs.shape[0] or
            self._cached_freqs.device != device):

            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            freqs = torch.cat([freqs, freqs], dim=-1)

            self._cached_freqs = freqs
            self._cached_cos = freqs.cos()
            self._cached_sin = freqs.sin()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[-2]
        self._update_cache(seq_len, x.device)

        cos = self._cached_cos[:seq_len]
        sin = self._cached_sin[:seq_len]

        return cos, sin


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary positional embedding to input tensor."""
    x1, x2 = x[..., ::2], x[..., 1::2]
    return torch.cat([
        x1 * cos - x2 * sin,
        x2 * cos + x1 * sin
    ], dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with rotary positional embedding."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_rotary: bool = True
    ):
        super().__init__()
        assert dim % num_heads == 0

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

        self.use_rotary = use_rotary
        if use_rotary:
            self.rotary = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply rotary positional embedding
        if self.use_rotary:
            cos, sin = self.rotary(x)
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ConformerBlock(nn.Module):
    """Conformer block combining convolution and attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        ff_mult: int = 4,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        use_rotary: bool = True
    ):
        super().__init__()

        # First feed-forward
        self.ff1 = FeedForward(dim, dim * ff_mult, dropout)
        self.norm1 = nn.LayerNorm(dim)

        # Multi-head attention
        self.attn = MultiHeadAttention(dim, num_heads, dropout, use_rotary)
        self.norm2 = nn.LayerNorm(dim)

        # Convolution module
        self.conv = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, dim * 2, 1),
            nn.GLU(dim=1),
            nn.Conv1d(dim, dim, conv_kernel_size, padding=conv_kernel_size // 2, groups=dim),
            nn.BatchNorm1d(dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
            nn.Dropout(dropout),
            Rearrange('b c n -> b n c')
        )

        # Second feed-forward
        self.ff2 = FeedForward(dim, dim * ff_mult, dropout)
        self.norm3 = nn.LayerNorm(dim)

        # Layer scale
        self.layer_scale = nn.Parameter(torch.ones(dim) * 1e-4)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Half-step feed-forward
        x = x + 0.5 * self.ff1(self.norm1(x))

        # Multi-head attention
        x = x + self.attn(self.norm2(x), mask)

        # Convolution
        x = x + self.conv(x)

        # Second half-step feed-forward
        x = x + 0.5 * self.ff2(self.norm3(x))

        return x * self.layer_scale


class ConformerTiny(nn.Module):
    """Tiny Conformer for efficient EEG processing."""

    def __init__(
        self,
        in_channels: int,
        dim: int = 192,
        depth: int = 6,
        num_heads: int = 8,
        conv_kernel_size: int = 31,
        dropout: float = 0.1,
        max_seq_len: int = 1000
    ):
        super().__init__()

        self.dim = dim

        # Channel embedding
        self.channel_embed = nn.Linear(in_channels, dim)

        # Conformer blocks
        self.blocks = nn.ModuleList([
            ConformerBlock(
                dim=dim,
                num_heads=num_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout
            )
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (batch, channels, time)
        x = rearrange(x, 'b c t -> b t c')  # (batch, time, channels)

        # Channel embedding
        x = self.channel_embed(x)  # (batch, time, dim)

        # Conformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)

        # Global pooling
        x = rearrange(x, 'b t c -> b c t')
        x = self.pool(x).squeeze(-1)  # (batch, dim)

        return x


class GradientReversalLayer(torch.autograd.Function):
    """Gradient reversal layer for domain adaptation."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.alpha, None


class DomainAdversarialNetwork(nn.Module):
    """Domain adversarial network for domain adaptation."""

    def __init__(
        self,
        feature_dim: int,
        num_domains: int,
        hidden_dim: int = 256,
        dropout: float = 0.5
    ):
        super().__init__()

        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_domains)
        )

    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        reversed_features = GradientReversalLayer.apply(features, alpha)
        return self.domain_classifier(reversed_features)


class EnhancedTemporalCNN(nn.Module):
    """Enhanced Temporal CNN with domain adaptation and attention."""

    def __init__(
        self,
        in_channels: int = 128,
        num_channels: List[int] = [64, 128, 256, 512],
        kernel_size: int = 7,
        dropout: float = 0.3,
        use_se: bool = True,
        use_conformer: bool = True,
        conformer_dim: int = 256,
        conformer_depth: int = 4,
        num_domains: int = 1000,
        enable_domain_adaptation: bool = True
    ):
        """
        Initialize Enhanced Temporal CNN.

        Args:
            in_channels: Number of input EEG channels
            num_channels: List of channel dimensions for each layer
            kernel_size: Convolution kernel size
            dropout: Dropout probability
            use_se: Whether to use Squeeze-and-Excitation
            use_conformer: Whether to add Conformer blocks
            conformer_dim: Conformer dimension
            conformer_depth: Number of Conformer blocks
            num_domains: Number of domains for adaptation
            enable_domain_adaptation: Whether to enable domain adaptation
        """
        super().__init__()

        self.in_channels = in_channels
        self.enable_domain_adaptation = enable_domain_adaptation

        # Input normalization
        self.input_norm = nn.BatchNorm1d(in_channels)

        # Temporal convolutional layers
        layers = []
        prev_channels = in_channels

        for i, out_channels in enumerate(num_channels):
            # Depthwise separable convolution with SE
            layers.extend([
                DepthwiseSeparableConv1d(
                    prev_channels, out_channels, kernel_size,
                    padding=kernel_size // 2, use_se=use_se
                ),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout1d(dropout),
                nn.MaxPool1d(2)
            ])
            prev_channels = out_channels

        self.temporal_layers = nn.Sequential(*layers)

        # Optional Conformer blocks
        self.use_conformer = use_conformer
        if use_conformer:
            self.conformer = ConformerTiny(
                in_channels=prev_channels,
                dim=conformer_dim,
                depth=conformer_depth,
                dropout=dropout
            )
            self.feature_dim = conformer_dim
        else:
            self.feature_dim = prev_channels
            self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Domain adaptation
        if enable_domain_adaptation:
            self.domain_classifier = DomainAdversarialNetwork(
                self.feature_dim, num_domains, dropout=dropout
            )

    def forward(
        self,
        x: torch.Tensor,
        alpha: float = 1.0,
        return_domain_logits: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input EEG data [batch, channels, time]
            alpha: Gradient reversal strength
            return_domain_logits: Whether to return domain predictions

        Returns:
            Dictionary with features and optional domain logits
        """
        # Input normalization
        x = self.input_norm(x)

        # Temporal convolution
        x = self.temporal_layers(x)

        # Conformer or global pooling
        if self.use_conformer:
            features = self.conformer(x)
        else:
            features = self.global_pool(x).squeeze(-1)

        outputs = {"features": features}

        # Domain adaptation
        if self.enable_domain_adaptation and return_domain_logits:
            domain_logits = self.domain_classifier(features, alpha)
            outputs["domain_logits"] = domain_logits

        return outputs


class ChannelAttention(nn.Module):
    """Channel attention for handling variable channel configurations."""

    def __init__(self, num_channels: int, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.Sequential(
            nn.Linear(num_channels, num_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_channels // 4, num_channels),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        # Global average pooling across time
        pooled = x.mean(dim=-1)  # (batch, channels)

        # Attention weights
        weights = self.attention(pooled)  # (batch, channels)

        # Apply attention
        return x * weights.unsqueeze(-1)


class RobustEEGBackbone(nn.Module):
    """Robust EEG backbone with channel masking and attention."""

    def __init__(
        self,
        in_channels: int = 128,
        num_channels: List[int] = [64, 128, 256],
        kernel_size: int = 7,
        dropout: float = 0.3,
        channel_dropout: float = 0.1,
        use_channel_attention: bool = True,
        **kwargs
    ):
        super().__init__()

        self.in_channels = in_channels
        self.channel_dropout = channel_dropout
        self.use_channel_attention = use_channel_attention

        # Channel attention
        if use_channel_attention:
            self.channel_attn = ChannelAttention(in_channels, dropout)

        # Enhanced temporal CNN
        self.backbone = EnhancedTemporalCNN(
            in_channels=in_channels,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            **kwargs
        )

    def _apply_channel_dropout(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random channel dropout during training."""
        if self.training and self.channel_dropout > 0:
            # Random channel mask
            batch_size, channels, time = x.shape
            keep_prob = 1 - self.channel_dropout
            mask = torch.bernoulli(torch.full((batch_size, channels, 1), keep_prob)).to(x.device)
            x = x * mask

        return x

    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """Forward pass with robust channel handling."""
        # Channel dropout for robustness
        x = self._apply_channel_dropout(x)

        # Channel attention
        if self.use_channel_attention:
            x = self.channel_attn(x)

        # Backbone processing
        return self.backbone(x, **kwargs)
