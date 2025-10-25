#!/usr/bin/env python3
"""Shared enhanced EEG components with ROCm-safe kernels."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """Temporal self-attention implemented with standard PyTorch ops."""

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)

    def _shape(self, tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        tensor = tensor.view(batch_size, -1, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, time_steps = x.shape
        x = x.transpose(1, 2)

        q = self._shape(self.q_proj(x), batch_size)
        k = self._shape(self.k_proj(x), batch_size)
        v = self._shape(self.v_proj(x), batch_size)

        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        context = torch.matmul(attention, v)

        context = context.transpose(1, 2).contiguous().view(batch_size, time_steps, self.embed_dim)
        output = self.out_proj(context)
        output = self.norm(output + x)
        return output.transpose(1, 2)


class MultiScaleFeaturesExtractor(nn.Module):
    """Extract temporal features at multiple receptive fields."""

    def __init__(self, in_channels: int, out_channels: int = 32) -> None:
        super().__init__()
        self.fast_conv = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.medium_conv = nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=7)
        self.slow_conv = nn.Conv1d(in_channels, out_channels, kernel_size=31, padding=15)

        self.bn_fast = nn.BatchNorm1d(out_channels)
        self.bn_medium = nn.BatchNorm1d(out_channels)
        self.bn_slow = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fast = F.relu(self.bn_fast(self.fast_conv(x)))
        medium = F.relu(self.bn_medium(self.medium_conv(x)))
        slow = F.relu(self.bn_slow(self.slow_conv(x)))
        return torch.cat([fast, medium, slow], dim=1)


class EnhancedEEGNeX(nn.Module):
    """EEGNeX backbone augmented with multi-scale features and attention."""

    def __init__(self, n_channels: int = 129, n_times: int = 200, n_outputs: int = 1) -> None:
        super().__init__()
        del n_times

        self.multiscale = MultiScaleFeaturesExtractor(n_channels, out_channels=32)
        self.temporal_attention = TemporalAttention(embed_dim=96, num_heads=4, dropout=0.1)
        self.fusion = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(96, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        multiscale_features = self.multiscale(x)
        attended = self.temporal_attention(multiscale_features)
        return self.fusion(attended)


def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Apply mixup augmentation."""

    if alpha <= 0:
        return x, y, y, 1.0

    lam = torch.distributions.beta.Beta(alpha, alpha).sample().item()
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1.0 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


__all__ = [
    "TemporalAttention",
    "MultiScaleFeaturesExtractor",
    "EnhancedEEGNeX",
    "mixup_data",
]
