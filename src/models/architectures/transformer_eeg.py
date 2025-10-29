"""Transformer-based EEG regressor inspired by recent EEG foundation models.

LUNA (Ingolfsson et al., 2025) and related work highlight the benefit of
patch-wise temporal attention with topology-agnostic embeddings. This module
implements a lightweight variant suitable for end-to-end training on the
Challenge 1 regression target.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len].to(x.dtype)


class EEGTransformerRegressor(nn.Module):
    """Temporal transformer that operates on EEG patches."""

    def __init__(
        self,
        n_channels: int = 129,
        n_times: int = 200,
        patch_size: int = 10,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
    ) -> None:
        super().__init__()

        if n_times % patch_size != 0:
            raise ValueError("n_times must be divisible by patch_size")

        self.patch_size = patch_size
        self.seq_len = n_times // patch_size

        self.patch_embed = nn.Conv1d(
            in_channels=n_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False,
        )
        self.norm_in = nn.LayerNorm(embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=self.seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            layer_norm_eps=layer_norm_eps,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, channels, time)
        patches = self.patch_embed(x)  # (batch, embed_dim, seq_len)
        patches = patches.transpose(1, 2)  # (batch, seq_len, embed_dim)
        tokens = self.norm_in(patches)
        tokens = self.pos_encoder(tokens)
        encoded = self.transformer(tokens)
        pooled = encoded.mean(dim=1)
        return self.head(pooled)


__all__ = ["EEGTransformerRegressor"]