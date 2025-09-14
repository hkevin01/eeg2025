"""
EEG Transformer Backbone
========================

Main transformer architecture for EEG processing with channel-aware attention.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[: x.size(0), :]


class EEGTransformerLayer(nn.Module):
    """Single transformer layer with EEG-specific modifications."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        # Self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = residual + self.dropout(attn_out)

        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)

        return x


class EEGTransformer(nn.Module):
    """
    EEG Transformer backbone for processing multi-channel EEG signals.

    Args:
        n_channels: Number of EEG channels
        d_model: Model dimension
        n_layers: Number of transformer layers
        n_heads: Number of attention heads
        dropout: Dropout rate
        max_seq_len: Maximum sequence length
    """

    def __init__(
        self,
        n_channels: int,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        dropout: float = 0.1,
        max_seq_len: int = 2048,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.d_model = d_model

        # Channel projection
        self.channel_projection = nn.Linear(n_channels, d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # Transformer layers
        self.layers = nn.ModuleList(
            [EEGTransformerLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using Xavier initialization."""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through EEG transformer.

        Args:
            x: Input tensor of shape [batch_size, n_channels, seq_len]
            mask: Attention mask of shape [seq_len, seq_len]

        Returns:
            Transformer output of shape [batch_size, seq_len, d_model]
        """
        batch_size, n_channels, seq_len = x.shape

        # Transpose to [batch_size, seq_len, n_channels]
        x = x.transpose(1, 2)

        # Project channels to model dimension
        x = self.channel_projection(x)

        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)

        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final normalization
        x = self.norm(x)

        return x

    def get_attention_weights(
        self, x: torch.Tensor, layer_idx: int = -1
    ) -> torch.Tensor:
        """
        Extract attention weights from a specific layer.

        Args:
            x: Input tensor
            layer_idx: Layer index (-1 for last layer)

        Returns:
            Attention weights
        """
        if layer_idx == -1:
            layer_idx = len(self.layers) - 1

        # Forward pass up to the specified layer
        x = x.transpose(1, 2)
        x = self.channel_projection(x)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for i, layer in enumerate(self.layers):
            if i <= layer_idx:
                if i == layer_idx:
                    # Extract attention weights from this layer
                    x_norm = layer.norm1(x)
                    _, attn_weights = layer.self_attn(x_norm, x_norm, x_norm)
                    return attn_weights
                else:
                    x = layer(x)

        return None


class EEGPatchEmbedding(nn.Module):
    """
    Alternative patch-based embedding for EEG signals.
    Divides the signal into patches and embeds them.
    """

    def __init__(
        self, n_channels: int, patch_size: int, d_model: int, overlap: float = 0.5
    ):
        super().__init__()

        self.n_channels = n_channels
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = int(patch_size * (1 - overlap))

        # Patch embedding
        self.patch_embed = nn.Linear(n_channels * patch_size, d_model)

        # Learnable class token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        Convert EEG signal to patch embeddings.

        Args:
            x: Input of shape [batch_size, n_channels, seq_len]

        Returns:
            Patch embeddings of shape [batch_size, n_patches + 1, d_model]
            Number of patches
        """
        batch_size, n_channels, seq_len = x.shape

        # Create patches with overlap
        patches = []
        for i in range(0, seq_len - self.patch_size + 1, self.stride):
            patch = x[:, :, i : i + self.patch_size]  # [batch, channels, patch_size]
            patch = patch.flatten(1)  # [batch, channels * patch_size]
            patches.append(patch)

        if patches:
            patches = torch.stack(
                patches, dim=1
            )  # [batch, n_patches, channels * patch_size]
            n_patches = patches.shape[1]
        else:
            # Handle case where sequence is too short
            patches = x.flatten(1).unsqueeze(1)  # [batch, 1, channels * seq_len]
            n_patches = 1
            # Adjust embedding size if needed
            if patches.shape[-1] != self.n_channels * self.patch_size:
                patches = F.pad(
                    patches, (0, self.n_channels * self.patch_size - patches.shape[-1])
                )

        # Embed patches
        embeddings = self.patch_embed(patches)  # [batch, n_patches, d_model]

        # Add class token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat([cls_tokens, embeddings], dim=1)

        return embeddings, n_patches


def create_eeg_transformer(config: dict) -> EEGTransformer:
    """
    Factory function to create EEG transformer from config.

    Args:
        config: Configuration dictionary

    Returns:
        EEG transformer model
    """
    return EEGTransformer(
        n_channels=config.get("n_channels", 128),
        d_model=config.get("d_model", 768),
        n_layers=config.get("n_layers", 12),
        n_heads=config.get("n_heads", 12),
        dropout=config.get("dropout", 0.1),
        max_seq_len=config.get("max_sequence_length", 2048),
    )
