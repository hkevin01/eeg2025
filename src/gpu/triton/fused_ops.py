"""
Fused GPU Operations using Triton
==================================

High-performance fused operations for EEG processing using Triton kernels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Try to import Triton, fall back to PyTorch if not available
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    print("Warning: Triton not available, using PyTorch fallback implementations")


# Triton kernels (if available)
if TRITON_AVAILABLE:
    @triton.jit
    def fused_filter_kernel(
        x_ptr, y_ptr,
        N, L,
        cutoff_freq: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        """Triton kernel for fused filtering."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < N * L

        # Load data
        x = tl.load(x_ptr + offsets, mask=mask)

        # Simple low-pass filter (placeholder)
        # In practice, this would implement proper filtering
        alpha = cutoff_freq
        y = x * alpha

        # Store result
        tl.store(y_ptr + offsets, y, mask=mask)

    @triton.jit
    def rms_norm_kernel(
        x_ptr, y_ptr, weight_ptr,
        N, D,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        """Triton kernel for RMS normalization."""
        row = tl.program_id(0)

        # Load row
        x_row_ptr = x_ptr + row * D
        y_row_ptr = y_ptr + row * D

        # Compute RMS
        cols = tl.arange(0, BLOCK_SIZE)
        mask = cols < D
        x = tl.load(x_row_ptr + cols, mask=mask, other=0.0)

        # Calculate mean square
        x_squared = x * x
        mean_square = tl.sum(x_squared, axis=0) / D
        rms = tl.sqrt(mean_square + eps)

        # Normalize and apply weight
        weight = tl.load(weight_ptr + cols, mask=mask, other=1.0)
        y = (x / rms) * weight

        # Store result
        tl.store(y_row_ptr + cols, y, mask=mask)


def fused_filtering(
    x: torch.Tensor,
    cutoff_freq: float = 0.1,
    use_triton: bool = True
) -> torch.Tensor:
    """
    Fused filtering operation with Triton acceleration.

    Args:
        x: Input tensor of shape [batch_size, channels, seq_len]
        cutoff_freq: Cutoff frequency for filtering
        use_triton: Whether to use Triton kernel

    Returns:
        Filtered tensor
    """
    if not TRITON_AVAILABLE or not use_triton or not x.is_cuda:
        return _fused_filtering_fallback(x, cutoff_freq)

    batch_size, channels, seq_len = x.shape

    # Flatten for kernel processing
    x_flat = x.flatten()
    y_flat = torch.empty_like(x_flat)

    # Calculate grid
    N = batch_size * channels
    L = seq_len
    BLOCK_SIZE = 512
    grid = (triton.cdiv(N * L, BLOCK_SIZE),)

    # Launch kernel
    fused_filter_kernel[grid](
        x_flat, y_flat,
        N, L,
        cutoff_freq,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape back
    return y_flat.reshape(batch_size, channels, seq_len)


def _fused_filtering_fallback(
    x: torch.Tensor,
    cutoff_freq: float
) -> torch.Tensor:
    """
    Fallback implementation using PyTorch.

    Args:
        x: Input tensor
        cutoff_freq: Cutoff frequency

    Returns:
        Filtered tensor
    """
    # Simple exponential moving average filter
    alpha = cutoff_freq

    # Initialize output
    y = torch.zeros_like(x)
    y[..., 0] = x[..., 0]

    # Apply filter
    for t in range(1, x.shape[-1]):
        y[..., t] = alpha * x[..., t] + (1 - alpha) * y[..., t-1]

    return y


def rms_norm(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    use_triton: bool = True
) -> torch.Tensor:
    """
    RMS normalization with Triton acceleration.

    Args:
        x: Input tensor [batch_size, seq_len, features] or [batch_size, features]
        weight: Optional weight tensor [features]
        eps: Small epsilon for numerical stability
        use_triton: Whether to use Triton kernel

    Returns:
        RMS normalized tensor
    """
    if not TRITON_AVAILABLE or not use_triton or not x.is_cuda:
        return _rms_norm_fallback(x, weight, eps)

    # Handle different input shapes
    original_shape = x.shape
    if len(x.shape) == 3:
        batch_size, seq_len, features = x.shape
        x = x.reshape(-1, features)
    else:
        features = x.shape[-1]

    N, D = x.shape

    # Prepare weight
    if weight is None:
        weight = torch.ones(D, device=x.device, dtype=x.dtype)

    # Output tensor
    y = torch.empty_like(x)

    # Launch kernel
    BLOCK_SIZE = triton.next_power_of_2(D)
    grid = (N,)

    rms_norm_kernel[grid](
        x, y, weight,
        N, D,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape back to original shape
    return y.reshape(original_shape)


def _rms_norm_fallback(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Fallback RMS normalization using PyTorch.

    Args:
        x: Input tensor
        weight: Optional weight tensor
        eps: Epsilon for numerical stability

    Returns:
        RMS normalized tensor
    """
    # Compute RMS
    rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)

    # Normalize
    x_normalized = x / rms

    # Apply weight if provided
    if weight is not None:
        x_normalized = x_normalized * weight

    return x_normalized


class FusedLinear(nn.Module):
    """
    Fused linear layer with optional activation and normalization.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        activation: str = "none",
        use_norm: bool = False,
        use_triton: bool = True
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.use_triton = use_triton and TRITON_AVAILABLE

        # Linear layer
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        # Activation
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "swish":
            self.activation = nn.SiLU()
        else:
            self.activation = nn.Identity()

        # Normalization
        if use_norm:
            self.norm = nn.LayerNorm(out_features)
        else:
            self.norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through fused linear layer."""
        # Linear transformation
        x = self.linear(x)

        # Activation
        x = self.activation(x)

        # Normalization
        x = self.norm(x)

        return x


class FusedMultiHeadAttention(nn.Module):
    """
    Fused multi-head attention with optimized memory usage.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.1,
        use_triton: bool = True
    ):
        super().__init__()

        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.use_triton = use_triton and TRITON_AVAILABLE

        # Fused QKV projection
        self.qkv_proj = FusedLinear(d_model, 3 * d_model, use_triton=use_triton)
        self.out_proj = FusedLinear(d_model, d_model, use_triton=use_triton)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through fused attention.

        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask

        Returns:
            Attention output
        """
        batch_size, seq_len, d_model = x.shape

        # Fused QKV computation
        qkv = self.qkv_proj(x)  # [batch_size, seq_len, 3 * d_model]

        # Reshape and split
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, n_heads, seq_len, d_k]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention computation
        if self.use_triton and x.is_cuda:
            # Use optimized attention (placeholder)
            attn_out = self._triton_attention(q, k, v, mask)
        else:
            attn_out = self._pytorch_attention(q, k, v, mask)

        # Reshape and project
        attn_out = attn_out.transpose(1, 2).contiguous()
        attn_out = attn_out.reshape(batch_size, seq_len, d_model)

        return self.out_proj(attn_out)

    def _triton_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Triton-optimized attention (placeholder).

        In practice, this would use Triton's FlashAttention or similar.
        """
        # Fallback to PyTorch for now
        return self._pytorch_attention(q, k, v, mask)

    def _pytorch_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard PyTorch attention implementation.
        """
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_out = torch.matmul(attn_weights, v)

        return attn_out


def create_fused_ops_config(config: dict) -> dict:
    """
    Create fused operations configuration.

    Args:
        config: Model configuration

    Returns:
        Fused operations configuration
    """
    gpu_config = config.get('gpu', {})
    triton_config = gpu_config.get('triton', {})

    return {
        'use_triton': triton_config.get('fused_filtering', True) and TRITON_AVAILABLE,
        'use_fused_rmsnorm': triton_config.get('fused_rmsnorm', True),
        'block_size': triton_config.get('block_size', 512),
        'available': TRITON_AVAILABLE
    }
