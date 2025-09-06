# -*- coding: utf-8 -*-
# File: src/gpu/triton/rmsnorm.py
"""
RMSNorm implementation using Triton for EEG data normalization.
Per-channel normalization over time dimension.
"""
from __future__ import annotations
import triton
import triton.language as tl
import torch
from typing import Optional
from .utils import to_contig_float, assert_device_cuda, validate_eeg_shape

@triton.jit
def rmsnorm_kernel(
    x_ptr, y_ptr, w_ptr, b_ptr,
    B: tl.constexpr, 
    C: tl.constexpr, 
    T: tl.constexpr,
    stride_b: tl.constexpr, 
    stride_c: tl.constexpr, 
    stride_t: tl.constexpr,
    eps: tl.constexpr,
    has_weight: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """
    RMSNorm kernel: per-channel normalization over time dimension.
    
    Computes: y = (x / sqrt(mean(x^2) + eps)) * weight + bias
    where mean is computed over the time dimension for each (batch, channel).
    """
    b_idx = tl.program_id(0)  # batch index
    c_idx = tl.program_id(1)  # channel index

    if c_idx >= C:
        return

    # Compute base pointer for this (batch, channel)
    base = b_idx * stride_b + c_idx * stride_c
    
    # First pass: compute mean of squares over time dimension
    sum_sq = 0.0
    count = 0.0

    for t0 in range(0, T, BLOCK_T):
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T
        
        # Load values for this time block
        x_ptrs = x_ptr + base + t_offsets * stride_t
        x_vals = tl.load(x_ptrs, mask=t_mask, other=0.0)
        
        # Accumulate sum of squares
        sum_sq += tl.sum(x_vals * x_vals, axis=0)
        count += tl.sum(t_mask.to(tl.float32), axis=0)

    # Compute RMS
    mean_sq = sum_sq / tl.maximum(count, 1.0)
    rms = tl.sqrt(mean_sq + eps)
    
    # Load weight and bias if provided
    if has_weight:
        w = tl.load(w_ptr + c_idx)
    else:
        w = 1.0
        
    if has_bias:
        b = tl.load(b_ptr + c_idx)
    else:
        b = 0.0

    # Second pass: normalize and apply affine transformation
    for t0 in range(0, T, BLOCK_T):
        t_offsets = t0 + tl.arange(0, BLOCK_T)
        t_mask = t_offsets < T
        
        # Load input values
        x_ptrs = x_ptr + base + t_offsets * stride_t
        x_vals = tl.load(x_ptrs, mask=t_mask, other=0.0)
        
        # Normalize and transform
        y_vals = (x_vals / rms) * w + b
        
        # Store output
        y_ptrs = y_ptr + base + t_offsets * stride_t
        tl.store(y_ptrs, y_vals, mask=t_mask)


def rmsnorm_time(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
    block_t: int = 256,
) -> torch.Tensor:
    """
    Per-channel RMSNorm over time dimension for EEG data.
    
    Args:
        x: (B, C, T) float32 CUDA tensor
        weight: (C,) optional weight parameter for affine transformation
        bias: (C,) optional bias parameter for affine transformation
        eps: Small constant for numerical stability
        block_t: Block size for time dimension processing
        
    Returns:
        y: (B, C, T) normalized tensor
        
    Note:
        Computes RMS over time dimension for each (batch, channel) pair.
        This is different from standard LayerNorm which normalizes over features.
        For EEG, this provides temporal normalization per electrode.
    """
    assert_device_cuda(x)
    x = to_contig_float(x)
    B, C, T = validate_eeg_shape(x)
    
    # Create output tensor
    y = torch.empty_like(x)
    
    # Handle weight and bias tensors
    has_weight = weight is not None and weight.is_cuda
    has_bias = bias is not None and bias.is_cuda
    
    # Create dummy pointers for unused parameters
    w_ptr = weight if has_weight else torch.zeros(1, device=x.device, dtype=torch.float32)
    b_ptr = bias if has_bias else torch.zeros(1, device=x.device, dtype=torch.float32)
    
    # Ensure parameters are contiguous and correct dtype
    if has_weight:
        w_ptr = w_ptr.contiguous().float()
        if w_ptr.shape != (C,):
            raise ValueError(f"Weight shape {w_ptr.shape} doesn't match channels {C}")
    
    if has_bias:
        b_ptr = b_ptr.contiguous().float()
        if b_ptr.shape != (C,):
            raise ValueError(f"Bias shape {b_ptr.shape} doesn't match channels {C}")

    # Launch kernel
    grid = (B, C)
    rmsnorm_kernel[grid](
        x, y, w_ptr, b_ptr,
        B, C, T,
        x.stride(0), x.stride(1), x.stride(2),
        eps,
        has_weight,
        has_bias,
        BLOCK_T=block_t,
        num_warps=4,
        num_stages=2,
    )
    
    return y


class RMSNormTime(torch.nn.Module):
    """
    PyTorch module wrapper for RMSNorm over time dimension.
    
    Useful for EEG temporal normalization where each electrode
    should be normalized independently over time.
    """
    
    def __init__(
        self, 
        num_channels: int, 
        eps: float = 1e-5, 
        elementwise_affine: bool = True,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ):
        """
        Initialize RMSNorm module.
        
        Args:
            num_channels: Number of EEG channels
            eps: Small constant for numerical stability
            elementwise_affine: If True, add learnable weight and bias
            device: Device for parameters
            dtype: Data type for parameters
        """
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if elementwise_affine:
            self.weight = torch.nn.Parameter(
                torch.ones(num_channels, device=device, dtype=dtype)
            )
            self.bias = torch.nn.Parameter(
                torch.zeros(num_channels, device=device, dtype=dtype)
            )
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply RMSNorm to input tensor.
        
        Args:
            x: (B, C, T) input tensor
            
        Returns:
            y: (B, C, T) normalized tensor
        """
        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"Input channels {x.shape[1]} doesn't match "
                f"module channels {self.num_channels}"
            )
        
        return rmsnorm_time(
            x, 
            weight=self.weight, 
            bias=self.bias, 
            eps=self.eps
        )
    
    def extra_repr(self) -> str:
        return f'num_channels={self.num_channels}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


# CPU fallback implementation
def rmsnorm_time_cpu(
    x: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    CPU fallback implementation of RMSNorm over time dimension.
    """
    # Move to CPU if needed
    x_cpu = x.cpu()
    B, C, T = x_cpu.shape
    
    # Compute RMS over time dimension (dim=2)
    x_sq_mean = torch.mean(x_cpu ** 2, dim=2, keepdim=True)  # (B, C, 1)
    rms = torch.sqrt(x_sq_mean + eps)
    
    # Normalize
    y = x_cpu / rms
    
    # Apply affine transformation if provided
    if weight is not None:
        weight_cpu = weight.cpu().view(1, -1, 1)  # (1, C, 1)
        y = y * weight_cpu
    
    if bias is not None:
        bias_cpu = bias.cpu().view(1, -1, 1)  # (1, C, 1)
        y = y + bias_cpu
    
    return y.to(x.device, dtype=x.dtype)
