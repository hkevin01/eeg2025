# -*- coding: utf-8 -*-
# File: src/gpu/triton/utils.py
"""
Utility functions for Triton GPU kernels.
"""
from __future__ import annotations

import torch


def to_contig_float(x: torch.Tensor) -> torch.Tensor:
    """Convert tensor to contiguous float32."""
    if x.dtype != torch.float32:
        x = x.float()
    if not x.is_contiguous():
        x = x.contiguous()
    return x


def assert_device_cuda(x: torch.Tensor):
    """Assert tensor is on CUDA device."""
    if not x.is_cuda:
        raise RuntimeError("Expected CUDA tensor")


def launch_pad(n: int, block: int) -> int:
    """Compute padded grid size for kernel launch."""
    return (n + block - 1) // block * block


def validate_eeg_shape(x: torch.Tensor) -> tuple[int, int, int]:
    """Validate and return EEG tensor dimensions (B, C, T)."""
    if x.dim() != 3:
        raise ValueError(f"Expected 3D tensor (B, C, T), got {x.dim()}D")
    B, C, T = x.shape
    if C <= 0 or T <= 0:
        raise ValueError(f"Invalid EEG dimensions: B={B}, C={C}, T={T}")
    return B, C, T


def check_triton_availability() -> bool:
    """Check if Triton is available and functional."""
    try:
        import triton

        return True
    except ImportError:
        return False
