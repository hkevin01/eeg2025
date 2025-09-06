# -*- coding: utf-8 -*-
# File: src/gpu/triton/fir_iir_fused.py
"""
Fused bandpass + notch + CAR filtering using Triton kernels.
Implements efficient IIR biquad cascades with common average reference.
"""
from __future__ import annotations
import triton
import triton.language as tl
import torch
from typing import Optional, Tuple
from .utils import to_contig_float, assert_device_cuda, validate_eeg_shape

@triton.jit
def biquad_step(xn, x1, x2, y1, y2, b0, b1, b2, a1, a2):
    """Single biquad filter step using Direct Form I."""
    y = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
    return y

@triton.jit
def fused_filter_kernel(
    x_ptr,        # *f32, input B*C*T
    y_ptr,        # *f32, output B*C*T
    car_tmp_ptr,  # *f32, workspace for CAR mean per (B, t_block)
    B: tl.constexpr,
    C: tl.constexpr,
    T: tl.constexpr,
    stride_b: tl.constexpr,
    stride_c: tl.constexpr,
    stride_t: tl.constexpr,
    # Bandpass biquad 1
    bp1_b0, bp1_b1, bp1_b2, bp1_a1, bp1_a2,
    # Bandpass biquad 2
    bp2_b0, bp2_b1, bp2_b2, bp2_a1, bp2_a2,
    # Notch biquad
    nt_b0, nt_b1, nt_b2, nt_a1, nt_a2,
    # block sizes
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """Fused filtering kernel: bandpass cascade + notch + CAR."""
    b_idx = tl.program_id(0)  # batch id
    t_block = tl.program_id(1)  # time block id

    # Channels vectorized
    c_offsets = tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    # Time range of this block
    t_start = t_block * BLOCK_T
    t_offsets = t_start + tl.arange(0, BLOCK_T)
    t_mask = t_offsets < T

    # pointers for this batch
    base_bc = b_idx * stride_b

    # Initialize biquad state per channel (for demo - resets per block)
    # For streaming: maintain persistent state arrays
    x1_bp1 = tl.zeros([BLOCK_C], dtype=tl.float32)
    x2_bp1 = tl.zeros([BLOCK_C], dtype=tl.float32)
    y1_bp1 = tl.zeros([BLOCK_C], dtype=tl.float32)
    y2_bp1 = tl.zeros([BLOCK_C], dtype=tl.float32)

    x1_bp2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    x2_bp2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    y1_bp2 = tl.zeros([BLOCK_C], dtype=tl.float32)
    y2_bp2 = tl.zeros([BLOCK_C], dtype=tl.float32)

    x1_nt = tl.zeros([BLOCK_C], dtype=tl.float32)
    x2_nt = tl.zeros([BLOCK_C], dtype=tl.float32)
    y1_nt = tl.zeros([BLOCK_C], dtype=tl.float32)
    y2_nt = tl.zeros([BLOCK_C], dtype=tl.float32)

    # Output buffer for this block (C, BLOCK_T)
    y_block = tl.zeros([BLOCK_C, BLOCK_T], dtype=tl.float32)

    # Iterate time inside the block
    for ti in range(0, BLOCK_T):
        t = t_start + ti
        if t >= T:
            break

        # Load x[:, t]
        x_ptr_t = x_ptr + base_bc + c_offsets * stride_c + t * stride_t
        xn = tl.load(x_ptr_t, mask=c_mask, other=0.0)

        # Bandpass biquad 1
        y_bp1 = biquad_step(xn, x1_bp1, x2_bp1, y1_bp1, y2_bp1,
                           bp1_b0, bp1_b1, bp1_b2, bp1_a1, bp1_a2)
        # Update states for bp1
        x2_bp1 = x1_bp1
        x1_bp1 = xn
        y2_bp1 = y1_bp1
        y1_bp1 = y_bp1

        # Bandpass biquad 2 (use bp1 output as input)
        y_bp2 = biquad_step(y_bp1, x1_bp2, x2_bp2, y1_bp2, y2_bp2,
                           bp2_b0, bp2_b1, bp2_b2, bp2_a1, bp2_a2)
        # Update states for bp2
        x2_bp2 = x1_bp2
        x1_bp2 = y_bp1
        y2_bp2 = y1_bp2
        y1_bp2 = y_bp2

        # Notch filter (use bp2 output as input)
        y_nt = biquad_step(y_bp2, x1_nt, x2_nt, y1_nt, y2_nt,
                          nt_b0, nt_b1, nt_b2, nt_a1, nt_a2)
        # Update states for notch
        x2_nt = x1_nt
        x1_nt = y_bp2
        y2_nt = y1_nt
        y1_nt = y_nt

        # Store to block buffer
        y_block[:, ti] = y_nt

    # Compute CAR per time step across channels within this block
    for ti in range(0, BLOCK_T):
        t = t_start + ti
        if t >= T:
            break
        vals = y_block[:, ti]
        # Compute mean over valid channels
        sum_val = tl.sum(vals, axis=0)
        cnt = tl.sum(c_mask.to(tl.float32), axis=0)
        mean_val = sum_val / tl.maximum(cnt, 1.0)
        # Store CAR mean for this (b, t)
        tl.store(car_tmp_ptr + b_idx * T + t, mean_val)

    # Subtract CAR mean and write output
    for ti in range(0, BLOCK_T):
        t = t_start + ti
        if t >= T:
            break
        mean_val = tl.load(car_tmp_ptr + b_idx * T + t)
        out_vals = y_block[:, ti] - mean_val
        y_ptr_t = y_ptr + base_bc + c_offsets * stride_c + t * stride_t
        tl.store(y_ptr_t, out_vals, mask=c_mask)


def fused_bandpass_notch_car(
    x: torch.Tensor,
    biquad_bp1: torch.Tensor,
    biquad_bp2: torch.Tensor,
    biquad_notch: torch.Tensor,
    block_t: int = 256,
    block_c: int = 128,
    workspace: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    GPU-fused bandpass + notch + CAR filtering.

    Args:
        x: (B, C, T) float32 CUDA tensor
        biquad_bp1: (5,) tensor [b0,b1,b2,a1,a2] for first bandpass stage
        biquad_bp2: (5,) tensor [b0,b1,b2,a1,a2] for second bandpass stage
        biquad_notch: (5,) tensor [b0,b1,b2,a1,a2] for notch filter
        block_t: Time block size for parallelization
        block_c: Channel block size for parallelization
        workspace: Optional workspace tensor for CAR computation

    Returns:
        y: (B, C, T) filtered tensor with CAR applied

    Note:
        For streaming applications, maintain biquad states across calls.
        Current implementation resets states per block for simplicity.
    """
    assert_device_cuda(x)
    x = to_contig_float(x)
    B, C, T = validate_eeg_shape(x)

    # Create output tensor
    y = torch.empty_like(x)

    # Create workspace for CAR computation if not provided
    if workspace is None or workspace.numel() < B * T:
        car_tmp = torch.empty(B * T, device=x.device, dtype=torch.float32)
    else:
        car_tmp = workspace

    # Ensure biquad coefficients are on correct device
    biquad_bp1 = biquad_bp1.to(x.device, dtype=torch.float32)
    biquad_bp2 = biquad_bp2.to(x.device, dtype=torch.float32)
    biquad_notch = biquad_notch.to(x.device, dtype=torch.float32)

    # Launch kernel
    grid = (B, triton.cdiv(T, block_t))
    fused_filter_kernel[grid](
        x, y, car_tmp,
        B, C, T,
        x.stride(0), x.stride(1), x.stride(2),
        biquad_bp1[0], biquad_bp1[1], biquad_bp1[2], biquad_bp1[3], biquad_bp1[4],
        biquad_bp2[0], biquad_bp2[1], biquad_bp2[2], biquad_bp2[3], biquad_bp2[4],
        biquad_notch[0], biquad_notch[1], biquad_notch[2], biquad_notch[3], biquad_notch[4],
        BLOCK_T=block_t,
        BLOCK_C=min(block_c, C),
        num_warps=4,
        num_stages=2,
    )
    return y


def make_biquad_coeffs(
    sfreq: float,
    bp_lo: float = 0.1,
    bp_hi: float = 40.0,
    notch: float = 60.0,
    Q: float = 30.0,
    device: str = "cuda"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate biquad coefficients for bandpass and notch filters.

    Args:
        sfreq: Sampling frequency in Hz
        bp_lo: Bandpass low cutoff in Hz
        bp_hi: Bandpass high cutoff in Hz
        notch: Notch frequency in Hz
        Q: Quality factor for notch filter
        device: Target device for tensors

    Returns:
        Tuple of (bp1_coeffs, bp2_coeffs, notch_coeffs) each as (5,) tensors
    """
    try:
        import numpy as np
        from scipy.signal import butter, iirnotch, sos2tf
    except ImportError:
        raise ImportError("scipy required for filter coefficient generation")

    # Bandpass filter (4th order = 2 biquads)
    sos_bp = butter(4, [bp_lo/(sfreq/2), bp_hi/(sfreq/2)],
                    btype="bandpass", output="sos")
    b1, a1 = sos2tf(sos_bp[0:1, :])
    b2, a2 = sos2tf(sos_bp[1:2, :])

    # Notch filter
    b_nt, a_nt = iirnotch(notch/(sfreq/2), Q=Q)

    def pack_coeffs(b, a):
        """Pack filter coefficients as [b0,b1,b2,a1,a2] (normalized)."""
        b = b / a[0]
        a = a / a[0]
        return torch.tensor([b[0], b[1], b[2], a[1], a[2]],
                          dtype=torch.float32, device=device)

    bp1 = pack_coeffs(b1, a1)
    bp2 = pack_coeffs(b2, a2)
    nt = pack_coeffs(b_nt, a_nt)

    return bp1, bp2, nt


# CPU fallback implementation
def fallback_bandpass_notch_car(
    x: torch.Tensor,
    sfreq: float,
    bp_lo: float = 0.1,
    bp_hi: float = 40.0,
    notch: float = 60.0,
    Q: float = 30.0
) -> torch.Tensor:
    """
    CPU fallback for filtering when GPU kernels unavailable.
    Uses scipy for filtering operations.
    """
    try:
        import numpy as np
        from scipy.signal import sosfilt, butter, iirnotch
    except ImportError:
        raise ImportError("scipy required for CPU fallback filtering")

    # Convert to numpy
    x_np = x.cpu().numpy()
    B, C, T = x_np.shape

    # Create filters
    sos_bp = butter(4, [bp_lo/(sfreq/2), bp_hi/(sfreq/2)],
                    btype="bandpass", output="sos")
    sos_notch = iirnotch(notch/(sfreq/2), Q=Q, output="sos")

    # Apply filters
    y_np = np.empty_like(x_np)
    for b in range(B):
        for c in range(C):
            # Bandpass
            filtered = sosfilt(sos_bp, x_np[b, c, :])
            # Notch
            filtered = sosfilt(sos_notch, filtered)
            y_np[b, c, :] = filtered

        # Apply CAR
        car_mean = np.mean(y_np[b, :, :], axis=0, keepdims=True)
        y_np[b, :, :] -= car_mean

    return torch.from_numpy(y_np).to(x.device, dtype=x.dtype)
