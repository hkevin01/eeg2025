# -*- coding: utf-8 -*-
# File: src/gpu/triton/__init__.py
"""
Triton GPU kernels for EEG processing.
"""
from .fir_iir_fused import (
    fallback_bandpass_notch_car,
    fused_bandpass_notch_car,
    make_biquad_coeffs,
)
from .rmsnorm import RMSNormTime, rmsnorm_time, rmsnorm_time_cpu
from .utils import (
    assert_device_cuda,
    check_triton_availability,
    launch_pad,
    to_contig_float,
    validate_eeg_shape,
)

__all__ = [
    # Utils
    "to_contig_float",
    "assert_device_cuda",
    "launch_pad",
    "validate_eeg_shape",
    "check_triton_availability",
    # Filtering
    "fused_bandpass_notch_car",
    "make_biquad_coeffs",
    "fallback_bandpass_notch_car",
    # Normalization
    "rmsnorm_time",
    "RMSNormTime",
    "rmsnorm_time_cpu",
]
