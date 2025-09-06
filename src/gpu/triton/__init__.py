# -*- coding: utf-8 -*-
# File: src/gpu/triton/__init__.py
"""
Triton GPU kernels for EEG processing.
"""
from .utils import (
    to_contig_float,
    assert_device_cuda,
    launch_pad,
    validate_eeg_shape,
    check_triton_availability
)

from .fir_iir_fused import (
    fused_bandpass_notch_car,
    make_biquad_coeffs,
    fallback_bandpass_notch_car
)

from .rmsnorm import (
    rmsnorm_time,
    RMSNormTime,
    rmsnorm_time_cpu
)

__all__ = [
    # Utils
    'to_contig_float',
    'assert_device_cuda',
    'launch_pad',
    'validate_eeg_shape',
    'check_triton_availability',

    # Filtering
    'fused_bandpass_notch_car',
    'make_biquad_coeffs',
    'fallback_bandpass_notch_car',

    # Normalization
    'rmsnorm_time',
    'RMSNormTime',
    'rmsnorm_time_cpu'
]
