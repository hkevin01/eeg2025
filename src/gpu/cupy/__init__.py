# -*- coding: utf-8 -*-
# File: src/gpu/cupy/__init__.py
"""
CuPy-based GPU acceleration for EEG processing.
"""
from .perceptual_quant import (
    perceptual_quantize,
    adaptive_wavelet_compress,
    predictive_coding_residual,
    perceptual_quantize_torch,
    compression_augmentation_suite,
    perceptual_quantize_cpu_fallback
)

__all__ = [
    'perceptual_quantize',
    'adaptive_wavelet_compress',
    'predictive_coding_residual',
    'perceptual_quantize_torch',
    'compression_augmentation_suite',
    'perceptual_quantize_cpu_fallback'
]
