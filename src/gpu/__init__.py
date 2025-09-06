# -*- coding: utf-8 -*-
# File: src/gpu/__init__.py
"""
GPU acceleration module for EEG Foundation Challenge 2025.

This module provides GPU-first implementations using:
- Triton kernels for fused operations
- CuPy for advanced CUDA operations
- Optimized streaming inference
"""

# Check availability of GPU libraries
TRITON_AVAILABLE = False
CUPY_AVAILABLE = False

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    pass

try:
    import cupy
    CUPY_AVAILABLE = True
except ImportError:
    pass

# Import modules with fallbacks
if TRITON_AVAILABLE:
    try:
        from .triton import *
    except ImportError:
        TRITON_AVAILABLE = False

if CUPY_AVAILABLE:
    try:
        from .cupy import *
    except ImportError:
        CUPY_AVAILABLE = False

__all__ = [
    'TRITON_AVAILABLE',
    'CUPY_AVAILABLE'
]

# Add exports from submodules if available
if TRITON_AVAILABLE:
    from .triton import __all__ as triton_exports
    __all__.extend(triton_exports)

if CUPY_AVAILABLE:
    from .cupy import __all__ as cupy_exports  
    __all__.extend(cupy_exports)
