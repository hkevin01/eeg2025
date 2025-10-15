#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced GPU Optimizer for CUDA and ROCm
========================================

Advanced GPU optimization system that maximizes performance on both
NVIDIA CUDA and AMD ROCm while maintaining stability.

Features:
- Intelligent operation routing per platform
- Advanced memory management
- Performance profiling and optimization
- Automatic batch size tuning
- Platform-specific acceleration
"""

import torch
import torch.nn as nn
import time
import threading
import queue
import warnings
from typing import Optional, Dict, Any, Tuple, List
from contextlib import contextmanager

class GPUProfiler:
    """GPU performance profiler for optimization"""

    def __init__(self):
        self.timing_cache = {}
        self.memory_cache = {}

    def profile_operation(self, operation_name: str, func, *args, **kwargs):
        """Profile operation and cache results"""
        if not torch.cuda.is_available():
            return func(*args, **kwargs)

        # Memory before
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

        # Time operation
        start_time = time.time()
        result = func(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.time()

        # Memory after
        mem_after = torch.cuda.memory_allocated()

        # Cache results
        self.timing_cache[operation_name] = end_time - start_time
        self.memory_cache[operation_name] = mem_after - mem_before

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics"""
        return {
            'timing': self.timing_cache.copy(),
            'memory': self.memory_cache.copy()
        }

class EnhancedGPUOptimizer:
    """
    Enhanced GPU optimizer for both NVIDIA and AMD platforms.

    Provides:
    - Platform-specific optimizations
    - Intelligent operation routing
    - Advanced memory management
    - Performance profiling
    - Automatic fallback strategies
    """

    def __init__(self, device: str = "cuda", enable_profiling: bool = True):
        self.gpu_available = torch.cuda.is_available()
        self.device = torch.device(device if self.gpu_available else "cpu")
        self.enable_profiling = enable_profiling

        # Platform detection
        self._detect_platform()

        # Initialize profiler
        if enable_profiling:
            self.profiler = GPUProfiler()

        # Configure platform-specific settings
        self._configure_platform()

        # Operation routing table
        self._setup_operation_routing()

        print(f"🚀 Enhanced GPU Optimizer initialized")
        print(f"   Platform: {self.platform}")
        print(f"   Device: {self.device}")
        print(f"   Memory: {self.gpu_memory_gb:.1f} GB" if self.gpu_available else "   Memory: N/A")

    def _detect_platform(self):
        """Detect GPU platform and capabilities"""
        if not self.gpu_available:
            self.vendor = "CPU"
            self.platform = "CPU"
            self.is_amd = False
            self.is_nvidia = False
            self.gpu_memory_gb = 0
            return

        # Detect vendor
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            self.vendor = "AMD"
            self.platform = "ROCm/HIP"
            self.is_amd = True
            self.is_nvidia = False
        else:
            self.vendor = "NVIDIA"
            self.platform = "CUDA"
            self.is_amd = False
            self.is_nvidia = True

        # Get device info
        if torch.cuda.device_count() > 0:
            self.device_name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            self.gpu_memory_gb = props.total_memory / (1024**3)
            self.compute_capability = (props.major, props.minor) if self.is_nvidia else None

    def _configure_platform(self):
        """Configure platform-specific optimizations"""
        if not self.gpu_available:
            return

        if self.is_nvidia:
            # NVIDIA optimizations
            try:
                # Enable TF32 for Ampere+ GPUs
                if self.compute_capability and self.compute_capability[0] >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    print("   ✅ TF32 enabled for Ampere GPU")

                # Optimize for specific GPU generations
                if self.compute_capability:
                    major, minor = self.compute_capability
                    if major >= 8:  # Ampere+
                        self.use_tensor_cores = True
                        self.optimal_batch_multiple = 8
                    elif major >= 7:  # Turing/Volta
                        self.use_tensor_cores = True
                        self.optimal_batch_multiple = 8
                    else:  # Older GPUs
                        self.use_tensor_cores = False
                        self.optimal_batch_multiple = 4

            except Exception as e:
                print(f"   ⚠️  NVIDIA optimization setup failed: {e}")

        elif self.is_amd:
            # AMD ROCm optimizations (especially for RX 5600 XT)
            try:
                print("   ✅ AMD ROCm optimizations enabled")
                self.use_tensor_cores = False  # AMD doesn't have tensor cores
                self.optimal_batch_multiple = 4  # Conservative for AMD

                # AMD-specific memory management
                self.conservative_memory = True
                self.frequent_cleanup = True

                # Fix hipBLASLt warning for unsupported architectures (RX 5600 XT)
                # RX 5600 XT is gfx1010 (RDNA 1.0) which doesn't support hipBLASLt
                try:
                    # Force hipBLAS backend instead of hipBLASLt
                    import os
                    os.environ['ROCBLAS_LAYER'] = '1'  # Disable advanced features
                    os.environ['HIPBLASLT_LOG_LEVEL'] = '0'  # Suppress warnings

                    # Disable experimental features for older AMD GPUs
                    if hasattr(torch.backends, 'cuda'):
                        if hasattr(torch.backends.cuda, 'preferred_blas_library'):
                            torch.backends.cuda.preferred_blas_library = 'hipblas'

                    print(f"   ✅ Configured hipBLAS for {self.device_name}")
                except Exception as inner_e:
                    print(f"   ⚠️  hipBLAS configuration note: {inner_e}")

            except Exception as e:
                print(f"   ⚠️  AMD optimization setup failed: {e}")

    def _setup_operation_routing(self):
        """Setup operation routing based on platform"""
        if self.is_amd:
            # AMD routing - conservative for stability
            self.operation_routing = {
                'fft': 'cpu',           # FFT operations on CPU for stability
                'ifft': 'cpu',          # Inverse FFT on CPU for stability
                'matmul': 'gpu',        # Matrix multiplication on GPU
                'conv': 'gpu',          # Convolution on GPU
                'linear': 'gpu',        # Linear layers on GPU
                'transformer': 'gpu',   # Transformer operations on GPU
                'attention': 'gpu',     # Attention on GPU
                'embedding': 'gpu',     # Embeddings on GPU
            }
        elif self.is_nvidia:
            # NVIDIA routing - full GPU utilization
            self.operation_routing = {
                'fft': 'gpu',           # FFT operations on GPU
                'ifft': 'gpu',          # Inverse FFT on GPU
                'matmul': 'gpu',        # Matrix multiplication on GPU
                'conv': 'gpu',          # Convolution on GPU
                'linear': 'gpu',        # Linear layers on GPU
                'transformer': 'gpu',   # Transformer operations on GPU
                'attention': 'gpu',     # Attention on GPU
                'embedding': 'gpu',     # Embeddings on GPU
            }
        else:
            # CPU routing
            self.operation_routing = {op: 'cpu' for op in [
                'fft', 'ifft', 'matmul', 'conv', 'linear',
                'transformer', 'attention', 'embedding'
            ]}

    def get_optimal_device(self, operation: str) -> torch.device:
        """Get optimal device for specific operation"""
        device_type = self.operation_routing.get(operation, 'gpu')
        if device_type == 'gpu' and self.gpu_available:
            return self.device
        else:
            return torch.device('cpu')

    def optimize_tensor_for_operation(self, tensor: torch.Tensor, operation: str) -> torch.Tensor:
        """Move tensor to optimal device for operation"""
        optimal_device = self.get_optimal_device(operation)
        if tensor.device != optimal_device:
            return tensor.to(optimal_device)
        return tensor

    @contextmanager
    def memory_management(self, operation_name: str = "operation"):
        """Context manager for memory management"""
        if self.gpu_available and self.is_amd and hasattr(self, 'frequent_cleanup'):
            # More aggressive cleanup for AMD
            torch.cuda.empty_cache()

        try:
            yield
        finally:
            if self.gpu_available:
                if self.is_amd and hasattr(self, 'frequent_cleanup'):
                    # Aggressive cleanup for AMD stability
                    torch.cuda.empty_cache()
                elif self.is_nvidia:
                    # Less aggressive for NVIDIA
                    if torch.cuda.memory_allocated() > self.gpu_memory_gb * 0.8 * 1024**3:
                        torch.cuda.empty_cache()

    def safe_fft(self, x: torch.Tensor, dim: int = -1, norm: str = "ortho") -> torch.Tensor:
        """Safe FFT with platform optimization"""
        with self.memory_management("fft"):
            x_opt = self.optimize_tensor_for_operation(x, 'fft')

            if self.enable_profiling:
                return self.profiler.profile_operation(
                    "fft", torch.fft.rfft, x_opt, dim=dim, norm=norm
                )
            else:
                return torch.fft.rfft(x_opt, dim=dim, norm=norm)

    def safe_ifft(self, X: torch.Tensor, n: Optional[int] = None,
                  dim: int = -1, norm: str = "ortho") -> torch.Tensor:
        """Safe inverse FFT with platform optimization"""
        with self.memory_management("ifft"):
            X_opt = self.optimize_tensor_for_operation(X, 'ifft')

            if self.enable_profiling:
                return self.profiler.profile_operation(
                    "ifft", torch.fft.irfft, X_opt, n=n, dim=dim, norm=norm
                )
            else:
                return torch.fft.irfft(X_opt, n=n, dim=dim, norm=norm)

    def optimized_matmul(self, a: torch.Tensor, b: torch.Tensor,
                        transpose_a: bool = False, transpose_b: bool = False) -> torch.Tensor:
        """Optimized matrix multiplication"""
        with self.memory_management("matmul"):
            # Move to optimal devices
            a_opt = self.optimize_tensor_for_operation(a, 'matmul')
            b_opt = self.optimize_tensor_for_operation(b, 'matmul')

            # Apply transposes if needed
            if transpose_a:
                a_opt = a_opt.transpose(-2, -1)
            if transpose_b:
                b_opt = b_opt.transpose(-2, -1)

            if self.enable_profiling:
                return self.profiler.profile_operation(
                    "matmul", torch.matmul, a_opt, b_opt
                )
            else:
                return torch.matmul(a_opt, b_opt)

    def optimize_batch_size(self, base_batch_size: int) -> int:
        """Optimize batch size for platform"""
        if not self.gpu_available:
            return base_batch_size

        # Align to optimal multiples
        optimal_multiple = getattr(self, 'optimal_batch_multiple', 4)
        optimized = ((base_batch_size + optimal_multiple - 1) // optimal_multiple) * optimal_multiple

        # Memory constraints
        if self.is_amd:
            # Conservative for AMD
            max_batch = min(32, int(self.gpu_memory_gb * 4))
        else:
            # More aggressive for NVIDIA
            max_batch = min(64, int(self.gpu_memory_gb * 8))

        return min(optimized, max_batch, base_batch_size * 2)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = {
            'platform': self.platform,
            'vendor': self.vendor,
            'device': str(self.device),
            'gpu_memory_gb': getattr(self, 'gpu_memory_gb', 0),
        }

        if self.enable_profiling and hasattr(self, 'profiler'):
            stats.update(self.profiler.get_stats())

        return stats

    def benchmark_operations(self, input_shapes: List[Tuple[int, ...]] = None) -> Dict[str, float]:
        """Benchmark key operations"""
        if input_shapes is None:
            input_shapes = [(32, 128, 1000), (16, 256, 500), (8, 512, 250)]

        benchmark_results = {}

        for shape in input_shapes:
            print(f"Benchmarking shape {shape}...")

            # Create test data
            x = torch.randn(*shape)

            # Benchmark FFT
            try:
                start = time.time()
                for _ in range(10):
                    _ = self.safe_fft(x, dim=-1)
                    if self.gpu_available:
                        torch.cuda.synchronize()
                fft_time = (time.time() - start) / 10
                benchmark_results[f'fft_{shape}'] = fft_time
            except Exception as e:
                print(f"FFT benchmark failed for {shape}: {e}")

            # Benchmark matrix multiplication
            try:
                a = torch.randn(shape[0], shape[1], 64)
                b = torch.randn(shape[0], 64, shape[1])

                start = time.time()
                for _ in range(10):
                    _ = self.optimized_matmul(a, b)
                    if self.gpu_available:
                        torch.cuda.synchronize()
                matmul_time = (time.time() - start) / 10
                benchmark_results[f'matmul_{shape}'] = matmul_time
            except Exception as e:
                print(f"Matmul benchmark failed for {shape}: {e}")

        return benchmark_results

# Global enhanced optimizer instance
_enhanced_gpu = EnhancedGPUOptimizer(enable_profiling=True)

# Convenience functions
def enhanced_fft(x, dim=-1, norm="ortho"):
    """Enhanced FFT with platform optimization"""
    return _enhanced_gpu.safe_fft(x, dim=dim, norm=norm)

def enhanced_ifft(X, n=None, dim=-1, norm="ortho"):
    """Enhanced inverse FFT with platform optimization"""
    return _enhanced_gpu.safe_ifft(X, n=n, dim=dim, norm=norm)

def enhanced_matmul(a, b, transpose_a=False, transpose_b=False):
    """Enhanced matrix multiplication with platform optimization"""
    return _enhanced_gpu.optimized_matmul(a, b, transpose_a=transpose_a, transpose_b=transpose_b)

def get_enhanced_optimizer():
    """Get the global enhanced optimizer instance"""
    return _enhanced_gpu
