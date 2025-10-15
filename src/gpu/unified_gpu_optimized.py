#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified GPU Optimized Operations for CUDA and ROCm
==================================================

Automatically detects and optimizes for:
- NVIDIA GPUs: CuFFT + CuBLAS (CUDA backend)
- AMD GPUs: hipFFT + rocBLAS (ROCm backend)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import warnings

class GPUPlatformDetector:
    """Detect and configure GPU platform (NVIDIA CUDA vs AMD ROCm)"""

    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.vendor = None
        self.platform = None
        self.version = None
        self.device_name = None

        if self.gpu_available:
            self._detect_platform()
            self._configure_optimizations()

    def _detect_platform(self):
        """Detect GPU vendor and platform"""
        try:
            # Check for ROCm/HIP (AMD)
            if hasattr(torch.version, 'hip') and torch.version.hip is not None:
                self.vendor = "AMD"
                self.platform = "ROCm/HIP"
                self.version = torch.version.hip
            else:
                # NVIDIA CUDA
                self.vendor = "NVIDIA"
                self.platform = "CUDA"
                self.version = torch.version.cuda

            # Get device name
            if torch.cuda.device_count() > 0:
                self.device_name = torch.cuda.get_device_name(0)

        except Exception as e:
            print(f"Warning: GPU detection failed: {e}")
            self.gpu_available = False

    def _configure_optimizations(self):
        """Configure platform-specific optimizations"""
        try:
            if self.vendor == "NVIDIA":
                # Enable TF32 for Ampere GPUs (RTX 30/40 series)
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print(f"✅ NVIDIA optimizations enabled (TF32)")

            elif self.vendor == "AMD":
                # AMD-specific optimizations
                print(f"✅ AMD optimizations enabled (ROCm)")
                # Note: AMD doesn't have TF32 equivalent

        except Exception as e:
            print(f"Warning: Optimization setup failed: {e}")

    def get_info(self) -> Dict[str, Any]:
        """Get platform information"""
        return {
            "gpu_available": self.gpu_available,
            "vendor": self.vendor,
            "platform": self.platform,
            "version": self.version,
            "device_name": self.device_name
        }

    def print_info(self):
        """Print platform information"""
        info = self.get_info()
        print(f"🔍 GPU Platform Detection:")
        print(f"   Available: {info['gpu_available']}")
        if info['gpu_available']:
            print(f"   Vendor: {info['vendor']}")
            print(f"   Platform: {info['platform']}")
            print(f"   Version: {info['version']}")
            print(f"   Device: {info['device_name']}")

# Global platform detector
_platform = GPUPlatformDetector()

class UnifiedFFTOptimizer:
    """
    Unified FFT optimizer that works with both CUDA and ROCm.

    Automatically uses:
    - CuFFT for NVIDIA GPUs
    - hipFFT for AMD GPUs
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if _platform.gpu_available else "cpu")
        self.platform_info = _platform.get_info()

    def rfft_batch(
        self,
        x: torch.Tensor,
        n_fft: Optional[int] = None,
        dim: int = -1,
        norm: str = "ortho"
    ) -> torch.Tensor:
        """
        Optimized real-to-complex FFT.

        Uses CuFFT (NVIDIA) or hipFFT (AMD) automatically.
        """
        if not x.is_cuda and _platform.gpu_available:
            x = x.to(self.device)

        return torch.fft.rfft(x, n=n_fft, dim=dim, norm=norm)

    def irfft_batch(
        self,
        X: torch.Tensor,
        n: Optional[int] = None,
        dim: int = -1,
        norm: str = "ortho",
        safe_mode: bool = True
    ) -> torch.Tensor:
        """
        Optimized complex-to-real inverse FFT with AMD GPU safety.

        Args:
            safe_mode: If True, uses special handling for AMD GPUs to prevent hangs
        """
        if not X.is_cuda and _platform.gpu_available:
            X = X.to(self.device)

        # AMD GPU specific handling - use safe FFT
        if _platform.vendor == "AMD" and safe_mode and X.is_cuda:
            try:
                from .amd_safe_fft import amd_safe_irfft
                return amd_safe_irfft(X, n=n, dim=dim, norm=norm)
            except ImportError:
                warnings.warn(
                    "AMD safe FFT module not available, using standard computation. "
                    "iFFT may hang on AMD GPUs.",
                    UserWarning
                )
                return torch.fft.irfft(X, n=n, dim=dim, norm=norm)
        else:
            # Standard computation for NVIDIA or CPU
            return torch.fft.irfft(X, n=n, dim=dim, norm=norm)

    def stft_optimized(
        self,
        x: torch.Tensor,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: Optional[torch.Tensor] = None,
        center: bool = True,
        return_complex: bool = True
    ) -> torch.Tensor:
        """
        Optimized Short-Time Fourier Transform.
        """
        if not x.is_cuda and _platform.gpu_available:
            x = x.to(self.device)

        if window is not None and not window.is_cuda:
            window = window.to(self.device)
        elif window is None:
            window = torch.hann_window(
                win_length or n_fft,
                device=self.device
            )

        # Handle batch dimensions
        original_shape = x.shape
        if x.dim() == 3:  # (batch, channels, time)
            batch, channels, time = x.shape
            x = x.reshape(batch * channels, time)
        else:
            batch, channels = 1, 1

        stft_result = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            return_complex=return_complex
        )

        # Reshape back
        if len(original_shape) == 3:
            freq_bins, time_frames = stft_result.shape[-2:]
            stft_result = stft_result.reshape(
                batch, channels, freq_bins, time_frames
            )

        return stft_result

class UnifiedBLASOptimizer:
    """
    Unified BLAS optimizer that works with both CUDA and ROCm.

    Automatically uses:
    - CuBLAS for NVIDIA GPUs
    - rocBLAS for AMD GPUs
    """

    def __init__(self, device: str = "cuda", use_tf32: bool = True):
        self.device = torch.device(device if _platform.gpu_available else "cpu")
        self.platform_info = _platform.get_info()

        # TF32 only available on NVIDIA
        self.use_tf32 = (
            use_tf32 and
            _platform.gpu_available and
            _platform.vendor == "NVIDIA"
        )

        if self.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

    def matmul_optimized(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        transpose_a: bool = False,
        transpose_b: bool = False
    ) -> torch.Tensor:
        """
        Optimized matrix multiplication.

        Uses CuBLAS (NVIDIA) or rocBLAS (AMD) automatically.
        """
        if not a.is_cuda and _platform.gpu_available:
            a = a.to(self.device)
        if not b.is_cuda and _platform.gpu_available:
            b = b.to(self.device)

        if transpose_a:
            a = a.transpose(-2, -1)
        if transpose_b:
            b = b.transpose(-2, -1)

        return torch.matmul(a, b)

    def bmm_optimized(
        self,
        batch1: torch.Tensor,
        batch2: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimized batch matrix multiplication.
        """
        if not batch1.is_cuda and _platform.gpu_available:
            batch1 = batch1.to(self.device)
        if not batch2.is_cuda and _platform.gpu_available:
            batch2 = batch2.to(self.device)

        return torch.bmm(batch1, batch2)

    def addmm_optimized(
        self,
        bias: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        Optimized fused add + matrix multiply: beta * bias + alpha * (input @ weight^T)
        """
        if not bias.is_cuda and _platform.gpu_available:
            bias = bias.to(self.device)
        if not input.is_cuda and _platform.gpu_available:
            input = input.to(self.device)
        if not weight.is_cuda and _platform.gpu_available:
            weight = weight.to(self.device)

        return torch.addmm(bias, input, weight.t(), beta=beta, alpha=alpha)

class UnifiedLinearLayer(nn.Module):
    """
    Linear layer optimized for both NVIDIA and AMD GPUs.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: str = "cuda"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = torch.device(device if _platform.gpu_available else "cpu")

        # Initialize weights
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=self.device)
        )

        if bias:
            self.bias = nn.Parameter(
                torch.empty(out_features, device=self.device)
            )
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        # Unified BLAS optimizer
        self.blas = UnifiedBLASOptimizer(device=str(self.device))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain('relu'))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using optimized operations.
        """
        if self.bias is not None:
            return self.blas.addmm_optimized(
                self.bias, x, self.weight, alpha=1.0, beta=1.0
            )
        else:
            return self.blas.matmul_optimized(x, self.weight, transpose_b=True)

def benchmark_unified_gpu(signal_length: int = 10000, batch_size: int = 32, num_channels: int = 129):
    """
    Benchmark unified GPU performance vs CPU.
    """
    import time

    print(f"\n{'='*60}")
    print("Unified GPU vs CPU Benchmark")
    print(f"{'='*60}")

    # Platform info
    _platform.print_info()
    print(f"Signal: {batch_size}x{num_channels}x{signal_length}")
    print(f"{'='*60}\n")

    # Generate test data
    x_cpu = torch.randn(batch_size, num_channels, signal_length)

    # CPU FFT
    print("CPU FFT...")
    start = time.time()
    for _ in range(10):
        _ = torch.fft.rfft(x_cpu, dim=-1)
    cpu_time = (time.time() - start) / 10
    print(f"  Time: {cpu_time*1000:.2f} ms")

    if _platform.gpu_available:
        # GPU FFT
        fft_opt = UnifiedFFTOptimizer()
        x_gpu = x_cpu.cuda()
        torch.cuda.synchronize()

        # Warmup
        for _ in range(5):
            _ = fft_opt.rfft_batch(x_gpu, dim=-1)
        torch.cuda.synchronize()

        print(f"\n{_platform.platform} FFT...")
        start = time.time()
        for _ in range(10):
            _ = fft_opt.rfft_batch(x_gpu, dim=-1)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 10
        print(f"  Time: {gpu_time*1000:.2f} ms")

        speedup = cpu_time / gpu_time
        print(f"\n🚀 Speedup: {speedup:.2f}x faster with {_platform.platform}")

        # Cleanup
        del x_gpu
        torch.cuda.empty_cache()
    else:
        print("\n⚠️  GPU not available")

def benchmark_unified_matmul(matrix_size: int = 1024, batch_size: int = 32):
    """
    Benchmark unified matrix multiplication.
    """
    import time

    print(f"\n{'='*60}")
    print("Unified Matrix Multiplication Benchmark")
    print(f"{'='*60}")
    print(f"Matrices: {batch_size}x{matrix_size}x{matrix_size}")
    print(f"{'='*60}\n")

    # Generate test matrices
    a_cpu = torch.randn(batch_size, matrix_size, matrix_size)
    b_cpu = torch.randn(batch_size, matrix_size, matrix_size)

    # CPU matmul
    print("CPU Matrix Multiplication...")
    start = time.time()
    for _ in range(10):
        _ = torch.bmm(a_cpu, b_cpu)
    cpu_time = (time.time() - start) / 10
    print(f"  Time: {cpu_time*1000:.2f} ms")

    if _platform.gpu_available:
        # GPU matmul
        blas_opt = UnifiedBLASOptimizer()
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        torch.cuda.synchronize()

        # Warmup
        for _ in range(5):
            _ = blas_opt.bmm_optimized(a_gpu, b_gpu)
        torch.cuda.synchronize()

        print(f"\n{_platform.platform} Matrix Multiplication...")
        start = time.time()
        for _ in range(10):
            _ = blas_opt.bmm_optimized(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 10
        print(f"  Time: {gpu_time*1000:.2f} ms")

        speedup = cpu_time / gpu_time
        print(f"\n🚀 Speedup: {speedup:.2f}x faster with {_platform.platform}")

        # Cleanup
        del a_gpu, b_gpu
        torch.cuda.empty_cache()
    else:
        print("\n⚠️  GPU not available")

# Export main classes and functions
__all__ = [
    "GPUPlatformDetector",
    "UnifiedFFTOptimizer",
    "UnifiedBLASOptimizer",
    "UnifiedLinearLayer",
    "benchmark_unified_gpu",
    "benchmark_unified_matmul"
]

if __name__ == "__main__":
    print("Unified GPU Optimization Module")
    print("=" * 60)
    _platform.print_info()
    print("=" * 60)

    # Run benchmarks if GPU available
    if _platform.gpu_available:
        try:
            benchmark_unified_gpu(signal_length=5000, batch_size=16, num_channels=64)
            benchmark_unified_matmul(matrix_size=512, batch_size=16)
        except Exception as e:
            print(f"Benchmark failed: {e}")
            print("This may be due to insufficient GPU memory or driver issues")
    else:
        print("\n⚠️  No GPU available. Install CUDA (NVIDIA) or ROCm (AMD) PyTorch.")
