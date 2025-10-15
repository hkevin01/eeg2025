#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified GPU Optimized Operations using CUDA/CuFFT/CuBLAS or ROCm/hipFFT/rocBLAS
===============================================================================

High-performance GPU operations that automatically detect and use:
- NVIDIA: CuFFT for Fast Fourier Transforms, CuBLAS for BLAS operations
- AMD: hipFFT for Fast Fourier Transforms, rocBLAS for BLAS operations

These libraries are automatically used by PyTorch when GPU is available,
but this module provides explicit controls and optimizations for both platforms.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# Detect GPU platform
def detect_gpu_platform():
    """Detect whether we're using CUDA (NVIDIA) or ROCm/HIP (AMD)"""
    if not torch.cuda.is_available():
        return None, "CPU"

    # Check for ROCm/HIP
    if hasattr(torch.version, 'hip') and torch.version.hip is not None:
        return "AMD", "ROCm/HIP"
    else:
        return "NVIDIA", "CUDA"

# Platform detection
GPU_VENDOR, GPU_PLATFORM = detect_gpu_platform()
GPU_AVAILABLE = torch.cuda.is_available()  # Works for both CUDA and ROCm
CUFFT_AVAILABLE = GPU_AVAILABLE and GPU_VENDOR == "NVIDIA"
CUBLAS_AVAILABLE = GPU_AVAILABLE and GPU_VENDOR == "NVIDIA"
HIPFFT_AVAILABLE = GPU_AVAILABLE and GPU_VENDOR == "AMD"
ROCBLAS_AVAILABLE = GPU_AVAILABLE and GPU_VENDOR == "AMD"

print("🔍 GPU Detection Results:")
print(f"   Platform: {GPU_PLATFORM}")
print(f"   Vendor: {GPU_VENDOR}")
print(f"   Available: {GPU_AVAILABLE}")

if GPU_AVAILABLE:
    try:
        # Enable optimizations based on platform
        if GPU_VENDOR == "NVIDIA":
            # Enable TF32 for better performance on Ampere GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            print("✅ NVIDIA optimizations: TF32 enabled for CuBLAS/CuDNN")
        elif GPU_VENDOR == "AMD":
            # AMD-specific optimizations
            print("✅ AMD optimizations: ROCm/HIP backend detected")
            # Note: AMD doesn't have TF32, but has other optimizations
    except AttributeError:
        pass


class CuFFTOptimizer:
    """
    Optimized FFT operations using CuFFT (NVIDIA) or hipFFT (AMD).

    Automatically uses the appropriate FFT library based on GPU platform:
    - NVIDIA: CuFFT
    - AMD: hipFFT (via ROCm)
    """

    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if GPU_AVAILABLE else "cpu")
        self.platform = GPU_PLATFORM

    def rfft_batch(
        self,
        x: torch.Tensor,
        n_fft: Optional[int] = None,
        dim: int = -1,
        norm: str = "ortho"
    ) -> torch.Tensor:
        """
        Optimized real-to-complex FFT for batch processing.

        Uses CuFFT (NVIDIA) or hipFFT (AMD) when on GPU device.

        Args:
            x: Input tensor (..., signal_length)
            n_fft: FFT size (default: signal length)
            dim: Dimension to apply FFT
            norm: Normalization mode

        Returns:
            Complex FFT coefficients
        """
        if not x.is_cuda and GPU_AVAILABLE:
            x = x.to(self.device)

        # CuFFT or hipFFT is automatically used for GPU tensors
        return torch.fft.rfft(x, n=n_fft, dim=dim, norm=norm)

    def irfft_batch(
        self,
        X: torch.Tensor,
        n: Optional[int] = None,
        dim: int = -1,
        norm: str = "ortho"
    ) -> torch.Tensor:
        """
        Optimized complex-to-real inverse FFT.

        Args:
            X: Complex FFT coefficients
            n: Output signal length
            dim: Dimension to apply IFFT
            norm: Normalization mode

        Returns:
            Real signal
        """
        if not X.is_cuda and GPU_AVAILABLE:
            X = X.to(self.device)

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
        Optimized Short-Time Fourier Transform using CuFFT.

        Args:
            x: Input signal (batch, channels, time)
            n_fft: FFT size
            hop_length: Hop size between frames
            win_length: Window size
            window: Window function
            center: Whether to center frames
            return_complex: Return complex or real/imag split

        Returns:
            STFT spectrogram
        """
        if not x.is_cuda and CUDA_AVAILABLE:
            x = x.to(self.device)

        if window is not None and not window.is_cuda:
            window = window.to(self.device)
        elif window is None:
            window = torch.hann_window(
                win_length or n_fft,
                device=self.device
            )

        # Reshape for torch.stft: (batch * channels, time)
        original_shape = x.shape
        if x.dim() == 3:  # (batch, channels, time)
            batch, channels, time = x.shape
            x = x.reshape(batch * channels, time)
        else:
            batch, channels = 1, 1

        # CuFFT-accelerated STFT
        stft_result = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            return_complex=return_complex
        )

        # Reshape back: (batch, channels, freq, time)
        if len(original_shape) == 3:
            freq_bins, time_frames = stft_result.shape[-2:]
            stft_result = stft_result.reshape(
                batch, channels, freq_bins, time_frames
            )

        return stft_result

    def istft_optimized(
        self,
        stft_matrix: torch.Tensor,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: Optional[torch.Tensor] = None,
        center: bool = True,
        length: Optional[int] = None
    ) -> torch.Tensor:
        """
        Optimized Inverse STFT using CuFFT.

        Args:
            stft_matrix: STFT spectrogram (batch, channels, freq, time)
            n_fft: FFT size
            hop_length: Hop size
            win_length: Window size
            window: Window function
            center: Whether frames were centered
            length: Target output length

        Returns:
            Time-domain signal
        """
        if not stft_matrix.is_cuda and CUDA_AVAILABLE:
            stft_matrix = stft_matrix.to(self.device)

        if window is not None and not window.is_cuda:
            window = window.to(self.device)
        elif window is None:
            window = torch.hann_window(
                win_length or n_fft,
                device=self.device
            )

        # Reshape for torch.istft
        original_shape = stft_matrix.shape
        if stft_matrix.dim() == 4:  # (batch, channels, freq, time)
            batch, channels = stft_matrix.shape[:2]
            stft_matrix = stft_matrix.reshape(
                batch * channels, *stft_matrix.shape[-2:]
            )
        else:
            batch, channels = 1, 1

        # CuFFT-accelerated ISTFT
        signal = torch.istft(
            stft_matrix,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            length=length
        )

        # Reshape back
        if len(original_shape) == 4:
            signal = signal.reshape(batch, channels, -1)

        return signal


class CuBLASOptimizer:
    """
    Optimized linear algebra operations using CuBLAS.

    CuBLAS provides highly optimized matrix operations on CUDA devices.
    PyTorch automatically uses CuBLAS, but this class provides additional
    optimizations and controls.
    """

    def __init__(self, device: str = "cuda", use_tf32: bool = True):
        self.device = torch.device(device if CUDA_AVAILABLE else "cpu")
        self.use_tf32 = use_tf32 and CUDA_AVAILABLE

        if self.use_tf32:
            # TF32 provides ~8x speedup on Ampere GPUs with minimal precision loss
            torch.backends.cuda.matmul.allow_tf32 = True

    def matmul_optimized(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        transpose_a: bool = False,
        transpose_b: bool = False
    ) -> torch.Tensor:
        """
        Optimized matrix multiplication using CuBLAS.

        Args:
            a: First matrix (*, m, k)
            b: Second matrix (*, k, n)
            transpose_a: Transpose first matrix
            transpose_b: Transpose second matrix

        Returns:
            Product matrix (*, m, n)
        """
        if not a.is_cuda and CUDA_AVAILABLE:
            a = a.to(self.device)
        if not b.is_cuda and CUDA_AVAILABLE:
            b = b.to(self.device)

        if transpose_a:
            a = a.transpose(-2, -1)
        if transpose_b:
            b = b.transpose(-2, -1)

        # CuBLAS is automatically used for CUDA tensors
        return torch.matmul(a, b)

    def bmm_optimized(
        self,
        batch1: torch.Tensor,
        batch2: torch.Tensor
    ) -> torch.Tensor:
        """
        Optimized batch matrix multiplication.

        Uses CuBLAS batched GEMM for efficient batch processing.

        Args:
            batch1: First batch (batch, m, k)
            batch2: Second batch (batch, k, n)

        Returns:
            Product batch (batch, m, n)
        """
        if not batch1.is_cuda and CUDA_AVAILABLE:
            batch1 = batch1.to(self.device)
        if not batch2.is_cuda and CUDA_AVAILABLE:
            batch2 = batch2.to(self.device)

        # CuBLAS batched GEMM
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

        This is a common pattern in neural networks (linear layers).
        CuBLAS provides fused kernel for better performance.

        Args:
            bias: Bias vector (out_features,)
            input: Input matrix (*, in_features)
            weight: Weight matrix (out_features, in_features)
            alpha: Multiplier for product
            beta: Multiplier for bias

        Returns:
            Result: beta * bias + alpha * (input @ weight^T)
        """
        if not bias.is_cuda and CUDA_AVAILABLE:
            bias = bias.to(self.device)
        if not input.is_cuda and CUDA_AVAILABLE:
            input = input.to(self.device)
        if not weight.is_cuda and CUDA_AVAILABLE:
            weight = weight.to(self.device)

        # Fused operation using CuBLAS
        return torch.addmm(bias, input, weight.t(), beta=beta, alpha=alpha)


class OptimizedLinearLayer(nn.Module):
    """
    Linear layer optimized with CuBLAS operations.

    Uses CuBLAS-accelerated addmm for fused bias addition and matrix multiplication.
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
        self.device = torch.device(device if CUDA_AVAILABLE else "cpu")

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

        # CuBLAS optimizer
        self.cublas = CuBLASOptimizer(device=str(self.device))

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain('relu'))
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass using optimized CuBLAS operations.

        Args:
            x: Input tensor (*, in_features)

        Returns:
            Output tensor (*, out_features)
        """
        if self.bias is not None:
            # Fused addmm: bias + (x @ weight^T)
            return self.cublas.addmm_optimized(
                self.bias, x, self.weight, alpha=1.0, beta=1.0
            )
        else:
            # Just matrix multiply
            return self.cublas.matmul_optimized(x, self.weight, transpose_b=True)


class OptimizedAttention(nn.Module):
    """
    Multi-head attention optimized with CuBLAS batched matrix operations.

    Uses CuBLAS for efficient batched QKV projections and attention computation.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        device: str = "cuda"
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.device = torch.device(device if CUDA_AVAILABLE else "cpu")

        # QKV projection (fused for efficiency)
        self.qkv_proj = OptimizedLinearLayer(
            embed_dim, 3 * embed_dim, device=str(self.device)
        )

        # Output projection
        self.out_proj = OptimizedLinearLayer(
            embed_dim, embed_dim, device=str(self.device)
        )

        self.dropout = nn.Dropout(dropout)
        self.cublas = CuBLASOptimizer(device=str(self.device))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optimized attention computation.

        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            Output tensor (batch, seq_len, embed_dim)
        """
        batch_size, seq_len, _ = x.shape

        # QKV projection (single optimized linear operation)
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores using CuBLAS batched matmul
        # (batch, heads, seq_len, head_dim) @ (batch, heads, head_dim, seq_len)
        # -> (batch, heads, seq_len, seq_len)
        scores = self.cublas.bmm_optimized(
            q.reshape(batch_size * self.num_heads, seq_len, self.head_dim),
            k.reshape(batch_size * self.num_heads, seq_len, self.head_dim).transpose(-2, -1)
        ).reshape(batch_size, self.num_heads, seq_len, seq_len)

        scores = scores / (self.head_dim ** 0.5)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Attention output using CuBLAS batched matmul
        # (batch, heads, seq_len, seq_len) @ (batch, heads, seq_len, head_dim)
        # -> (batch, heads, seq_len, head_dim)
        attn_output = self.cublas.bmm_optimized(
            attn_weights.reshape(batch_size * self.num_heads, seq_len, seq_len),
            v.reshape(batch_size * self.num_heads, seq_len, self.head_dim)
        ).reshape(batch_size, self.num_heads, seq_len, self.head_dim)

        # Concatenate heads
        attn_output = attn_output.permute(0, 2, 1, 3).reshape(
            batch_size, seq_len, self.embed_dim
        )

        # Output projection
        output = self.out_proj(attn_output)

        return output


def benchmark_cufft_vs_cpu(signal_length: int = 10000, batch_size: int = 32, num_channels: int = 129):
    """
    Benchmark CuFFT vs CPU FFT performance.

    Args:
        signal_length: Length of signal
        batch_size: Batch size
        num_channels: Number of channels
    """
    import time

    print(f"\n{'='*60}")
    print("CuFFT vs CPU FFT Benchmark")
    print(f"{'='*60}")
    print(f"Signal: {batch_size}x{num_channels}x{signal_length}")
    print(f"{'='*60}\n")

    # Generate random signal
    x_cpu = torch.randn(batch_size, num_channels, signal_length)

    # CPU FFT
    print("CPU FFT...")
    start = time.time()
    for _ in range(10):
        _ = torch.fft.rfft(x_cpu, dim=-1)
    cpu_time = (time.time() - start) / 10
    print(f"  Time: {cpu_time*1000:.2f} ms")

    if CUDA_AVAILABLE:
        # GPU FFT (CuFFT)
        x_gpu = x_cpu.cuda()
        torch.cuda.synchronize()

        # Warmup
        for _ in range(5):
            _ = torch.fft.rfft(x_gpu, dim=-1)
        torch.cuda.synchronize()

        print("\nCuFFT (GPU FFT)...")
        start = time.time()
        for _ in range(10):
            _ = torch.fft.rfft(x_gpu, dim=-1)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 10
        print(f"  Time: {gpu_time*1000:.2f} ms")

        speedup = cpu_time / gpu_time
        print(f"\n🚀 Speedup: {speedup:.2f}x faster with CuFFT")
    else:
        print("\n⚠️  CUDA not available - cannot benchmark GPU")


def benchmark_cublas_vs_cpu(matrix_size: int = 1024, batch_size: int = 32):
    """
    Benchmark CuBLAS vs CPU matrix multiplication.

    Args:
        matrix_size: Size of square matrices
        batch_size: Batch size
    """
    import time

    print(f"\n{'='*60}")
    print("CuBLAS vs CPU Matrix Multiplication Benchmark")
    print(f"{'='*60}")
    print(f"Matrices: {batch_size}x{matrix_size}x{matrix_size}")
    print(f"{'='*60}\n")

    # Generate random matrices
    a_cpu = torch.randn(batch_size, matrix_size, matrix_size)
    b_cpu = torch.randn(batch_size, matrix_size, matrix_size)

    # CPU matmul
    print("CPU Matrix Multiplication...")
    start = time.time()
    for _ in range(10):
        _ = torch.bmm(a_cpu, b_cpu)
    cpu_time = (time.time() - start) / 10
    print(f"  Time: {cpu_time*1000:.2f} ms")

    if CUDA_AVAILABLE:
        # GPU matmul (CuBLAS)
        a_gpu = a_cpu.cuda()
        b_gpu = b_cpu.cuda()
        torch.cuda.synchronize()

        # Warmup
        for _ in range(5):
            _ = torch.bmm(a_gpu, b_gpu)
        torch.cuda.synchronize()

        print("\nCuBLAS (GPU Matrix Multiplication)...")
        start = time.time()
        for _ in range(10):
            _ = torch.bmm(a_gpu, b_gpu)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 10
        print(f"  Time: {gpu_time*1000:.2f} ms")

        speedup = cpu_time / gpu_time
        print(f"\n🚀 Speedup: {speedup:.2f}x faster with CuBLAS")
    else:
        print("\n⚠️  CUDA not available - cannot benchmark GPU")


if __name__ == "__main__":
    print("CUDA/CuFFT/CuBLAS Optimization Module")
    print("=" * 60)
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    if CUDA_AVAILABLE:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CuFFT Available: {CUFFT_AVAILABLE}")
        print(f"CuBLAS Available: {CUBLAS_AVAILABLE}")
    print("=" * 60)

    # Run benchmarks
    if CUDA_AVAILABLE:
        benchmark_cufft_vs_cpu(signal_length=10000, batch_size=32, num_channels=129)
        benchmark_cublas_vs_cpu(matrix_size=512, batch_size=32)
    else:
        print("\n⚠️  CUDA not available. Install CUDA-enabled PyTorch to use CuFFT/CuBLAS optimizations.")
        print("   Visit: https://pytorch.org/get-started/locally/")
