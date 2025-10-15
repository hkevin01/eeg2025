#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMD ROCm Optimized Operations
=============================

GPU acceleration for AMD GPUs using ROCm/HIP.
PyTorch with ROCm uses the same CUDA API but calls ROCm libraries underneath.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
import warnings
import time

# Check ROCm availability
ROCM_AVAILABLE = torch.cuda.is_available()  # ROCm PyTorch uses cuda namespace
HIP_AVAILABLE = ROCM_AVAILABLE

if ROCM_AVAILABLE:
    # Check if this is actually ROCm
    if hasattr(torch.version, 'hip') and torch.version.hip:
        print(f"✅ ROCm/HIP {torch.version.hip} detected")
        # Enable optimizations for AMD GPUs
        torch.backends.cuda.matmul.allow_tf32 = False  # TF32 not available on AMD
        print("✅ AMD GPU optimizations enabled")
    else:
        print("⚠️  CUDA detected instead of ROCm")


class ROCmFFTOptimizer:
    """
    AMD ROCm optimized FFT operations using hipFFT.
    
    PyTorch automatically uses hipFFT when running on AMD GPUs with ROCm.
    """
    
    def __init__(self, device: str = "cuda"):
        # PyTorch ROCm uses "cuda" device name but maps to ROCm
        self.device = torch.device(device if ROCM_AVAILABLE else "cpu")
        
    def safe_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Safely move tensor to AMD GPU"""
        if not tensor.is_cuda and ROCM_AVAILABLE:
            return tensor.to(self.device, non_blocking=True)
        return tensor
        
    def rfft_amd(
        self, 
        x: torch.Tensor, 
        n_fft: Optional[int] = None,
        dim: int = -1,
        norm: str = "ortho"
    ) -> torch.Tensor:
        """
        AMD-optimized real-to-complex FFT using hipFFT.
        
        Args:
            x: Input tensor (..., signal_length)
            n_fft: FFT size
            dim: Dimension to apply FFT
            norm: Normalization mode
            
        Returns:
            Complex FFT coefficients
        """
        x = self.safe_to_device(x)
        
        # hipFFT is used automatically for ROCm builds
        return torch.fft.rfft(x, n=n_fft, dim=dim, norm=norm)
    
    def irfft_amd(
        self,
        X: torch.Tensor,
        n: Optional[int] = None,
        dim: int = -1,
        norm: str = "ortho"
    ) -> torch.Tensor:
        """AMD-optimized inverse FFT using hipFFT"""
        X = self.safe_to_device(X)
        return torch.fft.irfft(X, n=n, dim=dim, norm=norm)
    
    def stft_amd(
        self,
        x: torch.Tensor,
        n_fft: int = 512,
        hop_length: int = 256,
        win_length: Optional[int] = None,
        window: Optional[torch.Tensor] = None,
        center: bool = True,
        return_complex: bool = True
    ) -> torch.Tensor:
        """AMD-optimized STFT using hipFFT"""
        x = self.safe_to_device(x)
        
        if window is not None:
            window = self.safe_to_device(window)
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
            
        # hipFFT-accelerated STFT
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


class ROCmBLASOptimizer:
    """
    AMD ROCm optimized linear algebra using rocBLAS.
    
    PyTorch automatically uses rocBLAS for matrix operations on AMD GPUs.
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = torch.device(device if ROCM_AVAILABLE else "cpu")
        
    def safe_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Safely move tensor to AMD GPU"""
        if not tensor.is_cuda and ROCM_AVAILABLE:
            return tensor.to(self.device, non_blocking=True)
        return tensor
    
    def matmul_amd(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        transpose_a: bool = False,
        transpose_b: bool = False
    ) -> torch.Tensor:
        """AMD-optimized matrix multiplication using rocBLAS"""
        a = self.safe_to_device(a)
        b = self.safe_to_device(b)
        
        if transpose_a:
            a = a.transpose(-2, -1)
        if transpose_b:
            b = b.transpose(-2, -1)
            
        # rocBLAS used automatically
        return torch.matmul(a, b)
    
    def bmm_amd(
        self,
        batch1: torch.Tensor,
        batch2: torch.Tensor
    ) -> torch.Tensor:
        """AMD-optimized batch matrix multiplication using rocBLAS"""
        batch1 = self.safe_to_device(batch1)
        batch2 = self.safe_to_device(batch2)
        
        return torch.bmm(batch1, batch2)
    
    def addmm_amd(
        self,
        bias: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        alpha: float = 1.0,
        beta: float = 1.0
    ) -> torch.Tensor:
        """AMD-optimized fused add + matrix multiply using rocBLAS"""
        bias = self.safe_to_device(bias)
        input = self.safe_to_device(input)
        weight = self.safe_to_device(weight)
        
        return torch.addmm(bias, input, weight.t(), beta=beta, alpha=alpha)


class AMDOptimizedLinear(nn.Module):
    """Linear layer optimized for AMD GPUs using rocBLAS"""
    
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
        self.device = torch.device(device if ROCM_AVAILABLE else "cpu")
        
        # Initialize weights on AMD GPU
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
        self.rocblas = ROCmBLASOptimizer(device=str(self.device))
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=torch.nn.init.calculate_gain('relu'))
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using AMD-optimized rocBLAS"""
        if self.bias is not None:
            return self.rocblas.addmm_amd(self.bias, x, self.weight)
        else:
            return self.rocblas.matmul_amd(x, self.weight, transpose_b=True)


def benchmark_amd_vs_cpu(signal_length: int = 5000, batch_size: int = 16, num_channels: int = 64):
    """
    Benchmark AMD GPU vs CPU performance.
    
    Args:
        signal_length: Length of signal
        batch_size: Batch size
        num_channels: Number of channels
    """
    print(f"\n{'='*60}")
    print("AMD GPU (hipFFT/rocBLAS) vs CPU Benchmark")
    print(f"{'='*60}")
    print(f"Signal: {batch_size}x{num_channels}x{signal_length}")
    print(f"{'='*60}\n")
    
    # Generate test data
    x_cpu = torch.randn(batch_size, num_channels, signal_length)
    
    # CPU FFT
    print("CPU FFT...")
    start = time.time()
    for _ in range(5):
        _ = torch.fft.rfft(x_cpu, dim=-1)
    cpu_time = (time.time() - start) / 5
    print(f"  Time: {cpu_time*1000:.2f} ms")
    
    if ROCM_AVAILABLE:
        # AMD GPU FFT (hipFFT)
        x_amd = x_cpu.cuda()
        torch.cuda.synchronize()  # ROCm uses same sync
        
        # Warmup
        for _ in range(3):
            _ = torch.fft.rfft(x_amd, dim=-1)
        torch.cuda.synchronize()
        
        print("\nAMD GPU FFT (hipFFT)...")
        start = time.time()
        for _ in range(5):
            _ = torch.fft.rfft(x_amd, dim=-1)
        torch.cuda.synchronize()
        gpu_time = (time.time() - start) / 5
        print(f"  Time: {gpu_time*1000:.2f} ms")
        
        speedup = cpu_time / gpu_time
        print(f"\n🚀 AMD GPU Speedup: {speedup:.2f}x faster")
        
        # Cleanup
        del x_amd
        torch.cuda.empty_cache()
    else:
        print("\n⚠️  ROCm not available")


def test_amd_safe():
    """Safe test for AMD GPU operations"""
    print("\n🔧 AMD GPU Safe Test")
    print("-" * 40)
    
    if not ROCM_AVAILABLE:
        print("❌ ROCm not available")
        return False
    
    try:
        # Very small test
        print("Creating small tensor on AMD GPU...")
        x = torch.randn(64, 64, device='cuda')
        print(f"✅ Tensor created: {x.shape}")
        
        # Basic operation
        print("Testing basic operation...")
        y = x + 1
        torch.cuda.synchronize()
        print(f"✅ Operation completed: {y.shape}")
        
        # FFT test
        print("Testing small FFT...")
        signal = torch.randn(256, device='cuda')
        fft_result = torch.fft.rfft(signal)
        torch.cuda.synchronize()
        print(f"✅ FFT: {signal.shape} -> {fft_result.shape}")
        
        # Matrix multiplication
        print("Testing matrix multiplication...")
        a = torch.randn(128, 128, device='cuda')
        b = torch.randn(128, 128, device='cuda')
        c = torch.matmul(a, b)
        torch.cuda.synchronize()
        print(f"✅ MatMul: {a.shape} @ {b.shape} = {c.shape}")
        
        # Cleanup
        del x, y, signal, fft_result, a, b, c
        torch.cuda.empty_cache()
        print("✅ Cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ AMD test failed: {e}")
        # Emergency cleanup
        torch.cuda.empty_cache()
        return False


if __name__ == "__main__":
    print("AMD ROCm Optimization Module")
    print("=" * 60)
    print(f"ROCm Available: {ROCM_AVAILABLE}")
    
    if ROCM_AVAILABLE:
        if hasattr(torch.version, 'hip'):
            print(f"HIP Version: {torch.version.hip}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        # Run safe test first
        if test_amd_safe():
            print("\n✅ AMD GPU operations working safely")
            # Run benchmark
            benchmark_amd_vs_cpu(signal_length=5000, batch_size=16, num_channels=64)
        else:
            print("\n❌ AMD GPU operations not working")
    else:
        print("\n⚠️  ROCm not available")
        print("Make sure you have:")
        print("1. ROCm installed")
        print("2. PyTorch with ROCm support")
        print("3. Proper environment variables")
