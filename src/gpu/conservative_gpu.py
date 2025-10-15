#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Conservative GPU Operations
==========================

Ultra-safe GPU operations for problematic AMD GPUs.
Falls back to CPU for operations that commonly crash.
"""

import torch
import warnings
from typing import Optional

class ConservativeGPUOptimizer:
    """
    Conservative GPU operations that prioritize stability over performance.
    
    For AMD GPUs with known issues:
    - Uses GPU only for basic operations (matmul, simple ops)
    - Falls back to CPU for FFT operations that commonly hang/crash
    - Provides safe alternatives with automatic fallback
    """
    
    def __init__(self, device: str = "cuda"):
        self.gpu_available = torch.cuda.is_available()
        self.device = torch.device(device if self.gpu_available else "cpu")
        
        # Detect platform
        self.is_amd = False
        if self.gpu_available and hasattr(torch.version, 'hip'):
            self.is_amd = torch.version.hip is not None
        
        if self.is_amd:
            print("⚠️  AMD GPU detected - using conservative mode for stability")
        
    def safe_matmul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Safe matrix multiplication"""
        if self.gpu_available and not self.is_amd:
            # Use GPU for NVIDIA
            if not a.is_cuda:
                a = a.to(self.device)
            if not b.is_cuda:
                b = b.to(self.device)
            return torch.matmul(a, b)
        else:
            # CPU for AMD or no GPU
            a_cpu = a.cpu() if a.is_cuda else a
            b_cpu = b.cpu() if b.is_cuda else b
            return torch.matmul(a_cpu, b_cpu)
    
    def safe_fft(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Safe FFT - always uses CPU for AMD"""
        if self.is_amd:
            warnings.warn(
                "Using CPU FFT for AMD GPU to prevent crashes. "
                "GPU FFT disabled due to known stability issues.",
                UserWarning
            )
            x_cpu = x.cpu() if x.is_cuda else x
            return torch.fft.rfft(x_cpu, dim=dim)
        else:
            # NVIDIA or CPU
            if self.gpu_available and not x.is_cuda:
                x = x.to(self.device)
            return torch.fft.rfft(x, dim=dim)
    
    def safe_ifft(self, X: torch.Tensor, n: Optional[int] = None, dim: int = -1) -> torch.Tensor:
        """Safe inverse FFT - always uses CPU for AMD"""
        if self.is_amd:
            warnings.warn(
                "Using CPU iFFT for AMD GPU to prevent hangs. "
                "GPU iFFT disabled due to known stability issues.",
                UserWarning
            )
            X_cpu = X.cpu() if X.is_cuda else X
            return torch.fft.irfft(X_cpu, n=n, dim=dim)
        else:
            # NVIDIA or CPU
            if self.gpu_available and not X.is_cuda:
                X = X.to(self.device)
            return torch.fft.irfft(X, n=n, dim=dim)
    
    def get_optimal_device(self, operation: str = "general") -> torch.device:
        """Get optimal device for specific operation"""
        if operation in ["fft", "ifft"] and self.is_amd:
            return torch.device("cpu")
        elif self.gpu_available:
            return self.device
        else:
            return torch.device("cpu")
    
    def to_optimal_device(self, tensor: torch.Tensor, operation: str = "general") -> torch.Tensor:
        """Move tensor to optimal device for operation"""
        optimal_device = self.get_optimal_device(operation)
        return tensor.to(optimal_device)

# Global conservative optimizer
_conservative_gpu = ConservativeGPUOptimizer()

def conservative_fft(x, dim=-1):
    """Conservative FFT that avoids AMD GPU crashes"""
    return _conservative_gpu.safe_fft(x, dim=dim)

def conservative_ifft(X, n=None, dim=-1):
    """Conservative inverse FFT that avoids AMD GPU hangs"""
    return _conservative_gpu.safe_ifft(X, n=n, dim=dim)

def conservative_matmul(a, b):
    """Conservative matrix multiplication"""
    return _conservative_gpu.safe_matmul(a, b)
