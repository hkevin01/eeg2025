#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AMD-Safe FFT Operations
======================

Specific handling for AMD GPU FFT operations that may hang.
"""

import torch
import threading
import queue
import time
import warnings
from typing import Optional

class AMDSafeFFT:
    """AMD GPU-safe FFT operations with timeout and fallback"""
    
    def __init__(self, timeout_seconds: float = 5.0):
        self.timeout = timeout_seconds
        
    def rfft_safe(
        self,
        x: torch.Tensor,
        n: Optional[int] = None,
        dim: int = -1,
        norm: str = "ortho"
    ) -> torch.Tensor:
        """Safe real-to-complex FFT for AMD GPUs"""
        if not x.is_cuda:
            return torch.fft.rfft(x, n=n, dim=dim, norm=norm)
        
        # AMD GPUs handle forward FFT fine
        return torch.fft.rfft(x, n=n, dim=dim, norm=norm)
    
    def irfft_safe(
        self,
        X: torch.Tensor,
        n: Optional[int] = None,
        dim: int = -1,
        norm: str = "ortho"
    ) -> torch.Tensor:
        """Safe complex-to-real inverse FFT for AMD GPUs"""
        if not X.is_cuda:
            return torch.fft.irfft(X, n=n, dim=dim, norm=norm)
        
        # Try GPU first with thread timeout
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def compute_irfft():
            try:
                result = torch.fft.irfft(X, n=n, dim=dim, norm=norm)
                result_queue.put(result)
            except Exception as e:
                error_queue.put(str(e))
        
        # Start computation in thread
        thread = threading.Thread(target=compute_irfft)
        thread.daemon = True  # Dies when main thread dies
        thread.start()
        thread.join(timeout=self.timeout)
        
        if thread.is_alive():
            # Thread timed out - use CPU fallback
            warnings.warn(
                f"AMD GPU iFFT timed out after {self.timeout}s, using CPU fallback. "
                "This is a known ROCm/hipFFT issue.",
                UserWarning
            )
            
            X_cpu = X.cpu()
            result_cpu = torch.fft.irfft(X_cpu, n=n, dim=dim, norm=norm)
            
            # Return on GPU if original was on GPU
            return result_cpu.to(X.device)
        
        # Check results
        if not result_queue.empty():
            return result_queue.get()
        elif not error_queue.empty():
            error_msg = error_queue.get()
            warnings.warn(
                f"AMD GPU iFFT failed: {error_msg}, using CPU fallback",
                UserWarning
            )
            
            X_cpu = X.cpu()
            result_cpu = torch.fft.irfft(X_cpu, n=n, dim=dim, norm=norm)
            return result_cpu.to(X.device)
        else:
            # Shouldn't happen, but fallback anyway
            X_cpu = X.cpu()
            result_cpu = torch.fft.irfft(X_cpu, n=n, dim=dim, norm=norm)
            return result_cpu.to(X.device)

# Global AMD-safe FFT instance
_amd_fft = AMDSafeFFT(timeout_seconds=3.0)

def amd_safe_rfft(x, n=None, dim=-1, norm="ortho"):
    """AMD-safe real FFT"""
    return _amd_fft.rfft_safe(x, n=n, dim=dim, norm=norm)

def amd_safe_irfft(X, n=None, dim=-1, norm="ortho"):
    """AMD-safe inverse real FFT"""
    return _amd_fft.irfft_safe(X, n=n, dim=dim, norm=norm)
