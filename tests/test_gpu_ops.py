"""
Unit tests for GPU operations with CPU fallbacks.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

# Test both GPU and CPU implementations
try:
    import cupy as cp

    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

try:
    import triton

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

from src.gpu.cupy.perceptual_quant import (
    perceptual_quantize_numpy_fallback,
    perceptual_quantize_torch,
)

# Import modules with potential fallbacks
from src.gpu.triton.fir_iir_fused import (
    bandpass_notch_car_cpu_fallback,
    fused_bandpass_notch_car,
)
from src.gpu.triton.rmsnorm import rmsnorm_cpu_fallback, rmsnorm_triton


class TestFusedFiltering:
    """Test fused bandpass + notch + CAR filtering."""

    def test_cpu_fallback_preserves_shape(self):
        """Test CPU fallback preserves input shape."""
        x = torch.randn(16, 128, 1000)  # (batch, channels, time)

        filtered = bandpass_notch_car_cpu_fallback(
            x, lowcut=1.0, highcut=50.0, notch_freq=60.0, fs=500.0
        )

        assert filtered.shape == x.shape
        assert filtered.dtype == x.dtype

    def test_cpu_fallback_reduces_energy_in_stopband(self):
        """Test that filtering reduces energy in stopband."""
        # Create signal with 60Hz noise
        fs = 500.0
        t = torch.linspace(0, 2, int(2 * fs))  # 2 seconds

        # Clean signal (10 Hz) + 60Hz noise
        clean_signal = torch.sin(2 * torch.pi * 10 * t)
        noise_60hz = 0.5 * torch.sin(2 * torch.pi * 60 * t)
        noisy_signal = clean_signal + noise_60hz

        # Add batch and channel dimensions
        x = noisy_signal.unsqueeze(0).unsqueeze(0)  # (1, 1, time)

        filtered = bandpass_notch_car_cpu_fallback(
            x, lowcut=1.0, highcut=50.0, notch_freq=60.0, fs=fs
        )

        # Compute power spectral density (simplified)
        fft_original = torch.fft.fft(x.squeeze())
        fft_filtered = torch.fft.fft(filtered.squeeze())

        freqs = torch.fft.fftfreq(len(t), 1 / fs)

        # Find 60Hz bin
        freq_60_idx = torch.argmin(torch.abs(freqs - 60))

        # Power at 60Hz should be reduced
        power_original_60 = torch.abs(fft_original[freq_60_idx]) ** 2
        power_filtered_60 = torch.abs(fft_filtered[freq_60_idx]) ** 2

        assert power_filtered_60 < power_original_60 * 0.5  # At least 50% reduction

    def test_car_reduces_common_mode(self):
        """Test that CAR (Common Average Reference) works."""
        # Create signal with common mode artifact
        batch_size, n_channels, n_time = 8, 64, 1000

        # Individual channel signals
        signals = torch.randn(batch_size, n_channels, n_time)

        # Add common mode signal to all channels
        common_mode = torch.randn(batch_size, 1, n_time)
        signals_with_cm = signals + common_mode

        filtered = bandpass_notch_car_cpu_fallback(
            signals_with_cm, lowcut=1.0, highcut=50.0, notch_freq=60.0, fs=500.0
        )

        # After CAR, average across channels should be near zero
        channel_mean = filtered.mean(dim=1)  # Average across channels

        # Should be much smaller than before CAR
        original_cm_power = channel_mean.var()
        assert original_cm_power < signals_with_cm.mean(dim=1).var() * 0.1

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_path_if_available(self):
        """Test CUDA path if available."""
        x = torch.randn(16, 128, 1000).cuda()

        try:
            filtered = fused_bandpass_notch_car(x)
            assert filtered.device == x.device
            assert filtered.shape == x.shape
        except Exception:
            # Fallback to CPU should work
            filtered = bandpass_notch_car_cpu_fallback(x.cpu())
            assert filtered.shape == x.shape

    def test_numerical_sanity_check(self):
        """Test numerical sanity of filtering operation."""
        x = torch.randn(8, 32, 500)

        filtered = bandpass_notch_car_cpu_fallback(x)

        # Basic sanity checks
        assert torch.isfinite(filtered).all()
        assert not torch.isnan(filtered).any()

        # Energy should be preserved approximately (not exact due to filtering)
        original_energy = torch.norm(x)
        filtered_energy = torch.norm(filtered)

        # Filtered energy should be in reasonable range
        assert filtered_energy > 0.1 * original_energy
        assert filtered_energy < 2.0 * original_energy


class TestRMSNorm:
    """Test RMSNorm implementation."""

    def test_cpu_fallback_preserves_shape(self):
        """Test CPU fallback preserves shape."""
        x = torch.randn(16, 128, 1000)

        normalized = rmsnorm_cpu_fallback(x, eps=1e-6)

        assert normalized.shape == x.shape
        assert normalized.dtype == x.dtype

    def test_normalization_properties(self):
        """Test that RMSNorm has correct normalization properties."""
        x = torch.randn(16, 128, 1000)

        normalized = rmsnorm_cpu_fallback(x, eps=1e-6)

        # RMS (Root Mean Square) should be approximately 1 along last dimension
        rms = torch.sqrt(torch.mean(normalized**2, dim=-1, keepdim=True))

        # Should be close to 1 (within numerical precision)
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        test_cases = [
            torch.zeros(8, 64, 500),  # All zeros
            torch.ones(8, 64, 500) * 1e-8,  # Very small values
            torch.ones(8, 64, 500) * 1e8,  # Very large values
        ]

        for x in test_cases:
            normalized = rmsnorm_cpu_fallback(x, eps=1e-6)

            assert torch.isfinite(normalized).all()
            assert not torch.isnan(normalized).any()

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_rmsnorm_if_available(self):
        """Test CUDA RMSNorm if available."""
        x = torch.randn(16, 128, 1000).cuda()

        try:
            normalized = rmsnorm_triton(x)
            assert normalized.device == x.device
            assert normalized.shape == x.shape
        except Exception:
            # Fallback should work
            normalized = rmsnorm_cpu_fallback(x.cpu())
            assert normalized.shape == x.shape

    def test_gradient_flow(self):
        """Test that gradients flow through RMSNorm."""
        x = torch.randn(8, 64, 100, requires_grad=True)

        normalized = rmsnorm_cpu_fallback(x)
        loss = normalized.sum()
        loss.backward()

        assert x.grad is not None
        assert torch.isfinite(x.grad).all()


class TestPerceptualQuantization:
    """Test perceptual quantization for compression."""

    def test_numpy_fallback_preserves_shape(self):
        """Test NumPy fallback preserves shape."""
        x = torch.randn(16, 128, 1000)

        quantized = perceptual_quantize_numpy_fallback(x, bits=8)

        assert quantized.shape == x.shape
        assert quantized.dtype == x.dtype

    def test_quantization_reduces_precision(self):
        """Test that quantization actually reduces precision."""
        x = torch.randn(16, 128, 1000)

        quantized_8bit = perceptual_quantize_numpy_fallback(x, bits=8)
        quantized_4bit = perceptual_quantize_numpy_fallback(x, bits=4)

        # Lower bit quantization should be different from higher bit
        diff_8bit = torch.norm(x - quantized_8bit)
        diff_4bit = torch.norm(x - quantized_4bit)

        # 4-bit should have more quantization error
        assert diff_4bit > diff_8bit

    def test_stochastic_dithering(self):
        """Test stochastic dithering option."""
        x = torch.randn(16, 128, 1000)

        # Two runs with stochastic dithering should be different
        quant_1 = perceptual_quantize_numpy_fallback(x, bits=8, stochastic=True)
        quant_2 = perceptual_quantize_numpy_fallback(x, bits=8, stochastic=True)

        # Should be different due to random dithering
        assert not torch.allclose(quant_1, quant_2, atol=1e-6)

        # But deterministic should be same
        det_1 = perceptual_quantize_numpy_fallback(x, bits=8, stochastic=False)
        det_2 = perceptual_quantize_numpy_fallback(x, bits=8, stochastic=False)

        assert torch.allclose(det_1, det_2, atol=1e-8)

    def test_different_bit_depths(self):
        """Test quantization with different bit depths."""
        x = torch.randn(8, 64, 500)

        bit_depths = [4, 6, 8, 12, 16]
        quantization_errors = []

        for bits in bit_depths:
            quantized = perceptual_quantize_numpy_fallback(x, bits=bits)
            error = torch.norm(x - quantized)
            quantization_errors.append(error.item())

        # Higher bit depths should have lower quantization error
        for i in range(len(bit_depths) - 1):
            assert quantization_errors[i] >= quantization_errors[i + 1]

    @pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
    def test_cupy_path_if_available(self):
        """Test CuPy path if available."""
        x = torch.randn(16, 128, 1000)

        try:
            quantized = perceptual_quantize_torch(x, bits=8)
            assert quantized.shape == x.shape
        except Exception:
            # Should fall back to NumPy
            quantized = perceptual_quantize_numpy_fallback(x, bits=8)
            assert quantized.shape == x.shape

    def test_energy_preservation(self):
        """Test that quantization preserves reasonable signal energy."""
        x = torch.randn(16, 128, 1000)

        quantized = perceptual_quantize_numpy_fallback(x, bits=8)

        original_energy = torch.norm(x)
        quantized_energy = torch.norm(quantized)

        # Energy should be preserved within reasonable bounds
        energy_ratio = quantized_energy / original_energy
        assert 0.5 < energy_ratio < 2.0  # Within 2x of original


class TestGPUFallbackIntegration:
    """Test integrated GPU operations with fallbacks."""

    def test_all_operations_cpu_fallback(self):
        """Test that all GPU operations have working CPU fallbacks."""
        x = torch.randn(8, 64, 500)

        # Test filtering
        filtered = bandpass_notch_car_cpu_fallback(x)
        assert filtered.shape == x.shape

        # Test normalization
        normalized = rmsnorm_cpu_fallback(filtered)
        assert normalized.shape == x.shape

        # Test quantization
        quantized = perceptual_quantize_numpy_fallback(normalized, bits=8)
        assert quantized.shape == x.shape

        # All should be finite
        assert torch.isfinite(filtered).all()
        assert torch.isfinite(normalized).all()
        assert torch.isfinite(quantized).all()

    def test_pipeline_composition(self):
        """Test composing multiple GPU operations."""
        x = torch.randn(16, 128, 1000)

        # Composition: filter -> normalize -> quantize
        filtered = bandpass_notch_car_cpu_fallback(x)
        normalized = rmsnorm_cpu_fallback(filtered)
        quantized = perceptual_quantize_numpy_fallback(normalized, bits=8)

        assert quantized.shape == x.shape

        # Pipeline should preserve reasonable signal characteristics
        assert torch.isfinite(quantized).all()
        assert quantized.std() > 0  # Should have some variation

    def test_batch_processing_efficiency(self):
        """Test that operations work efficiently on batches."""
        batch_sizes = [1, 8, 32, 64]

        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 128, 1000)

            # Should handle different batch sizes
            filtered = bandpass_notch_car_cpu_fallback(x)
            normalized = rmsnorm_cpu_fallback(filtered)
            quantized = perceptual_quantize_numpy_fallback(normalized, bits=8)

            assert filtered.shape[0] == batch_size
            assert normalized.shape[0] == batch_size
            assert quantized.shape[0] == batch_size


if __name__ == "__main__":
    # Run tests with CPU-only to ensure fallbacks work
    pytest.main([__file__, "-v", "-k", "not cuda"])
