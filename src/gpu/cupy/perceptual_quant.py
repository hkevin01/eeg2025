# -*- coding: utf-8 -*-
# File: src/gpu/cupy/perceptual_quant.py
"""
CuPy-based perceptual quantization for EEG compression augmentation.
Implements frequency-domain compression with adaptive bit allocation.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple, Union

import cupy as cp
import numpy as np


def _stft_cupy(
    x: cp.ndarray, n_fft: int = 512, hop: int = 256, win: Optional[cp.ndarray] = None
) -> cp.ndarray:
    """
    Short-time Fourier transform using CuPy.

    Args:
        x: (B, C, T) input signal
        n_fft: FFT size
        hop: Hop length
        win: Window function (default: Hann)

    Returns:
        F: (B, C, n_frames, n_fft//2+1) complex STFT
    """
    B, C, T = x.shape

    if win is None:
        win = cp.hanning(n_fft, dtype=cp.float32)

    # Number of frames
    if T >= n_fft:
        n_frames = 1 + (T - n_fft) // hop
    else:
        n_frames = 1

    # Pre-allocate output
    n_freqs = n_fft // 2 + 1
    F = cp.zeros((B, C, n_frames, n_freqs), dtype=cp.complex64)

    # Compute STFT frame by frame
    for i in range(n_frames):
        start = i * hop
        end = start + n_fft

        # Extract frame with zero padding if needed
        if end <= T:
            frame = x[:, :, start:end]
        else:
            frame = cp.zeros((B, C, n_fft), dtype=cp.float32)
            valid_len = max(0, T - start)
            if valid_len > 0:
                frame[:, :, :valid_len] = x[:, :, start : start + valid_len]

        # Apply window
        windowed = frame * win[None, None, :]

        # Compute FFT
        F[:, :, i, :] = cp.fft.rfft(windowed, axis=-1)

    return F


def _istft_cupy(
    F: cp.ndarray,
    n_fft: int = 512,
    hop: int = 256,
    win: Optional[cp.ndarray] = None,
    T_out: Optional[int] = None,
) -> cp.ndarray:
    """
    Inverse short-time Fourier transform using CuPy.

    Args:
        F: (B, C, n_frames, n_freqs) complex STFT
        n_fft: FFT size
        hop: Hop length
        win: Window function (default: Hann)
        T_out: Target output length

    Returns:
        x: (B, C, T) reconstructed signal
    """
    B, C, n_frames, n_freqs = F.shape

    if win is None:
        win = cp.hanning(n_fft, dtype=cp.float32)

    # Estimate output length
    T_est = (n_frames - 1) * hop + n_fft
    T = T_out if T_out is not None else T_est

    # Pre-allocate buffers
    x = cp.zeros((B, C, T_est), dtype=cp.float32)
    window_sum = cp.zeros(T_est, dtype=cp.float32)

    # Reconstruct overlap-add
    for i in range(n_frames):
        # Inverse FFT
        frame = cp.fft.irfft(F[:, :, i, :], n=n_fft, axis=-1)

        # Apply window
        windowed_frame = frame * win[None, None, :]

        # Overlap-add
        start = i * hop
        end = start + n_fft
        x[:, :, start:end] += windowed_frame
        window_sum[start:end] += win

    # Normalize by window sum
    window_sum = cp.maximum(window_sum, 1e-6)
    x = x / window_sum[None, None, :]

    # Trim or pad to target length
    if T_out is not None and T_est != T_out:
        if T_est > T_out:
            x = x[:, :, :T_out]
        else:
            x_padded = cp.zeros((B, C, T_out), dtype=cp.float32)
            x_padded[:, :, :T_est] = x
            x = x_padded

    return x


def perceptual_quantize(
    x: cp.ndarray,
    snr_db: float = 30.0,
    n_fft: int = 512,
    hop: int = 256,
    band_edges: Tuple[int, ...] = (1, 4, 8, 16, 32, 64, 128, 257),
    stochastic: bool = True,
    seed: Optional[int] = None,
) -> cp.ndarray:
    """
    Perceptual quantization for EEG compression augmentation.

    Applies frequency-domain quantization with adaptive bit allocation
    to simulate real-world compression artifacts while preserving
    perceptually important signal components.

    Args:
        x: (B, C, T) CuPy float32 array
        snr_db: Target signal-to-noise ratio in dB
        n_fft: FFT size for STFT
        hop: Hop length for STFT
        band_edges: Frequency band boundaries for adaptive quantization
        stochastic: Whether to add dithering noise
        seed: Random seed for reproducibility

    Returns:
        x_quantized: (B, C, T) quantized signal with similar shape
    """
    if seed is not None:
        cp.random.seed(seed)

    B, C, T = x.shape

    # Compute STFT
    F = _stft_cupy(x, n_fft=n_fft, hop=hop)

    # Extract magnitude and phase
    magnitude = cp.abs(F)
    phase = cp.angle(F)

    # Noise scale from SNR
    noise_scale = 10 ** (-snr_db / 20.0)

    # Frequency bands for adaptive quantization
    n_freqs = magnitude.shape[-1]
    edges = list(band_edges)
    edges[-1] = min(edges[-1], n_freqs)

    # Ensure we have a valid frequency range
    if edges[0] != 0:
        edges = [0] + edges

    # Apply bandwise quantization
    quantized_magnitude = cp.empty_like(magnitude)

    for i in range(len(edges) - 1):
        freq_start, freq_end = edges[i], edges[i + 1]

        if freq_start >= freq_end:
            continue

        # Extract frequency band
        band_mag = magnitude[..., freq_start:freq_end]

        # Compute RMS energy per (B, C, frame) for this band
        band_energy = cp.sqrt(cp.mean(band_mag**2, axis=-1, keepdims=True) + 1e-8)

        # Adaptive quantization step based on energy and target SNR
        quant_step = noise_scale * band_energy

        # Uniform midrise quantization
        quantized_levels = cp.floor(band_mag / (quant_step + 1e-12) + 0.5)

        # Add dithering if requested
        if stochastic:
            dither_noise = cp.random.uniform(
                -0.5, 0.5, size=quantized_levels.shape, dtype=cp.float32
            )
            quantized_levels = quantized_levels + dither_noise

        # Reconstruct quantized magnitude
        quantized_band = quantized_levels * quant_step
        quantized_magnitude[..., freq_start:freq_end] = quantized_band

    # Reconstruct complex spectrum with quantized magnitude
    F_quantized = quantized_magnitude * cp.exp(1j * phase)

    # Inverse STFT
    x_quantized = _istft_cupy(F_quantized, n_fft=n_fft, hop=hop, T_out=T)

    return x_quantized


def adaptive_wavelet_compress(
    x: cp.ndarray,
    wavelet: str = "db4",
    compression_ratio: float = 0.75,
    threshold_mode: str = "soft",
    stochastic: bool = True,
) -> cp.ndarray:
    """
    Adaptive wavelet compression for EEG signals.

    Args:
        x: (B, C, T) input signal
        wavelet: Wavelet family for decomposition
        compression_ratio: Fraction of coefficients to zero out
        threshold_mode: 'soft' or 'hard' thresholding
        stochastic: Whether to add threshold noise

    Returns:
        x_compressed: Compressed and reconstructed signal
    """
    try:
        import pywt
    except ImportError:
        raise ImportError("PyWavelets required for wavelet compression")

    B, C, T = x.shape
    x_out = cp.empty_like(x)

    # Process each signal independently
    for b in range(B):
        for c in range(C):
            # Move to CPU for PyWavelets (TODO: use CuPy wavelets when available)
            signal_cpu = cp.asnumpy(x[b, c, :])

            # Wavelet decomposition
            coeffs = pywt.wavedec(signal_cpu, wavelet, mode="symmetric")

            # Flatten all coefficients except approximation
            detail_coeffs = cp.asarray(np.concatenate(coeffs[1:]))

            # Compute adaptive threshold
            threshold = cp.percentile(cp.abs(detail_coeffs), compression_ratio * 100)

            # Add stochastic noise to threshold if requested
            if stochastic:
                noise_scale = threshold * 0.1
                threshold += cp.random.normal(0, noise_scale)

            # Apply thresholding to detail coefficients
            coeffs_thresh = list(coeffs)
            detail_start = 0

            for i in range(1, len(coeffs)):
                detail_len = len(coeffs[i])
                detail_band = detail_coeffs[detail_start : detail_start + detail_len]

                if threshold_mode == "soft":
                    # Soft thresholding
                    detail_thresh = cp.sign(detail_band) * cp.maximum(
                        cp.abs(detail_band) - threshold, 0
                    )
                else:
                    # Hard thresholding
                    detail_thresh = detail_band * (cp.abs(detail_band) > threshold)

                coeffs_thresh[i] = cp.asnumpy(detail_thresh)
                detail_start += detail_len

            # Wavelet reconstruction
            reconstructed = pywt.waverec(coeffs_thresh, wavelet, mode="symmetric")

            # Handle length mismatch
            if len(reconstructed) != T:
                if len(reconstructed) > T:
                    reconstructed = reconstructed[:T]
                else:
                    padded = np.zeros(T)
                    padded[: len(reconstructed)] = reconstructed
                    reconstructed = padded

            x_out[b, c, :] = cp.asarray(reconstructed)

    return x_out


def predictive_coding_residual(
    x: cp.ndarray,
    predictor_order: int = 8,
    learning_rate: float = 0.01,
    add_noise: bool = True,
    noise_std: float = 0.02,
) -> cp.ndarray:
    """
    Predictive coding with adaptive linear prediction.

    Args:
        x: (B, C, T) input signal
        predictor_order: Number of prediction taps
        learning_rate: LMS adaptation rate
        add_noise: Whether to add prediction noise
        noise_std: Standard deviation of prediction noise

    Returns:
        residual: (B, C, T) prediction residual signal
    """
    B, C, T = x.shape
    P = predictor_order

    if T <= P:
        # Not enough samples for prediction
        return x

    residual = cp.zeros_like(x)

    # Process each signal independently
    for b in range(B):
        for c in range(C):
            signal = x[b, c, :]

            # Initialize predictor coefficients
            coeffs = cp.zeros(P, dtype=cp.float32)

            # Adaptive linear prediction using LMS
            for t in range(P, T):
                # Prediction input (past samples)
                x_vec = signal[t - P : t][::-1]  # Reverse for convolution

                # Prediction
                prediction = cp.dot(coeffs, x_vec)

                # Prediction error
                error = signal[t] - prediction
                residual[b, c, t] = error

                # Update coefficients (LMS)
                coeffs += learning_rate * error * x_vec

            # Copy initial samples that couldn't be predicted
            residual[b, c, :P] = signal[:P]

            # Add prediction noise if requested
            if add_noise:
                noise = cp.random.normal(0, noise_std, size=T).astype(cp.float32)
                residual[b, c, :] += noise

    return residual


def perceptual_quantize_torch(
    x_torch: "torch.Tensor",
    snr_db: float = 30.0,
    n_fft: int = 512,
    hop: int = 256,
    band_edges: Tuple[int, ...] = (1, 4, 8, 16, 32, 64, 128, 257),
    stochastic: bool = True,
    seed: Optional[int] = None,
) -> "torch.Tensor":
    """
    PyTorch wrapper for perceptual quantization.

    Converts PyTorch CUDA tensors to CuPy without host copies using DLPack,
    applies perceptual quantization, and converts back to PyTorch.

    Args:
        x_torch: (B, C, T) PyTorch CUDA tensor (float32)
        snr_db: Target SNR in dB
        n_fft: FFT size
        hop: Hop length
        band_edges: Frequency band boundaries
        stochastic: Whether to add dithering
        seed: Random seed

    Returns:
        quantized_torch: (B, C, T) PyTorch tensor with compression artifacts
    """
    import torch

    if not x_torch.is_cuda:
        raise RuntimeError("Input tensor must be on CUDA device")

    if x_torch.dtype != torch.float32:
        x_torch = x_torch.float()

    # Convert PyTorch -> CuPy via DLPack (zero-copy)
    try:
        cp_x = cp.fromDlpack(torch.utils.dlpack.to_dlpack(x_torch))
    except Exception as e:
        raise RuntimeError(f"Failed to convert PyTorch to CuPy: {e}")

    # Apply perceptual quantization
    cp_y = perceptual_quantize(
        cp_x,
        snr_db=snr_db,
        n_fft=n_fft,
        hop=hop,
        band_edges=band_edges,
        stochastic=stochastic,
        seed=seed,
    )

    # Convert CuPy -> PyTorch via DLPack (zero-copy)
    try:
        y_torch = torch.utils.dlpack.from_dlpack(cp_y.toDlpack())
    except Exception as e:
        raise RuntimeError(f"Failed to convert CuPy to PyTorch: {e}")

    return y_torch


def compression_augmentation_suite(
    x_torch: "torch.Tensor",
    compression_config: Optional[dict] = None,
) -> "torch.Tensor":
    """
    Complete compression augmentation suite combining multiple techniques.

    Args:
        x_torch: (B, C, T) PyTorch CUDA tensor
        compression_config: Configuration dict with compression parameters

    Returns:
        augmented: Compressed and distorted signal for robust training
    """
    import torch

    if compression_config is None:
        compression_config = {
            "perceptual_quant": {
                "snr_db_range": (25, 35),
                "stochastic": True,
            },
            "wavelet_compress": {
                "compression_ratio_range": (0.5, 0.8),
                "stochastic": True,
            },
            "predictive_coding": {
                "predictor_order": 8,
                "add_noise": True,
                "noise_std": 0.01,
            },
        }

    # Convert to CuPy
    cp_x = cp.fromDlpack(torch.utils.dlpack.to_dlpack(x_torch.float()))

    # Apply random compression techniques
    techniques = []

    # Perceptual quantization
    if "perceptual_quant" in compression_config and cp.random.rand() > 0.5:
        pq_config = compression_config["perceptual_quant"]
        snr_range = pq_config.get("snr_db_range", (25, 35))
        snr_db = float(cp.random.uniform(snr_range[0], snr_range[1]))
        cp_x = perceptual_quantize(
            cp_x, snr_db=snr_db, stochastic=pq_config.get("stochastic", True)
        )
        techniques.append(f"perceptual_quant(snr={snr_db:.1f}dB)")

    # Wavelet compression
    if "wavelet_compress" in compression_config and cp.random.rand() > 0.5:
        wc_config = compression_config["wavelet_compress"]
        ratio_range = wc_config.get("compression_ratio_range", (0.5, 0.8))
        ratio = float(cp.random.uniform(ratio_range[0], ratio_range[1]))
        cp_x = adaptive_wavelet_compress(
            cp_x, compression_ratio=ratio, stochastic=wc_config.get("stochastic", True)
        )
        techniques.append(f"wavelet_compress(ratio={ratio:.2f})")

    # Predictive coding residual
    if "predictive_coding" in compression_config and cp.random.rand() > 0.3:
        pc_config = compression_config["predictive_coding"]
        cp_x = predictive_coding_residual(
            cp_x,
            predictor_order=pc_config.get("predictor_order", 8),
            add_noise=pc_config.get("add_noise", True),
            noise_std=pc_config.get("noise_std", 0.01),
        )
        techniques.append("predictive_coding")

    # Convert back to PyTorch
    result = torch.utils.dlpack.from_dlpack(cp_x.toDlpack())

    # Store applied techniques as metadata (for debugging)
    if hasattr(result, "_compression_techniques"):
        result._compression_techniques = techniques

    return result


# CPU fallback functions
def perceptual_quantize_cpu_fallback(
    x: np.ndarray,
    snr_db: float = 30.0,
    n_fft: int = 512,
    hop: int = 256,
) -> np.ndarray:
    """CPU fallback for perceptual quantization when CuPy unavailable."""
    try:
        from scipy.signal import istft, stft
    except ImportError:
        warnings.warn("SciPy not available, returning original signal")
        return x

    B, C, T = x.shape
    result = np.empty_like(x)

    for b in range(B):
        for c in range(C):
            # STFT
            f, t, X = stft(x[b, c, :], nperseg=n_fft, noverlap=n_fft - hop)

            # Simple quantization (less sophisticated than CuPy version)
            magnitude = np.abs(X)
            phase = np.angle(X)

            # Global quantization step
            noise_scale = 10 ** (-snr_db / 20.0)
            rms = np.sqrt(np.mean(magnitude**2))
            quant_step = noise_scale * rms

            quantized_mag = np.round(magnitude / quant_step) * quant_step
            quantized_X = quantized_mag * np.exp(1j * phase)

            # ISTFT
            _, reconstructed = istft(quantized_X, nperseg=n_fft, noverlap=n_fft - hop)

            # Handle length mismatch
            if len(reconstructed) != T:
                if len(reconstructed) > T:
                    reconstructed = reconstructed[:T]
                else:
                    padded = np.zeros(T)
                    padded[: len(reconstructed)] = reconstructed
                    reconstructed = padded

            result[b, c, :] = reconstructed

    return result
