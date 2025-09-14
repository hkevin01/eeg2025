"""
Enhanced SSL augmentation pipeline for EEG data with compression-aware techniques.

This module implements various augmentation techniques for self-supervised learning
including time masking, channel dropout, temporal jitter, compression distortions,
and schedulable parameters for curriculum learning.
"""

import math
import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal


class SchedulableAugmentation(nn.Module):
    """
    Base class for augmentations with schedulable parameters.

    Enables dynamic adjustment of augmentation intensity during training.
    """

    def __init__(
        self,
        initial_param: float,
        final_param: float,
        param_name: str,
        schedule_strategy: str = "linear",
    ):
        """
        Initialize schedulable augmentation.

        Args:
            initial_param: Initial parameter value
            final_param: Final parameter value
            param_name: Name of the parameter to schedule
            schedule_strategy: Scheduling strategy ("linear", "cosine", "exponential")
        """
        super().__init__()
        self.initial_param = initial_param
        self.final_param = final_param
        self.param_name = param_name
        self.schedule_strategy = schedule_strategy
        self.current_step = 0
        self.total_steps = 1000  # Default, should be set by scheduler

        # Set initial parameter
        setattr(self, param_name, initial_param)

    def update_parameter(self, step: int, total_steps: int):
        """Update parameter based on training progress."""
        self.current_step = step
        self.total_steps = total_steps

        progress = min(step / total_steps, 1.0)

        if self.schedule_strategy == "linear":
            param_value = self.initial_param + progress * (
                self.final_param - self.initial_param
            )
        elif self.schedule_strategy == "cosine":
            param_value = (
                self.initial_param
                + (self.final_param - self.initial_param)
                * (1 - math.cos(math.pi * progress))
                / 2
            )
        elif self.schedule_strategy == "exponential":
            param_value = self.initial_param + (
                self.final_param - self.initial_param
            ) * (1 - math.exp(-5 * progress))
        else:
            param_value = self.final_param

        setattr(self, self.param_name, param_value)


class TimeMasking(SchedulableAugmentation):
    """
    Time masking augmentation for EEG signals with schedulable mask ratio.

    Randomly masks contiguous time segments to encourage temporal robustness.
    """

    def __init__(
        self,
        initial_mask_ratio: float = 0.1,
        final_mask_ratio: float = 0.3,
        mask_value: float = 0.0,
        schedule_strategy: str = "linear",
    ):
        super().__init__(
            initial_mask_ratio, final_mask_ratio, "mask_ratio", schedule_strategy
        )
        self.mask_value = mask_value

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply time masking to input signal.

        Args:
            x: Input EEG signal [batch_size, n_channels, seq_len]

        Returns:
            Time-masked signal and mask tensor
        """
        batch_size, n_channels, seq_len = x.shape
        mask = torch.ones_like(x, dtype=torch.bool)

        if self.training:
            x_masked = x.clone()

            for i in range(batch_size):
                # Random mask length
                mask_len = int(seq_len * self.mask_ratio * np.random.uniform(0.5, 1.5))
                mask_len = min(mask_len, seq_len)

                if mask_len > 0:
                    # Random start position
                    start_idx = np.random.randint(0, seq_len - mask_len + 1)

                    # Apply mask
                    x_masked[i, :, start_idx : start_idx + mask_len] = self.mask_value
                    mask[i, :, start_idx : start_idx + mask_len] = False

            return x_masked, mask

        return x, mask


class CompressionDistortion(SchedulableAugmentation):
    """
    Compression-aware distortion using wavelet compression and perceptual quantization.

    Simulates realistic compression artifacts that models might encounter in deployment.
    """

    def __init__(
        self,
        initial_distortion: float = 0.25,
        final_distortion: float = 1.5,
        wavelet: str = "db4",
        quantization_levels: int = 256,
        snr_db_range: Tuple[float, float] = (25.0, 35.0),
        schedule_strategy: str = "cosine",
    ):
        """
        Initialize compression distortion.

        Args:
            initial_distortion: Initial distortion percentage
            final_distortion: Final distortion percentage
            wavelet: Wavelet type for compression
            quantization_levels: Number of quantization levels
            snr_db_range: SNR range for perceptual quantization
            schedule_strategy: Parameter scheduling strategy
        """
        super().__init__(
            initial_distortion, final_distortion, "distortion_pct", schedule_strategy
        )
        self.wavelet = wavelet
        self.quantization_levels = quantization_levels
        self.snr_db_range = snr_db_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply compression distortion.

        Args:
            x: Input EEG signal [batch_size, n_channels, seq_len]

        Returns:
            Compression-distorted signal
        """
        if not self.training or self.distortion_pct == 0:
            return x

        batch_size, n_channels, seq_len = x.shape
        x_distorted = x.clone()

        for i in range(batch_size):
            for j in range(n_channels):
                signal_1d = x[i, j].cpu().numpy()

                # Apply distortion with probability
                if np.random.random() < 0.7:  # 70% chance of distortion
                    # Choose distortion type
                    distortion_type = np.random.choice(
                        ["wavelet", "quantization", "both"], p=[0.4, 0.4, 0.2]
                    )

                    if distortion_type in ["wavelet", "both"]:
                        signal_1d = self._apply_wavelet_compression(signal_1d)

                    if distortion_type in ["quantization", "both"]:
                        signal_1d = self._apply_perceptual_quantization(signal_1d)

                x_distorted[i, j] = torch.tensor(
                    signal_1d, dtype=x.dtype, device=x.device
                )

        return x_distorted

    def _apply_wavelet_compression(self, signal: np.ndarray) -> np.ndarray:
        """Apply wavelet compression with coefficient thresholding."""
        try:
            # Wavelet decomposition
            coeffs = pywt.wavedec(signal, self.wavelet, level=4)

            # Calculate threshold based on distortion percentage
            all_coeffs = np.concatenate(
                [c.flatten() for c in coeffs[1:]]
            )  # Skip approximation
            threshold = np.percentile(np.abs(all_coeffs), self.distortion_pct * 100)

            # Apply soft thresholding
            coeffs_thresh = [coeffs[0]]  # Keep approximation coefficients
            for detail_coeffs in coeffs[1:]:
                coeffs_thresh.append(
                    pywt.threshold(detail_coeffs, threshold, mode="soft")
                )

            # Reconstruct signal
            reconstructed = pywt.waverec(coeffs_thresh, self.wavelet)

            # Ensure same length as input
            if len(reconstructed) != len(signal):
                reconstructed = reconstructed[: len(signal)]

            return reconstructed

        except Exception:
            # Fallback to original signal if wavelet fails
            return signal

    def _apply_perceptual_quantization(self, signal: np.ndarray) -> np.ndarray:
        """Apply perceptual quantization with controlled SNR."""
        # Calculate signal power
        signal_power = np.mean(signal**2)

        # Target SNR (random within range)
        target_snr_db = np.random.uniform(*self.snr_db_range)
        target_snr_linear = 10 ** (target_snr_db / 10)

        # Calculate quantization step size for target SNR
        quantization_noise_power = signal_power / target_snr_linear
        quantization_step = np.sqrt(12 * quantization_noise_power)

        # Apply uniform quantization
        quantized = np.round(signal / quantization_step) * quantization_step

        # Add small amount of dither to break limit cycles
        dither = np.random.uniform(
            -quantization_step / 2, quantization_step / 2, size=signal.shape
        )
        quantized += dither * 0.1

        return quantized


class ChannelDropout(SchedulableAugmentation):
    """
    Channel dropout augmentation for EEG signals with schedulable dropout rate.

    Randomly sets entire channels to zero to encourage channel-robust representations.
    """

    def __init__(
        self,
        initial_dropout: float = 0.05,
        final_dropout: float = 0.2,
        schedule_strategy: str = "linear",
    ):
        super().__init__(
            initial_dropout, final_dropout, "dropout_prob", schedule_strategy
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel dropout to input signal.

        Args:
            x: Input EEG signal [batch_size, n_channels, seq_len]

        Returns:
            Channel-dropped signal
        """
        if self.training and self.dropout_prob > 0:
            batch_size, n_channels, seq_len = x.shape

            # Create channel dropout mask
            mask = torch.bernoulli(
                torch.full((batch_size, n_channels, 1), 1 - self.dropout_prob)
            ).to(x.device)

            return x * mask

        return x


class TemporalJitter(nn.Module):
    """
    Temporal jitter augmentation for EEG signals.

    Adds small random time shifts to encourage temporal invariance.
    """

    def __init__(self, jitter_std: float = 0.02, max_shift: Optional[int] = None):
        super().__init__()
        self.jitter_std = jitter_std
        self.max_shift = max_shift

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply temporal jitter to input signal.

        Args:
            x: Input EEG signal [batch_size, n_channels, seq_len]

        Returns:
            Temporally jittered signal
        """
        if not self.training:
            return x

        batch_size, n_channels, seq_len = x.shape

        # Determine max shift
        if self.max_shift is None:
            max_shift = int(seq_len * self.jitter_std)
        else:
            max_shift = min(self.max_shift, seq_len // 10)

        if max_shift <= 0:
            return x

        x_jittered = x.clone()

        for i in range(batch_size):
            # Random shift amount
            shift = np.random.randint(-max_shift, max_shift + 1)

            if shift != 0:
                if shift > 0:
                    # Shift right (pad left)
                    x_jittered[i, :, shift:] = x[i, :, :-shift]
                    x_jittered[i, :, :shift] = x[i, :, 0:1].expand(-1, shift)
                else:
                    # Shift left (pad right)
                    shift = -shift
                    x_jittered[i, :, :-shift] = x[i, :, shift:]
                    x_jittered[i, :, -shift:] = x[i, :, -1:].expand(-1, shift)

        return x_jittered


class WaveletDistortion(nn.Module):
    """
    Wavelet-based compression distortion for EEG signals.

    Applies wavelet compression and reconstruction to simulate data compression artifacts.
    """

    def __init__(
        self,
        wavelet: str = "db4",
        distortion_pct: float = 1.0,
        compression_ratio: float = 0.1,
    ):
        super().__init__()
        self.wavelet = wavelet
        self.distortion_pct = distortion_pct
        self.compression_ratio = compression_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply wavelet distortion to input signal.

        Args:
            x: Input EEG signal [batch_size, n_channels, seq_len]

        Returns:
            Wavelet-distorted signal
        """
        if not self.training or self.distortion_pct <= 0:
            return x

        # Apply distortion to random subset of batch
        apply_mask = torch.rand(x.shape[0]) < self.distortion_pct

        if not apply_mask.any():
            return x

        x_distorted = x.clone()

        # Convert to numpy for wavelet processing
        x_np = x.detach().cpu().numpy()

        for i in range(x.shape[0]):
            if apply_mask[i]:
                for ch in range(x.shape[1]):
                    signal = x_np[i, ch, :]

                    try:
                        # Wavelet decomposition
                        coeffs = pywt.wavedec(signal, self.wavelet, level=4)

                        # Compression: zero out smallest coefficients
                        for j in range(len(coeffs)):
                            if j > 0:  # Keep approximation coefficients
                                threshold = np.percentile(
                                    np.abs(coeffs[j]),
                                    (1 - self.compression_ratio) * 100,
                                )
                                coeffs[j] = coeffs[j] * (np.abs(coeffs[j]) >= threshold)

                        # Reconstruction
                        reconstructed = pywt.waverec(coeffs, self.wavelet)

                        # Handle length mismatch
                        if len(reconstructed) != len(signal):
                            if len(reconstructed) > len(signal):
                                reconstructed = reconstructed[: len(signal)]
                            else:
                                # Pad with zeros
                                pad_len = len(signal) - len(reconstructed)
                                reconstructed = np.pad(
                                    reconstructed, (0, pad_len), "constant"
                                )

                        x_distorted[i, ch, :] = torch.from_numpy(reconstructed).to(
                            x.device
                        )

                    except Exception:
                        # Skip distortion if wavelet processing fails
                        continue

        return x_distorted


class PerceptualQuantization(nn.Module):
    """
    Perceptual quantization for EEG compression simulation.

    Simulates lossy compression by quantizing signal values based on perceptual models.
    """

    def __init__(self, snr_db: float = 25.0, quantization_levels: int = 256):
        super().__init__()
        self.snr_db = snr_db
        self.quantization_levels = quantization_levels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply perceptual quantization to input signal.

        Args:
            x: Input EEG signal [batch_size, n_channels, seq_len]

        Returns:
            Quantized signal
        """
        if not self.training:
            return x

        # Calculate noise power from SNR
        signal_power = torch.mean(x**2)
        noise_power = signal_power / (10 ** (self.snr_db / 10))
        noise_std = torch.sqrt(noise_power)

        # Add quantization noise
        noise = torch.randn_like(x) * noise_std
        x_noisy = x + noise

        # Quantization
        x_range = torch.max(x_noisy) - torch.min(x_noisy)
        if x_range > 0:
            quantized = torch.round(
                (x_noisy - torch.min(x_noisy))
                / x_range
                * (self.quantization_levels - 1)
            )
            quantized = quantized / (
                self.quantization_levels - 1
            ) * x_range + torch.min(x_noisy)
        else:
            quantized = x_noisy

        return quantized


class GaussianNoise(nn.Module):
    """
    Gaussian noise augmentation for EEG signals.
    """

    def __init__(self, noise_std: float = 0.01):
        super().__init__()
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply Gaussian noise."""
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            return x + noise
        return x


class FrequencyMasking(nn.Module):
    """
    Frequency domain masking for EEG signals.

    Masks frequency bands to encourage frequency-robust representations.
    """

    def __init__(self, mask_ratio: float = 0.1, n_freq_bands: int = 8):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.n_freq_bands = n_freq_bands

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply frequency masking to input signal.

        Args:
            x: Input EEG signal [batch_size, n_channels, seq_len]

        Returns:
            Frequency-masked signal
        """
        if not self.training:
            return x

        # FFT to frequency domain
        x_fft = torch.fft.rfft(x, dim=-1)

        # Determine frequency bands to mask
        n_freqs = x_fft.shape[-1]
        band_size = n_freqs // self.n_freq_bands

        n_bands_to_mask = int(self.n_freq_bands * self.mask_ratio)

        for i in range(x.shape[0]):
            # Randomly select bands to mask
            bands_to_mask = random.sample(range(self.n_freq_bands), n_bands_to_mask)

            for band in bands_to_mask:
                start_freq = band * band_size
                end_freq = min((band + 1) * band_size, n_freqs)
                x_fft[i, :, start_freq:end_freq] = 0

        # IFFT back to time domain
        x_masked = torch.fft.irfft(x_fft, n=x.shape[-1], dim=-1)

        return x_masked


class SSLViewPipeline(nn.Module):
    """
    Complete SSL view generation pipeline for EEG data.

    Combines multiple augmentation techniques to create diverse views for contrastive learning.
    """

    def __init__(
        self,
        time_masking_ratio: float = 0.15,
        channel_dropout: float = 0.1,
        temporal_jitter_std: float = 0.02,
        wavelet_distortion_pct: float = 1.0,
        quant_snr_db: float = 25.0,
        gaussian_noise_std: float = 0.005,
        freq_masking_ratio: float = 0.1,
        apply_prob: float = 0.8,
    ):
        super().__init__()

        self.apply_prob = apply_prob

        # Initialize augmentation modules
        self.time_masking = TimeMasking(mask_ratio=time_masking_ratio)
        self.channel_dropout = ChannelDropout(dropout_prob=channel_dropout)
        self.temporal_jitter = TemporalJitter(jitter_std=temporal_jitter_std)
        self.wavelet_distortion = WaveletDistortion(
            distortion_pct=wavelet_distortion_pct
        )
        self.perceptual_quant = PerceptualQuantization(snr_db=quant_snr_db)
        self.gaussian_noise = GaussianNoise(noise_std=gaussian_noise_std)
        self.freq_masking = FrequencyMasking(mask_ratio=freq_masking_ratio)

        # List of available augmentations
        self.augmentations = [
            self.time_masking,
            self.channel_dropout,
            self.temporal_jitter,
            self.wavelet_distortion,
            self.perceptual_quant,
            self.gaussian_noise,
            self.freq_masking,
        ]

    def forward(
        self,
        x: torch.Tensor,
        distortion_pct: Optional[float] = None,
        n_augmentations: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply random augmentations to create a view.

        Args:
            x: Input EEG signal [batch_size, n_channels, seq_len]
            distortion_pct: Override distortion percentage
            n_augmentations: Number of augmentations to apply (random if None)

        Returns:
            Augmented view of the input signal
        """
        if not self.training:
            return x

        # Update distortion percentage if provided
        if distortion_pct is not None:
            self.wavelet_distortion.distortion_pct = distortion_pct

        # Determine number of augmentations to apply
        if n_augmentations is None:
            n_augmentations = np.random.randint(1, min(4, len(self.augmentations) + 1))

        # Randomly select augmentations
        selected_augs = np.random.choice(
            self.augmentations, size=n_augmentations, replace=False
        )

        # Apply selected augmentations
        x_aug = x
        for aug in selected_augs:
            if np.random.rand() < self.apply_prob:
                x_aug = aug(x_aug)

        return x_aug

    def update_parameters(self, **kwargs):
        """Update augmentation parameters dynamically."""
        if "time_masking_ratio" in kwargs:
            self.time_masking.mask_ratio = kwargs["time_masking_ratio"]

        if "channel_dropout" in kwargs:
            self.channel_dropout.dropout_prob = kwargs["channel_dropout"]

        if "temporal_jitter_std" in kwargs:
            self.temporal_jitter.jitter_std = kwargs["temporal_jitter_std"]

        if "wavelet_distortion_pct" in kwargs:
            self.wavelet_distortion.distortion_pct = kwargs["wavelet_distortion_pct"]

        if "quant_snr_db" in kwargs:
            self.perceptual_quant.snr_db = kwargs["quant_snr_db"]

        if "gaussian_noise_std" in kwargs:
            self.gaussian_noise.noise_std = kwargs["gaussian_noise_std"]

        if "freq_masking_ratio" in kwargs:
            self.freq_masking.mask_ratio = kwargs["freq_masking_ratio"]


def create_ssl_views(
    x: torch.Tensor, view_pipeline: SSLViewPipeline, n_views: int = 2
) -> List[torch.Tensor]:
    """
    Create multiple views of the input signal for SSL.

    Args:
        x: Input EEG signal [batch_size, n_channels, seq_len]
        view_pipeline: SSL view pipeline
        n_views: Number of views to create

    Returns:
        List of augmented views
    """
    views = []
    for _ in range(n_views):
        view = view_pipeline(x)
        views.append(view)

    return views
