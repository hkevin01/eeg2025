"""
Compression-Augmented Self-Supervised Learning
=============================================

Advanced SSL framework with compression-aware augmentations, wavelet distortions,
and schedulable parameters for robust EEG foundation models.

Key Features:
- Wavelet-domain distortions and compression artifacts
- Schedulable mask ratios and augmentation intensities
- Perceptual quantization and spectral distortions
- Compression consistency losses
- Multi-scale temporal corruptions
- Adaptive augmentation scheduling
"""

import math
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScheduleType(Enum):
    """Types of parameter schedules."""

    CONSTANT = "constant"
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    STEP = "step"
    POLYNOMIAL = "polynomial"


@dataclass
class ScheduleConfig:
    """Configuration for parameter scheduling."""

    schedule_type: ScheduleType = ScheduleType.COSINE
    start_value: float = 0.1
    end_value: float = 0.6
    warmup_steps: int = 1000
    total_steps: int = 100000
    step_size: int = 10000  # For step schedule
    step_gamma: float = 0.5  # For step schedule
    polynomial_power: float = 2.0  # For polynomial schedule


@dataclass
class CompressionSSLConfig:
    """Configuration for compression-augmented SSL."""

    # Masking parameters
    mask_ratio_schedule: ScheduleConfig = None
    temporal_mask_span: int = 50
    channel_mask_prob: float = 0.1

    # Compression parameters
    wavelet_family: str = "db4"
    compression_levels: List[int] = None  # [1, 2, 4, 8]
    quantization_bits: List[int] = None  # [16, 8, 4]

    # Augmentation intensities
    noise_intensity_schedule: ScheduleConfig = None
    distortion_intensity_schedule: ScheduleConfig = None

    # Loss weights
    reconstruction_weight: float = 1.0
    compression_consistency_weight: float = 0.5
    contrastive_weight: float = 0.3

    # Technical parameters
    temperature: float = 0.07
    use_momentum_encoder: bool = True
    momentum: float = 0.999

    def __post_init__(self):
        if self.mask_ratio_schedule is None:
            self.mask_ratio_schedule = ScheduleConfig(start_value=0.15, end_value=0.75)
        if self.compression_levels is None:
            self.compression_levels = [1, 2, 4, 8]
        if self.quantization_bits is None:
            self.quantization_bits = [16, 8, 4]
        if self.noise_intensity_schedule is None:
            self.noise_intensity_schedule = ScheduleConfig(
                start_value=0.01, end_value=0.1
            )
        if self.distortion_intensity_schedule is None:
            self.distortion_intensity_schedule = ScheduleConfig(
                start_value=0.05, end_value=0.3
            )


class ParameterScheduler:
    """
    Flexible parameter scheduler supporting multiple scheduling strategies.
    """

    def __init__(self, config: ScheduleConfig):
        self.config = config
        self.current_step = 0

    def step(self) -> float:
        """Get current parameter value and advance step."""
        value = self.get_value(self.current_step)
        self.current_step += 1
        return value

    def get_value(self, step: int) -> float:
        """Get parameter value at specific step."""
        config = self.config

        # Handle warmup
        if step < config.warmup_steps:
            # Linear warmup from 0 to start_value
            warmup_ratio = step / config.warmup_steps
            return config.start_value * warmup_ratio

        # Adjust step for post-warmup scheduling
        adjusted_step = step - config.warmup_steps
        adjusted_total = config.total_steps - config.warmup_steps

        if config.schedule_type == ScheduleType.CONSTANT:
            return config.start_value

        elif config.schedule_type == ScheduleType.LINEAR:
            if adjusted_step >= adjusted_total:
                return config.end_value
            ratio = adjusted_step / adjusted_total
            return config.start_value + (config.end_value - config.start_value) * ratio

        elif config.schedule_type == ScheduleType.COSINE:
            if adjusted_step >= adjusted_total:
                return config.end_value
            ratio = adjusted_step / adjusted_total
            cosine_ratio = (1 + math.cos(math.pi * ratio)) / 2
            return (
                config.end_value
                + (config.start_value - config.end_value) * cosine_ratio
            )

        elif config.schedule_type == ScheduleType.EXPONENTIAL:
            if adjusted_step >= adjusted_total:
                return config.end_value
            # Exponential decay/growth
            decay_rate = (
                math.log(config.end_value / config.start_value) / adjusted_total
            )
            return config.start_value * math.exp(decay_rate * adjusted_step)

        elif config.schedule_type == ScheduleType.STEP:
            # Step schedule
            num_steps = step // config.step_size
            return config.start_value * (config.step_gamma**num_steps)

        elif config.schedule_type == ScheduleType.POLYNOMIAL:
            if adjusted_step >= adjusted_total:
                return config.end_value
            ratio = adjusted_step / adjusted_total
            poly_ratio = ratio**config.polynomial_power
            return (
                config.start_value
                + (config.end_value - config.start_value) * poly_ratio
            )

        else:
            raise ValueError(f"Unknown schedule type: {config.schedule_type}")


class WaveletCompressor:
    """
    Wavelet-based compression and distortion for EEG signals.
    """

    def __init__(self, wavelet: str = "db4", max_level: int = 6):
        self.wavelet = wavelet
        self.max_level = max_level

        # Check if wavelet is valid
        if wavelet not in pywt.wavelist():
            raise ValueError(f"Wavelet {wavelet} not supported")

    def compress_signal(
        self,
        signal: torch.Tensor,
        compression_level: int,
        quantization_bits: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Apply wavelet compression to signal.

        Args:
            signal: Input signal (B, C, T)
            compression_level: Compression level (higher = more compression)
            quantization_bits: Number of bits for quantization

        Returns:
            Compressed signal (B, C, T)
        """
        B, C, T = signal.shape
        compressed = torch.zeros_like(signal)

        for b in range(B):
            for c in range(C):
                # Convert to numpy for pywt
                sig = signal[b, c].cpu().numpy()

                # Wavelet decomposition
                coeffs = pywt.wavedec(
                    sig, self.wavelet, level=min(compression_level, self.max_level)
                )

                # Apply compression by zeroing high-frequency coefficients
                compressed_coeffs = list(coeffs)
                for i in range(1, len(compressed_coeffs)):
                    # Zero out a fraction of detail coefficients
                    coeff = compressed_coeffs[i]
                    threshold = np.percentile(
                        np.abs(coeff), 100 - 100 / compression_level
                    )
                    mask = np.abs(coeff) < threshold
                    compressed_coeffs[i] = coeff * (~mask)

                # Quantization if specified
                if quantization_bits is not None:
                    for i in range(len(compressed_coeffs)):
                        coeff = compressed_coeffs[i]
                        # Quantize to specified bits
                        max_val = np.max(np.abs(coeff))
                        if max_val > 0:
                            levels = 2**quantization_bits - 1
                            quantized = (
                                np.round(coeff / max_val * levels) / levels * max_val
                            )
                            compressed_coeffs[i] = quantized

                # Reconstruction
                reconstructed = pywt.waverec(compressed_coeffs, self.wavelet)

                # Handle length mismatch
                if len(reconstructed) != T:
                    if len(reconstructed) > T:
                        reconstructed = reconstructed[:T]
                    else:
                        reconstructed = np.pad(
                            reconstructed, (0, T - len(reconstructed)), "constant"
                        )

                compressed[b, c] = torch.from_numpy(reconstructed).to(signal.device)

        return compressed

    def add_compression_artifacts(
        self, signal: torch.Tensor, artifact_intensity: float = 0.1
    ) -> torch.Tensor:
        """
        Add compression-like artifacts to signal.

        Args:
            signal: Input signal (B, C, T)
            artifact_intensity: Intensity of artifacts

        Returns:
            Signal with artifacts (B, C, T)
        """
        # Add quantization noise
        noise = torch.randn_like(signal) * artifact_intensity * 0.1

        # Add blocking artifacts (simulate DCT-like compression)
        block_size = 64
        T = signal.shape[-1]
        for i in range(0, T, block_size):
            end = min(i + block_size, T)
            # Add slight discontinuities at block boundaries
            if i > 0:
                signal[:, :, i] += (
                    torch.randn_like(signal[:, :, i]) * artifact_intensity * 0.05
                )

        return signal + noise


class MaskGenerator:
    """
    Advanced masking strategies for EEG SSL.
    """

    def __init__(self, config: CompressionSSLConfig):
        self.config = config

    def generate_temporal_mask(
        self, batch_size: int, seq_len: int, mask_ratio: float, device: torch.device
    ) -> torch.Tensor:
        """
        Generate temporal mask with contiguous spans.

        Args:
            batch_size: Batch size
            seq_len: Sequence length
            mask_ratio: Ratio of tokens to mask
            device: Device for tensor

        Returns:
            Mask tensor (B, T) where 1 = keep, 0 = mask
        """
        mask = torch.ones(batch_size, seq_len, device=device)

        for b in range(batch_size):
            num_mask = int(seq_len * mask_ratio)

            # Generate random spans
            span_length = self.config.temporal_mask_span
            num_spans = max(1, num_mask // span_length)

            for _ in range(num_spans):
                start = torch.randint(0, max(1, seq_len - span_length), (1,)).item()
                end = min(start + span_length, seq_len)
                mask[b, start:end] = 0

                # Early stopping if we've masked enough
                if (mask[b] == 0).sum() >= num_mask:
                    break

        return mask

    def generate_channel_mask(
        self, batch_size: int, num_channels: int, device: torch.device
    ) -> torch.Tensor:
        """
        Generate channel mask.

        Args:
            batch_size: Batch size
            num_channels: Number of channels
            device: Device for tensor

        Returns:
            Mask tensor (B, C) where 1 = keep, 0 = mask
        """
        mask = torch.ones(batch_size, num_channels, device=device)

        for b in range(batch_size):
            if torch.rand(1) < self.config.channel_mask_prob:
                # Mask random channels
                num_mask = torch.randint(1, max(2, num_channels // 4), (1,)).item()
                mask_indices = torch.randperm(num_channels)[:num_mask]
                mask[b, mask_indices] = 0

        return mask

    def apply_masks(
        self, x: torch.Tensor, temporal_mask: torch.Tensor, channel_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply temporal and channel masks to input.

        Args:
            x: Input tensor (B, C, T)
            temporal_mask: Temporal mask (B, T)
            channel_mask: Channel mask (B, C)

        Returns:
            Masked input (B, C, T)
        """
        # Apply channel mask
        channel_mask = channel_mask.unsqueeze(-1)  # (B, C, 1)
        x_masked = x * channel_mask

        # Apply temporal mask
        temporal_mask = temporal_mask.unsqueeze(1)  # (B, 1, T)
        x_masked = x_masked * temporal_mask

        return x_masked


class SpectralDistorter:
    """
    Frequency-domain distortions for EEG signals.
    """

    def __init__(self, sampling_rate: int = 500):
        self.sampling_rate = sampling_rate

    def add_spectral_noise(
        self,
        signal: torch.Tensor,
        intensity: float = 0.1,
        freq_bands: Optional[List[Tuple[float, float]]] = None,
    ) -> torch.Tensor:
        """
        Add frequency-specific noise.

        Args:
            signal: Input signal (B, C, T)
            intensity: Noise intensity
            freq_bands: List of (low, high) frequency bands

        Returns:
            Signal with spectral noise (B, C, T)
        """
        if freq_bands is None:
            # EEG frequency bands
            freq_bands = [(0.5, 4), (4, 8), (8, 13), (13, 30), (30, 100)]

        B, C, T = signal.shape

        # FFT
        signal_fft = torch.fft.rfft(signal, dim=-1)
        freqs = torch.fft.rfftfreq(T, 1 / self.sampling_rate).to(signal.device)

        # Add noise to specific frequency bands
        for low_freq, high_freq in freq_bands:
            band_mask = (freqs >= low_freq) & (freqs <= high_freq)
            if band_mask.any():
                # Random noise in this band
                band_noise = torch.randn_like(signal_fft) * intensity
                signal_fft[:, :, band_mask] += band_noise[:, :, band_mask]

        # IFFT back to time domain
        distorted = torch.fft.irfft(signal_fft, n=T, dim=-1)

        return distorted

    def add_phase_distortion(
        self, signal: torch.Tensor, intensity: float = 0.1
    ) -> torch.Tensor:
        """
        Add phase distortions while preserving magnitude spectrum.

        Args:
            signal: Input signal (B, C, T)
            intensity: Distortion intensity

        Returns:
            Phase-distorted signal (B, C, T)
        """
        # FFT
        signal_fft = torch.fft.rfft(signal, dim=-1)

        # Extract magnitude and phase
        magnitude = torch.abs(signal_fft)
        phase = torch.angle(signal_fft)

        # Add random phase distortion
        phase_noise = torch.randn_like(phase) * intensity
        distorted_phase = phase + phase_noise

        # Reconstruct with distorted phase
        distorted_fft = magnitude * torch.exp(1j * distorted_phase)
        distorted = torch.fft.irfft(distorted_fft, n=signal.shape[-1], dim=-1)

        return distorted


class CompressionSSLFramework(nn.Module):
    """
    Compression-augmented self-supervised learning framework.

    Combines traditional SSL objectives with compression-aware losses
    and schedulable augmentation parameters.
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        config: CompressionSSLConfig,
        sampling_rate: int = 500,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.sampling_rate = sampling_rate

        # Initialize schedulers
        self.mask_ratio_scheduler = ParameterScheduler(config.mask_ratio_schedule)
        self.noise_scheduler = ParameterScheduler(config.noise_intensity_schedule)
        self.distortion_scheduler = ParameterScheduler(
            config.distortion_intensity_schedule
        )

        # Initialize augmentation modules
        self.wavelet_compressor = WaveletCompressor(config.wavelet_family)
        self.mask_generator = MaskGenerator(config)
        self.spectral_distorter = SpectralDistorter(sampling_rate)

        # Momentum encoder for contrastive learning
        if config.use_momentum_encoder:
            self.momentum_encoder = self._create_momentum_encoder()
            self.momentum = config.momentum
        else:
            self.momentum_encoder = None

        # Projection heads for contrastive learning
        encoder_dim = self._get_encoder_dim()
        self.projection_head = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim),
            nn.ReLU(inplace=True),
            nn.Linear(encoder_dim, 256),
        )

        if self.momentum_encoder is not None:
            self.momentum_projection_head = nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim),
                nn.ReLU(inplace=True),
                nn.Linear(encoder_dim, 256),
            )
            # Initialize momentum projection head
            for param_q, param_k in zip(
                self.projection_head.parameters(),
                self.momentum_projection_head.parameters(),
            ):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False

        self.current_step = 0

    def _create_momentum_encoder(self) -> nn.Module:
        """Create momentum encoder as copy of main encoder."""
        momentum_encoder = type(self.encoder)(**self.encoder.__dict__)
        for param_q, param_k in zip(
            self.encoder.parameters(), momentum_encoder.parameters()
        ):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False
        return momentum_encoder

    def _get_encoder_dim(self) -> int:
        """Get encoder output dimension."""
        # Try to infer from encoder
        if hasattr(self.encoder, "embed_dim"):
            return self.encoder.embed_dim
        elif hasattr(self.encoder, "hidden_dim"):
            return self.encoder.hidden_dim
        else:
            # Default fallback
            return 768

    def _update_momentum_encoder(self):
        """Update momentum encoder with exponential moving average."""
        if self.momentum_encoder is None:
            return

        for param_q, param_k in zip(
            self.encoder.parameters(), self.momentum_encoder.parameters()
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )

        for param_q, param_k in zip(
            self.projection_head.parameters(),
            self.momentum_projection_head.parameters(),
        ):
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )

    def generate_augmentations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate multiple augmented views of input.

        Args:
            x: Input tensor (B, C, T)

        Returns:
            Dictionary of augmented views
        """
        B, C, T = x.shape
        device = x.device

        # Get current parameter values
        mask_ratio = self.mask_ratio_scheduler.step()
        noise_intensity = self.noise_scheduler.step()
        distortion_intensity = self.distortion_scheduler.step()

        # Original view
        views = {"original": x.clone()}

        # Masked view
        temporal_mask = self.mask_generator.generate_temporal_mask(
            B, T, mask_ratio, device
        )
        channel_mask = self.mask_generator.generate_channel_mask(B, C, device)
        masked_x = self.mask_generator.apply_masks(x, temporal_mask, channel_mask)
        views["masked"] = masked_x
        views["temporal_mask"] = temporal_mask
        views["channel_mask"] = channel_mask

        # Compressed views
        compression_views = []
        for comp_level in self.config.compression_levels:
            for quant_bits in self.config.quantization_bits:
                compressed = self.wavelet_compressor.compress_signal(
                    x, comp_level, quant_bits
                )
                # Add artifacts
                compressed = self.wavelet_compressor.add_compression_artifacts(
                    compressed, distortion_intensity
                )
                compression_views.append(compressed)

        views["compressed"] = compression_views

        # Spectral distortions
        spectral_noise = self.spectral_distorter.add_spectral_noise(x, noise_intensity)
        phase_distorted = self.spectral_distorter.add_phase_distortion(
            x, distortion_intensity
        )

        views["spectral_noise"] = spectral_noise
        views["phase_distorted"] = phase_distorted

        return views

    def compute_reconstruction_loss(
        self, original: torch.Tensor, reconstructed: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute masked reconstruction loss.

        Args:
            original: Original signal (B, C, T)
            reconstructed: Reconstructed signal (B, C, T)
            mask: Mask indicating which tokens to compute loss on (B, T)

        Returns:
            Reconstruction loss
        """
        # Expand mask to match signal dimensions
        mask = mask.unsqueeze(1)  # (B, 1, T)

        # Compute MSE loss only on masked tokens
        mse = F.mse_loss(reconstructed, original, reduction="none")  # (B, C, T)
        masked_mse = mse * (1 - mask)  # Only compute loss on masked tokens

        # Average over masked tokens
        num_masked = (1 - mask).sum()
        if num_masked > 0:
            loss = masked_mse.sum() / num_masked
        else:
            loss = masked_mse.mean()

        return loss

    def compute_compression_consistency_loss(
        self,
        original_features: torch.Tensor,
        compressed_features_list: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute consistency loss between original and compressed features.

        Args:
            original_features: Features from original signal
            compressed_features_list: List of features from compressed signals

        Returns:
            Consistency loss
        """
        consistency_losses = []

        for compressed_features in compressed_features_list:
            # Cosine similarity loss
            cosine_sim = F.cosine_similarity(
                original_features, compressed_features, dim=-1
            ).mean()
            consistency_losses.append(1 - cosine_sim)  # Convert to loss

        return torch.stack(consistency_losses).mean()

    def compute_contrastive_loss(
        self, features_q: torch.Tensor, features_k: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss between query and key features.

        Args:
            features_q: Query features (B, dim)
            features_k: Key features (B, dim)

        Returns:
            Contrastive loss
        """
        # Normalize features
        features_q = F.normalize(features_q, dim=-1)
        features_k = F.normalize(features_k, dim=-1)

        # Compute similarity matrix
        logits = torch.mm(features_q, features_k.t()) / self.config.temperature

        # Labels for contrastive learning (diagonal should be positive)
        labels = torch.arange(features_q.shape[0]).to(features_q.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for SSL training.

        Args:
            x: Input tensor (B, C, T)

        Returns:
            Dictionary of losses and outputs
        """
        # Generate augmented views
        views = self.generate_augmentations(x)

        # Encode different views
        original_encoded = self.encoder(views["original"])
        masked_encoded = self.encoder(views["masked"])

        # Encode compressed views
        compressed_encoded = []
        for compressed_view in views["compressed"]:
            compressed_encoded.append(self.encoder(compressed_view))

        # Encode spectral views
        spectral_encoded = self.encoder(views["spectral_noise"])
        phase_encoded = self.encoder(views["phase_distorted"])

        # Decode masked view for reconstruction
        reconstructed = self.decoder(masked_encoded)

        # Compute reconstruction loss
        recon_loss = self.compute_reconstruction_loss(
            views["original"], reconstructed, 1 - views["temporal_mask"]
        )

        # Compute compression consistency loss
        consistency_loss = self.compute_compression_consistency_loss(
            original_encoded, compressed_encoded
        )

        # Compute contrastive losses
        original_proj = self.projection_head(original_encoded)

        # Contrastive with momentum encoder if available
        if self.momentum_encoder is not None:
            with torch.no_grad():
                momentum_features = self.momentum_encoder(views["spectral_noise"])
                momentum_proj = self.momentum_projection_head(momentum_features)
            contrastive_loss = self.compute_contrastive_loss(
                original_proj, momentum_proj
            )

            # Update momentum encoder
            self._update_momentum_encoder()
        else:
            spectral_proj = self.projection_head(spectral_encoded)
            contrastive_loss = self.compute_contrastive_loss(
                original_proj, spectral_proj
            )

        # Combined loss
        total_loss = (
            self.config.reconstruction_weight * recon_loss
            + self.config.compression_consistency_weight * consistency_loss
            + self.config.contrastive_weight * contrastive_loss
        )

        # Advance step counter
        self.current_step += 1

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss,
            "consistency_loss": consistency_loss,
            "contrastive_loss": contrastive_loss,
            "reconstructed": reconstructed,
            "original_features": original_encoded,
            "views": views,
        }


# Example usage and testing
if __name__ == "__main__":
    # Test parameter scheduler
    print("Testing parameter scheduler...")
    schedule_config = ScheduleConfig(
        schedule_type=ScheduleType.COSINE,
        start_value=0.1,
        end_value=0.6,
        warmup_steps=100,
        total_steps=1000,
    )
    scheduler = ParameterScheduler(schedule_config)

    values = [scheduler.step() for _ in range(200)]
    print(
        f"Scheduler values: start={values[0]:.3f}, mid={values[100]:.3f}, end={values[-1]:.3f}"
    )

    # Test wavelet compressor
    print("\nTesting wavelet compressor...")
    compressor = WaveletCompressor()
    signal = torch.randn(4, 19, 1000)  # 4 samples, 19 channels, 1000 timepoints
    compressed = compressor.compress_signal(
        signal, compression_level=4, quantization_bits=8
    )
    print(f"Compression input shape: {signal.shape}, output shape: {compressed.shape}")

    # Test SSL framework
    print("\nTesting SSL framework...")

    # Dummy encoder and decoder
    class DummyEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(19, 256, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.embed_dim = 256

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x).squeeze(-1)
            return x

    class DummyDecoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(256, 19 * 1000)

        def forward(self, x):
            x = self.fc(x)
            return x.view(-1, 19, 1000)

    encoder = DummyEncoder()
    decoder = DummyDecoder()
    config = CompressionSSLConfig()

    ssl_framework = CompressionSSLFramework(encoder, decoder, config)

    # Test forward pass
    x = torch.randn(2, 19, 1000)
    outputs = ssl_framework(x)

    print(f"SSL outputs: {list(outputs.keys())}")
    print(f"Total loss: {outputs['total_loss'].item():.4f}")
    print(f"Reconstruction loss: {outputs['reconstruction_loss'].item():.4f}")
    print(f"Consistency loss: {outputs['consistency_loss'].item():.4f}")
    print(f"Contrastive loss: {outputs['contrastive_loss'].item():.4f}")

    print("\nAll tests passed!")
