"""
Compression-Augmented SSL
=========================

Self-supervised learning with compression-based data augmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

try:
    import pywt
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False


class CompressionAugmentation(nn.Module):
    """
    Compression-based data augmentation using wavelets and spectral distortion.
    """

    def __init__(
        self,
        wavelet_family: str = "db4",
        compression_levels: List[float] = [0.1, 0.3, 0.5, 0.7],
        noise_std: float = 0.02,
        freq_mask_prob: float = 0.3,
        time_mask_prob: float = 0.3
    ):
        super().__init__()

        self.wavelet_family = wavelet_family
        self.compression_levels = compression_levels
        self.noise_std = noise_std
        self.freq_mask_prob = freq_mask_prob
        self.time_mask_prob = time_mask_prob

        # Check if PyWavelets is available
        self.use_wavelets = PYWT_AVAILABLE
        if not self.use_wavelets:
            print("Warning: PyWavelets not available, using alternative compression")

    def wavelet_compress(
        self,
        x: torch.Tensor,
        compression_level: float
    ) -> torch.Tensor:
        """
        Apply wavelet compression to EEG signals.

        Args:
            x: Input tensor of shape [batch_size, n_channels, seq_len]
            compression_level: Compression level (0.0 = no compression, 1.0 = max compression)

        Returns:
            Compressed signal
        """
        if not self.use_wavelets:
            # Fallback: simple frequency-domain compression
            return self._fft_compress(x, compression_level)

        batch_size, n_channels, seq_len = x.shape
        compressed = torch.zeros_like(x)

        for b in range(batch_size):
            for c in range(n_channels):
                signal = x[b, c].cpu().numpy()

                # Wavelet decomposition
                coeffs = pywt.wavedec(signal, self.wavelet_family, level=4)

                # Apply compression by zeroing out small coefficients
                threshold = compression_level * np.std(coeffs[0])
                coeffs_compressed = []

                for coeff in coeffs:
                    coeff_compressed = coeff.copy()
                    coeff_compressed[np.abs(coeff_compressed) < threshold] = 0
                    coeffs_compressed.append(coeff_compressed)

                # Wavelet reconstruction
                compressed_signal = pywt.waverec(coeffs_compressed, self.wavelet_family)

                # Handle length mismatch
                if len(compressed_signal) != seq_len:
                    if len(compressed_signal) > seq_len:
                        compressed_signal = compressed_signal[:seq_len]
                    else:
                        padding = seq_len - len(compressed_signal)
                        compressed_signal = np.pad(compressed_signal, (0, padding), 'constant')

                compressed[b, c] = torch.from_numpy(compressed_signal).to(x.device)

        return compressed

    def _fft_compress(
        self,
        x: torch.Tensor,
        compression_level: float
    ) -> torch.Tensor:
        """
        Fallback compression using FFT.

        Args:
            x: Input tensor
            compression_level: Compression level

        Returns:
            Compressed signal using FFT
        """
        # FFT
        X_fft = torch.fft.fft(x, dim=-1)

        # Apply compression by zeroing out high frequencies
        seq_len = x.shape[-1]
        cutoff = int(seq_len * (1 - compression_level))

        X_compressed = X_fft.clone()
        X_compressed[..., cutoff:] = 0

        # IFFT
        x_compressed = torch.fft.ifft(X_compressed, dim=-1).real

        return x_compressed

    def spectral_distortion(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spectral distortion (frequency and time masking).

        Args:
            x: Input tensor of shape [batch_size, n_channels, seq_len]

        Returns:
            Spectrally distorted signal
        """
        batch_size, n_channels, seq_len = x.shape

        # Convert to frequency domain
        X_fft = torch.fft.fft(x, dim=-1)
        X_magnitude = torch.abs(X_fft)
        X_phase = torch.angle(X_fft)

        # Frequency masking
        if torch.rand(1) < self.freq_mask_prob:
            freq_mask_width = int(0.1 * seq_len)  # Mask 10% of frequencies
            freq_start = torch.randint(0, seq_len - freq_mask_width, (1,))
            X_magnitude[..., freq_start:freq_start + freq_mask_width] *= 0.1

        # Reconstruct and convert back to time domain
        X_distorted = X_magnitude * torch.exp(1j * X_phase)
        x_distorted = torch.fft.ifft(X_distorted, dim=-1).real

        # Time masking
        if torch.rand(1) < self.time_mask_prob:
            time_mask_width = int(0.05 * seq_len)  # Mask 5% of time
            time_start = torch.randint(0, seq_len - time_mask_width, (1,))
            x_distorted[..., time_start:time_start + time_mask_width] *= 0.1

        return x_distorted

    def add_noise(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add Gaussian noise to the signal.

        Args:
            x: Input tensor

        Returns:
            Noisy signal
        """
        noise = torch.randn_like(x) * self.noise_std * torch.std(x, dim=-1, keepdim=True)
        return x + noise

    def forward(
        self,
        x: torch.Tensor,
        compression_level: Optional[float] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Apply compression augmentation.

        Args:
            x: Input tensor of shape [batch_size, n_channels, seq_len]
            compression_level: Specific compression level (if None, randomly sample)

        Returns:
            Dictionary containing original and augmented views
        """
        if compression_level is None:
            compression_level = np.random.choice(self.compression_levels)

        # Create different augmented views
        views = {'original': x}

        # Wavelet compression
        views['compressed'] = self.wavelet_compress(x, compression_level)

        # Spectral distortion
        views['distorted'] = self.spectral_distortion(x)

        # Noisy version
        views['noisy'] = self.add_noise(x)

        # Combined augmentation
        x_combined = self.wavelet_compress(x, compression_level * 0.5)
        x_combined = self.spectral_distortion(x_combined)
        x_combined = self.add_noise(x_combined)
        views['combined'] = x_combined

        return views


class CompressionSSLLoss(nn.Module):
    """
    SSL loss combining consistency and contrastive objectives.
    """

    def __init__(
        self,
        temperature: float = 0.1,
        consistency_weight: float = 1.0,
        contrastive_weight: float = 0.5
    ):
        super().__init__()

        self.temperature = temperature
        self.consistency_weight = consistency_weight
        self.contrastive_weight = contrastive_weight

    def consistency_loss(
        self,
        features_orig: torch.Tensor,
        features_aug: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute consistency loss between original and augmented features.

        Args:
            features_orig: Features from original data
            features_aug: Features from augmented data

        Returns:
            Consistency loss
        """
        return F.mse_loss(features_orig, features_aug)

    def contrastive_loss(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            features: Feature representations
            labels: Optional labels for supervised contrastive learning

        Returns:
            Contrastive loss
        """
        batch_size = features.shape[0]

        # Normalize features
        features = F.normalize(features, dim=-1)

        # Compute similarity matrix
        similarity_matrix = torch.matmul(features, features.T) / self.temperature

        # Create mask for positive pairs
        if labels is not None:
            # Supervised contrastive learning
            labels = labels.unsqueeze(0)
            mask = torch.eq(labels, labels.T).float()
        else:
            # Self-supervised contrastive learning
            # Consider consecutive samples as positive pairs
            mask = torch.eye(batch_size, device=features.device)
            for i in range(batch_size - 1):
                mask[i, i + 1] = 1
                mask[i + 1, i] = 1

        # Remove diagonal (self-similarity)
        mask = mask - torch.eye(batch_size, device=features.device)

        # Compute loss
        exp_similarities = torch.exp(similarity_matrix)
        sum_exp_similarities = torch.sum(exp_similarities, dim=1, keepdim=True)

        log_prob = similarity_matrix - torch.log(sum_exp_similarities)
        loss = -torch.sum(mask * log_prob) / torch.sum(mask)

        return loss

    def forward(
        self,
        features_dict: Dict[str, torch.Tensor],
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total SSL loss.

        Args:
            features_dict: Dictionary of features from different views
            labels: Optional labels

        Returns:
            Total loss and loss components
        """
        losses = {}

        # Consistency loss
        features_orig = features_dict['original']
        consistency_losses = []

        for view_name, features_aug in features_dict.items():
            if view_name != 'original':
                consistency_losses.append(
                    self.consistency_loss(features_orig, features_aug)
                )

        if consistency_losses:
            losses['consistency'] = torch.stack(consistency_losses).mean()
        else:
            losses['consistency'] = torch.tensor(0.0, device=features_orig.device)

        # Contrastive loss
        # Concatenate all features for contrastive learning
        all_features = torch.cat(list(features_dict.values()), dim=0)
        if labels is not None:
            all_labels = torch.cat([labels] * len(features_dict), dim=0)
        else:
            all_labels = None

        losses['contrastive'] = self.contrastive_loss(all_features, all_labels)

        # Total loss
        total_loss = (
            self.consistency_weight * losses['consistency'] +
            self.contrastive_weight * losses['contrastive']
        )

        losses['total'] = total_loss

        return total_loss, losses


class CompressionAugmentedTrainer(nn.Module):
    """
    Trainer combining feature extraction with compression-augmented SSL.
    """

    def __init__(
        self,
        feature_extractor: nn.Module,
        augmentation: CompressionAugmentation,
        ssl_loss: CompressionSSLLoss,
        use_curriculum: bool = True,
        curriculum_epochs: int = 20
    ):
        super().__init__()

        self.feature_extractor = feature_extractor
        self.augmentation = augmentation
        self.ssl_loss = ssl_loss
        self.use_curriculum = use_curriculum
        self.curriculum_epochs = curriculum_epochs

        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        """Set current epoch for curriculum learning."""
        self.current_epoch = epoch

    def get_compression_level(self) -> float:
        """Get compression level based on curriculum."""
        if not self.use_curriculum:
            return np.random.choice(self.augmentation.compression_levels)

        # Start with light compression, gradually increase
        progress = min(1.0, self.current_epoch / self.curriculum_epochs)
        max_compression = max(self.augmentation.compression_levels)
        return progress * max_compression

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with compression augmentation.

        Args:
            x: Input EEG data
            labels: Optional labels

        Returns:
            Total loss and loss components
        """
        # Get compression level
        compression_level = self.get_compression_level()

        # Apply augmentation
        augmented_views = self.augmentation(x, compression_level)

        # Extract features from all views
        features_dict = {}
        for view_name, view_data in augmented_views.items():
            features = self.feature_extractor(view_data)
            # Global average pooling
            features = features.mean(dim=1)  # [batch_size, feature_dim]
            features_dict[view_name] = features

        # Compute SSL loss
        total_loss, loss_components = self.ssl_loss(features_dict, labels)

        # Add compression level to loss components
        loss_components['compression_level'] = compression_level

        return total_loss, loss_components


def create_compression_ssl(config: Dict[str, Any]) -> CompressionAugmentedTrainer:
    """
    Factory function to create compression-augmented SSL trainer.

    Args:
        config: Configuration dictionary

    Returns:
        CompressionAugmentedTrainer instance
    """
    ssl_config = config.get('compression_ssl', {})

    # Create augmentation
    augmentation = CompressionAugmentation(
        wavelet_family=ssl_config.get('wavelet_family', 'db4'),
        compression_levels=ssl_config.get('compression_levels', [0.1, 0.3, 0.5, 0.7]),
        noise_std=ssl_config.get('noise_std', 0.02),
        freq_mask_prob=ssl_config.get('spectral', {}).get('freq_mask_prob', 0.3),
        time_mask_prob=ssl_config.get('spectral', {}).get('time_mask_prob', 0.3)
    )

    # Create SSL loss
    ssl_loss = CompressionSSLLoss(
        temperature=ssl_config.get('contrastive', {}).get('temperature', 0.1),
        consistency_weight=ssl_config.get('loss_weights', {}).get('consistency', 1.0),
        contrastive_weight=ssl_config.get('loss_weights', {}).get('contrastive', 0.5)
    )

    # Note: feature_extractor should be passed separately
    # This factory function returns partially configured trainer
    def create_trainer(feature_extractor):
        return CompressionAugmentedTrainer(
            feature_extractor=feature_extractor,
            augmentation=augmentation,
            ssl_loss=ssl_loss,
            use_curriculum=ssl_config.get('curriculum', {}).get('enabled', True),
            curriculum_epochs=ssl_config.get('curriculum', {}).get('progression_epochs', 20)
        )

    return create_trainer
