"""
Unit tests for Compression-Augmented SSL implementation.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
try:
    import pywt
    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False

from src.models.compression_ssl import (
    CompressionAugmentation,
    CompressionSSLLoss,
    CompressionAugmentedTrainer
)


class TestCompressionAugmentation:
    """Test compression augmentation strategies."""

    def test_initialization(self):
        """Test proper initialization."""
        augmenter = CompressionAugmentation(
            wavelet='db4',
            compression_levels=[0.1, 0.3, 0.5],
            noise_std=0.01
        )

        assert augmenter.wavelet == 'db4'
        assert len(augmenter.compression_levels) == 3
        assert augmenter.noise_std == 0.01

    def test_forward_preserves_shape(self):
        """Test that augmentation preserves input shape."""
        augmenter = CompressionAugmentation()

        # Test different input shapes
        shapes = [
            (16, 128, 1000),  # (batch, channels, time)
            (8, 64, 500),
            (32, 256, 2000)
        ]

        for shape in shapes:
            x = torch.randn(*shape)
            augmented = augmenter(x)

            assert augmented.shape == x.shape
            assert augmented.dtype == x.dtype

    @pytest.mark.skipif(not HAS_PYWT, reason="PyWavelets not available")
    def test_wavelet_compression(self):
        """Test wavelet compression implementation."""
        augmenter = CompressionAugmentation(
            wavelet='db4',
            compression_levels=[0.5]
        )

        # Create a signal with clear frequency content
        t = torch.linspace(0, 1, 1000)
        signal = torch.sin(2 * torch.pi * 10 * t)  # 10 Hz sine wave
        signal = signal.unsqueeze(0).unsqueeze(0)  # (1, 1, 1000)

        compressed = augmenter._apply_wavelet_compression(signal, level=0.5)

        assert compressed.shape == signal.shape
        # Compressed signal should be different but similar
        assert not torch.allclose(signal, compressed, atol=1e-6)
        correlation = torch.corrcoef(torch.stack([signal.flatten(), compressed.flatten()]))[0, 1]
        assert correlation > 0.5  # Should maintain some correlation

    def test_spectral_distortion(self):
        """Test spectral distortion augmentation."""
        augmenter = CompressionAugmentation()

        x = torch.randn(16, 128, 1000)
        distorted = augmenter._apply_spectral_distortion(x, severity=0.1)

        assert distorted.shape == x.shape
        assert not torch.allclose(x, distorted, atol=1e-6)

    def test_temporal_masking(self):
        """Test temporal masking augmentation."""
        augmenter = CompressionAugmentation()

        x = torch.randn(16, 128, 1000)
        masked = augmenter._apply_temporal_masking(x, mask_ratio=0.1)

        assert masked.shape == x.shape

        # Check that some values are actually masked (set to zero)
        num_zeros_original = (x == 0).sum()
        num_zeros_masked = (masked == 0).sum()
        assert num_zeros_masked > num_zeros_original

    def test_adjustable_severity(self):
        """Test that compression severity is adjustable."""
        augmenter = CompressionAugmentation()

        x = torch.randn(8, 64, 500)

        # Test different severity levels
        mild = augmenter._apply_spectral_distortion(x, severity=0.01)
        severe = augmenter._apply_spectral_distortion(x, severity=0.5)

        # More severe distortion should be more different from original
        mild_diff = torch.norm(x - mild)
        severe_diff = torch.norm(x - severe)

        assert severe_diff > mild_diff

    def test_cpu_fallback(self):
        """Test CPU fallback when CUDA not available."""
        augmenter = CompressionAugmentation()

        x = torch.randn(8, 64, 500)  # CPU tensor
        augmented = augmenter(x)

        assert augmented.shape == x.shape
        assert augmented.device == x.device  # Should stay on CPU


class TestCompressionSSLLoss:
    """Test compression SSL loss computation."""

    def test_initialization(self):
        """Test loss function initialization."""
        loss_fn = CompressionSSLLoss(
            consistency_weight=1.0,
            contrastive_weight=0.5,
            temperature=0.1
        )

        assert loss_fn.consistency_weight == 1.0
        assert loss_fn.contrastive_weight == 0.5
        assert loss_fn.temperature == 0.1

    def test_consistency_loss_computation(self):
        """Test consistency loss between views."""
        loss_fn = CompressionSSLLoss()

        # Create similar but not identical features
        features_clean = torch.randn(16, 256)
        features_compressed = features_clean + 0.1 * torch.randn(16, 256)

        loss = loss_fn.consistency_loss(features_clean, features_compressed)

        assert loss.item() >= 0.0  # Loss should be non-negative
        assert loss.requires_grad  # Should be differentiable

    def test_contrastive_loss_computation(self):
        """Test contrastive loss (InfoNCE)."""
        loss_fn = CompressionSSLLoss()

        features = torch.randn(32, 256)
        loss = loss_fn.contrastive_loss(features)

        assert loss.item() >= 0.0
        assert loss.requires_grad

    def test_forward_computes_total_loss(self):
        """Test forward pass computes total weighted loss."""
        loss_fn = CompressionSSLLoss(
            consistency_weight=1.0,
            contrastive_weight=0.5
        )

        features_clean = torch.randn(16, 256)
        features_compressed = torch.randn(16, 256)

        total_loss, loss_dict = loss_fn(features_clean, features_compressed)

        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.item() >= 0.0
        assert "consistency_loss" in loss_dict
        assert "contrastive_loss" in loss_dict
        assert "total_loss" in loss_dict

    def test_loss_non_nan(self):
        """Test that loss computation doesn't produce NaN values."""
        loss_fn = CompressionSSLLoss()

        # Test with various input patterns
        test_cases = [
            torch.randn(16, 256),  # Normal case
            torch.zeros(16, 256),  # All zeros
            torch.ones(16, 256),   # All ones
            torch.randn(16, 256) * 1000,  # Large values
        ]

        for features in test_cases:
            features_2 = features + 0.1 * torch.randn_like(features)

            total_loss, _ = loss_fn(features, features_2)

            assert not torch.isnan(total_loss)
            assert not torch.isinf(total_loss)


class TestCompressionAugmentedTrainer:
    """Test compression-augmented SSL trainer."""

    def test_initialization(self):
        """Test trainer initialization."""
        # Simple encoder for testing
        encoder = nn.Sequential(
            nn.Linear(1000, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        trainer = CompressionAugmentedTrainer(
            encoder=encoder,
            augmentation_config={
                'compression_levels': [0.1, 0.3, 0.5],
                'noise_std': 0.01
            }
        )

        assert trainer.encoder is encoder
        assert isinstance(trainer.augmenter, CompressionAugmentation)
        assert isinstance(trainer.ssl_loss, CompressionSSLLoss)

    def test_training_step(self):
        """Test single training step."""
        encoder = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(256, 128)
        )

        trainer = CompressionAugmentedTrainer(encoder)
        optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)

        # Simulate EEG batch
        batch = torch.randn(16, 128, 1000)  # (batch, channels, time)

        # Training step
        loss_dict = trainer.training_step(batch, optimizer)

        assert "total_loss" in loss_dict
        assert "consistency_loss" in loss_dict
        assert "contrastive_loss" in loss_dict

        # Check that all losses are finite
        for loss_name, loss_value in loss_dict.items():
            assert torch.isfinite(torch.tensor(loss_value))

    def test_plug_and_play_interface(self):
        """Test that trainer works with different encoder architectures."""
        # Test different encoder types
        encoders = [
            # Simple MLP
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 1000, 256)
            ),
            # CNN-based
            nn.Sequential(
                nn.Conv1d(128, 64, 3),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(256)
            ),
            # Transformer-like (simplified)
            nn.Sequential(
                nn.Linear(1000, 256),
                nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True),
                nn.AdaptiveAvgPool1d(256)
            )
        ]

        for encoder in encoders:
            trainer = CompressionAugmentedTrainer(encoder)

            # Test that training step works
            batch = torch.randn(8, 128, 1000)
            optimizer = torch.optim.Adam(encoder.parameters())

            try:
                loss_dict = trainer.training_step(batch, optimizer)
                assert "total_loss" in loss_dict
            except Exception as e:
                pytest.fail(f"Encoder {type(encoder)} failed: {e}")


class TestCompressionRobustness:
    """Test that compression augmentation improves robustness."""

    def test_compression_changes_signal(self):
        """Test that compression actually changes the signal."""
        augmenter = CompressionAugmentation(
            compression_levels=[0.5],
            noise_std=0.05
        )

        x = torch.randn(16, 128, 1000)
        x_compressed = augmenter(x)

        # Signals should be different
        assert not torch.allclose(x, x_compressed, atol=1e-6)

        # But not completely uncorrelated
        correlation = torch.corrcoef(torch.stack([x.flatten(), x_compressed.flatten()]))[0, 1]
        assert correlation > 0.1  # Should maintain some structure

    def test_robustness_curriculum(self):
        """Test curriculum learning with increasing compression."""
        augmenter = CompressionAugmentation()

        x = torch.randn(8, 64, 500)

        # Test progression from light to heavy compression
        light_compression = augmenter._apply_spectral_distortion(x, severity=0.01)
        heavy_compression = augmenter._apply_spectral_distortion(x, severity=0.5)

        # Heavy compression should be more different
        light_diff = torch.norm(x - light_compression)
        heavy_diff = torch.norm(x - heavy_compression)

        assert heavy_diff > light_diff

    def test_multiple_augmentation_types(self):
        """Test that multiple augmentation types can be combined."""
        augmenter = CompressionAugmentation(
            use_wavelet=True,
            use_spectral=True,
            use_temporal=True
        )

        x = torch.randn(8, 64, 500)
        augmented = augmenter(x)

        assert augmented.shape == x.shape
        # Should be significantly different when all augmentations applied
        assert torch.norm(x - augmented) > 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
