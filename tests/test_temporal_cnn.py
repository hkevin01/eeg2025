"""Tests for TemporalCNN backbone."""

import pytest
import torch

from src.models.backbones.temporal_cnn import TemporalCNN, create_temporal_cnn


class TestTemporalCNN:
    """Test cases for TemporalCNN model."""

    def test_model_creation(self):
        """Test basic model creation."""
        model = TemporalCNN(n_channels=64, n_classes=128)
        assert model.n_channels == 64
        assert model.n_classes == 128

    def test_forward_pass(self, sample_eeg_data, device):
        """Test forward pass with sample data."""
        model = TemporalCNN(n_channels=64, n_classes=128)
        model.to(device)
        sample_eeg_data = sample_eeg_data.to(device)

        output = model(sample_eeg_data)

        batch_size = sample_eeg_data.shape[0]
        assert output.shape == (batch_size, 128)
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    def test_different_input_sizes(self, device):
        """Test model with different input sizes."""
        model = TemporalCNN(n_channels=32, n_classes=64)
        model.to(device)

        # Test different sequence lengths
        for seq_len in [100, 500, 1000]:
            x = torch.randn(2, 32, seq_len).to(device)
            output = model(x)
            assert output.shape == (2, 64)

    def test_feature_extraction(self, sample_eeg_data, device):
        """Test intermediate feature extraction."""
        model = TemporalCNN(n_channels=64, n_classes=128)
        model.to(device)
        sample_eeg_data = sample_eeg_data.to(device)

        # Extract features from different layers
        features = model.get_features(sample_eeg_data, layer_idx=0)
        assert features.shape[0] == sample_eeg_data.shape[0]  # Same batch size
        assert len(features.shape) == 3  # (batch, channels, time)

    def test_model_parameters(self):
        """Test model parameter count is reasonable."""
        model = TemporalCNN(n_channels=64, n_classes=128)
        param_count = sum(p.numel() for p in model.parameters())

        # Should be reasonable size (not too small, not too large)
        assert 10_000 < param_count < 10_000_000

    def test_model_gradients(self, sample_eeg_data, device):
        """Test that gradients flow properly."""
        model = TemporalCNN(n_channels=64, n_classes=128)
        model.to(device)
        sample_eeg_data = sample_eeg_data.to(device)

        # Forward pass
        output = model(sample_eeg_data)
        loss = output.mean()

        # Backward pass
        loss.backward()

        # Check that gradients exist and are finite
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"


class TestTemporalCNNFactory:
    """Test cases for TemporalCNN factory function."""

    @pytest.mark.parametrize("model_size", ["tiny", "small", "medium", "large"])
    def test_create_different_sizes(self, model_size, device):
        """Test creating models of different sizes."""
        model = create_temporal_cnn(n_channels=64, model_size=model_size)
        model.to(device)

        # Test with sample data
        x = torch.randn(2, 64, 500).to(device)
        output = model(x)

        assert output.shape[0] == 2  # Batch size
        assert len(output.shape) == 2  # (batch, features)

    def test_custom_parameters(self, device):
        """Test factory with custom parameters."""
        model = create_temporal_cnn(
            n_channels=32,
            model_size="small",
            dropout=0.2,
        )
        model.to(device)

        # Test forward pass
        x = torch.randn(1, 32, 250).to(device)
        output = model(x)

        assert output.shape[0] == 1
        assert not torch.isnan(output).any()


class TestTemporalCNNComponents:
    """Test individual components of TemporalCNN."""

    def test_depthwise_separable_conv(self):
        """Test depthwise separable convolution component."""
        from src.models.backbones.temporal_cnn import DepthwiseSeparableConv1d

        conv = DepthwiseSeparableConv1d(
            in_channels=64,
            out_channels=128,
            kernel_size=7,
            padding=3,
        )

        x = torch.randn(4, 64, 100)
        output = conv(x)

        assert output.shape == (4, 128, 100)

    def test_temporal_block(self):
        """Test temporal block component."""
        from src.models.backbones.temporal_cnn import TemporalBlock

        block = TemporalBlock(
            in_channels=64,
            out_channels=64,
            kernel_size=7,
            dilation=2,
        )

        x = torch.randn(4, 64, 100)
        output = block(x)

        assert output.shape == (4, 64, 100)

        # Test residual connection
        assert not torch.equal(output, x)  # Should be different due to processing

    def test_temporal_block_different_channels(self):
        """Test temporal block with different input/output channels."""
        from src.models.backbones.temporal_cnn import TemporalBlock

        block = TemporalBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
            dilation=1,
        )

        x = torch.randn(2, 32, 50)
        output = block(x)

        assert output.shape == (2, 64, 50)
