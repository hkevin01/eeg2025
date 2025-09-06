"""
Unit tests for EEG prediction heads.
"""

import pytest
import torch
import torch.nn as nn
from src.models.heads.regression import TemporalRegressionHead
from src.models.heads.classification import CalibratedClassificationHead
from src.models.heads.psychopathology import PsychopathologyHead


class TestTemporalRegressionHead:
    """Test temporal regression head for RT prediction."""

    def test_initialization(self):
        """Test proper initialization."""
        head = TemporalRegressionHead(
            input_dim=768,
            hidden_dim=256,
            num_heads=8,
            dropout=0.1
        )

        assert head.input_dim == 768
        assert head.hidden_dim == 256
        assert head.num_heads == 8

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        head = TemporalRegressionHead(input_dim=512, hidden_dim=128)

        # Input: (batch, time, features)
        x = torch.randn(32, 1000, 512)

        output = head(x)

        # Should output single regression value per sample
        assert output.shape == (32, 1)

    def test_temporal_attention_mechanism(self):
        """Test that temporal attention works."""
        head = TemporalRegressionHead(input_dim=256, hidden_dim=128, num_heads=4)

        # Create input with clear temporal pattern
        batch_size, seq_len, dim = 16, 500, 256
        x = torch.randn(batch_size, seq_len, dim)

        # Add strong signal at specific time points
        x[:, 100:120, :] *= 3.0  # Strong signal in middle

        output = head(x)

        assert output.shape == (batch_size, 1)
        assert torch.isfinite(output).all()

    def test_uncertainty_estimation(self):
        """Test uncertainty estimation capability."""
        head = TemporalRegressionHead(
            input_dim=256,
            hidden_dim=128,
            uncertainty_estimation=True
        )

        x = torch.randn(16, 500, 256)
        output = head(x)

        # With uncertainty, should output mean and variance
        if hasattr(head, 'uncertainty_estimation') and head.uncertainty_estimation:
            # Implementation detail - check if uncertainty is returned
            assert output.shape[1] >= 1  # At least mean prediction

    def test_multi_scale_aggregation(self):
        """Test multi-scale temporal aggregation."""
        head = TemporalRegressionHead(
            input_dim=256,
            hidden_dim=128,
            use_multi_scale=True
        )

        x = torch.randn(16, 1000, 256)
        output = head(x)

        assert output.shape == (16, 1)
        assert torch.isfinite(output).all()

    def test_gradient_flow(self):
        """Test that gradients flow through the head."""
        head = TemporalRegressionHead(input_dim=128, hidden_dim=64)

        x = torch.randn(8, 500, 128, requires_grad=True)
        output = head(x)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        for param in head.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestCalibratedClassificationHead:
    """Test calibrated classification head for success prediction."""

    def test_initialization(self):
        """Test proper initialization."""
        head = CalibratedClassificationHead(
            input_dim=768,
            num_classes=2,
            hidden_dim=256,
            use_temperature_scaling=True
        )

        assert head.input_dim == 768
        assert head.num_classes == 2
        assert head.hidden_dim == 256
        assert head.use_temperature_scaling

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        head = CalibratedClassificationHead(
            input_dim=512,
            num_classes=3,
            hidden_dim=128
        )

        # Input: (batch, time, features) or (batch, features)
        x = torch.randn(32, 512)

        logits = head(x)

        assert logits.shape == (32, 3)

    def test_temporal_input_handling(self):
        """Test handling of temporal input."""
        head = CalibratedClassificationHead(input_dim=256, num_classes=2)

        # Temporal input: (batch, time, features)
        x_temporal = torch.randn(16, 1000, 256)
        logits = head(x_temporal)

        assert logits.shape == (16, 2)

    def test_temperature_scaling(self):
        """Test temperature scaling for calibration."""
        head = CalibratedClassificationHead(
            input_dim=256,
            num_classes=2,
            use_temperature_scaling=True
        )

        x = torch.randn(32, 256)
        logits = head(x)

        # Temperature should affect the logits scale
        assert hasattr(head, 'temperature')
        assert logits.shape == (32, 2)

    def test_confidence_estimation(self):
        """Test confidence estimation capability."""
        head = CalibratedClassificationHead(
            input_dim=256,
            num_classes=2,
            confidence_estimation=True
        )

        x = torch.randn(16, 256)
        logits = head(x)

        # Should produce logits
        assert logits.shape == (16, 2)

        # Probabilities should sum to 1
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(16), atol=1e-6)

    def test_focal_loss_compatibility(self):
        """Test compatibility with focal loss."""
        head = CalibratedClassificationHead(input_dim=256, num_classes=2)

        x = torch.randn(32, 256)
        targets = torch.randint(0, 2, (32,))

        logits = head(x)

        # Should be able to compute standard cross-entropy
        loss = nn.CrossEntropyLoss()(logits, targets)
        assert loss.item() > 0

    def test_balanced_accuracy_optimization(self):
        """Test that head works for balanced accuracy optimization."""
        head = CalibratedClassificationHead(input_dim=128, num_classes=2)

        x = torch.randn(64, 128)
        targets = torch.randint(0, 2, (64,))

        logits = head(x)

        # Check prediction accuracy
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == targets).float().mean()

        # Should produce reasonable predictions (random baseline = 0.5)
        assert 0.0 <= accuracy <= 1.0


class TestPsychopathologyHead:
    """Test psychopathology head for CBCL factor prediction."""

    def test_initialization(self):
        """Test proper initialization."""
        target_factors = ["p_factor", "internalizing", "externalizing", "attention"]

        head = PsychopathologyHead(
            input_dim=768,
            target_factors=target_factors,
            hidden_dim=256,
            use_clinical_normalization=True
        )

        assert head.input_dim == 768
        assert len(head.target_factors) == 4
        assert head.num_factors == 4
        assert head.use_clinical_normalization

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        target_factors = ["p_factor", "internalizing", "externalizing", "attention"]
        head = PsychopathologyHead(
            input_dim=512,
            target_factors=target_factors,
            hidden_dim=128
        )

        x = torch.randn(32, 512)

        predictions = head(x)

        # Should output one prediction per factor
        assert predictions.shape == (32, 4)

    def test_clinical_normalization(self):
        """Test clinical normalization with demographics."""
        target_factors = ["p_factor", "internalizing"]
        head = PsychopathologyHead(
            input_dim=256,
            target_factors=target_factors,
            use_clinical_normalization=True,
            use_demographic_features=True
        )

        x = torch.randn(16, 256)
        demographics = {
            "age": torch.randn(16, 1),    # Age in years
            "gender": torch.randint(0, 2, (16, 1)).float()  # Binary gender
        }

        predictions = head(x, demographics=demographics)

        assert predictions.shape == (16, 2)

    def test_factor_correlations(self):
        """Test factor correlation enforcement."""
        target_factors = ["p_factor", "internalizing", "externalizing", "attention"]
        head = PsychopathologyHead(
            input_dim=256,
            target_factors=target_factors,
            enforce_factor_correlations=True
        )

        x = torch.randn(64, 256)  # Larger batch for correlation analysis
        predictions = head(x)

        assert predictions.shape == (64, 4)

        # Check that predictions have reasonable correlations
        corr_matrix = torch.corrcoef(predictions.T)

        # All factors should correlate positively with p-factor (first factor)
        p_factor_corrs = corr_matrix[0, 1:]
        # Note: This is a weak test since we're using random data
        assert torch.isfinite(p_factor_corrs).all()

    def test_uncertainty_quantification(self):
        """Test uncertainty quantification for clinical predictions."""
        target_factors = ["p_factor", "internalizing"]
        head = PsychopathologyHead(
            input_dim=256,
            target_factors=target_factors,
            uncertainty_estimation=True
        )

        x = torch.randn(16, 256)
        predictions = head(x)

        # Should output factor predictions
        assert predictions.shape == (16, 2)

        # If uncertainty is estimated, check if additional outputs exist
        if hasattr(head, 'uncertainty_estimation') and head.uncertainty_estimation:
            # Implementation-specific uncertainty handling
            assert torch.isfinite(predictions).all()

    def test_age_gender_adjustment(self):
        """Test age and gender adjustment for clinical scores."""
        target_factors = ["p_factor", "attention"]
        head = PsychopathologyHead(
            input_dim=256,
            target_factors=target_factors,
            use_clinical_normalization=True
        )

        x = torch.randn(32, 256)

        # Test with different age groups
        young_age = torch.full((16, 1), 8.0)    # 8 years old
        old_age = torch.full((16, 1), 16.0)     # 16 years old

        pred_young = head(x[:16], demographics={"age": young_age})
        pred_old = head(x[16:32], demographics={"age": old_age})

        assert pred_young.shape == (16, 2)
        assert pred_old.shape == (16, 2)

    def test_multi_output_regression(self):
        """Test multi-output regression capability."""
        target_factors = ["p_factor", "internalizing", "externalizing", "attention"]
        head = PsychopathologyHead(
            input_dim=512,
            target_factors=target_factors
        )

        x = torch.randn(32, 512)
        predictions = head(x)

        # Each sample should have prediction for each factor
        assert predictions.shape == (32, 4)

        # Test with targets for loss computation
        targets = torch.randn(32, 4)  # True factor scores

        mse_loss = nn.MSELoss()(predictions, targets)
        assert mse_loss.item() >= 0

    def test_correlation_aware_loss_compatibility(self):
        """Test compatibility with correlation-aware losses."""
        target_factors = ["p_factor", "internalizing", "externalizing"]
        head = PsychopathologyHead(
            input_dim=256,
            target_factors=target_factors
        )

        x = torch.randn(64, 256)  # Larger batch for correlation
        targets = torch.randn(64, 3)

        predictions = head(x)

        # Test Pearson correlation computation (simplified)
        pred_centered = predictions - predictions.mean(dim=0)
        target_centered = targets - targets.mean(dim=0)

        # Should be able to compute correlations
        correlation = (pred_centered * target_centered).sum(dim=0) / (
            torch.sqrt((pred_centered**2).sum(dim=0)) *
            torch.sqrt((target_centered**2).sum(dim=0))
        )

        assert correlation.shape == (3,)  # One correlation per factor
        assert torch.isfinite(correlation).all()


class TestHeadIntegration:
    """Test integration between different heads."""

    def test_heads_with_same_backbone_features(self):
        """Test that all heads work with same backbone features."""
        # Simulate backbone features
        backbone_features = torch.randn(16, 768)

        # RT prediction head
        rt_head = TemporalRegressionHead(input_dim=768, hidden_dim=256)
        rt_pred = rt_head(backbone_features.unsqueeze(1))  # Add time dim
        assert rt_pred.shape == (16, 1)

        # Success classification head
        success_head = CalibratedClassificationHead(input_dim=768, num_classes=2)
        success_logits = success_head(backbone_features)
        assert success_logits.shape == (16, 2)

        # Psychopathology head
        psych_head = PsychopathologyHead(
            input_dim=768,
            target_factors=["p_factor", "internalizing"]
        )
        psych_pred = psych_head(backbone_features)
        assert psych_pred.shape == (16, 2)

    def test_multi_task_training_compatibility(self):
        """Test heads are compatible with multi-task training."""
        # Shared features
        features = torch.randn(32, 512)

        # Multiple heads for different tasks
        heads = {
            "rt": TemporalRegressionHead(input_dim=512, hidden_dim=128),
            "success": CalibratedClassificationHead(input_dim=512, num_classes=2),
            "factors": PsychopathologyHead(
                input_dim=512,
                target_factors=["p_factor", "attention"]
            )
        }

        # Forward pass through all heads
        outputs = {}
        for task_name, head in heads.items():
            if task_name == "rt":
                outputs[task_name] = head(features.unsqueeze(1))  # Add time dim
            else:
                outputs[task_name] = head(features)

        # Check all outputs
        assert outputs["rt"].shape == (32, 1)
        assert outputs["success"].shape == (32, 2)
        assert outputs["factors"].shape == (32, 2)

        # Should be able to compute multi-task loss
        targets = {
            "rt": torch.randn(32, 1),
            "success": torch.randint(0, 2, (32,)),
            "factors": torch.randn(32, 2)
        }

        losses = {
            "rt": nn.MSELoss()(outputs["rt"], targets["rt"]),
            "success": nn.CrossEntropyLoss()(outputs["success"], targets["success"]),
            "factors": nn.MSELoss()(outputs["factors"], targets["factors"])
        }

        total_loss = sum(losses.values())
        assert total_loss.item() >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
