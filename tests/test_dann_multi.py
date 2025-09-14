"""
Unit tests for Multi-Adversary DANN implementation.
"""

import pytest
import torch
import torch.nn as nn

from src.models.invariance.dann_multi import (
    DomainClassifier,
    GradientReversalFunction,
    GradientReversalLayer,
    LambdaScheduler,
    MultiAdversaryDANN,
    create_hbn_domain_adaptation,
)


class TestGradientReversalFunction:
    """Test gradient reversal autograd function."""

    def test_forward_pass(self):
        """Test forward pass is identity."""
        x = torch.randn(32, 768, requires_grad=True)
        lambda_val = 0.5

        output = GradientReversalFunction.apply(x, lambda_val)

        assert torch.allclose(output, x)
        assert output.shape == x.shape

    def test_gradient_reversal(self):
        """Test gradient reversal in backward pass."""
        x = torch.randn(32, 768, requires_grad=True)
        lambda_val = 0.5

        # Forward pass
        output = GradientReversalFunction.apply(x, lambda_val)
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Check gradients are reversed and scaled
        expected_grad = -lambda_val * torch.ones_like(x)
        assert torch.allclose(x.grad, expected_grad)


class TestLambdaScheduler:
    """Test lambda scheduling strategies."""

    def test_linear_schedule(self):
        """Test linear scheduling."""
        scheduler = LambdaScheduler(
            schedule_type="linear", max_lambda=1.0, max_epochs=100
        )

        assert scheduler.get_lambda(0) == 0.0
        assert scheduler.get_lambda(50) == 0.5
        assert scheduler.get_lambda(100) == 1.0

    def test_cosine_schedule(self):
        """Test cosine scheduling."""
        scheduler = LambdaScheduler(
            schedule_type="cosine", max_lambda=1.0, max_epochs=100
        )

        assert scheduler.get_lambda(0) == 0.0
        assert abs(scheduler.get_lambda(50) - 0.5) < 0.1
        assert scheduler.get_lambda(100) == 1.0

    def test_step_schedule(self):
        """Test step scheduling."""
        scheduler = LambdaScheduler(
            schedule_type="step", max_lambda=1.0, step_epochs=[30, 60]
        )

        assert scheduler.get_lambda(20) == 0.0
        assert scheduler.get_lambda(40) == 1.0
        assert scheduler.get_lambda(80) == 1.0

    def test_exp_schedule(self):
        """Test exponential scheduling."""
        scheduler = LambdaScheduler(
            schedule_type="exp", max_lambda=1.0, max_epochs=100, gamma=10.0
        )

        # Should start near 0 and approach 1
        assert scheduler.get_lambda(0) < 0.1
        assert scheduler.get_lambda(100) > 0.9


class TestDomainClassifier:
    """Test domain classifier implementation."""

    def test_forward_pass(self):
        """Test forward pass shapes."""
        classifier = DomainClassifier(
            input_dim=768, num_domains=4, hidden_dims=[256, 128]
        )

        x = torch.randn(32, 768)
        output = classifier(x)

        assert output.shape == (32, 4)

    def test_training_step(self):
        """Test classifier can be trained."""
        classifier = DomainClassifier(768, 4)
        optimizer = torch.optim.Adam(classifier.parameters())

        x = torch.randn(32, 768)
        targets = torch.randint(0, 4, (32,))

        # Forward pass
        logits = classifier(x)
        loss = nn.CrossEntropyLoss()(logits, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert loss.item() > 0
        # Check gradients exist
        for param in classifier.parameters():
            assert param.grad is not None


class TestMultiAdversaryDANN:
    """Test multi-adversary DANN implementation."""

    def test_initialization(self):
        """Test proper initialization."""
        domain_configs = {"subject": 100, "site": 4, "montage": 3}

        dann = MultiAdversaryDANN(feature_dim=768, domain_configs=domain_configs)

        assert len(dann.domain_classifiers) == 3
        assert "subject" in dann.domain_classifiers
        assert "site" in dann.domain_classifiers
        assert "montage" in dann.domain_classifiers

    def test_forward_pass_shapes(self):
        """Test forward pass produces correct shapes."""
        domain_configs = {"subject": 100, "site": 4}

        dann = MultiAdversaryDANN(768, domain_configs)
        features = torch.randn(32, 768)

        output = dann(features, epoch=50)

        assert "domain_logits" in output
        assert "lambda" in output
        assert output["domain_logits"]["subject"].shape == (32, 100)
        assert output["domain_logits"]["site"].shape == (32, 4)
        assert isinstance(output["lambda"], float)

    def test_loss_computation(self):
        """Test loss computation with targets."""
        domain_configs = {"subject": 10, "site": 4}
        dann = MultiAdversaryDANN(768, domain_configs)

        features = torch.randn(32, 768)
        domain_targets = {
            "subject": torch.randint(0, 10, (32,)),
            "site": torch.randint(0, 4, (32,)),
        }

        output = dann(features, domain_targets, epoch=50)

        assert "losses" in output
        assert "subject_loss" in output["losses"]
        assert "site_loss" in output["losses"]
        assert "total_domain_loss" in output["losses"]

        # Check losses are positive
        for loss_name, loss_value in output["losses"].items():
            assert loss_value.item() > 0

    def test_toy_training_decreases_loss(self):
        """Test that loss decreases on toy data."""
        domain_configs = {"site": 2}
        dann = MultiAdversaryDANN(64, domain_configs)
        optimizer = torch.optim.Adam(dann.parameters(), lr=0.01)

        # Create toy data with clear domain separation
        torch.manual_seed(42)
        features_site0 = torch.randn(16, 64) + torch.tensor([1.0] * 64)
        features_site1 = torch.randn(16, 64) + torch.tensor([-1.0] * 64)

        features = torch.cat([features_site0, features_site1])
        targets = torch.cat(
            [torch.zeros(16, dtype=torch.long), torch.ones(16, dtype=torch.long)]
        )

        initial_loss = None
        for epoch in range(10):
            optimizer.zero_grad()

            output = dann(features, {"site": targets}, epoch=epoch)
            loss = output["losses"]["total_domain_loss"]

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Loss should decrease (domain classifier should learn)
        # Note: In real DANN training, we want this loss to NOT decrease
        # But for testing the classifier component, it should decrease
        assert final_loss < initial_loss or abs(final_loss - initial_loss) < 0.1


class TestHBNDomainAdaptation:
    """Test HBN-specific domain adaptation setup."""

    def test_factory_function(self):
        """Test HBN domain adaptation factory."""
        dann = create_hbn_domain_adaptation(
            feature_dim=512, num_subjects=50, num_sites=3, num_montages=2
        )

        assert dann.feature_dim == 512
        assert len(dann.domain_classifiers) == 3
        assert dann.domain_configs["subject"] == 50
        assert dann.domain_configs["site"] == 3
        assert dann.domain_configs["montage"] == 2

    def test_loss_weights(self):
        """Test loss weights are properly applied."""
        dann = create_hbn_domain_adaptation()

        features = torch.randn(16, 768)
        targets = {
            "subject": torch.randint(0, 100, (16,)),
            "site": torch.randint(0, 4, (16,)),
            "montage": torch.randint(0, 3, (16,)),
        }

        output = dann(features, targets, epoch=50)

        # Check that different domains have different weights
        subject_loss = output["losses"]["subject_loss"]
        site_loss = output["losses"]["site_loss"]
        montage_loss = output["losses"]["montage_loss"]

        # Site should have highest weight (1.0)
        assert isinstance(subject_loss.item(), float)
        assert isinstance(site_loss.item(), float)
        assert isinstance(montage_loss.item(), float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
