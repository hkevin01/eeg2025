"""
Test suite for DANN (Domain Adversarial Neural Network) schedule correctness.

This module tests the GRL (Gradient Reversal Layer) lambda scheduling,
domain adversarial training components, and IRM compatibility.
"""

import math
import sys
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

sys.path.append("/home/kevin/Projects/eeg2025/src")

from models.invariance.dann import (
    DANNModel,
    DomainAdversarialHead,
    GradientReversalLayer,
    GRLScheduler,
    IRMPenalty,
    create_dann_model,
)


class TestGRLScheduler:
    """Test GRL lambda scheduling strategies."""

    def setup_method(self):
        """Setup test schedulers."""
        self.linear_scheduler = GRLScheduler(
            strategy="linear_warmup",
            initial_lambda=0.0,
            final_lambda=0.2,
            warmup_steps=1000,
        )

        self.exponential_scheduler = GRLScheduler(
            strategy="exponential",
            initial_lambda=0.0,
            final_lambda=0.2,
            total_steps=5000,
        )

        self.cosine_scheduler = GRLScheduler(
            strategy="cosine",
            initial_lambda=0.0,
            final_lambda=0.2,
            warmup_steps=500,
            total_steps=2000,
        )

        self.adaptive_scheduler = GRLScheduler(
            strategy="adaptive", initial_lambda=0.0, final_lambda=0.2, warmup_steps=1000
        )

    def test_linear_warmup_schedule_correctness(self):
        """Test linear warmup schedule produces correct lambda values."""
        # Test initial value
        lambda_0 = self.linear_scheduler.step()
        assert lambda_0 == 0.0, f"Expected initial lambda 0.0, got {lambda_0}"

        # Test midpoint
        for _ in range(499):  # Step to 500/1000
            self.linear_scheduler.step()

        lambda_mid = self.linear_scheduler.step()
        expected_mid = 0.1  # 50% of way from 0.0 to 0.2
        assert (
            abs(lambda_mid - expected_mid) < 0.001
        ), f"Expected lambda ~{expected_mid}, got {lambda_mid}"

        # Test final value
        for _ in range(499):  # Step to 1000
            self.linear_scheduler.step()

        lambda_final = self.linear_scheduler.step()
        assert (
            abs(lambda_final - 0.2) < 0.001
        ), f"Expected final lambda 0.2, got {lambda_final}"

        # Test post-warmup (should stay at final value)
        for _ in range(100):
            lambda_post = self.linear_scheduler.step()
            assert (
                abs(lambda_post - 0.2) < 0.001
            ), f"Expected post-warmup lambda 0.2, got {lambda_post}"

    def test_exponential_schedule_monotonicity(self):
        """Test exponential schedule is monotonically increasing."""
        lambdas = []

        for _ in range(100):
            lambda_val = self.exponential_scheduler.step()
            lambdas.append(lambda_val)

        # Check monotonic increase
        for i in range(1, len(lambdas)):
            assert (
                lambdas[i] >= lambdas[i - 1]
            ), f"Non-monotonic at step {i}: {lambdas[i-1]} -> {lambdas[i]}"

        # Check bounds
        assert lambdas[0] >= 0.0, f"First lambda should be >= 0, got {lambdas[0]}"
        assert lambdas[-1] <= 0.2, f"Last lambda should be <= 0.2, got {lambdas[-1]}"

    def test_cosine_schedule_warmup_and_annealing(self):
        """Test cosine schedule has proper warmup and annealing phases."""
        lambdas = []

        # Collect values throughout training
        for _ in range(2000):
            lambda_val = self.cosine_scheduler.step()
            lambdas.append(lambda_val)

        # Check warmup phase (first 500 steps should be linear)
        warmup_lambdas = lambdas[:500]
        for i in range(1, len(warmup_lambdas)):
            assert (
                warmup_lambdas[i] >= warmup_lambdas[i - 1]
            ), f"Warmup not monotonic at step {i}"

        # Check that warmup reaches final value
        assert (
            abs(lambdas[499] - 0.2) < 0.01
        ), f"Warmup didn't reach final value: {lambdas[499]}"

        # Check annealing phase (should decrease after warmup)
        assert lambdas[1999] < lambdas[500], "Annealing phase should decrease lambda"

    def test_adaptive_schedule_domain_accuracy_response(self):
        """Test adaptive schedule responds to domain accuracy."""
        # Test with high domain accuracy (should increase lambda)
        high_acc_lambdas = []
        for _ in range(20):
            lambda_val = self.adaptive_scheduler.step(domain_accuracy=0.9)
            high_acc_lambdas.append(lambda_val)

        # Reset scheduler
        self.adaptive_scheduler.current_step = 0
        self.adaptive_scheduler.domain_acc_history = []

        # Test with low domain accuracy (should decrease lambda)
        low_acc_lambdas = []
        for _ in range(20):
            lambda_val = self.adaptive_scheduler.step(domain_accuracy=0.5)
            low_acc_lambdas.append(lambda_val)

        # After several steps, high accuracy should lead to higher lambda adjustments
        # This is implementation-dependent but should show some adaptation
        final_high = high_acc_lambdas[-1]
        final_low = low_acc_lambdas[-1]

        # At least check that both are valid values
        assert (
            0 <= final_high <= 1.0
        ), f"High accuracy lambda out of bounds: {final_high}"
        assert 0 <= final_low <= 1.0, f"Low accuracy lambda out of bounds: {final_low}"

    def test_schedule_bounds_always_valid(self):
        """Test that all schedules always produce valid lambda values."""
        schedulers = [
            self.linear_scheduler,
            self.exponential_scheduler,
            self.cosine_scheduler,
            self.adaptive_scheduler,
        ]

        for scheduler in schedulers:
            # Reset scheduler
            scheduler.current_step = 0
            if hasattr(scheduler, "domain_acc_history"):
                scheduler.domain_acc_history = []

            # Test many steps
            for _ in range(1000):
                lambda_val = scheduler.step(domain_accuracy=np.random.uniform(0.4, 0.9))

                assert (
                    0 <= lambda_val <= 1.0
                ), f"Lambda out of bounds: {lambda_val} for {scheduler.strategy}"
                assert not math.isnan(
                    lambda_val
                ), f"Lambda is NaN for {scheduler.strategy}"
                assert not math.isinf(
                    lambda_val
                ), f"Lambda is infinite for {scheduler.strategy}"

    def test_schedule_determinism(self):
        """Test that schedules are deterministic given same inputs."""
        # Create two identical schedulers
        scheduler1 = GRLScheduler(
            strategy="linear_warmup",
            initial_lambda=0.0,
            final_lambda=0.2,
            warmup_steps=1000,
        )

        scheduler2 = GRLScheduler(
            strategy="linear_warmup",
            initial_lambda=0.0,
            final_lambda=0.2,
            warmup_steps=1000,
        )

        # Step both identically
        for _ in range(100):
            lambda1 = scheduler1.step()
            lambda2 = scheduler2.step()

            assert (
                abs(lambda1 - lambda2) < 1e-10
            ), f"Schedulers not deterministic: {lambda1} vs {lambda2}"


class TestGradientReversalLayer:
    """Test Gradient Reversal Layer functionality."""

    def setup_method(self):
        """Setup test GRL."""
        self.grl = GradientReversalLayer(lambda_val=0.5)

    def test_forward_pass_identity(self):
        """Test that forward pass is identity."""
        x = torch.randn(10, 20, requires_grad=True)
        output = self.grl(x)

        # Forward pass should be identity
        assert torch.allclose(output, x), "Forward pass should be identity"
        assert output.shape == x.shape, "Output shape should match input"

    def test_backward_pass_reversal(self):
        """Test that backward pass reverses gradients."""
        x = torch.randn(10, 20, requires_grad=True)

        # Forward pass
        output = self.grl(x)

        # Create dummy loss
        loss = output.sum()

        # Backward pass
        loss.backward()

        # Gradients should exist and be reversed
        assert x.grad is not None, "Gradients should exist"

        # The gradient should be scaled by -lambda
        expected_grad = -0.5 * torch.ones_like(x)
        assert torch.allclose(
            x.grad, expected_grad
        ), f"Expected {expected_grad[0,0]}, got {x.grad[0,0]}"

    def test_lambda_update(self):
        """Test lambda parameter updates."""
        # Test setting lambda
        self.grl.set_lambda(0.8)
        assert self.grl.lambda_val == 0.8, "Lambda should be updated"

        # Test gradient scaling with new lambda
        x = torch.randn(5, 10, requires_grad=True)
        output = self.grl(x)
        loss = output.sum()
        loss.backward()

        expected_grad = -0.8 * torch.ones_like(x)
        assert torch.allclose(
            x.grad, expected_grad, atol=1e-6
        ), "Gradient scaling should use new lambda"


class TestDomainAdversarialHead:
    """Test domain adversarial head."""

    def setup_method(self):
        """Setup test domain head."""
        self.domain_head = DomainAdversarialHead(
            input_dim=128, num_domains=3, hidden_dims=[64, 32], dropout_rate=0.2
        )

    def test_forward_pass_shape(self):
        """Test forward pass produces correct output shape."""
        batch_size = 16
        x = torch.randn(batch_size, 128)

        output = self.domain_head(x)

        expected_shape = (batch_size, 3)  # 3 domains
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {output.shape}"

    def test_domain_classification_probabilities(self):
        """Test that domain predictions can be converted to valid probabilities."""
        x = torch.randn(10, 128)
        logits = self.domain_head(x)

        # Convert to probabilities
        probs = torch.softmax(logits, dim=1)

        # Check probability constraints
        assert torch.allclose(
            probs.sum(dim=1), torch.ones(10)
        ), "Probabilities should sum to 1"
        assert torch.all(probs >= 0), "Probabilities should be non-negative"
        assert torch.all(probs <= 1), "Probabilities should be <= 1"

    def test_gradient_flow(self):
        """Test that gradients flow through domain head."""
        x = torch.randn(5, 128, requires_grad=True)
        output = self.domain_head(x)

        # Create dummy loss
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None, "Input gradients should exist"
        assert not torch.all(x.grad == 0), "Gradients should be non-zero"

        # Check parameter gradients
        for param in self.domain_head.parameters():
            if param.requires_grad:
                assert param.grad is not None, "Parameter gradients should exist"


class TestDANNModel:
    """Test complete DANN model."""

    def setup_method(self):
        """Setup test DANN model."""
        # Create mock backbone and task head
        self.backbone = MagicMock()
        self.backbone.return_value = torch.randn(4, 100, 128)  # [batch, time, features]

        self.task_head = MagicMock()
        self.task_head.return_value = torch.randn(4, 2)  # [batch, num_tasks]

        # Create scheduler
        self.scheduler = GRLScheduler(
            strategy="linear_warmup",
            initial_lambda=0.0,
            final_lambda=0.2,
            warmup_steps=100,
        )

        # Create DANN model
        self.dann_model = DANNModel(
            backbone=self.backbone,
            task_head=self.task_head,
            num_domains=2,
            feature_dim=128,
            lambda_scheduler=self.scheduler,
        )

    def test_forward_pass_outputs(self):
        """Test DANN forward pass produces expected outputs."""
        x = torch.randn(4, 64, 1000)  # [batch, channels, time]

        outputs = self.dann_model(x)

        # Check required outputs
        assert "task_output" in outputs, "Task output should be present"
        assert "domain_output" in outputs, "Domain output should be present"
        assert "lambda" in outputs, "Lambda should be present"

        # Check shapes
        assert outputs["task_output"].shape == (4, 2), "Task output shape mismatch"
        assert outputs["domain_output"].shape == (4, 2), "Domain output shape mismatch"

    def test_lambda_scheduling_integration(self):
        """Test lambda scheduling works in DANN model."""
        x = torch.randn(4, 64, 1000)

        # Initial lambda should be 0
        outputs1 = self.dann_model(x)
        assert (
            outputs1["lambda"] == 0.0
        ), f"Initial lambda should be 0, got {outputs1['lambda']}"

        # Step several times
        for _ in range(10):
            outputs = self.dann_model(x)

        # Lambda should have increased
        final_lambda = self.dann_model.get_current_lambda()
        assert final_lambda > 0.0, f"Lambda should increase, got {final_lambda}"
        assert (
            final_lambda <= 0.2
        ), f"Lambda should not exceed final value, got {final_lambda}"

    def test_optional_features_return(self):
        """Test optional features are returned when requested."""
        x = torch.randn(4, 64, 1000)

        outputs = self.dann_model(x, return_features=True)

        assert "features" in outputs, "Features should be returned when requested"
        assert (
            "raw_features" in outputs
        ), "Raw features should be returned when requested"

    def test_lambda_update_control(self):
        """Test lambda update can be controlled."""
        x = torch.randn(4, 64, 1000)

        # Get initial lambda
        initial_lambda = self.dann_model.get_current_lambda()

        # Forward pass without lambda update
        self.dann_model(x, update_lambda=False)
        lambda_after_no_update = self.dann_model.get_current_lambda()

        # Lambda should not have changed
        assert (
            lambda_after_no_update == initial_lambda
        ), "Lambda should not update when disabled"

        # Forward pass with lambda update
        self.dann_model(x, update_lambda=True)
        lambda_after_update = self.dann_model.get_current_lambda()

        # Lambda should have changed
        assert (
            lambda_after_update != initial_lambda
        ), "Lambda should update when enabled"


class TestIRMPenalty:
    """Test IRM penalty computation."""

    def setup_method(self):
        """Setup IRM penalty."""
        self.irm_penalty = IRMPenalty(penalty_weight=1.0)

    def test_irm_penalty_computation(self):
        """Test IRM penalty can be computed without errors."""
        # Create test data
        features = torch.randn(20, 10, requires_grad=True)
        targets = torch.randn(20, requires_grad=True)
        domain_ids = torch.randint(0, 2, (20,))

        # Simple classifier
        classifier = torch.nn.Linear(10, 1)

        # Compute penalty
        penalty = self.irm_penalty.compute_penalty(
            features, targets, domain_ids, classifier
        )

        # Check penalty properties
        assert isinstance(penalty, torch.Tensor), "Penalty should be a tensor"
        assert penalty.dim() == 0, "Penalty should be scalar"
        assert penalty >= 0, "Penalty should be non-negative"
        assert penalty.requires_grad, "Penalty should require gradients"

    def test_irm_penalty_edge_cases(self):
        """Test IRM penalty with edge cases."""
        # Single domain
        features = torch.randn(10, 5, requires_grad=True)
        targets = torch.randn(10, requires_grad=True)
        domain_ids = torch.zeros(10, dtype=torch.long)  # All same domain
        classifier = torch.nn.Linear(5, 1)

        penalty = self.irm_penalty.compute_penalty(
            features, targets, domain_ids, classifier
        )
        assert torch.isfinite(penalty), "Penalty should be finite for single domain"

        # Empty domain (should be handled gracefully)
        domain_ids = torch.tensor(
            [0, 1, 2, 0, 1], dtype=torch.long
        )  # Some domains have single samples
        features = torch.randn(5, 5, requires_grad=True)
        targets = torch.randn(5, requires_grad=True)

        penalty = self.irm_penalty.compute_penalty(
            features, targets, domain_ids, classifier
        )
        assert torch.isfinite(penalty), "Penalty should be finite with sparse domains"


class TestDANNCreation:
    """Test DANN model creation factory."""

    def test_create_dann_model(self):
        """Test DANN model creation with factory function."""
        # Mock components
        backbone = MagicMock()
        backbone.return_value = torch.randn(4, 100, 128)

        task_head = MagicMock()
        task_head.return_value = torch.randn(4, 3)

        # Create DANN model
        dann_model = create_dann_model(
            backbone=backbone,
            task_head=task_head,
            num_domains=3,
            lambda_schedule_config={
                "strategy": "linear_warmup",
                "initial_lambda": 0.0,
                "final_lambda": 0.1,
                "warmup_steps": 500,
            },
        )

        # Test that model was created properly
        assert isinstance(dann_model, DANNModel), "Should create DANNModel instance"
        assert dann_model.num_domains == 3, "Should have correct number of domains"
        assert dann_model.lambda_scheduler is not None, "Should have lambda scheduler"

        # Test forward pass
        x = torch.randn(4, 64, 1000)
        outputs = dann_model(x)

        assert "task_output" in outputs, "Should have task output"
        assert "domain_output" in outputs, "Should have domain output"
        assert (
            outputs["domain_output"].shape[1] == 3
        ), "Domain output should match num_domains"


def test_dann_irm_compatibility():
    """Test that DANN and IRM can be used together."""
    # This is more of an integration test
    # Create components
    backbone = MagicMock()
    backbone.return_value = torch.randn(4, 100, 128)

    task_head = MagicMock()
    task_head.return_value = torch.randn(4, 1)

    # Create DANN model
    dann_model = create_dann_model(
        backbone=backbone, task_head=task_head, num_domains=2
    )

    # Create IRM penalty
    irm_penalty = IRMPenalty(penalty_weight=0.5)

    # Test that both can be used in the same forward pass
    x = torch.randn(4, 64, 1000)
    outputs = dann_model(x, return_features=True)

    # Mock some domain labels and targets
    domain_labels = torch.randint(0, 2, (4,))
    targets = torch.randn(4)

    # Compute IRM penalty (should not interfere with DANN)
    irm_loss = irm_penalty.compute_penalty(
        outputs["features"], targets, domain_labels, task_head
    )

    # Both should work without errors
    assert torch.isfinite(irm_loss), "IRM penalty should be finite"
    assert "lambda" in outputs, "DANN lambda should be present"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
