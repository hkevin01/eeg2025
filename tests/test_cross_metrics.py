"""
Test suite for cross-task transfer metrics and computations.

This module tests the official metrics used in the EEG Challenge 2025
cross-task evaluation, including Pearson correlation, RMSE, AUROC, AUPRC,
and balanced accuracy for reaction time and success prediction tasks.
"""

import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    roc_auc_score,
)

sys.path.append("/home/kevin/Projects/eeg2025/src")

from models.losses.corr_mse import AdaptiveCorrMSELoss, CorrMSELoss, RobustCorrMSELoss
from training.train_cross_task import OfficialMetrics


class TestOfficialMetrics:
    """Test official metrics computation."""

    def setup_method(self):
        """Setup test data."""
        self.metrics = OfficialMetrics()

        # Create test data
        np.random.seed(42)
        self.n_samples = 100

        # RT test data (regression)
        self.rt_true = np.random.normal(0.5, 0.2, self.n_samples)
        self.rt_pred = self.rt_true + np.random.normal(0, 0.1, self.n_samples)

        # Success test data (classification)
        self.success_true = np.random.binomial(1, 0.6, self.n_samples)
        self.success_pred = np.random.uniform(0, 1, self.n_samples)

        # Perfect correlation data
        self.perfect_rt_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.perfect_rt_pred = np.array(
            [2.0, 4.0, 6.0, 8.0, 10.0]
        )  # Perfect positive correlation

        # No correlation data
        self.no_corr_rt_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.no_corr_rt_pred = np.array([3.0, 1.0, 4.0, 2.0, 5.0])  # No correlation

    def test_pearson_correlation_perfect(self):
        """Test Pearson correlation with perfect correlation."""
        corr = self.metrics.pearson_correlation(
            self.perfect_rt_true, self.perfect_rt_pred
        )
        assert abs(corr - 1.0) < 1e-6, f"Expected perfect correlation, got {corr}"

    def test_pearson_correlation_normal(self):
        """Test Pearson correlation with normal data."""
        corr = self.metrics.pearson_correlation(self.rt_true, self.rt_pred)

        # Compare with scipy implementation
        expected_corr, _ = pearsonr(self.rt_true, self.rt_pred)
        assert abs(corr - expected_corr) < 1e-6, f"Expected {expected_corr}, got {corr}"

        # Should be positive correlation
        assert corr > 0.5, f"Expected positive correlation, got {corr}"

    def test_pearson_correlation_edge_cases(self):
        """Test Pearson correlation edge cases."""
        # Empty arrays
        corr = self.metrics.pearson_correlation(np.array([]), np.array([]))
        assert corr == 0.0, f"Expected 0 for empty arrays, got {corr}"

        # Single value
        corr = self.metrics.pearson_correlation(np.array([1.0]), np.array([2.0]))
        assert corr == 0.0, f"Expected 0 for single value, got {corr}"

        # NaN values
        rt_with_nan = np.array([1.0, 2.0, np.nan, 4.0])
        pred_with_nan = np.array([2.0, 4.0, 6.0, 8.0])
        corr = self.metrics.pearson_correlation(rt_with_nan, pred_with_nan)

        # Should handle NaN gracefully
        assert not np.isnan(corr), "Correlation should not be NaN"

        # Constant arrays (zero variance)
        constant_true = np.array([1.0, 1.0, 1.0, 1.0])
        varying_pred = np.array([1.0, 2.0, 3.0, 4.0])
        corr = self.metrics.pearson_correlation(constant_true, varying_pred)
        assert corr == 0.0, f"Expected 0 for constant true values, got {corr}"

    def test_rmse_computation(self):
        """Test RMSE computation."""
        rmse = self.metrics.rmse(self.rt_true, self.rt_pred)

        # Compute expected RMSE
        expected_rmse = np.sqrt(np.mean((self.rt_true - self.rt_pred) ** 2))
        assert abs(rmse - expected_rmse) < 1e-6, f"Expected {expected_rmse}, got {rmse}"

        # RMSE should be positive
        assert rmse >= 0, f"RMSE should be non-negative, got {rmse}"

    def test_rmse_edge_cases(self):
        """Test RMSE edge cases."""
        # Perfect predictions
        perfect_pred = self.rt_true.copy()
        rmse = self.metrics.rmse(self.rt_true, perfect_pred)
        assert (
            rmse < 1e-10
        ), f"Expected near-zero RMSE for perfect predictions, got {rmse}"

        # Empty arrays
        rmse = self.metrics.rmse(np.array([]), np.array([]))
        assert rmse == float("inf"), f"Expected inf for empty arrays, got {rmse}"

        # NaN values
        rt_with_nan = np.array([1.0, 2.0, np.nan, 4.0])
        pred_with_nan = np.array([2.0, 4.0, 6.0, 8.0])
        rmse = self.metrics.rmse(rt_with_nan, pred_with_nan)
        assert not np.isnan(rmse), "RMSE should not be NaN"
        assert rmse != float("inf"), "RMSE should be finite with some valid values"

    def test_auroc_computation(self):
        """Test AUROC computation."""
        auroc = self.metrics.auroc(self.success_true, self.success_pred)

        # Compare with sklearn implementation
        expected_auroc = roc_auc_score(self.success_true, self.success_pred)
        assert (
            abs(auroc - expected_auroc) < 1e-6
        ), f"Expected {expected_auroc}, got {auroc}"

        # AUROC should be between 0 and 1
        assert 0 <= auroc <= 1, f"AUROC should be in [0, 1], got {auroc}"

    def test_auroc_edge_cases(self):
        """Test AUROC edge cases."""
        # Perfect predictions
        perfect_true = np.array([0, 0, 1, 1])
        perfect_pred = np.array([0.1, 0.2, 0.8, 0.9])
        auroc = self.metrics.auroc(perfect_true, perfect_pred)
        assert auroc == 1.0, f"Expected perfect AUROC, got {auroc}"

        # All same class
        same_class_true = np.array([1, 1, 1, 1])
        same_class_pred = np.array([0.1, 0.2, 0.8, 0.9])
        auroc = self.metrics.auroc(same_class_true, same_class_pred)
        assert auroc == 0.5, f"Expected 0.5 AUROC for edge case, got {auroc}"

    def test_auprc_computation(self):
        """Test AUPRC computation."""
        auprc = self.metrics.auprc(self.success_true, self.success_pred)

        # Compare with sklearn implementation
        expected_auprc = average_precision_score(self.success_true, self.success_pred)
        assert (
            abs(auprc - expected_auprc) < 1e-6
        ), f"Expected {expected_auprc}, got {auprc}"

        # AUPRC should be between 0 and 1
        assert 0 <= auprc <= 1, f"AUPRC should be in [0, 1], got {auprc}"

    def test_balanced_accuracy_computation(self):
        """Test balanced accuracy computation."""
        bal_acc = self.metrics.balanced_accuracy(self.success_true, self.success_pred)

        # Compare with sklearn implementation
        y_pred_binary = (self.success_pred > 0.5).astype(int)
        expected_bal_acc = balanced_accuracy_score(self.success_true, y_pred_binary)
        assert (
            abs(bal_acc - expected_bal_acc) < 1e-6
        ), f"Expected {expected_bal_acc}, got {bal_acc}"

        # Balanced accuracy should be between 0 and 1
        assert (
            0 <= bal_acc <= 1
        ), f"Balanced accuracy should be in [0, 1], got {bal_acc}"

    def test_compute_all_metrics(self):
        """Test computing all metrics together."""
        metrics_dict = self.metrics.compute_all_metrics(
            self.rt_true, self.rt_pred, self.success_true, self.success_pred
        )

        # Check all required metrics are present
        required_metrics = [
            "rt_pearson",
            "rt_rmse",
            "success_auroc",
            "success_auprc",
            "success_balanced_acc",
            "combined_score",
        ]

        for metric in required_metrics:
            assert metric in metrics_dict, f"Missing metric: {metric}"
            assert not np.isnan(metrics_dict[metric]), f"Metric {metric} is NaN"

        # Check combined score computation
        normalized_rt_corr = (metrics_dict["rt_pearson"] + 1) / 2
        expected_combined = (normalized_rt_corr + metrics_dict["success_auroc"]) / 2
        assert (
            abs(metrics_dict["combined_score"] - expected_combined) < 1e-6
        ), f"Combined score mismatch: {metrics_dict['combined_score']} vs {expected_combined}"

        # Combined score should be in [0, 1]
        assert (
            0 <= metrics_dict["combined_score"] <= 1
        ), f"Combined score should be in [0, 1], got {metrics_dict['combined_score']}"

    def test_metrics_shapes(self):
        """Test that metrics handle different input shapes correctly."""
        # Test with different array shapes
        rt_true_2d = self.rt_true.reshape(-1, 1)
        rt_pred_2d = self.rt_pred.reshape(-1, 1)

        # Should work with 2D arrays
        corr = self.metrics.pearson_correlation(rt_true_2d, rt_pred_2d)
        assert not np.isnan(corr), "Pearson correlation failed with 2D input"

        rmse = self.metrics.rmse(rt_true_2d, rt_pred_2d)
        assert not np.isnan(rmse), "RMSE failed with 2D input"

    def test_metrics_consistency(self):
        """Test consistency across multiple calls."""
        # Metrics should be deterministic
        corr1 = self.metrics.pearson_correlation(self.rt_true, self.rt_pred)
        corr2 = self.metrics.pearson_correlation(self.rt_true, self.rt_pred)
        assert corr1 == corr2, "Pearson correlation should be deterministic"

        rmse1 = self.metrics.rmse(self.rt_true, self.rt_pred)
        rmse2 = self.metrics.rmse(self.rt_true, self.rt_pred)
        assert rmse1 == rmse2, "RMSE should be deterministic"


class TestCorrMSELoss:
    """Test CorrMSE loss functions."""

    def setup_method(self):
        """Setup test data."""
        torch.manual_seed(42)
        self.batch_size = 32

        # Create test tensors
        self.y_true = torch.randn(self.batch_size)
        self.y_pred = self.y_true + 0.1 * torch.randn(self.batch_size)

        # Perfect correlation data
        self.perfect_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        self.perfect_pred = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])

        # Loss functions
        self.corr_mse_loss = CorrMSELoss(alpha=1.0, beta=1.0)
        self.adaptive_loss = AdaptiveCorrMSELoss(initial_alpha=1.0, initial_beta=1.0)
        self.robust_loss = RobustCorrMSELoss(alpha=1.0, beta=1.0)

    def test_corr_mse_basic(self):
        """Test basic CorrMSE loss computation."""
        loss = self.corr_mse_loss(self.y_pred, self.y_true)

        # Loss should be a scalar
        assert loss.dim() == 0, "Loss should be scalar"

        # Loss should require gradients
        assert loss.requires_grad, "Loss should require gradients"

        # Loss should be finite
        assert torch.isfinite(loss), "Loss should be finite"

    def test_corr_mse_perfect_correlation(self):
        """Test CorrMSE with perfect correlation."""
        loss = self.corr_mse_loss(self.perfect_pred, self.perfect_true)

        # With perfect correlation, the correlation component should be -1.0
        # So loss = alpha * mse - beta * 1.0
        mse_component = torch.nn.functional.mse_loss(
            self.perfect_pred, self.perfect_true
        )
        expected_loss = 1.0 * mse_component - 1.0 * 1.0

        assert (
            abs(loss.item() - expected_loss.item()) < 1e-5
        ), f"Loss mismatch: {loss.item()} vs {expected_loss.item()}"

    def test_corr_mse_component_weights(self):
        """Test CorrMSE with different component weights."""
        # High alpha (emphasize MSE)
        high_mse_loss = CorrMSELoss(alpha=10.0, beta=1.0)
        loss_high_mse = high_mse_loss(self.y_pred, self.y_true)

        # High beta (emphasize correlation)
        high_corr_loss = CorrMSELoss(alpha=1.0, beta=10.0)
        loss_high_corr = high_corr_loss(self.y_pred, self.y_true)

        # Losses should be different
        assert (
            abs(loss_high_mse.item() - loss_high_corr.item()) > 1e-3
        ), "Different weights should produce different losses"

    def test_corr_mse_edge_cases(self):
        """Test CorrMSE edge cases."""
        # Empty tensors
        empty_pred = torch.tensor([])
        empty_true = torch.tensor([])
        loss = self.corr_mse_loss(empty_pred, empty_true)
        assert loss.item() == 0.0, "Empty tensors should produce zero loss"

        # Single value
        single_pred = torch.tensor([1.0])
        single_true = torch.tensor([2.0])
        loss = self.corr_mse_loss(single_pred, single_true)
        assert torch.isfinite(loss), "Single value should produce finite loss"

        # Constant values (zero variance)
        constant_pred = torch.tensor([1.0, 1.0, 1.0, 1.0])
        varying_true = torch.tensor([1.0, 2.0, 3.0, 4.0])
        loss = self.corr_mse_loss(constant_pred, varying_true)
        assert torch.isfinite(loss), "Constant prediction should produce finite loss"

    def test_corr_mse_gradients(self):
        """Test that CorrMSE loss produces valid gradients."""
        # Create parameters that require gradients
        y_pred = self.y_pred.clone().requires_grad_(True)

        loss = self.corr_mse_loss(y_pred, self.y_true)
        loss.backward()

        # Gradients should exist and be finite
        assert y_pred.grad is not None, "Gradients should exist"
        assert torch.all(torch.isfinite(y_pred.grad)), "Gradients should be finite"
        assert not torch.all(y_pred.grad == 0), "Gradients should be non-zero"

    def test_adaptive_corr_mse(self):
        """Test adaptive CorrMSE loss."""
        # Initial weights
        initial_alpha = self.adaptive_loss.alpha.item()
        initial_beta = self.adaptive_loss.beta.item()

        # Run multiple forward passes to trigger adaptation
        for _ in range(150):  # More than adapt_frequency
            loss = self.adaptive_loss(self.y_pred, self.y_true)

        # Weights should have potentially changed
        final_alpha = self.adaptive_loss.alpha.item()
        final_beta = self.adaptive_loss.beta.item()

        # At least check that the mechanism works (weights stay in valid ranges)
        assert (
            self.adaptive_loss.alpha_range[0]
            <= final_alpha
            <= self.adaptive_loss.alpha_range[1]
        ), "Alpha should stay in valid range"
        assert (
            self.adaptive_loss.beta_range[0]
            <= final_beta
            <= self.adaptive_loss.beta_range[1]
        ), "Beta should stay in valid range"

    def test_robust_corr_mse(self):
        """Test robust CorrMSE loss."""
        # Add outliers to test data
        outlier_pred = self.y_pred.clone()
        outlier_pred[0] = 100.0  # Large outlier
        outlier_pred[1] = -100.0  # Large outlier

        # Robust loss should handle outliers better
        robust_loss = self.robust_loss(outlier_pred, self.y_true)
        standard_loss = self.corr_mse_loss(outlier_pred, self.y_true)

        # Both should be finite
        assert torch.isfinite(robust_loss), "Robust loss should be finite"
        assert torch.isfinite(standard_loss), "Standard loss should be finite"

        # Test Spearman correlation computation
        spearman_corr = self.robust_loss.spearman_correlation(self.y_pred, self.y_true)
        assert -1 <= spearman_corr <= 1, "Spearman correlation should be in [-1, 1]"

    def test_loss_shapes(self):
        """Test loss functions with different input shapes."""
        # 2D inputs
        y_pred_2d = self.y_pred.unsqueeze(1)  # [batch_size, 1]
        y_true_2d = self.y_true.unsqueeze(1)  # [batch_size, 1]

        loss = self.corr_mse_loss(y_pred_2d, y_true_2d)
        assert loss.dim() == 0, "Loss should be scalar even with 2D input"

        # Different batch sizes (broadcasting)
        y_pred_small = self.y_pred[:10]
        y_true_small = self.y_true[:10]

        loss_small = self.corr_mse_loss(y_pred_small, y_true_small)
        assert torch.isfinite(loss_small), "Loss should work with smaller batches"

    def test_normalization_effect(self):
        """Test the effect of target normalization."""
        # Loss with normalization
        norm_loss = CorrMSELoss(normalize_targets=True)
        loss_norm = norm_loss(self.y_pred, self.y_true)

        # Loss without normalization
        no_norm_loss = CorrMSELoss(normalize_targets=False)
        loss_no_norm = no_norm_loss(self.y_pred, self.y_true)

        # Both should be finite but potentially different
        assert torch.isfinite(loss_norm), "Normalized loss should be finite"
        assert torch.isfinite(loss_no_norm), "Non-normalized loss should be finite"


def test_integration():
    """Integration test for cross-task metrics and losses."""
    # Simulate a complete evaluation scenario
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = 200

    # Generate realistic test data
    rt_true = np.random.exponential(
        0.5, n_samples
    )  # Reaction times (exponential distribution)
    rt_pred = rt_true * np.random.normal(1.0, 0.2, n_samples)  # Predictions with noise

    # Success depends on RT (faster RT = higher success probability)
    success_prob = 1 / (1 + np.exp((rt_true - 0.5) * 5))  # Sigmoid relationship
    success_true = np.random.binomial(1, success_prob, n_samples)
    success_pred = success_prob + np.random.normal(0, 0.1, n_samples)
    success_pred = np.clip(success_pred, 0, 1)  # Clip to valid probability range

    # Test official metrics
    metrics = OfficialMetrics()
    all_metrics = metrics.compute_all_metrics(
        rt_true, rt_pred, success_true, success_pred
    )

    # Validate all metrics
    assert all(
        [not np.isnan(v) for v in all_metrics.values()]
    ), "All metrics should be finite"
    assert 0 <= all_metrics["combined_score"] <= 1, "Combined score should be in [0, 1]"
    assert all_metrics["rt_rmse"] >= 0, "RMSE should be non-negative"
    assert 0 <= all_metrics["success_auroc"] <= 1, "AUROC should be in [0, 1]"

    # Test loss function
    rt_pred_tensor = torch.tensor(rt_pred, dtype=torch.float32)
    rt_true_tensor = torch.tensor(rt_true, dtype=torch.float32)

    corr_mse_loss = CorrMSELoss(alpha=1.0, beta=1.0)
    loss = corr_mse_loss(rt_pred_tensor, rt_true_tensor)

    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.requires_grad, "Loss should require gradients"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
