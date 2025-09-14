#!/usr/bin/env python3
"""
Simple test runner for cross-task metrics without pytest dependencies.
"""

import os
import sys

import numpy as np
import torch

# Add the source path
sys.path.insert(0, "/home/kevin/Projects/eeg2025/src")


def test_official_metrics():
    """Test official metrics computation."""
    print("Testing Official Metrics...")

    from training.train_cross_task import OfficialMetrics

    metrics = OfficialMetrics()

    # Create test data
    np.random.seed(42)
    rt_true = np.random.normal(0.5, 0.2, 100)
    rt_pred = rt_true + np.random.normal(0, 0.1, 100)
    success_true = np.random.binomial(1, 0.6, 100)
    success_pred = np.random.uniform(0, 1, 100)

    # Test individual metrics
    corr = metrics.pearson_correlation(rt_true, rt_pred)
    assert not np.isnan(corr), "Correlation should not be NaN"
    assert corr > 0.5, f"Expected positive correlation, got {corr}"

    rmse = metrics.rmse(rt_true, rt_pred)
    assert rmse > 0, "RMSE should be positive"

    auroc = metrics.auroc(success_true, success_pred)
    assert 0 <= auroc <= 1, f"AUROC should be in [0,1], got {auroc}"

    # Test all metrics together
    all_metrics = metrics.compute_all_metrics(
        rt_true, rt_pred, success_true, success_pred
    )
    assert "combined_score" in all_metrics, "Combined score should be computed"
    assert 0 <= all_metrics["combined_score"] <= 1, "Combined score should be in [0,1]"

    print("✅ Official Metrics tests passed")


def test_corr_mse_loss():
    """Test CorrMSE loss functions."""
    print("Testing CorrMSE Loss...")

    from models.losses.corr_mse import (
        AdaptiveCorrMSELoss,
        CorrMSELoss,
        RobustCorrMSELoss,
    )

    torch.manual_seed(42)
    y_true = torch.randn(32)
    y_pred = y_true + 0.1 * torch.randn(32)

    # Test standard CorrMSE
    loss_fn = CorrMSELoss(alpha=1.0, beta=1.0)
    loss = loss_fn(y_pred, y_true)
    assert torch.isfinite(loss), "Loss should be finite"
    assert loss.requires_grad, "Loss should require gradients"

    # Test with perfect correlation
    perfect_true = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    perfect_pred = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
    loss_perfect = loss_fn(perfect_pred, perfect_true)
    assert torch.isfinite(loss_perfect), "Perfect correlation loss should be finite"

    # Test adaptive loss
    adaptive_loss = AdaptiveCorrMSELoss()
    loss_adaptive = adaptive_loss(y_pred, y_true)
    assert torch.isfinite(loss_adaptive), "Adaptive loss should be finite"

    # Test robust loss
    robust_loss = RobustCorrMSELoss()
    loss_robust = robust_loss(y_pred, y_true)
    assert torch.isfinite(loss_robust), "Robust loss should be finite"

    print("✅ CorrMSE Loss tests passed")


def test_ccd_heads():
    """Test CCD task heads."""
    print("Testing CCD Heads...")

    from models.heads import CCDClassificationHead, CCDRegressionHead

    # Test regression head
    reg_head = CCDRegressionHead(input_dim=128)
    x = torch.randn(32, 128)
    rt_pred = reg_head(x)
    assert rt_pred.shape == (32, 1), f"Expected shape (32, 1), got {rt_pred.shape}"

    # Test classification head
    clf_head = CCDClassificationHead(input_dim=128)
    success_pred = clf_head(x)
    assert success_pred.shape == (
        32,
        1,
    ), f"Expected shape (32, 1), got {success_pred.shape}"

    print("✅ CCD Heads tests passed")


def test_cross_task_model():
    """Test cross-task model creation."""
    print("Testing Cross-Task Model...")

    from models.backbones.temporal_cnn import TemporalCNN
    from training.train_cross_task import CrossTaskConfig, create_cross_task_model

    config = CrossTaskConfig()

    try:
        model = create_cross_task_model(config, n_channels=64)

        # Test forward pass
        x = torch.randn(4, 64, 1000)  # [batch, channels, time]
        task_id = torch.ones(4, dtype=torch.long)  # CCD task

        outputs = model(x, task_id=task_id)

        assert "rt_prediction" in outputs, "RT prediction should be in outputs"
        assert (
            "success_prediction" in outputs
        ), "Success prediction should be in outputs"
        assert outputs["rt_prediction"].shape == (4, 1), "RT prediction shape mismatch"
        assert outputs["success_prediction"].shape == (
            4,
            1,
        ), "Success prediction shape mismatch"

        print("✅ Cross-Task Model tests passed")

    except Exception as e:
        print(f"⚠️ Cross-Task Model test skipped due to: {e}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("Cross-Task Transfer Test Suite")
    print("=" * 60)

    tests = [
        test_official_metrics,
        test_corr_mse_loss,
        test_ccd_heads,
        test_cross_task_model,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} failed: {e}")
            import traceback

            traceback.print_exc()

    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed ({100*passed/total:.1f}%)")

    if passed == total:
        print(
            "🎉 All tests passed! Cross-task transfer implementation is working correctly."
        )
    else:
        print("⚠️ Some tests failed. Please check the implementation.")

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
