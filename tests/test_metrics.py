#!/usr/bin/env python3
"""Test metric calculations"""
import pytest
import numpy as np
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, accuracy_score

def test_pearson_correlation():
    """Test Pearson correlation calculation"""
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.2, 2.9, 4.1, 5.0])
    r, p = pearsonr(y_true, y_pred)
    assert r > 0.9, f"Expected high correlation, got {r}"
    assert p < 0.05, f"Expected significant p-value, got {p}"

def test_pearson_perfect():
    """Test perfect correlation"""
    y = np.array([1, 2, 3, 4, 5])
    r, p = pearsonr(y, y)
    assert abs(r - 1.0) < 0.001, f"Expected r=1.0, got {r}"

def test_auroc_calculation():
    """Test AUROC calculation"""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0.1, 0.4, 0.6, 0.9])
    auroc = roc_auc_score(y_true, y_pred)
    assert auroc == 1.0, f"Expected AUROC=1.0, got {auroc}"

def test_accuracy_calculation():
    """Test accuracy calculation"""
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 0])
    acc = accuracy_score(y_true, y_pred)
    assert acc == 0.8, f"Expected accuracy=0.8, got {acc}"

def test_age_range_validation():
    """Test age range validation"""
    ages = np.array([6.0, 8.5, 10.0, 12.5, 14.0])
    assert ages.min() >= 5.0, "Ages should be >= 5"
    assert ages.max() <= 22.0, "Ages should be <= 22"
    assert ages.mean() > 8.0, "Mean age should be reasonable"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
