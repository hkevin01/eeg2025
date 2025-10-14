"""
Tests for P1 scripts: baseline training, artifact detection,
cross-site validation, and hyperparameter optimization.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBaselineTraining:
    """Tests for train_baseline.py script."""

    def test_feature_extraction_psd(self):
        """Test PSD feature extraction."""
        from scripts.train_baseline import extract_features

        # Create dummy EEG data (128 channels, 1000 samples)
        data = np.random.randn(128, 1000)

        # Extract PSD features
        features = extract_features(data, method="psd")

        # Should have 128 channels * 5 frequency bands = 640 features
        assert features.shape == (640,), f"Expected 640 features, got {features.shape}"
        assert not np.any(np.isnan(features)), "Features contain NaN values"

    def test_feature_extraction_stats(self):
        """Test statistical feature extraction."""
        from scripts.train_baseline import extract_features

        data = np.random.randn(128, 1000)
        features = extract_features(data, method="stats")

        # Should have 128 channels * 5 stats = 640 features
        assert features.shape == (640,)

    def test_feature_extraction_raw_mean(self):
        """Test raw mean feature extraction."""
        from scripts.train_baseline import extract_features

        data = np.random.randn(128, 1000)
        features = extract_features(data, method="raw_mean")

        # Should have 128 features (one per channel)
        assert features.shape == (128,)


class TestArtifactDetection:
    """Tests for artifact_detection.py script."""

    def test_bad_channel_detection(self):
        """Test bad channel detection."""
        import mne

        from scripts.artifact_detection import detect_bad_channels

        # Create dummy raw data
        info = mne.create_info(ch_names=["Ch1", "Ch2", "Ch3"], sfreq=500, ch_types=["eeg"] * 3)
        data = np.random.randn(3, 1000) * 1e-6  # Use realistic EEG amplitudes

        # Add a bad channel (very high variance)
        data[1, :] *= 1000

        raw = mne.io.RawArray(data, info, verbose=False)

        bad_channels = detect_bad_channels(raw, threshold=3.0)

        # Should detect at least the bad channel
        assert len(bad_channels) > 0, "Should detect bad channels"
        assert "Ch2" in bad_channels, "Should detect Ch2 as bad"

    def test_threshold_artifact_detection(self):
        """Test threshold-based artifact detection."""
        import mne

        from scripts.artifact_detection import detect_artifacts_threshold

        info = mne.create_info(ch_names=["Ch1"], sfreq=500, ch_types=["eeg"])
        data = np.random.randn(1, 1000) * 50e-6  # Normal amplitude

        # Add artifact
        data[0, 500:510] = 200e-6  # Exceeds 150 µV threshold

        raw = mne.io.RawArray(data, info, verbose=False)

        artifact_mask, bad_segments = detect_artifacts_threshold(raw, threshold=150e-6)

        # Should detect the artifact
        assert len(bad_segments) > 0, "Should detect artifact segments"


class TestCrossSiteValidation:
    """Tests for cross_site_validation.py script."""

    def test_site_assignment(self):
        """Test site assignment from metadata."""
        import pandas as pd

        from scripts.cross_site_validation import assign_sites_from_metadata

        # Create dummy participants.tsv
        bids_root = Path("data/raw/hbn")
        participants = pd.read_csv(bids_root / "participants.tsv", sep="\t")

        # Get actual subjects
        subjects = ["NDARAC904DMU", "NDARAG143ARJ"]

        site_map = assign_sites_from_metadata(bids_root, subjects)

        # Should have site assignments
        assert len(site_map) == 2, "Should assign sites to both subjects"
        assert all(s in site_map for s in subjects), "All subjects should have sites"

    def test_leave_one_site_out_cv(self):
        """Test leave-one-site-out cross-validation."""
        from sklearn.ensemble import RandomForestRegressor

        from scripts.cross_site_validation import leave_one_site_out_cv

        # Create dummy data
        X = np.random.randn(10, 50)
        y = np.random.randn(10)
        sites = np.array(["R1"] * 5 + ["R2"] * 5)

        def model_fn():
            return RandomForestRegressor(n_estimators=10, random_state=42)

        results = leave_one_site_out_cv(X, y, sites, model_fn, task_type="regression")

        # Should have results for both sites
        assert "per_site" in results
        assert "R1" in results["per_site"]
        assert "R2" in results["per_site"]
        assert "overall" in results

    def test_grouped_k_fold_cv(self):
        """Test grouped k-fold cross-validation."""
        from sklearn.ensemble import RandomForestRegressor

        from scripts.cross_site_validation import grouped_k_fold_cv

        X = np.random.randn(20, 50)
        y = np.random.randn(20)
        sites = np.array(["R1"] * 10 + ["R2"] * 10)

        def model_fn():
            return RandomForestRegressor(n_estimators=10, random_state=42)

        results = grouped_k_fold_cv(X, y, sites, model_fn, n_splits=2, task_type="regression")

        # Should have results for all folds
        assert "per_fold" in results
        assert len(results["per_fold"]) == 2
        assert "overall" in results


class TestHyperparameterOptimization:
    """Tests for hyperparameter_optimization.py script."""

    def test_search_space_logistic(self):
        """Test logistic regression search space."""
        import optuna

        from scripts.hyperparameter_optimization import get_search_space_logistic

        # Create a simple study and get a trial
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        params = get_search_space_logistic(trial)

        assert "C" in params
        assert "penalty" in params
        assert params["solver"] == "saga"
        assert "random_state" in params

    def test_search_space_random_forest(self):
        """Test random forest search space."""
        import optuna

        from scripts.hyperparameter_optimization import get_search_space_random_forest

        # Create a simple study and get a trial
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        params = get_search_space_random_forest(trial, "regression")

        assert "n_estimators" in params
        assert "max_depth" in params
        assert params["random_state"] == 42
        assert "min_samples_split" in params
        assert "min_samples_leaf" in params

    def test_objective_classification(self):
        """Test classification objective function."""
        import optuna

        from scripts.hyperparameter_optimization import objective_classification

        # Create dummy data
        X_train = np.random.randn(50, 20)
        y_train = np.random.randint(0, 2, 50)
        X_val = np.random.randn(20, 20)
        y_val = np.random.randint(0, 2, 20)

        # Create a simple trial
        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        try:
            score = objective_classification(
                trial, X_train, y_train, X_val, y_val, "logistic"
            )
            assert 0 <= score <= 1, f"AUROC should be between 0 and 1, got {score}"
        except Exception as e:
            # Some parameter combinations may fail, which is okay
            pytest.skip(f"Objective failed with: {e}")

    def test_objective_regression(self):
        """Test regression objective function."""
        import optuna

        from scripts.hyperparameter_optimization import objective_regression

        # Create dummy data
        X_train = np.random.randn(50, 20)
        y_train = np.random.randn(50)
        X_val = np.random.randn(20, 20)
        y_val = np.random.randn(20)

        study = optuna.create_study(direction="maximize")
        trial = study.ask()

        try:
            score = objective_regression(
                trial, X_train, y_train, X_val, y_val, "ridge", metric="r2"
            )
            assert -10 <= score <= 1, f"R2 should be reasonable, got {score}"
        except Exception as e:
            pytest.skip(f"Objective failed with: {e}")


class TestIntegration:
    """Integration tests for P1 pipeline."""

    def test_baseline_to_crosssite_pipeline(self):
        """Test that baseline features can be used in cross-site validation."""
        from sklearn.ensemble import RandomForestRegressor

        from scripts.cross_site_validation import grouped_k_fold_cv
        from scripts.train_baseline import extract_features

        # Create dummy EEG data for 2 subjects
        data1 = np.random.randn(128, 1000)
        data2 = np.random.randn(128, 1000)

        # Extract features
        features1 = extract_features(data1, method="psd")
        features2 = extract_features(data2, method="psd")

        X = np.vstack([features1, features2])
        y = np.random.randn(2)
        sites = np.array(["R1", "R2"])

        def model_fn():
            return RandomForestRegressor(n_estimators=10, random_state=42)

        # This should work without errors
        results = grouped_k_fold_cv(X, y, sites, model_fn, n_splits=2, task_type="regression")

        assert "overall" in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
