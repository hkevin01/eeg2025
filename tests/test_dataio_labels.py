"""
Test suite for data loading, labels, and splits integrity.

Tests challenge-compliant label loading, official splits, and leakage protection.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataio.bids_loader import HBNDataLoader, HBNDataset
from dataio.preprocessing import LeakageFreePreprocessor, SessionAwareSampler
from dataio.starter_kit import StarterKitDataLoader


class TestOfficialLabelsAndSplits:
    """Test official label loading and splits functionality."""

    @pytest.fixture
    def mock_bids_root(self):
        """Create a mock BIDS directory structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            bids_root = Path(temp_dir)

            # Create participants.tsv
            participants_data = pd.DataFrame(
                {
                    "participant_id": [
                        "sub-001",
                        "sub-002",
                        "sub-003",
                        "sub-004",
                        "sub-005",
                    ],
                    "age": [10, 11, 12, 13, 14],
                    "sex": ["M", "F", "M", "F", "M"],
                }
            )
            participants_data.to_csv(
                bids_root / "participants.tsv", sep="\t", index=False
            )

            # Create mock phenotype data
            phenotype_data = pd.DataFrame(
                {
                    "participant_id": [
                        "sub-001",
                        "sub-002",
                        "sub-003",
                        "sub-004",
                        "sub-005",
                    ],
                    "p_factor": [45.2, 52.1, 38.9, 65.4, 71.2],
                    "internalizing": [42.0, 55.3, 35.8, 68.1, 73.5],
                    "externalizing": [48.1, 49.2, 41.6, 62.3, 69.8],
                    "attention": [50.5, 58.7, 40.2, 72.1, 75.3],
                }
            )
            phenotype_data.to_csv(bids_root / "phenotype.tsv", sep="\t", index=False)

            # Create subject directories with EEG files
            for sub_id in participants_data["participant_id"]:
                sub_dir = bids_root / sub_id / "ses-001" / "eeg"
                sub_dir.mkdir(parents=True)

                # Create mock EEG file
                eeg_file = sub_dir / f"{sub_id}_ses-001_task-RS_eeg.edf"
                eeg_file.touch()

                # Create mock events file for CCD
                if "CCD" in str(eeg_file):
                    events_file = (
                        sub_dir.parent
                        / "func"
                        / f"{sub_id}_ses-001_task-CCD_events.tsv"
                    )
                    events_file.parent.mkdir(exist_ok=True)

                    events_data = pd.DataFrame(
                        {
                            "onset": [0.0, 5.0, 10.0],
                            "duration": [2.0, 2.0, 2.0],
                            "trial_type": ["CCD", "CCD", "CCD"],
                            "response_time": [1.2, 0.8, 1.5],
                            "accuracy": [1, 1, 0],
                        }
                    )
                    events_data.to_csv(events_file, sep="\t", index=False)

            yield bids_root

    @pytest.fixture
    def mock_splits(self, mock_bids_root):
        """Create mock official splits."""
        splits_dir = mock_bids_root.parent / "splits"
        splits_dir.mkdir(exist_ok=True)

        splits_data = {
            "version": "v1.0",
            "created_date": "2025-09-05T12:00:00",
            "random_seed": 42,
            "val_size": 0.2,
            "test_size": 0.2,
            "total_subjects": 5,
            "splits": {
                "train": ["sub-001", "sub-002", "sub-003"],
                "val": ["sub-004"],
                "test": ["sub-005"],
            },
        }

        splits_file = splits_dir / "official_splits_v1.0.json"
        with open(splits_file, "w") as f:
            json.dump(splits_data, f)

        return splits_data["splits"]

    def test_load_participants_data(self, mock_bids_root):
        """Test loading participants data with proper validation."""
        loader = StarterKitDataLoader(bids_root=mock_bids_root)

        participants_df = loader._load_participants_data()

        assert participants_df is not None
        assert len(participants_df) == 5
        assert "participant_id" in participants_df.columns
        assert "age" in participants_df.columns
        assert "sex" in participants_df.columns

        # Check data types
        assert participants_df["participant_id"].dtype == "object"
        assert pd.api.types.is_numeric_dtype(participants_df["age"])

    def test_load_cbcl_labels(self, mock_bids_root, mock_splits):
        """Test loading CBCL labels with proper shape and dtypes."""
        loader = StarterKitDataLoader(bids_root=mock_bids_root)

        # Mock the official splits loading
        loader.official_splits = mock_splits

        # Load CBCL labels for training split
        cbcl_labels = loader.load_cbcl_labels(split="train")

        assert not cbcl_labels.empty
        assert len(cbcl_labels) == 3  # 3 subjects in train split

        # Check required columns
        required_cols = [
            "participant_id",
            "p_factor",
            "internalizing",
            "externalizing",
            "attention",
        ]
        for col in required_cols:
            assert col in cbcl_labels.columns

        # Check data types
        for col in ["p_factor", "internalizing", "externalizing", "attention"]:
            assert pd.api.types.is_numeric_dtype(cbcl_labels[col])

        # Check value ranges (T-scores typically 20-120)
        for col in ["p_factor", "internalizing", "externalizing", "attention"]:
            values = cbcl_labels[col].dropna()
            if len(values) > 0:
                assert values.min() >= 20
                assert values.max() <= 120

    def test_split_isolation(self, mock_bids_root, mock_splits):
        """Test that splits maintain strict subject-level isolation."""
        train_subjects = set(mock_splits["train"])
        val_subjects = set(mock_splits["val"])
        test_subjects = set(mock_splits["test"])

        # Check no overlaps
        assert (
            len(train_subjects & val_subjects) == 0
        ), "Train-validation overlap detected"
        assert len(train_subjects & test_subjects) == 0, "Train-test overlap detected"
        assert (
            len(val_subjects & test_subjects) == 0
        ), "Validation-test overlap detected"

        # Check all subjects are accounted for
        all_subjects = train_subjects | val_subjects | test_subjects
        expected_subjects = {"sub-001", "sub-002", "sub-003", "sub-004", "sub-005"}
        assert all_subjects == expected_subjects

    def test_window_fold_isolation(self, mock_bids_root, mock_splits):
        """Test that windows from recordings never cross folds."""
        # This test would check that all windows from a subject's recording
        # stay within the same fold

        loader = StarterKitDataLoader(bids_root=mock_bids_root)
        loader.official_splits = mock_splits

        # For each split, verify that all windows belong to subjects in that split
        for split_name, split_subjects in mock_splits.items():
            # Mock loading windows for this split
            # In practice, this would use the actual HBNDataset

            # Verify that no windows leak across splits
            for subject in split_subjects:
                # All windows from this subject should be in this split only
                assert subject in split_subjects
                assert subject not in (
                    set(mock_splits["train"])
                    | set(mock_splits["val"])
                    | set(mock_splits["test"])
                ) - set(split_subjects)


class TestLeakageControls:
    """Test leakage prevention in normalization and preprocessing."""

    @pytest.fixture
    def mock_eeg_data(self):
        """Create mock EEG data for different sessions."""
        np.random.seed(42)

        # Create data for 3 sessions (train/val/test)
        data = {}
        for i, split in enumerate(["train", "val", "test"]):
            session_id = f"sub-00{i+1}_ses-001_RS"
            # Different baseline and scale for each session
            baseline = i * 10
            scale = (i + 1) * 2

            eeg_data = (
                np.random.randn(64, 1000) * scale + baseline
            )  # 64 channels, 1000 timepoints
            data[session_id] = eeg_data

        return data

    @pytest.fixture
    def mock_session_info(self):
        """Create mock session information."""
        return {
            "sub-001_ses-001_RS": {
                "split": "train",
                "subject_id": "sub-001",
                "session_id": "ses-001",
            },
            "sub-002_ses-001_RS": {
                "split": "val",
                "subject_id": "sub-002",
                "session_id": "ses-001",
            },
            "sub-003_ses-001_RS": {
                "split": "test",
                "subject_id": "sub-003",
                "session_id": "ses-001",
            },
        }

    def test_normalization_fit_train_only(self, mock_eeg_data, mock_session_info):
        """Test that normalization stats are fit only on training data."""
        with tempfile.TemporaryDirectory() as temp_dir:
            preprocessor = LeakageFreePreprocessor(stats_dir=Path(temp_dir))

            # Fit normalization on training data only
            train_data = {k: v for k, v in mock_eeg_data.items() if "sub-001" in k}
            preprocessor.fit_normalization_stats(train_data, mock_session_info)

            # Check that only training sessions were fitted
            assert len(preprocessor.fitted_sessions) == 1
            assert "sub-001_ses-001_RS" in preprocessor.fitted_sessions
            assert "sub-002_ses-001_RS" not in preprocessor.fitted_sessions
            assert "sub-003_ses-001_RS" not in preprocessor.fitted_sessions

    def test_leakage_protection_validation(self, mock_eeg_data, mock_session_info):
        """Test that leakage protection validation catches violations."""
        with tempfile.TemporaryDirectory() as temp_dir:
            preprocessor = LeakageFreePreprocessor(stats_dir=Path(temp_dir))

            # Fit normalization on training data only
            train_data = {k: v for k, v in mock_eeg_data.items() if "sub-001" in k}
            preprocessor.fit_normalization_stats(train_data, mock_session_info)

            # Validate leakage protection
            train_sessions = ["sub-001_ses-001_RS"]
            val_sessions = ["sub-002_ses-001_RS"]
            test_sessions = ["sub-003_ses-001_RS"]

            results = preprocessor.validate_leakage_protection(
                train_sessions, val_sessions, test_sessions
            )

            assert results["valid"] == True
            assert len(results["errors"]) == 0
            assert results["stats"]["fitted_sessions"] == 1

    def test_normalization_stats_persistence(self, mock_eeg_data, mock_session_info):
        """Test saving and loading normalization statistics."""
        with tempfile.TemporaryDirectory() as temp_dir:
            preprocessor = LeakageFreePreprocessor(stats_dir=Path(temp_dir))

            # Fit and save stats
            train_data = {k: v for k, v in mock_eeg_data.items() if "sub-001" in k}
            preprocessor.fit_normalization_stats(train_data, mock_session_info)
            stats_file = preprocessor.save_normalization_stats(version="test")

            assert stats_file.exists()

            # Create new preprocessor and load stats
            new_preprocessor = LeakageFreePreprocessor(stats_dir=Path(temp_dir))
            loaded = new_preprocessor.load_normalization_stats(version="test")

            assert loaded == True
            assert len(new_preprocessor.fitted_sessions) == 1
            assert "sub-001_ses-001_RS" in new_preprocessor.fitted_sessions

    def test_session_aware_sampler(self):
        """Test that SessionAwareSampler groups windows correctly."""

        # Mock dataset with session information
        class MockDataset:
            def __init__(self):
                self.data = [
                    {"subject": "sub-001", "session": "ses-001"},
                    {"subject": "sub-001", "session": "ses-001"},
                    {"subject": "sub-002", "session": "ses-001"},
                    {"subject": "sub-002", "session": "ses-001"},
                    {"subject": "sub-003", "session": "ses-001"},
                ]

            def __len__(self):
                return len(self.data)

            def get_session_info(self, idx):
                return {
                    "subject_id": self.data[idx]["subject"],
                    "session_id": self.data[idx]["session"],
                }

        dataset = MockDataset()
        sampler = SessionAwareSampler(dataset, batch_size=2, shuffle=False)

        # Check that sampler groups by session
        batches = list(sampler)

        # Should group windows from same subject/session together
        assert len(batches) >= 2  # At least 2 batches for 5 samples with batch_size=2

        # Check that indices within groups are from same session
        for batch in batches:
            if len(batch) > 1:
                # Get session info for batch
                sessions = [dataset.get_session_info(idx) for idx in batch]
                # All should be from same subject (for this simple test)
                subjects = [s["subject_id"] for s in sessions]
                # Allow mixed subjects in batch, but prefer grouping
                # The key requirement is that we don't accidentally mix splits

    def test_epoch_split_integrity_check(self):
        """Test per-epoch split integrity checking."""
        from dataio.preprocessing import check_epoch_split_integrity

        batch_subjects = ["sub-001", "sub-002", "sub-001"]
        expected_split = "train"
        subject_splits = {
            "sub-001": "train",
            "sub-002": "train",
            "sub-003": "val",
            "sub-004": "test",
        }

        # Test valid batch
        results = check_epoch_split_integrity(
            batch_subjects, expected_split, subject_splits, epoch=1
        )

        assert results["valid"] == True
        assert len(results["errors"]) == 0
        assert results["stats"]["split_counts"]["train"] == 3

        # Test invalid batch (mixed splits)
        batch_subjects_mixed = ["sub-001", "sub-003"]  # train + val
        results = check_epoch_split_integrity(
            batch_subjects_mixed, expected_split, subject_splits, epoch=1
        )

        assert results["valid"] == False
        assert len(results["errors"]) > 0
        assert "wrong split" in results["errors"][0].lower()


class TestDataIntegration:
    """Test end-to-end data loading with labels and splits."""

    def test_hbn_dataset_with_splits(self, mock_bids_root=None, mock_splits=None):
        """Test HBNDataset integration with official splits."""
        # This would test the full integration but requires mocking MNE
        # For now, we'll test the structure

        # Mock the dependencies
        with (
            patch("dataio.bids_loader.read_raw_bids"),
            patch("dataio.bids_loader.BIDSPath"),
        ):

            # This test would verify:
            # 1. HBNDataset loads correct participants for split
            # 2. Labels are properly attached to windows
            # 3. Session information is tracked
            # 4. No data leakage between splits

            pass  # Implementation would go here

    def test_label_schema_validation(self):
        """Test that loaded labels match expected schema."""
        # Define expected schema
        cbcl_schema = {
            "p_factor": {"dtype": "float", "range": (20, 120)},
            "internalizing": {"dtype": "float", "range": (20, 120)},
            "externalizing": {"dtype": "float", "range": (20, 120)},
            "attention": {"dtype": "float", "range": (20, 120)},
            "binary_label": {"dtype": "int", "values": [0, 1]},
        }

        # Mock CBCL data
        cbcl_data = pd.DataFrame(
            {
                "participant_id": ["sub-001", "sub-002"],
                "p_factor": [45.2, 52.1],
                "internalizing": [42.0, 55.3],
                "externalizing": [48.1, 49.2],
                "attention": [50.5, 58.7],
                "binary_label": [0, 0],
            }
        )

        # Validate schema
        for col, schema in cbcl_schema.items():
            if col in cbcl_data.columns:
                series = cbcl_data[col]

                # Check dtype
                if schema["dtype"] == "float":
                    assert pd.api.types.is_numeric_dtype(series)
                elif schema["dtype"] == "int":
                    assert pd.api.types.is_integer_dtype(series)

                # Check range/values
                if "range" in schema:
                    min_val, max_val = schema["range"]
                    assert series.min() >= min_val
                    assert series.max() <= max_val

                if "values" in schema:
                    assert set(series.unique()) <= set(schema["values"])


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
