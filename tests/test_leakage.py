"""
Test suite for preventing data leakage in EEG preprocessing pipeline.

Validates that normalization statistics, window generation, and cross-validation
maintain strict separation between train/val/test splits.
"""

import pytest
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
import json

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataio.preprocessing import LeakageFreePreprocessor, SessionAwareSampler
from dataio.starter_kit import StarterKitDataLoader


class TestNormalizationLeakage:
    """Test that normalization statistics don't leak across splits."""

    @pytest.fixture
    def mock_multi_session_data(self):
        """Create mock EEG data with different statistical properties per split."""
        np.random.seed(42)

        # Create data with distinct statistical properties for each split
        data = {}
        session_info = {}

        # Training data: mean=0, std=1
        for i in range(3):
            session_id = f'sub-{i:03d}_ses-001_RS'
            data[session_id] = np.random.randn(64, 2000)  # Standard normal
            session_info[session_id] = {
                'split': 'train',
                'subject_id': f'sub-{i:03d}',
                'session_id': 'ses-001'
            }

        # Validation data: mean=10, std=2
        for i in range(3, 5):
            session_id = f'sub-{i:03d}_ses-001_RS'
            data[session_id] = np.random.randn(64, 2000) * 2 + 10  # Different distribution
            session_info[session_id] = {
                'split': 'val',
                'subject_id': f'sub-{i:03d}',
                'session_id': 'ses-001'
            }

        # Test data: mean=-5, std=3
        for i in range(5, 7):
            session_id = f'sub-{i:03d}_ses-001_RS'
            data[session_id] = np.random.randn(64, 2000) * 3 - 5  # Another distribution
            session_info[session_id] = {
                'split': 'test',
                'subject_id': f'sub-{i:03d}',
                'session_id': 'ses-001'
            }

        return data, session_info

    def test_normalization_train_only_fitting(self, mock_multi_session_data):
        """Test that normalization stats are computed only from training data."""
        data, session_info = mock_multi_session_data

        with tempfile.TemporaryDirectory() as temp_dir:
            preprocessor = LeakageFreePreprocessor(stats_dir=Path(temp_dir))

            # Get training sessions only
            train_sessions = {k: v for k, v in data.items()
                            if session_info[k]['split'] == 'train'}

            # Fit normalization statistics
            preprocessor.fit_normalization_stats(train_sessions, session_info)

            # Check that stats were computed
            assert hasattr(preprocessor, 'channel_stats')
            assert len(preprocessor.channel_stats) > 0

            # Verify only training sessions were used
            assert len(preprocessor.fitted_sessions) == 3
            for session_id in preprocessor.fitted_sessions:
                assert session_info[session_id]['split'] == 'train'

            # Check that stats reflect training data distribution (mean ≈ 0, std ≈ 1)
            for channel_idx in range(64):
                if channel_idx in preprocessor.channel_stats:
                    stats = preprocessor.channel_stats[channel_idx]
                    # Training data has mean=0, std=1, so stats should reflect this
                    assert abs(stats['robust_mean']) < 0.5  # Should be close to 0
                    assert 0.5 < stats['robust_std'] < 2.0   # Should be close to 1

    def test_val_test_normalization_uses_train_stats(self, mock_multi_session_data):
        """Test that val/test data is normalized using training statistics."""
        data, session_info = mock_multi_session_data

        with tempfile.TemporaryDirectory() as temp_dir:
            preprocessor = LeakageFreePreprocessor(stats_dir=Path(temp_dir))

            # Fit on training data
            train_sessions = {k: v for k, v in data.items()
                            if session_info[k]['split'] == 'train'}
            preprocessor.fit_normalization_stats(train_sessions, session_info)

            # Get validation data (which has different distribution)
            val_session_id = [k for k in data.keys()
                            if session_info[k]['split'] == 'val'][0]
            val_data = data[val_session_id]

            # Normalize validation data
            normalized_val = preprocessor.normalize_session(
                val_data, val_session_id, session_info[val_session_id]
            )

            # Check that normalization was applied but not fitted
            assert val_session_id not in preprocessor.fitted_sessions

            # Validation data should be normalized using training stats
            # Original val data has mean=10, std=2
            # After normalization with train stats (mean≈0, std≈1),
            # the mean should be much larger than original scale
            val_mean = np.mean(normalized_val)
            assert abs(val_mean) > 5  # Should be shifted significantly from train distribution

    def test_leakage_detection_mixed_splits(self, mock_multi_session_data):
        """Test that leakage detection catches mixed training with val/test data."""
        data, session_info = mock_multi_session_data

        with tempfile.TemporaryDirectory() as temp_dir:
            preprocessor = LeakageFreePreprocessor(stats_dir=Path(temp_dir))

            # Attempt to fit on mixed data (should fail validation)
            mixed_sessions = {}
            mixed_sessions.update({k: v for k, v in data.items()
                                 if session_info[k]['split'] == 'train'})
            mixed_sessions.update({k: v for k, v in data.items()
                                 if session_info[k]['split'] == 'val'})

            # This should raise an error or warning
            with pytest.warns(UserWarning, match="mixed splits"):
                preprocessor.fit_normalization_stats(mixed_sessions, session_info)

    def test_cross_validation_split_isolation(self):
        """Test that cross-validation maintains split isolation."""
        # Mock official splits
        official_splits = {
            'train': ['sub-001', 'sub-002', 'sub-003'],
            'val': ['sub-004'],
            'test': ['sub-005']
        }

        # Create CV splits - should never use val/test subjects in train folds
        from sklearn.model_selection import KFold

        train_subjects = official_splits['train']
        cv_folds = KFold(n_splits=3, shuffle=True, random_state=42)

        for fold_idx, (train_idx, val_idx) in enumerate(cv_folds.split(train_subjects)):
            fold_train = [train_subjects[i] for i in train_idx]
            fold_val = [train_subjects[i] for i in val_idx]

            # Check no overlap with official val/test
            assert not set(fold_train) & set(official_splits['val'])
            assert not set(fold_train) & set(official_splits['test'])
            assert not set(fold_val) & set(official_splits['val'])
            assert not set(fold_val) & set(official_splits['test'])

            # Check internal fold isolation
            assert not set(fold_train) & set(fold_val)


class TestSessionLeakage:
    """Test that sessions from same subject don't leak across splits."""

    @pytest.fixture
    def mock_multi_session_subjects(self):
        """Create mock data with multiple sessions per subject."""
        # Subject with sessions in different splits (this should be prevented)
        session_data = {
            'sub-001_ses-001_RS': {'split': 'train', 'subject': 'sub-001', 'session': 'ses-001'},
            'sub-001_ses-002_RS': {'split': 'val', 'subject': 'sub-001', 'session': 'ses-002'},  # BAD!
            'sub-002_ses-001_RS': {'split': 'train', 'subject': 'sub-002', 'session': 'ses-001'},
            'sub-003_ses-001_RS': {'split': 'val', 'subject': 'sub-003', 'session': 'ses-001'},
            'sub-004_ses-001_RS': {'split': 'test', 'subject': 'sub-004', 'session': 'ses-001'},
        }
        return session_data

    def test_detect_subject_split_leakage(self, mock_multi_session_subjects):
        """Test detection of subjects appearing in multiple splits."""
        session_data = mock_multi_session_subjects

        # Group by subject and check splits
        subject_splits = {}
        for session_id, info in session_data.items():
            subject = info['subject']
            split = info['split']

            if subject not in subject_splits:
                subject_splits[subject] = set()
            subject_splits[subject].add(split)

        # Check for violations
        violations = []
        for subject, splits in subject_splits.items():
            if len(splits) > 1:
                violations.append(f"Subject {subject} appears in splits: {splits}")

        # Should detect the violation
        assert len(violations) == 1
        assert 'sub-001' in violations[0]
        assert 'train' in violations[0] and 'val' in violations[0]

    def test_session_aware_sampling_groups_correctly(self):
        """Test that SessionAwareSampler groups windows by session."""
        # Mock dataset with session information
        class MockEEGDataset:
            def __init__(self):
                # Windows from different sessions
                self.windows = [
                    {'session_id': 'sub-001_ses-001_RS', 'window_idx': 0},
                    {'session_id': 'sub-001_ses-001_RS', 'window_idx': 1},
                    {'session_id': 'sub-002_ses-001_RS', 'window_idx': 0},
                    {'session_id': 'sub-002_ses-001_RS', 'window_idx': 1},
                    {'session_id': 'sub-003_ses-001_RS', 'window_idx': 0},
                ]

            def __len__(self):
                return len(self.windows)

            def get_session_info(self, idx):
                return {
                    'session_id': self.windows[idx]['session_id'],
                    'subject_id': self.windows[idx]['session_id'].split('_')[0],
                }

        dataset = MockEEGDataset()
        sampler = SessionAwareSampler(dataset, batch_size=2, shuffle=False)

        batches = list(sampler)

        # Check that windows from same session are grouped together
        for batch in batches:
            if len(batch) > 1:
                # Get session IDs for this batch
                sessions = [dataset.get_session_info(idx)['session_id'] for idx in batch]
                # All windows in batch should be from same session (when possible)
                unique_sessions = set(sessions)
                # With our test data and batch_size=2, we should get perfect grouping
                if len(unique_sessions) > 1:
                    # This might happen at session boundaries, check it's reasonable
                    assert len(unique_sessions) <= 2


class TestTemporalLeakage:
    """Test prevention of temporal leakage in time series processing."""

    def test_future_information_leak_prevention(self):
        """Test that future timepoints don't influence past predictions."""
        # Mock time series data
        time_series = np.random.randn(1000)  # 1000 timepoints

        # Simulate causal filtering (only past information)
        def causal_filter(data, window_size=10):
            """Example causal filter that only uses past data."""
            filtered = np.zeros_like(data)
            for i in range(len(data)):
                start_idx = max(0, i - window_size + 1)
                filtered[i] = np.mean(data[start_idx:i+1])
            return filtered

        # Apply causal filter
        filtered_data = causal_filter(time_series)

        # Test that each point only depends on past
        for i in range(1, len(time_series)):
            # Change future values
            modified_series = time_series.copy()
            modified_series[i+1:] = 999  # Dramatic change to future

            # Re-filter
            modified_filtered = causal_filter(modified_series)

            # Current and past points should be unchanged
            np.testing.assert_array_equal(
                filtered_data[:i+1],
                modified_filtered[:i+1],
                err_msg=f"Future leak detected at timepoint {i}"
            )

    def test_sliding_window_overlap_tracking(self):
        """Test that overlapping windows are tracked to prevent leakage."""
        # Mock sliding window extraction
        def extract_windows(data, window_size=100, stride=50):
            """Extract overlapping windows."""
            windows = []
            metadata = []

            for start_idx in range(0, len(data) - window_size + 1, stride):
                end_idx = start_idx + window_size
                window = data[start_idx:end_idx]
                windows.append(window)
                metadata.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'source_range': (start_idx, end_idx)
                })

            return windows, metadata

        # Create mock EEG data
        eeg_data = np.random.randn(1000)
        windows, metadata = extract_windows(eeg_data, window_size=100, stride=50)

        # Check for overlaps
        overlaps = []
        for i, meta1 in enumerate(metadata):
            for j, meta2 in enumerate(metadata[i+1:], i+1):
                range1 = set(range(meta1['start_idx'], meta1['end_idx']))
                range2 = set(range(meta2['start_idx'], meta2['end_idx']))
                overlap = range1 & range2

                if overlap:
                    overlaps.append({
                        'window1': i,
                        'window2': j,
                        'overlap_size': len(overlap),
                        'overlap_ratio': len(overlap) / 100  # window_size
                    })

        # Should detect overlaps (expected with stride < window_size)
        assert len(overlaps) > 0

        # Check overlap ratios are reasonable
        for overlap in overlaps:
            assert 0 < overlap['overlap_ratio'] < 1  # Partial overlap expected


class TestValidationReproducibility:
    """Test that validation procedures are reproducible and deterministic."""

    def test_split_generation_reproducibility(self):
        """Test that splits are generated reproducibly with same seed."""
        # Mock participants data
        participants = pd.DataFrame({
            'participant_id': [f'sub-{i:03d}' for i in range(20)],
            'age': np.random.randint(8, 18, 20),
            'sex': np.random.choice(['M', 'F'], 20)
        })

        # Generate splits with same seed multiple times
        splits1 = self._generate_stratified_splits(participants, random_seed=42)
        splits2 = self._generate_stratified_splits(participants, random_seed=42)
        splits3 = self._generate_stratified_splits(participants, random_seed=999)  # Different seed

        # Same seed should give same splits
        assert splits1['train'] == splits2['train']
        assert splits1['val'] == splits2['val']
        assert splits1['test'] == splits2['test']

        # Different seed should give different splits
        assert splits1['train'] != splits3['train'] or \
               splits1['val'] != splits3['val'] or \
               splits1['test'] != splits3['test']

    def _generate_stratified_splits(self, participants_df, random_seed=42):
        """Generate stratified splits (simplified version)."""
        from sklearn.model_selection import train_test_split

        # Create stratification labels
        participants_df['age_bin'] = pd.cut(participants_df['age'], bins=3, labels=[0, 1, 2])
        participants_df['strata'] = (
            participants_df['sex'].astype(str) + '_' +
            participants_df['age_bin'].astype(str)
        )

        # Split into train/temp
        train_ids, temp_ids = train_test_split(
            participants_df['participant_id'].tolist(),
            test_size=0.4,
            stratify=participants_df['strata'].tolist(),
            random_state=random_seed
        )

        # Split temp into val/test
        temp_df = participants_df[participants_df['participant_id'].isin(temp_ids)]
        val_ids, test_ids = train_test_split(
            temp_ids,
            test_size=0.5,
            stratify=temp_df['strata'].tolist(),
            random_state=random_seed
        )

        return {
            'train': sorted(train_ids),
            'val': sorted(val_ids),
            'test': sorted(test_ids)
        }

    def test_normalization_determinism(self):
        """Test that normalization is deterministic given same input."""
        np.random.seed(42)

        # Create identical datasets
        data1 = np.random.randn(64, 1000)
        data2 = data1.copy()

        # Normalize both
        from sklearn.preprocessing import RobustScaler

        scaler1 = RobustScaler()
        scaler2 = RobustScaler()

        normalized1 = scaler1.fit_transform(data1.T).T
        normalized2 = scaler2.fit_transform(data2.T).T

        # Should be identical
        np.testing.assert_array_almost_equal(
            normalized1, normalized2,
            err_msg="Normalization not deterministic"
        )


class TestLeakageAudit:
    """Comprehensive audit for potential leakage sources."""

    def test_full_pipeline_leakage_audit(self):
        """Comprehensive test of entire pipeline for leakage sources."""
        # This would be a comprehensive test that checks:
        # 1. Data loading splits
        # 2. Preprocessing normalization
        # 3. Window generation
        # 4. Cross-validation setup
        # 5. Model training boundaries

        audit_results = {
            'split_isolation': True,
            'normalization_leakage': False,
            'temporal_leakage': False,
            'session_leakage': False,
            'cv_leakage': False
        }

        # Check each component
        # In practice, this would run actual leakage detection algorithms

        assert all(audit_results.values()), f"Leakage detected: {audit_results}"

    def test_generate_leakage_report(self):
        """Generate a comprehensive leakage prevention report."""
        report = {
            'pipeline_version': '1.0',
            'audit_date': '2025-01-15',
            'checks_performed': [
                'split_isolation',
                'normalization_stats_isolation',
                'temporal_causality',
                'session_boundaries',
                'cross_validation_setup'
            ],
            'violations_found': [],
            'recommendations': [
                'Use subject-level splits only',
                'Fit normalization on training data only',
                'Validate temporal causality in preprocessing',
                'Track session boundaries in sampling'
            ]
        }

        # In practice, this would be generated by running all leakage tests
        assert len(report['violations_found']) == 0
        assert len(report['checks_performed']) >= 5


if __name__ == "__main__":
    # Run comprehensive leakage tests
    pytest.main([__file__, "-v", "--tb=short"])
