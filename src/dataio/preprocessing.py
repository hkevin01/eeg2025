"""
Preprocessing pipeline with strict leakage controls for EEG Foundation Challenge 2025.

This module implements:
- Per-session robust z-scoring fitted ONLY on training data
- Leakage-free normalization with saved statistics
- Window grouping to prevent fold contamination
- Session-aware sampling for proper minibatch construction

Features comprehensive error handling and validation to ensure no data leakage
between train/validation/test splits.
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler, Dataset
from sklearn.preprocessing import RobustScaler
from scipy import stats
import mne

logger = logging.getLogger(__name__)


@dataclass
class NormalizationStats:
    """Store normalization statistics for leakage-free preprocessing."""

    median: np.ndarray
    scale: np.ndarray
    channel_names: List[str]
    split: str
    session_info: Dict[str, Any] = field(default_factory=dict)

    def save(self, filepath: Path):
        """Save normalization stats to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Saved normalization stats to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> 'NormalizationStats':
        """Load normalization stats from disk."""
        with open(filepath, 'rb') as f:
            stats = pickle.load(f)
        logger.info(f"Loaded normalization stats from {filepath}")
        return stats


class LeakageFreePreprocessor:
    """
    Preprocessor that ensures no data leakage between train/val/test splits.

    Key features:
    - Normalization statistics computed ONLY on training data
    - Per-session robust z-scoring
    - Saved statistics for consistent preprocessing
    - Validation checks to prevent leakage
    """

    def __init__(
        self,
        stats_dir: Path,
        bandpass_freqs: Optional[Tuple[float, float]] = (0.1, 40.0),
        notch_freq: Optional[float] = 60.0,
        reref_method: str = "CAR",
        robust_scaling: bool = True
    ):
        """
        Initialize leakage-free preprocessor.

        Args:
            stats_dir: Directory to save/load normalization statistics
            bandpass_freqs: Bandpass filter frequencies (low, high)
            notch_freq: Notch filter frequency (line noise)
            reref_method: Re-referencing method ('CAR', 'average', None)
            robust_scaling: Use robust scaling (less sensitive to outliers)
        """
        self.stats_dir = Path(stats_dir)
        self.stats_dir.mkdir(parents=True, exist_ok=True)

        self.bandpass_freqs = bandpass_freqs
        self.notch_freq = notch_freq
        self.reref_method = reref_method
        self.robust_scaling = robust_scaling

        # Store normalization statistics by session
        self.normalization_stats = {}
        self.fitted_sessions = set()

        logger.info(f"Initialized leakage-free preprocessor with stats dir: {stats_dir}")

    def fit_normalization_stats(
        self,
        train_data: Dict[str, np.ndarray],
        session_info: Dict[str, Dict[str, Any]]
    ) -> None:
        """
        Fit normalization statistics on training data ONLY.

        Args:
            train_data: Dictionary mapping session_id -> EEG data (channels x time)
            session_info: Dictionary mapping session_id -> session metadata
        """
        logger.info("Fitting normalization statistics on training data...")

        # Validate that we're only fitting on training data
        for session_id in train_data.keys():
            if session_id in session_info:
                split = session_info[session_id].get('split', 'unknown')
                if split != 'train':
                    raise ValueError(f"Attempting to fit normalization on non-training data: {session_id} (split: {split})")

        # Fit per-session normalization
        for session_id, eeg_data in train_data.items():
            logger.debug(f"Fitting normalization for session {session_id}")

            # Validate data shape
            if eeg_data.ndim != 2:
                raise ValueError(f"Expected 2D data (channels x time), got {eeg_data.shape}")

            n_channels, n_timepoints = eeg_data.shape

            if n_timepoints == 0:
                logger.warning(f"Empty data for session {session_id}")
                continue

            # Compute robust statistics per channel
            if self.robust_scaling:
                # Use robust scaling (median and IQR)
                median = np.median(eeg_data, axis=1)
                q25, q75 = np.percentile(eeg_data, [25, 75], axis=1)
                scale = q75 - q25

                # Avoid division by zero
                scale = np.where(scale == 0, 1.0, scale)

            else:
                # Use standard scaling (mean and std)
                median = np.mean(eeg_data, axis=1)
                scale = np.std(eeg_data, axis=1)

                # Avoid division by zero
                scale = np.where(scale == 0, 1.0, scale)

            # Get channel names if available from session info
            channel_names = session_info.get(session_id, {}).get('channel_names',
                                           [f'ch_{i}' for i in range(n_channels)])

            # Store normalization statistics
            norm_stats = NormalizationStats(
                median=median,
                scale=scale,
                channel_names=channel_names,
                split='train',
                session_info=session_info.get(session_id, {})
            )

            self.normalization_stats[session_id] = norm_stats
            self.fitted_sessions.add(session_id)

            logger.debug(f"Fitted stats for {session_id}: "
                        f"median range [{median.min():.3f}, {median.max():.3f}], "
                        f"scale range [{scale.min():.3f}, {scale.max():.3f}]")

        logger.info(f"Fitted normalization statistics for {len(self.normalization_stats)} training sessions")

    def save_normalization_stats(self, version: str = "v1.0") -> Path:
        """Save normalization statistics to disk."""
        stats_file = self.stats_dir / f"normalization_stats_{version}.pkl"

        stats_data = {
            'stats': self.normalization_stats,
            'fitted_sessions': self.fitted_sessions,
            'config': {
                'bandpass_freqs': self.bandpass_freqs,
                'notch_freq': self.notch_freq,
                'reref_method': self.reref_method,
                'robust_scaling': self.robust_scaling
            },
            'version': version,
            'created_date': pd.Timestamp.now().isoformat()
        }

        with open(stats_file, 'wb') as f:
            pickle.dump(stats_data, f)

        logger.info(f"Saved normalization statistics to {stats_file}")
        return stats_file

    def load_normalization_stats(self, version: str = "v1.0") -> bool:
        """Load normalization statistics from disk."""
        stats_file = self.stats_dir / f"normalization_stats_{version}.pkl"

        if not stats_file.exists():
            logger.warning(f"Normalization stats file not found: {stats_file}")
            return False

        try:
            with open(stats_file, 'rb') as f:
                stats_data = pickle.load(f)

            self.normalization_stats = stats_data['stats']
            self.fitted_sessions = stats_data['fitted_sessions']

            # Validate config compatibility
            config = stats_data.get('config', {})
            if config.get('robust_scaling') != self.robust_scaling:
                logger.warning("Loaded stats were created with different scaling method")

            logger.info(f"Loaded normalization statistics for {len(self.normalization_stats)} sessions")
            return True

        except Exception as e:
            logger.error(f"Error loading normalization stats: {e}")
            return False

    def apply_normalization(
        self,
        eeg_data: np.ndarray,
        session_id: str,
        split: str
    ) -> np.ndarray:
        """
        Apply normalization using pre-fitted statistics.

        Args:
            eeg_data: EEG data to normalize (channels x time)
            session_id: Session identifier
            split: Data split ('train', 'val', 'test')

        Returns:
            Normalized EEG data
        """
        # Validate that normalization stats are available
        if session_id not in self.normalization_stats:
            # For validation/test data, we need to find a suitable reference session
            # This should use stats from a training session with similar characteristics
            reference_session = self._find_reference_session(session_id, split)
            if reference_session is None:
                raise ValueError(f"No normalization stats available for session {session_id} and no suitable reference found")
            logger.debug(f"Using reference session {reference_session} for {session_id}")
            stats = self.normalization_stats[reference_session]
        else:
            stats = self.normalization_stats[session_id]

        # Validate data leakage protection
        if split in ['val', 'test'] and session_id in self.fitted_sessions:
            logger.warning(f"Session {session_id} was used for fitting but is now in {split} split - potential leakage!")

        # Apply normalization
        normalized_data = (eeg_data - stats.median[:, np.newaxis]) / stats.scale[:, np.newaxis]

        # Validate output
        if not np.isfinite(normalized_data).all():
            logger.warning(f"Non-finite values detected after normalization for session {session_id}")
            # Replace non-finite values with zeros
            normalized_data = np.where(np.isfinite(normalized_data), normalized_data, 0.0)

        logger.debug(f"Applied normalization to session {session_id}: "
                    f"output range [{normalized_data.min():.3f}, {normalized_data.max():.3f}]")

        return normalized_data

    def _find_reference_session(self, session_id: str, split: str) -> Optional[str]:
        """Find a suitable reference session for normalization."""
        # For now, use the first available training session
        # In a more sophisticated implementation, this could match by subject characteristics
        training_sessions = [sid for sid in self.fitted_sessions
                           if self.normalization_stats[sid].split == 'train']

        if training_sessions:
            return training_sessions[0]

        return None

    def preprocess_raw(
        self,
        raw: mne.io.Raw,
        session_id: str,
        split: str,
        apply_normalization: bool = True
    ) -> mne.io.Raw:
        """
        Preprocess raw MNE object with leakage-free normalization.

        Args:
            raw: MNE Raw object
            session_id: Session identifier
            split: Data split
            apply_normalization: Whether to apply normalization

        Returns:
            Preprocessed Raw object
        """
        logger.debug(f"Preprocessing raw data for session {session_id}")

        # Copy to avoid modifying original
        raw_copy = raw.copy()

        # Apply bandpass filter
        if self.bandpass_freqs is not None:
            raw_copy.filter(
                l_freq=self.bandpass_freqs[0],
                h_freq=self.bandpass_freqs[1],
                fir_design='firwin',
                verbose=False
            )
            logger.debug(f"Applied bandpass filter: {self.bandpass_freqs}")

        # Apply notch filter
        if self.notch_freq is not None:
            raw_copy.notch_filter(
                freqs=self.notch_freq,
                verbose=False
            )
            logger.debug(f"Applied notch filter: {self.notch_freq}")

        # Apply re-referencing
        if self.reref_method == "CAR":
            raw_copy.set_eeg_reference(ref_channels='average', verbose=False)
            logger.debug("Applied common average reference")
        elif self.reref_method == "average":
            raw_copy.set_eeg_reference(ref_channels='average', verbose=False)
            logger.debug("Applied average reference")

        # Apply normalization
        if apply_normalization:
            # Get data
            data = raw_copy.get_data()

            # Apply normalization
            normalized_data = self.apply_normalization(data, session_id, split)

            # Update the raw object with normalized data
            raw_copy._data = normalized_data

            logger.debug(f"Applied normalization to session {session_id}")

        return raw_copy

    def validate_leakage_protection(
        self,
        train_sessions: List[str],
        val_sessions: List[str],
        test_sessions: List[str]
    ) -> Dict[str, Any]:
        """
        Validate that no data leakage occurred during preprocessing.

        Returns:
            Dictionary with validation results
        """
        logger.info("Validating leakage protection...")

        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }

        # Check that fitted sessions are only from training
        non_train_fitted = self.fitted_sessions - set(train_sessions)
        if non_train_fitted:
            results['valid'] = False
            results['errors'].append(f"Normalization fitted on non-training sessions: {non_train_fitted}")

        # Check session isolation
        all_sessions = set(train_sessions + val_sessions + test_sessions)
        overlaps = []

        for i, (name1, sessions1) in enumerate([('train', train_sessions), ('val', val_sessions), ('test', test_sessions)]):
            for name2, sessions2 in [('train', train_sessions), ('val', val_sessions), ('test', test_sessions)][i+1:]:
                overlap = set(sessions1) & set(sessions2)
                if overlap:
                    overlaps.append(f"{name1}-{name2}: {overlap}")

        if overlaps:
            results['valid'] = False
            results['errors'].append(f"Session overlaps detected: {overlaps}")

        # Statistics
        results['stats'] = {
            'fitted_sessions': len(self.fitted_sessions),
            'train_sessions': len(train_sessions),
            'val_sessions': len(val_sessions),
            'test_sessions': len(test_sessions),
            'total_sessions': len(all_sessions)
        }

        if results['valid']:
            logger.info("✅ Leakage protection validation passed")
        else:
            logger.error(f"❌ Leakage protection validation failed: {results['errors']}")

        return results


class SessionAwareSampler(Sampler):
    """
    Sampler that groups windows by session/subject to prevent fold contamination.

    Ensures that minibatches don't accidentally mix data from different splits
    by grouping windows from the same session together.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False
    ):
        """
        Initialize session-aware sampler.

        Args:
            dataset: Dataset with session/subject grouping information
            batch_size: Batch size for grouping
            shuffle: Whether to shuffle the order of groups
            drop_last: Whether to drop the last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        # Build session/subject groups
        self.groups = self._build_groups()

        logger.info(f"Initialized SessionAwareSampler with {len(self.groups)} groups")

    def _build_groups(self) -> Dict[str, List[int]]:
        """Build groups of indices by session/subject."""
        groups = {}

        for idx in range(len(self.dataset)):
            # Get session/subject information from dataset
            if hasattr(self.dataset, 'get_session_info'):
                session_info = self.dataset.get_session_info(idx)
                group_key = f"{session_info.get('subject_id', 'unknown')}_{session_info.get('session_id', 'unknown')}"
            else:
                # Fallback grouping
                group_key = f"group_{idx // self.batch_size}"

            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(idx)

        # Log group statistics
        group_sizes = [len(indices) for indices in groups.values()]
        logger.debug(f"Group sizes: min={min(group_sizes)}, max={max(group_sizes)}, mean={np.mean(group_sizes):.1f}")

        return groups

    def __iter__(self):
        """Iterate over grouped indices."""
        # Get group keys
        group_keys = list(self.groups.keys())

        # Shuffle groups if requested
        if self.shuffle:
            np.random.shuffle(group_keys)

        # Yield indices grouped by session/subject
        batch_indices = []

        for group_key in group_keys:
            group_indices = self.groups[group_key].copy()

            # Shuffle within group if requested
            if self.shuffle:
                np.random.shuffle(group_indices)

            # Add to current batch
            batch_indices.extend(group_indices)

            # Yield complete batches
            while len(batch_indices) >= self.batch_size:
                yield batch_indices[:self.batch_size]
                batch_indices = batch_indices[self.batch_size:]

        # Handle remaining indices
        if batch_indices and not self.drop_last:
            yield batch_indices

    def __len__(self):
        """Return number of batches."""
        total_samples = sum(len(indices) for indices in self.groups.values())

        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


def validate_preprocessing_pipeline(
    preprocessor: LeakageFreePreprocessor,
    train_data: Dict[str, np.ndarray],
    val_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    session_info: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate the entire preprocessing pipeline for leakage protection.

    Args:
        preprocessor: The preprocessor to validate
        train_data: Training data by session
        val_data: Validation data by session
        test_data: Test data by session
        session_info: Session metadata

    Returns:
        Validation results dictionary
    """
    logger.info("Validating preprocessing pipeline...")

    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    try:
        # Validate session isolation
        train_sessions = set(train_data.keys())
        val_sessions = set(val_data.keys())
        test_sessions = set(test_data.keys())

        # Check for overlaps
        train_val_overlap = train_sessions & val_sessions
        train_test_overlap = train_sessions & test_sessions
        val_test_overlap = val_sessions & test_sessions

        if train_val_overlap:
            results['errors'].append(f"Train-validation session overlap: {train_val_overlap}")
            results['valid'] = False

        if train_test_overlap:
            results['errors'].append(f"Train-test session overlap: {train_test_overlap}")
            results['valid'] = False

        if val_test_overlap:
            results['errors'].append(f"Validation-test session overlap: {val_test_overlap}")
            results['valid'] = False

        # Validate normalization was fitted only on training data
        leakage_results = preprocessor.validate_leakage_protection(
            list(train_sessions), list(val_sessions), list(test_sessions)
        )

        if not leakage_results['valid']:
            results['errors'].extend(leakage_results['errors'])
            results['valid'] = False

        results['warnings'].extend(leakage_results.get('warnings', []))

        # Statistics
        results['stats'] = {
            'train_sessions': len(train_sessions),
            'val_sessions': len(val_sessions),
            'test_sessions': len(test_sessions),
            'fitted_sessions': len(preprocessor.fitted_sessions),
            'total_data_points': {
                'train': sum(data.shape[1] for data in train_data.values()),
                'val': sum(data.shape[1] for data in val_data.values()),
                'test': sum(data.shape[1] for data in test_data.values())
            }
        }

        if results['valid']:
            logger.info("✅ Preprocessing pipeline validation passed")
        else:
            logger.error(f"❌ Preprocessing pipeline validation failed: {results['errors']}")

        return results

    except Exception as e:
        logger.error(f"Error during preprocessing validation: {e}")
        results['valid'] = False
        results['errors'].append(str(e))
        return results


# Utility functions for epoch checking during training

def check_epoch_split_integrity(
    batch_subjects: List[str],
    expected_split: str,
    subject_splits: Dict[str, str],
    epoch: int
) -> Dict[str, Any]:
    """
    Check that all subjects in a batch belong to the expected split.

    Args:
        batch_subjects: List of subject IDs in the current batch
        expected_split: Expected split name ('train', 'val', 'test')
        subject_splits: Mapping of subject_id -> split
        epoch: Current epoch number

    Returns:
        Validation results
    """
    results = {
        'valid': True,
        'errors': [],
        'warnings': [],
        'stats': {}
    }

    # Count subjects by split
    split_counts = {}
    wrong_split_subjects = []

    for subject_id in batch_subjects:
        actual_split = subject_splits.get(subject_id, 'unknown')

        if actual_split not in split_counts:
            split_counts[actual_split] = 0
        split_counts[actual_split] += 1

        if actual_split != expected_split:
            wrong_split_subjects.append((subject_id, actual_split))

    # Validate
    if wrong_split_subjects:
        results['valid'] = False
        results['errors'].append(
            f"Epoch {epoch}: Found subjects from wrong split in {expected_split} batch: {wrong_split_subjects}"
        )

    results['stats'] = {
        'epoch': epoch,
        'expected_split': expected_split,
        'split_counts': split_counts,
        'total_subjects': len(batch_subjects),
        'unique_subjects': len(set(batch_subjects))
    }

    return results


def log_split_statistics(
    epoch: int,
    split_name: str,
    batch_subjects: List[str],
    subject_splits: Dict[str, str]
) -> None:
    """Log statistics about split integrity for the current epoch."""
    unique_subjects = set(batch_subjects)

    # Count by actual split
    actual_splits = {}
    for subject in unique_subjects:
        actual_split = subject_splits.get(subject, 'unknown')
        if actual_split not in actual_splits:
            actual_splits[actual_split] = 0
        actual_splits[actual_split] += 1

    logger.info(f"Epoch {epoch} {split_name}: {len(unique_subjects)} unique subjects, splits: {actual_splits}")

    # Warn if contamination detected
    expected_count = actual_splits.get(split_name, 0)
    contamination_count = sum(count for split, count in actual_splits.items() if split != split_name)

    if contamination_count > 0:
        logger.warning(f"Epoch {epoch} {split_name}: Detected {contamination_count} subjects from other splits!")
