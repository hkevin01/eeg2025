"""
BIDS-compliant data loader for the Healthy Brain Network EEG dataset.

This module provides functionality to load and preprocess EEG data
following the Brain Imaging Data Structure (BIDS) standard with
challenge-compliant labels and splits integration.

Features:
- Official label loading (CCD and CBCL targets)
- Strict subject-level split isolation
- Leakage-free preprocessing integration
- Session-aware data organization
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from mne_bids import BIDSPath, read_raw_bids

from .starter_kit import StarterKitDataLoader
from .preprocessing import LeakageFreePreprocessor, SessionAwareSampler

logger = logging.getLogger(__name__)


class HBNDataset(Dataset):
    """
    PyTorch Dataset for HBN EEG data with challenge-compliant labels and splits.

    Features:
    - Integration with official splits (strict subject-level isolation)
    - CCD and CBCL label loading per challenge specifications
    - Leakage-free preprocessing
    - Session-aware data organization

    Args:
        bids_root: Path to BIDS root directory
        split: Data split to load ('train', 'val', 'test')
        tasks: List of task names to include
        window_length: Length of each window in seconds
        overlap: Overlap between windows (0-1)
        preprocessing_config: Configuration for preprocessing steps
        splits_version: Version of official splits to use
        load_labels: Whether to load challenge labels
    """

    def __init__(
        self,
        bids_root: Union[str, Path],
        split: str,
        tasks: List[str],
        window_length: float = 2.0,
        overlap: float = 0.5,
        preprocessing_config: Optional[Dict] = None,
        splits_version: str = "v1.0",
        load_labels: bool = True,
        transform=None,
    ):
        self.bids_root = Path(bids_root)
        self.split = split
        self.tasks = tasks
        self.window_length = window_length
        self.overlap = overlap
        self.preprocessing_config = preprocessing_config or {}
        self.splits_version = splits_version
        self.load_labels = load_labels
        self.transform = transform

        # Validate split
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be one of ['train', 'val', 'test']")

        # Initialize challenge data loader
        self.starter_kit = StarterKitDataLoader(bids_root=self.bids_root)

        # Load official splits
        self.official_splits = self._load_official_splits()
        self.participants = self.official_splits[split]

        logger.info(f"Loaded {len(self.participants)} participants for {split} split")

        # Initialize preprocessor if needed
        self.preprocessor = None
        if self.preprocessing_config:
            stats_dir = Path(self.preprocessing_config.get('stats_dir', 'preprocessing_stats'))
            self.preprocessor = LeakageFreePreprocessor(stats_dir=stats_dir)

        # Load labels if requested
        self.labels = {}
        if self.load_labels:
            self._load_challenge_labels()

        # Build index of all available files
        self.file_index = self._build_file_index()

        # Extract windows from all files
        self.windows = self._extract_windows()

        logger.info(f"Initialized HBNDataset with {len(self.windows)} windows for {split} split")
        logger.info(f"Initialized HBNDataset with {len(self.windows)} windows for {split} split")

    def _load_official_splits(self) -> Dict[str, List[str]]:
        """Load official challenge splits with validation."""
        try:
            # Try to load from starter kit first
            splits = self.starter_kit.load_official_splits(version=self.splits_version)

            if splits is None:
                # Fallback: try to load from splits directory
                splits_dir = Path("data/splits")
                splits_file = splits_dir / f"official_splits_{self.splits_version}.json"

                if splits_file.exists():
                    import json
                    with open(splits_file, 'r') as f:
                        splits_data = json.load(f)
                    splits = splits_data['splits']
                else:
                    raise FileNotFoundError(f"Official splits not found for version {self.splits_version}")

            # Validate splits
            self._validate_splits(splits)

            logger.info(f"Loaded official splits {self.splits_version}: "
                       f"train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

            return splits

        except Exception as e:
            logger.error(f"Failed to load official splits: {e}")
            raise

    def _validate_splits(self, splits: Dict[str, List[str]]) -> None:
        """Validate that splits meet challenge requirements."""
        required_splits = {'train', 'val', 'test'}

        # Check all required splits are present
        if not all(split in splits for split in required_splits):
            missing = required_splits - set(splits.keys())
            raise ValueError(f"Missing required splits: {missing}")

        # Check no subject appears in multiple splits
        all_subjects = []
        for split_subjects in splits.values():
            all_subjects.extend(split_subjects)

        if len(all_subjects) != len(set(all_subjects)):
            raise ValueError("Subject overlap detected between splits")

        # Log split sizes
        for split_name, split_subjects in splits.items():
            logger.debug(f"Split {split_name}: {len(split_subjects)} subjects")

    def _load_challenge_labels(self) -> None:
        """Load challenge-specific labels (CCD and CBCL)."""
        try:
            # Load CCD labels for Challenge 1 (response time + success)
            if any(task in ['CCD'] for task in self.tasks):
                self.labels['ccd'] = self.starter_kit.load_ccd_labels(split=self.split)
                logger.info(f"Loaded CCD labels: {len(self.labels['ccd'])} records")

            # Load CBCL labels for Challenge 2 (behavioral factors)
            self.labels['cbcl'] = self.starter_kit.load_cbcl_labels(split=self.split)
            logger.info(f"Loaded CBCL labels: {len(self.labels['cbcl'])} records")

        except Exception as e:
            logger.warning(f"Could not load some challenge labels: {e}")
            # Initialize empty labels to prevent errors
            if 'ccd' not in self.labels:
                self.labels['ccd'] = pd.DataFrame()
            if 'cbcl' not in self.labels:
                self.labels['cbcl'] = pd.DataFrame()

    def _build_file_index(self) -> List[Dict]:
        """Build index of all available EEG files for split participants."""
        file_index = []

        for participant in self.participants:
            for task in self.tasks:
                # Find all sessions for this participant/task
                bids_path = BIDSPath(
                    subject=participant,
                    task=task,
                    datatype="eeg",
                    root=self.bids_root
                )

                try:
                    # Get all matching files
                    matches = bids_path.match()
                    for match in matches:
                        # Validate that this participant is in the correct split
                        if participant not in self.participants:
                            logger.warning(f"Participant {participant} not in {self.split} split, skipping")
                            continue

                        file_info = {
                            "bids_path": match,
                            "participant": participant,
                            "task": task,
                            "session": match.session,
                            "split": self.split,  # Track split for validation
                        }
                        file_index.append(file_info)

                except Exception as e:
                    logger.warning(f"Could not find files for {participant}/{task}: {e}")

        logger.info(f"Found {len(file_index)} EEG files for {self.split} split")

        # Log split integrity check
        unique_participants = set(info['participant'] for info in file_index)
        if len(unique_participants) != len(self.participants):
            missing = set(self.participants) - unique_participants
            logger.warning(f"Some participants in {self.split} split have no EEG files: {missing}")

        return file_index

    def _extract_windows(self) -> List[Dict]:
        """Extract windows from all EEG files with session tracking."""
        windows = []

        for file_info in self.file_index:
            try:
                # Load raw data
                raw = read_raw_bids(file_info["bids_path"], verbose=False)

                # Apply preprocessing with leakage protection
                if self.preprocessor:
                    session_id = f"{file_info['participant']}_{file_info['session']}_{file_info['task']}"
                    raw = self.preprocessor.preprocess_raw(
                        raw=raw,
                        session_id=session_id,
                        split=self.split
                    )
                else:
                    # Basic preprocessing fallback
                    raw = self._basic_preprocessing(raw)

                # Extract windows
                file_windows = self._extract_windows_from_raw(raw, file_info)
                windows.extend(file_windows)

            except Exception as e:
                logger.warning(f"Could not process {file_info['bids_path']}: {e}")

        logger.info(f"Extracted {len(windows)} windows from {len(self.file_index)} files")
        return windows

    def _basic_preprocessing(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply basic preprocessing when no preprocessor is configured."""
        # Apply basic filters
        raw_copy = raw.copy()

        # Bandpass filter
        if "bandpass" in self.preprocessing_config:
            freqs = self.preprocessing_config["bandpass"]
            raw_copy.filter(l_freq=freqs[0], h_freq=freqs[1], verbose=False)

        # Notch filter
        if "notch" in self.preprocessing_config:
            notch_freq = self.preprocessing_config["notch"]
            raw_copy.notch_filter(freqs=notch_freq, verbose=False)

        # Re-referencing
        if self.preprocessing_config.get("reref") == "CAR":
            raw_copy.set_eeg_reference(ref_channels='average', verbose=False)

        return raw_copy

    def _extract_windows_from_raw(self, raw: mne.io.Raw, file_info: Dict) -> List[Dict]:
        """Extract sliding windows from a single raw file."""
        windows = []

        # Get sampling parameters
        sfreq = raw.info["sfreq"]
        n_samples = len(raw.times)

        # Calculate window parameters
        window_samples = int(self.window_length * sfreq)
        stride_samples = int((1 - self.overlap) * window_samples)

        # Extract windows
        start_idx = 0
        window_idx = 0

        while start_idx + window_samples <= n_samples:
            end_idx = start_idx + window_samples

            # Create window info
            window_info = {
                "file_info": file_info,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "window_idx": window_idx,
                "start_time": start_idx / sfreq,
                "end_time": end_idx / sfreq,
                "participant": file_info["participant"],
                "session": file_info["session"],
                "task": file_info["task"],
                "split": file_info["split"],  # Track split for validation
                "raw": raw,  # Store reference to preprocessed raw data
            }

            # Add labels if available
            window_info.update(self._get_window_labels(window_info))

            windows.append(window_info)

            start_idx += stride_samples
            window_idx += 1

        return windows

    def _get_window_labels(self, window_info: Dict) -> Dict:
        """Get challenge labels for a specific window."""
        labels = {}

        participant = window_info["participant"]
        task = window_info["task"]

        # CCD labels (Challenge 1)
        if task == "CCD" and "ccd" in self.labels and not self.labels["ccd"].empty:
            # Find CCD trials that overlap with this window
            ccd_data = self.labels["ccd"]
            participant_ccd = ccd_data[ccd_data["subject_id"] == participant]

            if not participant_ccd.empty:
                # For simplicity, use session-level aggregates
                # In practice, you might want more sophisticated temporal alignment
                labels["response_time_target"] = participant_ccd["response_time_target"].mean()
                labels["success_target"] = participant_ccd["success_target"].mean()

        # CBCL labels (Challenge 2) - subject-level
        if "cbcl" in self.labels and not self.labels["cbcl"].empty:
            cbcl_data = self.labels["cbcl"]
            participant_cbcl = cbcl_data[cbcl_data["participant_id"] == participant]

            if not participant_cbcl.empty:
                # Get CBCL factors
                for col in ["p_factor", "internalizing", "externalizing", "attention", "binary_label"]:
                    if col in participant_cbcl.columns:
                        labels[col] = participant_cbcl[col].iloc[0]  # Subject-level label

        return labels

    def get_session_info(self, idx: int) -> Dict[str, Any]:
        """Get session information for a given window index (for SessionAwareSampler)."""
        if idx >= len(self.windows):
            raise IndexError(f"Window index {idx} out of range")

        window = self.windows[idx]
        return {
            "subject_id": window["participant"],
            "session_id": window["session"],
            "task": window["task"],
            "split": window["split"]
        }

    def __len__(self) -> int:
        """Return the number of windows."""
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """Get a single window and its labels."""
        if idx >= len(self.windows):
            raise IndexError(f"Window index {idx} out of range")

        window_info = self.windows[idx]

        # Extract EEG data for this window
        raw = window_info["raw"]
        start_idx = window_info["start_idx"]
        end_idx = window_info["end_idx"]

        # Get data (channels x time)
        data = raw.get_data(start=start_idx, stop=end_idx)

        # Convert to tensor
        eeg_tensor = torch.from_numpy(data).float()

        # Apply transform if specified
        if self.transform:
            eeg_tensor = self.transform(eeg_tensor)

        # Prepare labels
        labels = {}
        for key in ["response_time_target", "success_target", "p_factor",
                   "internalizing", "externalizing", "attention", "binary_label"]:
            if key in window_info:
                value = window_info[key]
                if pd.notna(value):  # Only include non-NaN labels
                    labels[key] = torch.tensor(value).float()

        # Add metadata
        metadata = {
            "participant": window_info["participant"],
            "session": window_info["session"],
            "task": window_info["task"],
            "split": window_info["split"],
            "window_idx": window_info["window_idx"],
            "start_time": window_info["start_time"],
            "end_time": window_info["end_time"]
        }

        return eeg_tensor, {"labels": labels, "metadata": metadata}


class HBNDataLoader:
    """
    DataLoader factory for HBN dataset with leakage-free configuration.

    Features:
    - Session-aware sampling to prevent fold contamination
    - Automatic split validation
    - Preprocessor integration
    - Challenge-compliant data loading
    """

    def __init__(
        self,
        bids_root: Union[str, Path],
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        splits_version: str = "v1.0",
        preprocessing_config: Optional[Dict] = None
    ):
        self.bids_root = Path(bids_root)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.splits_version = splits_version
        self.preprocessing_config = preprocessing_config or {}

        logger.info(f"Initialized HBNDataLoader with batch_size={batch_size}")

    def get_dataloader(
        self,
        split: str,
        tasks: List[str],
        shuffle: bool = None,
        **dataset_kwargs
    ) -> DataLoader:
        """
        Get DataLoader for a specific split with session-aware sampling.

        Args:
            split: Data split ('train', 'val', 'test')
            tasks: List of tasks to include
            shuffle: Whether to shuffle (defaults based on split)
            **dataset_kwargs: Additional arguments for HBNDataset

        Returns:
            DataLoader with session-aware sampling
        """
        # Default shuffle behavior
        if shuffle is None:
            shuffle = (split == 'train')

        # Create dataset
        dataset = HBNDataset(
            bids_root=self.bids_root,
            split=split,
            tasks=tasks,
            preprocessing_config=self.preprocessing_config,
            splits_version=self.splits_version,
            **dataset_kwargs
        )

        # Create session-aware sampler
        sampler = SessionAwareSampler(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=shuffle
        )

        # Create DataLoader
        dataloader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self._collate_fn
        )

        logger.info(f"Created DataLoader for {split} split: {len(dataset)} windows, {len(dataloader)} batches")

        return dataloader

    def _collate_fn(self, batch):
        """Custom collate function to handle variable-length labels."""
        eeg_data = []
        all_labels = {}
        all_metadata = []

        for eeg, data_dict in batch:
            eeg_data.append(eeg)
            all_metadata.append(data_dict["metadata"])

            # Collect labels
            for key, value in data_dict["labels"].items():
                if key not in all_labels:
                    all_labels[key] = []
                all_labels[key].append(value)

        # Stack EEG data
        eeg_batch = torch.stack(eeg_data)

        # Stack labels (handling missing values)
        labels_batch = {}
        for key, values in all_labels.items():
            if values:  # Only if we have values
                try:
                    labels_batch[key] = torch.stack(values)
                except:
                    # Handle case where not all samples have this label
                    logger.debug(f"Could not stack labels for {key}")

        return eeg_batch, {"labels": labels_batch, "metadata": all_metadata}

    def validate_splits_integrity(self) -> Dict[str, Any]:
        """Validate that data loading maintains split integrity."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "stats": {}
        }

        try:
            # Load a small sample from each split
            test_tasks = ["RS"]  # Use a common task for testing

            split_datasets = {}
            for split in ["train", "val", "test"]:
                try:
                    dataset = HBNDataset(
                        bids_root=self.bids_root,
                        split=split,
                        tasks=test_tasks,
                        preprocessing_config=self.preprocessing_config,
                        splits_version=self.splits_version,
                        load_labels=False  # Faster for validation
                    )
                    split_datasets[split] = dataset
                except Exception as e:
                    results["errors"].append(f"Could not load {split} dataset: {e}")
                    results["valid"] = False

            # Check for subject overlaps
            split_subjects = {}
            for split, dataset in split_datasets.items():
                subjects = set(dataset.participants)
                split_subjects[split] = subjects
                results["stats"][f"{split}_subjects"] = len(subjects)

            # Check overlaps
            for i, (split1, subjects1) in enumerate(split_subjects.items()):
                for split2, subjects2 in list(split_subjects.items())[i+1:]:
                    overlap = subjects1 & subjects2
                    if overlap:
                        results["errors"].append(f"Subject overlap between {split1} and {split2}: {list(overlap)}")
                        results["valid"] = False

            # Check session isolation at the window level
            for split, dataset in split_datasets.items():
                session_ids = set()
                for window in dataset.windows[:100]:  # Sample first 100 windows
                    session_id = f"{window['participant']}_{window['session']}"
                    session_ids.add(session_id)

                results["stats"][f"{split}_sessions_sample"] = len(session_ids)

            if results["valid"]:
                logger.info("✅ Split integrity validation passed")
            else:
                logger.error(f"❌ Split integrity validation failed: {results['errors']}")

        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Validation error: {e}")
            logger.error(f"Split validation failed: {e}")

        return results

    def _preprocess_raw(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Apply preprocessing steps to raw EEG data."""
        # Make a copy to avoid modifying original
        raw = raw.copy()

        # Basic filtering
        l_freq = self.preprocessing_config.get("l_freq", 0.1)
        h_freq = self.preprocessing_config.get("h_freq", 40.0)
        raw.filter(l_freq, h_freq, verbose=False)

        # Notch filter
        notch_freq = self.preprocessing_config.get("notch_freq", 60.0)
        if notch_freq:
            raw.notch_filter(notch_freq, verbose=False)

        # Re-referencing
        reference = self.preprocessing_config.get("reference", "average")
        if reference == "average":
            raw.set_eeg_reference("average", projection=True, verbose=False)

        # Bad channel detection (simplified)
        if self.preprocessing_config.get("bad_channel_detection", False):
            # This is a placeholder - implement proper bad channel detection
            pass

        return raw

    def _extract_windows_from_raw(self, raw: mne.io.Raw, file_info: Dict) -> List[Dict]:
        """Extract windowed segments from raw EEG data."""
        windows = []

        # Calculate window parameters
        sfreq = raw.info["sfreq"]
        window_samples = int(self.window_length * sfreq)
        step_samples = int(window_samples * (1 - self.overlap))

        # Get data
        data = raw.get_data()  # Shape: (n_channels, n_samples)
        n_samples = data.shape[1]

        # Extract windows
        start = 0
        while start + window_samples <= n_samples:
            window_data = data[:, start:start + window_samples]

            window_info = {
                "data": window_data,
                "participant": file_info["participant"],
                "task": file_info["task"],
                "session": file_info["session"],
                "start_time": start / sfreq,
                "sfreq": sfreq,
                "ch_names": raw.ch_names,
            }
            windows.append(window_info)

            start += step_samples

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        window = self.windows[idx]

        # Convert to tensor
        data = torch.from_numpy(window["data"]).float()

        # Apply transforms if provided
        if self.transform:
            data = self.transform(data)

        return {
            "eeg": data,
            "participant": window["participant"],
            "task": window["task"],
            "session": window["session"],
            "start_time": window["start_time"],
            "sfreq": window["sfreq"],
        }


class HBNDataLoader:
    """
    Data loader factory for HBN EEG dataset.

    This class provides convenient methods to create train/val/test
    data loaders with proper subject-level splitting.
    """

    def __init__(self, bids_root: Union[str, Path]):
        self.bids_root = Path(bids_root)
        self.participants_df = self._load_participants()

    def _load_participants(self) -> pd.DataFrame:
        """Load participants.tsv file."""
        participants_file = self.bids_root / "participants.tsv"
        if participants_file.exists():
            return pd.read_csv(participants_file, sep="\t")
        else:
            logger.warning("participants.tsv not found, creating empty DataFrame")
            return pd.DataFrame()

    def get_dataset(
        self,
        participants: Optional[List[str]] = None,
        tasks: List[str] = ["SuS", "CCD"],
        split: str = "train",
        **kwargs
    ) -> HBNDataset:
        """
        Create a dataset for specified participants and tasks.

        Args:
            participants: List of participant IDs (None for all)
            tasks: List of task names
            split: Split name (for logging purposes)
            **kwargs: Additional arguments passed to HBNDataset

        Returns:
            HBNDataset instance
        """
        if participants is None:
            participants = self._get_available_participants(tasks)

        logger.info(f"Creating {split} dataset with {len(participants)} participants")

        return HBNDataset(
            bids_root=self.bids_root,
            participants=participants,
            tasks=tasks,
            **kwargs
        )

    def get_dataloader(
        self,
        dataset: HBNDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        **kwargs
    ) -> DataLoader:
        """Create PyTorch DataLoader."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            **kwargs
        )

    def _get_available_participants(self, tasks: List[str]) -> List[str]:
        """Get list of participants with data for specified tasks."""
        # This is a placeholder - implement proper participant discovery
        # based on available BIDS files
        participants = []

        # Scan BIDS directory for available participants
        if self.bids_root.exists():
            for sub_dir in self.bids_root.glob("sub-*"):
                participant_id = sub_dir.name.replace("sub-", "")

                # Check if participant has data for required tasks
                has_all_tasks = True
                for task in tasks:
                    task_files = list(sub_dir.glob(f"**/sub-{participant_id}_task-{task}_eeg.*"))
                    if not task_files:
                        has_all_tasks = False
                        break

                if has_all_tasks:
                    participants.append(participant_id)

        return participants

    def create_splits(
        self,
        tasks: List[str],
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Create train/val/test splits at participant level.

        Returns:
            Tuple of (train_participants, val_participants, test_participants)
        """
        participants = self._get_available_participants(tasks)

        # Shuffle participants
        np.random.seed(random_state)
        shuffled_participants = np.random.permutation(participants)

        # Calculate split sizes
        n_participants = len(shuffled_participants)
        n_train = int(n_participants * train_ratio)
        n_val = int(n_participants * val_ratio)

        # Create splits
        train_participants = shuffled_participants[:n_train].tolist()
        val_participants = shuffled_participants[n_train:n_train + n_val].tolist()
        test_participants = shuffled_participants[n_train + n_val:].tolist()

        logger.info(f"Split {n_participants} participants: "
                   f"train={len(train_participants)}, "
                   f"val={len(val_participants)}, "
                   f"test={len(test_participants)}")

        return train_participants, val_participants, test_participants
