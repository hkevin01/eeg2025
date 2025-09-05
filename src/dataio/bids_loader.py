"""
BIDS-compliant data loader for the Healthy Brain Network EEG dataset.

This module provides functionality to load and preprocess EEG data
following the Brain Imaging Data Structure (BIDS) standard.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from mne_bids import BIDSPath, read_raw_bids

logger = logging.getLogger(__name__)


class HBNDataset(Dataset):
    """
    PyTorch Dataset for HBN EEG data.

    Args:
        bids_root: Path to BIDS root directory
        participants: List of participant IDs to include
        tasks: List of task names to include
        window_length: Length of each window in seconds
        overlap: Overlap between windows (0-1)
        preprocessing_config: Configuration for preprocessing steps
    """

    def __init__(
        self,
        bids_root: Union[str, Path],
        participants: List[str],
        tasks: List[str],
        window_length: float = 2.0,
        overlap: float = 0.5,
        preprocessing_config: Optional[Dict] = None,
        transform=None,
    ):
        self.bids_root = Path(bids_root)
        self.participants = participants
        self.tasks = tasks
        self.window_length = window_length
        self.overlap = overlap
        self.preprocessing_config = preprocessing_config or {}
        self.transform = transform

        # Build index of all available files
        self.file_index = self._build_file_index()

        # Extract windows from all files
        self.windows = self._extract_windows()

    def _build_file_index(self) -> List[Dict]:
        """Build index of all available EEG files."""
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
                        file_info = {
                            "bids_path": match,
                            "participant": participant,
                            "task": task,
                            "session": match.session,
                        }
                        file_index.append(file_info)

                except Exception as e:
                    logger.warning(f"Could not find files for {participant}/{task}: {e}")

        logger.info(f"Found {len(file_index)} EEG files")
        return file_index

    def _extract_windows(self) -> List[Dict]:
        """Extract windows from all EEG files."""
        windows = []

        for file_info in self.file_index:
            try:
                # Load raw data
                raw = read_raw_bids(file_info["bids_path"], verbose=False)

                # Apply preprocessing
                raw = self._preprocess_raw(raw)

                # Extract windows
                file_windows = self._extract_windows_from_raw(raw, file_info)
                windows.extend(file_windows)

            except Exception as e:
                logger.warning(f"Could not process {file_info['bids_path']}: {e}")
                continue

        logger.info(f"Extracted {len(windows)} windows")
        return windows

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
