"""
HBN Dataset implementation for EEG Foundation Challenge 2025.

This module provides BIDS-compliant data loading for the Healthy Brain Network
EEG dataset with proper label extraction and windowing.
"""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset
import mne
from mne_bids import BIDSPath, read_raw_bids

from .bids_loader import BidsWindowDataset

logger = logging.getLogger(__name__)


class HBNDataset(Dataset):
    """
    HBN EEG Dataset for the Foundation Challenge.

    Loads EEG data from BIDS format and provides windowed samples with
    appropriate labels for different tasks (pretraining, cross-task, psychopathology).
    """

    def __init__(
        self,
        bids_root: Union[str, Path],
        split: str = "train",
        window_length: float = 4.0,
        overlap: float = 0.5,
        sample_rate: int = 500,
        task_type: str = "cross_task",
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        preprocessing_params: Optional[Dict] = None
    ):
        """
        Initialize HBN dataset.

        Args:
            bids_root: Path to BIDS dataset root
            split: Data split ('train', 'val', 'test')
            window_length: Length of each window in seconds
            overlap: Overlap between windows (0-1)
            sample_rate: Target sample rate in Hz
            task_type: Type of task ('pretraining', 'cross_task', 'psychopathology')
            subjects: List of subject IDs to include
            sessions: List of session IDs to include
            tasks: List of task names to include
            preprocessing_params: Preprocessing parameters
        """
        self.bids_root = Path(bids_root)
        self.split = split
        self.window_length = window_length
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.task_type = task_type

        # Preprocessing parameters
        self.preprocessing_params = preprocessing_params or {
            "l_freq": 1.0,
            "h_freq": 40.0,
            "notch_freq": 60.0,
            "reject_criteria": {"eeg": 150e-6}  # 150 µV
        }

        # Initialize BIDS dataset
        self.bids_dataset = BidsWindowDataset(
            bids_root=self.bids_root,
            window_length=window_length,
            overlap=overlap,
            target_sfreq=sample_rate,
            subjects=subjects,
            sessions=sessions,
            tasks=tasks,
            splits=[split] if split != "all" else None
        )

        # Load metadata and labels
        self._load_metadata()
        self._create_labels()

        # Filter data based on task type
        self._filter_by_task_type()

        logger.info(f"Loaded {len(self)} samples for {task_type} task, split: {split}")

    def _load_metadata(self):
        """Load participant metadata and behavioral data."""
        # Load participants.tsv
        participants_file = self.bids_root / "participants.tsv"
        if participants_file.exists():
            self.participants_df = pd.read_csv(participants_file, sep="\t")
        else:
            logger.warning(f"participants.tsv not found at {participants_file}")
            self.participants_df = pd.DataFrame()

        # Load phenotypic data (for psychopathology labels)
        phenotype_dir = self.bids_root / "phenotype"
        self.phenotype_data = {}

        if phenotype_dir.exists():
            for phenotype_file in phenotype_dir.glob("*.tsv"):
                df = pd.read_csv(phenotype_file, sep="\t")
                self.phenotype_data[phenotype_file.stem] = df

        logger.info(f"Loaded metadata for {len(self.participants_df)} participants")
        logger.info(f"Available phenotype files: {list(self.phenotype_data.keys())}")

    def _create_labels(self):
        """Create labels for different tasks."""
        # Get all available files from BIDS dataset
        self.file_labels = []

        for window_idx in range(len(self.bids_dataset)):
            window_info = self.bids_dataset.get_window_info(window_idx)

            # Extract metadata from filename
            bids_path = BIDSPath.from_path(window_info["file_path"])
            subject_id = bids_path.subject
            session_id = bids_path.session
            task_name = bids_path.task

            # Create label dictionary
            labels = {
                "subject_id": subject_id,
                "session_id": session_id,
                "task_name": task_name,
                "file_path": str(window_info["file_path"]),
                "window_start": window_info["window_start"],
                "window_end": window_info["window_end"]
            }

            # Add task-specific labels
            if self.task_type in ["cross_task", "pretraining"]:
                labels.update(self._get_task_labels(task_name))

            if self.task_type in ["psychopathology", "cross_task"]:
                labels.update(self._get_psychopathology_labels(subject_id))

            self.file_labels.append(labels)

    def _get_task_labels(self, task_name: str) -> Dict[str, int]:
        """Get task classification labels."""
        # Map HBN task names to integer labels
        task_mapping = {
            "rest": 0,
            "movie": 1,
            "DM": 2,  # Dot Motion
            "Nback": 3,
            "GoNogo": 4,
            "flanker": 5,
            "MSIT": 6,  # Multi-Source Interference Task
            "cuedts": 7,  # Cued Task Switching
            "srt": 8,  # Simple Reaction Time
            "PEER1": 9,  # PEER Task 1
            "PEER2": 10  # PEER Task 2
        }

        task_label = task_mapping.get(task_name, -1)  # -1 for unknown tasks

        return {
            "task_label": task_label,
            "is_rest": 1 if task_name == "rest" else 0,
            "is_cognitive": 1 if task_name in ["DM", "Nback", "GoNogo", "flanker", "MSIT", "cuedts"] else 0
        }

    def _get_psychopathology_labels(self, subject_id: str) -> Dict[str, int]:
        """Get psychopathology prediction labels."""
        # Initialize with default values
        labels = {
            "psych_label": 0,  # 0 = typical, 1 = atypical
            "adhd_score": 0.0,
            "anxiety_score": 0.0,
            "depression_score": 0.0
        }

        # Look for subject in phenotype data
        subject_key = f"sub-{subject_id}"

        # Check CBCL (Child Behavior Checklist) data
        if "CBCL" in self.phenotype_data:
            cbcl_df = self.phenotype_data["CBCL"]
            subject_data = cbcl_df[cbcl_df["participant_id"] == subject_key]

            if not subject_data.empty:
                # Extract relevant scores (these column names are examples)
                if "CBCL_Attention_T" in subject_data.columns:
                    adhd_score = subject_data["CBCL_Attention_T"].iloc[0]
                    labels["adhd_score"] = float(adhd_score) if pd.notna(adhd_score) else 0.0

                if "CBCL_Anxious_T" in subject_data.columns:
                    anxiety_score = subject_data["CBCL_Anxious_T"].iloc[0]
                    labels["anxiety_score"] = float(anxiety_score) if pd.notna(anxiety_score) else 0.0

                if "CBCL_Withdrawn_T" in subject_data.columns:
                    depression_score = subject_data["CBCL_Withdrawn_T"].iloc[0]
                    labels["depression_score"] = float(depression_score) if pd.notna(depression_score) else 0.0

                # Create binary psychopathology label based on clinical cutoffs
                # T-scores >= 65 are considered clinically significant
                is_atypical = (
                    labels["adhd_score"] >= 65 or
                    labels["anxiety_score"] >= 65 or
                    labels["depression_score"] >= 65
                )
                labels["psych_label"] = 1 if is_atypical else 0

        return labels

    def _filter_by_task_type(self):
        """Filter data based on task type requirements."""
        if self.task_type == "pretraining":
            # Use all data for pretraining
            pass
        elif self.task_type == "cross_task":
            # Keep samples with valid task labels
            self.file_labels = [
                labels for labels in self.file_labels
                if labels.get("task_label", -1) >= 0
            ]
        elif self.task_type == "psychopathology":
            # Keep samples with valid psychopathology labels
            self.file_labels = [
                labels for labels in self.file_labels
                if "psych_label" in labels
            ]

    def __len__(self) -> int:
        """Return number of windows in dataset."""
        return len(self.file_labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing EEG data and labels
        """
        # Get window data
        window_data = self.bids_dataset[idx]
        labels = self.file_labels[idx]

        # Convert to tensors
        sample = {
            "eeg": torch.FloatTensor(window_data["eeg"]),
            "channels": window_data["channels"]
        }

        # Add labels as tensors
        for key, value in labels.items():
            if isinstance(value, (int, float)):
                if key.endswith("_id"):
                    # Convert subject/session IDs to integers for classification
                    try:
                        sample[key] = torch.LongTensor([hash(str(value)) % 10000])
                    except:
                        sample[key] = torch.LongTensor([0])
                elif key.endswith("_label"):
                    sample[key] = torch.LongTensor([int(value)])
                elif key.endswith("_score"):
                    sample[key] = torch.FloatTensor([float(value)])
                else:
                    sample[key] = torch.LongTensor([int(value)])

        return sample

    def get_class_weights(self, label_type: str = "psych_label") -> torch.Tensor:
        """
        Compute class weights for balanced training.

        Args:
            label_type: Type of labels to compute weights for

        Returns:
            Class weights tensor
        """
        if label_type not in [labels.get(label_type) for labels in self.file_labels]:
            logger.warning(f"Label type {label_type} not found in dataset")
            return torch.ones(2)

        # Count classes
        labels = [labels_dict.get(label_type, 0) for labels_dict in self.file_labels]
        unique_labels, counts = np.unique(labels, return_counts=True)

        # Compute weights (inverse frequency)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(weights)

        return torch.FloatTensor(weights)

    def get_subject_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Dict[str, List[str]]:
        """
        Get subject-wise data splits to prevent data leakage.

        Args:
            train_ratio: Ratio of subjects for training
            val_ratio: Ratio of subjects for validation

        Returns:
            Dictionary with subject IDs for each split
        """
        # Get unique subjects
        subjects = list(set([labels["subject_id"] for labels in self.file_labels]))
        subjects.sort()  # For reproducibility

        # Create splits
        n_subjects = len(subjects)
        n_train = int(n_subjects * train_ratio)
        n_val = int(n_subjects * val_ratio)

        splits = {
            "train": subjects[:n_train],
            "val": subjects[n_train:n_train + n_val],
            "test": subjects[n_train + n_val:]
        }

        logger.info(f"Subject splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

        return splits

    def save_predictions(self, predictions: Dict[str, np.ndarray], output_path: Union[str, Path]):
        """
        Save predictions in competition format.

        Args:
            predictions: Dictionary with prediction arrays
            output_path: Path to save predictions CSV
        """
        # Create submission DataFrame
        submission_data = []

        for idx, labels in enumerate(self.file_labels):
            row = {
                "subject_id": labels["subject_id"],
                "session_id": labels.get("session_id", "001"),
                "task_name": labels.get("task_name", "unknown"),
                "window_start": labels["window_start"],
                "window_end": labels["window_end"]
            }

            # Add predictions
            for pred_type, pred_array in predictions.items():
                if idx < len(pred_array):
                    row[f"pred_{pred_type}"] = pred_array[idx]

            submission_data.append(row)

        # Save to CSV
        submission_df = pd.DataFrame(submission_data)
        submission_df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")


def create_hbn_datasets(
    bids_root: Union[str, Path],
    window_length: float = 4.0,
    overlap: float = 0.5,
    sample_rate: int = 500,
    task_type: str = "cross_task",
    subject_splits: Optional[Dict[str, List[str]]] = None
) -> Dict[str, HBNDataset]:
    """
    Create train/val/test datasets with proper subject-wise splitting.

    Args:
        bids_root: Path to BIDS dataset root
        window_length: Length of each window in seconds
        overlap: Overlap between windows (0-1)
        sample_rate: Target sample rate in Hz
        task_type: Type of task
        subject_splits: Pre-defined subject splits

    Returns:
        Dictionary containing train, val, and test datasets
    """
    if subject_splits is None:
        # Create a temporary dataset to get subject splits
        temp_dataset = HBNDataset(
            bids_root=bids_root,
            split="all",
            window_length=window_length,
            overlap=overlap,
            sample_rate=sample_rate,
            task_type=task_type
        )
        subject_splits = temp_dataset.get_subject_splits()

    # Create datasets for each split
    datasets = {}
    for split_name, subjects in subject_splits.items():
        datasets[split_name] = HBNDataset(
            bids_root=bids_root,
            split=split_name,
            window_length=window_length,
            overlap=overlap,
            sample_rate=sample_rate,
            task_type=task_type,
            subjects=subjects
        )

    return datasets
