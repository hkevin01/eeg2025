"""
HBN Dataset implementation for EEG Foundation Challenge 2025.

This module provides BIDS-compliant data loading for the Healthy Brain Network
EEG dataset with official Starter Kit label extraction and proper windowing.
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
from .starter_kit import StarterKitDataLoader, OfficialMetrics

logger = logging.getLogger(__name__)


class HBNDataset(Dataset):
    """
    HBN EEG Dataset for the Foundation Challenge with official Starter Kit integration.

    Loads EEG data from BIDS format and provides windowed samples with
    official challenge labels for different tasks (pretraining, cross-task, psychopathology).
    """

    def __init__(
        self,
        bids_root: Union[str, Path],
        split: str = "train",
        window_length: float = 2.0,  # Official challenge window length
        overlap: float = 0.5,
        sample_rate: int = 500,
        task_type: str = "cross_task",
        subjects: Optional[List[str]] = None,
        sessions: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        preprocessing_params: Optional[Dict] = None,
        use_official_splits: bool = True
    ):
        """
        Initialize HBN dataset with Starter Kit integration.

        Args:
            bids_root: Path to BIDS dataset root
            split: Data split ('train', 'val', 'test')
            window_length: Length of each window in seconds (2.0s for challenge)
            overlap: Overlap between windows (0-1)
            sample_rate: Target sample rate in Hz
            task_type: Type of task ('pretraining', 'cross_task', 'psychopathology')
            subjects: List of subject IDs to include
            sessions: List of session IDs to include
            tasks: List of task names to include
            preprocessing_params: Preprocessing parameters
            use_official_splits: Whether to use official challenge splits
        """
        self.bids_root = Path(bids_root)
        self.split = split
        self.window_length = window_length
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.task_type = task_type
        self.use_official_splits = use_official_splits

        # Initialize Starter Kit data loader
        self.starter_kit = StarterKitDataLoader(bids_root)

        # Preprocessing parameters
        self.preprocessing_params = preprocessing_params or {
            "l_freq": 0.1,  # Official challenge preprocessing
            "h_freq": 40.0,
            "notch_freq": 60.0,
            "reject_criteria": {"eeg": 150e-6}  # 150 µV
        }

        # Get official splits if requested
        if use_official_splits and subjects is None:
            subjects = self.starter_kit.official_splits.get(split, [])
            # Remove 'sub-' prefix if present
            subjects = [s.replace('sub-', '') for s in subjects]

        # Filter tasks based on challenge requirements
        if tasks is None:
            if task_type == "pretraining":
                tasks = ["SuS", "RS", "MW"]  # Passive tasks for SSL
            elif task_type == "cross_task":
                tasks = ["CCD", "SuS"]  # CCD for testing, SuS for pretraining
            elif task_type == "psychopathology":
                tasks = ["RS", "SuS", "MW", "CCD"]  # All available tasks

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

        # Load official labels
        self._load_official_labels()

        # Filter data based on task type and label availability
        self._filter_by_task_and_labels()

        logger.info(f"Loaded {len(self)} samples for {task_type} task, split: {split}")
        logger.info(f"Using official splits: {use_official_splits}")

    def _load_official_labels(self):
        """Load official challenge labels using Starter Kit."""
        self.file_labels = []

        for window_idx in range(len(self.bids_dataset)):
            window_info = self.bids_dataset.get_window_info(window_idx)

            # Extract metadata from filename
            bids_path = BIDSPath.from_path(window_info["file_path"])
            subject_id = bids_path.subject
            session_id = bids_path.session or "001"
            task_name = bids_path.task

            # Initialize label dictionary
            labels = {
                "subject_id": subject_id,
                "session_id": session_id,
                "task_name": task_name,
                "file_path": str(window_info["file_path"]),
                "window_start": window_info["window_start"],
                "window_end": window_info["window_end"]
            }

            # Add subject metadata for domain adaptation
            subject_metadata = self.starter_kit.get_subject_metadata(subject_id)
            labels.update(subject_metadata)

            # Add task-specific official labels
            if self.task_type in ["cross_task", "pretraining"] and task_name == "CCD":
                ccd_labels = self.starter_kit.get_ccd_labels(subject_id, session_id)
                labels.update(ccd_labels)

            if self.task_type in ["psychopathology", "cross_task"]:
                cbcl_labels = self.starter_kit.get_cbcl_labels(subject_id)
                labels.update(cbcl_labels)

            # Add SSL pretext labels for pretraining
            if self.task_type == "pretraining":
                labels.update(self._get_ssl_labels(task_name, window_info))

            self.file_labels.append(labels)

    def _get_ssl_labels(self, task_name: str, window_info: Dict) -> Dict[str, int]:
        """Get self-supervised learning labels."""
        # Task classification for contrastive learning
        task_mapping = {
            "RS": 0,   # Resting state
            "SuS": 1,  # Sustained attention to response
            "MW": 2,   # Mind wandering
            "CCD": 3,  # Continuous cognitive demand
            "SL": 4,   # Spatial learning
            "SyS": 5   # Symbol search
        }

        task_label = task_mapping.get(task_name, -1)

        # Temporal prediction labels (next window prediction)
        window_idx = int(window_info.get("window_start", 0) / self.window_length)

        return {
            "task_label": task_label,
            "temporal_position": window_idx,
            "is_passive": 1 if task_name in ["RS", "SuS", "MW"] else 0,
            "is_active": 1 if task_name in ["CCD", "SL", "SyS"] else 0
        }

    def _filter_by_task_and_labels(self):
        """Filter data based on task type and label availability."""
        filtered_labels = []

        for labels in self.file_labels:
            include_sample = True

            if self.task_type == "pretraining":
                # Include all samples with valid task labels
                if labels.get("task_label", -1) < 0:
                    include_sample = False

            elif self.task_type == "cross_task":
                # Include CCD samples with valid RT/success labels
                if labels["task_name"] == "CCD":
                    if np.isnan(labels.get("mean_rt", np.nan)) and labels.get("trial_count", 0) == 0:
                        include_sample = False
                # Include other tasks for pretraining
                elif labels.get("task_label", -1) < 0:
                    include_sample = False

            elif self.task_type == "psychopathology":
                # Include samples with valid CBCL labels
                has_cbcl = any(
                    labels.get(key, 0.0) != 0.0
                    for key in ["p_factor", "internalizing", "externalizing", "attention"]
                )
                if not has_cbcl and "binary_label" not in labels:
                    include_sample = False

            if include_sample:
                filtered_labels.append(labels)

        self.file_labels = filtered_labels
        logger.info(f"Filtered to {len(self.file_labels)} samples with valid labels")

    def __len__(self) -> int:
        """Return number of windows in dataset."""
        return len(self.file_labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing EEG data and official labels
        """
        # Get window data
        window_data = self.bids_dataset[idx]
        labels = self.file_labels[idx]

        # Convert to tensors
        sample = {
            "eeg": torch.FloatTensor(window_data["eeg"]),
            "channels": window_data["channels"]
        }

        # Add metadata
        sample["subject_id"] = labels["subject_id"]
        sample["task_name"] = labels["task_name"]
        sample["session_id"] = labels["session_id"]

        # Add task-specific labels as tensors
        if self.task_type in ["cross_task", "pretraining"]:
            # Task classification
            if "task_label" in labels:
                sample["task_label"] = torch.LongTensor([labels["task_label"]])

            # Domain adaptation (subject ID)
            subject_hash = hash(labels["subject_id"]) % 10000
            sample["domain_label"] = torch.LongTensor([subject_hash])

            # CCD-specific labels
            if labels["task_name"] == "CCD":
                if not np.isnan(labels.get("mean_rt", np.nan)):
                    sample["response_time"] = torch.FloatTensor([labels["mean_rt"]])
                    sample["response_time_target"] = torch.FloatTensor([labels["mean_rt"]])

                if "success_rate" in labels:
                    sample["success"] = torch.FloatTensor([labels["success_rate"]])
                    sample["success_target"] = torch.FloatTensor([labels["success_rate"]])

        if self.task_type in ["psychopathology", "cross_task"]:
            # CBCL dimensions
            cbcl_dims = ["p_factor", "internalizing", "externalizing", "attention"]
            for dim in cbcl_dims:
                if dim in labels:
                    sample[f"{dim}_target"] = torch.FloatTensor([labels[dim]])

            # Binary classification
            if "binary_label" in labels:
                sample["binary_target"] = torch.LongTensor([labels["binary_label"]])

        # Add subject metadata for domain adaptation
        if "age" in labels and not np.isnan(labels["age"]):
            sample["age"] = torch.FloatTensor([labels["age"]])

        if "site" in labels:
            site_hash = hash(str(labels["site"])) % 100
            sample["site_label"] = torch.LongTensor([site_hash])

        return sample

    def get_official_metrics(self, predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Compute official challenge metrics.

        Args:
            predictions: Dictionary with model predictions

        Returns:
            Dictionary with official metrics
        """
        # Extract ground truth labels
        targets = {}

        if self.task_type == "cross_task":
            # Challenge 1 targets
            rt_targets = []
            success_targets = []

            for labels in self.file_labels:
                if labels["task_name"] == "CCD":
                    rt_targets.append(labels.get("mean_rt", np.nan))
                    success_targets.append(labels.get("success_rate", 0.0))

            if rt_targets:
                targets["response_time"] = np.array(rt_targets)
                targets["success"] = np.array(success_targets)

            return OfficialMetrics.compute_challenge1_metrics(predictions, targets)

        elif self.task_type == "psychopathology":
            # Challenge 2 targets
            cbcl_dims = ["p_factor", "internalizing", "externalizing", "attention"]

            for dim in cbcl_dims:
                dim_targets = [labels.get(dim, 0.0) for labels in self.file_labels]
                targets[dim] = np.array(dim_targets)

            binary_targets = [labels.get("binary_label", 0) for labels in self.file_labels]
            targets["binary_label"] = np.array(binary_targets)

            return OfficialMetrics.compute_challenge2_metrics(predictions, targets)

        return {}

    def create_official_submission(
        self,
        predictions: Dict[str, np.ndarray],
        output_path: Union[str, Path],
        aggregate_subjects: bool = True
    ) -> pd.DataFrame:
        """
        Create official submission format.

        Args:
            predictions: Model predictions
            output_path: Path to save submission CSV
            aggregate_subjects: Whether to aggregate to subject level

        Returns:
            Submission DataFrame
        """
        from .starter_kit import SubmissionValidator

        # Create submission data
        submission_data = []

        for idx, labels in enumerate(self.file_labels):
            row = {
                "subject_id": f"sub-{labels['subject_id']}",
                "session_id": labels["session_id"]
            }

            # Add predictions based on task type
            if self.task_type == "cross_task" and labels["task_name"] == "CCD":
                if "response_time" in predictions and idx < len(predictions["response_time"]):
                    row["response_time_pred"] = predictions["response_time"][idx]
                if "success" in predictions and idx < len(predictions["success"]):
                    row["success_pred"] = predictions["success"][idx]

            elif self.task_type == "psychopathology":
                cbcl_dims = ["p_factor", "internalizing", "externalizing", "attention"]
                for dim in cbcl_dims:
                    if dim in predictions and idx < len(predictions[dim]):
                        row[f"{dim}_pred"] = predictions[dim][idx]

                if "binary_label" in predictions and idx < len(predictions["binary_label"]):
                    row["binary_pred"] = predictions["binary_label"][idx]

            submission_data.append(row)

        submission_df = pd.DataFrame(submission_data)

        # Aggregate to subject level if requested
        if aggregate_subjects and len(submission_df) > 0:
            numeric_cols = [col for col in submission_df.columns
                          if col.endswith('_pred') and submission_df[col].dtype in ['float64', 'int64']]

            agg_dict = {col: 'mean' for col in numeric_cols}
            agg_dict['session_id'] = 'first'  # Keep first session

            submission_df = submission_df.groupby('subject_id').agg(agg_dict).reset_index()

        # Validate submission format
        validator = SubmissionValidator()

        if self.task_type == "cross_task":
            validation_result = validator.validate_challenge1_submission(submission_df)
        elif self.task_type == "psychopathology":
            validation_result = validator.validate_challenge2_submission(submission_df)
        else:
            validation_result = {"valid": True, "errors": [], "warnings": []}

        if not validation_result["valid"]:
            logger.error(f"Submission validation failed: {validation_result['errors']}")

        if validation_result["warnings"]:
            logger.warning(f"Submission warnings: {validation_result['warnings']}")

        # Save submission
        submission_df.to_csv(output_path, index=False)
        logger.info(f"Saved official submission to {output_path}")

        return submission_df


def create_hbn_datasets(
    bids_root: Union[str, Path],
    window_length: float = 2.0,
    overlap: float = 0.5,
    sample_rate: int = 500,
    task_type: str = "cross_task",
    subject_splits: Optional[Dict[str, List[str]]] = None,
    use_official_splits: bool = True
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
        use_official_splits: Whether to use official challenge splits

    Returns:
        Dictionary containing train, val, and test datasets
    """
    if not use_official_splits and subject_splits is None:
        # Create a temporary dataset to get subject splits
        temp_dataset = HBNDataset(
            bids_root=bids_root,
            split="all",
            window_length=window_length,
            overlap=overlap,
            sample_rate=sample_rate,
            task_type=task_type,
            use_official_splits=False
        )
        subject_splits = temp_dataset.starter_kit.official_splits

    # Create datasets for each split
    datasets = {}
    splits = ["train", "val", "test"]

    for split_name in splits:
        datasets[split_name] = HBNDataset(
            bids_root=bids_root,
            split=split_name,
            window_length=window_length,
            overlap=overlap,
            sample_rate=sample_rate,
            task_type=task_type,
            use_official_splits=use_official_splits
        )

    return datasets

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
