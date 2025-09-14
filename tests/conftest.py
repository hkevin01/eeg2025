"""Test configuration for pytest."""

from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def device():
    """Device fixture for tests."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_eeg_data():
    """Sample EEG data for testing."""
    batch_size = 4
    n_channels = 64
    seq_length = 500  # 2 seconds at 250 Hz

    return torch.randn(batch_size, n_channels, seq_length)


@pytest.fixture
def sample_metadata():
    """Sample metadata for testing."""
    return {
        "participant": ["sub-001", "sub-002", "sub-003", "sub-004"],
        "task": ["SuS", "SuS", "CCD", "CCD"],
        "session": ["ses-001", "ses-001", "ses-001", "ses-001"],
        "sfreq": 250.0,
    }


@pytest.fixture
def temp_bids_dir(tmp_path):
    """Create temporary BIDS directory structure for testing."""
    bids_root = tmp_path / "bids"
    bids_root.mkdir()

    # Create basic BIDS files
    (bids_root / "participants.tsv").write_text(
        "participant_id\tage\tsex\n" "sub-001\t25\tM\n" "sub-002\t30\tF\n"
    )

    (bids_root / "dataset_description.json").write_text(
        '{"Name": "Test Dataset", "BIDSVersion": "1.8.0"}'
    )

    # Create subject directories
    for sub_id in ["sub-001", "sub-002"]:
        sub_dir = bids_root / sub_id / "ses-001" / "eeg"
        sub_dir.mkdir(parents=True)

        # Create dummy EEG files
        for task in ["SuS", "CCD"]:
            eeg_file = sub_dir / f"{sub_id}_ses-001_task-{task}_eeg.edf"
            eeg_file.touch()

    return bids_root


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    return {
        "model": {
            "backbone": {
                "name": "temporal_cnn",
                "n_channels": 64,
                "hidden_dims": [32, 64],
                "kernel_sizes": [5, 5],
                "dilations": [1, 2],
                "dropout": 0.1,
            }
        },
        "data": {
            "batch_size": 4,
            "window_length": 2.0,
            "overlap": 0.5,
            "preprocessing": {
                "l_freq": 0.1,
                "h_freq": 40.0,
                "notch_freq": 60.0,
                "reference": "average",
            },
        },
        "training": {
            "max_epochs": 2,
            "optimizer": {
                "name": "adamw",
                "lr": 1e-3,
                "weight_decay": 0.01,
            },
        },
    }
