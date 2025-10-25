#!/usr/bin/env python3
"""Test data loading functionality"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.models.eeg_dataset_simple import SimpleEEGDataset


def test_dataset_loads():
    """Test that dataset can load"""
    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
    if not data_dir.exists():
        pytest.skip("Data directory not found")

    dataset = SimpleEEGDataset(data_dir=data_dir, max_subjects=2)
    assert len(dataset) > 0, "Dataset should have samples"

def test_dataset_shape():
    """Test that samples have correct shape"""
    data_dir = Path(__file__).parent.parent / "data" / "raw" / "hbn"
    if not data_dir.exists():
        pytest.skip("Data directory not found")

    dataset = SimpleEEGDataset(data_dir=data_dir, max_subjects=1)
    if len(dataset) > 0:
        sample = dataset[0]
        # Handle tuple return (sample, metadata)
        if isinstance(sample, tuple):
            sample = sample[0]
        assert sample.shape == (129, 1000), f"Expected (129, 1000), got {sample.shape}"

def test_participants_tsv_exists():
    """Test that participants.tsv exists and has required columns"""
    participants_file = Path(__file__).parent.parent / "data" / "raw" / "hbn" / "participants.tsv"
    assert participants_file.exists(), "participants.tsv should exist"

    import pandas as pd
    df = pd.read_csv(participants_file, sep='\t')
    assert 'participant_id' in df.columns
    assert 'age' in df.columns
    assert 'sex' in df.columns
    assert len(df) > 0, "Should have participants"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
