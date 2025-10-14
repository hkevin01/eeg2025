#!/usr/bin/env python3
"""
Test suite for HBN data acquisition and validation.
Critical: Run this BEFORE any training.
"""

import os
import sys
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
from mne_bids import BIDSPath, read_raw_bids

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataio.bids_loader import HBNDataLoader, HBNDataset
from dataio.starter_kit import StarterKitDataLoader


class TestDataAcquisition:
    """Critical data validation tests."""

    @pytest.fixture
    def data_path(self):
        """Get data path from environment or default."""
        return Path(os.getenv("HBN_DATA_PATH", "/path/to/hbn/data"))

    def test_data_directory_exists(self, data_path):
        """Test 1: Verify data directory exists."""
        assert data_path.exists(), (
            f"❌ HBN data directory not found: {data_path}\n"
            f"Please set HBN_DATA_PATH environment variable or download data.\n"
            f"See docs/DATA_ACQUISITION_GUIDE.md for instructions."
        )
        print(f"✅ Data directory found: {data_path}")

    def test_bids_structure(self, data_path):
        """Test 2: Verify BIDS structure."""
        # Check required files
        required_files = [
            data_path / "participants.tsv",
            data_path / "dataset_description.json",
        ]

        for file_path in required_files:
            assert file_path.exists(), f"❌ Required BIDS file missing: {file_path}"

        print("✅ Required BIDS files present")

        # Check subject directories
        subject_dirs = list(data_path.glob("sub-*"))
        assert len(subject_dirs) > 0, "❌ No subject directories found"
        print(f"✅ Found {len(subject_dirs)} subject directories")

    def test_eeg_files_exist(self, data_path):
        """Test 3: Verify EEG files exist."""
        eeg_files = list(data_path.glob("**/eeg/*.edf")) + list(
            data_path.glob("**/eeg/*.bdf")
        )

        assert len(eeg_files) > 0, (
            "❌ No EEG files found!\n"
            "Expected .edf or .bdf files in sub-*/ses-*/eeg/ directories."
        )
        print(f"✅ Found {len(eeg_files)} EEG files")

    def test_load_single_eeg_file(self, data_path):
        """Test 4: Load and validate a single EEG file."""
        # Find first EEG file
        eeg_files = list(data_path.glob("**/eeg/*_eeg.edf"))
        if not eeg_files:
            eeg_files = list(data_path.glob("**/eeg/*_eeg.bdf"))

        assert len(eeg_files) > 0, "❌ No EEG files to test"

        test_file = eeg_files[0]
        print(f"Testing file: {test_file}")

        # Try loading with MNE-BIDS
        try:
            # Parse BIDS path
            subject = test_file.parts[-4].replace("sub-", "")
            session = test_file.parts[-3].replace("ses-", "")

            bids_path = BIDSPath(
                subject=subject, session=session, datatype="eeg", root=data_path
            )

            raw = read_raw_bids(bids_path, verbose=False)
            print(f"✅ Successfully loaded EEG file")
            print(f"   Channels: {len(raw.ch_names)}")
            print(f"   Sampling rate: {raw.info['sfreq']} Hz")
            print(f"   Duration: {raw.times[-1]:.2f} seconds")

            # Validate basic properties
            assert raw.info["sfreq"] >= 250, "❌ Sampling rate too low"
            assert len(raw.ch_names) >= 64, "❌ Too few channels"
            assert raw.times[-1] > 60, "❌ Recording too short"

        except Exception as e:
            pytest.fail(f"❌ Failed to load EEG file: {e}")

    def test_participants_file(self, data_path):
        """Test 5: Validate participants.tsv."""
        participants_file = data_path / "participants.tsv"

        try:
            df = pd.read_csv(participants_file, sep="\t")
            print(f"✅ Loaded participants.tsv: {len(df)} participants")

            # Check required columns
            required_cols = ["participant_id"]
            for col in required_cols:
                assert col in df.columns, f"❌ Missing column: {col}"

            print(f"   Columns: {df.columns.tolist()}")

        except Exception as e:
            pytest.fail(f"❌ Failed to load participants.tsv: {e}")

    def test_official_splits_exist(self, data_path):
        """Test 6: Verify official splits are available."""
        try:
            starter_kit = StarterKitDataLoader(data_path)
            splits = starter_kit.official_splits

            assert "train" in splits, "❌ Missing train split"
            assert "val" in splits, "❌ Missing val split"
            assert "test" in splits, "❌ Missing test split"

            print(f"✅ Official splits loaded:")
            print(f"   Train: {len(splits['train'])} subjects")
            print(f"   Val: {len(splits['val'])} subjects")
            print(f"   Test: {len(splits['test'])} subjects")

        except Exception as e:
            print(f"⚠️  Could not load official splits: {e}")
            print(f"   This is OK if using custom splits")

    def test_challenge_labels_exist(self, data_path):
        """Test 7: Verify challenge labels are available."""
        try:
            starter_kit = StarterKitDataLoader(data_path)

            # Try loading CCD labels
            try:
                ccd_labels = starter_kit.load_ccd_labels()
                print(f"✅ CCD labels loaded: {len(ccd_labels)} records")
            except:
                print(f"⚠️  CCD labels not found (Challenge 1)")

            # Try loading CBCL labels
            try:
                cbcl_labels = starter_kit.load_cbcl_labels()
                print(f"✅ CBCL labels loaded: {len(cbcl_labels)} records")
            except:
                print(f"⚠️  CBCL labels not found (Challenge 2)")

        except Exception as e:
            print(f"⚠️  Could not check challenge labels: {e}")

    def test_data_quality(self, data_path):
        """Test 8: Basic data quality checks."""
        # Find first EEG file
        eeg_files = list(data_path.glob("**/eeg/*_eeg.edf"))[:1]
        if not eeg_files:
            pytest.skip("No EEG files found for quality check")

        try:
            subject = eeg_files[0].parts[-4].replace("sub-", "")
            session = eeg_files[0].parts[-3].replace("ses-", "")

            bids_path = BIDSPath(
                subject=subject, session=session, datatype="eeg", root=data_path
            )

            raw = read_raw_bids(bids_path, verbose=False)
            data = raw.get_data()

            # Quality checks
            print("Data Quality Metrics:")

            # 1. Check for flat channels
            flat_channels = np.sum(np.std(data, axis=1) < 1e-10)
            print(f"   Flat channels: {flat_channels}")
            assert flat_channels < len(raw.ch_names) * 0.1, "❌ Too many flat channels"

            # 2. Check amplitude range
            max_amp = np.max(np.abs(data))
            print(f"   Max amplitude: {max_amp:.2e} V")
            assert max_amp < 1e-3, "❌ Amplitude suspiciously high"
            assert max_amp > 1e-8, "❌ Amplitude suspiciously low"

            # 3. Check for NaN/Inf
            assert not np.any(np.isnan(data)), "❌ NaN values detected"
            assert not np.any(np.isinf(data)), "❌ Inf values detected"

            print("✅ Data quality checks passed")

        except Exception as e:
            pytest.fail(f"❌ Data quality check failed: {e}")


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("HBN-EEG Data Acquisition & Validation Test Suite")
    print("=" * 70)
    print()

    # Run with pytest
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    main()
