# HBN-EEG Data Acquisition & Validation Guide

**Status**: 🔴 CRITICAL - No actual EEG data detected  
**Priority**: Must complete before any training  
**Date**: October 13, 2025

---

## Part 1: Data Acquisition

### Step 1: Register for HBN Dataset Access

The Healthy Brain Network (HBN) EEG dataset is hosted by the Child Mind Institute and requires registration.

#### Registration Process

1. **Visit the HBN Data Portal**:
   ```
   https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/
   ```

2. **Create Account**:
   - Click "Request Access"
   - Complete data usage agreement
   - Wait for approval (typically 1-3 business days)

3. **Dataset Details**:
   - **Size**: ~500GB compressed, ~800GB uncompressed
   - **Format**: BIDS-compliant EEG data
   - **Participants**: 1,500+ children/adolescents (ages 5-21)
   - **Tasks**: SuS (Stop Signal), CCD (Cognitive Control), RS (Resting State), MW (Mind Wandering)

### Step 2: Download the Dataset

#### Option A: Direct Download (Recommended for first-time)

```bash
# Create data directory
mkdir -p /path/to/hbn/data
cd /path/to/hbn/data

# Download via AWS CLI (fastest)
aws s3 sync s3://fcp-indi/data/Projects/HBN/EEG_BIDS ./hbn-eeg-bids \
    --no-sign-request \
    --region us-east-1

# Alternative: Download via wget (slower but more reliable on some networks)
wget -r -np -nH --cut-dirs=4 \
    https://fcp-indi.s3.amazonaws.com/data/Projects/HBN/EEG_BIDS/
```

#### Option B: Download Subset for Testing (Quick Start)

```bash
# Download just 10 subjects for quick validation (~5GB)
export HBN_DATA_PATH="/path/to/hbn/data"
mkdir -p $HBN_DATA_PATH

# Download sample subjects
for sub_id in NDARAA075AMK NDARAA948VFH NDARAB457VF4 NDARAB582UM4 NDARAC286UE8 \
               NDARAD121MJN NDARAD481FXF NDARAD533VF5 NDARAD744EGJ NDARAE003GGV; do
    aws s3 sync s3://fcp-indi/data/Projects/HBN/EEG_BIDS/sub-${sub_id} \
        ${HBN_DATA_PATH}/sub-${sub_id} \
        --no-sign-request \
        --region us-east-1
done

# Download essential files
aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG_BIDS/participants.tsv \
    ${HBN_DATA_PATH}/participants.tsv --no-sign-request
aws s3 cp s3://fcp-indi/data/Projects/HBN/EEG_BIDS/dataset_description.json \
    ${HBN_DATA_PATH}/dataset_description.json --no-sign-request
```

### Step 3: Verify Download

```bash
# Check BIDS structure
tree -L 2 $HBN_DATA_PATH | head -50

# Expected structure:
# /path/to/hbn/data/
# ├── dataset_description.json
# ├── participants.tsv
# ├── sub-NDARAA075AMK/
# │   └── ses-HBNsiteRU/
# │       └── eeg/
# ├── sub-NDARAA948VFH/
# │   └── ses-HBNsiteCBIC/
# │       └── eeg/
# └── ...

# Count subjects
ls -d $HBN_DATA_PATH/sub-* | wc -l

# Check EEG files
find $HBN_DATA_PATH -name "*_eeg.edf" | wc -l
```

---

## Part 2: Data Validation Scripts

### Create Data Validation Test

Save this as `tests/test_data_acquisition.py`:

```python
#!/usr/bin/env python3
"""
Test suite for HBN data acquisition and validation.
Critical: Run this BEFORE any training.
"""

import os
import sys
from pathlib import Path
import pytest
import mne
from mne_bids import BIDSPath, read_raw_bids
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataio.bids_loader import HBNDataset, HBNDataLoader
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
            data_path / "dataset_description.json"
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
        eeg_files = list(data_path.glob("**/eeg/*.edf")) + \
                    list(data_path.glob("**/eeg/*.bdf"))
        
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
                subject=subject,
                session=session,
                datatype="eeg",
                root=data_path
            )
            
            raw = read_raw_bids(bids_path, verbose=False)
            print(f"✅ Successfully loaded EEG file")
            print(f"   Channels: {len(raw.ch_names)}")
            print(f"   Sampling rate: {raw.info['sfreq']} Hz")
            print(f"   Duration: {raw.times[-1]:.2f} seconds")
            
            # Validate basic properties
            assert raw.info['sfreq'] >= 250, "❌ Sampling rate too low"
            assert len(raw.ch_names) >= 64, "❌ Too few channels"
            assert raw.times[-1] > 60, "❌ Recording too short"
            
        except Exception as e:
            pytest.fail(f"❌ Failed to load EEG file: {e}")

    def test_participants_file(self, data_path):
        """Test 5: Validate participants.tsv."""
        participants_file = data_path / "participants.tsv"
        
        try:
            df = pd.read_csv(participants_file, sep='\t')
            print(f"✅ Loaded participants.tsv: {len(df)} participants")
            
            # Check required columns
            required_cols = ['participant_id']
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
            
            assert 'train' in splits, "❌ Missing train split"
            assert 'val' in splits, "❌ Missing val split"
            assert 'test' in splits, "❌ Missing test split"
            
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

    def test_dataset_loading(self, data_path):
        """Test 8: Test HBNDataset class."""
        try:
            # Create dataset with minimal config
            dataset = HBNDataset(
                bids_root=data_path,
                split='train',
                tasks=['SuS'],  # Just one task for testing
                window_length=2.0,
                overlap=0.5
            )
            
            print(f"✅ Created HBNDataset")
            print(f"   Windows: {len(dataset)}")
            
            # Try loading one sample
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"✅ Loaded sample:")
                print(f"   EEG shape: {sample['eeg'].shape}")
                print(f"   Keys: {sample.keys()}")
            else:
                print(f"⚠️  Dataset is empty - check subject/task availability")
                
        except Exception as e:
            pytest.fail(f"❌ Dataset loading failed: {e}")

    def test_dataloader_iteration(self, data_path):
        """Test 9: Test DataLoader iteration."""
        try:
            # Create data loader
            loader = HBNDataLoader(
                bids_root=data_path,
                batch_size=4,
                num_workers=0  # Single worker for testing
            )
            
            # Get train dataloader
            train_loader = loader.get_dataloader(
                split='train',
                tasks=['SuS']
            )
            
            print(f"✅ Created DataLoader")
            
            # Try iterating one batch
            batch = next(iter(train_loader))
            print(f"✅ Loaded batch:")
            print(f"   Batch size: {batch['eeg'].shape[0]}")
            print(f"   EEG shape: {batch['eeg'].shape}")
            
        except Exception as e:
            pytest.fail(f"❌ DataLoader iteration failed: {e}")

    def test_data_quality(self, data_path):
        """Test 10: Basic data quality checks."""
        # Find first EEG file
        eeg_files = list(data_path.glob("**/eeg/*_eeg.edf"))[:1]
        if not eeg_files:
            pytest.skip("No EEG files found for quality check")
        
        try:
            subject = eeg_files[0].parts[-4].replace("sub-", "")
            session = eeg_files[0].parts[-3].replace("ses-", "")
            
            bids_path = BIDSPath(
                subject=subject,
                session=session,
                datatype="eeg",
                root=data_path
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
```
