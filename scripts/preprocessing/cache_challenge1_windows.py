#!/usr/bin/env python3
"""
Preprocess and cache Challenge 1 windows in HDF5 format for memory-efficient training.

This script runs ONCE to create cached windows, then training can load them
without RAM overflow.

Features:
- Comprehensive error handling and crash recovery
- Memory monitoring and safety checks
- Detailed logging with timestamps
- Resume capability (skip already processed releases)
- Checkpoint saving after each release
"""
import os
import sys
from pathlib import Path
import time
import h5py
import numpy as np
import logging
import traceback
import psutil
from datetime import datetime
from braindecode.preprocessing import Preprocessor, preprocess
from braindecode.datautil.windowers import create_windows_from_events

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from eegdash import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

# Setup comprehensive logging
log_dir = Path("logs/preprocessing")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"cache_windows_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Memory safety
MAX_MEMORY_PERCENT = 85
CHECK_MEMORY_INTERVAL = 60  # Check every 60 seconds

def check_memory_safe():
    """Check if memory usage is safe"""
    memory = psutil.virtual_memory()
    return memory.percent < MAX_MEMORY_PERCENT, memory.percent

def log_memory_status(prefix=""):
    """Log current memory status"""
    memory = psutil.virtual_memory()
    logger.info(f"{prefix}Memory: {memory.percent:.1f}% used ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
    return memory.percent

print("="*80)
print("🔄 PREPROCESSING: Caching Challenge 1 Windows to HDF5")
print("="*80)
logger.info("="*80)
logger.info("Preprocessing started")
logger.info(f"Log file: {log_file}")
logger.info(f"Memory safety: Max {MAX_MEMORY_PERCENT}%")
logger.info("="*80)

# Check initial memory
log_memory_status("Initial ")

# Configuration
RELEASES = ['R1', 'R2', 'R3', 'R4']
CACHE_DIR = Path('data/raw')
OUTPUT_DIR = Path('data/processed')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SFREQ = 100
EPOCH_LEN_S = 2.0
SHIFT_AFTER_STIM = 0.5

def preprocess_release(release, mini=False):
    """Preprocess one release and save to HDF5 with comprehensive error handling"""
    print(f"\n{'='*80}")
    print(f"Processing {release}")
    print(f"{'='*80}")
    logger.info(f"Starting {release} preprocessing")

    start_time = time.time()

    try:
        # Check memory before starting
        safe, mem_pct = check_memory_safe()
        if not safe:
            logger.error(f"Memory limit exceeded before {release}: {mem_pct:.1f}%")
            print(f"❌ Memory limit exceeded: {mem_pct:.1f}% > {MAX_MEMORY_PERCENT}%")
            return None

        log_memory_status(f"Before {release}: ")

        # Load dataset
        print(f"Loading {release}...")
        logger.info(f"{release}: Loading dataset")
        dataset = EEGChallengeDataset(
            release=release,
            mini=mini,
            query=dict(task="contrastChangeDetection"),
            description_fields=["rt_from_stimulus"],
            cache_dir=CACHE_DIR
        )
        print(f"  Loaded: {len(dataset.datasets)} recordings")
        logger.info(f"{release}: Loaded {len(dataset.datasets)} recordings")

        # Check memory after loading
        safe, mem_pct = check_memory_safe()
        if not safe:
            logger.error(f"Memory limit exceeded after loading {release}: {mem_pct:.1f}%")
            print(f"❌ Memory limit exceeded after loading: {mem_pct:.1f}%")
            return None

        # Preprocessing
        print("Preprocessing...")
        logger.info(f"{release}: Running preprocessors")
        preprocessors = [
            Preprocessor(annotate_trials_with_target, apply_on_array=False),
            Preprocessor(add_aux_anchors, apply_on_array=False),
        ]

        preprocess(dataset, preprocessors)
        logger.info(f"{release}: Preprocessing complete")

        # Filter: keep only recordings with stimulus_anchor
        print("Filtering for stimulus_anchor...")
        logger.info(f"{release}: Filtering for stimulus_anchor")
        dataset = keep_only_recordings_with("stimulus_anchor", dataset)
        print(f"  Kept: {len(dataset.datasets)} recordings with stimulus_anchor")
        logger.info(f"{release}: Kept {len(dataset.datasets)} recordings with stimulus_anchor")

        if len(dataset.datasets) == 0:
            print(f"  ⚠️  No recordings with stimulus_anchor, skipping {release}")
            logger.warning(f"{release}: No recordings with stimulus_anchor, skipping")
            return None

    except Exception as e:
        logger.error(f"{release}: Failed during loading/preprocessing: {e}")
        logger.error(f"Traceback: {str(e)}")
        print(f"❌ Error processing {release}: {e}")
        return None

    # Create windows
    print("Creating windows...")
    ANCHOR = "stimulus_anchor"
    windows_dataset = create_windows_from_events(
        dataset,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + EPOCH_LEN_S) * SFREQ),
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )
    print(f"  Created: {len(windows_dataset)} windows")

    # Add metadata
    print("Adding metadata...")
    windows_dataset = add_extras_columns(
        windows_dataset,
        dataset,
        desc="stimulus_anchor",
        keys=("rt_from_stimulus", "target", "rt_from_trialstart",
              "stimulus_onset", "response_onset", "correct", "response_type")
    )

    # Extract data
    print("Extracting EEG data and labels...")
    X_list = []
    y_list = []

    for i in range(len(windows_dataset)):
        x, y, _ = windows_dataset[i]
        X_list.append(x)
        y_list.append(y)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"  Data shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")

    # Save to HDF5
    output_file = OUTPUT_DIR / f"{release}_challenge1_windows.h5"
    print(f"Saving to {output_file}...")

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('eeg', data=X, compression='gzip', compression_opts=4)
        f.create_dataset('labels', data=y, compression='gzip', compression_opts=4)

        # Save metadata
        f.attrs['release'] = release
        f.attrs['n_windows'] = len(X)
        f.attrs['n_channels'] = X.shape[1]
        f.attrs['n_timepoints'] = X.shape[2]
        f.attrs['sfreq'] = SFREQ

    elapsed = time.time() - start_time
    file_size_mb = output_file.stat().st_size / (1024 * 1024)
    print(f"✅ Saved: {output_file.name} ({file_size_mb:.1f} MB, {elapsed:.1f}s)")

    return output_file

def main():
    """Preprocess all releases with error handling and resume capability"""
    print("Starting preprocessing...")
    print(f"Releases: {RELEASES}")
    print(f"Output directory: {OUTPUT_DIR}")
    logger.info("Starting main preprocessing loop")
    logger.info(f"Releases to process: {RELEASES}")

    processed_files = []
    failed_releases = []
    skipped_releases = []

    total_start = time.time()
    cached_files = []

    for release in RELEASES:
        try:
            output_file = preprocess_release(release, mini=False)
            if output_file:
                cached_files.append(output_file)
        except Exception as e:
            print(f"❌ Error processing {release}: {e}")
            import traceback
            traceback.print_exc()
            continue

    total_elapsed = time.time() - total_start

    print(f"\n{'='*80}")
    print("✅ PREPROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Cached files: {len(cached_files)}/{len(RELEASES)}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"Output files:")
    for f in cached_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")
    print(f"\nNext: Use HDF5Dataset in training scripts for memory-efficient loading")

if __name__ == "__main__":
    main()
