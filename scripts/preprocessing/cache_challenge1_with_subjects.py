#!/usr/bin/env python3
"""
Re-cache Challenge 1 windows WITH subject IDs for subject-aware validation.

This creates new cached files with subject_id included for proper cross-validation.
"""
import os
import sys
from pathlib import Path
import time
import h5py
import numpy as np
import logging
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from eegdash import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)
from braindecode.preprocessing import Preprocessor, preprocess
from braindecode.datautil.windowers import create_windows_from_events

# Setup logging
log_dir = Path("logs/preprocessing")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"cache_with_subjects_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

print("="*80)
print("🔄 CACHING: Challenge 1 Windows WITH Subject IDs")
print("="*80)

# Configuration
RELEASES = ['R1', 'R2', 'R3', 'R4']
CACHE_DIR = Path('data/raw')
OUTPUT_DIR = Path('data/cached')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SFREQ = 100
EPOCH_LEN_S = 2.0
SHIFT_AFTER_STIM = 0.5

def cache_release_with_subjects(release):
    """Cache one release WITH subject IDs"""
    print(f"\n{'='*80}")
    print(f"Processing {release}")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    # Check if output file already exists
    output_file = OUTPUT_DIR / f"challenge1_{release}_windows_with_subjects.h5"
    if output_file.exists():
        print(f"⚠️  {output_file.name} already exists, skipping...")
        logger.info(f"{release}: Already exists, skipping")
        return output_file
    
    try:
        # Load dataset
        print(f"Loading {release}...")
        dataset = EEGChallengeDataset(
            release=release,
            mini=False,
            query=dict(task="contrastChangeDetection"),
            description_fields=["rt_from_stimulus"],
            cache_dir=CACHE_DIR
        )
        print(f"  Loaded: {len(dataset.datasets)} recordings")
        
        # Preprocessing
        print("Preprocessing...")
        preprocessors = [
            Preprocessor(annotate_trials_with_target, apply_on_array=False),
            Preprocessor(add_aux_anchors, apply_on_array=False),
        ]
        preprocess(dataset, preprocessors)
        
        # Filter: keep only recordings with stimulus_anchor
        print("Filtering for stimulus_anchor...")
        dataset = keep_only_recordings_with("stimulus_anchor", dataset)
        print(f"  Kept: {len(dataset.datasets)} recordings with stimulus_anchor")
        
        if len(dataset.datasets) == 0:
            print(f"  ⚠️  No recordings with stimulus_anchor, skipping {release}")
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
        
        # Extract data AND subject IDs
        print("Extracting EEG data, labels, and SUBJECT IDs...")
        X_list = []
        y_list = []
        subject_ids = []
        
        for i in range(len(windows_dataset)):
            x, y, window_ind = windows_dataset[i]
            X_list.append(x)
            y_list.append(y)
            
            # Extract subject ID from description
            description = windows_dataset.datasets[window_ind[0]].description
            if hasattr(description, 'subject_info'):
                subject_id = description.subject_info.his_id
            else:
                # Fallback: extract from path
                path_parts = str(description).split('/')
                for part in path_parts:
                    if part.startswith('sub-'):
                        subject_id = part.replace('sub-', '')
                        break
                else:
                    subject_id = f"unknown_{i}"
            
            subject_ids.append(subject_id)
        
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Convert subject IDs to string array
        subject_ids = np.array(subject_ids, dtype='S20')  # 20-char strings
        
        print(f"  Data shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")
        print(f"  Subject IDs shape: {subject_ids.shape}")
        print(f"  Unique subjects: {len(np.unique(subject_ids))}")
        
        # Save to HDF5 with subject IDs
        print(f"Saving to {output_file}...")
        with h5py.File(output_file, 'w') as f:
            f.create_dataset('eeg', data=X, compression='gzip', compression_opts=4)
            f.create_dataset('labels', data=y, compression='gzip', compression_opts=4)
            f.create_dataset('subject_ids', data=subject_ids, compression='gzip')
            
            # Save metadata
            f.attrs['release'] = release
            f.attrs['n_windows'] = len(X)
            f.attrs['n_channels'] = X.shape[1]
            f.attrs['n_timepoints'] = X.shape[2]
            f.attrs['n_subjects'] = len(np.unique(subject_ids))
            f.attrs['sfreq'] = SFREQ
        
        elapsed = time.time() - start_time
        file_size_mb = output_file.stat().st_size / (1024 * 1024)
        print(f"✅ Saved: {output_file.name} ({file_size_mb:.1f} MB, {elapsed:.1f}s)")
        logger.info(f"{release}: Cached with subjects ({file_size_mb:.1f} MB, {elapsed:.1f}s)")
        
        return output_file
        
    except Exception as e:
        logger.error(f"{release}: Failed: {e}")
        print(f"❌ Error processing {release}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Cache all releases with subject IDs"""
    print("Starting caching with subject IDs...")
    print(f"Releases: {RELEASES}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    total_start = time.time()
    cached_files = []
    
    for release in RELEASES:
        output_file = cache_release_with_subjects(release)
        if output_file:
            cached_files.append(output_file)
    
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*80}")
    print("✅ CACHING WITH SUBJECTS COMPLETE")
    print(f"{'='*80}")
    print(f"Cached files: {len(cached_files)}/{len(RELEASES)}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")
    print(f"\nOutput files:")
    for f in cached_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        with h5py.File(f, 'r') as hf:
            n_subjects = hf.attrs.get('n_subjects', 'unknown')
            n_windows = hf.attrs.get('n_windows', 'unknown')
        print(f"  - {f.name} ({size_mb:.1f} MB)")
        print(f"      {n_windows} windows from {n_subjects} subjects")
    
    print(f"\n🎯 Next: Use these files for subject-aware validation!")

if __name__ == "__main__":
    main()
