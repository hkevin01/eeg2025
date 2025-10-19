#!/usr/bin/env python3
"""
Create remaining HDF5 cache files for Challenge 2 (R3, R4, R5)
Runs in tmux to survive VS Code crashes
"""

import os
import sys
import h5py
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append('src')

# Import from eegdash
from eegdash import EEGChallengeDataset

# Bad subjects to filter out (from starter kit)
SUB_RM = [
    'sub-NDARAL647NM6', 'sub-NDARWU142AMH', 'sub-NDARJW869MU2', 'sub-NDARRB569VKZ',
    'sub-NDARUE982PK4', 'sub-NDARAT872FL3', 'sub-NDARRL391TZP', 'sub-NDARZN047ZTP',
    'sub-NDAREF427HZE', 'sub-NDARHP662LXP', 'sub-NDARUB376JBT', 'sub-NDARAA075AMK',
    'sub-NDARLU306PYZ', 'sub-NDARZY256HBB', 'sub-NDARHF528WMY', 'sub-NDARKJ606PKB',
    'sub-NDARAG384NU5', 'sub-NDARME802GCG', 'sub-NDARUA953DWV', 'sub-NDARFN640BFD',
    'sub-NDARRF427VT0', 'sub-NDARWM746JYV', 'sub-NDARDC823NCK', 'sub-NDARKD635ACL',
    'sub-NDARVN859ZWE'
]

def create_cache_for_release(release, is_validation=False):
    """Create HDF5 cache for a single release"""
    print(f"\n{'='*70}")
    print(f"Processing {release} ({'validation' if is_validation else 'training'})")
    print(f"{'='*70}\n")

    # Load dataset with correct API
    print(f"📥 Loading EEGChallengeDataset for {release}...")
    from braindecode.preprocessing import create_fixed_length_windows
    from braindecode.datasets.base import BaseConcatDataset
    
    ds = EEGChallengeDataset(
        release=release,
        task="contrastChangeDetection",
        mini=False,
        description_fields=["subject", "session", "run", "task", "age", "sex", "p_factor"],
        cache_dir="/home/kevin/Projects/eeg2025/data/training",
    )
    
    print(f"✅ Raw dataset loaded: {len(ds.datasets)} subjects")
    
    # Filter bad subjects and short recordings
    print("Filtering subjects...")
    datasets = BaseConcatDataset([ds])
    filtered_datasets = BaseConcatDataset([
        d for d in datasets.datasets
        if (d.description.subject not in SUB_RM and 
            d.raw.n_times >= 4 * 100)  # At least 4 seconds at 100Hz
    ])
    
    print(f"After filtering: {len(filtered_datasets.datasets)} subjects")
    
    # Create windows (4 seconds, 2 second stride)
    print("Creating windows...")
    windowed = create_fixed_length_windows(
        filtered_datasets,
        start_offset_samples=0,
        stop_offset_samples=None,
        window_size_samples=400,  # 4 seconds at 100Hz
        window_stride_samples=200,  # 2 second stride
        drop_last_window=True,
    )
    
    print(f"✅ Created {len(windowed)} windows\n")

    # Prepare data arrays
    all_data = []
    all_subjects = []
    all_p_factors = []
    all_metadata = []

    print("🔄 Extracting windows to cache...")
    failed_count = 0

    for i in tqdm(range(len(windowed)), desc="Processing windows"):
        try:
            X, y = windowed[i]
            desc = windowed.datasets[i].description

            all_data.append(X.numpy())
            all_subjects.append(desc.get('subject', 'unknown'))
            all_p_factors.append(y)

            # Metadata with safe defaults
            metadata = {
                'age': desc.get('age', -1),
                'sex': desc.get('sex', 'unknown'),
                'session': desc.get('session', 'unknown'),
                'run': desc.get('run', -1)
            }
            all_metadata.append(str(metadata))

        except Exception as e:
            failed_count += 1
            if failed_count <= 5:  # Only log first 5
                print(f"\n⚠️  Warning: Failed window {i}: {e}")

    if failed_count > 0:
        print(f"\n⚠️  Total failed: {failed_count}/{len(windowed)} ({100*failed_count/len(windowed):.2f}%)")

    # Convert to numpy
    print("\n💾 Converting to numpy...")
    data_array = np.array(all_data, dtype=np.float32)
    subjects_array = np.array(all_subjects, dtype='S50')
    p_factors_array = np.array(all_p_factors, dtype=np.float32)
    metadata_array = np.array(all_metadata, dtype='S200')

    # Save to HDF5
    output_file = f'data/cached/challenge2_{release}_windows.h5'
    print(f"\n💾 Saving to {output_file}...")

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('data', data=data_array, compression='gzip', compression_opts=4)
        f.create_dataset('subjects', data=subjects_array)
        f.create_dataset('p_factors', data=p_factors_array)
        f.create_dataset('metadata', data=metadata_array)

        # Attributes
        f.attrs['n_windows'] = len(all_data)
        f.attrs['n_channels'] = data_array.shape[1]
        f.attrs['n_timepoints'] = data_array.shape[2]
        f.attrs['release'] = release
        f.attrs['is_validation'] = is_validation

    file_size_gb = os.path.getsize(output_file) / (1024**3)
    print(f"✅ Saved {len(all_data)} windows ({file_size_gb:.2f} GB)")
    print(f"   Shape: {data_array.shape}")

    return len(all_data)

def main():
    """Create cache for R3, R4, R5"""
    print("\n" + "="*70)
    print("CHALLENGE 2 CACHE - REMAINING RELEASES (R3, R4, R5)")
    print("="*70)

    os.makedirs('data/cached', exist_ok=True)

    releases = [
        ('R3', False),
        ('R4', False),
        ('R5', True),
    ]

    total_windows = 0

    for release, is_val in releases:
        try:
            n_windows = create_cache_for_release(release, is_val)
            total_windows += n_windows
        except Exception as e:
            print(f"\n❌ ERROR: {release}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "="*70)
    print("COMPLETE!")
    print("="*70)
    print(f"Total: {total_windows} windows\n")

    for release, _ in releases:
        f = f'data/cached/challenge2_{release}_windows.h5'
        if os.path.exists(f):
            print(f"  ✅ {f} ({os.path.getsize(f)/(1024**3):.2f} GB)")
        else:
            print(f"  ❌ {f} FAILED")

    print("\n🚀 Ready for training!")
    print("   tmux new -s training 'python3 train_challenge2_fast.py'\n")

if __name__ == "__main__":
    main()
