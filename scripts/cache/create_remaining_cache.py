#!/usr/bin/env python3
"""Create remaining Challenge 2 cache files (R3, R4, R5)"""

import os
import sys
import h5py
import numpy as np
from tqdm import tqdm
import pandas as pd

# Add src to path
sys.path.append('src')

from eegdash import EEGChallengeDataset

# Subject removal list (from starter kit)
SUB_RM = ["NDARAA075AMK", "NDARAA304MH2", "NDARAB509GGT", "NDARAT590BGV",
          "NDARAV894XWT", "NDARAW244CE5", "NDARBA215DEL", "NDARBW459VN7",
          "NDARCJ402VF3", "NDARCV072BRH", "NDARCV911CEE", "NDARDA719ALL",
          "NDARDB228FVJ", "NDARDD143MNP", "NDARDF632JKG", "NDARDG770CCD",
          "NDARDK161LEF", "NDARDM943JCT", "NDARDV991JD9", "NDAREE006PRM",
          "NDAREF472PW0", "NDAREP782JEE", "NDARFA172LGN", "NDARFD110VFD",
          "NDARFD506MNV", "NDARFV620PAW", "NDARGR639LM8", "NDARHA512HDB",
          "NDARJX095DTG"]

def create_cache_for_release(release, is_validation=False):
    """Create HDF5 cache for one release"""
    print(f"\n{'='*60}")
    print(f"Processing Release {release}")
    print(f"{'='*60}")
    
    # Output file
    output_file = f"data/cached/challenge2_R{release}_windows.h5"
    
    if os.path.exists(output_file):
        print(f"✅ {output_file} already exists, skipping...")
        return
    
    # Load dataset
    print(f"Loading EEG Challenge Dataset for R{release}...")
    dataset = EEGChallengeDataset(
        task="contrastChangeDetection",
        releases=[release],
        description_fields=["subject", "session", "run", "age", "sex", "p_factor"],
        cache_path="data/cached",
        num_jobs=4
    )
    
    print(f"Dataset loaded: {len(dataset.datasets)} raw objects")
    
    # Create windows with 4-second duration, 2-second stride
    print("Creating windows (4s windows, 2s stride)...")
    windows = dataset.create_windows(
        duration=4.0,
        stride=2.0,
        is_validation=is_validation
    )
    
    print(f"Created {len(windows.datasets)} windows")
    
    # Filter out bad subjects
    good_indices = []
    for idx in range(len(windows.datasets)):
        desc = windows.datasets[idx].description
        subject = desc.get('subject', '')
        if subject not in SUB_RM:
            good_indices.append(idx)
    
    print(f"Filtered: {len(good_indices)}/{len(windows.datasets)} windows (removed {len(SUB_RM)} bad subjects)")
    
    # Extract data
    print("Extracting data to numpy arrays...")
    data_list = []
    subject_list = []
    p_factor_list = []
    age_list = []
    sex_list = []
    session_list = []
    run_list = []
    
    failed_count = 0
    
    for idx in tqdm(good_indices, desc="Processing windows"):
        try:
            X, y, i = windows[idx]
            desc = windows.datasets[idx].description
            
            data_list.append(X)
            subject_list.append(desc.get('subject', ''))
            p_factor_list.append(desc.get('p_factor', np.nan))
            age_list.append(desc.get('age', np.nan))
            sex_list.append(desc.get('sex', ''))
            session_list.append(desc.get('session', ''))
            run_list.append(desc.get('run', ''))
            
        except Exception as e:
            failed_count += 1
            # Continue on errors
            continue
    
    if failed_count > 0:
        print(f"⚠️  {failed_count} windows failed to process (continuing)")
    
    # Stack arrays
    print(f"Stacking {len(data_list)} windows...")
    data_array = np.stack(data_list, axis=0)  # (n_windows, n_channels, n_times)
    
    print(f"Data shape: {data_array.shape}")
    print(f"Data size: {data_array.nbytes / 1e9:.2f} GB")
    
    # Save to HDF5
    print(f"Saving to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        # Create dataset with compression
        f.create_dataset('data', data=data_array, compression='gzip', compression_opts=4)
        
        # Save metadata
        f.create_dataset('subjects', data=np.array(subject_list, dtype='S'))
        f.create_dataset('p_factors', data=np.array(p_factor_list, dtype=np.float32))
        f.create_dataset('ages', data=np.array(age_list, dtype=np.float32))
        f.create_dataset('sexes', data=np.array(sex_list, dtype='S'))
        f.create_dataset('sessions', data=np.array(session_list, dtype='S'))
        f.create_dataset('runs', data=np.array(run_list, dtype='S'))
        
        # Store attributes
        f.attrs['release'] = release
        f.attrs['task'] = 'contrastChangeDetection'
        f.attrs['n_windows'] = len(data_list)
        f.attrs['n_channels'] = data_array.shape[1]
        f.attrs['n_times'] = data_array.shape[2]
        f.attrs['is_validation'] = is_validation
    
    file_size = os.path.getsize(output_file) / 1e9
    print(f"✅ Saved: {output_file} ({file_size:.2f} GB)")
    print(f"   Windows: {len(data_list)}, Channels: {data_array.shape[1]}, Times: {data_array.shape[2]}")

if __name__ == "__main__":
    print("Creating remaining Challenge 2 cache files (R3, R4, R5)")
    print(f"Output directory: data/cached/")
    
    # Create R3, R4, R5 training caches
    for release in [3, 4]:
        create_cache_for_release(release, is_validation=False)
    
    # Create R5 validation cache
    create_cache_for_release(5, is_validation=True)
    
    print("\n" + "="*60)
    print("✅ ALL CACHE FILES CREATED!")
    print("="*60)
    
    # List all files
    print("\nCache files:")
    for r in range(1, 6):
        filepath = f"data/cached/challenge2_R{r}_windows.h5"
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1e9
            print(f"  ✅ R{r}: {size:.2f} GB")
        else:
            print(f"  ❌ R{r}: Not found")
