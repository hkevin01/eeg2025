#!/usr/bin/env python3
"""
Prepare Challenge 1 Data for Training
Loads CCD task data and saves as HDF5 for fast loading
"""
import sys
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import h5py
import mne
from tqdm import tqdm

print("="*80)
print("Challenge 1 - Data Preparation")
print("="*80)

# Data sources
DATA_DIRS = [
    Path("data/ds005507-bdf"),  # CCD task data
    Path("data/ds005506-bdf"),  # Alternative CCD data
]

def load_ccd_data():
    """Load all CCD task data with response times"""
    all_segments = []
    all_response_times = []
    
    print("\nLoading CCD data...")
    
    for data_dir in DATA_DIRS:
        if not data_dir.exists():
            print(f"  ⚠️  Skipping {data_dir} (not found)")
            continue
            
        print(f"\n  Processing {data_dir.name}...")
        
        # Find subjects
        subjects = sorted(data_dir.glob("sub-*"))
        print(f"    Found {len(subjects)} subjects")
        
        for subject_dir in tqdm(subjects, desc=f"    {data_dir.name}"):
            subject_id = subject_dir.name
            eeg_dir = subject_dir / "eeg"
            
            if not eeg_dir.exists():
                continue
            
            # Find CCD EEG files
            ccd_files = list(eeg_dir.glob("*contrastChangeDetection*.bdf"))
            if not ccd_files:
                continue
            
            for eeg_file in ccd_files:
                try:
                    # Load EEG
                    raw = mne.io.read_raw_bdf(eeg_file, preload=True, verbose=False)
                    
                    # Resample to 100Hz
                    if raw.info['sfreq'] != 100:
                        raw.resample(100, verbose=False)
                    
                    data = raw.get_data()  # (channels, time)
                    
                    # Check channel count
                    if data.shape[0] != 129:
                        continue
                    
                    # Z-score normalize per channel
                    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
                    
                    # Load events to get response times
                    events_file = eeg_file.with_name(eeg_file.name.replace('_eeg.bdf', '_events.tsv'))
                    if not events_file.exists():
                        continue
                    
                    events_df = pd.read_csv(events_file, sep='\t')
                    
                    # Find trial start and button press events
                    trial_starts = events_df[events_df['value'].str.contains('Trial_start', na=False, case=False)]
                    button_presses = events_df[events_df['value'].str.contains('buttonPress', na=False)]
                    
                    # Match trials to button presses
                    for _, trial_row in trial_starts.iterrows():
                        trial_time = trial_row['onset']
                        
                        # Find corresponding button press (response)
                        response_mask = (button_presses['onset'] > trial_time) & (button_presses['onset'] < trial_time + 10)
                        if response_mask.sum() == 0:
                            continue
                        
                        response_time = button_presses[response_mask].iloc[0]['onset'] - trial_time
                        
                        # Validate RT range
                        if response_time < 0.1 or response_time > 5.0:
                            continue
                        
                        # Extract 2-second segment starting from trial
                        start_sample = int(trial_time * 100)  # 100Hz
                        end_sample = start_sample + 200  # 2 seconds
                        
                        if end_sample > data.shape[1]:
                            continue
                        
                        segment = data[:, start_sample:end_sample]
                        
                        all_segments.append(segment)
                        all_response_times.append(response_time)
                
                except Exception as e:
                    print(f"      Error processing {eeg_file.name}: {e}")
                    continue
    
    print(f"\n  ✅ Loaded {len(all_segments)} segments")
    
    return np.array(all_segments, dtype=np.float32), np.array(all_response_times, dtype=np.float32)

def split_train_val(X, y, val_ratio=0.2):
    """Split data into train and validation"""
    n_samples = len(X)
    n_val = int(n_samples * val_ratio)
    
    # Shuffle
    indices = np.random.permutation(n_samples)
    
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    
    return X_train, y_train, X_val, y_val

def main():
    # Load data
    X, y = load_ccd_data()
    
    if len(X) == 0:
        print("\n❌ No data loaded! Check data directories.")
        return
    
    print(f"\nData shape: X {X.shape}, y {y.shape}")
    print(f"Response time range: {y.min():.2f}s - {y.max():.2f}s")
    print(f"Response time mean: {y.mean():.2f}s ± {y.std():.2f}s")
    
    # Split train/val
    print("\nSplitting train/val...")
    X_train, y_train, X_val, y_val = split_train_val(X, y, val_ratio=0.2)
    
    print(f"  Train: X {X_train.shape}, y {y_train.shape}")
    print(f"  Val:   X {X_val.shape}, y {y_val.shape}")
    
    # Save to HDF5
    output_dir = Path("data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "challenge1_data.h5"
    
    print(f"\nSaving to {output_file}...")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('X_train', data=X_train, compression='gzip')
        f.create_dataset('y_train', data=y_train, compression='gzip')
        f.create_dataset('X_val', data=X_val, compression='gzip')
        f.create_dataset('y_val', data=y_val, compression='gzip')
    
    print(f"✅ Data saved successfully!")
    print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()
