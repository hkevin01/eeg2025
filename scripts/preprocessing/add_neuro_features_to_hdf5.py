"""
Pre-compute neuroscience features and add them to existing HDF5 files.

This script:
1. Opens existing HDF5 files (challenge1_R{1-4}_windows.h5)
2. Extracts neuroscience features for all windows
3. Adds 'neuro_features' dataset to each file
4. Makes training much faster (no on-the-fly extraction)

Runtime: ~30 minutes for 41,071 windows
"""

import sys
from pathlib import Path
import h5py
import numpy as np
from tqdm import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

from features.neuroscience_features import extract_all_neuro_features


def add_features_to_hdf5(hdf5_path, overwrite=False):
    """
    Add neuroscience features to an existing HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        overwrite: Whether to overwrite existing features
    """
    print(f"\n{'='*80}")
    print(f"Processing: {hdf5_path.name}")
    print(f"{'='*80}")
    
    # Open file in read/write mode
    with h5py.File(hdf5_path, 'a') as f:
        # Check if features already exist
        if 'neuro_features' in f and not overwrite:
            print(f"✅ Features already exist (use overwrite=True to recompute)")
            n_features = f['neuro_features'].shape[0]
            print(f"   Found {n_features} feature vectors")
            return
        
        # Get EEG data
        eeg_data = f['eeg']
        n_windows = eeg_data.shape[0]
        
        print(f"Total windows: {n_windows}")
        print(f"EEG shape: {eeg_data.shape}")
        
        # Create feature array (6 features per window)
        if 'neuro_features' in f:
            del f['neuro_features']  # Remove old features
        
        features_dataset = f.create_dataset(
            'neuro_features',
            shape=(n_windows, 6),
            dtype='float32',
            compression='gzip',
            compression_opts=4
        )
        
        print(f"\nExtracting features...")
        
        # Process in batches for progress tracking
        batch_size = 100
        failed_count = 0
        
        for i in tqdm(range(0, n_windows, batch_size), desc="Extracting"):
            end_idx = min(i + batch_size, n_windows)
            batch_size_actual = end_idx - i
            
            # Extract features for batch
            batch_features = np.zeros((batch_size_actual, 6), dtype=np.float32)
            
            for j in range(batch_size_actual):
                idx = i + j
                eeg_window = eeg_data[idx]
                
                try:
                    # Extract features (already normalized)
                    features = extract_all_neuro_features(
                        eeg_window,
                        sfreq=100.0,
                        channel_groups=None,
                        stimulus_time=0.0,
                        normalize=True
                    )
                    batch_features[j] = features
                except Exception as e:
                    # Use zeros for failed extractions
                    batch_features[j] = 0.0
                    failed_count += 1
            
            # Save batch to HDF5
            features_dataset[i:end_idx] = batch_features
        
        # Add metadata
        features_dataset.attrs['description'] = 'Neuroscience features: p300_amp, p300_lat, motor_slope, motor_amp, n200_amp, alpha_supp'
        features_dataset.attrs['feature_names'] = 'p300_amplitude,p300_latency,motor_slope,motor_amplitude,n200_amplitude,alpha_suppression'
        features_dataset.attrs['normalized'] = True
        features_dataset.attrs['failed_extractions'] = failed_count
        
        print(f"\n✅ Features added successfully!")
        print(f"   Shape: {features_dataset.shape}")
        print(f"   Failed extractions: {failed_count}/{n_windows} ({100*failed_count/n_windows:.2f}%)")
        
        # Verify a sample
        sample_features = features_dataset[0]
        print(f"\n   Sample features (window 0):")
        feature_names = ['p300_amp', 'p300_lat', 'motor_slope', 'motor_amp', 'n200_amp', 'alpha_supp']
        for name, val in zip(feature_names, sample_features):
            print(f"     {name}: {val:.4f}")


def main():
    print("="*80)
    print("🧠 PRE-COMPUTING NEUROSCIENCE FEATURES FOR HDF5 FILES")
    print("="*80)
    print("\nThis will add 'neuro_features' dataset to existing HDF5 files.")
    print("Training will be MUCH faster after this (~10x speedup).\n")
    
    # Find all HDF5 files
    data_dir = PROJECT_ROOT / 'data' / 'cached'
    hdf5_files = sorted(data_dir.glob('challenge1_R*.h5'))
    
    if not hdf5_files:
        print(f"❌ No HDF5 files found in {data_dir}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files:")
    for f in hdf5_files:
        print(f"  - {f.name}")
    
    # Process each file
    for hdf5_path in hdf5_files:
        try:
            add_features_to_hdf5(hdf5_path, overwrite=False)
        except Exception as e:
            print(f"❌ Error processing {hdf5_path.name}: {e}")
            continue
    
    print(f"\n{'='*80}")
    print("✅ ALL FILES PROCESSED")
    print(f"{'='*80}")
    print("\nYou can now train models much faster!")
    print("The 'neuro_features' dataset is available in each HDF5 file.")
    print("\nNext step: Run training scripts")


if __name__ == "__main__":
    main()
