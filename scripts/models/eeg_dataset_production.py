#!/usr/bin/env python3
"""
Production EEG Dataset Loader
- Efficient loading of HBN EEG data
- Caching preprocessed windows
- Memory-efficient windowing
"""

import os
from pathlib import Path
import numpy as np
import mne
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle
import hashlib

class ProductionEEGDataset(Dataset):
    """
    Production-ready EEG dataset with caching
    """
    def __init__(
        self,
        data_dir,
        cache_dir=None,
        max_subjects=None,
        window_size=2.0,  # seconds
        overlap=0.5,  # 50% overlap
        target_sfreq=500,
        bandpass=(0.5, 45),
        notch=60,
        preload_all=True,
        verbose=True
    ):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else self.data_dir.parent / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.window_size = window_size
        self.overlap = overlap
        self.target_sfreq = target_sfreq
        self.bandpass = bandpass
        self.notch = notch
        self.preload_all = preload_all
        self.verbose = verbose
        
        # Storage
        self.windows = []
        self.labels = []
        self.subjects = []
        self.metadata = []
        
        # Load data
        if self.verbose:
            print(f"\n{'='*60}")
            print("📂 Production EEG Dataset Loader")
            print(f"{'='*60}")
            print(f"Data directory: {self.data_dir}")
            print(f"Cache directory: {self.cache_dir}")
        
        self._load_dataset(max_subjects)
        
        if self.verbose:
            print(f"\n✅ Dataset ready!")
            print(f"   Total windows: {len(self.windows)}")
            print(f"   Total subjects: {len(set(self.subjects))}")
            if len(self.windows) > 0:
                print(f"   Window shape: {self.windows[0].shape}")
                print(f"   Memory usage: ~{self._estimate_memory():.1f} MB")
    
    def _get_cache_key(self, subject_dir):
        """Generate cache key for a subject"""
        params = f"{self.window_size}_{self.overlap}_{self.target_sfreq}_{self.bandpass}_{self.notch}"
        key = f"{subject_dir.name}_{params}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_path(self, cache_key):
        """Get cache file path"""
        return self.cache_dir / f"{cache_key}.pkl"
    
    def _load_from_cache(self, cache_path):
        """Load preprocessed data from cache"""
        try:
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            return data
        except Exception as e:
            if self.verbose:
                print(f"   Cache load failed: {e}")
            return None
    
    def _save_to_cache(self, cache_path, data):
        """Save preprocessed data to cache"""
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            if self.verbose:
                print(f"   Cache save failed: {e}")
    
    def _process_subject(self, subject_dir):
        """Process a single subject's EEG data"""
        # Check cache first
        cache_key = self._get_cache_key(subject_dir)
        cache_path = self._get_cache_path(cache_key)
        
        if cache_path.exists():
            cached_data = self._load_from_cache(cache_path)
            if cached_data is not None:
                if self.verbose:
                    print(f"   ✓ Loaded from cache: {len(cached_data['windows'])} windows")
                return cached_data
        
        # Process from raw files
        eeg_dir = subject_dir / "eeg"
        if not eeg_dir.exists():
            return None
        
        subject_windows = []
        subject_labels = []
        subject_metadata = []
        
        # Find all .set files
        set_files = list(eeg_dir.glob("*.set"))
        if len(set_files) == 0:
            return None
        
        for set_file in set_files:
            try:
                # Load EEG
                raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)
                
                # Preprocessing
                if self.bandpass:
                    raw.filter(self.bandpass[0], self.bandpass[1], verbose=False)
                if self.notch:
                    raw.notch_filter(self.notch, verbose=False)
                
                # Resample if needed
                if raw.info['sfreq'] != self.target_sfreq:
                    raw.resample(self.target_sfreq, verbose=False)
                
                # Get data
                data = raw.get_data()  # (n_channels, n_samples)
                sfreq = raw.info['sfreq']
                
                # Create windows
                window_samples = int(self.window_size * sfreq)
                step_samples = int(window_samples * (1 - self.overlap))
                
                for start in range(0, data.shape[1] - window_samples + 1, step_samples):
                    window = data[:, start:start + window_samples]
                    
                    # Store
                    subject_windows.append(window.astype(np.float32))
                    subject_labels.append(0)  # Placeholder label
                    subject_metadata.append({
                        'subject': subject_dir.name,
                        'file': set_file.name,
                        'start_sample': start
                    })
                
            except Exception as e:
                if self.verbose:
                    print(f"   Error loading {set_file.name}: {e}")
                continue
        
        if len(subject_windows) == 0:
            return None
        
        # Prepare cache data
        cache_data = {
            'windows': subject_windows,
            'labels': subject_labels,
            'metadata': subject_metadata
        }
        
        # Save to cache
        self._save_to_cache(cache_path, cache_data)
        
        if self.verbose:
            print(f"   ✓ Processed: {len(subject_windows)} windows")
        
        return cache_data
    
    def _load_dataset(self, max_subjects=None):
        """Load all subjects"""
        # Find subjects
        subjects = sorted([d for d in self.data_dir.iterdir() 
                          if d.is_dir() and d.name.startswith("sub-")])
        
        if max_subjects:
            subjects = subjects[:max_subjects]
        
        if self.verbose:
            print(f"\nFound {len(subjects)} subjects")
            print(f"Loading and preprocessing...")
        
        # Process each subject
        for subject_dir in tqdm(subjects, desc="Loading subjects", disable=not self.verbose):
            subject_data = self._process_subject(subject_dir)
            
            if subject_data is None:
                continue
            
            # Add to dataset
            for window, label, metadata in zip(
                subject_data['windows'],
                subject_data['labels'],
                subject_data['metadata']
            ):
                self.windows.append(window)
                self.labels.append(label)
                self.subjects.append(metadata['subject'])
                self.metadata.append(metadata)
    
    def _estimate_memory(self):
        """Estimate memory usage in MB"""
        if len(self.windows) == 0:
            return 0
        window_size = self.windows[0].nbytes / (1024 ** 2)  # MB
        return window_size * len(self.windows)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        """Get a single window"""
        window = torch.from_numpy(self.windows[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return window, label
    
    def get_metadata(self, idx):
        """Get metadata for a window"""
        return self.metadata[idx]
    
    def get_subject_indices(self, subject_id):
        """Get all indices for a specific subject"""
        return [i for i, subj in enumerate(self.subjects) if subj == subject_id]
    
    def get_stats(self):
        """Get dataset statistics"""
        if len(self.windows) == 0:
            return {}
        
        sample_window = self.windows[0]
        unique_subjects = list(set(self.subjects))
        
        return {
            'n_windows': len(self.windows),
            'n_subjects': len(unique_subjects),
            'n_channels': sample_window.shape[0],
            'window_length': sample_window.shape[1],
            'sampling_rate': self.target_sfreq,
            'window_duration': self.window_size,
            'overlap': self.overlap,
            'memory_mb': self._estimate_memory(),
            'subjects': unique_subjects
        }

# Test if run directly
if __name__ == "__main__":
    import sys
    
    # Test loading
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "hbn"
    
    print("Testing Production EEG Dataset Loader")
    print("="*60)
    
    # Load with caching
    dataset = ProductionEEGDataset(
        data_dir=data_dir,
        max_subjects=3,  # Test with 3 subjects
        verbose=True
    )
    
    if len(dataset) > 0:
        # Test getting an item
        window, label = dataset[0]
        print(f"\n📊 Sample window:")
        print(f"   Shape: {window.shape}")
        print(f"   Type: {window.dtype}")
        print(f"   Label: {label}")
        
        # Show stats
        stats = dataset.get_stats()
        print(f"\n📈 Dataset Statistics:")
        for key, value in stats.items():
            if key != 'subjects':
                print(f"   {key}: {value}")
        
        print("\n✅ Dataset loader test passed!")
    else:
        print("\n❌ No data loaded!")
        sys.exit(1)
