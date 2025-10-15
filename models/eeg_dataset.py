"""
EEG Dataset for PyTorch Training
Loads HBN BIDS EEG data and prepares it for model training
"""

import glob
from pathlib import Path
import numpy as np
import mne
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional

class EEGDataset(Dataset):
    """PyTorch Dataset for HBN EEG data"""
    
    def __init__(
        self,
        data_dir: str = "data/raw/hbn",
        window_size: int = 1000,  # 2 seconds at 500 Hz
        overlap: float = 0.5,
        l_freq: float = 0.5,
        h_freq: float = 45.0,
        notch_freq: float = 60.0,
        max_files_per_subject: Optional[int] = None
    ):
        """
        Args:
            data_dir: Path to HBN data directory
            window_size: Window size in samples (500 Hz)
            overlap: Overlap fraction for windows
            l_freq: Low-pass filter frequency
            h_freq: High-pass filter frequency
            notch_freq: Notch filter frequency
            max_files_per_subject: Max EEG files to load per subject
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.overlap = overlap
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freq = notch_freq
        self.max_files_per_subject = max_files_per_subject
        
        # Load and prepare all data
        self.samples = []
        self._load_data()
        
    def _load_data(self):
        """Load all EEG files and create windowed samples"""
        print(f"Loading data from {self.data_dir}...")
        
        subjects = sorted(self.data_dir.glob("sub-*"))
        print(f"Found {len(subjects)} subjects")
        
        for subj_path in subjects:
            subj_name = subj_path.name
            print(f"  Processing {subj_name}...")
            
            eeg_files = list(subj_path.glob("**/*.set"))
            
            if self.max_files_per_subject:
                eeg_files = eeg_files[:self.max_files_per_subject]
            
            for eeg_file in eeg_files:
                try:
                    raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
                    
                    # Preprocessing
                    raw.filter(self.l_freq, self.h_freq, verbose=False)
                    raw.notch_filter(self.notch_freq, verbose=False)
                    
                    # Create windows
                    data = raw.get_data()  # Shape: (n_channels, n_samples)
                    n_channels, n_samples = data.shape
                    
                    step = int(self.window_size * (1 - self.overlap))
                    
                    for start_idx in range(0, n_samples - self.window_size, step):
                        end_idx = start_idx + self.window_size
                        window = data[:, start_idx:end_idx]
                        
                        # Create sample dict
                        sample = {
                            'eeg': torch.FloatTensor(window),
                            'subject': subj_name,
                            'file': eeg_file.name,
                            'start_idx': start_idx
                        }
                        self.samples.append(sample)
                        
                except Exception as e:
                    print(f"    ⚠️  Error loading {eeg_file.name}: {e}")
                    continue
        
        print(f"✅ Loaded {len(self.samples)} windows from {len(subjects)} subjects")
        print(f"   Window size: {self.window_size} samples ({self.window_size/500:.2f}s at 500Hz)")
        print(f"   EEG shape per window: {self.samples[0]['eeg'].shape if self.samples else 'N/A'}")
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> dict:
        return self.samples[idx]
    
    def get_feature_dim(self) -> Tuple[int, int]:
        """Get (n_channels, window_size) for model input"""
        if self.samples:
            return self.samples[0]['eeg'].shape
        return (0, 0)


def test_dataset():
    """Test the dataset loading"""
    print("=" * 60)
    print("Testing EEGDataset")
    print("=" * 60)
    
    dataset = EEGDataset(
        window_size=1000,  # 2 seconds at 500 Hz
        overlap=0.5,
        max_files_per_subject=2  # Limit for testing
    )
    
    print()
    print(f"Dataset size: {len(dataset)} samples")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print()
        print("Sample 0:")
        print(f"  EEG shape: {sample['eeg'].shape}")
        print(f"  Subject: {sample['subject']}")
        print(f"  File: {sample['file']}")
        print(f"  EEG stats: min={sample['eeg'].min():.3f}, max={sample['eeg'].max():.3f}, mean={sample['eeg'].mean():.3f}")
    
    print()
    print("=" * 60)
    print("✅ Dataset test complete!")


if __name__ == "__main__":
    test_dataset()
