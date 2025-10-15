#!/usr/bin/env python3
"""
Simple and Robust EEG Dataset Loader
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import mne
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class SimpleEEGDataset(Dataset):
    """Simple EEG dataset loader with robust error handling"""

    def __init__(self, data_dir, max_subjects=None, window_size=1000, verbose=False):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.verbose = verbose

        self.data = []
        self.labels = []

        # Find all subjects with eeg data
        all_subject_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith('sub-')])
        subject_dirs = [d for d in all_subject_dirs if (d / "eeg").exists()]

        if max_subjects:
            subject_dirs = subject_dirs[:max_subjects]

        if self.verbose:
            print(f"\n🔍 Found {len(subject_dirs)} subjects")
            print(f"   Loading with window size: {window_size}")

        # Load data from each subject
        for subject_dir in tqdm(subject_dirs, desc="Loading subjects", disable=not verbose):
            self._load_subject(subject_dir)

        if self.verbose:
            print(f"\n✅ Loaded {len(self.data)} windows")

    def _load_subject(self, subject_dir):
        """Load data from one subject"""
        eeg_dir = subject_dir / "eeg"

        if not eeg_dir.exists():
            if self.verbose:
                print(f"   ⚠️  No eeg directory in {subject_dir.name}")
            return

        # Find all .set files
        set_files = list(eeg_dir.glob("*.set"))

        if self.verbose and len(set_files) > 0:
            print(f"   Found {len(set_files)} files in {subject_dir.name}")

        for set_file in set_files:
            try:
                if self.verbose:
                    print(f"      Loading {set_file.name}...")

                # Load with MNE
                raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=False)

                # Get data
                data = raw.get_data()  # Shape: (n_channels, n_timepoints)
                n_channels, n_timepoints = data.shape

                if self.verbose:
                    print(f"         Shape: {n_channels} channels x {n_timepoints} samples")

                # Create windows
                n_windows = 0
                for start_idx in range(0, n_timepoints - self.window_size, self.window_size // 2):
                    end_idx = start_idx + self.window_size

                    if end_idx > n_timepoints:
                        break

                    window = data[:, start_idx:end_idx]

                    # Simple preprocessing - normalize each channel
                    window_norm = np.zeros_like(window, dtype=np.float32)
                    for ch in range(n_channels):
                        ch_data = window[ch, :]
                        mean = np.mean(ch_data)
                        std = np.std(ch_data)
                        if std > 1e-10:  # Avoid division by zero
                            window_norm[ch, :] = (ch_data - mean) / std
                        else:
                            window_norm[ch, :] = ch_data - mean

                    self.data.append(window_norm)
                    # Binary label: 0 or 1 (for now, random)
                    self.labels.append(np.random.randint(0, 2))
                    n_windows += 1

                if self.verbose:
                    print(f"         Created {n_windows} windows")

            except Exception as e:
                if self.verbose:
                    print(f"   ❌ Error loading {set_file.name}: {e}")
                    import traceback
                    traceback.print_exc()
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.FloatTensor(self.data[idx])
        label = torch.LongTensor([self.labels[idx]])[0]
        return data, label

if __name__ == "__main__":
    # Test
    dataset = SimpleEEGDataset(
        data_dir="/home/kevin/Projects/eeg2025/data/raw/hbn",
        max_subjects=2,
        window_size=1000,
        verbose=True
    )

    if len(dataset) > 0:
        print(f"\n✅ Dataset working!")
        print(f"   Total windows: {len(dataset)}")
        data, label = dataset[0]
        print(f"   Sample shape: {data.shape}")
        print(f"   Sample label: {label}")
    else:
        print("\n❌ No data loaded")
