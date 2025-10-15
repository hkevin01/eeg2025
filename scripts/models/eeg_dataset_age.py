#!/usr/bin/env python3
"""
EEG Dataset with Real Age Labels
=================================
Load EEG data and map to real age labels from participants.tsv
"""
from pathlib import Path
import mne
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class AgeEEGDataset(Dataset):
    """EEG dataset with real age labels from participants.tsv"""

    def __init__(self, data_dir, max_subjects=None, segment_length=512):
        """
        Args:
            data_dir: Path to HBN data directory
            max_subjects: Maximum number of subjects to load
            segment_length: Length of EEG segments in samples
        """
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length

        # Load participants metadata
        print("📋 Loading participants metadata...")
        participants_file = self.data_dir / "participants.tsv"
        self.participants_df = pd.read_csv(participants_file, sep='\t')
        print(f"   Found {len(self.participants_df)} participants")

        # Find available EEG files
        print("🔍 Finding EEG files...")
        self.eeg_files = []
        self.ages = []
        self.sexes = []

        for _, row in self.participants_df.iterrows():
            subject_id = row['participant_id']
            age = row['age']
            sex = row['sex']

            # Look for RestingState EEG
            subject_dir = self.data_dir / subject_id / "eeg"
            if not subject_dir.exists():
                continue

            # Find EEG files
            eeg_files = list(subject_dir.glob("*RestingState*.set"))
            if not eeg_files:
                continue

            # Use first available file
            eeg_file = eeg_files[0]
            self.eeg_files.append(eeg_file)
            self.ages.append(age)
            self.sexes.append(1 if sex == 'M' else 0)

            if max_subjects and len(self.eeg_files) >= max_subjects:
                break

        print(f"   Found {len(self.eeg_files)} subjects with EEG data")
        print(f"   Age range: {min(self.ages):.1f} - {max(self.ages):.1f} years")

        # Pre-load and segment data
        print("📂 Loading EEG data...")
        self.segments = []
        self.segment_ages = []

        for eeg_file, age in zip(self.eeg_files, self.ages):
            try:
                # Load EEG
                raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
                data = raw.get_data()

                # Standardize per channel
                data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)

                # Create segments
                n_samples = data.shape[1]
                n_segments = n_samples // segment_length

                for i in range(n_segments):
                    start = i * segment_length
                    end = start + segment_length
                    segment = data[:, start:end]

                    self.segments.append(torch.FloatTensor(segment))
                    self.segment_ages.append(age)

            except Exception as e:
                print(f"   ⚠️  Error loading {eeg_file.name}: {e}")
                continue

        print(f"   Created {len(self.segments)} segments")

        # Normalize ages for training (0-1 range)
        self.ages_array = np.array(self.segment_ages)
        self.age_min = self.ages_array.min()
        self.age_max = self.ages_array.max()
        self.ages_normalized = (self.ages_array - self.age_min) / (self.age_max - self.age_min)

        print(f"   Age normalized: {self.age_min:.1f} - {self.age_max:.1f} → 0.0 - 1.0")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        """Get segment and normalized age"""
        segment = self.segments[idx]
        age_normalized = self.ages_normalized[idx]

        return segment, torch.FloatTensor([age_normalized])

    def denormalize_age(self, age_normalized):
        """Convert normalized age back to years"""
        return age_normalized * (self.age_max - self.age_min) + self.age_min


if __name__ == "__main__":
    # Test the dataset
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw" / "hbn"

    print("\n" + "="*80)
    print("Testing AgeEEGDataset")
    print("="*80)

    dataset = AgeEEGDataset(data_dir=data_dir, max_subjects=5)

    print("\n📊 Dataset Info:")
    print(f"   Total segments: {len(dataset)}")
    print(f"   Age range (normalized): {dataset.ages_normalized.min():.3f} - {dataset.ages_normalized.max():.3f}")
    print(f"   Age range (years): {dataset.age_min:.1f} - {dataset.age_max:.1f}")

    # Test a sample
    print("\n🧪 Testing sample:")
    segment, age = dataset[0]
    print(f"   Segment shape: {segment.shape}")
    print(f"   Age (normalized): {age.item():.3f}")
    print(f"   Age (years): {dataset.denormalize_age(age.item()):.1f}")

    print("\n✅ Dataset test passed!")
