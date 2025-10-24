#!/usr/bin/env python3
"""
Quick test to verify HBN data loading works
"""
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

# Import dataset from training script
import mne
import pandas as pd
from tqdm import tqdm

class ResponseTimeDataset(torch.utils.data.Dataset):
    """Load EEG windows with response times from BIDS events"""
    
    def __init__(self, data_dirs, max_subjects=None, augment=False):
        self.segments = []
        self.response_times = []
        self.subject_ids = []
        self.augment = augment
        
        print(f"Loading HBN Challenge 1 data (max_subjects={max_subjects})...")
        
        for data_dir in data_dirs:
            data_dir = Path(data_dir)
            participants_file = data_dir / "participants.tsv"
            
            if not participants_file.exists():
                print(f"  Warning: {data_dir}: participants.tsv not found")
                continue
            
            df = pd.read_csv(participants_file, sep='\t')
            
            if max_subjects:
                df = df.head(max_subjects)
            
            print(f"  {data_dir.name}: {len(df)} subjects")
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"  {data_dir.name}", leave=False):
                subject_id = row['participant_id']
                subject_dir = data_dir / subject_id / "eeg"
                
                if not subject_dir.exists():
                    continue
                
                eeg_files = list(subject_dir.glob("*contrastChangeDetection*.bdf"))
                if not eeg_files:
                    continue
                
                for eeg_file in eeg_files:
                    try:
                        raw = mne.io.read_raw_bdf(eeg_file, preload=True, verbose=False)
                        
                        if raw.info['sfreq'] != 100:
                            raw.resample(100, verbose=False)
                        
                        data = raw.get_data()
                        
                        if data.shape[0] != 129:
                            continue
                        
                        data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
                        
                        events_file = eeg_file.with_name(eeg_file.name.replace('_eeg.bdf', '_events.tsv'))
                        if not events_file.exists():
                            continue
                        
                        events_df = pd.read_csv(events_file, sep='\t')
                        
                        for _, event_row in events_df.iterrows():
                            if 'response_time' in events_df.columns:
                                rt = event_row.get('response_time', np.nan)
                            elif 'rt' in events_df.columns:
                                rt = event_row.get('rt', np.nan)
                            else:
                                continue
                            
                            if pd.isna(rt) or rt <= 0:
                                continue
                            
                            onset = event_row.get('onset', np.nan)
                            if pd.isna(onset):
                                continue
                            
                            start_sample = int((onset + 0.5) * 100)
                            end_sample = start_sample + 200
                            
                            if end_sample > data.shape[1]:
                                continue
                            
                            segment = data[:, start_sample:end_sample]
                            
                            self.segments.append(segment)
                            self.response_times.append(rt)
                            self.subject_ids.append(subject_id)
                    
                    except:
                        continue
        
        self.segments = np.array(self.segments, dtype=np.float32)
        self.response_times = np.array(self.response_times, dtype=np.float32)
        self.subject_ids = np.array(self.subject_ids)
        
        print(f"\n  Loaded {len(self)} windows with response times")
        if len(self) > 0:
            print(f"  RT range: {self.response_times.min():.3f} - {self.response_times.max():.3f} seconds")
            print(f"  Unique subjects: {len(np.unique(self.subject_ids))}")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.segments[idx])
        y = torch.FloatTensor([self.response_times[idx]])
        return X, y


if __name__ == '__main__':
    print("="*70)
    print("Testing HBN Data Loading")
    print("="*70)
    
    # Test with small subset
    dataset = ResponseTimeDataset(
        data_dirs=['data/ds005506-bdf', 'data/ds005507-bdf'],
        max_subjects=5  # Just 5 subjects for quick test
    )
    
    if len(dataset) == 0:
        print("\nERROR: No data loaded!")
        sys.exit(1)
    
    print("\nTesting data shapes...")
    X, y = dataset[0]
    print(f"  X shape: {X.shape} (expected: [129, 200])")
    print(f"  y shape: {y.shape} (expected: [1])")
    
    print("\nTesting DataLoader...")
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch_X, batch_y = next(iter(loader))
    print(f"  Batch X shape: {batch_X.shape} (expected: [8, 129, 200])")
    print(f"  Batch y shape: {batch_y.shape} (expected: [8, 1])")
    
    print("\nTesting Subject-level split...")
    subject_ids = dataset.subject_ids
    unique_subjects = np.unique(subject_ids)
    subject_id_map = {sid: i for i, sid in enumerate(unique_subjects)}
    subject_groups = np.array([subject_id_map[sid] for sid in subject_ids])
    
    gkf = GroupKFold(n_splits=min(3, len(unique_subjects)))
    splits = list(gkf.split(np.arange(len(dataset)), groups=subject_groups))
    
    print(f"  Created {len(splits)} folds")
    train_idx, val_idx = splits[0]
    print(f"  Fold 1 - Train: {len(train_idx)}, Val: {len(val_idx)}")
    print(f"  Train subjects: {len(np.unique(subject_groups[train_idx]))}")
    print(f"  Val subjects: {len(np.unique(subject_groups[val_idx]))}")
    
    print("\n" + "="*70)
    print("SUCCESS! Data loading integration complete!")
    print("="*70)
