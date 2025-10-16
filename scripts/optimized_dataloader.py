#!/usr/bin/env python3
"""
Optimized Data Loader for EEG 2025 Competition
==============================================
Features:
- Smart device detection (CUDA, ROCm, CPU fallback)
- Pandas optimization with chunking
- Fast MNE data loading with memory mapping
- Progress bars for all operations
"""
import os
import sys
from pathlib import Path
import time
import warnings

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def detect_device(verbose=True):
    """Detect best available device: CUDA, ROCm, or CPU"""
    device_name = "cpu"
    device_info = "CPU (default)"
    
    # Check for CUDA
    if torch.cuda.is_available():
        try:
            # Test if CUDA actually works
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            torch.cuda.empty_cache()
            device_name = "cuda"
            device_info = f"CUDA (GPU: {torch.cuda.get_device_name(0)})"
        except Exception as e:
            if verbose:
                print(f"⚠️  CUDA available but not working: {e}")
            device_name = "cpu"
            device_info = "CPU (CUDA failed)"
    
    # Check for ROCm
    elif hasattr(torch.version, 'hip') and torch.version.hip is not None:
        try:
            # ROCm uses 'cuda' backend in PyTorch
            if torch.cuda.is_available():
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                device_name = "cuda"  # ROCm uses cuda backend
                device_info = f"ROCm (HIP: {torch.version.hip})"
        except Exception as e:
            if verbose:
                print(f"⚠️  ROCm available but not working: {e}")
            device_name = "cpu"
            device_info = "CPU (ROCm failed)"
    
    device = torch.device(device_name)
    
    if verbose:
        print(f"🖥️  Device: {device_info}")
    
    return device, device_info


def load_participants_optimized(data_dir, required_columns=None):
    """Load participants.tsv with pandas optimization"""
    participants_file = Path(data_dir) / "participants.tsv"
    
    print(f"📋 Loading participants from: {participants_file}")
    
    # Use pandas with optimal settings
    df = pd.read_csv(
        participants_file,
        sep='\t',
        engine='c',  # Fast C engine
        low_memory=False  # Better type inference
    )
    
    print(f"   Total participants: {len(df)}")
    
    # Filter for required columns
    if required_columns:
        valid_mask = df[required_columns].notna().all(axis=1)
        df = df[valid_mask]
        print(f"   With required data: {len(df)}")
    
    return df


def load_eeg_fast(eeg_file, target_sr=100, channels=129):
    """Load EEG with MNE using memory-efficient methods"""
    import mne
    
    # Load with memory mapping for large files
    raw = mne.io.read_raw_eeglab(
        eeg_file,
        preload=False,  # Don't load to memory yet
        verbose=False
    )
    
    # Check channels
    if len(raw.ch_names) != channels:
        return None, f"Wrong channel count: {len(raw.ch_names)} != {channels}"
    
    # Resample if needed (in-place, efficient)
    if raw.info['sfreq'] != target_sr:
        raw.resample(target_sr, verbose=False)
    
    # Now load to memory (after resampling reduces size)
    raw.load_data(verbose=False)
    
    # Get data as numpy array
    data = raw.get_data()
    
    # Standardize per channel
    data = (data - data.mean(axis=1, keepdims=True)) / (data.std(axis=1, keepdims=True) + 1e-8)
    
    return data, None


def create_segments_vectorized(data, segment_length):
    """Create segments using vectorized numpy operations (much faster)"""
    n_channels, n_samples = data.shape
    n_segments = n_samples // segment_length
    
    if n_segments == 0:
        return []
    
    # Trim to multiple of segment_length
    trimmed_length = n_segments * segment_length
    data_trimmed = data[:, :trimmed_length]
    
    # Reshape to create segments (vectorized, no loops!)
    segments = data_trimmed.reshape(n_channels, n_segments, segment_length)
    segments = np.transpose(segments, (1, 0, 2))  # (n_segments, n_channels, segment_length)
    
    return segments


class FastEEGDataset(torch.utils.data.Dataset):
    """Optimized EEG dataset with caching and vectorization"""
    
    def __init__(self, data_dir, target_column, segment_length=200, sampling_rate=100, 
                 cache_segments=True, device='cpu'):
        """
        Args:
            data_dir: Path to HBN data
            target_column: Column name for target (e.g., 'externalizing', 'response_time')
            segment_length: Samples per segment (200 = 2sec @ 100Hz)
            sampling_rate: Target sampling rate
            cache_segments: Cache segments in memory (faster but uses more RAM)
            device: Device for tensors
        """
        self.data_dir = Path(data_dir)
        self.segment_length = segment_length
        self.target_sr = sampling_rate
        self.target_column = target_column
        self.cache_segments = cache_segments
        self.device = device
        
        # Load participants with optimization
        print(f"\n📊 Creating dataset for: {target_column}")
        self.df = load_participants_optimized(data_dir, required_columns=[target_column])
        
        # Pre-allocate lists for speed
        self.segments = []
        self.targets = []
        
        # Load all data with progress bar
        print(f"🔄 Loading EEG data...")
        for _, row in tqdm(self.df.iterrows(), total=len(self.df), desc="Loading subjects"):
            subject_id = row['participant_id']
            subject_dir = self.data_dir / subject_id / "eeg"
            
            if not subject_dir.exists():
                continue
            
            # Find RestingState EEG (fastest to load)
            eeg_files = list(subject_dir.glob("*RestingState*.set"))
            if not eeg_files:
                continue
            
            # Load EEG with fast method
            data, error = load_eeg_fast(eeg_files[0], self.target_sr, 129)
            if error:
                continue
            
            # Create segments (vectorized - very fast!)
            segments = create_segments_vectorized(data, segment_length)
            
            if len(segments) == 0:
                continue
            
            # Get target value
            target = float(row[target_column])
            
            # Store segments
            for segment in segments:
                if cache_segments:
                    # Cache as torch tensor for speed
                    self.segments.append(torch.from_numpy(segment).float())
                else:
                    # Store as numpy (less memory)
                    self.segments.append(segment.astype(np.float32))
                self.targets.append(target)
        
        print(f"   Total segments: {len(self.segments)}")
        print(f"   Memory cached: {cache_segments}")
        
        # Compute statistics
        self.targets_array = np.array(self.targets, dtype=np.float32)
        self.target_mean = self.targets_array.mean()
        self.target_std = self.targets_array.std()
        print(f"   Target stats: mean={self.target_mean:.3f}, std={self.target_std:.3f}")
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        if self.cache_segments:
            segment = self.segments[idx]  # Already a tensor
        else:
            segment = torch.from_numpy(self.segments[idx]).float()
        
        target = torch.FloatTensor([self.targets[idx]])
        return segment, target


def test_optimized_loader():
    """Test the optimized data loader"""
    print("="*80)
    print("🚀 Testing Optimized Data Loader")
    print("="*80)
    
    # Detect device
    device, device_info = detect_device()
    
    # Test loading
    data_dir = Path("data/raw/hbn")
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    # Create dataset
    start_time = time.time()
    dataset = FastEEGDataset(
        data_dir=data_dir,
        target_column='externalizing',
        segment_length=200,
        sampling_rate=100,
        cache_segments=True,
        device=device
    )
    load_time = time.time() - start_time
    
    print(f"\n⏱️  Loading time: {load_time:.1f} seconds")
    print(f"📦 Dataset size: {len(dataset)} segments")
    
    # Test data loading speed
    print("\n🧪 Testing batch loading speed...")
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=0  # Test single-threaded first
    )
    
    start_time = time.time()
    for i, (X, y) in enumerate(dataloader):
        if i >= 10:  # Test first 10 batches
            break
    batch_time = time.time() - start_time
    
    print(f"   10 batches loaded in: {batch_time:.2f} seconds")
    print(f"   Speed: {10/batch_time:.1f} batches/sec")
    
    print("\n✅ Optimized loader test complete!")


if __name__ == "__main__":
    test_optimized_loader()
