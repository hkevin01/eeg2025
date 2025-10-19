"""
Memory-efficient dataset that loads windows from HDF5 files on-demand.

This prevents loading all data into RAM, enabling training on large datasets
without memory overflow.
"""
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

class HDF5Dataset(Dataset):
    """
    PyTorch Dataset that loads EEG windows from HDF5 files on-demand.

    Features:
    - Memory-mapped file access (doesn't load all data into RAM)
    - Multi-file support (combine R1, R2, R3, R4)
    - Lazy loading (only loads requested windows)
    - Thread-safe (works with DataLoader num_workers > 0)

    Example:
        >>> dataset = HDF5Dataset([
        ...     'data/processed/R1_challenge1_windows.h5',
        ...     'data/processed/R2_challenge1_windows.h5'
        ... ])
        >>> loader = DataLoader(dataset, batch_size=32, num_workers=4)
        >>> for eeg, labels in loader:
        ...     # Training loop
        ...     pass
    """

    def __init__(self, hdf5_files, transform=None):
        """
        Args:
            hdf5_files: List of paths to HDF5 files
            transform: Optional transform to apply to windows
        """
        self.hdf5_files = [Path(f) for f in hdf5_files]
        self.transform = transform

        # Verify files exist
        for f in self.hdf5_files:
            if not f.exists():
                raise FileNotFoundError(f"HDF5 file not found: {f}")

        # Open files in read mode (memory-mapped)
        self.files = [h5py.File(f, 'r') for f in self.hdf5_files]

        # Calculate cumulative lengths for multi-file indexing
        self.lengths = [len(f['eeg']) for f in self.files]
        self.cumulative = np.cumsum([0] + self.lengths)
        self.total_length = self.cumulative[-1]

        print(f"HDF5Dataset initialized:")
        print(f"  Files: {len(self.hdf5_files)}")
        for i, (f, length) in enumerate(zip(self.hdf5_files, self.lengths)):
            print(f"    {i+1}. {f.name}: {length} windows")
        print(f"  Total windows: {self.total_length}")

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        """
        Load a single window (memory-efficient).

        Args:
            idx: Global index across all files

        Returns:
            (eeg, label): Tuple of torch tensors
        """
        if idx < 0 or idx >= self.total_length:
            raise IndexError(f"Index {idx} out of range [0, {self.total_length})")

        # Find which file contains this index
        file_idx = np.searchsorted(self.cumulative, idx, side='right') - 1
        local_idx = idx - self.cumulative[file_idx]

        # Load single window from HDF5 (memory-mapped, not copied to RAM)
        eeg = self.files[file_idx]['eeg'][local_idx]
        label = self.files[file_idx]['labels'][local_idx]

        # Convert to PyTorch tensors
        eeg = torch.from_numpy(eeg).float()
        label = torch.tensor([label], dtype=torch.float32)

        if self.transform:
            eeg = self.transform(eeg)

        return eeg, label

    def get_metadata(self):
        """Get metadata from all files"""
        metadata = {}
        for i, f in enumerate(self.files):
            file_meta = dict(f.attrs)
            metadata[f"file_{i}"] = file_meta
        return metadata

    def close(self):
        """Close all HDF5 files"""
        for f in self.files:
            f.close()

    def __del__(self):
        """Cleanup when dataset is deleted"""
        try:
            self.close()
        except:
            pass

def test_hdf5_dataset():
    """Test the HDF5Dataset class"""
    print("Testing HDF5Dataset...")

    # Look for cached files
    cached_dir = Path("data/processed")
    cached_files = sorted(cached_dir.glob("*_challenge1_windows.h5"))

    if not cached_files:
        print("❌ No cached files found. Run cache_challenge1_windows.py first.")
        return

    print(f"Found {len(cached_files)} cached files")

    # Create dataset
    dataset = HDF5Dataset(cached_files)

    # Test loading a few samples
    print("\nTesting sample loading...")
    for i in range(min(3, len(dataset))):
        eeg, label = dataset[i]
        print(f"  Sample {i}: eeg shape={eeg.shape}, label={label.item():.4f}")

    # Test DataLoader
    print("\nTesting with DataLoader...")
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=8, num_workers=2, shuffle=True)

    for i, (eeg_batch, label_batch) in enumerate(loader):
        print(f"  Batch {i}: eeg={eeg_batch.shape}, labels={label_batch.shape}")
        if i >= 2:  # Just test a few batches
            break

    print("\n✅ HDF5Dataset test passed!")
    dataset.close()

if __name__ == "__main__":
    test_hdf5_dataset()
