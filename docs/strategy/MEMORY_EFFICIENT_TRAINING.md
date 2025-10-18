# Memory-Efficient Training Strategy

## Problem
Loading R1-R4 datasets (719 subjects, thousands of trials) causes:
- Memory overflow (>80% RAM usage)
- VS Code crashes
- System instability

## Solution: Lazy Loading + Memory Mapping

### Option 1: PyTorch IterableDataset (BEST for our case)
Instead of loading all data into RAM, load windows on-demand:

```python
class LazyEEGDataset(IterableDataset):
    """Load EEG windows on-demand, not all at once"""
    def __init__(self, releases, ...):
        # Store PATHS to datasets, not the data itself
        self.dataset_paths = []
        self.window_indices = []
    
    def __iter__(self):
        # Load windows ONE AT A TIME during training
        for path, idx in zip(self.dataset_paths, self.window_indices):
            data = load_single_window(path, idx)  # Load from disk
            yield data
```

**Advantages:**
- RAM usage: ~2GB instead of ~40GB
- Can train on unlimited data
- Standard PyTorch approach

### Option 2: HDF5 Memory Mapping
Store preprocessed windows in HDF5 format:

```python
import h5py

# Preprocess once, save to HDF5
with h5py.File('windows.h5', 'w') as f:
    f.create_dataset('eeg', data=all_windows, compression='gzip')
    f.create_dataset('labels', data=all_labels)

# During training: memory-mapped access
with h5py.File('windows.h5', 'r') as f:
    for i in range(len(f['eeg'])):
        window = f['eeg'][i]  # Loaded from disk, not RAM
```

**Advantages:**
- Fast random access
- Compressed storage
- Used by large-scale ML (ImageNet, etc.)

### Option 3: LMDB (Lightning Memory-Mapped Database)
Used by Caffe, PyTorch ImageNet loaders:

```python
import lmdb

# Store windows in LMDB
env = lmdb.open('eeg_data.lmdb', map_size=int(1e12))
with env.begin(write=True) as txn:
    for i, window in enumerate(windows):
        txn.put(str(i).encode(), pickle.dumps(window))

# Load during training
with env.begin() as txn:
    window = pickle.loads(txn.get(str(i).encode()))
```

### Option 4: Zarr (Cloud-optimized arrays)
Modern alternative to HDF5:

```python
import zarr

# Save
z = zarr.open('windows.zarr', mode='w', shape=(n_windows, channels, time), 
              chunks=(100, channels, time), dtype='float32')
z[:] = all_windows

# Load lazily
z = zarr.open('windows.zarr', mode='r')
window = z[i]  # Memory-mapped
```

## Recommended Approach for EEG Challenge

**Two-stage process:**

### Stage 1: Preprocess & Cache (Run Once)
```python
# scripts/preprocessing/cache_windows.py
for release in ['R1', 'R2', 'R3', 'R4']:
    dataset = load_and_preprocess(release)
    windows = create_windows(dataset)
    save_to_hdf5(f'data/processed/{release}_windows.h5', windows)
```

**Output:**
- `data/processed/R1_windows.h5` (2GB)
- `data/processed/R2_windows.h5` (3GB)
- `data/processed/R3_windows.h5` (3GB)
- `data/processed/R4_windows.h5` (4GB)
- **Total:** 12GB on disk, minimal RAM

### Stage 2: Memory-Efficient Training
```python
class HDF5Dataset(Dataset):
    def __init__(self, hdf5_files):
        self.files = [h5py.File(f, 'r') for f in hdf5_files]
        self.lengths = [len(f['eeg']) for f in self.files]
        self.cumulative = np.cumsum([0] + self.lengths)
    
    def __getitem__(self, idx):
        # Find which file + local index
        file_idx = np.searchsorted(self.cumulative, idx, side='right') - 1
        local_idx = idx - self.cumulative[file_idx]
        
        # Load single window (memory-mapped, not loaded into RAM)
        eeg = self.files[file_idx]['eeg'][local_idx]
        label = self.files[file_idx]['labels'][local_idx]
        return torch.from_numpy(eeg), torch.from_numpy(label)
    
    def __len__(self):
        return self.cumulative[-1]

# Training uses minimal RAM
dataset = HDF5Dataset(['R1_windows.h5', 'R2_windows.h5', 'R3_windows.h5', 'R4_windows.h5'])
loader = DataLoader(dataset, batch_size=32, num_workers=4)  # Workers load in parallel
```

## Implementation Plan

```markdown
## Memory-Efficient Training - TODO

- [ ] Create preprocessing script to cache windows
  - [ ] Load releases one at a time
  - [ ] Create windows with proper metadata
  - [ ] Save to HDF5 format
  - [ ] Verify file integrity

- [ ] Create HDF5Dataset class
  - [ ] Memory-mapped file access
  - [ ] Multi-file support (R1-R4)
  - [ ] Proper indexing across files
  - [ ] Test with small batch

- [ ] Update training scripts
  - [ ] Challenge 1: Use HDF5Dataset
  - [ ] Challenge 2: Use HDF5Dataset
  - [ ] Monitor memory during training
  - [ ] Verify no crashes

- [ ] Optimize for speed
  - [ ] Add caching for frequently accessed windows
  - [ ] Tune num_workers for DataLoader
  - [ ] Profile memory usage
```

## Expected Improvements

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| RAM Usage | ~40GB | ~4GB | 90% reduction |
| Crash Risk | High | None | Eliminated |
| Training Speed | N/A | Same | No penalty |
| Max Data Size | Limited | Unlimited | Scalable |

## References

- PyTorch IterableDataset: https://pytorch.org/docs/stable/data.html#torch.utils.data.IterableDataset
- HDF5 for PyTorch: https://discuss.pytorch.org/t/hdf5-a-pytorch-alternative/25
- LMDB for ML: https://github.com/Lyken17/Efficient-PyTorch
- Zarr documentation: https://zarr.readthedocs.io/
