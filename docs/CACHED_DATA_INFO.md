# 📦 Cached Data Information

## Challenge 1 Cached Data (READY ✅)

### Location
```
data/cached/challenge1_*.h5
```

### Files
| File | Size | Windows | Status |
|------|------|---------|--------|
| `challenge1_R1_windows.h5` | 660 MB | 7,276 | ✅ Ready |
| `challenge1_R2_windows.h5` | 682 MB | 7,524 | ✅ Ready |
| `challenge1_R3_windows.h5` | 853 MB | 9,551 | ✅ Ready |
| `challenge1_R4_windows.h5` | 1.5 GB | 16,554 | ✅ Ready |
| **TOTAL** | **3.6 GB** | **40,905** | **✅ Ready** |

### H5 Structure
```python
h5_file = h5py.File('challenge1_R1_windows.h5', 'r')

# Keys (NOT 'segments' or 'response_times'!)
h5_file.keys()  # ['eeg', 'labels', 'neuro_features']

# Data shapes
eeg = h5_file['eeg'][:]        # (n_windows, 129, 200) float32
labels = h5_file['labels'][:]  # (n_windows,) float64

# Attributes
attrs = dict(h5_file.attrs)
# {'n_channels': 129, 'n_timepoints': 200, 'sfreq': 100, 
#  'release': 'R1', 'n_windows': 7276}
```

### Usage Example
```python
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class CachedResponseTimeDataset(Dataset):
    def __init__(self, h5_files):
        self.segments = []
        self.response_times = []
        
        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                # CRITICAL: Use 'eeg' and 'labels', not 'segments' or 'response_times'
                segments = f['eeg'][:]
                rts = f['labels'][:]
                
                # Filter valid RTs (0.1 to 5.0 seconds)
                valid_mask = (rts >= 0.1) & (rts <= 5.0)
                self.segments.append(segments[valid_mask])
                self.response_times.append(rts[valid_mask])
        
        self.segments = np.concatenate(self.segments, axis=0)
        self.response_times = np.concatenate(self.response_times, axis=0)
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx):
        segment = torch.from_numpy(self.segments[idx]).float()
        rt = torch.tensor(self.response_times[idx], dtype=torch.float32)
        return segment, rt

# Load all R1-R4
h5_files = [
    'data/cached/challenge1_R1_windows.h5',
    'data/cached/challenge1_R2_windows.h5',
    'data/cached/challenge1_R3_windows.h5',
    'data/cached/challenge1_R4_windows.h5'
]

dataset = CachedResponseTimeDataset(h5_files)
loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Training loop
for X, y in loader:
    # X: (batch, 129, 200)
    # y: (batch,)
    pass
```

### Performance Benefits
- **Raw BDF loading:** ~2-4 hours per dataset (slow file I/O + MNE processing)
- **Cached H5 loading:** ~2 minutes for all 40K windows (500x faster!)
- **Training speedup:** Can iterate quickly, run multiple experiments

### Training Script
```bash
# Use the cached data training script
python3 training/train_c1_cached.py

# Monitor progress
./monitor_training.sh

# View logs
tail -f logs/c1_cached_training.log
```

---

## Challenge 2 Cached Data (PARTIAL)

### Known Files
- `data/cached/challenge2_R1_windows.h5` - 11 GB (61,889 windows) ✅
- `data/cached/challenge2_R2_windows.h5` - 12 GB (62,000+ windows) ✅
- R3, R4, R5 status unknown

### Structure
Similar to Challenge 1 but for externalizing factor prediction:
- Keys: `eeg`, `labels` (p_factor values)
- Shape: `eeg` = (n_windows, 129, 200)

---

## Key Lessons Learned (Oct 26, 2025)

### ❌ Common Mistakes
1. **Wrong H5 keys:** Using `segments`/`response_times` instead of `eeg`/`labels`
2. **Loading raw BDF:** Extremely slow, caused training to hang for hours
3. **Not checking cache:** Wasted time re-processing data that was already cached

### ✅ Best Practices
1. **Always check for cached data first:** Look in `data/cached/` before raw loading
2. **Use correct H5 keys:** `eeg` and `labels` (not `segments`/`response_times`)
3. **Verify structure:** Check with `h5py.File(file, 'r').keys()` before assuming
4. **Fast iteration:** Cached data enables rapid experimentation

### 🔧 Debugging Cached Data Issues
```python
import h5py

# Check what keys exist
with h5py.File('data/cached/challenge1_R1_windows.h5', 'r') as f:
    print("Keys:", list(f.keys()))
    print("Attributes:", dict(f.attrs))
    for key in f.keys():
        print(f"{key}: shape={f[key].shape}, dtype={f[key].dtype}")
```

---

**Last Updated:** October 26, 2025  
**Status:** Challenge 1 cached data fully documented and in use for training

