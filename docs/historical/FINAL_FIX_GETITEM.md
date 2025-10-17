# 🐛 FINAL FIX: __getitem__ Method Bug

## Problem Found!

✅ Metadata extraction **WORKED PERFECTLY**:
```
✅ Response times extracted: 21948/21948 non-zero
   Range: [0.010, 2.402]
   Mean: 1.561, Std: 0.404
```

❌ But training showed **NRMSE = 0.0000**!

## Root Cause

The `__getitem__` method was **ignoring** the pre-extracted `self.response_times` list and trying to extract from metadata **at runtime** (which returned 0.0).

### Broken Code (lines 292-299):
```python
def __getitem__(self, idx):
    windows_ds, rel_idx = self._get_dataset_and_index(idx)
    X, y, metadata = windows_ds[rel_idx]
    
    # ... normalization ...
    
    # ❌ WRONG: Tries to extract from metadata at runtime
    if isinstance(metadata, list):
        meta_dict = metadata[0] if len(metadata) > 0 else {}
    else:
        meta_dict = metadata
    
    response_time = meta_dict.get('rt_from_stimulus', 0.0) if isinstance(meta_dict, dict) else 0.0
    # This always returned 0.0!
    
    return torch.FloatTensor(X), torch.FloatTensor([response_time])
```

### Why It Failed

- `metadata` from `windows_ds[rel_idx]` is still just sample indices `[0, 50, 250]`
- The metadata DataFrame from `get_metadata()` is only accessible at the dataset level, not per-window
- We already extracted all response times during `__init__` into `self.response_times`!

## The Fix

Simply use the pre-extracted values:

```python
def __getitem__(self, idx):
    windows_ds, rel_idx = self._get_dataset_and_index(idx)
    X, y, metadata = windows_ds[rel_idx]
    
    # ... normalization ...
    
    # ✅ CORRECT: Use pre-extracted response times from __init__
    response_time = self.response_times[idx] if idx < len(self.response_times) else 0.0
    if np.isnan(response_time):
        response_time = 0.0
    
    return torch.FloatTensor(X), torch.FloatTensor([response_time])
```

## Training Status

**Challenge 1:**
- ✅ Metadata extraction: WORKING
- ✅ __getitem__ method: FIXED
- 🔄 Training: Restarted (v12)
- 📝 Log: `logs/challenge1_training_v12_FINAL_FINAL.log`

## Verification

In ~30 minutes, check for:
- ✅ Train NRMSE: 0.5-2.0 (NOT 0.0000)
- ✅ Val NRMSE: 0.5-2.0 (NOT 0.0000)

## Lessons Learned

1. ✅ Use `add_extras_columns` to inject metadata
2. ✅ Use `get_metadata()` to extract during `__init__`
3. ✅ Store in a list (`self.response_times`)
4. ✅ **Return from the list in `__getitem__`** ← This was the missing piece!

---
**Status:** Training restarted with complete fix
**Expected:** NRMSE > 0 in Epoch 1
