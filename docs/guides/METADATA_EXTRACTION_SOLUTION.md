# Metadata Extraction Solution - Challenge 1

## Problem
Challenge 1 training showing **NRMSE = 0.0000** because response times are all zeros.

**Root Cause:** Windows metadata returns `[0, 50, 250]` (sample indices) instead of actual metadata dict with `rt_from_stimulus`.

## Official Starter Kit Approach

According to `starter_kit_integration/challenge_1.py`, the correct workflow is:

### 1. Annotate Trials
```python
from eegdash.hbn.windows import annotate_trials_with_target

preprocessors = [
    Preprocessor(
        annotate_trials_with_target,
        apply_on_array=False,
        target_field="rt_from_stimulus",  # Adds RT to annotation extras
        epoch_length=EPOCH_LEN_S,
        require_stimulus=True,
        require_response=True,
    ),
]
preprocess(dataset, preprocessors)
```

### 2. Create Windows
```python
from braindecode.preprocessing import create_windows_from_events

windows = create_windows_from_events(
    dataset,
    trial_start_offset_samples=0,
    trial_stop_offset_samples=0,
    window_size_samples=200,  # 2 seconds @ 100Hz
    window_stride_samples=200,
    preload=True,
)
```

### 3. **KEY STEP**: Inject Metadata into Windows
```python
from eegdash.hbn.windows import add_extras_columns

windows = add_extras_columns(
    windows,           # Windowed dataset
    dataset,           # Original dataset with annotations
    desc=ANCHOR,       # Annotation description (e.g., "contrast_trial_start")
    keys=("target", "rt_from_stimulus", "rt_from_trialstart", 
          "stimulus_onset", "response_onset", "correct", "response_type")
)
```

**This step copies metadata from annotation `extras` into the windows metadata DataFrame!**

### 4. Extract Metadata
```python
# Get metadata as DataFrame
meta_information = windows.get_metadata()

# Now meta_information has columns:
# - subject
# - rt_from_stimulus  ← Response time!
# - rt_from_trialstart
# - target
# - etc.
```

### 5. Use in DataLoader
```python
# The windows dataset now returns (X, y, metadata)
# where y is the target field specified in annotate_trials_with_target

for X, y, meta in dataloader:
    # y already contains rt_from_stimulus!
    # No need to extract from metadata in __getitem__
    pass
```

## What We're Missing

Our current approach in `train_challenge1_multi_release.py`:

❌ **Wrong:**
```python
# After create_windows_from_events:
windows_dataset[0]  # Returns (X, y, [0, 50, 250])
                    # Metadata is just sample indices!
```

✅ **Correct:**
```python
# After create_windows_from_events:
windows = create_windows_from_events(...)

# ADD THIS STEP:
windows = add_extras_columns(
    windows, 
    dataset,  # Original preprocessed dataset
    desc="contrast_trial_start",
    keys=("rt_from_stimulus",)
)

# Now:
metadata = windows.get_metadata()
# metadata is a DataFrame with rt_from_stimulus column!
```

## Solution Options

### Option 1: Use add_extras_columns (Recommended - Official Approach)
Modify `train_challenge1_multi_release.py` to:
1. After `create_windows_from_events`, call `add_extras_columns`
2. Extract metadata DataFrame with `windows.get_metadata()`
3. Store response times during dataset initialization
4. Return them in `__getitem__`

### Option 2: Direct Extraction from Raw Dataset
Keep our current approach but fix the extraction:
- Access the original dataset (not the windows)
- Map window index → trial index → dataset → get RT from raw annotations
- More complex, not the official way

## Recommendation

**Use Option 1 (add_extras_columns)** because:
- ✅ It's the official starter kit approach
- ✅ Simpler and cleaner
- ✅ Guarantees compatibility with evaluation
- ✅ Properly tested by competition organizers

## Next Steps

1. ✅ Stop current training (NRMSE = 0.0)
2. Modify `MultiReleaseDataset.__init__` to use `add_extras_columns`
3. Extract metadata with `get_metadata()` after windowing
4. Store response times in list during initialization
5. Restart training and verify NRMSE > 0

## References

- `starter_kit_integration/challenge_1.py` lines 153-164
- `eegdash.hbn.windows.add_extras_columns` documentation
- Competition guidelines emphasize using official starter kit approaches

---
**Status:** Challenge 1 training stopped. Ready to implement fix.
**ETA:** ~30 minutes to fix + 3 hours to retrain
