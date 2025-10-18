# Stimulus-Aligned Training Strategy

## 🎯 Critical Issue Identified

**Problem:** Current training uses `"contrast_trial_start"` as window anchor
**Solution:** Must use `"stimulus_anchor"` for proper stimulus alignment

## Why Stimulus Alignment Matters

Response time (RT) is measured **from stimulus onset**, not trial start:
- Trial start = Beginning of entire trial sequence
- Stimulus onset = When the actual visual stimulus appears
- Response time = Time from **stimulus onset** to button press

**If windows are not stimulus-aligned:**
- Model sees variable delays before stimulus
- RT predictions include irrelevant pre-stimulus activity
- Poor generalization to test data

## Current vs Correct Approach

### ❌ Current (INCORRECT):
```python
ANCHOR = "contrast_trial_start"  # Wrong! This is trial start, not stimulus
windows_dataset = create_windows_from_events(
    dataset,
    mapping={ANCHOR: 0},
    trial_start_offset_samples=int(0.5 * 100),  # 0.5s after TRIAL start
    # ... but we need 0.5s after STIMULUS!
)
```

### ✅ Correct (STIMULUS-ALIGNED):
```python
ANCHOR = "stimulus_anchor"  # Correct! This is added by add_aux_anchors
windows_dataset = create_windows_from_events(
    dataset,
    mapping={ANCHOR: 0},
    trial_start_offset_samples=int(0.5 * 100),  # 0.5s after STIMULUS
    trial_stop_offset_samples=int(2.5 * 100),   # 2.5s after STIMULUS
    window_size_samples=int(2.0 * 100),         # 2.0 second window
    preload=True,
)
```

## What `add_aux_anchors` Does

The preprocessor `add_aux_anchors` creates stimulus-aligned anchors:
```python
Preprocessor(add_aux_anchors, apply_on_array=False)
```

This adds:
- `"stimulus_anchor"` - Marks exact stimulus onset time
- `"response_anchor"` - Marks exact response time

These are the events we should lock our windows to!

## Window Configuration

**For Response Time Prediction:**
```
Stimulus onset (t=0)
    |
    |--- 0.5s buffer --->| Start window
    |                    |
    |                    |--- 2.0s window --->| End window
    |                                         |
    |--- Response typically occurs here ----->
    |
    |--- 2.5s total from stimulus ----------->
```

**Key parameters:**
- `trial_start_offset_samples = 50` (0.5s * 100Hz = 50 samples)
  - Start window 0.5s AFTER stimulus
  - Captures neural activity AFTER stimulus presentation
  
- `window_size_samples = 200` (2.0s * 100Hz = 200 samples)
  - 2 second window of EEG data
  - Enough to capture most response times
  
- `trial_stop_offset_samples = 250` (2.5s * 100Hz = 250 samples)
  - End window 2.5s after stimulus
  - trial_stop = trial_start + window_size

## Implementation Checklist

- [x] Use `add_aux_anchors` preprocessor (already doing this ✅)
- [ ] Change anchor from `"contrast_trial_start"` → `"stimulus_anchor"`
- [ ] Verify windows are stimulus-aligned in logs
- [ ] Check metadata extraction still works with new anchor
- [ ] Update `add_extras_columns` to use `desc="stimulus_anchor"`

## Expected Improvements

1. **Better alignment:** Windows now correctly centered on stimulus
2. **Less noise:** No pre-stimulus activity in windows
3. **Better generalization:** Model learns stimulus→response relationship
4. **Correct RT measurement:** RT is relative to window start (stimulus)

## Code Changes Required

### File: `scripts/training/challenge1/train_challenge1_multi_release.py`

**Line ~186:** Change anchor definition
```python
# OLD:
ANCHOR = "contrast_trial_start"  # Event marker for trials

# NEW:
ANCHOR = "stimulus_anchor"  # Stimulus-aligned anchor from add_aux_anchors
```

**Line ~218:** Update metadata extraction descriptor
```python
# OLD:
windows_dataset = add_extras_columns(
    windows_dataset,
    dataset,
    desc="contrast_trial_start",  # Wrong descriptor
    keys=("rt_from_stimulus", "target", ...)
)

# NEW:
windows_dataset = add_extras_columns(
    windows_dataset,
    dataset,
    desc="stimulus_anchor",  # Correct descriptor
    keys=("rt_from_stimulus", "target", ...)
)
```

## Testing the Fix

After making changes, verify in logs:
```
Creating windows from trials...
Using anchor: stimulus_anchor (stimulus-aligned) ✅
Windows created: XXX
```

Check first few windows have valid RT values:
```python
metadata_df = windows_dataset.get_metadata()
print(metadata_df[['rt_from_stimulus', 'stimulus_onset', 'response_onset']].head())
```

## References

- eegdash documentation: `add_aux_anchors` creates stimulus/response anchors
- braindecode: `create_windows_from_events` locks windows to events
- Challenge 1 objective: Predict RT **from stimulus onset**

## Summary

**Critical fix:** Change window anchor from `"contrast_trial_start"` → `"stimulus_anchor"`

This ensures:
1. Windows start at stimulus onset (t=0)
2. RT measurements are relative to window start
3. Model learns correct stimulus→response relationship
4. Better generalization to test data

**Implementation time:** ~5 minutes (2 line changes)
**Expected improvement:** 5-15% better NRMSE due to proper alignment
