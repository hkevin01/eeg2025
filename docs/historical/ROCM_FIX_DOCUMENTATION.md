# PyTorch ROCm Compatibility Fix

## Problem

When using PyTorch with ROCm (AMD GPU support), there's a known compatibility issue with `torch.arange()` when called with certain parameters during MNE/braindecode data loading operations.

### Error Message
```
arange: cannot compute length
```

This error occurs during `create_windows_from_events()` in braindecode, which internally uses PyTorch operations that conflict with ROCm's implementation of `arange`.

## Root Cause

The issue stems from how PyTorch ROCm handles floating-point step values in `torch.arange()`. Some internal operations in MNE-Python's event handling code trigger this edge case.

## Solution

We implemented a temporary monkey-patch that forces all `torch.arange()` calls to use CPU during data loading, then restores the original function for GPU training.

### Implementation

```python
# Monkey-patch torch.arange to force CPU during data loading
original_arange = torch.arange

def cpu_arange(*args, **kwargs):
    # Force all arange calls to use CPU during data loading
    kwargs.pop('device', None)  # Remove any device argument
    return original_arange(*args, device='cpu', **kwargs)

torch.arange = cpu_arange
try:
    windows_ds = create_windows_from_events(...)
finally:
    # Restore original arange
    torch.arange = original_arange
```

## Impact

- **Data Loading**: Uses CPU (unavoidable due to MNE/braindecode limitations)
- **Training**: Uses GPU (4-5x speedup maintained)
- **Compatibility**: Works with PyTorch 2.5.1+rocm6.2
- **Competition**: Safe - doesn't affect submission code

## Files Modified

1. `scripts/train_challenge1_robust_gpu.py` - Lines ~176-200
2. `scripts/train_challenge2_robust_gpu.py` - Lines ~186-210

## Verification

Run training and check logs for successful R1 data loading:
```bash
bash restart_training.sh
# Wait ~60 seconds
tail -f logs/train_c1_robust_final.log
# Should see "Reading 0 ..." messages without "arange" errors
```

## Alternative Solutions Considered

1. **Disable CUDA environment variable** - Too invasive, affects entire PyTorch session
2. **Upgrade PyTorch** - ROCm 6.3 not yet stable
3. **CPU-only training** - Loses 4-5x speedup
4. **Monkey-patch (chosen)** - Minimal, isolated, easily reversible

## Future

This workaround should be removed when:
- PyTorch ROCm fixes the arange bug (likely 2.6.0+)
- MNE-Python updates to avoid triggering the issue
- Braindecode adds ROCm-specific handling

## Testing

Date: 2025-10-16
PyTorch: 2.5.1+rocm6.2
ROCm: 6.2.2
GPU: AMD Radeon RX 5600 XT
Status: ✅ Working - 220+ files loaded successfully
