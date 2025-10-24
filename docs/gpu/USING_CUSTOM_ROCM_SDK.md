# Using Custom ROCm SDK for gfx1010 GPU Training

**Date:** October 23, 2025  
**GPU:** AMD Radeon RX 5600 XT (gfx1010)  
**SDK Location:** `/opt/rocm_sdk_612`

---

## Problem Summary

Standard PyTorch ROCm builds cause `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` on consumer AMD GPUs (gfx1010) because official ROCm binaries only include kernels for server GPUs.

## Solution

A custom ROCm SDK was built using [ROCm SDK Builder](https://github.com/lamikr/rocm_sdk_builder) that includes full gfx1010 support.

**SDK Details:**
- Location: `/opt/rocm_sdk_612`
- PyTorch: 2.4.1 (custom build with gfx1010 kernels)
- Python: 3.11
- ROCm: 6.1.2

---

## Current Status

### ✅ What Works
- SDK is successfully built at `/opt/rocm_sdk_612`
- Contains PyTorch 2.4.1 with gfx1010 support
- Basic tensor operations on GPU work

### ⚠️ Current Issue
- Python environment compatibility issue
  - SDK built with Python 3.11
  - Has conflicting `typing` module
  - Need to use SDK's Python directly or create isolated environment

---

## How to Use the SDK

### Option 1: Use SDK Python Directly (Recommended)

The SDK has its own Python 3.11 installation with PyTorch pre-installed:

```bash
# SDK Python location
/opt/rocm_sdk_612/bin/python3

# Example: Test PyTorch import
/opt/rocm_sdk_612/bin/python3 -c "import torch; print(torch.__version__)"

# Run training with SDK Python
/opt/rocm_sdk_612/bin/python3 scripts/training/train_challenge2_fast.py
```

**Missing Dependencies:**
The SDK Python needs additional packages installed:
```bash
# Install braindecode (if pip works in SDK)
/opt/rocm_sdk_612/bin/python3 -m ensurepip
/opt/rocm_sdk_612/bin/pip3 install braindecode h5py numpy pandas

# Or copy from system Python site-packages
cp -r ~/.local/lib/python3.*/site-packages/braindecode \
      /opt/rocm_sdk_612/lib/python3.11/site-packages/
```

### Option 2: Set Environment Variables

For using SDK libraries with system Python (has compatibility issues currently):

```bash
export PYTHONPATH="/opt/rocm_sdk_612/lib/python3.11/site-packages:$PYTHONPATH"
export LD_LIBRARY_PATH="/opt/rocm_sdk_612/lib:$LD_LIBRARY_PATH"
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HIP_VISIBLE_DEVICES=0

python3.11 your_script.py
```

**Note:** This approach currently has Python `typing` module conflicts.

---

## Testing GPU with SDK

### Quick Test

```bash
# Test if GPU works with SDK PyTorch
/opt/rocm_sdk_612/bin/python3 << 'PYEOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.randn(100, 100, device="cuda:0")
    y = torch.randn(100, 100, device="cuda:0")
    z = torch.mm(x, y)
    print("✅ GPU tensor operations work!")
PYEOF
```

### Full EEGNeX Test

Once braindecode is installed in SDK:
```bash
/opt/rocm_sdk_612/bin/python3 tests/test_custom_sdk_eegnex.py
```

---

## Next Steps

### To Enable GPU Training

1. **Install Missing Packages in SDK Python:**
   ```bash
   # Ensure pip works
   /opt/rocm_sdk_612/bin/python3 -m ensurepip --upgrade
   
   # Install required packages
   /opt/rocm_sdk_612/bin/pip3 install \
       braindecode \
       h5py \
       numpy \
       pandas \
       mne \
       scikit-learn
   ```

2. **Update Training Script:**
   Modify `scripts/training/train_challenge2_fast.py` to detect and use SDK:
   ```python
   def select_device():
       sdk_path = "/opt/rocm_sdk_612"
       if os.path.exists(sdk_path):
           # SDK available - GPU should work
           if torch.cuda.is_available():
               return torch.device("cuda:0")
       # Fallback to CPU
       return torch.device("cpu")
   ```

3. **Run Training with SDK:**
   ```bash
   /opt/rocm_sdk_612/bin/python3 scripts/training/train_challenge2_fast.py
   ```

---

## Alternative: Wait for Official ROCm Support

If the SDK Python environment issues are too complex:

**Short-term:** Continue with CPU training (stable, works now)
**Long-term:** Wait for PyTorch 2.6+ which may have better gfx1010 support

---

## Performance Expectations

Once working, GPU training should provide:
- **Speedup:** 3-5x faster than CPU
- **Per-batch:** ~50-80ms vs 271ms on CPU
- **Training time:** ~30-45 min per epoch vs ~2 hours on CPU

---

## References

- [ROCm SDK Builder](https://github.com/lamikr/rocm_sdk_builder) - Tool used to build the SDK
- [AMD ROCm Documentation](https://docs.amd.com/)
- [PyTorch ROCm](https://pytorch.org/get-started/locally/)

---

## Troubleshooting

### Issue: `typing` module conflicts
**Cause:** SDK's Python 3.11 packages conflict with system Python  
**Solution:** Use SDK's Python directly: `/opt/rocm_sdk_612/bin/python3`

### Issue: braindecode not found
**Cause:** Not installed in SDK Python environment  
**Solution:** Install with SDK's pip: `/opt/rocm_sdk_612/bin/pip3 install braindecode`

### Issue: Memory aperture violation still occurs
**Cause:** Not using SDK PyTorch  
**Solution:** Verify with `which python3` and use full SDK path

---

**Last Updated:** October 23, 2025, 9:15 PM EDT
