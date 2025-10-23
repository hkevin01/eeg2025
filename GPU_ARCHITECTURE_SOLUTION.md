# GPU Architecture Solution - CRITICAL FIX REQUIRED

**Date:** October 23, 2025  
**Status:** 🔴 MISCONFIGURATION DETECTED - EASY FIX AVAILABLE

---

## 🎯 THE PROBLEM

Your AMD Radeon RX 5600 XT is **INCORRECTLY CONFIGURED**:

### Hardware Reality
```
GPU Model:        AMD Radeon RX 5600 XT
Chip ID:          0x731f
Real Architecture: gfx1010 (Navi 10, RDNA 1.0)
ROCm Support:     ✅ NATIVE - Fully supported in ROCm 6.2!
```

### Current Configuration (WRONG)
```
HSA_OVERRIDE_GFX_VERSION: 10.3.0  ❌ WRONG!
ROCm Reports:             gfx1030 ❌ WRONG!
Result:                   Memory aperture violations
```

---

## 🔍 ROOT CAUSE

Your `.bashrc` contains **TWO duplicate entries** forcing the GPU to be treated as gfx1030:

```bash
# In ~/.bashrc (LINES TO REMOVE):
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # ❌ WRONG FOR YOUR GPU!
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # ❌ DUPLICATE!
```

This override was likely added to work around gfx1030 issues on a DIFFERENT GPU, but your RX 5600 XT is **gfx1010** and doesn't need any override!

---

## ✅ THE SOLUTION

### Step 1: Remove the Incorrect Override

Edit `~/.bashrc` and **REMOVE or COMMENT OUT** these lines:

```bash
# Comment out or delete these lines:
# export HSA_OVERRIDE_GFX_VERSION=10.3.0  # NOT NEEDED FOR gfx1010!
# export HIP_VISIBLE_DEVICES=0            # NOT NEEDED
```

Keep these lines (they're correct):
```bash
export PATH=$PATH:/opt/rocm/bin:/opt/rocm/rocprofiler/bin:/opt/rocm/opencl/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/lib64
```

### Step 2: Apply Changes

```bash
# Reload bashrc
source ~/.bashrc

# Verify the override is gone
echo $HSA_OVERRIDE_GFX_VERSION  # Should be empty

# Start a new terminal session
# (or unset manually: unset HSA_OVERRIDE_GFX_VERSION)
```

### Step 3: Verify Correct Detection

```bash
cd /home/kevin/Projects/eeg2025
python3 -c "
import torch
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'Architecture: {props.gcnArchName}')
"
```

Expected output:
```
CUDA Available: True
GPU: AMD Radeon RX 5600 XT
Architecture: gfx1010  # ✅ CORRECT!
```

### Step 4: Test GPU Training

```bash
# Run our GPU test suite
python3 tests/test_rocm_eegnex_gpu.py
```

---

## 📊 Why This Will Work

### gfx1010 (Your GPU) vs gfx1030
| Feature | gfx1010 (RX 5600 XT) | gfx1030 (RX 6600 XT) |
|---------|---------------------|---------------------|
| RDNA Version | RDNA 1.0 | RDNA 2.0 |
| ROCm 6.2 Support | ✅ Native | ⚠️ Requires override |
| PyTorch ROCm | ✅ Fully supported | ⚠️ Issues reported |
| Memory Aperture | ✅ No known issues | ❌ Known violations |
| Override Needed | ❌ NO | ✅ YES (10.3.0) |

### Your Current Situation
```
You have:  gfx1010 (native support)
Configured as: gfx1030 (needs workarounds)
Result:    Using wrong kernels → crashes
```

### After Fix
```
You have:  gfx1010
Configured as: gfx1010  ✅
Result:    Native support → stable GPU training!
```

---

## 🚀 Expected Performance After Fix

### Before (CPU Only)
```
Batch Size 32: 271.60 ms
Per Sample:    8.49 ms
```

### After (GPU)
```
Batch Size 32: ~30-50 ms (estimated)
Per Sample:    ~1-2 ms (estimated)
Speedup:       5-8x faster! 🚀
```

---

## 📝 Post-Fix Checklist

After removing the override and restarting your terminal:

- [ ] Verify `HSA_OVERRIDE_GFX_VERSION` is unset
- [ ] Run `rocminfo | grep "Name:" | grep gfx` → should show gfx1010
- [ ] Run `python3 tests/test_rocm_eegnex_gpu.py` → should pass all tests
- [ ] Update `train_challenge2_fast.py` to allow gfx1010 GPU
- [ ] Start Challenge 2 training on GPU
- [ ] Monitor for stability (should be stable now!)

---

## 🔧 If You Still Need the Custom SDK

**YOU DON'T!** The custom SDK at `/opt/rocm_sdk_612` was built to work around gfx1030 issues, but:

1. Your GPU is **gfx1010**, not gfx1030
2. gfx1010 is **natively supported** in ROCm 6.2
3. The system ROCm at `/opt/rocm` is perfect for your hardware

**Recommendation:** Use the system ROCm. The custom SDK is **not needed** for gfx1010.

---

## 🎯 Action Plan

### Immediate (5 minutes)
1. Edit `~/.bashrc` and remove HSA_OVERRIDE_GFX_VERSION lines
2. Open new terminal or `source ~/.bashrc`
3. Verify with `echo $HSA_OVERRIDE_GFX_VERSION` (should be empty)

### Verification (5 minutes)
4. Run `python3 tests/test_rocm_eegnex_gpu.py`
5. Should pass all 5 tests now!

### Training (2-3 hours)
6. Update `train_challenge2_fast.py` device selection
7. Start GPU training: `python3 scripts/training/train_challenge2_fast.py`
8. Monitor for stability

---

## 🐛 Why This Mistake Happened

The `HSA_OVERRIDE_GFX_VERSION=10.3.0` override is commonly recommended for:
- RX 6600/6700 XT (actual gfx1030 GPUs)
- Working around RDNA 2.0 issues

Someone (possibly from online tutorials for RX 6000 series) added this to your `.bashrc`, but your RX 5600 XT doesn't need it!

---

## 📚 Reference

### Correct Architecture Mapping
```
Chip ID | GPU Model        | Architecture | ROCm 6.2 Support
--------|------------------|--------------|------------------
0x731f  | RX 5600/5700 XT  | gfx1010      | ✅ Native
0x73af  | RX 6800 XT       | gfx1030      | ⚠️ Override
0x73bf  | RX 6700 XT       | gfx1030      | ⚠️ Override  
0x73df  | RX 6600 XT       | gfx1030      | ⚠️ Override
```

Your chip: **0x731f** = **gfx1010** = **Native support!** ✅

---

**Last Updated:** October 23, 2025, 9:25 PM EDT

**NEXT STEP:** Remove the HSA_OVERRIDE_GFX_VERSION from ~/.bashrc NOW!
