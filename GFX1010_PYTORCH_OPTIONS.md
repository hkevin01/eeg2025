# PyTorch gfx1010 Support Options for AMD RX 5600 XT

**Date**: October 23, 2025  
**GPU**: AMD Radeon RX 5600 XT (Navi 10, gfx1010)  
**Current System**: ROCm 6.2.41133, PyTorch 2.5.1+rocm6.2  
**Issue**: gfx1010 architecture support was dropped in ROCm 6.0+

---

## ✅ **VERIFIED SOLUTION: Custom ROCm SDK**

**You already have a working solution!** Your custom ROCm SDK at `/opt/rocm_sdk_612` has PyTorch 2.4.1 built with **native gfx1010 support**.

### Quick Start
```bash
# Activate SDK environment
source /home/kevin/Projects/eeg2025/activate_sdk.sh

# Run your training
sdk_python your_training_script.py
```

### Verification Results
- ✅ PyTorch 2.4.1 with ROCm 6.1.2
- ✅ Native gfx1010:xnack- detection (no HSA override needed)
- ✅ GPU tensor operations work correctly
- ✅ Matrix multiplication validated

---

## ⚠️ Critical Understanding

AMD dropped official support for gfx1010 (RDNA 1.0) starting with ROCm 6.0. The last officially supported versions were:
- **ROCm 5.7.x** (last release with gfx1010)
- **PyTorch 2.2.2** (last version with ROCm 5.7 wheels)

Your current setup (ROCm 6.2 + PyTorch 2.5.1) does **NOT** have gfx1010 kernels, which is why you're seeing memory aperture violations and crashes.

