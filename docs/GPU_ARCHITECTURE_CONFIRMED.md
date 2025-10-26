# GPU Architecture Confirmed - October 25, 2025

## ✅ Hardware Identification

### GPU Details
- **Model**: AMD Radeon RX 5600 XT
- **Architecture**: RDNA 1.0 (Navi 14)
- **GCN Arch Name**: **gfx1030** (NOT gfx1010!)
- **Compute Capability**: 10.3
- **VRAM**: 6 GB (6128 MB usable)
- **Compute Units**: 18 (36 on higher models)
- **ISA**: `amdgcn-amd-amdhsa--gfx1030`

### Verification Commands
```bash
# ROCm info
rocminfo | grep -E "Name:|ISA"
# Output: Name: gfx1030
#         ISA: amdgcn-amd-amdhsa--gfx1030

# PyTorch detection
python -c "import torch; print(torch.cuda.get_device_properties(0))"
# Output: gcnArchName='gfx1030'
```

## 🔧 Correct Configuration

### Environment Variables (CORRECTED)
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # NOT 10.3.0 - this is CORRECT!
export PYTORCH_ROCM_ARCH="gfx1010"      # Should be "gfx1030"!
```

### Why the Confusion?
- **gfx1010**: Navi 10 (RX 5700 series) - RDNA 1.0
- **gfx1012**: Navi 12 (Pro cards) - RDNA 1.0
- **gfx1030**: Navi 14 (RX 5500/5600 series) - RDNA 1.0 ← **OUR CARD**
- **gfx1031**: Navi 14 variant

All are RDNA 1.0 but different chips!

## 🐛 HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION Issue

### Problem
Training with ROCm 6.1.2 SDK causes:
```
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: 
The agent attempted to access memory beyond the largest legal address.
```

### GitHub Issue
- Issue: https://github.com/hashcat/hashcat/issues/3932
- **Resolution**: "with the updated version of rocm will work" (Oct 24, 2024)
- Suggests upgrading ROCm fixes this

### Testing Results

#### ROCm SDK 6.1.2 (Custom Build)
- ✅ Basic operations work
- ✅ Model loading works  
- ❌ Training crashes with aperture violation
- Location: `/opt/rocm_sdk_612`
- PyTorch: 2.4.1

#### ROCm 6.2.2 (System Install)
- ✅ Basic tensor ops work
- ✅ Matrix multiplication works
- ✅ Neural network forward/backward pass works
- ❌ **Convolution operations FREEZE**
- Location: `/opt/rocm` → `/opt/rocm-6.2.2`
- PyTorch: 2.5.1+rocm6.2
- HIP: 6.2.41133

## 🎯 Current Status

### What Works
1. **Submission Fixed**: 
   - File renamed from `submission_sam_fixed.py` → `submission.py`
   - Zip created: `submission_sam_corrected_20251025_171136.zip`
   - Ready to upload at: `/home/kevin/Projects/eeg2025/submissions/fixed_submission_correct/`

2. **ROCm 6.2.2 Environment**:
   - Virtual env: `venv_rocm622`
   - PyTorch 2.5.1+rocm6.2 installed
   - All dependencies installed
   - GPU detected correctly as gfx1030

### What Doesn't Work
1. **ROCm 6.2.2 Convolutions**: Freeze on conv1d operations (EEG models use these!)
2. **ROCm 6.1.2 Training**: Memory aperture violations during training

## 🤔 Next Steps

### Option 1: Debug ROCm 6.2.2 Freeze
- Investigate why convolutions freeze
- Check if it's a gfx1030-specific issue
- May need workarounds or patches

### Option 2: Stay with ROCm SDK 6.1.2
- Known to work for basic ops
- Aperture violation might be intermittent
- Could try tuning batch size, workers, etc.

### Option 3: Try Different ROCm Version
- ROCm 6.0.x might be more stable
- ROCm 6.1.3 if available
- Check AMD's official gfx1030 support matrix

### Immediate Action
1. **Upload corrected submission** ← DO THIS FIRST!
2. Test if ROCm SDK 6.1.2 has same conv freeze
3. Research gfx1030 + ROCm 6.2 compatibility
4. Consider training on CPU if GPU unstable (slower but working)

## 📝 Configuration Files to Update

### Files with Wrong Architecture
1. `~/.bashrc`: `PYTORCH_ROCM_ARCH="gfx1010"` → should be `"gfx1030"`
2. All launcher scripts in `scripts/launchers/`
3. Documentation in `docs/guides/`

### Correct Configuration
```bash
export ROCM_PATH="/opt/rocm"  # or "/opt/rocm_sdk_612"
export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Correct for gfx1030
export PYTORCH_ROCM_ARCH="gfx1030"      # Corrected!
export HIP_VISIBLE_DEVICES=0
export HSA_XNACK=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
```

## 🎉 Summary

**Confirmed**: AMD Radeon RX 5600 XT is **gfx1030** architecture.

**Submission**: Fixed and ready to upload!

**Training**: Needs investigation - both ROCm versions have issues with this consumer GPU.

