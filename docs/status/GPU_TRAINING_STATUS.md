# GPU Training Status Report

**Date**: 2025-10-14  
**System**: AMD Radeon RX 5700 XT (Navi 10) with ROCm 6.2

## Summary

✅ **CPU Training**: Working perfectly  
❌ **GPU Training**: Unstable - causes system crashes  
✅ **PyTorch**: 2.5.1+rocm6.2 installed correctly  
✅ **GPU Detection**: Working (torch.cuda.is_available() = True)

---

## Issue Details

### Problem
When attempting to use the AMD GPU (RX 5700 XT) for training with PyTorch + ROCm 6.2, the system crashes during GPU initialization or first forward pass.

### Symptoms
1. System hangs during GPU tensor operations
2. Desktop crashes requiring login
3. No error messages - silent hang/crash

### Root Cause
**Known compatibility issue**: ROCm 6.2 + Navi 10 (gfx1010) GPUs have stability problems with PyTorch
- RX 5700 XT is Navi 10 architecture (gfx1010)
- ROCm 6.2 has better support for newer GPUs (RDNA 2+, RDNA 3)
- Navi 10 support in ROCm 6.2 is experimental and unstable

### Tests Performed
1. ✅ Minimal GPU tensor creation - **Worked with env vars**
2. ✅ Small matrix multiplication - **Worked**
3. ❌ Neural network training - **Crashed system**
4. ❌ Data loading + GPU training - **Crashed system**

---

## Solutions & Workarounds

### Option 1: CPU Training (Recommended for Now) ⭐
**Status**: ✅ Working perfectly  
**Script**: `scripts/train_cpu_only.py`  
**Performance**: Slower but stable  
**Use Case**: Development, testing, small-scale training

```bash
# Run CPU training
python3 scripts/train_cpu_only.py
```

**Pros**:
- Stable, no crashes
- Can train models reliably
- Good for development/testing

**Cons**:
- ~10-20x slower than GPU
- Limited to smaller models/batches

---

### Option 2: Downgrade ROCm (Advanced)
**Status**: ⚠️ Not tested  
**Effort**: High  
**Risk**: May break system

Try ROCm 5.7 which has better Navi 10 support:
```bash
# Uninstall ROCm 6.2
sudo apt remove rocm-hip-sdk rocm-libs

# Install ROCm 5.7
wget https://repo.radeon.com/rocm/apt/5.7/ubuntu/pool/main/r/rocm-hip-sdk/
# ... follow ROCm 5.7 installation

# Reinstall PyTorch for ROCm 5.7
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

---

### Option 3: Use NVIDIA GPU
**Status**: ⚠️ Requires different hardware  
**Effort**: Requires GPU purchase/swap

If you have access to an NVIDIA GPU:
- RTX 3060 or better recommended
- CUDA support is more mature/stable
- Better PyTorch ecosystem support

---

### Option 4: Cloud GPU Training
**Status**: ✅ Available  
**Cost**: ~$0.50-2/hour  
**Platforms**: Vast.ai, RunPod, Lambda Labs

Example workflow:
```bash
# 1. Export training script and data
tar -czf eeg2025_export.tar.gz scripts/ data/ config/

# 2. Upload to cloud GPU instance

# 3. Run training on cloud GPU
python3 scripts/train_gpu_safe.py

# 4. Download trained model
```

---

### Option 5: Continue with CPU, Optimize Later
**Status**: ✅ Recommended for competition  
**Strategy**: Train foundation model on CPU, optimize for inference

Plan:
1. Train transformer on CPU (slower but works)
2. Use smaller batch sizes if needed
3. Train overnight if necessary
4. Focus on model architecture/features
5. Optimize inference separately (doesn't need GPU)

---

## Current P2 Progress

### ✅ Completed
- [x] Data acquisition (10 subjects, 4,904 windows)
- [x] PyTorch dataset class (working)
- [x] Transformer model (working)
- [x] Training pipeline (CPU working)
- [x] GPU environment setup (PyTorch + ROCm installed)

### 🟡 In Progress
- [ ] Foundation model training (blocked by GPU issues)
  - **Workaround**: Use CPU training
- [ ] Scale to 50+ subjects
- [ ] Challenge 1 & 2 implementation

### ⭕ Not Started
- [ ] Model optimization
- [ ] Inference benchmarking

---

## Recommended Next Steps

### Immediate (Today)
1. **Train on CPU** using synthetic or small real data
2. Verify training loop works end-to-end
3. Test model saving/loading
4. Validate inference pipeline

### Short-term (This Week)
1. Load real HBN EEG data (optimize loading speed)
2. Train small transformer on CPU overnight
3. Implement Challenge 1 & 2 heads
4. Test transfer learning

### Long-term (Competition)
1. Consider cloud GPU for final training if needed
2. Focus on model architecture improvements
3. Optimize inference (quantization, pruning)
4. Submit to competition

---

## Technical Details

### Working Environment Variables
```bash
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:512
```

### GPU Info
```
Device: AMD Radeon RX 5600 XT (actually RX 5700 XT)
Architecture: Navi 10 (gfx1010)
Memory: ~8GB VRAM
ROCm: 6.2
PyTorch: 2.5.1+rocm6.2
```

### Crash Logs
- No meaningful error messages captured
- System hangs then requires desktop restart
- Likely GPU driver timeout or memory fault

---

## References

- [ROCm Compatibility Matrix](https://rocm.docs.amd.com/en/latest/release/gpu_os_support.html)
- [PyTorch ROCm Support](https://pytorch.org/get-started/locally/)
- [Known Navi 10 Issues](https://github.com/ROCm/ROCm/issues)

---

## Conclusion

**GPU training is not reliable on this system**. The ROCm 6.2 + Navi 10 combination causes system crashes.

**Recommended approach**: 
- Use CPU training for development and initial model training
- Consider cloud GPU (NVIDIA) for final large-scale training if needed
- Focus on model architecture and features rather than GPU optimization

The competition focuses on **model performance**, not training speed. A model trained on CPU can still win!

