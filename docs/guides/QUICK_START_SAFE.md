# 🛡️ QUICK START - SAFE MODE

## ⚠️ YOUR GPU IS UNSTABLE - USE CPU ONLY

Your AMD Radeon RX 5600 XT causes **system crashes** with visual artifacts when doing deep learning. Use CPU-only mode.

## ✅ Safe Commands (Use These)

```bash
# Test that CPU-only works
python scripts/test_cpu_minimal.py

# Train on CPU safely
python scripts/train_cpu_only_safe.py

# Check NO GPU is being used
rocm-smi  # Should show 0% usage
```

## ❌ Dangerous Commands (DO NOT USE)

```bash
# These will CRASH your system:
python scripts/train_amd_optimized.py              # ⚠️ CRASHES
python scripts/test_enhanced_gpu_system.py         # ⚠️ CRASHES
python scripts/train_enhanced_gpu.py               # ⚠️ CRASHES
```

## 🚨 Crash Symptoms

If you see these, **immediately Ctrl+C**:
- Checkerboard/RGB artifacts on screen
- System slowing down
- Display becoming fuzzy
- Any GPU memory usage > 0%

## 📊 Files Created

### ✅ SAFE to use:
- `scripts/train_cpu_only_safe.py` - CPU training
- `scripts/test_cpu_minimal.py` - CPU test
- `docs/GPU_OPTIMIZATION_SUMMARY.md` - Full documentation

### ⚠️ UNSAFE (will crash):
- `scripts/train_amd_optimized.py`
- `scripts/train_enhanced_gpu.py`
- `scripts/test_enhanced_gpu_system.py`
- `src/gpu/enhanced_gpu_optimizer.py`
- `src/gpu/amd_optimized_gpu.py`
- `src/models/enhanced_gpu_layers.py`

## 💡 Recommendation

**Use Google Colab** for GPU training instead:
1. Upload your data to Google Drive
2. Use Colab's free GPU (NVIDIA Tesla T4)
3. Train there instead of on your unstable AMD GPU

## 🎯 For the Competition

Train on CPU with:
- Smaller models
- Lower batch sizes
- More epochs (since it's safe to leave running)
- Focus on feature engineering over model complexity

---

**Remember:** A slow CPU is better than a crashed system! 🛡️
