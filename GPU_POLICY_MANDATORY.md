# 🔥 GPU Training Policy - MANDATORY

**Date Created**: October 24, 2025, 21:45 UTC  
**Status**: ✅ ENFORCED

---

## 🚨 CRITICAL RULE

**ALL TRAINING MUST USE GPU - NO EXCEPTIONS**

This is a **mandatory policy** for the EEG 2025 competition project. Training on CPU is **strictly prohibited** except for testing/debugging purposes.

---

## Why GPU is Mandatory

### Performance Impact

| Device | Challenge 1 (30 epochs) | Challenge 2 (20 epochs) |
|--------|-------------------------|-------------------------|
| **GPU (SDK)** | 2-4 hours | 2-4 hours |
| **CPU** | 8-12 hours | 12-16 hours |
| **Speedup** | **3-6x faster** | **4-8x faster** |

### Competition Deadline Pressure

- **Deadline**: November 3, 2025 (9 days remaining)
- **Available Time**: Limited window for iterations
- **Need**: Fast training to test hyperparameters, architectures, strategies
- **Reality**: CPU training is too slow for competitive work

---

## Hardware Configuration

### Current System
- **GPU**: AMD Radeon RX 5600 XT
- **VRAM**: 5.98 GB (6 GB total)
- **Architecture**: gfx1010:xnack- (Navi 10)
- **TDP**: 150W
- **Memory Bandwidth**: 288 GB/s

### Why This GPU Requires Custom SDK

**Problem**: Standard PyTorch ROCm doesn't support consumer GPUs
- Official ROCm: Only MI100, MI200, MI300 (server GPUs)
- Consumer GPUs: RX 5000/6000/7000 series **NOT officially supported**
- Result: `RuntimeError: HIP error: invalid device function`

**Solution**: ROCm SDK Builder creates custom PyTorch with gfx1010 kernels

---

## ROCm SDK Setup

### SDK Location
```
/opt/rocm_sdk_612/
├── bin/
│   └── python3 -> python3.11
├── lib/
│   └── python3.11/
│       └── site-packages/
│           ├── torch/           (PyTorch 2.4.1 with gfx1010)
│           ├── braindecode/     (1.2.0)
│           └── eegdash/         (0.4.1)
└── lib64/
```

### Quick Activation

**Method 1: Use activation script** (easiest):
```bash
source activate_sdk.sh
sdk_python your_training_script.py
```

**Method 2: Manual environment** (for tmux):
```bash
export ROCM_SDK_PATH="/opt/rocm_sdk_612"
export PYTHONPATH="${ROCM_SDK_PATH}/lib/python3.11/site-packages"
export LD_LIBRARY_PATH="${ROCM_SDK_PATH}/lib:${ROCM_SDK_PATH}/lib64:${LD_LIBRARY_PATH}"
export PATH="${ROCM_SDK_PATH}/bin:${PATH}"
unset HSA_OVERRIDE_GFX_VERSION

${ROCM_SDK_PATH}/bin/python3 train.py
```

---

## Training Templates

### Tmux Training Session (Recommended)

```bash
#!/bin/bash
# Start GPU training in tmux (crash-resistant, SSH-safe)

SESSION_NAME="training_c2"
SCRIPT="train_c2_sam_real_data.py"
LOG="training_sam_c2_sdk.log"

tmux new-session -d -s $SESSION_NAME "
export ROCM_SDK_PATH='/opt/rocm_sdk_612'
export PYTHONPATH=\"\${ROCM_SDK_PATH}/lib/python3.11/site-packages\"
export LD_LIBRARY_PATH=\"\${ROCM_SDK_PATH}/lib:\${ROCM_SDK_PATH}/lib64:\${LD_LIBRARY_PATH}\"
export PATH=\"\${ROCM_SDK_PATH}/bin:\${PATH}\"
unset HSA_OVERRIDE_GFX_VERSION

echo '✅ Using ROCm SDK with gfx1010 PyTorch support'
echo 'GPU: AMD Radeon RX 5600 XT'
echo 'PyTorch: 2.4.1'
echo ''

\${ROCM_SDK_PATH}/bin/python3 -u $SCRIPT 2>&1 | tee $LOG
"

echo "✅ Training started in tmux session: $SESSION_NAME"
echo "   Attach: tmux attach -t $SESSION_NAME"
echo "   Monitor: tail -f $LOG"
```

### Direct Execution (Quick Tests)

```bash
#!/bin/bash
# For quick tests only (not crash-resistant)

source activate_sdk.sh
sdk_python test_script.py
```

---

## Verification Checklist

Before starting ANY training, verify:

### ✅ GPU Check
```bash
rocm-smi
# Should show: AMD Radeon RX 5600 XT, gfx1010
```

### ✅ SDK Check
```bash
ls /opt/rocm_sdk_612/bin/python3
# Should exist and be a symlink to python3.11
```

### ✅ PyTorch Check
```bash
/opt/rocm_sdk_612/bin/python3 -c "
import torch
assert torch.cuda.is_available(), 'GPU not available!'
assert 'AMD' in torch.cuda.get_device_name(0), 'Wrong GPU!'
print('✅ GPU ready:', torch.cuda.get_device_name(0))
"
```

### ✅ Import Check
```bash
/opt/rocm_sdk_612/bin/python3 -c "
import torch
import braindecode
import eegdash
print('✅ All imports successful')
print(f'   PyTorch: {torch.__version__}')
print(f'   Braindecode: {braindecode.__version__}')
"
```

---

## Common Issues & Solutions

### Issue 1: "HIP error: invalid device function"
**Cause**: Using standard PyTorch ROCm instead of SDK  
**Solution**: Verify PYTHONPATH points to SDK, use SDK Python

### Issue 2: "ModuleNotFoundError: No module named 'torch'"
**Cause**: PYTHONPATH not set correctly  
**Solution**: Source activate_sdk.sh or set PYTHONPATH manually

### Issue 3: Training on CPU despite GPU available
**Cause**: Script hardcoded device="cpu" or wrong PyTorch  
**Solution**: Use SDK Python and verify device selection in script

### Issue 4: Tmux session not persisting
**Cause**: Wrong tmux syntax or session naming  
**Solution**: Use template above, verify with `tmux ls`

---

## Training Logs & Monitoring

### Log Files
- **Challenge 1**: `training_sam_c1_cpu.log` (historical, CPU-based)
- **Challenge 2**: `training_sam_c2_sdk.log` (current, GPU-based)

### Monitoring Commands

```bash
# Check tmux sessions
tmux ls

# Attach to training session
tmux attach -t sam_c2

# Monitor log in real-time
tail -f training_sam_c2_sdk.log

# Check GPU usage
watch -n 1 rocm-smi

# Extract key metrics
grep -E "Epoch|Best|NRMSE|Loss" training_sam_c2_sdk.log | tail -20
```

---

## Success Metrics

### Challenge 1 (Complete)
- ✅ **Result**: 0.3008 NRMSE
- ✅ **Improvement**: 70% over baseline (1.0015)
- ✅ **Device**: CPU (EEGNeX GPU incompatible at that time)
- ✅ **Status**: READY FOR SUBMISSION

### Challenge 2 (In Progress)
- 🔄 **Status**: Training on GPU with SDK
- 🔄 **Target**: NRMSE < 0.9 (baseline: 1.0087)
- 🔄 **Expected**: 2-4 hours training time
- 🔄 **Log**: training_sam_c2_sdk.log

---

## Documentation References

- **Memory Bank**: `.github/instructions/memory.instruction.md` (GPU policy section)
- **README**: `README.md` (AMD GPU ROCm SDK Builder Solution section)
- **SDK Activation**: `activate_sdk.sh`
- **Status Document**: `C2_SDK_TRAINING_STATUS.md`
- **ROCm SDK Builder**: https://github.com/lamikr/rocm_sdk_builder

---

## Enforcement

This policy is **mandatory** and will be:

1. ✅ **Documented** in memory bank (permanent reminder)
2. ✅ **Documented** in README.md (visible to all)
3. ✅ **Enforced** in training scripts (GPU checks)
4. ✅ **Monitored** during training (verify GPU usage)
5. ✅ **Reviewed** before submission (ensure GPU training)

**No CPU training will be accepted** for final models unless explicitly justified and approved.

---

**Last Updated**: October 24, 2025, 21:45 UTC  
**Status**: ✅ ACTIVE & ENFORCED
