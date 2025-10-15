# GPU Safety Testing Guide

**⚠️ IMPORTANT: Your GPU has been causing system crashes/blackouts**

This guide provides a safe way to test GPU compatibility without crashing your system.

## Safety Features Implemented

### 1. Ultra-Safe Test Script (`scripts/gpu_ultra_safe_test.py`)
- **Progressive testing**: Tests get more complex step-by-step
- **Timeout protection**: Each test has a timeout (5-15s)
- **Early termination**: Stops at first sign of trouble
- **Minimal operations**: Only 5x5 tensors initially
- **No training**: Just basic GPU operations

### 2. Monitored Wrapper (`scripts/run_gpu_test_monitored.sh`)
- **External timeout**: Kills process after 120s
- **Logging**: All output saved to log file
- **Abort option**: 3-second countdown to abort
- **Clear reporting**: Success/failure clearly indicated

## How to Run the Safe Test

```bash
cd /home/kevin/Projects/eeg2025
./scripts/run_gpu_test_monitored.sh
```

### What to Expect

1. **3-second countdown**: Press Ctrl+C to abort if needed
2. **Progressive tests**: 8 tests run sequentially
3. **Each test has timeout**: 5-15 seconds max per test
4. **Stops on failure**: Won't continue if any test fails/hangs
5. **Total max time**: 2 minutes (120s external timeout)

### Test Sequence

1. ✅ Import PyTorch (5s timeout)
2. ✅ Check CUDA availability (5s timeout)
3. ⚠️ Get GPU device name (10s timeout) - **May hang here**
4. ⚠️ Create device object (5s timeout)
5. ✅ Create CPU tensor (5s timeout)
6. 🔴 Create TINY GPU tensor 5x5 (15s timeout) - **CRITICAL**
7. 🔴 Tiny GPU matmul 5x5 (15s timeout) - **CRITICAL**
8. ⚠️ Move GPU→CPU (10s timeout)

### Interpreting Results

#### ✅ ALL 8 TESTS PASS
- GPU works for basic operations
- **However**: Full training may still crash
- **Recommendation**: Test with very small model first

#### ⚠️ 5-7 TESTS PASS
- GPU partially works
- Likely hangs on actual operations
- **Recommendation**: DO NOT attempt training
- **Recommendation**: Use CPU (already running)

#### ❌ 0-4 TESTS PASS
- GPU not compatible
- System likely to crash on GPU use
- **Recommendation**: CPU only

## Current Status

### CPU Training: ✅ RUNNING (Safe & Stable)
- Process: `train_foundation_cpu.py`
- Status: Active, loading dataset
- Device: CPU
- No crash risk

### GPU Testing: ⚠️ READY TO TEST
- Ultra-safe script: Created
- Monitored wrapper: Created
- Guardrails: Multiple layers
- Risk: Minimized (but not zero)

## Recommendations

### Option 1: Test GPU (With Caution)
```bash
# Run the monitored test
./scripts/run_gpu_test_monitored.sh

# Watch output carefully
# Press Ctrl+C if system shows any instability
```

### Option 2: Skip GPU (Safest)
- CPU training is already running
- CPU is stable and proven working
- Will complete in 2-4 hours
- No crash risk

## What We Know

### From OpenNLP-GPU Analysis
- Your `opennlp-gpu` project works with same hardware
- Uses same environment variables (HSA_OVERRIDE_GFX_VERSION=10.3.0)
- Has graceful CPU fallback
- Focus is on CPU for production

### From Previous Tests
- `check_gpu.py` showed "GPU READY"
- Basic tensor operations (10x10, 100x100) worked
- But full training caused crashes
- Issue: RX 5700 XT + ROCm 6.2 incompatibility

### The Problem
- AMD RX 5700 XT (gfx1010/Navi 10)
- Not officially supported in ROCm 6.0+
- hipBLASLt not available (uses fallback)
- Can work for basic ops but unstable for training

## Decision Tree

```
Do you need GPU urgently?
├─ YES → Run monitored test
│         ├─ All pass → Try tiny training test
│         └─ Any fail → Use CPU
└─ NO  → Let CPU training continue (RECOMMENDED)
          ├─ Safe and stable
          ├─ Will complete in 2-4 hours
          └─ Can test GPU later if needed
```

## Emergency Procedures

### If System Starts to Hang
1. **Immediately press Ctrl+C**
2. If unresponsive, switch to TTY (Ctrl+Alt+F2)
3. Login and run: `pkill -9 python3`
4. Reboot if necessary

### If System Blacks Out
1. Hard reset (power button)
2. After reboot, check logs: `logs/gpu_safe_test_*.log`
3. **DO NOT** attempt GPU training again
4. Use CPU exclusively

## Files Created

- `scripts/gpu_ultra_safe_test.py` - Progressive GPU test (305 lines)
- `scripts/run_gpu_test_monitored.sh` - Monitored wrapper (50 lines)
- `GPU_SAFETY_GUIDE.md` - This guide

## Next Steps

Choose one:

### A. Test GPU (Cautious)
```bash
cd /home/kevin/Projects/eeg2025
./scripts/run_gpu_test_monitored.sh
# Watch for any signs of hang
# Press Ctrl+C if concerned
```

### B. Continue with CPU (Safe) ✅ RECOMMENDED
```bash
# Monitor CPU training
./monitor_training.sh

# Training will complete in ~2-4 hours
# Can work on other tasks meanwhile
```

## Summary

**Safe Path**: Let CPU training finish → Then decide about GPU
**Risky Path**: Test GPU now → May crash system → May learn if GPU usable

**Your choice!** Both options documented and ready.
