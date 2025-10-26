# GPU Test Suite

Comprehensive test suite for AMD ROCm GPU functionality and HSA aperture violation detection.

## 🎯 Purpose

This test suite validates:
1. Basic GPU operations (tensor creation, arithmetic, memory transfer)
2. Convolution operations (Conv1d, Conv2d - critical for EEG models)
3. Complete training loops (forward, backward, optimizer)
4. Memory stress testing (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION detection)

## 📋 Test Files

| Test | Description | Duration |
|------|-------------|----------|
| `test_01_basic_operations.py` | Fundamental PyTorch GPU ops | ~5s |
| `test_02_convolutions.py` | Conv1d/Conv2d operations | ~10s |
| `test_03_training_loop.py` | Full training workflow | ~15s |
| `test_04_memory_stress.py` | HSA aperture bug detection | ~120s |

## 🚀 Usage

### Test Individual Environment

```bash
# Test ROCm SDK 6.1.2
./run_all_tests.sh rocm_sdk

# Test ROCm 6.2.2 (venv_rocm622)
./run_all_tests.sh rocm622

# Test venv_rocm57
./run_all_tests.sh rocm57
```

### Run Specific Test

```bash
# ROCm SDK 6.1.2
/opt/rocm_sdk_612/bin/python3 test_01_basic_operations.py

# venv_rocm622
bash -c 'unset PYTHONPATH; unset LD_LIBRARY_PATH; source ../../../venv_rocm622/bin/activate; python test_01_basic_operations.py'
```

## 📊 Expected Results

### ROCm SDK 6.1.2 (gfx1010 build)
- ✅ Test 01: Should PASS
- ✅ Test 02: Should PASS
- ✅ Test 03: Should PASS
- ⚠️ Test 04: May FAIL with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION

### venv_rocm622 (ROCm 6.2.2, gfx1030)
- ✅ Test 01: Should PASS
- ❌ Test 02: May FREEZE on convolutions
- ❓ Test 03: Unknown (depends on conv fix)
- ❓ Test 04: Unknown (needs conv fix first)

## 🐛 Known Issues

### ROCm SDK 6.1.2
**Issue:** `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`
- **Cause:** Built for gfx1010 instead of gfx1030
- **Symptom:** Crashes after extended training (500+ iterations)
- **Fix:** Rebuild SDK with `GPU_BUILD_AMD_NAVI14_GFX1030=1`

### ROCm 6.2.2 (venv_rocm622)
**Issue:** Convolution operations freeze
- **Cause:** Unknown compatibility issue with gfx1030
- **Symptom:** `test_02_convolutions.py` hangs
- **Fix:** Under investigation

## �� Debugging

### Check GPU Architecture
```bash
rocminfo | grep "Name:.*gfx"
# Should show: gfx1030 (NOT gfx1010!)
```

### Monitor GPU During Tests
```bash
watch -n 1 'rocm-smi'
```

### Check for Crashes
```bash
dmesg | tail -50
```

## 📝 Test Checklist

When testing a new environment:

- [ ] Test 01: Basic operations work
- [ ] Test 02: Convolutions work (no freeze)
- [ ] Test 03: Training loop completes
- [ ] Test 04: 500 iterations without aperture violation
- [ ] Extended training (30+ minutes) stable
- [ ] GPU memory usage reasonable
- [ ] No kernel errors in dmesg

## 🎯 Next Steps

1. **Run tests on all environments**
   ```bash
   ./run_all_tests.sh rocm_sdk 2>&1 | tee test_results_rocm_sdk.log
   ./run_all_tests.sh rocm622 2>&1 | tee test_results_rocm622.log
   ```

2. **Compare results** - Document which environment is most stable

3. **Rebuild SDK if needed** - If ROCm SDK fails test 04, rebuild for gfx1030

4. **Debug ROCm 6.2.2** - If venv_rocm622 freezes, investigate workarounds

5. **Choose training environment** - Based on test results

