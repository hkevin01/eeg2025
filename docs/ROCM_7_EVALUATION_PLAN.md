# ROCm 7.0.2 Evaluation Plan

## 🎯 Goal
Determine if ROCm 7.0.2 with official gfx1030 support resolves HSA aperture violations and convolution freezing issues on AMD Radeon RX 5600 XT.

## 📋 Pre-Installation Checklist

- [ ] Upload corrected competition submission first (highest priority)
- [ ] Run baseline GPU tests on ROCm SDK 6.1.2
- [ ] Run baseline GPU tests on venv_rocm622
- [ ] Document current performance metrics
- [ ] Save disk space (delete old logs/checkpoints if needed)

## 🔧 Installation Steps

### 1. Check System Requirements
```bash
# Verify Ubuntu version (needs 22.04.5 or 24.04.3)
lsb_release -a

# Check current ROCm installations
ls -la /opt/rocm*

# Check available disk space
df -h
```

### 2. Install ROCm 7.0.2
```bash
# Add ROCm 7.0 repository (if not already present)
wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/jammy/amdgpu-install_6.3.60300-1_all.deb
sudo dpkg -i amdgpu-install_*_all.deb
sudo apt update

# Install ROCm 7.0.2
sudo amdgpu-install --usecase=rocm --rocmrelease=7.0.2

# Verify installation
ls -la /opt/rocm-7.0.2
/opt/rocm-7.0.2/bin/rocminfo | grep "Name:.*gfx"
```

### 3. Create Python Environment
```bash
# Create new venv
cd /home/kevin/Projects/eeg2025
python3 -m venv venv_rocm702

# Activate and upgrade pip
source venv_rocm702/bin/activate
pip install --upgrade pip wheel setuptools

# Install PyTorch for ROCm 7.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.0

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

### 4. Install Project Dependencies
```bash
# Still in venv_rocm702
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

## 🧪 Testing Protocol

### Phase 1: Basic Verification
```bash
# Test 1: ROCm detection
/opt/rocm-7.0.2/bin/rocminfo | grep -A5 "Name:.*gfx"
/opt/rocm-7.0.2/bin/rocm-smi

# Test 2: PyTorch GPU detection
source venv_rocm702/bin/activate
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_properties(0))"

# Test 3: Architecture verification
python -c "import torch; print(torch.cuda.get_device_properties(0).gcnArchName)"
```

### Phase 2: GPU Test Suite
```bash
cd tests/gpu

# Run all tests with ROCm 7.0.2
source ../../venv_rocm702/bin/activate
export ROCM_PATH=/opt/rocm-7.0.2
export LD_LIBRARY_PATH=/opt/rocm-7.0.2/lib:$LD_LIBRARY_PATH

# Test 1: Basic operations (5s)
python test_01_basic_operations.py

# Test 2: Convolutions (10s) - CRITICAL TEST
python test_02_convolutions.py

# Test 3: Training loop (15s)
python test_03_training_loop.py

# Test 4: Memory stress (120s) - APERTURE VIOLATION TEST
python test_04_memory_stress.py
```

### Phase 3: Real Training Test
```bash
# Short training run (100 iterations)
cd /home/kevin/Projects/eeg2025
source venv_rocm702/bin/activate

python scripts/train_challenge1.py \
    --config configs/challenge1_tcn_config.yaml \
    --max_iterations 100 \
    --output_dir test_rocm702
```

## 📊 Success Criteria

### Must Pass (Critical)
- ✅ ROCm 7.0.2 detects gfx1030
- ✅ PyTorch sees GPU and reports correct architecture
- ✅ Test 01: Basic operations complete without errors
- ✅ Test 02: Convolutions complete without freezing (ROCm 6.2.2 FAILS this)
- ✅ Test 03: Training loop completes successfully
- ✅ Test 04: Memory stress test runs 500 iterations (ROCm SDK FAILS this)

### Should Pass (Desired)
- ✅ Training run completes 100 iterations
- ✅ No HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
- ✅ Performance similar or better than ROCm SDK 6.1.2
- ✅ Memory usage reasonable (~4-5GB for training)

### Documentation
- ✅ Record all test results
- ✅ Note any warnings or errors
- ✅ Compare performance with older ROCm versions
- ✅ Document GPU utilization and memory usage

## 🎯 Decision Matrix

| Test Results | Decision | Next Steps |
|--------------|----------|------------|
| All tests pass ✅ | **Use ROCm 7.0.2** | Delete SDK 6.1.2, train enhanced models |
| Test 02 freezes ❌ | Skip ROCm 7.0.2 | Continue investigating ROCm 6.2.2 or rebuild SDK |
| Test 04 fails ❌ | ROCm 7.0.2 has same bug | Investigate kernel parameters or hardware issue |
| Training crashes ❌ | Not production ready | Stick with SDK 6.1.2 or rebuild for gfx1030 |

## 📝 Notes

### Known Issues to Watch For
1. **HSA Aperture Violation:** If this occurs in test 04, ROCm 7.0.2 doesn't fix the issue
2. **Conv Freeze:** If test 02 hangs, ROCm 7.0.2 has same bug as 6.2.2
3. **Performance Regression:** If training is significantly slower, may need optimization
4. **Memory Leaks:** Monitor VRAM usage during extended tests

### Environment Variables
```bash
export ROCM_PATH=/opt/rocm-7.0.2
export HIP_VISIBLE_DEVICES=0
export PYTORCH_ROCM_ARCH=gfx1030
export HSA_OVERRIDE_GFX_VERSION=10.3.0
```

### Rollback Plan
If ROCm 7.0.2 doesn't work:
1. Deactivate venv_rocm702
2. Continue using ROCm SDK 6.1.2
3. Consider rebuilding SDK with gfx1030 flag
4. Document issues for AMD bug report

## 🚀 Expected Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Installation | 30 min | Including downloads |
| Basic verification | 10 min | Quick sanity checks |
| GPU test suite | ~3 min | All 4 tests |
| Training test | 5-10 min | 100 iterations |
| Documentation | 15 min | Record results |
| **Total** | **~1 hour** | Plus decision making |

## ✅ Final Checklist

### Before Installation
- [ ] Upload competition submission
- [ ] Baseline tests completed
- [ ] Disk space verified (>10GB free)
- [ ] Backup current environment variables

### During Installation
- [ ] ROCm 7.0.2 installed successfully
- [ ] venv_rocm702 created
- [ ] PyTorch installed with ROCm 7.0 support
- [ ] Dependencies installed

### After Installation
- [ ] All 4 GPU tests pass
- [ ] Training test completes
- [ ] Results documented
- [ ] Decision made on environment choice

### Cleanup (if adopting ROCm 7.0.2)
- [ ] Delete venv_rocm57
- [ ] Delete venv
- [ ] Consider removing ROCm SDK 6.1.2
- [ ] Update launcher scripts to use ROCm 7.0.2
- [ ] Update README.md with final environment

