# ROCm 5.2 System-Wide Installation Guide for gfx1030 (RX 5600 XT)

## ⚠️ Important Context

This guide addresses the **convolution HSA aperture violation** issue that affects RDNA1 consumer GPUs (gfx1030) with ROCm ≥5.3. Based on extensive research, ROCm 5.2 is the **last stable version** for these cards, but even ROCm 5.2 may experience issues with certain operations.

### Known Issues
- **Convolution operations may fail** with HSA memory aperture violations
- RDNA1/Navi 10 cards (RX 5600 XT, RX 5700 series) dropped from official support after ROCm 5.2
- PyTorch wheels for ROCm 5.2 are available but may have limited functionality
- Basic GPU operations (tensor creation, matrix multiply) typically work
- **Convolutions specifically are problematic** and may require workarounds or custom builds

---

## Prerequisites

### 1. Verify Your GPU Architecture
```bash
# Check your actual ISA (should show gfx1030, not gfx1010)
rocminfo | grep "Name:" | grep "gfx"
# OR
/opt/rocm/bin/rocminfo | grep "gfx"
```

**Critical:** Hardware reports as "Navi 10" but ISA is **gfx1030**

### 2. Check Python Version
```bash
python3.10 --version
```
Python 3.10 is **optimal** for ROCm 5.2. If not installed:
```bash
sudo apt update
sudo apt install python3.10 python3.10-dev python3.10-venv
```

### 3. Verify PCIe Atomics Support
```bash
lspci -vvv | grep -i atomic
```
Look for "AtomicOpsCap" in your GPU entry. Lack of atomics may cause issues.

---

## System-Wide ROCm 5.2 Installation

### Step 1: Remove Existing ROCm Installation
```bash
# Remove all ROCm packages
sudo apt autoremove --purge rocm-* hip-* miopen-* rocblas-* rocsolver-* 

# Clean up
sudo apt autoremove
sudo apt autoclean

# Verify removal
dpkg -l | grep rocm
```

### Step 2: Add ROCm 5.2 Repository
```bash
# Download repo setup
wget https://repo.radeon.com/rocm/apt/5.2/rocm.gpg.key
sudo apt-key add rocm.gpg.key

# Add repository
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.2/ ubuntu main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list

# Update package list
sudo apt update
```

### Step 3: Install ROCm 5.2 Core
```bash
# Install ROCm development package
sudo apt install rocm-dev rocm-libs

# Add user to groups
sudo usermod -aG render $LOGNAME
sudo usermod -aG video $LOGNAME

# REBOOT REQUIRED
echo "⚠️  REBOOT NOW before continuing"
```

### Step 4: Set Environment Variables System-Wide
```bash
# Edit system-wide environment
sudo tee -a /etc/environment << 'ENVEOF'
# ROCm 5.2 Configuration
ROCM_PATH=/opt/rocm-5.2.0
HIP_PATH=/opt/rocm-5.2.0/hip
HSA_PATH=/opt/rocm-5.2.0
PATH=/opt/rocm-5.2.0/bin:/opt/rocm-5.2.0/opencl/bin:$PATH
LD_LIBRARY_PATH=/opt/rocm-5.2.0/lib:/opt/rocm-5.2.0/lib64:$LD_LIBRARY_PATH

# Critical overrides for gfx1030
HSA_OVERRIDE_GFX_VERSION=10.3.0
PYTORCH_ROCM_ARCH=gfx1030
GPU_DEVICE_ORDINAL=0

# Debug flags (optional - enable if troubleshooting)
# AMD_LOG_LEVEL=3
# HIP_VISIBLE_DEVICES=0
ENVEOF

# Reload environment
source /etc/environment
```

### Step 5: Verify ROCm Installation
```bash
# After reboot, check ROCm version
/opt/rocm-5.2.0/bin/rocm-smi --showproduct

# Check device visibility
/opt/rocm-5.2.0/bin/rocminfo | grep -A 10 "Agent 2"

# Test HIP
/opt/rocm-5.2.0/bin/hipconfig
```

---

## Install PyTorch 1.13.1 + ROCm 5.2 (System Python)

### Option A: System-Wide pip Install (Recommended)
```bash
# Use Python 3.10
python3.10 -m pip install --user --upgrade pip

# Install PyTorch 1.13.1 with ROCm 5.2
python3.10 -m pip install --user torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 \
    --index-url https://download.pytorch.org/whl/rocm5.2

# Install NumPy fix
python3.10 -m pip install --user "numpy<2.0"
```

### Option B: User-Local Install
```bash
pip3 install --user torch==1.13.1+rocm5.2 torchvision==0.14.1+rocm5.2 \
    --index-url https://download.pytorch.org/whl/rocm5.2
```

---

## Testing & Validation

### Test 1: Basic GPU Detection
```bash
python3.10 << 'PYTEST'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"ROCm available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"GPU count: {torch.cuda.device_count()}")
PYTEST
```

### Test 2: Simple Tensor Operations
```bash
python3.10 << 'PYTEST'
import torch
x = torch.randn(10, 10).cuda()
y = torch.randn(10, 10).cuda()
z = x @ y  # Matrix multiply
print("✅ Matrix operations work")
PYTEST
```

### Test 3: Convolution Test (⚠️ May Fail)
```bash
python3.10 << 'PYTEST'
import torch
import torch.nn as nn

try:
    x = torch.randn(1, 3, 224, 224).cuda()
    conv = nn.Conv2d(3, 64, kernel_size=3, padding=1).cuda()
    y = conv(x)
    print("✅ Convolution works!")
except RuntimeError as e:
    print(f"❌ Convolution failed: {e}")
    print("This is a known issue with gfx1030 on ROCm 5.2+")
PYTEST
```

---

## Troubleshooting Convolution HSA Errors

### Issue: `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`

This error indicates ROCm cannot properly manage GPU memory for convolution operations on gfx1030.

### Attempted Solutions (Research-Based)

#### 1. Environment Variable Overrides
```bash
# Add to /etc/environment or ~/.bashrc
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH=gfx1030
export GPU_DEVICE_ORDINAL=0
export HSA_ENABLE_SDMA=0  # Disable system DMA
export GPU_MAX_HW_QUEUES=1  # Limit hardware queues
```

#### 2. Kernel Parameter (Requires Reboot)
```bash
# Check current atomics support
lspci -vvv | grep -i atomic

# If needed, add kernel parameter
sudo nano /etc/default/grub
# Add: amdgpu.pcie_gen_cap=0x4 amdgpu.pcie_lane_cap=16

sudo update-grub
sudo reboot
```

#### 3. Use ROCm 5.1 or 5.0 Instead
ROCm 5.0-5.1 may have better gfx1030 support before regressions were introduced:
```bash
# Same process but use:
deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.1.3/ ubuntu main
# OR
deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.0.2/ ubuntu main
```

#### 4. Build PyTorch from Source
Some users report success building PyTorch from source with `PYTORCH_ROCM_ARCH=gfx1030`:
```bash
git clone --recursive https://github.com/pytorch/pytorch
cd pytorch
git checkout v1.13.1

export PYTORCH_ROCM_ARCH=gfx1030
export HSA_OVERRIDE_GFX_VERSION=10.3.0
python setup.py install
```

#### 5. Use MIOpen Environment Tuning
```bash
export MIOPEN_FIND_MODE=NORMAL
export MIOPEN_DEBUG_DISABLE_FIND_DB=0
export MIOPEN_FIND_ENFORCE=3
```

---

## Alternative: Docker with Older ROCm

If system-wide installation fails, consider using Docker with ROCm 4.5 or 5.0:
```bash
docker pull rocm/pytorch:rocm5.0_ubuntu20.04_py3.7_pytorch_1.10.0

docker run -it --device=/dev/kfd --device=/dev/dri \
    --group-add video --group-add render \
    rocm/pytorch:rocm5.0_ubuntu20.04_py3.7_pytorch_1.10.0
```

---

## Known Limitations & Workarounds

### ❌ What Doesn't Work
- **Convolution layers in PyTorch** (HSA aperture errors)
- Training most CNNs directly
- Some MIOpen kernels

### ✅ What Works
- Basic tensor operations (add, multiply, matmul)
- Linear layers (fully connected)
- Transformer models (attention, embeddings)
- Data loading and preprocessing

### 🔧 Workarounds
1. **Use CPU for convolutions**: Offload conv layers to CPU, keep other ops on GPU
2. **Switch to transformer architectures**: Avoid CNNs, use attention-based models
3. **Use older ROCm in Docker**: ROCm 4.5 may work better
4. **Consider upgrading hardware**: RDNA2 (RX 6000) or RDNA3 (RX 7000) have better support

---

## References & Community Resources

- [ROCm GitHub Issue #1659](https://github.com/ROCm/ROCm/issues/1659) - Polaris/Vega support discussion
- [PyTorch ROCm gfx803 Build](https://github.com/tsl0922/pytorch-gfx803) - Custom build reference
- [ROCm Documentation](https://rocm.docs.amd.com/)
- Environment variables: [HIP Environment Docs](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/env_variables.html)

---

## Summary Checklist

```bash
# Quick validation script
cat > ~/validate_rocm52.sh << 'VALEOF'
#!/bin/bash
echo "🔍 ROCm 5.2 Validation for gfx1030"
echo "=================================="
echo ""

echo "1. Checking ROCm installation..."
/opt/rocm-5.2.0/bin/rocm-smi --showproductname || echo "❌ ROCm not found"
echo ""

echo "2. Checking environment variables..."
echo "HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH"
echo ""

echo "3. Testing PyTorch GPU..."
python3.10 -c "import torch; print(f'PyTorch {torch.__version__}'); print(f'GPU: {torch.cuda.is_available()}')"
echo ""

echo "4. Testing basic operations..."
python3.10 -c "import torch; x=torch.randn(10,10).cuda(); print('✅ GPU ops work')" 2>/dev/null || echo "❌ GPU ops failed"
echo ""

echo "5. Testing convolution (may fail)..."
python3.10 -c "import torch,torch.nn as nn; x=torch.randn(1,3,32,32).cuda(); c=nn.Conv2d(3,8,3).cuda(); y=c(x); print('✅ Conv works')" 2>/dev/null || echo "⚠️  Conv failed (expected on gfx1030)"
VALEOF

chmod +x ~/validate_rocm52.sh
~/validate_rocm52.sh
```

---

**Last Updated:** October 25, 2025  
**Status:** ROCm 5.2 installed, convolution issues remain unresolved due to hardware limitations
