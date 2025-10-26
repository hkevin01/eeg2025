# ROCm 5.2 Quick Start for gfx1030

## TL;DR - Fast Installation

```bash
# 1. Remove old ROCm
sudo apt autoremove --purge rocm-* hip-* miopen-*

# 2. Add ROCm 5.2 repo
wget https://repo.radeon.com/rocm/apt/5.2/rocm.gpg.key
sudo apt-key add rocm.gpg.key
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/5.2/ ubuntu main' | \
    sudo tee /etc/apt/sources.list.d/rocm.list

# 3. Install
sudo apt update
sudo apt install rocm-dev rocm-libs

# 4. Add user to groups
sudo usermod -aG render $LOGNAME
sudo usermod -aG video $LOGNAME

# 5. Set environment (add to /etc/environment)
ROCM_PATH=/opt/rocm-5.2.0
HSA_OVERRIDE_GFX_VERSION=10.3.0
PYTORCH_ROCM_ARCH=gfx1030

# 6. REBOOT

# 7. Install PyTorch 1.13.1
python3.10 -m pip install --user torch==1.13.1+rocm5.2 \
    --index-url https://download.pytorch.org/whl/rocm5.2

# 8. Test
python3.10 -c "import torch; print(torch.cuda.is_available())"
```

## ⚠️ Known Issue: Convolution Failure

**Problem:** Convolutions fail with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`

**Root Cause:** RDNA1 (gfx1030) support dropped after ROCm 5.2, aperture issues remain

**Workarounds:**
1. Use CPU for convolutions
2. Try ROCm 5.0 or 5.1 instead
3. Build PyTorch from source with gfx1030 target
4. Switch to transformer-based models (no convolutions)

## Full Documentation

See [ROCM_52_SYSTEM_INSTALL.md](./ROCM_52_SYSTEM_INSTALL.md) for complete guide with troubleshooting.
