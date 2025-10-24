# ROCm 5.x Downgrade Guide for RX 5600 XT

**Issue**: RX 5600 XT (gfx1030) + ROCm 6.2/7.x + EEGNeX = Memory violation  
**Solution**: Use ROCm 5.x which has better gfx1030 support  
**Source**: Community recommendation - ROCm 5.x more stable for RDNA1/2 GPUs

## Current State

```
GPU: AMD Radeon RX 5600 XT (gfx1030)
System ROCm: 6.0-7.3.0 (mixed)
PyTorch: 2.5.1+rocm6.2
Status: EEGNeX fails with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
```

## Option 1: Install PyTorch with ROCm 5.7 (RECOMMENDED)

### Why ROCm 5.7?
- Last stable ROCm 5.x release
- Known good compatibility with gfx1030
- PyTorch officially supports it
- Less risky than building from source

### Steps

```bash
# 1. Create new conda/venv environment (optional but recommended)
conda create -n rocm5 python=3.11
conda activate rocm5

# 2. Install PyTorch with ROCm 5.7
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7

# 3. Verify
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# 4. Install other dependencies
pip3 install braindecode mne scikit-learn h5py

# 5. Test EEGNeX
python3 test_rocm_eegnex.py
```

### Pros & Cons

✅ **Pros**:
- Official PyTorch wheels available
- Just a pip install (5 minutes)
- Can keep existing ROCm 6.x system
- Easy to roll back

❌ **Cons**:
- Older PyTorch version (may be 2.1.x or 2.2.x)
- Some newer features might not be available
- Need separate environment

## Option 2: Downgrade System ROCm to 5.7

### Warning
⚠️ This affects system-wide ROCm installation!  
⚠️ May break other apps using ROCm 6.x!  
⚠️ More complex rollback!

### Steps (Ubuntu/Debian)

```bash
# 1. Remove current ROCm packages
sudo apt remove --purge rocm* hip* hsa*
sudo apt autoremove

# 2. Add ROCm 5.7 repository
wget https://repo.radeon.com/amdgpu-install/5.7/ubuntu/jammy/amdgpu-install_5.7.50700-1_all.deb
sudo apt install ./amdgpu-install_5.7.50700-1_all.deb

# 3. Install ROCm 5.7
sudo amdgpu-install --usecase=rocm

# 4. Reboot
sudo reboot

# 5. Verify
rocminfo | grep "Marketing Name"
```

### Pros & Cons

✅ **Pros**:
- System-wide consistent ROCm version
- May fix other ROCm 6.x issues

❌ **Cons**:
- Complex installation
- Requires sudo/root access
- May break other tools
- Time-consuming (1-2 hours)

## Option 3: The "Ollama Approach" 🚀

Based on your suggestion about Ollama working well:

### Concept
If Ollama runs well on your system, it means:
1. Your GPU hardware is fine
2. Some ROCm configuration works
3. There's a working GPU software stack

### Investigation Steps

```bash
# 1. Check if Ollama is installed
which ollama
ollama --version

# 2. Check what ROCm/HIP version Ollama uses
ldd $(which ollama) | grep -i "hip\|rocm\|hsa"

# 3. Check Ollama's environment
ps aux | grep ollama
cat /proc/$(pgrep ollama)/environ | tr '\0' '\n' | grep -E "HSA|HIP|ROCM"

# 4. Try using same environment for PyTorch
# Copy Ollama's environment variables to our training script
```

### If Ollama Works

We can potentially:
1. Use Ollama's ROCm configuration
2. Match its environment variables
3. Use its library paths
4. Replicate its success

## Recommendation: Try Option 1 First

**Action Plan** (30 minutes):

```bash
# Quick test with ROCm 5.7 PyTorch
conda create -n eeg_rocm5 python=3.11 -y
conda activate eeg_rocm5
pip3 install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/rocm5.7
pip3 install braindecode mne scikit-learn h5py numpy pandas
cd /home/kevin/Projects/eeg2025
python3 test_rocm_eegnex.py
```

**Expected Result**:
- If it works → Start GPU training with ROCm 5.7 PyTorch
- If it fails → Fall back to fast CPU training (already running)

## Current Fast CPU Training

**Status**: ✅ Running (13+ hours elapsed)
- Batch size: 128
- Expected completion: ~8 hours remaining
- This is our **backup plan** if GPU doesn't work

## Timeline

- **ROCm 5.7 PyTorch test**: 30 minutes
- **If successful, GPU training**: 30-60 minutes per epoch
- **Total for 3 epochs**: ~2 hours (vs 21 hours on CPU)

**Worth trying!** 🎯

## References

- PyTorch ROCm wheels: https://pytorch.org/get-started/locally/
- ROCm 5.7 docs: https://rocm.docs.amd.com/en/docs-5.7.0/
- RX 5600 XT compatibility: Known to work better with ROCm 5.x
- Community reports: ROCm 6.x has regressions for gfx1030

---

**Your call**: Want to try the 30-minute ROCm 5.7 PyTorch test while the CPU training continues in the background? 🚀
