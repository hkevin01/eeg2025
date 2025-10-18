# VSCode + AMD ROCm/HIP Setup Guide

## The Problem

VSCode terminals crash when running PyTorch GPU operations on AMD GPUs because:

1. **GPU Conflict**: VSCode's built-in GPU acceleration conflicts with PyTorch ROCm operations
2. **Memory Competition**: Both VSCode and PyTorch try to use the GPU simultaneously
3. **Driver Issues**: ROCm's HIP backend has specific requirements for stable operation
4. **FFT Operations**: AMD's hipFFT can cause system hangs without proper configuration

## The Solution

We've implemented a comprehensive fix with **three layers of protection**:

### Layer 1: Global VSCode Settings

**Location**: `~/.config/Code/User/settings.json`

**Key Settings**:
```json
{
    "terminal.integrated.gpuAcceleration": "off",  // Disable VSCode GPU use
    "disable-hardware-acceleration": true,         // Force software rendering
    "terminal.integrated.env.linux": {
        "HSA_OVERRIDE_GFX_VERSION": "10.3.0",      // AMD GPU compatibility
        "HSA_ENABLE_SDMA": "0",                     // Disable problematic DMA
        "PYTORCH_HIP_ALLOC_CONF": "max_split_size_mb:128"  // Memory management
    }
}
```

**What it does**:
- Prevents VSCode from using the GPU for rendering
- Sets up proper ROCm environment variables automatically
- Applies to ALL projects in VSCode

### Layer 2: Project-Specific Settings

**Location**: `/home/kevin/Projects/eeg2025/.vscode/settings.json`

**Additional Features**:
- Project-specific GPU configurations
- Python interpreter settings
- File watcher exclusions to reduce system load

### Layer 3: Environment Scripts

**Setup Script**: `setup_gpu_env.sh`
```bash
#!/bin/bash
export HSA_OVERRIDE_GFX_VERSION="10.3.0"
export HIP_VISIBLE_DEVICES="0"
export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:128"
export HSA_ENABLE_SDMA="0"
```

**Safe Execution Wrapper**: `run_gpu_safe.sh`
```bash
# Runs scripts with timeout and cleanup
./run_gpu_safe.sh test_gpu_debug.py
```

## How to Use

### Option 1: Restart VSCode (Recommended)

1. **Close VSCode completely** (important!)
2. **Reopen VSCode** - settings will be applied automatically
3. **Open a new terminal** - environment variables are now set
4. **Run GPU scripts normally**:
   ```bash
   python3 test_gpu_debug.py
   ```

### Option 2: Manual Environment Setup

If you don't want to restart VSCode:

```bash
# In your terminal
source setup_gpu_env.sh
python3 test_gpu_debug.py
```

### Option 3: Safe Wrapper (Extra Protection)

For scripts that might hang:

```bash
./run_gpu_safe.sh test_gpu_debug.py
```

This adds:
- 60-second timeout
- Automatic GPU cleanup on crash
- Better error reporting

## Testing the Setup

### Step 1: Quick Environment Test

```bash
python3 -c "import os; print('HSA_OVERRIDE_GFX_VERSION:', os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'NOT SET'))"
```

**Expected**: Should show `10.3.0` (after VSCode restart)

### Step 2: GPU Detection Test

```bash
python3 test_gpu_debug.py
```

**Expected Output**:
```
======================================================================
GPU DEBUG TEST - Verbose Mode
======================================================================

STEP 1: Checking environment variables...
   HSA_OVERRIDE_GFX_VERSION: 10.3.0
   HIP_VISIBLE_DEVICES: 0
   ...

STEP 4: Checking GPU availability...
   CUDA/ROCm available: True
   Platform: AMD ROCm/HIP
   Device name: AMD Radeon RX 5600 XT
```

### Step 3: Full Unified Module Test

```bash
./run_gpu_safe.sh test_unified_gpu_safe.py
```

## Environment Variables Explained

| Variable | Value | Purpose |
|----------|-------|---------|
| `HSA_OVERRIDE_GFX_VERSION` | `10.3.0` | AMD GPU architecture override for RX 5600 XT (Navi 10) |
| `HIP_VISIBLE_DEVICES` | `0` | Use only GPU 0 (prevents multi-GPU issues) |
| `ROCR_VISIBLE_DEVICES` | `0` | Same as above, but for ROCm runtime |
| `GPU_MAX_HEAP_SIZE` | `100` | Maximum heap size percentage |
| `GPU_MAX_ALLOC_PERCENT` | `100` | Maximum allocation percentage |
| `PYTORCH_HIP_ALLOC_CONF` | `max_split_size_mb:128` | Limit memory fragmentation |
| `HSA_ENABLE_SDMA` | `0` | Disable SDMA (can cause hangs on some operations) |
| `PYTHONUNBUFFERED` | `1` | Immediate output (no buffering) |

## Troubleshooting

### Problem: VSCode still crashes

**Solution**: Make sure you **completely closed and reopened VSCode**. The settings only apply on restart.

```bash
# Verify settings were applied
code --list-extensions
# Should show extensions without errors

# Check if environment is set
python3 -c "import os; print(os.environ.get('HSA_OVERRIDE_GFX_VERSION'))"
# Should output: 10.3.0
```

### Problem: Scripts hang on FFT operations

**Solution**: This is a known issue with AMD GPUs. Use the safe wrapper:

```bash
./run_gpu_safe.sh your_script.py
```

Or add timeout protection in your code:

```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)  # 5 second timeout

try:
    result = fft_operation()  # Your FFT code
    signal.alarm(0)  # Cancel timeout if successful
except TimeoutError:
    print("FFT timed out - using CPU fallback")
    result = cpu_fft_operation()
```

### Problem: Out of memory errors

**Solutions**:

1. **Reduce batch size**:
   ```python
   CONFIG = {'batch_size': 8}  # Instead of 32
   ```

2. **Clear GPU cache frequently**:
   ```python
   torch.cuda.empty_cache()
   ```

3. **Adjust memory allocation**:
   ```bash
   export PYTORCH_HIP_ALLOC_CONF="max_split_size_mb:64"  # Smaller chunks
   ```

### Problem: Environment variables not set in terminal

**Check**:
```bash
# Are you in a VSCode terminal?
echo $TERM_PROGRAM  # Should show "vscode"

# Is VSCode using the right settings?
cat ~/.config/Code/User/settings.json | grep HSA_OVERRIDE
```

**Solution**: Restart VSCode OR manually source the environment:
```bash
source setup_gpu_env.sh
```

## Performance Tips

### For Training Scripts

```python
# Use smaller batches for AMD GPUs
CONFIG = {
    'batch_size': 16,        # vs 32 for NVIDIA
    'max_samples': 3000,     # vs 5000 for NVIDIA
    'sleep_between_epochs': 2,  # Let GPU cool down
}

# Add cleanup between epochs
if device.type == 'cuda':
    torch.cuda.empty_cache()
```

### For FFT Operations

```python
# Prefer CPU FFT for very large signals on AMD
if signal_length > 10000:
    fft_result = torch.fft.rfft(signal.cpu())
else:
    fft_result = fft_opt.rfft_batch(signal)  # GPU
```

### Memory Management

```python
# Aggressive cleanup for AMD
def cleanup_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# Call after each batch or epoch
cleanup_gpu()
```

## File Reference

| File | Purpose | When to Use |
|------|---------|-------------|
| `~/.config/Code/User/settings.json` | Global VSCode settings | Automatic (restart required) |
| `.vscode/settings.json` | Project settings | Automatic in this project |
| `setup_gpu_env.sh` | Manual environment setup | When not using VSCode |
| `run_gpu_safe.sh` | Safe script execution | For scripts that might hang |
| `test_gpu_debug.py` | Verbose GPU testing | Troubleshooting issues |
| `test_unified_gpu_safe.py` | Full module test | Verify installation |

## Verification Checklist

After setup, verify each item:

- [ ] VSCode completely closed and reopened
- [ ] Global settings file exists: `~/.config/Code/User/settings.json`
- [ ] Project settings file exists: `.vscode/settings.json`
- [ ] Environment variables set (check with `echo $HSA_OVERRIDE_GFX_VERSION`)
- [ ] GPU detected: `python3 -c "import torch; print(torch.cuda.is_available())"`
- [ ] Platform correct: `python3 test_gpu_debug.py` (should show AMD ROCm/HIP)
- [ ] Basic operations work: Scripts complete without hanging
- [ ] Unified module imports: `from gpu.unified_gpu_optimized import GPUPlatformDetector`

## What Makes This Setup "Forever"

1. **Global Settings**: Applied to ALL VSCode projects automatically
2. **Persistent Configuration**: Settings survive VSCode updates
3. **Project-Specific Overrides**: Can customize per-project if needed
4. **Environment Scripts**: Work outside VSCode too
5. **Safe Wrappers**: Protection for problematic operations

## Summary

The crashes were caused by VSCode and PyTorch competing for GPU resources. We fixed this by:

1. ✅ **Disabling VSCode GPU acceleration** globally
2. ✅ **Setting proper ROCm environment variables** automatically
3. ✅ **Creating safe execution wrappers** for problematic scripts
4. ✅ **Implementing aggressive memory management** for AMD stability

**After restarting VSCode, all GPU operations should work safely!**

---

## Quick Commands Reference

```bash
# Test environment
python3 test_gpu_debug.py

# Safe execution
./run_gpu_safe.sh your_script.py

# Manual environment
source setup_gpu_env.sh

# Check GPU status
rocm-smi

# Clear GPU memory
python3 -c "import torch; torch.cuda.empty_cache()"
```
