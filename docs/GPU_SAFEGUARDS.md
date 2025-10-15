# GPU Training Safeguards

## Problem Summary

The AMD Radeon RX 5700 XT (Navi 10, gfx1010) causes system crashes or hangs when used with PyTorch + ROCm 6.2+ for training. This is due to:

1. **Incompatibility**: RX 5700 XT is NOT officially supported in ROCm 6.0+
2. **Driver Issues**: ROCm dropped Navi 10 (gfx1010) support
3. **Hanging**: GPU operations hang indefinitely, freezing the system

## Safeguards Implemented

### 1. Timeout Protection (`train_gpu_timeout.py`)

**How it works:**
- GPU test runs in a **separate process**
- **15-second timeout** on GPU operations
- If timeout occurs, process is **terminated** and training falls back to CPU
- Prevents system hangs and crashes

**Key Features:**
```python
# Test GPU in isolated process with timeout
def test_gpu_with_timeout(queue, timeout=10):
    # GPU operations here
    pass

# Main process waits with timeout
p.join(timeout=15)
if p.is_alive():
    p.terminate()  # Kill hung process
    return "cpu"   # Use CPU instead
```

### 2. Progressive GPU Testing

**Test Sequence:**
1. **Availability Check**: `torch.cuda.is_available()`
2. **Memory Check**: Query GPU memory (less likely to hang)
3. **Small Tensor Test**: Create 10x10 tensor on GPU
4. **Matrix Operation**: Perform simple matrix multiplication
5. **Transfer Back**: Move result back to CPU

**Each test has individual timeout protection.**

### 3. Automatic CPU Fallback

If GPU test fails or times out:
- Automatically switches to CPU training
- Logs the reason for fallback
- Training continues without interruption

### 4. Crash-Safe Training Loop

**During Training:**
```python
try:
    data = data.to(device)
    output = model(data)
    loss.backward()
except RuntimeError as e:
    if "out of memory" in str(e):
        torch.cuda.empty_cache()  # Clear memory
        continue                   # Skip batch
    else:
        raise  # Real error
```

### 5. Memory Management

**Safeguards:**
- Set memory fraction limit: `torch.cuda.set_per_process_memory_fraction(0.3)`
- Clear cache after each batch: `torch.cuda.empty_cache()`
- Use small batch sizes (1-2 samples)
- Minimize model size for testing

### 6. Environment Variables

```bash
HSA_OVERRIDE_GFX_VERSION=10.3.0    # Pretend to be RDNA2
CUDA_LAUNCH_BLOCKING=1              # Synchronous ops
PYTORCH_HIP_ALLOC_CONF=max_split_size_mb:64  # Small allocations
GPU_MAX_HEAP_SIZE=50                # Limit heap
GPU_MAX_ALLOC_PERCENT=50            # Limit allocation
PYTHONUNBUFFERED=1                  # Immediate output
```

## Usage

### Quick Test (Recommended)
```bash
python3 scripts/train_gpu_timeout.py
```

**What happens:**
1. Tests GPU with 15s timeout
2. If GPU hangs → Falls back to CPU
3. Trains on CPU successfully
4. Saves model checkpoint

### Monitor Training
```bash
# Run in background
python3 scripts/train_gpu_timeout.py > logs/training.log 2>&1 &

# Monitor progress
watch -n 2 tail -20 logs/training.log

# Or use monitoring script
bash scripts/monitor_gpu_training.sh
```

### Force CPU Training
```bash
export CUDA_VISIBLE_DEVICES=""  # Hide GPU
python3 scripts/train_gpu_timeout.py
```

## Test Results

### ✅ What Works
- **CPU Training**: 100% stable, 2 epochs in ~10-30 seconds (10 samples)
- **GPU Detection**: `torch.cuda.is_available()` returns True
- **Small GPU Tests**: Sometimes work with `HSA_OVERRIDE_GFX_VERSION`

### ❌ What Fails
- **GPU Training**: Hangs after 15+ seconds on tensor operations
- **Large Models**: Even small models cause hangs on GPU
- **Sustained GPU Use**: System becomes unresponsive

### Tested Configurations

| Configuration | Result | Notes |
|--------------|--------|-------|
| CPU only | ✅ Works | Stable, slower but reliable |
| GPU with timeout | ⚠️ Detects hang | Falls back to CPU |
| GPU without safeguards | ❌ System crash | Requires reboot |
| Small model + GPU | ❌ Still hangs | Issue is driver-level |

## Recommendations

### For Development (Current System)
1. **Use CPU training** with the safeguarded scripts
2. **Enable multi-processing** for faster CPU training
3. **Use cloud GPU** for production training (Vast.ai, Colab, etc.)

### For Production
1. **Get RDNA2+ GPU** (RX 6000/7000 series) with ROCm 6.0+ support
2. **Use NVIDIA GPU** (better PyTorch support)
3. **Use cloud instances** with supported hardware

### Quick Commands

```bash
# Safe training (auto-detects GPU, falls back to CPU)
python3 scripts/train_gpu_timeout.py

# Force CPU (fastest on current hardware)
CUDA_VISIBLE_DEVICES="" python3 scripts/train_gpu_timeout.py

# Monitor background training
bash scripts/monitor_gpu_training.sh

# Stop training
pkill -f train_gpu
```

## Technical Details

### Why GPU Hangs
1. **Driver Mismatch**: ROCm 6.2 drivers don't properly support gfx1010
2. **Missing Kernels**: Optimized kernels (hipBLASLt, etc.) missing for Navi 10
3. **Memory Management**: ROCm memory allocator unstable on unsupported GPUs
4. **Instruction Set**: Some GPU instructions cause hangs on RDNA1

### Why Timeout Works
- **Process Isolation**: Hung GPU process doesn't freeze main program
- **Clean Termination**: Parent process can kill child process
- **No System Impact**: Hung process is contained and terminated
- **Graceful Fallback**: Training continues on CPU

## Files

### Training Scripts
- `scripts/train_gpu_timeout.py` - **Recommended**: Timeout-protected training
- `scripts/train_gpu_quick.py` - Quick test (may hang)
- `scripts/train_gpu_safeguarded.py` - Full safeguards (slower)

### Monitoring
- `scripts/monitor_gpu_training.sh` - Check training status
- `logs/gpu_*.log` - Training logs

### Documentation
- `docs/GPU_TRAINING_STATUS.md` - Initial investigation
- `docs/ROCM_GPU_ANALYSIS.md` - Deep technical analysis
- `docs/GPU_SAFEGUARDS.md` - This file

## Summary

**Bottom Line:** 
- GPU training crashes due to hardware incompatibility (RX 5700 XT not supported in ROCm 6.0+)
- Safeguards prevent crashes by detecting hangs and falling back to CPU
- CPU training is stable and recommended for current hardware
- For GPU training, use cloud instances or upgrade to supported hardware

**Status:** ✅ Problem solved with automatic CPU fallback

