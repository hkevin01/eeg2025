# ROCm Troubleshooting Guide

## HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION

This error indicates the GPU attempted to access memory beyond its legal address space. Based on extensive research from AMD ROCm community issues, PyTorch forums, and official documentation, here's what we know:

### Root Causes

1. **Physical Address Limitation**
   - AMD Radeon RX 5600 XT (GFX1010) has specific memory aperture constraints
   - ROCm may attempt to access memory regions beyond supported address space
   - Related to PCIe BAR (Base Address Register) configuration

2. **Driver/Hardware Incompatibility**
   - Some hardware+driver combinations have known issues (ROCm GitHub #2153)
   - Resolved in ROCm 5.5-5.6+ for certain cards, but RX 5600 XT may still have issues
   - BIOS MMIO aperture settings can interfere

3. **Memory Management**
   - PyTorch ROCm caching allocator may allocate beyond safe limits
   - Workspace allocation for hipBLAS/rocBLAS can trigger violations

### Diagnostic Steps

#### 1. Check Environment Variables

```bash
# Verify ROCm environment
echo "PYTORCH_ROCM_ARCH: $PYTORCH_ROCM_ARCH"
echo "HSA_OVERRIDE_GFX_VERSION: $HSA_OVERRIDE_GFX_VERSION"
echo "HIP_VISIBLE_DEVICES: $HIP_VISIBLE_DEVICES"
```

**For RX 5600 XT (gfx1010):**
```bash
export PYTORCH_ROCM_ARCH=gfx1010
export HSA_OVERRIDE_GFX_VERSION=10.1.0
```

#### 2. Enable Debug Logging

```bash
# Enable HSA/KMT debug output
export HSAKMT_DEBUG_LEVEL=7  # Maximum verbosity

# Disable PyTorch memory caching (debugging only)
export PYTORCH_NO_HIP_MEMORY_CACHING=1
export PYTORCH_NO_CUDA_MEMORY_CACHING=1  # Also supported for compatibility
```

#### 3. Check GPU Firmware & Driver

```bash
# Dump firmware versions
sudo cat /sys/kernel/debug/dri/1/amdgpu_firmware_info

# Check kernel version
uname -a

# Verify ROCm version
rocm-smi --showproductname
```

#### 4. Memory Allocation Limits

```bash
# Check available VRAM
rocm-smi --showmeminfo vram

# Monitor real-time usage
watch -n 1 rocm-smi
```

### Mitigation Strategies

#### Strategy 1: Reduce Batch Size & Memory Pressure

```python
# Start very conservative
batch_size = 8  # Instead of 64
num_workers = 1  # Reduce parallel data loading

# Limit memory fraction
torch.cuda.set_per_process_memory_fraction(0.7, device=0)
```

#### Strategy 2: Disable Memory Features

```python
# Disable pin_memory (avoids CUDA pinned memory pool)
train_loader = DataLoader(dataset, pin_memory=False)

# Disable persistent workers
train_loader = DataLoader(dataset, persistent_workers=False)
```

#### Strategy 3: Disable Mixed Precision

```python
# AMP may trigger additional memory allocations
use_amp = False  # Disable automatic mixed precision
```

#### Strategy 4: Configure hipBLAS Workspace

```bash
# Force hipBLAS to avoid using workspaces
export HIPBLAS_WORKSPACE_CONFIG=:0:0
export CUBLAS_WORKSPACE_CONFIG=:0:0  # Also set for compatibility
```

#### Strategy 5: CPU Fallback (Current Implementation)

Our training script automatically detects ROCm failures and falls back to CPU:

```python
try:
    output = model(input)
except RuntimeError as e:
    if "HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION" in str(e):
        # Switch to CPU, disable AMP, clear cache
        device = torch.device("cpu")
        model = model.cpu()
        torch.cuda.empty_cache()
```

### Hardware-Level Checks

#### PCIe Atomics Support

Some older motherboards don't properly support PCIe atomics required by ROCm:

```bash
# Check if system supports PCIe atomics
lspci -vv | grep -i atomic
```

#### IOMMU Configuration

Enabling IOMMU can sometimes help with address translation:

```bash
# Check IOMMU status
dmesg | grep -i iommu

# Enable in GRUB (add to /etc/default/grub):
GRUB_CMDLINE_LINUX_DEFAULT="... iommu=pt amd_iommu=on"
sudo update-grub
```

#### BIOS Settings

- **Above 4G Decoding**: Should be ENABLED
- **Re-Size BAR Support**: Try DISABLED (some cards have issues)
- **IOMMU**: Try both AUTO and ENABLED

### Known Limitations

Based on research from multiple sources:

1. **RX 5600 XT Not Officially Supported**
   - AMD's official ROCm support targets server GPUs (MI series) and some high-end consumer cards
   - Consumer RDNA1 cards (like RX 5600 XT) have limited/experimental support

2. **ROCm Version Matters**
   - ROCm 5.5+ fixed some memory aperture issues for Radeon VII
   - RX 5600 XT may require ROCm 5.7+ or 6.0+

3. **Driver Quality**
   - Some users report returning hardware as "faulty" when issue was driver-related
   - Try both `amdgpu-pro` and open-source `amdgpu` drivers

### Recommended Workflow

For stable training on RX 5600 XT with ROCm:

1. **Start with CPU baseline** - Verify model/data pipeline works
2. **Try GPU with minimal config** - batch_size=8, no AMP, workers=1
3. **If GPU fails, auto-fallback to CPU** - Our current implementation
4. **Monitor for success** - Check if any batches complete on GPU
5. **Incrementally increase** - If stable, slowly increase batch size

### When to Use CPU vs GPU

| Scenario | Recommendation |
|----------|----------------|
| Quick prototyping | CPU (more stable) |
| Small models (<1M params) | CPU often competitive |
| Large batches needed | GPU (if stable) |
| Long training runs | Start GPU, fallback to CPU on first error |
| Production inference | CPU (more reliable for edge deployment) |

### References

- [ROCm GitHub Issue #2153](https://github.com/ROCm/ROCm/issues/2153) - Radeon VII aperture violation
- [PyTorch ROCm Documentation](https://pytorch.org/docs/stable/notes/hip.html)
- [AMD ROCm System Debugging](https://rocm.docs.amd.com/en/latest/how-to/system-debugging.html)
- [PyTorch Forums - CUDA OOM Fallback](https://discuss.pytorch.org/t/cuda-out-of-memory-run-time-error-handling-fall-back-to-cpu-possibility/153749)
- [AskUbuntu - ROCm HSA Errors](https://askubuntu.com/questions/1258945/amd-rocm-issues-hsa-status-error-out-of-resources-and-more)

### Support & Community

- **ROCm GitHub**: https://github.com/ROCm/ROCm/issues
- **PyTorch Forums**: https://discuss.pytorch.org/c/rocm
- **AMD Developer Discord**: https://discord.com/invite/amd-dev
