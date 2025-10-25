---
applyTo: '**'
---

# ROCm SDK Usage Rule

## CRITICAL RULE: Always Use ROCm SDK for GPU Training

When training models on AMD GPU, **ALWAYS** use the ROCm SDK Python and PyTorch:

### Required Environment Setup

```bash
# ROCm SDK paths
export ROCM_PATH="/opt/rocm_sdk_612"
export LD_LIBRARY_PATH="/opt/rocm_sdk_612/lib:/opt/rocm_sdk_612/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/rocm_sdk_612/lib/python3.11/site-packages:$PYTHONPATH"
export PATH="/opt/rocm_sdk_612/bin:$PATH"

# ROCm GPU configuration
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH="gfx1010"

# Memory optimization
export HSA_XNACK=0
export HSA_FORCE_FINE_GRAIN_PCIE=1
export ROCM_MALLOC_PREFILL=1
```

### Python Executable

**ALWAYS use**: `/opt/rocm_sdk_612/bin/python3`

**NEVER use**: `/usr/bin/python3` or system Python

### Why This Matters

The ROCm SDK includes:
- PyTorch 2.4.1 with ROCm/HIP support
- Optimized memory management for AMD GPUs
- Proper kernel compilation for gfx1010 (RX 5600 XT)
- MIGraphX acceleration libraries

Using system Python bypasses these optimizations and can cause:
- Memory access violations (HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION)
- Suboptimal GPU utilization
- Missing ROCm-specific features

### Verification

Before training, always verify:

```bash
/opt/rocm_sdk_612/bin/python3 -c "import torch; print(torch.__version__); print(torch.version.hip); print(torch.cuda.is_available())"
```

Should output:
- PyTorch: 2.4.1 (or similar)
- ROCm/HIP: 6.1.2 (or similar)
- CUDA Available: True

### Training Script Template

```bash
#!/bin/bash
# Setup ROCm SDK environment
export ROCM_PATH="/opt/rocm_sdk_612"
export LD_LIBRARY_PATH="/opt/rocm_sdk_612/lib:/opt/rocm_sdk_612/lib/python3.11/site-packages/torch/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/rocm_sdk_612/lib/python3.11/site-packages:$PYTHONPATH"
export PATH="/opt/rocm_sdk_612/bin:$PATH"
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export PYTORCH_ROCM_ARCH="gfx1010"
export HSA_XNACK=0
export HSA_FORCE_FINE_GRAIN_PCIE=1

# Run training with ROCm SDK Python
/opt/rocm_sdk_612/bin/python3 your_training_script.py
```

## Hardware Details

- GPU: AMD Radeon RX 5600 XT
- Architecture: gfx1010 (RDNA 1.0)
- VRAM: 6.43 GB
- ROCm SDK: 6.1.2
- PyTorch: 2.4.1+rocm6.1.2

## Memory Guidelines

With 6.43GB VRAM:
- Batch size 16: EnhancedEEGNeX (254K params) - May exceed memory
- Batch size 8: EnhancedEEGNeX (254K params) - Should fit
- Batch size 4: EnhancedEEGNeX (254K params) - Safe
- Batch size 16: EEGNeX (62K params) - Safe

Always monitor GPU memory usage during training initialization.
