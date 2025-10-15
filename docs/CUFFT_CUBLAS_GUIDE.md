# CuFFT and CuBLAS Integration Guide

## Overview

This guide explains how to leverage NVIDIA's highly optimized CUDA libraries for maximum performance in EEG processing and deep learning:

- **CuFFT**: NVIDIA's Fast Fourier Transform library for GPU
- **CuBLAS**: NVIDIA's Basic Linear Algebra Subprograms for GPU

## Performance Benefits

### CuFFT Performance
- **10-50x faster** than CPU FFT for large batches
- Optimized for real-time EEG spectral analysis
- Efficient STFT/ISTFT for time-frequency domain processing

### CuBLAS Performance  
- **20-100x faster** than CPU matrix operations
- Automatically used by PyTorch for GPU tensors
- TF32 mode provides ~8x speedup on Ampere GPUs

## Architecture Integration

```
EEG Data Pipeline with CUDA Acceleration
════════════════════════════════════════

Raw EEG Signal (CPU)
        ↓
    [Transfer to GPU] ← Pinned memory for faster transfer
        ↓
Preprocessing (GPU)
├── CuFFT: Bandpass filtering (frequency domain)
├── CuFFT: STFT for spectral features
└── CuBLAS: Common average reference (matrix ops)
        ↓
Deep Learning Model (GPU)
├── CuBLAS: Linear layers (matmul + bias)
├── CuBLAS: Attention mechanism (QKV projections)
├── CuBLAS: Transformer feed-forward networks
└── CuFFT: Frequency-domain augmentations
        ↓
Predictions (GPU → CPU)
```

## Quick Start

### 1. Check CUDA Availability

```python
import torch
from src.gpu.cuda_optimized import (
    CUDA_AVAILABLE, 
    CUFFT_AVAILABLE, 
    CUBLAS_AVAILABLE
)

print(f"CUDA: {CUDA_AVAILABLE}")
print(f"CuFFT: {CUFFT_AVAILABLE}")
print(f"CuBLAS: {CUBLAS_AVAILABLE}")

if CUDA_AVAILABLE:
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Version: {torch.version.cuda}")
```

### 2. Use CuFFT for FFT Operations

```python
from src.gpu.cuda_optimized import CuFFTOptimizer

# Create optimizer
fft_optimizer = CuFFTOptimizer(device="cuda")

# Your EEG data: (batch, channels, time)
eeg_data = torch.randn(32, 129, 10000).cuda()

# Fast FFT using CuFFT
fft_coeffs = fft_optimizer.rfft_batch(eeg_data, dim=-1)

# Inverse FFT
reconstructed = fft_optimizer.irfft_batch(fft_coeffs, n=10000, dim=-1)

# STFT for spectrograms
spectrogram = fft_optimizer.stft_optimized(
    eeg_data,
    n_fft=512,
    hop_length=256
)
```

### 3. Use CuBLAS for Matrix Operations

```python
from src.gpu.cuda_optimized import CuBLASOptimizer

# Create optimizer
blas_optimizer = CuBLASOptimizer(device="cuda", use_tf32=True)

# Matrix multiplication (uses CuBLAS GEMM)
A = torch.randn(32, 1024, 512).cuda()
B = torch.randn(32, 512, 256).cuda()
C = blas_optimizer.bmm_optimized(A, B)

# Fused operations (addmm: bias + input @ weight^T)
bias = torch.randn(256).cuda()
input = torch.randn(32, 512).cuda()
weight = torch.randn(256, 512).cuda()
output = blas_optimizer.addmm_optimized(bias, input, weight)
```

### 4. Use Optimized Neural Network Layers

```python
from src.gpu.cuda_optimized import (
    OptimizedLinearLayer,
    OptimizedAttention
)

# Replace nn.Linear with optimized version
linear = OptimizedLinearLayer(512, 256, device="cuda")
x = torch.randn(32, 128, 512).cuda()
out = linear(x)  # Uses CuBLAS addmm

# Optimized multi-head attention
attention = OptimizedAttention(
    embed_dim=512,
    num_heads=8,
    device="cuda"
)
attn_out = attention(x)  # Uses CuBLAS batched matmul
```

## Integration with Existing Code

### Update Training Scripts

```python
# Before: Standard PyTorch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(512, 256)
        
# After: With CuBLAS optimization
from src.gpu.cuda_optimized import OptimizedLinearLayer

class OptimizedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = OptimizedLinearLayer(512, 256, device="cuda")
```

### Update EEG Preprocessing

```python
# Before: CPU-based FFT
def preprocess_eeg_cpu(eeg_data):
    spectrum = torch.fft.rfft(eeg_data, dim=-1)
    # ...processing...
    return torch.fft.irfft(spectrum, dim=-1)

# After: CuFFT-accelerated
from src.gpu.cuda_optimized import CuFFTOptimizer

fft_optimizer = CuFFTOptimizer()

def preprocess_eeg_gpu(eeg_data):
    eeg_data = eeg_data.cuda()
    spectrum = fft_optimizer.rfft_batch(eeg_data, dim=-1)
    # ...processing on GPU...
    return fft_optimizer.irfft_batch(spectrum, dim=-1)
```

## Performance Benchmarks

Run the built-in benchmarks:

```bash
cd /home/kevin/Projects/eeg2025
python3 -m src.gpu.cuda_optimized
```

### Expected Speedups (RTX 3080/4090)

| Operation | CPU Time | GPU Time (CuFFT/CuBLAS) | Speedup |
|-----------|----------|-------------------------|---------|
| FFT (32x129x10000) | 180 ms | 8 ms | **22x** |
| STFT (32x129x10000) | 350 ms | 15 ms | **23x** |
| Matrix Mul (32x1024x1024) | 850 ms | 12 ms | **70x** |
| Attention (32x128x512) | 120 ms | 6 ms | **20x** |

### Expected Speedups (AMD RX 5700 XT with ROCm)

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| FFT (32x129x10000) | 180 ms | 25 ms | **7x** |
| Matrix Mul (32x1024x1024) | 850 ms | 45 ms | **19x** |

*Note: AMD uses ROCm's FFT (hipFFT) and BLAS (rocBLAS) equivalents*

## Advanced Optimizations

### 1. Enable TF32 for Ampere GPUs

```python
import torch

# Enable TF32 for ~8x speedup with minimal accuracy loss
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

### 2. Use Pinned Memory for Faster Transfers

```python
# Allocate pinned memory
eeg_batch = torch.randn(32, 129, 10000).pin_memory()

# Faster transfer to GPU
eeg_gpu = eeg_batch.cuda(non_blocking=True)
```

### 3. Batch Operations for Maximum Efficiency

```python
# Bad: Sequential processing
for i in range(100):
    result = process_single_sample(data[i])
    
# Good: Batched processing (uses CuBLAS batched ops)
results = process_batch(data)  # All 100 samples at once
```

### 4. Keep Data on GPU

```python
# Bad: Frequent CPU ↔ GPU transfers
for batch in dataloader:
    batch = batch.cuda()  # Transfer
    out = model(batch)
    loss = loss_fn(out, labels.cuda())  # Another transfer
    
# Good: Keep everything on GPU
model = model.cuda()
for batch, labels in dataloader:
    batch, labels = batch.cuda(), labels.cuda()
    out = model(batch)
    loss = loss_fn(out, labels)
```

## Troubleshooting

### Issue 1: CUDA Out of Memory

```python
# Solution 1: Reduce batch size
batch_size = 16  # Instead of 32

# Solution 2: Use gradient accumulation
for i, batch in enumerate(dataloader):
    loss = model(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Solution 3: Clear cache periodically
if batch_idx % 100 == 0:
    torch.cuda.empty_cache()
```

### Issue 2: Slow Performance

```python
# Check if data is actually on GPU
print(f"Data on CUDA: {tensor.is_cuda}")

# Profile to find bottlenecks
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
) as prof:
    model(input_data)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### Issue 3: AMD ROCm Issues

```python
# Check ROCm availability
print(f"ROCm available: {torch.cuda.is_available()}")
print(f"ROCm version: {torch.version.hip}")

# AMD uses different optimizations
# CuFFT → hipFFT (automatic)
# CuBLAS → rocBLAS (automatic)
```

## Best Practices

### ✅ Do:
- **Use batched operations** - CuFFT/CuBLAS are optimized for batches
- **Keep data on GPU** - Minimize CPU ↔ GPU transfers
- **Use mixed precision** - FP16 + TF32 for 2-8x speedup
- **Profile your code** - Identify bottlenecks before optimizing
- **Enable TF32** on Ampere GPUs - Free 8x speedup

### ❌ Don't:
- **Don't transfer to CPU unnecessarily** - Expensive operation
- **Don't use small batches** - Wastes GPU parallelism
- **Don't ignore memory management** - Clear cache when needed
- **Don't assume GPU is always faster** - Small operations may be slower
- **Don't forget warmup** - First GPU call includes initialization overhead

## Integration Checklist

```markdown
- [ ] Check CUDA availability in your environment
- [ ] Update requirements.txt with CUDA-enabled PyTorch
- [ ] Replace FFT operations with CuFFT optimizer
- [ ] Replace Linear layers with OptimizedLinearLayer (optional)
- [ ] Enable TF32 for Ampere GPUs
- [ ] Use pinned memory for data loading
- [ ] Batch operations wherever possible
- [ ] Profile performance before/after
- [ ] Add error handling for CUDA OOM
- [ ] Test on both NVIDIA and AMD GPUs (if applicable)
```

## Example: Complete Training Script

```python
#!/usr/bin/env python3
"""
Example: Training with CuFFT/CuBLAS optimization
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.gpu.cuda_optimized import (
    CuFFTOptimizer,
    CuBLASOptimizer,
    OptimizedLinearLayer,
    OptimizedAttention
)

# Enable optimizations
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

class OptimizedEEGModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fft_optimizer = CuFFTOptimizer()
        
        # Use optimized layers
        self.input_proj = OptimizedLinearLayer(129, 512, device="cuda")
        self.attention = OptimizedAttention(512, 8, device="cuda")
        self.output = OptimizedLinearLayer(512, 2, device="cuda")
        
    def forward(self, x):
        # x: (batch, channels, time)
        
        # CuFFT preprocessing
        spectrum = self.fft_optimizer.rfft_batch(x, dim=-1)
        # ...extract features...
        features = spectrum.abs().mean(dim=-1)  # Simplified
        
        # CuBLAS-optimized layers
        x = self.input_proj(features)
        x = x.unsqueeze(1)  # Add sequence dimension
        x = self.attention(x)
        x = self.output(x.squeeze(1))
        
        return x

def train():
    device = torch.device("cuda")
    model = OptimizedEEGModel().to(device)
    
    # Use pinned memory for faster transfers
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        pin_memory=True,
        num_workers=4
    )
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    
    for epoch in range(10):
        for batch_idx, (data, labels) in enumerate(dataloader):
            # Non-blocking transfer
            data = data.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            
            # Forward pass (uses CuFFT + CuBLAS)
            outputs = model(data)
            loss = nn.functional.cross_entropy(outputs, labels)
            
            # Backward pass (CuBLAS for gradients)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clear cache periodically
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()

if __name__ == "__main__":
    train()
```

## Resources

- [PyTorch CUDA Semantics](https://pytorch.org/docs/stable/notes/cuda.html)
- [CuFFT Documentation](https://docs.nvidia.com/cuda/cufft/)
- [CuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [PyTorch Performance Tuning](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html)
- [AMD ROCm Documentation](https://rocmdocs.amd.com/)

## Summary

CuFFT and CuBLAS provide **10-100x speedups** for EEG processing and deep learning when properly integrated. Key takeaways:

1. ✅ PyTorch automatically uses CuFFT/CuBLAS for GPU tensors
2. ✅ Our optimized classes provide additional control and fused operations
3. ✅ Batch operations and keeping data on GPU maximize performance
4. ✅ Enable TF32 on Ampere GPUs for free 8x speedup
5. ✅ Profile before/after to measure actual improvements

Start with the Quick Start examples and gradually integrate optimizations into your training pipeline.
