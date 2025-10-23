"""
CPU-Only EEGNeX Wrapper
=======================
Force EEGNeX to always run on CPU to avoid AMD gfx1030 GPU issues.
"""

import torch
import torch.nn as nn
from braindecode.models import EEGNeX as BraindecodeEEGNeX

class EEGNeXCPUOnly(nn.Module):
    """
    Wrapper around braindecode EEGNeX that forces CPU execution.
    
    This avoids the AMD gfx1030 memory aperture violation issue
    by keeping the model on CPU even when training script uses GPU.
    """
    
    def __init__(self, **kwargs):
        super().__init__()
        self.model = BraindecodeEEGNeX(**kwargs)
        self.model.cpu()  # Force CPU
        
    def forward(self, x):
        # Move input to CPU, compute, return on same device as input
        input_device = x.device
        x_cpu = x.cpu()
        y_cpu = self.model(x_cpu)
        return y_cpu.to(input_device)
    
    def to(self, device):
        # Ignore device moves - always stay on CPU
        return self
    
    def cuda(self, device=None):
        # Ignore cuda() calls
        return self
    
    def cpu(self):
        # Already on CPU
        return self


if __name__ == "__main__":
    print("Testing CPU-Only EEGNeX Wrapper")
    print("=" * 60)
    
    model = EEGNeXCPUOnly(
        n_chans=129,
        n_outputs=1,
        n_times=200,
        sfreq=100,
    )
    
    print(f"✅ Model created: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    x = torch.randn(2, 129, 200)
    y = model(x)
    print(f"✅ Forward pass: {x.shape} -> {y.shape}")
    
    # Try moving to GPU (should be ignored)
    if torch.cuda.is_available():
        model_gpu = model.cuda()
        x_gpu = torch.randn(2, 129, 200, device='cuda')
        y_gpu = model_gpu(x_gpu)
        print(f"✅ GPU input handled (computed on CPU): {x_gpu.shape} -> {y_gpu.shape}")
        print("   Model stayed on CPU, avoiding gfx1030 issues!")
    
    print("=" * 60)
