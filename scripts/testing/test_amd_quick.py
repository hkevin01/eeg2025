#!/usr/bin/env python3
"""Quick AMD GPU validation"""
import sys
import os
from pathlib import Path

# Suppress hipBLASLt warning BEFORE importing torch
os.environ['PYTORCH_ROCBLAS_ALLOW_FALLBACK'] = '1'

import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gpu.enhanced_gpu_optimizer import get_enhanced_optimizer
from models.enhanced_gpu_layers import create_enhanced_eeg_model

print("="*80)
print("🚀 AMD RX 5600 XT Quick Validation")
print("="*80)

# Test GPU optimizer
gpu_opt = get_enhanced_optimizer()
print(f"✅ Platform: {gpu_opt.platform}")
print(f"✅ Device: {gpu_opt.get_optimal_device('general')}")

# Test model creation
model = create_enhanced_eeg_model(
    n_channels=129,
    num_classes=1,
    d_model=64,  # Smaller for quick test
    n_heads=4,
    n_layers=2,
    use_enhanced_ops=True
)

device = gpu_opt.get_optimal_device("transformer")
model = model.to(device)

print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

# Test forward pass
test_input = torch.randn(2, 129, 500).to(device)
output = model(test_input)

print(f"✅ Forward pass successful: {test_input.shape} → {output.shape}")
print(f"✅ No crashes or hangs!")
print("="*80)
print("🎉 AMD RX 5600 XT optimization validated!")
