#!/usr/bin/env python3
"""
Incremental GPU Testing with OpenNLP-GPU patterns
Step-by-step verification with detailed monitoring
"""

import os
import sys
from pathlib import Path
import time

# Set environment variables FIRST (from OpenNLP-GPU)
print("=" * 70)
print("STEP 1: Setting Environment Variables (OpenNLP-GPU Pattern)")
print("=" * 70)

os.environ['ROCM_PATH'] = '/opt/rocm'
os.environ['HIP_PATH'] = '/opt/rocm'
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HIP_PLATFORM'] = 'amd'
os.environ['PYTORCH_HIP_ALLOC_CONF'] = 'max_split_size_mb:128'

print("✅ ROCM_PATH:", os.environ.get('ROCM_PATH'))
print("✅ HIP_PATH:", os.environ.get('HIP_PATH'))
print("✅ HSA_OVERRIDE_GFX_VERSION:", os.environ.get('HSA_OVERRIDE_GFX_VERSION'))
print("✅ HIP_PLATFORM:", os.environ.get('HIP_PLATFORM'))
print("✅ PYTORCH_HIP_ALLOC_CONF:", os.environ.get('PYTORCH_HIP_ALLOC_CONF'))

# Import torch
print("\n" + "=" * 70)
print("STEP 2: Import PyTorch")
print("=" * 70)
import torch
import warnings
warnings.filterwarnings('ignore', message='.*hipBLASLt.*')
print("✅ PyTorch imported successfully")
print(f"   Version: {torch.__version__}")

# Check CUDA availability
print("\n" + "=" * 70)
print("STEP 3: Check GPU Availability")
print("=" * 70)
if torch.cuda.is_available():
    print("✅ GPU detected by PyTorch")
    print(f"   Device count: {torch.cuda.device_count()}")
    print(f"   Device 0: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"   Total memory: {props.total_memory / (1024**3):.2f} GB")
    print(f"   Compute capability: {props.major}.{props.minor}")
else:
    print("❌ No GPU detected - stopping test")
    sys.exit(1)

# Test basic tensor operations
print("\n" + "=" * 70)
print("STEP 4: Test Basic Tensor Operations")
print("=" * 70)
try:
    device = torch.device('cuda:0')
    print(f"Creating 10x10 tensor on {device}...")
    t1 = torch.randn(10, 10, device=device)
    print(f"✅ Tensor created: {t1.shape}, device={t1.device}")
    
    print("Testing matrix multiplication...")
    result = t1 @ t1.T
    print(f"✅ Matmul successful: {result.shape}")
    
    print("Clearing cache...")
    del t1, result
    torch.cuda.empty_cache()
    print("✅ Cache cleared")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test medium tensors
print("\n" + "=" * 70)
print("STEP 5: Test Medium Tensor Operations")
print("=" * 70)
try:
    print(f"Creating 100x100 tensor on {device}...")
    t2 = torch.randn(100, 100, device=device)
    print(f"✅ Tensor created: {t2.shape}")
    
    print("Testing matrix multiplication...")
    result2 = t2 @ t2.T
    print(f"✅ Matmul successful: {result2.shape}")
    
    mem_alloc = torch.cuda.memory_allocated() / (1024**2)
    mem_reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"   GPU Memory: {mem_alloc:.2f} MB allocated, {mem_reserved:.2f} MB reserved")
    
    print("Clearing cache...")
    del t2, result2
    torch.cuda.empty_cache()
    print("✅ Cache cleared")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test neural network layer
print("\n" + "=" * 70)
print("STEP 6: Test Neural Network Layer")
print("=" * 70)
try:
    import torch.nn as nn
    
    print("Creating linear layer...")
    layer = nn.Linear(100, 100).to(device)
    print(f"✅ Layer created on {device}")
    
    print("Creating input tensor...")
    input_tensor = torch.randn(32, 100, device=device)
    print(f"✅ Input: {input_tensor.shape}")
    
    print("Forward pass...")
    output = layer(input_tensor)
    print(f"✅ Output: {output.shape}")
    
    print("Backward pass...")
    loss = output.sum()
    loss.backward()
    print(f"✅ Backward pass successful")
    
    mem_alloc = torch.cuda.memory_allocated() / (1024**2)
    mem_reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"   GPU Memory: {mem_alloc:.2f} MB allocated, {mem_reserved:.2f} MB reserved")
    
    print("Clearing...")
    del layer, input_tensor, output, loss
    torch.cuda.empty_cache()
    print("✅ Cleared")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test small transformer
print("\n" + "=" * 70)
print("STEP 7: Test Small Transformer")
print("=" * 70)
try:
    print("Creating transformer encoder layer...")
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=64,
        nhead=4,
        dim_feedforward=256,
        dropout=0.1,
        batch_first=True
    ).to(device)
    print("✅ Transformer layer created")
    
    print("Creating input (batch=2, seq=100, dim=64)...")
    input_tensor = torch.randn(2, 100, 64, device=device)
    print(f"✅ Input created: {input_tensor.shape}")
    
    print("Forward pass...")
    output = encoder_layer(input_tensor)
    print(f"✅ Output: {output.shape}")
    
    print("Backward pass...")
    loss = output.sum()
    loss.backward()
    print(f"✅ Backward pass successful")
    
    mem_alloc = torch.cuda.memory_allocated() / (1024**2)
    mem_reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"   GPU Memory: {mem_alloc:.2f} MB allocated, {mem_reserved:.2f} MB reserved")
    
    print("Clearing...")
    del encoder_layer, input_tensor, output, loss
    torch.cuda.empty_cache()
    print("✅ Cleared")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test with actual EEG-sized data
print("\n" + "=" * 70)
print("STEP 8: Test EEG-Sized Data")
print("=" * 70)
try:
    print("Creating EEG-sized tensor (batch=4, channels=129, time=1000)...")
    eeg_data = torch.randn(4, 129, 1000, device=device)
    print(f"✅ EEG tensor created: {eeg_data.shape}")
    
    print("Creating projection layer (129 -> 128)...")
    proj = nn.Linear(129, 128).to(device)
    
    print("Transposing and projecting...")
    eeg_transposed = eeg_data.transpose(1, 2)  # (4, 1000, 129)
    projected = proj(eeg_transposed)  # (4, 1000, 128)
    print(f"✅ Projected: {projected.shape}")
    
    mem_alloc = torch.cuda.memory_allocated() / (1024**2)
    mem_reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"   GPU Memory: {mem_alloc:.2f} MB allocated, {mem_reserved:.2f} MB reserved")
    
    print("Clearing...")
    del eeg_data, proj, eeg_transposed, projected
    torch.cuda.empty_cache()
    print("✅ Cleared")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test full mini model
print("\n" + "=" * 70)
print("STEP 9: Test Mini Foundation Model")
print("=" * 70)
try:
    class MiniFoundation(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(129, 64)
            self.pos = nn.Parameter(torch.randn(1, 1000, 64) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=256,
                dropout=0.1, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.classifier = nn.Linear(64, 2)
        
        def forward(self, x):
            x = x.transpose(1, 2)  # (B, T, C)
            x = self.proj(x)
            x = x + self.pos[:, :x.size(1), :]
            x = self.transformer(x)
            x = x.mean(dim=1)
            return self.classifier(x)
    
    print("Creating mini model...")
    model = MiniFoundation().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model created: {n_params:,} parameters")
    
    print("Creating input batch (4, 129, 1000)...")
    batch = torch.randn(4, 129, 1000, device=device)
    print(f"✅ Batch created")
    
    print("Forward pass...")
    output = model(batch)
    print(f"✅ Output: {output.shape}")
    
    print("Backward pass...")
    loss = output.sum()
    loss.backward()
    print(f"✅ Backward successful")
    
    mem_alloc = torch.cuda.memory_allocated() / (1024**2)
    mem_reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"   GPU Memory: {mem_alloc:.2f} MB allocated, {mem_reserved:.2f} MB reserved")
    
    print("Clearing...")
    del model, batch, output, loss
    torch.cuda.empty_cache()
    print("✅ Cleared")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test optimizer
print("\n" + "=" * 70)
print("STEP 10: Test Optimizer")
print("=" * 70)
try:
    print("Creating model and optimizer...")
    model = MiniFoundation().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    print("✅ Setup complete")
    
    print("Training 5 mini-batches...")
    model.train()
    for i in range(5):
        batch = torch.randn(4, 129, 1000, device=device)
        target = torch.randint(0, 2, (4,), device=device)
        
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        print(f"   Batch {i+1}/5: loss={loss.item():.4f}")
    
    print("✅ Training loop successful!")
    
    mem_alloc = torch.cuda.memory_allocated() / (1024**2)
    mem_reserved = torch.cuda.memory_reserved() / (1024**2)
    print(f"   GPU Memory: {mem_alloc:.2f} MB allocated, {mem_reserved:.2f} MB reserved")
    
    print("Clearing...")
    del model, optimizer, criterion, batch, target, output, loss
    torch.cuda.empty_cache()
    print("✅ Cleared")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "=" * 70)
print("🎉 ALL TESTS PASSED!")
print("=" * 70)
print("✅ Environment variables set")
print("✅ GPU detected and accessible")
print("✅ Basic tensor operations")
print("✅ Medium tensor operations")
print("✅ Neural network layers")
print("✅ Transformer layers")
print("✅ EEG-sized data")
print("✅ Mini foundation model")
print("✅ Training loop with optimizer")
print("\n🚀 GPU is ready for full training!")
print("=" * 70)
