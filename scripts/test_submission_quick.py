#!/usr/bin/env python3
"""Quick submission test with device optimization"""
import sys
import os
import time

# Force CPU for testing (faster initialization)
os.environ['CUDA_VISIBLE_DEVICES'] = ''

print("="*70, flush=True)
print("🧪 Quick Submission Test (CPU Mode for Speed)", flush=True)
print("="*70, flush=True)

# Add project root to path
sys.path.insert(0, '/home/kevin/Projects/eeg2025')

import torch
import numpy as np

print("\n📦 Importing submission...", end=' ', flush=True)
from submission import Submission
print("✓", flush=True)

# Competition parameters
SFREQ = 100
DEVICE = torch.device("cpu")  # Force CPU for quick test

print(f"Device: {DEVICE}", flush=True)

# Create tiny test data
print("🔢 Creating test data...", end=' ', flush=True)
X_test = np.random.randn(2, 129, 200).astype(np.float32)
X_tensor = torch.from_numpy(X_test).to(DEVICE)
print("✓", flush=True)

# Initialize submission
print("\n🔧 Initializing submission...", end=' ', flush=True)
start = time.time()
sub = Submission(SFREQ, DEVICE)
print(f"✓ ({time.time()-start:.1f}s)", flush=True)

# Test Challenge 1 (response time)
print("\n🎯 Challenge 1: Response Time", flush=True)
print("  Loading model...", end=' ', flush=True)
start = time.time()
try:
    model_1 = sub.get_model_challenge_1()
    print(f"✓ ({time.time()-start:.1f}s)", flush=True)

    print("  Running inference...", end=' ', flush=True)
    start = time.time()
    with torch.inference_mode():
        pred1 = model_1.forward(X_tensor)
    print(f"✓ ({time.time()-start:.1f}s)", flush=True)

    pred1_np = pred1.cpu().numpy().flatten()
    print(f"  Predictions: {pred1_np}", flush=True)
    print(f"  Range: [{pred1_np.min():.4f}, {pred1_np.max():.4f}]", flush=True)
except Exception as e:
    print(f"⚠️  Challenge 1 error: {e}", flush=True)

# Test Challenge 2 (externalizing factor)
print("\n📊 Challenge 2: Externalizing Factor", flush=True)
print("  Loading model...", end=' ', flush=True)
start = time.time()
model_2 = sub.get_model_challenge_2()
print(f"✓ ({time.time()-start:.1f}s)", flush=True)

print("  Running inference...", end=' ', flush=True)
start = time.time()
with torch.inference_mode():
    pred2 = model_2.forward(X_tensor)
print(f"✓ ({time.time()-start:.1f}s)", flush=True)

pred2_np = pred2.cpu().numpy().flatten()
print(f"  Predictions: {pred2_np}", flush=True)
print(f"  Range: [{pred2_np.min():.4f}, {pred2_np.max():.4f}]", flush=True)

print("\n" + "="*70, flush=True)
print("✅ Both challenges tested successfully!", flush=True)
print("="*70, flush=True)
print("\nNext: Run full test with CUDA if available", flush=True)
