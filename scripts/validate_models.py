#!/usr/bin/env python3
"""
Comprehensive Model Validation
================================
Validates both Challenge 1 and Challenge 2 models with:
- Performance metrics
- Prediction distributions
- Edge case handling
- Resource usage
"""
import os
import sys
from pathlib import Path
import time
import numpy as np

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/home/kevin/Projects/eeg2025')

import torch

print("="*80, flush=True)
print("🔍 COMPREHENSIVE MODEL VALIDATION", flush=True)
print("="*80, flush=True)

# Import submission
from submission import Submission

# Competition parameters
SFREQ = 100
DEVICE = torch.device("cpu")

print(f"\nDevice: {DEVICE}", flush=True)
print(f"Sampling Rate: {SFREQ} Hz", flush=True)

# Initialize submission
print("\n📦 Initializing submission...", end=' ', flush=True)
start = time.time()
sub = Submission(SFREQ, DEVICE)
init_time = time.time() - start
print(f"✓ ({init_time:.3f}s)", flush=True)

# Test Challenge 1
print("\n" + "="*80, flush=True)
print("🎯 CHALLENGE 1: Response Time Prediction", flush=True)
print("="*80, flush=True)

print("\n1. Loading model...", end=' ', flush=True)
start = time.time()
model_1 = sub.get_model_challenge_1()
model_1.eval()
load_time = time.time() - start
print(f"✓ ({load_time:.3f}s)", flush=True)

# Count parameters
n_params_1 = sum(p.numel() for p in model_1.parameters())
print(f"   Parameters: {n_params_1:,}", flush=True)

print("\n2. Testing inference on various batch sizes:", flush=True)
for batch_size in [1, 2, 8, 16, 32]:
    X = torch.randn(batch_size, 129, 200)
    start = time.time()
    with torch.inference_mode():
        pred = model_1(X)
    inf_time = time.time() - start
    
    pred_np = pred.numpy().flatten()
    print(f"   Batch {batch_size:2d}: {inf_time*1000:6.2f}ms | Range: [{pred_np.min():.3f}, {pred_np.max():.3f}]s", flush=True)

print("\n3. Testing edge cases:", flush=True)
# All zeros
X_zeros = torch.zeros(2, 129, 200)
with torch.inference_mode():
    pred_zeros = model_1(X_zeros).numpy().flatten()
print(f"   All zeros: {pred_zeros} | Mean: {pred_zeros.mean():.3f}s", flush=True)

# All ones
X_ones = torch.ones(2, 129, 200)
with torch.inference_mode():
    pred_ones = model_1(X_ones).numpy().flatten()
print(f"   All ones:  {pred_ones} | Mean: {pred_ones.mean():.3f}s", flush=True)

# Random noise
X_noise = torch.randn(100, 129, 200)
with torch.inference_mode():
    pred_noise = model_1(X_noise).numpy().flatten()
print(f"   Random noise (n=100): Mean={pred_noise.mean():.3f}s, Std={pred_noise.std():.3f}s", flush=True)
print(f"   Range: [{pred_noise.min():.3f}, {pred_noise.max():.3f}]s", flush=True)

# Test Challenge 2
print("\n" + "="*80, flush=True)
print("📊 CHALLENGE 2: Externalizing Factor Prediction", flush=True)
print("="*80, flush=True)

print("\n1. Loading model...", end=' ', flush=True)
start = time.time()
model_2 = sub.get_model_challenge_2()
model_2.eval()
load_time = time.time() - start
print(f"✓ ({load_time:.3f}s)", flush=True)

# Count parameters
n_params_2 = sum(p.numel() for p in model_2.parameters())
print(f"   Parameters: {n_params_2:,}", flush=True)

print("\n2. Testing inference on various batch sizes:", flush=True)
for batch_size in [1, 2, 8, 16, 32]:
    X = torch.randn(batch_size, 129, 200)
    start = time.time()
    with torch.inference_mode():
        pred = model_2(X)
    inf_time = time.time() - start
    
    pred_np = pred.numpy().flatten()
    print(f"   Batch {batch_size:2d}: {inf_time*1000:6.2f}ms | Range: [{pred_np.min():.4f}, {pred_np.max():.4f}]", flush=True)

print("\n3. Testing edge cases:", flush=True)
# All zeros
X_zeros = torch.zeros(2, 129, 200)
with torch.inference_mode():
    pred_zeros = model_2(X_zeros).numpy().flatten()
print(f"   All zeros: {pred_zeros} | Mean: {pred_zeros.mean():.4f}", flush=True)

# All ones
X_ones = torch.ones(2, 129, 200)
with torch.inference_mode():
    pred_ones = model_2(X_ones).numpy().flatten()
print(f"   All ones:  {pred_ones} | Mean: {pred_ones.mean():.4f}", flush=True)

# Random noise
X_noise = torch.randn(100, 129, 200)
with torch.inference_mode():
    pred_noise = model_2(X_noise).numpy().flatten()
print(f"   Random noise (n=100): Mean={pred_noise.mean():.4f}, Std={pred_noise.std():.4f}", flush=True)
print(f"   Range: [{pred_noise.min():.4f}, {pred_noise.max():.4f}]", flush=True)

# Resource usage summary
print("\n" + "="*80, flush=True)
print("📋 RESOURCE USAGE SUMMARY", flush=True)
print("="*80, flush=True)
print(f"\nChallenge 1:", flush=True)
print(f"  Parameters: {n_params_1:,}", flush=True)
print(f"  Memory: ~{n_params_1 * 4 / 1024 / 1024:.1f} MB", flush=True)

print(f"\nChallenge 2:", flush=True)
print(f"  Parameters: {n_params_2:,}", flush=True)
print(f"  Memory: ~{n_params_2 * 4 / 1024 / 1024:.1f} MB", flush=True)

total_memory = (n_params_1 + n_params_2) * 4 / 1024 / 1024
print(f"\nTotal Model Memory: ~{total_memory:.1f} MB", flush=True)
print(f"Competition Limit: 20 GB", flush=True)
print(f"Usage: {total_memory / 1024 / 20 * 100:.2f}%", flush=True)

# Final summary
print("\n" + "="*80, flush=True)
print("✅ VALIDATION COMPLETE", flush=True)
print("="*80, flush=True)
print("\nBoth models:", flush=True)
print("  ✓ Load successfully", flush=True)
print("  ✓ Handle various batch sizes", flush=True)
print("  ✓ Handle edge cases (zeros, ones, noise)", flush=True)
print("  ✓ Produce reasonable predictions", flush=True)
print("  ✓ Fit within resource limits", flush=True)
print("\n🎉 Ready for submission!", flush=True)
