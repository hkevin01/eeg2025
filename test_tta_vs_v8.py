#!/usr/bin/env python3
"""
Quick test: Compare V8 vs V8+TTA performance
"""
import sys
sys.path.insert(0, 'submissions/phase1_v8')
sys.path.insert(0, 'submissions/phase1_v8_tta')

import numpy as np
import h5py
import time

print("="*80)
print("🔬 Testing V8 vs V8+TTA")
print("="*80)
print()

# Load a small sample of validation data
print("Loading validation data...")
with h5py.File('data/cached/challenge1_R4_windows.h5', 'r') as f:
    X_val = f['eeg'][:100]  # Just 100 samples for quick test
    y_val = f['labels'][:100]

print(f"Loaded {len(X_val)} validation samples")
print()

# Test V8 (standard)
print("="*80)
print("Testing V8 (Standard)")
print("="*80)

import importlib
v8_module = importlib.import_module('submission', 'submissions.phase1_v8')
v8_sub = v8_module.Submission(SFREQ=100, DEVICE='cpu')

v8_preds = []
start = time.time()
for i, x in enumerate(X_val):
    pred = v8_sub.challenge_1(x)
    v8_preds.append(pred)
    if (i+1) % 20 == 0:
        print(f"  Processed {i+1}/100 samples...")
v8_time = time.time() - start

v8_preds = np.array(v8_preds)
v8_mse = np.mean((v8_preds - y_val) ** 2)
v8_rmse = np.sqrt(v8_mse)

print(f"✅ V8 Results:")
print(f"   MSE: {v8_mse:.6f}")
print(f"   RMSE: {v8_rmse:.6f}")
print(f"   Time: {v8_time:.2f}s ({v8_time/len(X_val)*1000:.1f}ms/sample)")
print()

# Test V8+TTA
print("="*80)
print("Testing V8+TTA (Test-Time Augmentation)")
print("="*80)

# Reload to get fresh import
if 'submission' in sys.modules:
    del sys.modules['submission']
sys.path.insert(0, 'submissions/phase1_v8_tta')
tta_module = importlib.import_module('submission')
tta_sub = tta_module.Submission(SFREQ=100, DEVICE='cpu')

tta_preds = []
start = time.time()
for i, x in enumerate(X_val):
    pred = tta_sub.challenge_1(x)
    tta_preds.append(pred)
    if (i+1) % 20 == 0:
        print(f"  Processed {i+1}/100 samples...")
tta_time = time.time() - start

tta_preds = np.array(tta_preds)
tta_mse = np.mean((tta_preds - y_val) ** 2)
tta_rmse = np.sqrt(tta_mse)

print(f"✅ TTA Results:")
print(f"   MSE: {tta_mse:.6f}")
print(f"   RMSE: {tta_rmse:.6f}")
print(f"   Time: {tta_time:.2f}s ({tta_time/len(X_val)*1000:.1f}ms/sample)")
print()

# Comparison
print("="*80)
print("📊 Comparison")
print("="*80)
print(f"MSE Improvement: {(v8_mse - tta_mse)/v8_mse*100:+.2f}%")
print(f"RMSE Improvement: {(v8_rmse - tta_rmse)/v8_rmse*100:+.2f}%")
print(f"Time Overhead: {(tta_time/v8_time - 1)*100:+.1f}%")
print()

if tta_mse < v8_mse:
    improvement = (v8_mse - tta_mse) / v8_mse * 100
    print(f"✅ TTA is BETTER by {improvement:.2f}%")
    print("   Recommendation: Use TTA version for submission")
else:
    degradation = (tta_mse - v8_mse) / v8_mse * 100
    print(f"❌ TTA is WORSE by {degradation:.2f}%")
    print("   Recommendation: Stick with V8 (no TTA)")

print()
print("="*80)
