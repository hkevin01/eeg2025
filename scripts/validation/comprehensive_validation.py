#!/usr/bin/env python3
"""
Comprehensive Validation Script for EEG 2025 Competition
=========================================================
Tests both Challenge 1 and Challenge 2 models with:
- Multiple random inputs
- Different batch sizes
- Inference timing
- Memory usage monitoring
- Output validation
"""
import os
import sys
import time
import psutil
import torch
import numpy as np
from pathlib import Path

# Force CPU for consistent testing
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HIP_VISIBLE_DEVICES'] = ''

# Add project root to path
sys.path.insert(0, '/home/kevin/Projects/eeg2025')

print("="*80, flush=True)
print("🧪 COMPREHENSIVE VALIDATION - EEG 2025 Competition", flush=True)
print("="*80, flush=True)

# Competition parameters
SFREQ = 100
DEVICE = torch.device("cpu")
NUM_CHANNELS = 129
SEGMENT_LENGTH = 200

print(f"\n📋 Configuration:", flush=True)
print(f"   Device: {DEVICE}", flush=True)
print(f"   Sampling Rate: {SFREQ} Hz", flush=True)
print(f"   Input Shape: (batch, {NUM_CHANNELS}, {SEGMENT_LENGTH})", flush=True)

# Import submission
print("\n📦 Importing submission...", end=' ', flush=True)
from submission import Submission
print("✓", flush=True)

# Initialize submission
print("🔧 Initializing submission...", end=' ', flush=True)
start = time.time()
sub = Submission(SFREQ, DEVICE)
init_time = time.time() - start
print(f"✓ ({init_time:.2f}s)", flush=True)

# Get initial memory usage
process = psutil.Process()
initial_memory = process.memory_info().rss / 1024 / 1024  # MB
print(f"💾 Initial memory: {initial_memory:.1f} MB", flush=True)

# Test configurations
test_configs = [
    {"name": "Single sample", "batch_size": 1, "num_batches": 5},
    {"name": "Small batch", "batch_size": 8, "num_batches": 3},
    {"name": "Competition batch", "batch_size": 32, "num_batches": 2},
    {"name": "Large batch", "batch_size": 64, "num_batches": 1},
]

print("\n" + "="*80, flush=True)
print("🎯 CHALLENGE 1: RESPONSE TIME PREDICTION", flush=True)
print("="*80, flush=True)

# Load Challenge 1 model
print("\n📥 Loading Challenge 1 model...", end=' ', flush=True)
start = time.time()
try:
    model_1 = sub.get_model_challenge_1()
    load_time = time.time() - start
    print(f"✓ ({load_time:.2f}s)", flush=True)
    
    current_memory = process.memory_info().rss / 1024 / 1024
    model_memory = current_memory - initial_memory
    print(f"💾 Model memory: {model_memory:.1f} MB", flush=True)
    
    # Test different batch sizes
    all_predictions = []
    all_times = []
    
    for config in test_configs:
        batch_size = config["batch_size"]
        num_batches = config["num_batches"]
        
        print(f"\n🧪 Testing: {config['name']} (batch={batch_size}, n={num_batches})", flush=True)
        
        batch_times = []
        batch_predictions = []
        
        for i in range(num_batches):
            # Create test data
            X_test = np.random.randn(batch_size, NUM_CHANNELS, SEGMENT_LENGTH).astype(np.float32)
            X_tensor = torch.from_numpy(X_test).to(DEVICE)
            
            # Time inference
            start = time.time()
            with torch.inference_mode():
                pred = model_1.forward(X_tensor)
            inference_time = time.time() - start
            batch_times.append(inference_time)
            
            pred_np = pred.cpu().numpy().flatten()
            batch_predictions.extend(pred_np)
            
            print(f"   Batch {i+1}/{num_batches}: {inference_time*1000:.1f}ms, "
                  f"predictions: [{pred_np.min():.2f}, {pred_np.max():.2f}]", flush=True)
        
        avg_time = np.mean(batch_times)
        per_sample_time = avg_time / batch_size * 1000
        all_times.extend(batch_times)
        all_predictions.extend(batch_predictions)
        
        print(f"   ⏱️  Avg: {avg_time*1000:.1f}ms total, {per_sample_time:.2f}ms/sample", flush=True)
    
    # Summary statistics
    all_predictions = np.array(all_predictions)
    print(f"\n📊 Challenge 1 Summary:", flush=True)
    print(f"   Total predictions: {len(all_predictions)}", flush=True)
    print(f"   Range: [{all_predictions.min():.3f}, {all_predictions.max():.3f}] seconds", flush=True)
    print(f"   Mean: {all_predictions.mean():.3f}s, Std: {all_predictions.std():.3f}s", flush=True)
    print(f"   Avg inference time: {np.mean(all_times)*1000:.1f}ms", flush=True)
    
    # Validation checks
    print(f"\n✅ Validation Checks:", flush=True)
    if all_predictions.min() >= 0:
        print(f"   ✓ All predictions positive", flush=True)
    else:
        print(f"   ⚠️  Some predictions negative!", flush=True)
    
    if 0.1 <= all_predictions.mean() <= 10.0:
        print(f"   ✓ Predictions in reasonable range (0.1-10s)", flush=True)
    else:
        print(f"   ⚠️  Predictions outside typical response time range!", flush=True)
    
    challenge1_success = True
    
except Exception as e:
    print(f"❌ Error: {e}", flush=True)
    challenge1_success = False

print("\n" + "="*80, flush=True)
print("📊 CHALLENGE 2: EXTERNALIZING FACTOR PREDICTION", flush=True)
print("="*80, flush=True)

# Load Challenge 2 model
print("\n📥 Loading Challenge 2 model...", end=' ', flush=True)
start = time.time()
try:
    model_2 = sub.get_model_challenge_2()
    load_time = time.time() - start
    print(f"✓ ({load_time:.2f}s)", flush=True)
    
    current_memory = process.memory_info().rss / 1024 / 1024
    total_memory = current_memory - initial_memory
    print(f"💾 Total memory (both models): {total_memory:.1f} MB", flush=True)
    
    # Test different batch sizes
    all_predictions = []
    all_times = []
    
    for config in test_configs:
        batch_size = config["batch_size"]
        num_batches = config["num_batches"]
        
        print(f"\n🧪 Testing: {config['name']} (batch={batch_size}, n={num_batches})", flush=True)
        
        batch_times = []
        batch_predictions = []
        
        for i in range(num_batches):
            # Create test data
            X_test = np.random.randn(batch_size, NUM_CHANNELS, SEGMENT_LENGTH).astype(np.float32)
            X_tensor = torch.from_numpy(X_test).to(DEVICE)
            
            # Time inference
            start = time.time()
            with torch.inference_mode():
                pred = model_2.forward(X_tensor)
            inference_time = time.time() - start
            batch_times.append(inference_time)
            
            pred_np = pred.cpu().numpy().flatten()
            batch_predictions.extend(pred_np)
            
            print(f"   Batch {i+1}/{num_batches}: {inference_time*1000:.1f}ms, "
                  f"predictions: [{pred_np.min():.2f}, {pred_np.max():.2f}]", flush=True)
        
        avg_time = np.mean(batch_times)
        per_sample_time = avg_time / batch_size * 1000
        all_times.extend(batch_times)
        all_predictions.extend(batch_predictions)
        
        print(f"   ⏱️  Avg: {avg_time*1000:.1f}ms total, {per_sample_time:.2f}ms/sample", flush=True)
    
    # Summary statistics
    all_predictions = np.array(all_predictions)
    print(f"\n📊 Challenge 2 Summary:", flush=True)
    print(f"   Total predictions: {len(all_predictions)}", flush=True)
    print(f"   Range: [{all_predictions.min():.3f}, {all_predictions.max():.3f}]", flush=True)
    print(f"   Mean: {all_predictions.mean():.3f}, Std: {all_predictions.std():.3f}", flush=True)
    print(f"   Avg inference time: {np.mean(all_times)*1000:.1f}ms", flush=True)
    
    # Validation checks
    print(f"\n✅ Validation Checks:", flush=True)
    if -5 <= all_predictions.min() and all_predictions.max() <= 5:
        print(f"   ✓ Predictions in reasonable range (-5 to 5)", flush=True)
    else:
        print(f"   ⚠️  Predictions outside typical normalized range!", flush=True)
    
    challenge2_success = True
    
except Exception as e:
    print(f"❌ Error: {e}", flush=True)
    challenge2_success = False

# Final memory check
final_memory = process.memory_info().rss / 1024 / 1024
total_memory_used = final_memory - initial_memory

print("\n" + "="*80, flush=True)
print("📋 FINAL VALIDATION REPORT", flush=True)
print("="*80, flush=True)

print(f"\n💾 Memory Usage:", flush=True)
print(f"   Initial: {initial_memory:.1f} MB", flush=True)
print(f"   Final: {final_memory:.1f} MB", flush=True)
print(f"   Total used: {total_memory_used:.1f} MB", flush=True)

if total_memory_used < 1024:
    print(f"   ✅ Well under 20GB limit", flush=True)
else:
    print(f"   ⚠️  High memory usage!", flush=True)

print(f"\n🎯 Challenge Results:", flush=True)
print(f"   Challenge 1: {'✅ PASS' if challenge1_success else '❌ FAIL'}", flush=True)
print(f"   Challenge 2: {'✅ PASS' if challenge2_success else '❌ FAIL'}", flush=True)

if challenge1_success and challenge2_success:
    print(f"\n{'='*80}", flush=True)
    print(f"✅ ALL VALIDATION TESTS PASSED!", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"\n🎉 Submission is ready for Codabench!", flush=True)
    print(f"\nNext steps:", flush=True)
    print(f"1. Write 2-page methods document", flush=True)
    print(f"2. Create submission ZIP: cd submission_package && zip -r ../submission_complete.zip .", flush=True)
    print(f"3. Upload to https://www.codabench.org/competitions/4287/", flush=True)
else:
    print(f"\n{'='*80}", flush=True)
    print(f"⚠️  VALIDATION ISSUES DETECTED", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"\nPlease fix issues before submitting.", flush=True)

print(f"\n{'='*80}", flush=True)
