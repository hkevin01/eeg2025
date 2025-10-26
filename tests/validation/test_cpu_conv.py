#!/usr/bin/env python3
"""
Test convolutions on CPU to verify the issue is GPU-specific
"""

import torch
import time

def test_cpu_conv():
    print("🖥️  Testing Convolutions on CPU")
    print("=" * 40)
    
    device = torch.device('cpu')
    print(f"Using device: {device}")
    
    try:
        # Test 1: Conv1d on CPU
        print("1️⃣ Testing Conv1d on CPU...")
        conv1d = torch.nn.Conv1d(4, 8, kernel_size=5, padding=2).to(device)
        input_1d = torch.randn(4, 4, 200).to(device)  # Same size as original test
        
        start_time = time.time()
        with torch.no_grad():
            output_1d = conv1d(input_1d)
        elapsed = time.time() - start_time
        print(f"   ✅ Conv1d successful! Shape: {input_1d.shape} -> {output_1d.shape}, Time: {elapsed:.3f}s")
        
        # Test 2: Conv2d on CPU
        print("2️⃣ Testing Conv2d on CPU...")
        conv2d = torch.nn.Conv2d(3, 16, kernel_size=3, padding=1).to(device)
        input_2d = torch.randn(2, 3, 32, 32).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            output_2d = conv2d(input_2d)
        elapsed = time.time() - start_time
        print(f"   ✅ Conv2d successful! Shape: {input_2d.shape} -> {output_2d.shape}, Time: {elapsed:.3f}s")
        
        print()
        print("🎉 All CPU convolution tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ CPU convolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_cpu_conv()
