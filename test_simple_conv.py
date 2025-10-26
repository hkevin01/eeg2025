#!/usr/bin/env python3
"""
Simple convolution test for ROCm 7.0 + PyTorch + gfx1030
Based on GitHub issue #5195 solution
"""

import torch
import time
import sys

def test_simple_conv():
    print("🧪 Simple Convolution Test for ROCm 7.0 + gfx1030")
    print("=" * 50)
    
    # Check environment
    print(f"PyTorch: {torch.__version__}")
    print(f"ROCm available: {torch.cuda.is_available()}")
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print()
    
    if not torch.cuda.is_available():
        print("❌ ROCm not available, exiting")
        return False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    try:
        # Test 1: Very simple Conv1d
        print("1️⃣ Testing minimal Conv1d...")
        conv1d = torch.nn.Conv1d(1, 1, kernel_size=3, padding=1).to(device)
        input_1d = torch.randn(1, 1, 10).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            output_1d = conv1d(input_1d)
        elapsed = time.time() - start_time
        print(f"   ✅ Conv1d successful! Shape: {input_1d.shape} -> {output_1d.shape}, Time: {elapsed:.3f}s")
        
        # Test 2: Very simple Conv2d  
        print("2️⃣ Testing minimal Conv2d...")
        conv2d = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1).to(device)
        input_2d = torch.randn(1, 1, 8, 8).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            output_2d = conv2d(input_2d)
        elapsed = time.time() - start_time
        print(f"   ✅ Conv2d successful! Shape: {input_2d.shape} -> {output_2d.shape}, Time: {elapsed:.3f}s")
        
        # Test 3: EEG-style Conv1d (small)
        print("3️⃣ Testing small EEG-style Conv1d...")
        eeg_conv = torch.nn.Conv1d(in_channels=4, out_channels=8, kernel_size=5, padding=2).to(device)
        eeg_input = torch.randn(2, 4, 20).to(device)  # Small: batch=2, channels=4, time=20
        
        start_time = time.time()
        with torch.no_grad():
            eeg_output = eeg_conv(eeg_input)
        elapsed = time.time() - start_time
        print(f"   ✅ EEG Conv1d successful! Shape: {eeg_input.shape} -> {eeg_output.shape}, Time: {elapsed:.3f}s")
        
        print()
        print("🎉 All basic convolution tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Convolution test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_conv()
    sys.exit(0 if success else 1)
