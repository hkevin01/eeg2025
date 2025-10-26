"""
Test ROCm RDNA1 Convolution Fixes
Tests the MIOpen environment variable fixes for gfx1030/gfx1031
Based on: https://github.com/ROCm/MIOpen/issues/3540
"""

import os
import sys
import torch
import torch.nn as nn
import time

print("="*70)
print("🧪 ROCm RDNA1 Convolution Fix Test Suite")
print("="*70)

# Display environment variables
print("\n🔧 Environment Variables:")
print(f"   MIOPEN_DEBUG_CONV_GEMM = {os.environ.get('MIOPEN_DEBUG_CONV_GEMM', 'NOT SET')}")
print(f"   MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2 = {os.environ.get('MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2', 'NOT SET')}")
print(f"   MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53 = {os.environ.get('MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53', 'NOT SET')}")
print(f"   HSA_OVERRIDE_GFX_VERSION = {os.environ.get('HSA_OVERRIDE_GFX_VERSION', 'NOT SET')}")
print(f"   HIP_VISIBLE_DEVICES = {os.environ.get('HIP_VISIBLE_DEVICES', 'NOT SET')}")

# Verify fixes are active
fixes_active = (
    os.environ.get('MIOPEN_DEBUG_CONV_GEMM') == '0' and
    os.environ.get('MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW2') == '0' and
    os.environ.get('MIOPEN_DEBUG_CONV_DIRECT_OCL_WRW53') == '0'
)

if fixes_active:
    print("\n✅ Convolution fixes are ACTIVE")
else:
    print("\n⚠️  WARNING: Convolution fixes NOT fully active!")
    print("   Run: source ~/.bashrc")
    print("   Or set variables manually (see CONVOLUTION_FIX_SUMMARY.md)")

# GPU Detection
print("\n📦 PyTorch Info:")
print(f"   Version: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    print("\n❌ GPU not available - cannot test convolutions")
    sys.exit(1)

device = torch.device("cuda")
print(f"   GPU: {torch.cuda.get_device_name(0)}")
print(f"   GPU count: {torch.cuda.device_count()}")

# Test Suite
test_results = []

def run_test(test_name, test_func):
    """Run a test and track results"""
    print(f"\n{'='*70}")
    print(f"TEST: {test_name}")
    print(f"{'='*70}")
    try:
        test_func()
        print(f"✅ PASSED: {test_name}")
        test_results.append((test_name, True, None))
        return True
    except Exception as e:
        print(f"❌ FAILED: {test_name}")
        print(f"   Error: {e}")
        test_results.append((test_name, False, str(e)))
        return False

# Test 1: Simple Conv2d (from MIOpen issue)
def test_simple_conv2d():
    """Test from MIOpen issue #3540"""
    print("Testing simple Conv2d (3x3 kernel)...")
    x = torch.randn(1, 3, 64, 64).cuda()
    conv = nn.Conv2d(3, 16, kernel_size=3, padding=1).cuda()
    y = conv(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == (1, 16, 64, 64), f"Unexpected output shape: {y.shape}"

# Test 2: EEG-style Conv1d
def test_eeg_conv1d():
    """Test Conv1d for EEG (129 channels)"""
    print("Testing EEG-style Conv1d (129 channels)...")
    x = torch.randn(4, 129, 200).cuda()
    conv = nn.Conv1d(129, 64, kernel_size=7).cuda()
    y = conv(x)
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    expected_length = 200 - 7 + 1  # no padding
    assert y.shape == (4, 64, expected_length), f"Unexpected output shape: {y.shape}"

# Test 3: 5x5 Conv (from MIOpen test case)
def test_conv_5x5():
    """Test 5x5 convolution (MIOpen test case)"""
    print("Testing 5x5 convolution (1024 batch, 256 channels)...")
    # Smaller version of MIOpen test: -n 1024 -c 256 -H 32 -W 32 -k 1 -y 5 -x 5
    x = torch.randn(32, 256, 32, 32).cuda()  # Reduced batch for speed
    conv = nn.Conv2d(256, 1, kernel_size=5, padding=2).cuda()
    
    start = time.time()
    y = conv(x)
    elapsed = time.time() - start
    
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    print(f"   Time: {elapsed*1000:.2f} ms")
    assert y.shape == (32, 1, 32, 32), f"Unexpected output shape: {y.shape}"

# Test 4: Training (forward + backward)
def test_training_loop():
    """Test full training loop with convolutions"""
    print("Testing training loop (forward + backward + optimizer)...")
    
    model = nn.Sequential(
        nn.Conv2d(3, 32, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, 10)
    ).cuda()
    
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    # Training step
    x = torch.randn(4, 3, 64, 64).cuda()
    y_true = torch.randint(0, 10, (4,)).cuda()
    
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y_true)
    
    print(f"   Forward pass: {x.shape} -> {y_pred.shape}")
    print(f"   Loss: {loss.item():.4f}")
    
    loss.backward()
    optimizer.step()
    
    print(f"   Backward pass: ✅")
    print(f"   Optimizer step: ✅")

# Test 5: Multi-layer CNN (like ResNet block)
def test_resnet_style():
    """Test ResNet-style convolutional block"""
    print("Testing ResNet-style block...")
    
    class ResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            identity = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            out += identity
            return self.relu(out)
    
    block = ResBlock(64).cuda()
    x = torch.randn(8, 64, 32, 32).cuda()
    y = block(x)
    
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == x.shape, f"Unexpected output shape: {y.shape}"

# Test 6: Depthwise Separable (MobileNet-style)
def test_depthwise_separable():
    """Test depthwise separable convolution"""
    print("Testing depthwise separable convolution...")
    
    # Depthwise
    dw_conv = nn.Conv2d(32, 32, 3, padding=1, groups=32).cuda()
    # Pointwise
    pw_conv = nn.Conv2d(32, 64, 1).cuda()
    
    x = torch.randn(4, 32, 64, 64).cuda()
    y = pw_conv(dw_conv(x))
    
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    assert y.shape == (4, 64, 64, 64), f"Unexpected output shape: {y.shape}"

# Test 7: Large batch stress test
def test_large_batch():
    """Test large batch convolution (memory stress)"""
    print("Testing large batch (64 samples)...")
    
    conv = nn.Conv2d(3, 64, 3, padding=1).cuda()
    x = torch.randn(64, 3, 128, 128).cuda()
    
    start = time.time()
    y = conv(x)
    elapsed = time.time() - start
    
    print(f"   Batch size: 64")
    print(f"   Input: {x.shape} -> Output: {y.shape}")
    print(f"   Time: {elapsed*1000:.2f} ms")
    assert y.shape == (64, 64, 128, 128), f"Unexpected output shape: {y.shape}"

# Test 8: Mixed Conv1d and Conv2d
def test_mixed_convolutions():
    """Test model with both Conv1d and Conv2d"""
    print("Testing mixed Conv1d and Conv2d...")
    
    # Simulate EEG -> Image pipeline
    # Conv1d for temporal
    conv1d = nn.Conv1d(129, 64, 7).cuda()
    # Conv2d for spatial
    conv2d = nn.Conv2d(1, 32, 3).cuda()
    
    x1d = torch.randn(4, 129, 200).cuda()
    y1d = conv1d(x1d)
    print(f"   Conv1d: {x1d.shape} -> {y1d.shape}")
    
    x2d = torch.randn(4, 1, 64, 64).cuda()
    y2d = conv2d(x2d)
    print(f"   Conv2d: {x2d.shape} -> {y2d.shape}")

# Run all tests
print("\n" + "="*70)
print("🚀 RUNNING TEST SUITE")
print("="*70)

run_test("1. Simple Conv2d (MIOpen test case)", test_simple_conv2d)
run_test("2. EEG-style Conv1d (129 channels)", test_eeg_conv1d)
run_test("3. Large 5x5 Convolution", test_conv_5x5)
run_test("4. Training Loop (forward + backward)", test_training_loop)
run_test("5. ResNet-style Block", test_resnet_style)
run_test("6. Depthwise Separable Conv", test_depthwise_separable)
run_test("7. Large Batch Stress Test", test_large_batch)
run_test("8. Mixed Conv1d and Conv2d", test_mixed_convolutions)

# Summary
print("\n" + "="*70)
print("📊 TEST SUMMARY")
print("="*70)

passed = sum(1 for _, result, _ in test_results if result)
failed = len(test_results) - passed

print(f"\nTotal tests: {len(test_results)}")
print(f"✅ Passed: {passed}")
print(f"❌ Failed: {failed}")

if failed > 0:
    print("\n❌ FAILED TESTS:")
    for name, result, error in test_results:
        if not result:
            print(f"   • {name}")
            if error:
                print(f"     {error[:100]}")

print("\n" + "="*70)

if failed == 0:
    print("🎉 ALL TESTS PASSED! 🎉")
    print("Your RDNA1 GPU convolutions are working perfectly!")
    print("="*70)
    sys.exit(0)
else:
    print("⚠️  SOME TESTS FAILED")
    print("Check ROCM_CONVOLUTION_FIX.md for troubleshooting")
    print("="*70)
    sys.exit(1)
