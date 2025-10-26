"""
GPU Test 04: Memory Stress Test
Tests for HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION
"""

import torch
import torch.nn as nn
import sys
import time
import gc

def test_memory_stress():
    """Test memory-intensive operations"""
    print("="*70)
    print("TEST 04: Memory Stress (HSA Aperture Violation Detection)")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ GPU not available")
        return False
    
    device = torch.device("cuda")
    
    # Test 1: Large tensor allocation
    print("\n1️⃣ Large Tensor Allocation:")
    try:
        sizes = [1000, 2000, 3000, 4000]
        for size in sizes:
            x = torch.randn(size, size, device=device)
            mem_mb = x.numel() * 4 / 1024**2
            print(f"   {size}x{size}: {mem_mb:.1f} MB - ✅")
            del x
        torch.cuda.empty_cache()
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 2: Rapid allocate/deallocate
    print("\n2️⃣ Rapid Allocate/Deallocate (100 iterations):")
    try:
        for i in range(100):
            x = torch.randn(1000, 1000, device=device)
            y = x @ x.T
            del x, y
            if (i + 1) % 25 == 0:
                print(f"   Iteration {i+1}/100")
                torch.cuda.empty_cache()
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 3: Multiple large models
    print("\n3️⃣ Multiple Large Models:")
    try:
        models = []
        for i in range(3):
            model = nn.Sequential(
                nn.Conv1d(129, 256, 7),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Conv1d(256, 256, 5),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256 * 188, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            ).to(device)
            models.append(model)
            n_params = sum(p.numel() for p in model.parameters())
            print(f"   Model {i+1}: {n_params:,} parameters")
        
        # Clean up
        del models
        torch.cuda.empty_cache()
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 4: Large batch training
    print("\n4️⃣ Large Batch Training (64 batch size):")
    try:
        model = nn.Sequential(
            nn.Conv1d(129, 64, 7),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 32, 5),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.MSELoss()
        
        for i in range(10):
            x = torch.randn(64, 129, 200, device=device)  # Large batch
            y = torch.randn(64, 32, 188, device=device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 5 == 0:
                print(f"   Iteration {i+1}: loss={loss.item():.6f}")
        
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 5: Extended run (500 iterations)
    print("\n5️⃣ Extended Run (500 iterations - HSA Aperture Test):")
    print("   This may take 1-2 minutes...")
    try:
        model = nn.Conv1d(129, 64, 7).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        
        start_time = time.time()
        for i in range(500):
            x = torch.randn(16, 129, 200, device=device)
            y = torch.randn(16, 64, 194, device=device)
            
            optimizer.zero_grad()
            output = model(x)
            loss = ((output - y) ** 2).mean()
            loss.backward()
            optimizer.step()
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"   Iteration {i+1}/500 ({elapsed:.1f}s elapsed)")
        
        total_time = time.time() - start_time
        print(f"   Completed 500 iterations in {total_time:.1f}s")
        print("   ✅ PASSED - NO HSA APERTURE VIOLATION!")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        print("   ⚠️  This may indicate HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION")
        return False
    
    print("\n" + "="*70)
    print("✅ ALL MEMORY STRESS TESTS PASSED")
    print("🎉 NO HSA APERTURE VIOLATIONS DETECTED!")
    print("="*70)
    return True

if __name__ == "__main__":
    success = test_memory_stress()
    sys.exit(0 if success else 1)
