"""
GPU Test 03: Training Loop
Tests forward pass, backward pass, optimizer step
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
import time

def test_training_loop():
    """Test complete training loop"""
    print("="*70)
    print("TEST 03: Training Loop")
    print("="*70)
    
    if not torch.cuda.is_available():
        print("❌ GPU not available")
        return False
    
    device = torch.device("cuda")
    
    # Create a simple model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv1d(129, 64, 7)
            self.bn1 = nn.BatchNorm1d(64)
            self.conv2 = nn.Conv1d(64, 32, 5)
            self.bn2 = nn.BatchNorm1d(32)
            self.fc = nn.Linear(32 * 188, 1)
        
        def forward(self, x):
            x = torch.relu(self.bn1(self.conv1(x)))
            x = torch.relu(self.bn2(self.conv2(x)))
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    # Test 1: Model creation
    print("\n1️⃣ Model Creation:")
    try:
        model = SimpleModel().to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {n_params:,}")
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 2: Forward pass
    print("\n2️⃣ Forward Pass:")
    try:
        x = torch.randn(8, 129, 200, device=device)
        y = torch.randn(8, 1, device=device)
        
        start = time.time()
        output = model(x)
        elapsed = time.time() - start
        
        print(f"   Output shape: {output.shape}")
        print(f"   Time: {elapsed*1000:.2f} ms")
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 3: Loss computation
    print("\n3️⃣ Loss Computation:")
    try:
        criterion = nn.MSELoss()
        loss = criterion(output, y)
        print(f"   Loss: {loss.item():.6f}")
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 4: Backward pass
    print("\n4️⃣ Backward Pass:")
    try:
        start = time.time()
        loss.backward()
        elapsed = time.time() - start
        
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"   Gradient norm: {grad_norm:.6f}")
        print(f"   Time: {elapsed*1000:.2f} ms")
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 5: Optimizer step
    print("\n5️⃣ Optimizer Step:")
    try:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        optimizer.step()
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    # Test 6: Multiple training steps
    print("\n6️⃣ Multiple Training Steps (10 iterations):")
    try:
        model.train()
        total_time = 0
        
        for i in range(10):
            x = torch.randn(8, 129, 200, device=device)
            y = torch.randn(8, 1, device=device)
            
            optimizer.zero_grad()
            start = time.time()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            elapsed = time.time() - start
            total_time += elapsed
            
            if (i + 1) % 5 == 0:
                print(f"   Iter {i+1}: loss={loss.item():.6f}, time={elapsed*1000:.2f}ms")
        
        avg_time = total_time / 10 * 1000
        print(f"   Average time per iteration: {avg_time:.2f} ms")
        print("   ✅ PASSED")
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
        return False
    
    print("\n" + "="*70)
    print("✅ ALL TRAINING LOOP TESTS PASSED")
    print("="*70)
    return True

if __name__ == "__main__":
    success = test_training_loop()
    sys.exit(0 if success else 1)
