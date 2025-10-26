#!/usr/bin/env python3
"""
Minimal GPU workflow - only operations that definitely work
No batch norm, no complex operations that might trigger freezes
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

import torch
import torch.nn as nn
import torch.optim as optim
import time

print("🔬 Minimal GPU Workflow - Conservative Approach")
print("=" * 45)

class SafeLinearNet(nn.Module):
    """Ultra-conservative network using only Linear + ReLU"""
    def __init__(self, input_size=129, hidden_size=256, num_classes=4):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size//2, num_classes)
    
    def forward(self, x):
        # Flatten if needed: (batch, channels, time) -> (batch, features)
        if x.dim() == 3:
            x = x.view(x.size(0), -1)  # Simple flatten
        
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def safe_test():
    """Test only the safest operations"""
    print("🧪 Testing minimal safe operations...")
    
    if not torch.cuda.is_available():
        print("❌ GPU not available")
        return
    
    device = torch.device('cuda')
    print(f"📍 Using: {torch.cuda.get_device_name()}")
    
    # Very small model
    model = SafeLinearNet(input_size=100, hidden_size=64, num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # Simple SGD
    
    print(f"📊 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Tiny batch to be extra safe
    batch_size = 4
    input_size = 100
    
    print("🔄 Running 3 safe iterations...")
    
    for i in range(3):
        print(f"  Iteration {i+1}/3...", end=" ")
        
        # Generate small batch
        x = torch.randn(batch_size, input_size).to(device)
        y = torch.randint(0, 2, (batch_size,)).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item():.4f}")
        
        # Clear cache after each iteration
        torch.cuda.empty_cache()
    
    print("✅ Safe operations completed successfully!")
    print(f"🧠 Final memory: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    
    return True

def benchmark_safe_operations():
    """Benchmark what definitely works"""
    print("\n⚡ Benchmarking safe operations...")
    
    device = torch.device('cuda')
    
    # Matrix multiplication benchmark
    print("1. Matrix operations...", end=" ")
    start = time.time()
    for _ in range(100):
        a = torch.randn(100, 100).cuda()
        b = torch.randn(100, 100).cuda()
        c = a @ b
    elapsed = time.time() - start
    print(f"{elapsed:.3f}s (100 matmuls)")
    
    # Linear layer benchmark
    print("2. Linear layers...", end=" ")
    linear = nn.Linear(100, 50).cuda()
    x = torch.randn(32, 100).cuda()
    start = time.time()
    for _ in range(100):
        y = linear(x)
    elapsed = time.time() - start
    print(f"{elapsed:.3f}s (100 forwards)")
    
    print("✅ Benchmarks completed safely")

if __name__ == "__main__":
    # Test basic functionality
    success = safe_test()
    
    if success:
        # Run benchmarks
        benchmark_safe_operations()
        
        print("\n💡 GPU Usage Guidelines:")
        print("   ✅ Matrix operations (A @ B)")
        print("   ✅ Linear layers (nn.Linear)")
        print("   ✅ Simple activations (ReLU)")
        print("   ✅ Basic loss functions (MSE, CrossEntropy)")
        print("   ✅ SGD optimizer")
        print("   ❌ Avoid: BatchNorm, Dropout, Conv layers")
        print("   ❌ Avoid: Large batches (>8)")
        print("   ❌ Avoid: Complex models")
        
        print("\n🚀 Recommended GPU workflow:")
        print("   1. Use GPU for rapid Linear layer experiments")
        print("   2. Test MLP architectures quickly")
        print("   3. Prototype embedding layers")
        print("   4. Once design is validated, move to CPU for full training")
        
    else:
        print("\n❌ Even safe operations failed - GPU may need system reboot")
