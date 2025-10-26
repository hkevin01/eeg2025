#!/usr/bin/env python3
"""
Practical GPU workflow based on what we learned
Uses GPU for what works, falls back to CPU for complex operations
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

import torch
import torch.nn as nn
import time

def setup_hybrid_environment():
    """Set up hybrid GPU/CPU environment"""
    print("🔧 Setting up hybrid GPU/CPU environment...")
    
    gpu_available = torch.cuda.is_available()
    print(f"GPU Available: {gpu_available}")
    
    if gpu_available:
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"PyTorch: {torch.__version__}")
    
    return gpu_available

def test_operation_safety(operation_name, operation_func):
    """Test if an operation works safely on GPU"""
    try:
        start_time = time.time()
        result = operation_func()
        elapsed = time.time() - start_time
        print(f"   ✅ {operation_name}: {elapsed:.3f}s")
        return True, elapsed
    except Exception as e:
        print(f"   ❌ {operation_name}: Failed - {str(e)[:50]}...")
        return False, None

def determine_device_strategy():
    """Determine what operations should use GPU vs CPU"""
    print("\n🧪 Testing operation compatibility...")
    
    if not torch.cuda.is_available():
        print("GPU not available - using CPU for everything")
        return {'default': 'cpu'}
    
    strategy = {}
    
    # Test basic operations
    def test_basic():
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        return x @ y
    
    def test_linear():
        linear = nn.Linear(100, 50).cuda()
        x = torch.randn(32, 100).cuda()
        return linear(x)
    
    def test_small_conv():
        conv = nn.Conv1d(2, 4, 3).cuda()
        x = torch.randn(2, 2, 10).cuda()
        return conv(x)
    
    def test_realistic_conv():
        conv = nn.Conv1d(32, 64, 5).cuda()
        x = torch.randn(8, 32, 100).cuda()
        return conv(x)
    
    # Test each operation
    basic_safe, _ = test_operation_safety("Basic tensors", test_basic)
    linear_safe, _ = test_operation_safety("Linear layers", test_linear)
    small_conv_safe, _ = test_operation_safety("Small convolutions", test_small_conv)
    real_conv_safe, _ = test_operation_safety("Realistic convolutions", test_realistic_conv)
    
    # Build strategy
    strategy['basic_ops'] = 'cuda' if basic_safe else 'cpu'
    strategy['linear'] = 'cuda' if linear_safe else 'cpu'
    strategy['small_conv'] = 'cuda' if small_conv_safe else 'cpu'
    strategy['conv'] = 'cuda' if real_conv_safe else 'cpu'
    strategy['default'] = 'cpu'  # Safe fallback
    
    return strategy

class HybridModel(nn.Module):
    """Model that intelligently uses GPU/CPU based on operation safety"""
    
    def __init__(self, strategy, input_size=129, num_classes=4):
        super().__init__()
        self.strategy = strategy
        
        # Use GPU for linear layers if safe
        linear_device = strategy.get('linear', 'cpu')
        print(f"📍 Linear layers on: {linear_device}")
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        
        if linear_device == 'cuda':
            self.feature_extractor = self.feature_extractor.cuda()
        
        self.classifier = nn.Linear(128, num_classes)
        if linear_device == 'cuda':
            self.classifier = self.classifier.cuda()
    
    def forward(self, x):
        # Handle device placement intelligently
        if x.dim() == 3:  # (batch, channels, time)
            x = x.mean(dim=2)  # Global average pooling
        
        # Move to appropriate device for linear ops
        linear_device = self.strategy.get('linear', 'cpu')
        if linear_device == 'cuda' and x.device.type == 'cpu':
            x = x.cuda()
        elif linear_device == 'cpu' and x.device.type == 'cuda':
            x = x.cpu()
        
        features = self.feature_extractor(x)
        output = self.classifier(features)
        
        return output

def rapid_prototype_session():
    """Run a rapid prototyping session using hybrid approach"""
    print("\n🚀 Rapid Prototyping Session")
    print("-" * 30)
    
    strategy = determine_device_strategy()
    print(f"\n📋 Device Strategy: {strategy}")
    
    # Create hybrid model
    model = HybridModel(strategy)
    
    # Test different architectures quickly
    architectures = [
        (64, 32),
        (128, 64), 
        (256, 128),
        (512, 256)
    ]
    
    results = []
    
    for hidden1, hidden2 in architectures:
        print(f"\n🧪 Testing architecture: {hidden1} -> {hidden2}")
        
        # Create model for this architecture
        test_model = nn.Sequential(
            nn.Linear(129, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 4)
        )
        
        # Use appropriate device
        device = strategy.get('linear', 'cpu')
        test_model = test_model.to(device)
        
        # Time the forward pass
        batch_size = 16
        x = torch.randn(batch_size, 129).to(device)
        
        start_time = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = test_model(x)
        elapsed = time.time() - start_time
        
        throughput = (100 * batch_size) / elapsed
        params = sum(p.numel() for p in test_model.parameters())
        
        print(f"   ⏱️  Time: {elapsed:.3f}s")
        print(f"   🏃 Throughput: {throughput:.0f} samples/sec")
        print(f"   📊 Parameters: {params:,}")
        
        results.append({
            'arch': (hidden1, hidden2),
            'time': elapsed,
            'throughput': throughput,
            'params': params,
            'device': device
        })
    
    # Show results
    print(f"\n🏆 Prototyping Results:")
    print("-" * 40)
    for result in results:
        arch = result['arch']
        print(f"{arch}: {result['throughput']:.0f} samples/sec "
              f"({result['params']:,} params) on {result['device']}")
    
    # Best architecture
    best = max(results, key=lambda x: x['throughput'])
    print(f"\n🥇 Fastest: {best['arch']} at {best['throughput']:.0f} samples/sec")
    
    return results

if __name__ == "__main__":
    setup_hybrid_environment()
    results = rapid_prototype_session()
    
    print(f"\n💡 Hybrid Workflow Summary:")
    print("✅ GPU acceleration for linear layers and basic operations")
    print("✅ Rapid prototyping of MLP architectures") 
    print("✅ Fast iteration on network designs")
    print("❗ Use CPU for convolutions and complex models")
    print("❗ Transfer best designs to CPU for full training")
    
    print(f"\n🎯 Recommendation:")
    print("Use this hybrid approach for rapid experimentation,")
    print("then scale successful architectures to full CNN models on CPU.")
