#!/usr/bin/env python3
"""
Rapid GPU experimentation workflow
Uses PyTorch 1.13.1+rocm5.2 for fast MLP prototyping
"""

import os
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

print("🚀 Rapid GPU Experimentation - EEG Model Prototyping")
print("=" * 55)

class FastMLPClassifier(nn.Module):
    """GPU-friendly MLP for rapid EEG experiments"""
    def __init__(self, input_size=129, hidden_sizes=[512, 256, 128], num_classes=4):
        super().__init__()
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_size = hidden_size
            
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        if x.dim() == 3:  # (batch, channels, time) -> (batch, features)
            x = x.mean(dim=2)  # Global average pooling
        return self.network(x)

def rapid_experiment(architecture_params):
    """Run a quick experiment with given architecture"""
    print(f"\n🧪 Testing architecture: {architecture_params}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📍 Device: {device}")
    
    # Create model
    model = FastMLPClassifier(**architecture_params).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Synthetic EEG data (batch_size, channels, time_points)
    batch_size = 32
    channels = 129
    time_points = 200
    num_classes = 4
    
    # Training simulation
    model.train()
    start_time = time.time()
    
    total_loss = 0
    for epoch in range(10):
        # Generate batch
        x = torch.randn(batch_size, channels, time_points).to(device)
        y = torch.randint(0, num_classes, (batch_size,)).to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if epoch % 2 == 0:
            print(f"  Epoch {epoch:2d}: Loss = {loss.item():.4f}")
    
    train_time = time.time() - start_time
    avg_loss = total_loss / 10
    
    # Test inference speed
    model.eval()
    with torch.no_grad():
        test_x = torch.randn(batch_size, channels, time_points).to(device)
        inference_start = time.time()
        _ = model(test_x)
        inference_time = time.time() - inference_start
    
    print(f"  ⏱️  Training time: {train_time:.2f}s")
    print(f"  ⏱️  Inference time: {inference_time*1000:.2f}ms")
    print(f"  📉 Final loss: {avg_loss:.4f}")
    print(f"  🧠 Memory used: {torch.cuda.memory_allocated()/1024**2:.1f} MB")
    
    return {
        'architecture': architecture_params,
        'train_time': train_time,
        'inference_time': inference_time,
        'final_loss': avg_loss,
        'memory_mb': torch.cuda.memory_allocated()/1024**2
    }

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("❌ GPU not available - switching to CPU")
        
    # Rapid architecture search
    architectures = [
        {'hidden_sizes': [256, 128]},
        {'hidden_sizes': [512, 256, 128]}, 
        {'hidden_sizes': [1024, 512, 256]},
        {'hidden_sizes': [512, 256, 128, 64]},
    ]
    
    results = []
    for arch in architectures:
        try:
            result = rapid_experiment(arch)
            results.append(result)
            torch.cuda.empty_cache()  # Clean memory
        except Exception as e:
            print(f"❌ Failed: {e}")
    
    # Summary
    print("\n🏆 Experiment Summary:")
    print("-" * 50)
    for i, result in enumerate(results):
        arch = result['architecture']['hidden_sizes']
        print(f"{i+1}. {arch}: {result['train_time']:.1f}s, "
              f"Loss: {result['final_loss']:.3f}, "
              f"Mem: {result['memory_mb']:.1f}MB")
    
    if results:
        best = min(results, key=lambda x: x['final_loss'])
        print(f"\n🥇 Best architecture: {best['architecture']['hidden_sizes']}")
        print(f"   Loss: {best['final_loss']:.4f}, Time: {best['train_time']:.2f}s")
    
    print("\n💡 Next steps:")
    print("   1. Take best architecture from GPU experiments")
    print("   2. Scale to full CNN model")
    print("   3. Train on CPU with real EEG data")
    print("   4. Use GPU for rapid hyperparameter search")
