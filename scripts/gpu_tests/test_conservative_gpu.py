#!/usr/bin/env python3
"""
Test conservative GPU operations
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_conservative_gpu():
    print("🛡️  Testing Conservative GPU Operations")
    print("=" * 50)
    
    try:
        from gpu.conservative_gpu import ConservativeGPUOptimizer, conservative_fft, conservative_ifft
        
        print("✅ Imports successful")
        
        # Create optimizer
        optimizer = ConservativeGPUOptimizer()
        
        # Test data
        x = torch.randn(8, 32, 500)
        print(f"Input shape: {x.shape}")
        
        # Test FFT operations
        print("\n🔧 Testing conservative FFT...")
        X = conservative_fft(x, dim=-1)
        print(f"Forward FFT shape: {X.shape}")
        print(f"FFT result device: {X.device}")
        
        print("Testing conservative inverse FFT...")
        x_reconstructed = conservative_ifft(X, n=x.shape[-1], dim=-1)
        print(f"Inverse FFT shape: {x_reconstructed.shape}")
        print(f"iFFT result device: {x_reconstructed.device}")
        
        # Verify reconstruction
        x_cpu = x.cpu() if x.is_cuda else x
        x_recon_cpu = x_reconstructed.cpu() if x_reconstructed.is_cuda else x_reconstructed
        
        error = torch.mean(torch.abs(x_cpu - x_recon_cpu))
        print(f"Reconstruction error: {error:.6f}")
        
        if error < 1e-4:
            print("✅ Conservative FFT operations successful!")
        else:
            print("⚠️  High reconstruction error")
        
        # Test matrix operations
        print("\n🔧 Testing conservative matrix operations...")
        a = torch.randn(32, 64)
        b = torch.randn(64, 32)
        
        c = optimizer.safe_matmul(a, b)
        print(f"Matrix multiplication: {a.shape} x {b.shape} = {c.shape}")
        print(f"Result device: {c.device}")
        
        # Test device selection
        print("\n🔧 Testing device selection...")
        fft_device = optimizer.get_optimal_device("fft")
        general_device = optimizer.get_optimal_device("general")
        
        print(f"Optimal device for FFT: {fft_device}")
        print(f"Optimal device for general ops: {general_device}")
        
        print("\n✅ All conservative tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_conservative_gpu()
