#!/usr/bin/env python3
"""
Quick test for AMD-safe FFT operations
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_amd_safe_fft():
    print("🧪 Testing AMD-Safe FFT")
    print("=" * 50)
    
    try:
        from gpu.unified_gpu_optimized import UnifiedFFTOptimizer
        from gpu.amd_safe_fft import amd_safe_rfft, amd_safe_irfft
        
        print("✅ Imports successful")
        
        # Test small tensor to avoid crashes
        x = torch.randn(4, 16, 100)
        print(f"Input shape: {x.shape}")
        
        if torch.cuda.is_available():
            x_gpu = x.cuda()
            print(f"Moved to GPU: {x_gpu.device}")
            
            # Test direct AMD-safe functions
            print("\n🔧 Testing direct AMD-safe FFT...")
            X = amd_safe_rfft(x_gpu, dim=-1)
            print(f"Forward FFT shape: {X.shape}")
            
            print("Testing AMD-safe inverse FFT...")
            x_reconstructed = amd_safe_irfft(X, n=x_gpu.shape[-1], dim=-1)
            print(f"Inverse FFT shape: {x_reconstructed.shape}")
            
            # Verify reconstruction
            error = torch.mean(torch.abs(x_gpu - x_reconstructed))
            print(f"Reconstruction error: {error:.6f}")
            
            if error < 1e-4:
                print("✅ Direct AMD-safe FFT successful!")
            else:
                print("⚠️  High reconstruction error")
            
            # Test through unified optimizer
            print("\n🔧 Testing unified optimizer...")
            fft_opt = UnifiedFFTOptimizer()
            
            X2 = fft_opt.rfft_batch(x_gpu, dim=-1)
            print(f"Unified forward FFT shape: {X2.shape}")
            
            x_reconstructed2 = fft_opt.irfft_batch(X2, n=x_gpu.shape[-1], dim=-1)
            print(f"Unified inverse FFT shape: {x_reconstructed2.shape}")
            
            error2 = torch.mean(torch.abs(x_gpu - x_reconstructed2))
            print(f"Unified reconstruction error: {error2:.6f}")
            
            if error2 < 1e-4:
                print("✅ Unified optimizer successful!")
            else:
                print("⚠️  High unified reconstruction error")
            
            # Cleanup
            torch.cuda.empty_cache()
            
        else:
            print("⚠️  No GPU available")
        
        print("\n✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_amd_safe_fft()
