#!/usr/bin/env python3
"""Test SDK PyTorch with EEG project requirements"""

import sys
import time

def test_pytorch():
    """Test PyTorch GPU functionality"""
    print("=" * 60)
    print("1. Testing PyTorch GPU Support")
    print("=" * 60)
    
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ ROCm version: {torch.version.hip}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"✅ Architecture: {props.gcnArchName}")
            
            # Test GPU computation
            print("\n🧪 Testing GPU computation...")
            start = time.time()
            x = torch.randn(1000, 1000, device='cuda')
            y = torch.randn(1000, 1000, device='cuda')
            z = torch.matmul(x, y)
            elapsed = (time.time() - start) * 1000
            print(f"✅ Matrix multiplication: {elapsed:.2f}ms")
            
            return True
        else:
            print("❌ GPU not available")
            return False
            
    except Exception as e:
        print(f"❌ PyTorch test failed: {e}")
        return False

def test_eeg_imports():
    """Test EEG project imports"""
    print("\n" + "=" * 60)
    print("2. Testing EEG Project Dependencies")
    print("=" * 60)
    
    sys.path.insert(0, '/home/kevin/Projects/eeg2025/src')
    
    results = {}
    
    # Core dependencies
    packages = [
        ('numpy', 'NumPy'),
        ('torch', 'PyTorch'),
        ('scipy', 'SciPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('mne', 'MNE'),
        ('braindecode', 'Braindecode'),
        ('h5py', 'H5Py'),
    ]
    
    for module, name in packages:
        try:
            __import__(module)
            print(f"✅ {name}")
            results[name] = True
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            results[name] = False
    
    return all(results.values())

def test_project_models():
    """Test EEG project model imports"""
    print("\n" + "=" * 60)
    print("3. Testing EEG Project Models")
    print("=" * 60)
    
    sys.path.insert(0, '/home/kevin/Projects/eeg2025/src')
    
    models = [
        ('models.baseline.tcn', 'TemporalConvNet'),
        ('models.baseline.eegnex', 'EEGNeX'),
        ('models.baseline.cnn', 'BaselineCNN'),
    ]
    
    results = {}
    for module_path, model_name in models:
        try:
            parts = module_path.split('.')
            module = __import__(module_path, fromlist=[model_name])
            model_class = getattr(module, model_name)
            print(f"✅ {model_name}")
            results[model_name] = True
        except Exception as e:
            print(f"❌ {model_name}: {e}")
            results[model_name] = False
    
    return all(results.values())

def test_gpu_model():
    """Test model on GPU"""
    print("\n" + "=" * 60)
    print("4. Testing Model on GPU")
    print("=" * 60)
    
    try:
        import torch
        sys.path.insert(0, '/home/kevin/Projects/eeg2025/src')
        
        # Try simplest model first
        print("🧪 Testing BaselineCNN on GPU...")
        from models.baseline.cnn import BaselineCNN
        
        model = BaselineCNN(n_channels=64, n_outputs=1).cuda()
        x = torch.randn(2, 64, 1000).cuda()  # batch=2, channels=64, time=1000
        
        start = time.time()
        y = model(x)
        elapsed = (time.time() - start) * 1000
        
        print(f"✅ Model forward pass successful!")
        print(f"   Input: {x.shape}")
        print(f"   Output: {y.shape}")
        print(f"   Time: {elapsed:.2f}ms")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("\n" + "🧠 EEG2025 SDK Validation Test ".center(60, "="))
    print()
    
    results = []
    
    results.append(("PyTorch GPU", test_pytorch()))
    results.append(("Dependencies", test_eeg_imports()))
    
    # Only test models if dependencies are available
    if results[-1][1]:
        results.append(("Model Imports", test_project_models()))
        
        # Only test GPU if PyTorch GPU works
        if results[0][1]:
            results.append(("GPU Model", test_gpu_model()))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(r[1] for r in results)
    
    print("\n" + ("=" * 60))
    if all_passed:
        print("🎉 ALL TESTS PASSED - SDK is ready for GPU training!")
    else:
        print("⚠️  Some tests failed - install missing dependencies:")
        print("\n  source activate_sdk.sh")
        print("  sdk_pip install -r requirements.txt")
    print("=" * 60 + "\n")
    
    return 0 if all_passed else 1

if __name__ == '__main__':
    sys.exit(main())
