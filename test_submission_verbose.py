"""
Verbose test for submission.py with step-by-step progress.
"""
import sys
import os

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''

print("=" * 70)
print("VERBOSE SUBMISSION TEST - CPU MODE")
print("=" * 70)
print()

# Step 1: Import torch
print("Step 1/10: Importing PyTorch...")
sys.stdout.flush()
import torch
print(f"  ✓ PyTorch version: {torch.__version__}")
print(f"  ✓ CUDA available: {torch.cuda.is_available()}")
sys.stdout.flush()

# Step 2: Import submission module
print("\nStep 2/10: Importing submission.py...")
sys.stdout.flush()
try:
    from submission import Submission, select_device
    print("  ✓ Submission module imported")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.stdout.flush()

# Step 3: Test device selection
print("\nStep 3/10: Testing device selection...")
sys.stdout.flush()
try:
    device, device_info = select_device(verbose=True)
    print(f"  ✓ Device selected: {device}")
    print(f"  ✓ Device info: {device_info}")
except Exception as e:
    print(f"  ✗ Device selection failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.stdout.flush()

# Step 4: Create submission instance
print("\nStep 4/10: Creating Submission instance...")
sys.stdout.flush()
try:
    SFREQ = 100
    sub = Submission(SFREQ, DEVICE=torch.device('cpu'))
    print("  ✓ Submission instance created")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.stdout.flush()

# Step 5: Load Challenge 1 model
print("\nStep 5/10: Loading Challenge 1 model...")
sys.stdout.flush()
try:
    model_1 = sub.get_model_challenge_1()
    n_params = sum(p.numel() for p in model_1.parameters())
    print(f"  ✓ Model loaded: {n_params:,} parameters")
except Exception as e:
    print(f"  ✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.stdout.flush()

# Step 6: Create test data
print("\nStep 6/10: Creating test data...")
sys.stdout.flush()
try:
    batch_size = 4
    n_channels = 129
    n_times = int(2 * SFREQ)
    X_test = torch.randn(batch_size, n_channels, n_times, device=torch.device('cpu'))
    print(f"  ✓ Test data shape: {X_test.shape}")
    print(f"  ✓ Data range: [{X_test.min():.3f}, {X_test.max():.3f}]")
except Exception as e:
    print(f"  ✗ Failed: {e}")
    sys.exit(1)
sys.stdout.flush()

# Step 7: Test inference Challenge 1
print("\nStep 7/10: Testing Challenge 1 inference...")
sys.stdout.flush()
try:
    model_1.eval()
    print("  ✓ Model set to eval mode")
    
    with torch.inference_mode():
        print("  → Running forward pass...")
        sys.stdout.flush()
        y_pred = model_1(X_test)
        print(f"  ✓ Output shape: {y_pred.shape}")
        print(f"  ✓ Sample predictions: {y_pred[:3, 0].detach().cpu().numpy()}")
except Exception as e:
    print(f"  ✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.stdout.flush()

# Step 8: Clean up
print("\nStep 8/10: Cleaning up Challenge 1 model...")
sys.stdout.flush()
del model_1
print("  ✓ Model deleted")
sys.stdout.flush()

# Step 9: Load Challenge 2 model
print("\nStep 9/10: Loading Challenge 2 model...")
sys.stdout.flush()
try:
    model_2 = sub.get_model_challenge_2()
    n_params = sum(p.numel() for p in model_2.parameters())
    print(f"  ✓ Model loaded: {n_params:,} parameters")
except Exception as e:
    print(f"  ✗ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.stdout.flush()

# Step 10: Test inference Challenge 2
print("\nStep 10/10: Testing Challenge 2 inference...")
sys.stdout.flush()
try:
    model_2.eval()
    print("  ✓ Model set to eval mode")
    
    with torch.inference_mode():
        print("  → Running forward pass...")
        sys.stdout.flush()
        y_pred = model_2(X_test)
        print(f"  ✓ Output shape: {y_pred.shape}")
        print(f"  ✓ Sample predictions: {y_pred[:3, 0].detach().cpu().numpy()}")
except Exception as e:
    print(f"  ✗ Inference failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
sys.stdout.flush()

# Success!
print("\n" + "=" * 70)
print("ALL TESTS PASSED - SUBMISSION FILE READY FOR COMPETITION")
print("=" * 70)
print("\nNext steps:")
print("  1. Test with GPU: python submission.py")
print("  2. Package: zip submission.zip submission.py weights_challenge_*.pt")
print("  3. Submit to competition platform")
print("=" * 70)
