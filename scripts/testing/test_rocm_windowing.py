#!/usr/bin/env python3
"""
Test to isolate torch.arange ROCm bug in braindecode windowing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 80)
print("Testing ROCm + Braindecode Windowing")
print("=" * 80)

# Test 1: Basic torch operations
print("\n1️⃣ Testing basic torch operations...")
import torch
print(f"   PyTorch: {torch.__version__}")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   Device: {torch.cuda.get_device_name(0)}")

# Test 2: torch.arange on CPU
print("\n2️⃣ Testing torch.arange on CPU...")
try:
    x = torch.arange(0, 10, 0.1, device='cpu')
    print(f"   ✅ CPU arange works: shape {x.shape}")
except Exception as e:
    print(f"   ❌ CPU arange failed: {e}")

# Test 3: torch.arange on GPU
if torch.cuda.is_available():
    print("\n3️⃣ Testing torch.arange on GPU...")
    try:
        x = torch.arange(0, 10, 0.1, device='cuda')
        print(f"   ✅ GPU arange works: shape {x.shape}")
    except Exception as e:
        print(f"   ❌ GPU arange failed: {e}")

# Test 4: Load a single EEG file
print("\n4️⃣ Testing EEG data loading...")
try:
    from eegdash import EEGChallengeDataset
    dataset = EEGChallengeDataset(
        release='R1',
        mini=True,  # Use mini for quick test
        query=dict(task="contrastChangeDetection"),
        cache_dir=Path('data/raw')
    )
    print(f"   ✅ Loaded {len(dataset.datasets)} subjects")
    
    # Get first valid subject
    first_ds = dataset.datasets[0]
    print(f"   Subject: {first_ds.description}")
    print(f"   Channels: {len(first_ds.raw.ch_names)}")
    print(f"   Duration: {first_ds.raw.n_times / first_ds.raw.info['sfreq']:.1f}s")
    
except Exception as e:
    print(f"   ❌ EEG loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: MNE preprocessing
print("\n5️⃣ Testing MNE preprocessing...")
try:
    from braindecode.preprocessing import Preprocessor, preprocess
    import numpy as np
    
    preprocessors = [
        Preprocessor('set_eeg_reference', ref_channels='average', ch_type='eeg'),
        Preprocessor(lambda data: np.clip(data, -800e-6, 800e-6), apply_on_array=True),
    ]
    
    # Test on single subject
    single_ds = EEGChallengeDataset(
        release='R1',
        mini=True,
        query=dict(task="contrastChangeDetection"),
        cache_dir=Path('data/raw')
    )
    single_ds.datasets = [single_ds.datasets[0]]  # Just one subject
    
    preprocess(single_ds, preprocessors)
    print(f"   ✅ Preprocessing successful")
    
except Exception as e:
    print(f"   ❌ Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Braindecode windowing - THE CRITICAL TEST
print("\n6️⃣ Testing braindecode create_windows_from_events...")
print("   (This is where the arange error likely occurs)")

try:
    from braindecode.preprocessing import create_windows_from_events
    
    # Force CPU mode for torch during this operation
    print("   6a. Testing with CPU-forced torch.arange...")
    
    # Monkey patch torch.arange to CPU
    _original_arange = torch.arange
    def cpu_arange(*args, **kwargs):
        kwargs.pop('device', None)
        result = _original_arange(*args, device='cpu', **kwargs)
        return result
    
    torch.arange = cpu_arange
    
    windows_ds = create_windows_from_events(
        single_ds,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        picks='eeg',
        preload=True
    )
    
    print(f"   ✅ Windowing successful: {len(windows_ds)} windows")
    
    # Restore original
    torch.arange = _original_arange
    
except Exception as e:
    print(f"   ❌ Windowing failed: {e}")
    import traceback
    traceback.print_exc()
    
    # Try to get more details
    print("\n   Detailed error analysis:")
    print(f"   Error type: {type(e).__name__}")
    print(f"   Error message: {str(e)}")
    
    sys.exit(1)

# Test 7: Try without monkey patch to see if it fails
print("\n7️⃣ Testing windowing WITHOUT CPU-forced arange (to confirm bug)...")
try:
    # Load fresh dataset
    test_ds = EEGChallengeDataset(
        release='R1',
        mini=True,
        query=dict(task="contrastChangeDetection"),
        cache_dir=Path('data/raw')
    )
    test_ds.datasets = [test_ds.datasets[0]]
    
    preprocessors = [
        Preprocessor('set_eeg_reference', ref_channels='average', ch_type='eeg'),
        Preprocessor(lambda data: np.clip(data, -800e-6, 800e-6), apply_on_array=True),
    ]
    preprocess(test_ds, preprocessors)
    
    # No monkey patch - use default torch.arange
    windows_ds = create_windows_from_events(
        test_ds,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        picks='eeg',
        preload=True
    )
    
    print(f"   ✅ Works without patch! {len(windows_ds)} windows")
    print("   ℹ️  The bug may have been fixed or doesn't occur in this configuration")
    
except Exception as e:
    print(f"   ❌ CONFIRMED BUG: {e}")
    print("   ℹ️  This confirms the arange bug exists and our patch is needed")

print("\n" + "=" * 80)
print("✅ Test complete!")
print("=" * 80)
