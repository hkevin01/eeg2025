#!/usr/bin/env python3
"""Debug data loading"""

from pathlib import Path
import mne

data_dir = Path("/home/kevin/Projects/eeg2025/data/raw/hbn")

print("🔍 Debug Data Loading")
print("="*60)

# Find subjects
subjects = sorted(list(data_dir.glob("sub-NDAR*")))
print(f"Found {len(subjects)} subjects")

if len(subjects) == 0:
    print("❌ No subjects found!")
    exit(1)

# Test first subject
subj_dir = subjects[0]
print(f"\n📂 Testing subject: {subj_dir.name}")

eeg_dir = subj_dir / "eeg"
print(f"   EEG dir exists: {eeg_dir.exists()}")

if not eeg_dir.exists():
    print("❌ EEG directory doesn't exist!")
    exit(1)

# Find .set files
set_files = list(eeg_dir.glob("*.set"))
print(f"   Found {len(set_files)} .set files")

if len(set_files) == 0:
    print("❌ No .set files found!")
    exit(1)

# Try loading first file
set_file = set_files[0]
print(f"\n📄 Loading: {set_file.name}")
print(f"   Size: {set_file.stat().st_size / (1024**2):.1f} MB")

try:
    print("   Loading with MNE...")
    raw = mne.io.read_raw_eeglab(set_file, preload=True, verbose=True)
    print(f"✅ Loaded successfully!")
    print(f"   Channels: {len(raw.ch_names)}")
    print(f"   Duration: {raw.times[-1]:.1f} seconds")
    print(f"   Sampling rate: {raw.info['sfreq']} Hz")
    
    # Get data
    print("\n   Getting data array...")
    data = raw.get_data()
    print(f"   Data shape: {data.shape}")
    
    # Calculate windows
    sfreq = raw.info['sfreq']
    window_samples = int(2.0 * sfreq)
    step = window_samples // 2
    n_windows = (data.shape[1] - window_samples) // step + 1
    print(f"   Window size: {window_samples} samples (2.0s)")
    print(f"   Step size: {step} samples (50% overlap)")
    print(f"   Expected windows: {n_windows}")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
