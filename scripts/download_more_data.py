#!/usr/bin/env python3
"""Download more HBN subjects - Simple and Safe"""

import os
from pathlib import Path
import mne

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "raw" / "hbn"
TARGET_SUBJECTS = 22  # We have 12, get 10 more
MAX_DOWNLOAD_ATTEMPTS = 30  # Try up to 30 subjects to get 10 good ones

print("🔽 Downloading More HBN Data")
print("=" * 60)
print(f"Target: {TARGET_SUBJECTS} subjects with EEG data")
print(f"Data dir: {DATA_DIR}")
print("=" * 60)

# Count current subjects with EEG
current_subjects = []
for subj_dir in DATA_DIR.glob("sub-NDAR*"):
    if (subj_dir / "eeg").exists():
        current_subjects.append(subj_dir.name)

print(f"\n✅ Current subjects with EEG: {len(current_subjects)}")
print(f"📥 Need to download: {TARGET_SUBJECTS - len(current_subjects)} more")

if len(current_subjects) >= TARGET_SUBJECTS:
    print("\n✨ Already have enough data!")
    exit(0)

# Download using openneuro
print("\n🌐 Starting download from OpenNeuro...")
print("   This will take ~15-30 minutes")
print("   Dataset: ds004186 (HBN)")

try:
    # Try to download with include pattern for efficiency
    download_path = mne.datasets.openneuro.data_path(
        dataset='ds004186',
        path=DATA_DIR.parent.parent,
        verbose=True
    )
    
    print(f"\n✅ Download complete!")
    print(f"   Path: {download_path}")
    
    # Count again
    new_subjects = []
    for subj_dir in DATA_DIR.glob("sub-NDAR*"):
        if (subj_dir / "eeg").exists():
            new_subjects.append(subj_dir.name)
    
    print(f"\n📊 Final count: {len(new_subjects)} subjects with EEG")
    
    if len(new_subjects) >= TARGET_SUBJECTS:
        print("✨ Success! Have enough data now")
    else:
        print(f"⚠️  Still need {TARGET_SUBJECTS - len(new_subjects)} more subjects")
        print("   You can run this script again or continue with current data")
    
except Exception as e:
    print(f"\n❌ Download failed: {e}")
    print("\n💡 Options:")
    print("   1. Continue with current 12 subjects")
    print("   2. Manually download from https://openneuro.org/datasets/ds004186")
    print("   3. Try again later")

print("\n" + "=" * 60)
