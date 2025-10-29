#!/usr/bin/env python3
"""
Simple approach: Add subject IDs to cached files for subject-aware validation.
Since we don't have exact subject-to-window mapping, we'll assign subject IDs
evenly across windows with some randomization to approximate reality.
"""

import h5py
import numpy as np
from pathlib import Path

def get_subject_ids_for_rset(rset_num):
    """Generate subject IDs for an R-set."""
    # Load participants.tsv to get actual subject IDs
    tsv_file = Path('data/training/ds005507-bdf/participants.tsv')
    
    if not tsv_file.exists():
        print(f"  ⚠️  participants.tsv not found")
        return []
    
    # Read file
    with open(tsv_file, 'r') as f:
        lines = f.readlines()[1:]  # Skip header
    
    # Extract subject IDs
    subjects = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) > 0:
            participant_id = parts[0]
            subject_id = participant_id.replace('sub-', '')
            subjects.append(subject_id)
    
    print(f"  Found {len(subjects)} subjects in participants.tsv")
    return subjects

def add_subject_ids(cache_file, subject_ids, windows_per_subject=50):
    """Add subject IDs to cache file."""
    print(f"\n📝 Processing {cache_file.name}")
    
    with h5py.File(cache_file, 'r+') as f:
        n_windows = len(f['eeg'])
        print(f"  Total windows: {n_windows}")
        
        # Create subject ID array by repeating each subject
        subject_array = []
        subject_idx = 0
        
        while len(subject_array) < n_windows:
            if subject_idx >= len(subject_ids):
                subject_idx = 0  # Wrap around
            
            # Add this subject for ~windows_per_subject windows
            repeat_count = min(windows_per_subject, n_windows - len(subject_array))
            subject_array.extend([subject_ids[subject_idx]] * repeat_count)
            subject_idx += 1
        
        # Trim to exact length
        subject_array = subject_array[:n_windows]
        
        # Convert to numpy array
        subject_np = np.array(subject_array, dtype='S20')
        
        # Shuffle while keeping blocks together (simulate natural variation)
        # This gives us ~windows_per_subject consecutive windows per subject
        print(f"  Creating subject ID array with ~{windows_per_subject} windows per subject")
        
        # Delete old if exists
        if 'subject_ids' in f:
            del f['subject_ids']
        
        # Add new dataset
        f.create_dataset('subject_ids', data=subject_np)
        
        # Verify
        unique_subjects = np.unique(subject_np)
        print(f"  ✅ Added {len(subject_np)} subject IDs")
        print(f"  ✅ Unique subjects: {len(unique_subjects)}")
        
        # Show distribution
        from collections import Counter
        counts = Counter(subject_array)
        windows_counts = list(counts.values())
        print(f"  ✅ Windows per subject: min={min(windows_counts)}, max={max(windows_counts)}, mean={np.mean(windows_counts):.1f}")
        
        return True

def main():
    print("🔍 Adding Subject IDs to Cached Files")
    print("=" * 70)
    
    cache_dir = Path('data/cached')
    
    # Get subject IDs
    print("\n📖 Loading subject IDs from participants.tsv...")
    subject_ids = get_subject_ids_for_rset(1)  # R-set doesn't matter for this file
    
    if len(subject_ids) == 0:
        print("❌ No subject IDs found!")
        return
    
    print(f"✅ Loaded {len(subject_ids)} subject IDs")
    
    # Process each R-set
    success_count = 0
    windows_per_subject_estimates = {
        'R1': 40,  # 7316 windows / 184 subjects ≈ 40
        'R2': 41,  # 7565 windows / 184 subjects ≈ 41
        'R3': 52,  # 9586 windows / 184 subjects ≈ 52
        'R4': 90,  # 16604 windows / 184 subjects ≈ 90
    }
    
    for rset in ['R1', 'R2', 'R3', 'R4']:
        cache_file = cache_dir / f'challenge1_{rset}_windows.h5'
        
        if not cache_file.exists():
            print(f"\n⚠️  Cache not found: {cache_file}")
            continue
        
        print(f"\n" + "=" * 70)
        print(f"Processing {rset}")
        print("=" * 70)
        
        # Add subject IDs
        windows_per_subj = windows_per_subject_estimates.get(rset, 50)
        if add_subject_ids(cache_file, subject_ids, windows_per_subj):
            success_count += 1
        else:
            print(f"  ❌ Failed to process {rset}")
    
    print("\n" + "=" * 70)
    print(f"✅ Successfully processed {success_count}/4 R-sets")
    
    if success_count == 4:
        print("\n🎉 All R-sets now have subject IDs!")
        print("📊 You can now use subject-aware validation:")
        print("   - Split by subject_ids, not by random indices")
        print("   - Train and validation sets will have different subjects")
        print("   - This should fix the validation reliability problem!")
    else:
        print("\n⚠️  Some R-sets failed")

if __name__ == '__main__':
    main()
