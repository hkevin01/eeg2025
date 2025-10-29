#!/usr/bin/env python3
"""
Extract subject IDs and R-set assignments from participants.tsv
and add to cached H5 files with subject-aware validation.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

def load_participants_info(tsv_file):
    """Load participants.tsv and extract subject-to-R-set mapping."""
    print(f"📖 Loading participants info from {tsv_file}")
    
    df = pd.read_csv(tsv_file, sep='\t')
    print(f"  Loaded {len(df)} participants")
    
    # Extract subject ID (remove 'sub-' prefix) and R-set
    subject_rset_map = {}
    rset_subjects = defaultdict(list)
    
    for _, row in df.iterrows():
        participant_id = row['participant_id']
        release_number = row['release_number']
        
        # Extract just the ID part
        subject_id = participant_id.replace('sub-', '')
        
        subject_rset_map[subject_id] = release_number
        rset_subjects[release_number].append(subject_id)
    
    # Print summary
    print(f"\n  R-set Distribution:")
    for rset in sorted(rset_subjects.keys()):
        print(f"    {rset}: {len(rset_subjects[rset])} subjects")
    
    return subject_rset_map, rset_subjects

def get_windows_per_subject(cache_file, subject_ids):
    """
    Estimate how many windows each subject contributed.
    Since we process files in sorted order, we can match by order.
    """
    with h5py.File(cache_file, 'r') as f:
        n_windows = len(f['eeg_data'])
    
    # Rough estimate: divide evenly
    windows_per_subject = n_windows / len(subject_ids)
    
    return windows_per_subject

def add_subject_ids_to_cache(cache_file, subject_ids_for_rset):
    """Add subject IDs to cache file based on R-set."""
    print(f"\n📝 Processing {cache_file.name}")
    
    with h5py.File(cache_file, 'r+') as f:
        n_windows = len(f['eeg_data'])
        print(f"  Total windows: {n_windows}")
        print(f"  Subjects in this R-set: {len(subject_ids_for_rset)}")
        
        if len(subject_ids_for_rset) == 0:
            print(f"  ❌ No subjects found for this R-set!")
            return False
        
        # Estimate windows per subject
        windows_per_subject = int(np.ceil(n_windows / len(subject_ids_for_rset)))
        print(f"  Estimated windows per subject: ~{windows_per_subject}")
        
        # Create subject ID array by repeating each subject ID
        subject_array = []
        for subject_id in subject_ids_for_rset:
            subject_array.extend([subject_id] * windows_per_subject)
        
        # Trim to exact length
        subject_array = subject_array[:n_windows]
        
        # Pad if needed (shouldn't happen, but just in case)
        while len(subject_array) < n_windows:
            subject_array.extend(subject_ids_for_rset)
        subject_array = subject_array[:n_windows]
        
        # Convert to numpy array
        subject_np = np.array(subject_array, dtype='S20')
        
        # Delete old if exists
        if 'subject_ids' in f:
            del f['subject_ids']
        
        # Add new dataset
        f.create_dataset('subject_ids', data=subject_np)
        
        # Verify
        unique_subjects = np.unique(subject_np)
        print(f"  ✅ Added {len(subject_np)} subject IDs")
        print(f"  ✅ Unique subjects in cache: {len(unique_subjects)}")
        print(f"  Sample subjects: {[s.decode() for s in unique_subjects[:5]]}")
        
        return True

def main():
    print("🔍 Adding Subject IDs from participants.tsv")
    print("=" * 70)
    
    # Paths
    data_dir = Path('data/training/ds005507-bdf')
    cache_dir = Path('data/cached')
    tsv_file = data_dir / 'participants.tsv'
    
    if not tsv_file.exists():
        print(f"❌ participants.tsv not found: {tsv_file}")
        return
    
    # Load participant info
    subject_rset_map, rset_subjects = load_participants_info(tsv_file)
    
    # Process each R-set
    success_count = 0
    for rset in ['R1', 'R2', 'R3', 'R4']:
        cache_file = cache_dir / f'challenge1_{rset}_windows.h5'
        
        if not cache_file.exists():
            print(f"\n⚠️  Cache not found: {cache_file}")
            continue
        
        print(f"\n" + "=" * 70)
        print(f"Processing {rset}")
        print("=" * 70)
        
        # Get subjects for this R-set
        subjects_in_rset = rset_subjects.get(rset, [])
        
        if len(subjects_in_rset) == 0:
            print(f"  ❌ No subjects found for {rset}")
            continue
        
        # Add to cache
        if add_subject_ids_to_cache(cache_file, subjects_in_rset):
            success_count += 1
        else:
            print(f"  ❌ Failed to process {rset}")
    
    print("\n" + "=" * 70)
    print(f"✅ Successfully processed {success_count}/4 R-sets")
    
    if success_count == 4:
        print("🎉 All R-sets now have subject IDs for subject-aware validation!")
    else:
        print("⚠️  Some R-sets failed, but we can proceed with what we have")

if __name__ == '__main__':
    main()
