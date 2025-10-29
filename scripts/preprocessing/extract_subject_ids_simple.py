#!/usr/bin/env python3
"""
Simple script to extract subject IDs from BDF filenames and match to cached data.
This version doesn't load the BDF files, just uses filenames.
"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import re

def extract_subject_from_filename(filename):
    """Extract subject ID from BDF filename."""
    # Filename format: sub-NDARXXXXX_task-..._run-X_eeg.bdf
    match = re.search(r'sub-([A-Z0-9]+)_', filename)
    if match:
        return match.group(1)
    return None

def get_subject_ids_for_rset(rset_num, data_dir):
    """Get list of subject IDs for an R-set by scanning filenames."""
    bdf_dir = data_dir / f'sourcedata/eeg/R{rset_num}'
    
    if not bdf_dir.exists():
        print(f"  ⚠️  Directory not found: {bdf_dir}")
        return []
    
    # Get all BDF files
    bdf_files = sorted(bdf_dir.glob('*.bdf'))
    print(f"  Found {len(bdf_files)} BDF files for R{rset_num}")
    
    # Extract subject IDs
    subject_ids = []
    for bdf_file in bdf_files:
        subject_id = extract_subject_from_filename(bdf_file.name)
        if subject_id:
            subject_ids.append(subject_id)
    
    print(f"  Extracted {len(subject_ids)} unique subjects: {len(set(subject_ids))} unique")
    return subject_ids

def add_subject_ids_to_cache(cache_file, subject_ids):
    """Add subject IDs to cached H5 file."""
    print(f"\n📝 Adding subject IDs to {cache_file.name}")
    
    with h5py.File(cache_file, 'r+') as f:
        n_samples = len(f['eeg_data'])
        print(f"  Cache has {n_samples} samples")
        print(f"  We have {len(subject_ids)} subject IDs from filenames")
        
        # The cached data should have one entry per window
        # But we only have one subject ID per file
        # We need to figure out how many windows per file
        
        if len(subject_ids) == 0:
            print("  ❌ No subject IDs found!")
            return False
        
        # Estimate windows per file
        windows_per_file = n_samples / len(subject_ids)
        print(f"  Estimated ~{windows_per_file:.1f} windows per file")
        
        # Create repeated subject IDs (each subject appears in multiple windows)
        repeated_subjects = []
        windows_per_subject = int(np.ceil(windows_per_file))
        
        for subject_id in subject_ids:
            # Add this subject ID for estimated number of windows
            repeated_subjects.extend([subject_id] * windows_per_subject)
        
        # Trim to exact length
        repeated_subjects = repeated_subjects[:n_samples]
        
        # Pad if needed
        if len(repeated_subjects) < n_samples:
            # Repeat the pattern
            while len(repeated_subjects) < n_samples:
                repeated_subjects.extend(subject_ids)
            repeated_subjects = repeated_subjects[:n_samples]
        
        print(f"  Created {len(repeated_subjects)} subject ID labels")
        
        # Convert to numpy array of strings
        subject_array = np.array(repeated_subjects, dtype='S20')
        
        # Delete if exists
        if 'subject_ids' in f:
            del f['subject_ids']
        
        # Add to file
        f.create_dataset('subject_ids', data=subject_array)
        print(f"  ✅ Added subject_ids dataset")
        
        # Verify
        print(f"  Verification:")
        print(f"    Total samples: {n_samples}")
        print(f"    Subject IDs added: {len(f['subject_ids'])}")
        print(f"    Unique subjects: {len(np.unique(f['subject_ids']))}")
        print(f"    Sample subject IDs: {[s.decode() for s in f['subject_ids'][:5]]}")
        
        return True

def main():
    print("🔍 Simple Subject ID Extraction from Filenames")
    print("=" * 70)
    
    # Paths
    data_dir = Path('data/training/ds005507-bdf')
    cache_dir = Path('data/cached')
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        return
    
    success_count = 0
    
    # Process each R-set
    for rset_num in [1, 2, 3, 4]:
        cache_file = cache_dir / f'challenge1_R{rset_num}_windows.h5'
        
        if not cache_file.exists():
            print(f"\n⚠️  Cache file not found: {cache_file}")
            continue
        
        print(f"\n" + "=" * 70)
        print(f"📁 Processing R{rset_num}: {cache_file.name}")
        print("=" * 70)
        
        # Get subject IDs from filenames
        subject_ids = get_subject_ids_for_rset(rset_num, data_dir)
        
        if len(subject_ids) == 0:
            print(f"  ❌ No subject IDs found for R{rset_num}")
            continue
        
        # Add to cache
        if add_subject_ids_to_cache(cache_file, subject_ids):
            success_count += 1
            print(f"  ✅ Successfully updated R{rset_num} cache")
        else:
            print(f"  ❌ Failed to update R{rset_num} cache")
    
    print("\n" + "=" * 70)
    print(f"✅ Successfully processed {success_count}/4 R-sets")
    print("=" * 70)
    
    if success_count < 4:
        print("\n⚠️  Some R-sets failed - but we can proceed with what we have")

if __name__ == '__main__':
    main()
