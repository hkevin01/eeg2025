#!/usr/bin/env python3
"""
Add subject IDs to cached H5 files by analyzing the source BDF files.
This enables subject-aware train/val splitting to prevent leakage.
"""
import h5py
import numpy as np
from pathlib import Path
import mne
from tqdm import tqdm
import re

def extract_subject_from_filename(bdf_path):
    """Extract subject ID from BDF filename (e.g., sub-NDARJZ526HN3)"""
    match = re.search(r'sub-([A-Z0-9]+)', str(bdf_path))
    return match.group(1) if match else None

def get_subject_mapping_for_rset(rset_num):
    """
    Build mapping of window index to subject ID for a given R-set.
    """
    print(f"\n🔍 Processing R{rset_num}...")
    
    # Find all BDF files for this R-set
    data_dir = Path(f"data/training/ds005507-bdf")
    bdf_files = sorted(data_dir.glob(f"*/eeg/*_run-{rset_num}_eeg.bdf"))
    
    print(f"  Found {len(bdf_files)} BDF files for R{rset_num}")
    
    # Build subject mapping based on window extraction
    subject_ids = []
    window_counts = {}
    
    for bdf_file in tqdm(bdf_files, desc=f"  Analyzing R{rset_num} files"):
        subject_id = extract_subject_from_filename(bdf_file)
        if not subject_id:
            continue
        
        try:
            # Load raw data to count windows
            raw = mne.io.read_raw_bdf(bdf_file, preload=False, verbose=False)
            events = mne.find_events(raw, stim_channel='Status', verbose=False)
            
            # Filter for relevant events (11, 12, 21, 22, 31, 32)
            relevant_events = events[np.isin(events[:, 2], [11, 12, 21, 22, 31, 32])]
            n_windows = len(relevant_events)
            
            # Add this many subject IDs
            subject_ids.extend([subject_id] * n_windows)
            window_counts[subject_id] = window_counts.get(subject_id, 0) + n_windows
            
        except Exception as e:
            print(f"    ⚠️  Error processing {bdf_file.name}: {e}")
            continue
    
    print(f"  Total windows: {len(subject_ids)}")
    print(f"  Unique subjects: {len(window_counts)}")
    print(f"  Samples per subject: {list(window_counts.items())[:5]}...")
    
    return np.array(subject_ids), window_counts

def add_subject_ids_to_h5(rset_num):
    """
    Add subject_id dataset to existing H5 file.
    """
    h5_file = Path(f"data/cached/challenge1_R{rset_num}_windows.h5")
    
    if not h5_file.exists():
        print(f"❌ File not found: {h5_file}")
        return False
    
    print(f"\n📁 Processing: {h5_file.name}")
    
    # Check current structure
    with h5py.File(h5_file, 'r') as f:
        n_samples = f['eeg'].shape[0]
        print(f"  Current samples: {n_samples}")
        
        if 'subject_id' in f.keys():
            print(f"  ⚠️  subject_id already exists, skipping...")
            return True
    
    # Get subject mapping
    subject_ids, counts = get_subject_mapping_for_rset(rset_num)
    
    # Verify counts match
    if len(subject_ids) != n_samples:
        print(f"  ❌ Mismatch: {len(subject_ids)} subject IDs vs {n_samples} samples")
        print(f"     This might happen if preprocessing filtered some windows")
        print(f"     We'll need to match by order - assumes same processing")
        
        if len(subject_ids) < n_samples:
            print(f"  ❌ Not enough subject IDs generated")
            return False
        elif len(subject_ids) > n_samples:
            print(f"  ⚠️  Trimming subject IDs to match sample count")
            subject_ids = subject_ids[:n_samples]
    
    # Add subject_id dataset
    with h5py.File(h5_file, 'a') as f:
        # Convert string IDs to integer hashes for storage efficiency
        unique_subjects = sorted(set(subject_ids))
        subject_to_int = {subj: i for i, subj in enumerate(unique_subjects)}
        subject_int_ids = np.array([subject_to_int[s] for s in subject_ids])
        
        # Create dataset
        f.create_dataset('subject_id', data=subject_int_ids, dtype='int32')
        
        # Store mapping as attributes
        f['subject_id'].attrs['subject_names'] = ','.join(unique_subjects)
        f['subject_id'].attrs['n_unique'] = len(unique_subjects)
        
        print(f"  ✅ Added subject_id dataset")
        print(f"     Unique subjects: {len(unique_subjects)}")
        print(f"     Subject names (first 5): {unique_subjects[:5]}")
    
    return True

def main():
    print("🧠 Adding Subject IDs to Cached Data")
    print("=" * 70)
    
    success_count = 0
    for rset in [1, 2, 3, 4]:
        try:
            if add_subject_ids_to_h5(rset):
                success_count += 1
        except Exception as e:
            print(f"❌ Error processing R{rset}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"✅ Successfully processed {success_count}/4 R-sets")
    
    if success_count == 4:
        print(f"\n🎉 All R-sets now have subject IDs!")
        print(f"   Ready for subject-aware train/val splitting")
    else:
        print(f"\n⚠️  Some R-sets failed - check errors above")

if __name__ == "__main__":
    main()
