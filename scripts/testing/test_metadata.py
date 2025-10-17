#!/usr/bin/env python3
"""Quick test to check what's in metadata"""
import sys
sys.path.insert(0, 'venv/lib/python3.12/site-packages')

from eegdash import EEGChallengeDataset
from braindecode.preprocessing import create_fixed_length_windows

# Load one dataset
dataset = EEGChallengeDataset(release='R1', query=dict(task="RestingState"))
print(f"Loaded {len(dataset)} datasets")

# Get first dataset
raw_data = dataset.datasets[0]
print(f"\nFirst dataset keys: {raw_data.keys() if hasattr(raw_data, 'keys') else 'not a dict'}")

# Check description
if hasattr(raw_data, 'description'):
    print(f"Description keys: {raw_data.description.keys()}")
    print(f"\nChecking for externalizing:")
    print(f"  'externalizing' in description: {'externalizing' in raw_data.description}")
    
    # Print all keys
    print(f"\nAll description keys:")
    for key in sorted(raw_data.description.keys()):
        value = raw_data.description[key]
        if not isinstance(value, (list, dict)) or len(str(value)) < 100:
            print(f"  {key}: {value}")

# Try creating windows and checking metadata
print(f"\nCreating windows to check metadata...")
windows_ds = create_fixed_length_windows(
    dataset[:1],  # Just first dataset
    start_offset_samples=0,
    stop_offset_samples=None,
    window_size_samples=200,
    window_stride_samples=100,
    drop_last_window=False,
    mapping=None,
    preload=False,
    targets_from="metadata",
    on_missing="ignore",
)

if len(windows_ds) > 0:
    X, y, metadata = windows_ds[0]
    print(f"\nFirst window metadata type: {type(metadata)}")
    if isinstance(metadata, list):
        print(f"Metadata is list of length: {len(metadata)}")
        if len(metadata) > 0:
            print(f"First item type: {type(metadata[0])}")
            print(f"First item keys: {metadata[0].keys() if hasattr(metadata[0], 'keys') else 'not a dict'}")
            if hasattr(metadata[0], 'keys'):
                print(f"First item: {dict(list(metadata[0].items())[:10])}")
    elif isinstance(metadata, dict):
        print(f"Metadata keys: {metadata.keys()}")
        print(f"Metadata: {dict(list(metadata.items())[:10])}")
    else:
        print(f"Metadata: {metadata}")

