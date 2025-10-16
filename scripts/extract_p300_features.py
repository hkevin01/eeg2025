"""
Extract P300 features from Challenge 1 data for Phase 2 training
This creates a cache of P300 features to speed up training

NOTE: This uses the same preprocessing as train_challenge1_multi_release.py
"""

import sys
import os
import pickle
import numpy as np
from pathlib import Path
import torch
from tqdm import tqdm
import time
import logging

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.features.erp import ERPExtractor
from eegdash import EEGChallengeDataset
from braindecode.preprocessing import (
    create_windows_from_events,
    preprocess,
    Preprocessor,
)
from eegdash.hbn.windows import (
    add_extras_columns,
    add_aux_anchors,
)

print("="*70)
print("🧠 EXTRACTING P300 FEATURES FOR CHALLENGE 1")
print("="*70)
print()

# Configuration (match train_challenge1_multi_release.py)
RELEASES = ['R1', 'R2', 'R3']  # Same as Phase 1 training
CACHE_DIR = Path('data/processed/p300_cache')
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Trial window configuration (match training script)
SHIFT_AFTER_STIM = 0.5  # Shift 0.5s after stimulus onset
EPOCH_LEN_S = 2.0       # 2-second window
SFREQ = 100             # 100 Hz sampling rate
ANCHOR = "contrast_trial_start"

# Initialize extractor
extractor = ERPExtractor(sampling_rate=SFREQ)

# Standard channel names (129 channels)
CHANNEL_NAMES = [
    'Fp1', 'Fpz', 'Fp2', 'AF7', 'AF3', 'AFz', 'AF4', 'AF8',
    'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8',
    'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10',
    'T9', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'T8', 'T10',
    'TP9', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10',
    'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10',
    'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Iz',
] + [f'E{i}' for i in range(1, 61)]  # Add extra electrodes to make 129

def extract_features_for_release(release_name):
    """Extract P300 features for one release using same preprocessing as training"""
    print(f"\n📂 Processing {release_name}...")

    cache_file = CACHE_DIR / f'{release_name}_p300_features.pkl'

    if cache_file.exists():
        print(f"✅ Cache exists: {cache_file}")
        with open(cache_file, 'rb') as f:
            data = pickle.load(f)
        print(f"   Loaded {len(data['features'])} cached features")
        return data

    # Load release data (same as training script)
    print(f"   Loading EEG data...")
    try:
        dataset = EEGChallengeDataset(
            release=release_name,
            mini=False,
            query=dict(task="contrastChangeDetection"),
            cache_dir=Path('data/cache')
        )
        print(f"    Datasets: {len(dataset.datasets)}")
    except Exception as e:
        print(f"   ❌ Failed to load {release_name}: {e}")
        raise

    # Filter corrupted datasets (same as training script)
    print(f"    Checking for corrupted files...")
    valid_datasets = []
    for i, ds in enumerate(dataset.datasets):
        try:
            raw = ds.raw
            if raw is not None and raw.get_data().shape[1] > 0:
                valid_datasets.append(ds)
        except Exception:
            continue

    dataset.datasets = valid_datasets
    print(f"    Valid datasets: {len(valid_datasets)}")

    if len(valid_datasets) == 0:
        print(f"   ⚠️ No valid datasets, skipping {release_name}")
        return {'release': release_name, 'features': [], 'n_trials': 0, 'channel_names': CHANNEL_NAMES}

    # Preprocessing (same as training script)
    print(f"    Preprocessing...")
    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False, eog=False, ecg=False, exclude='bads'),
        Preprocessor(lambda data: np.clip(data, *np.percentile(data, [0.5, 99.5])), apply_on_array=True),
        Preprocessor('filter', l_freq=1.0, h_freq=40.0, method='iir'),
        Preprocessor(add_aux_anchors, apply_on_array=False),  # CRITICAL: Add event markers
    ]

    try:
        preprocess(dataset, preprocessors)
    except Exception as e:
        print(f"   ⚠️ Preprocessing error: {e}")

    # Filter out bad trials
    valid_trials = sum(1 for ds in dataset.datasets if ds.raw is not None and ds.raw.get_data().shape[1] > 0)
    print(f"    Valid trials after preprocessing: {valid_trials}")

    if valid_trials == 0:
        print(f"   ⚠️ No valid trials, skipping {release_name}")
        return {'release': release_name, 'features': [], 'n_trials': 0, 'channel_names': CHANNEL_NAMES}

    # Create windows from events (same as training script)
    print("    Creating windows from trials...")
    windows_dataset = create_windows_from_events(
        dataset,
        mapping={ANCHOR: 0},
        trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
        trial_stop_offset_samples=int((SHIFT_AFTER_STIM + EPOCH_LEN_S) * SFREQ),
        window_size_samples=int(EPOCH_LEN_S * SFREQ),
        window_stride_samples=SFREQ,
        preload=True,
    )

    print(f"    Windows created: {len(windows_dataset)}")

    # Add metadata (same as training script)
    print(f"    Adding metadata...")
    windows_dataset.datasets = add_extras_columns(windows_dataset.datasets)

    # Extract P300 features from each window
    print(f"    Extracting P300 features...")
    features_list = []

    for i in tqdm(range(len(windows_dataset)), desc=f"   {release_name} P300"):
        try:
            # Get window data and metadata
            X, y, window_idx = windows_dataset[i]  # X shape: (channels, samples)

            # Get response time from metadata
            ds = windows_dataset.datasets[window_idx[0]]
            rt = ds.rt_ms

            if rt is None or rt <= 0:
                continue

            # Extract P300 features
            p300_features = extractor.extract_p300(
                X,
                channel_names=CHANNEL_NAMES[:X.shape[0]],
                baseline_correct=True
            )

            # Store
            features_list.append({
                'window_idx': i,
                'dataset_idx': window_idx[0],
                'p300_features': p300_features,
                'raw_shape': X.shape,
                'target_rt': rt
            })

        except Exception as e:
            continue

    print(f"   ✅ Extracted {len(features_list)} trial features")

    # Save cache
    cache_data = {
        'release': release_name,
        'features': features_list,
        'n_trials': len(features_list),
        'channel_names': CHANNEL_NAMES
    }

    with open(cache_file, 'wb') as f:
        pickle.dump(cache_data, f)

    print(f"   💾 Saved to: {cache_file}")

    return cache_data


def main():
    """Extract features for all releases"""

    all_features = {}
    total_trials = 0

    for release in RELEASES:
        data = extract_features_for_release(release)
        all_features[release] = data
        total_trials += data['n_trials']

    print("\n" + "="*70)
    print("📊 EXTRACTION SUMMARY")
    print("="*70)
    for release, data in all_features.items():
        print(f"{release}: {data['n_trials']:,} trials")
    print(f"Total: {total_trials:,} trials")
    print()

    # Analyze P300 statistics
    print("📈 P300 STATISTICS")
    print("="*70)

    all_latencies = []
    all_amplitudes = []
    all_rts = []

    for release, data in all_features.items():
        for trial in data['features']:
            p300 = trial['p300_features']
            all_latencies.append(p300['p300_peak_latency'])
            all_amplitudes.append(p300['p300_peak_amplitude'])
            all_rts.append(trial['target_rt'])

    all_latencies = np.array(all_latencies)
    all_amplitudes = np.array(all_amplitudes)
    all_rts = np.array(all_rts)

    print(f"P300 Latency:  {all_latencies.mean():.1f} ± {all_latencies.std():.1f} ms")
    print(f"P300 Amplitude: {all_amplitudes.mean():.2f} ± {all_amplitudes.std():.2f} μV")
    print(f"Response Time:  {all_rts.mean():.1f} ± {all_rts.std():.1f} ms")
    print()

    # Check correlation (P300 latency should correlate with RT!)
    correlation = np.corrcoef(all_latencies, all_rts)[0, 1]
    print(f"💡 Correlation (P300 latency ↔ RT): {correlation:.3f}")
    if abs(correlation) > 0.1:
        print(f"   ✅ Good! P300 features should help prediction")
    else:
        print(f"   ⚠️  Weak correlation - may not help much")

    print()
    print("="*70)
    print("✅ FEATURE EXTRACTION COMPLETE!")
    print("="*70)
    print(f"📁 Cache directory: {CACHE_DIR}")
    print(f"📊 Ready for Phase 2 training!")
    print()


if __name__ == '__main__':
    main()
