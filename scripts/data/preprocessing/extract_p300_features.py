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
from typing import List, Dict
import torch
from tqdm import tqdm
import time
import logging
from multiprocessing import Pool, cpu_count

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
    annotate_trials_with_target,
)

print("="*70)
print("🧠 EXTRACTING P300 FEATURES FOR CHALLENGE 1")
print("="*70)
print()

# Check GPU/acceleration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_cpus = cpu_count()

if torch.cuda.is_available():
    print(f"🚀 GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    USE_GPU = True
    BATCH_SIZE = 512
else:
    print(f"💻 No GPU detected (PyTorch needs ROCm build for AMD)")
    print(f"🔧 Using CPU parallelization: {n_cpus} cores")
    USE_GPU = False
    BATCH_SIZE = 128
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

# Parallel processing
N_WORKERS = max(1, n_cpus - 2)  # Leave 2 cores for system

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


def extract_p300_batch_vectorized(
    batch_data: np.ndarray,
    sampling_rate: int = 100
) -> List[Dict[str, float]]:
    """
    Vectorized batch P300 feature extraction (CPU-optimized with numpy)
    Much faster than loop-based extraction!

    Args:
        batch_data: Array of shape [batch_size, channels, time_samples]
        sampling_rate: Sampling rate in Hz

    Returns:
        List of feature dictionaries for each trial
    """
    batch_size, n_channels, n_samples = batch_data.shape

    # Baseline correction (first 200ms = 20 samples @ 100Hz) - vectorized!
    baseline = batch_data[:, :, :20].mean(axis=2, keepdims=True)
    batch_data = batch_data - baseline

    # Select parietal channels (middle 1/3 approximation)
    parietal_start = n_channels // 3
    parietal_end = 2 * n_channels // 3
    parietal_data = batch_data[:, parietal_start:parietal_end, :]

    # Average over parietal channels - vectorized
    parietal_avg = parietal_data.mean(axis=1)  # [batch_size, time_samples]

    # P300 window: 300-600ms @ 100Hz = samples 30-60
    p300_start = int(300 * sampling_rate / 1000)
    p300_end = int(600 * sampling_rate / 1000)
    p300_window = parietal_avg[:, p300_start:p300_end]  # [batch_size, window_size]

    # Extract features for entire batch at once - all vectorized!
    peak_idx = p300_window.argmax(axis=1)  # [batch_size]
    peak_latency = 300 + (peak_idx * 1000 / sampling_rate)  # Convert to ms

    # Get peak amplitudes using fancy indexing
    batch_indices = np.arange(batch_size)
    peak_amplitude = p300_window[batch_indices, peak_idx]

    mean_amplitude = p300_window.mean(axis=1)

    # Area under curve (trapezoidal rule) - vectorized
    area_under_curve = np.trapz(p300_window, dx=1.0/sampling_rate, axis=1)

    # Onset detection (first point > 25% of peak) - vectorized
    threshold = peak_amplitude[:, np.newaxis] * 0.25
    onset_mask = p300_window > threshold
    onset_idx = onset_mask.argmax(axis=1)
    onset_latency = 300 + (onset_idx * 1000 / sampling_rate)

    # Rise time (onset to peak)
    rise_time = peak_latency - onset_latency

    # Create feature dictionaries
    features_list = []
    for i in range(batch_size):
        features_list.append({
            'p300_peak_latency': float(peak_latency[i]),
            'p300_peak_amplitude': float(peak_amplitude[i]),
            'p300_mean_amplitude': float(mean_amplitude[i]),
            'p300_area_under_curve': float(area_under_curve[i]),
            'p300_onset_latency': float(onset_latency[i]),
            'p300_rise_time': float(rise_time[i]),
        })

    return features_list
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

    # Preprocessing (EXACTLY same as training script)
    print(f"    Preprocessing...")
    preprocessors = [
        Preprocessor(
            annotate_trials_with_target,
            apply_on_array=False,
            target_field="rt_from_stimulus",
            epoch_length=EPOCH_LEN_S,
            require_stimulus=True,
            require_response=True,
        ),
        Preprocessor(add_aux_anchors, apply_on_array=False),
    ]

    try:
        preprocess(dataset, preprocessors, n_jobs=-1)
    except Exception as e:
        print(f"   ⚠️ Preprocessing error: {e}")
        return {'release': release_name, 'features': [], 'n_trials': 0, 'channel_names': CHANNEL_NAMES}

    # Filter out datasets with no valid trials (check annotations)
    valid_trials = sum(1 for ds in dataset.datasets if len(ds.raw.annotations) > 0)
    print(f"    Datasets with valid trials after preprocessing: {valid_trials}/{len(valid_datasets)}")

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

    if len(windows_dataset) == 0:
        print(f"   ⚠️ No windows created, skipping {release_name}")
        return {'release': release_name, 'features': [], 'n_trials': 0, 'channel_names': CHANNEL_NAMES}

    # Add metadata (EXACTLY same as training script)
    print(f"    Injecting trial metadata...")
    try:
        windows_dataset = add_extras_columns(
            windows_dataset,  # Windowed dataset
            dataset,          # Original preprocessed dataset with annotations
            desc="contrast_trial_start",  # Annotation description
            keys=("rt_from_stimulus", "target", "rt_from_trialstart",
                  "stimulus_onset", "response_onset", "correct", "response_type")
        )
    except Exception as e:
        print(f"   ⚠️ Metadata injection error: {e}")
        return {'release': release_name, 'features': [], 'n_trials': 0, 'channel_names': CHANNEL_NAMES}

    # Extract metadata and P300 features from each window
    print(f"    Extracting P300 features...")

    # Get metadata DataFrame (has rt_from_stimulus column)
    metadata_df = windows_dataset.get_metadata()
    print(f"    Metadata: {len(metadata_df)} rows")

    if 'rt_from_stimulus' not in metadata_df.columns:
        print(f"   ⚠️ rt_from_stimulus not in metadata columns!")
        print(f"   Available columns: {metadata_df.columns.tolist()}")
        return {'release': release_name, 'features': [], 'n_trials': 0, 'channel_names': CHANNEL_NAMES}

    rt_values = metadata_df['rt_from_stimulus'].values
    features_list = []

    # Vectorized batch processing (much faster than loop!)
    print(f"    🚀 Using vectorized batch processing (batch_size={BATCH_SIZE}, {N_WORKERS} workers)")

    n_windows = len(windows_dataset)
    n_batches = (n_windows + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in tqdm(range(n_batches), desc=f"   {release_name} P300 (Fast)"):
        batch_start = batch_idx * BATCH_SIZE
        batch_end = min(batch_start + BATCH_SIZE, n_windows)

        # Collect batch data
        batch_data = []
        batch_rts = []
        batch_indices = []

        for i in range(batch_start, batch_end):
            try:
                X, y, window_idx = windows_dataset[i]
                rt = rt_values[i]

                if np.isnan(rt) or rt <= 0:
                    continue

                batch_data.append(X)
                batch_rts.append(float(rt))
                batch_indices.append({
                    'window_idx': i,
                    'dataset_idx': window_idx[0] if isinstance(window_idx, (list, tuple)) else window_idx,
                    'raw_shape': X.shape
                })
            except Exception:
                continue

        if len(batch_data) == 0:
            continue

        # Convert to numpy array and extract features in batch (vectorized!)
        try:
            batch_array = np.array(batch_data)

            # Extract P300 features in batch using vectorized operations
            batch_features = extract_p300_batch_vectorized(
                batch_array,
                sampling_rate=SFREQ
            )

            # Store results
            for idx, (feats, rt, meta) in enumerate(zip(batch_features, batch_rts, batch_indices)):
                features_list.append({
                    'window_idx': meta['window_idx'],
                    'dataset_idx': meta['dataset_idx'],
                    'p300_features': feats,
                    'raw_shape': meta['raw_shape'],
                    'target_rt': rt
                })
        except Exception as e:
            # Fallback to single extraction if batch fails
            print(f"\n   ⚠️ Batch failed, using fallback: {e}")
            for idx, (X, rt, meta) in enumerate(zip(batch_data, batch_rts, batch_indices)):
                p300_features = extractor.extract_p300(
                    X,
                    channel_names=CHANNEL_NAMES[:X.shape[0]],
                    baseline_correct=True
                )
                features_list.append({
                    'window_idx': meta['window_idx'],
                    'dataset_idx': meta['dataset_idx'],
                    'p300_features': p300_features,
                    'raw_shape': meta['raw_shape'],
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
