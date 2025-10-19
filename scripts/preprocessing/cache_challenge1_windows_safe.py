#!/usr/bin/env python3
"""
MEMORY-SAFE Challenge 1 Window Caching with Comprehensive Error Handling

Features:
- Memory monitoring and safety checks (stops at 85% RAM)
- Resume capability (skips already processed releases)
- Detailed logging with timestamps
- Crash recovery and checkpointing
- Progress tracking

Usage:
    python cache_challenge1_windows_safe.py [--mini] [--releases R1 R2 R3 R4]
"""
import sys
from pathlib import Path
import time
import h5py
import numpy as np
import logging
import psutil
import argparse
from datetime import datetime
from braindecode.preprocessing import Preprocessor, preprocess
from braindecode.preprocessing import create_windows_from_events

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / "src"))

from eegdash import EEGChallengeDataset
from eegdash.hbn.windows import (
    annotate_trials_with_target,
    add_aux_anchors,
    add_extras_columns,
    keep_only_recordings_with,
)

# Configuration
CACHE_DIR = Path('data/raw')
OUTPUT_DIR = Path('data/cached')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SFREQ = 100
EPOCH_LEN_S = 2.0
SHIFT_AFTER_STIM = 0.5

# Memory safety
MAX_MEMORY_PERCENT = 85

# Setup logging
log_dir = Path("logs/preprocessing")
log_dir.mkdir(parents=True, exist_ok=True)
log_file = log_dir / f"cache_safe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_memory_safe():
    """Check if memory usage is safe"""
    memory = psutil.virtual_memory()
    return memory.percent < MAX_MEMORY_PERCENT, memory.percent

def log_memory():
    """Log current memory status"""
    memory = psutil.virtual_memory()
    logger.info(f"Memory: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
    return memory.percent

def preprocess_release(release, mini=False):
    """
    Preprocess one release and save to HDF5

    Returns:
        Path to output file on success, None on failure
    """
    logger.info(f"{'='*60}")
    logger.info(f"Processing {release}")
    logger.info(f"{'='*60}")
    print(f"\n{'='*60}")
    print(f"Processing {release}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        # Memory check
        safe, mem_pct = check_memory_safe()
        if not safe:
            logger.error(f"{release}: Memory limit exceeded: {mem_pct:.1f}%")
            print(f"❌ Memory limit exceeded: {mem_pct:.1f}%")
            return None

        log_memory()

        # Load dataset
        logger.info(f"{release}: Loading dataset")
        print("Loading dataset...")
        dataset = EEGChallengeDataset(
            release=release,
            mini=mini,
            query=dict(task="contrastChangeDetection"),
            description_fields=["rt_from_stimulus"],
            cache_dir=CACHE_DIR
        )
        logger.info(f"{release}: Loaded {len(dataset.datasets)} recordings")
        print(f"  Loaded: {len(dataset.datasets)} recordings")

        # Memory check after loading
        safe, mem_pct = check_memory_safe()
        if not safe:
            logger.error(f"{release}: Memory limit exceeded after loading: {mem_pct:.1f}%")
            print(f"❌ Memory limit exceeded: {mem_pct:.1f}%")
            return None

        # Preprocessing
        logger.info(f"{release}: Running preprocessors")
        print("Running preprocessors...")
        preprocessors = [
            Preprocessor(annotate_trials_with_target, apply_on_array=False),
            Preprocessor(add_aux_anchors, apply_on_array=False),
        ]
        preprocess(dataset, preprocessors)

        # Filter for stimulus_anchor
        logger.info(f"{release}: Filtering for stimulus_anchor")
        print("Filtering for stimulus_anchor...")
        dataset = keep_only_recordings_with("stimulus_anchor", dataset)
        logger.info(f"{release}: Kept {len(dataset.datasets)} recordings")
        print(f"  Kept: {len(dataset.datasets)} recordings")

        if len(dataset.datasets) == 0:
            logger.warning(f"{release}: No recordings with stimulus_anchor")
            print("⚠️  No recordings with stimulus_anchor, skipping")
            return None

        # Create windows
        logger.info(f"{release}: Creating windows")
        print("Creating windows...")
        ANCHOR = "stimulus_anchor"
        windows_dataset = create_windows_from_events(
            dataset,
            mapping={ANCHOR: 0},
            trial_start_offset_samples=int(SHIFT_AFTER_STIM * SFREQ),
            trial_stop_offset_samples=int((SHIFT_AFTER_STIM + EPOCH_LEN_S) * SFREQ),
            window_size_samples=int(EPOCH_LEN_S * SFREQ),
            window_stride_samples=SFREQ,
            preload=True,
        )
        logger.info(f"{release}: Created {len(windows_dataset)} windows")
        print(f"  Created: {len(windows_dataset)} windows")

        # Add metadata
        logger.info(f"{release}: Adding metadata")
        windows_dataset = add_extras_columns(
            windows_dataset,
            dataset,
            desc="stimulus_anchor",
            keys=("rt_from_stimulus", "target", "rt_from_trialstart",
                  "stimulus_onset", "response_onset", "correct", "response_type")
        )

        # Extract data
        logger.info(f"{release}: Extracting data to arrays")
        print("Extracting data...")

        # Extract EEG data
        X_list = []
        for i in range(len(windows_dataset)):
            try:
                X_i, _, _ = windows_dataset[i]
                # X_i shape: (1, n_channels, n_timepoints) or (n_channels, n_timepoints)
                if X_i.ndim == 3:
                    X_i = X_i.squeeze(0)
                X_list.append(X_i)
            except Exception as e:
                logger.warning(f"{release}: Failed to extract window {i}: {e}")
                continue

        X = np.array(X_list)

        # Extract labels from metadata DataFrame
        logger.info(f"{release}: Extracting labels from metadata")
        print("Extracting labels from metadata...")
        metadata_df = windows_dataset.get_metadata()

        if 'rt_from_stimulus' in metadata_df.columns:
            y = metadata_df['rt_from_stimulus'].values
            # Replace NaN with 0.0
            y = np.nan_to_num(y, nan=0.0)
            logger.info(f"{release}: Extracted {len(y)} labels, {np.sum(y != 0)} non-zero")
            print(f"  Labels: {len(y)} total, {np.sum(y != 0)} non-zero")
        else:
            logger.warning(f"{release}: rt_from_stimulus not in metadata, using zeros")
            y = np.zeros(len(X))

        logger.info(f"{release}: Data shape: {X.shape}, Labels shape: {y.shape}")
        print(f"  Data shape: {X.shape}")
        print(f"  Labels shape: {y.shape}")

        # Save to HDF5
        output_file = OUTPUT_DIR / f"challenge1_{release}_windows.h5"
        logger.info(f"{release}: Saving to {output_file}")
        print(f"Saving to {output_file}...")

        with h5py.File(output_file, 'w') as f:
            f.create_dataset('eeg', data=X, compression='gzip', compression_opts=4)
            f.create_dataset('labels', data=y, compression='gzip', compression_opts=4)
            f.attrs['release'] = release
            f.attrs['n_windows'] = len(X)
            f.attrs['n_channels'] = X.shape[1]
            f.attrs['n_timepoints'] = X.shape[2]
            f.attrs['sfreq'] = SFREQ

        elapsed = time.time() - start_time
        file_size_mb = output_file.stat().st_size / (1024 * 1024)

        logger.info(f"{release}: ✅ Success! {file_size_mb:.1f}MB in {elapsed:.1f}s")
        print(f"✅ Saved: {output_file.name} ({file_size_mb:.1f}MB in {elapsed:.1f}s)")

        return output_file

    except KeyboardInterrupt:
        logger.warning(f"{release}: Interrupted by user")
        raise
    except Exception as e:
        logger.error(f"{release}: Failed with error: {e}")
        print(f"❌ Error: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Cache Challenge 1 windows to HDF5')
    parser.add_argument('--mini', action='store_true', help='Use mini dataset')
    parser.add_argument('--releases', nargs='+', default=['R1', 'R2', 'R3', 'R4'],
                       help='Releases to process')
    args = parser.parse_args()

    print("="*80)
    print("🔄 MEMORY-SAFE PREPROCESSING: Caching Challenge 1 Windows")
    print("="*80)
    logger.info("="*80)
    logger.info("Preprocessing started")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Memory limit: {MAX_MEMORY_PERCENT}%")
    logger.info(f"Releases: {args.releases}")
    logger.info(f"Mini dataset: {args.mini}")
    logger.info("="*80)

    log_memory()

    processed = []
    failed = []
    skipped = []
    total_start = time.time()

    for release in args.releases:
        # Check if already processed
        output_file = OUTPUT_DIR / f"challenge1_{release}_windows.h5"
        if output_file.exists():
            print(f"\n⏭️  {release} already processed, skipping")
            logger.info(f"{release}: Already exists, skipping")
            skipped.append(release)
            processed.append(output_file)
            continue

        # Check memory before starting
        safe, mem_pct = check_memory_safe()
        if not safe:
            logger.error(f"Memory limit exceeded before {release}: {mem_pct:.1f}%")
            print(f"\n❌ Memory limit exceeded ({mem_pct:.1f}%), stopping")
            break

        # Process release
        try:
            result = preprocess_release(release, mini=args.mini)
            if result:
                processed.append(result)
                log_memory()
            else:
                failed.append(release)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user")
            print("\n⚠️  Interrupted by user")
            break
        except Exception as e:
            logger.error(f"{release}: Unexpected error: {e}")
            print(f"\n❌ Unexpected error in {release}: {e}")
            failed.append(release)
            continue

    # Summary
    total_elapsed = time.time() - total_start
    successful = len(processed) - len(skipped)

    print(f"\n{'='*80}")
    print("✅ PREPROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Successfully processed: {successful}")
    print(f"Skipped (already done): {len(skipped)}")
    print(f"Failed: {len(failed)}")
    print(f"Total time: {total_elapsed/60:.1f} minutes")

    logger.info("="*80)
    logger.info("Preprocessing complete")
    logger.info(f"Successful: {successful}, Skipped: {len(skipped)}, Failed: {len(failed)}")
    logger.info(f"Total time: {total_elapsed/60:.1f} minutes")

    if processed:
        print("\nCached files:")
        total_size = 0
        for f in processed:
            if f.exists():
                size_mb = f.stat().st_size / (1024 * 1024)
                total_size += size_mb
                print(f"  {f.name}: {size_mb:.1f}MB")
                logger.info(f"File: {f.name} ({size_mb:.1f}MB)")
        print(f"\nTotal size: {total_size:.1f}MB")
        logger.info(f"Total size: {total_size:.1f}MB")

    if failed:
        print(f"\n⚠️  Failed releases: {failed}")
        logger.warning(f"Failed releases: {failed}")

    print(f"\n✅ Next: Use HDF5Dataset to load these files for training")
    print(f"📋 Log saved to: {log_file}")
    logger.info("Session complete")

    return len(processed) > 0

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        logger.warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n❌ Fatal error: {e}")
        sys.exit(1)
