#!/usr/bin/env python3
"""
Validate EEG data quality and statistics
"""

import logging
from pathlib import Path
from typing import Dict, List

import mne
import mne_bids
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def validate_subject_data(bids_path: mne_bids.BIDSPath) -> Dict[str, any]:
    """Validate a single subject's EEG data."""
    results = {
        "subject": bids_path.subject,
        "task": bids_path.task,
        "valid": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }

    try:
        # Load data
        raw = mne_bids.read_raw_bids(bids_path, verbose=False)

        # Basic stats
        results["stats"]["sfreq"] = raw.info["sfreq"]
        results["stats"]["n_channels"] = len(raw.ch_names)
        results["stats"]["duration"] = raw.times[-1]
        results["stats"]["n_samples"] = len(raw.times)

        # Get data for quality checks
        data = raw.get_data()

        # Check for NaNs
        if np.isnan(data).any():
            results["valid"] = False
            results["errors"].append("Data contains NaN values")

        # Check for Infs
        if np.isinf(data).any():
            results["valid"] = False
            results["errors"].append("Data contains Inf values")

        # Check amplitude range (reasonable for EEG in V)
        data_min = np.min(data)
        data_max = np.max(data)
        results["stats"]["amplitude_min"] = float(data_min)
        results["stats"]["amplitude_max"] = float(data_max)

        # Warn if amplitudes seem unreasonable (typical EEG is microvolts, so 1e-6 to 1e-4 V)
        if abs(data_max) > 1e-3 or abs(data_min) > 1e-3:
            results["warnings"].append(f"Unusual amplitude range: [{data_min:.2e}, {data_max:.2e}] V")

        # Check for flat channels
        std_per_channel = np.std(data, axis=1)
        flat_channels = np.where(std_per_channel < 1e-10)[0]
        if len(flat_channels) > 0:
            results["warnings"].append(f"{len(flat_channels)} flat channels detected")
            results["stats"]["n_flat_channels"] = len(flat_channels)

        # Check sampling rate
        if results["stats"]["sfreq"] < 100:
            results["warnings"].append(f"Low sampling rate: {results['stats']['sfreq']} Hz")

        # Check duration
        if results["stats"]["duration"] < 10:
            results["warnings"].append(f"Short recording: {results['stats']['duration']:.1f}s")

    except Exception as e:
        results["valid"] = False
        results["errors"].append(f"Failed to load/validate data: {str(e)}")

    return results


def validate_dataset(bids_root: Path, max_subjects: int = None) -> Dict[str, any]:
    """Validate entire BIDS dataset."""
    logger.info(f"🔍 Validating dataset at: {bids_root}\n")

    # Find all subjects
    subject_dirs = sorted([d for d in bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")])

    if max_subjects:
        subject_dirs = subject_dirs[:max_subjects]

    all_results = []
    n_valid = 0
    n_errors = 0
    n_warnings = 0

    for sub_dir in subject_dirs:
        subject_id = sub_dir.name.replace("sub-", "")

        # Find EEG files for this subject
        eeg_dir = sub_dir / "eeg"
        if not eeg_dir.exists():
            continue

        eeg_files = list(eeg_dir.glob("*.set"))

        logger.info(f"📁 Subject: {subject_id} ({len(eeg_files)} files)")

        # Validate first file (usually RestingState) for each subject
        for eeg_file in eeg_files[:1]:  # Just validate first task for speed
            # Parse task from filename
            parts = eeg_file.stem.split("_")
            task = None
            for part in parts:
                if part.startswith("task-"):
                    task = part.replace("task-", "")
                    break

            if not task:
                continue

            bids_path = mne_bids.BIDSPath(
                subject=subject_id, task=task, datatype="eeg", root=bids_root
            )

            result = validate_subject_data(bids_path)
            all_results.append(result)

            if result["valid"]:
                n_valid += 1
                logger.info(f"  ✓ {task}: {result['stats']['duration']:.1f}s, {result['stats']['sfreq']}Hz, {result['stats']['n_channels']} ch")
            else:
                n_errors += 1
                logger.error(f"  ✗ {task}: {', '.join(result['errors'])}")

            if result["warnings"]:
                n_warnings += len(result["warnings"])
                for warning in result["warnings"]:
                    logger.warning(f"    ⚠️  {warning}")

    # Summary
    logger.info(f"\n📊 Validation Summary:")
    logger.info(f"  Total validated: {len(all_results)}")
    logger.info(f"  Valid: {n_valid}")
    logger.info(f"  Errors: {n_errors}")
    logger.info(f"  Warnings: {n_warnings}")

    if n_errors == 0:
        logger.info("\n✅ All data validated successfully!")
    else:
        logger.error("\n❌ Some data validation failed")

    return {"results": all_results, "n_valid": n_valid, "n_errors": n_errors, "n_warnings": n_warnings}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate EEG data quality")
    parser.add_argument(
        "--bids_root",
        type=Path,
        default="data/raw/hbn",
        help="Path to BIDS dataset root",
    )
    parser.add_argument(
        "--max_subjects", type=int, help="Maximum number of subjects to validate"
    )
    args = parser.parse_args()

    summary = validate_dataset(args.bids_root, args.max_subjects)

    # Return exit code based on validation
    return 0 if summary["n_errors"] == 0 else 1


if __name__ == "__main__":
    exit(main())
