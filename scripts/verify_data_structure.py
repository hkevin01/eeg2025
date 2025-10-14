#!/usr/bin/env python3
"""
Verify BIDS data structure compliance
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import mne_bids

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def verify_bids_structure(bids_root: Path) -> Dict[str, any]:
    """Verify BIDS dataset structure."""
    results = {
        "valid": True,
        "errors": [],
        "warnings": [],
        "subjects": [],
        "tasks": set(),
        "sessions": set(),
    }

    # Check if BIDS root exists
    if not bids_root.exists():
        results["valid"] = False
        results["errors"].append(f"BIDS root does not exist: {bids_root}")
        return results

    # Check for required BIDS files
    required_files = ["dataset_description.json"]
    for req_file in required_files:
        file_path = bids_root / req_file
        if not file_path.exists():
            results["valid"] = False
            results["errors"].append(f"Missing required file: {req_file}")
        else:
            logger.info(f"✓ Found {req_file}")

    # Check dataset_description.json
    desc_file = bids_root / "dataset_description.json"
    if desc_file.exists():
        try:
            with open(desc_file) as f:
                desc = json.load(f)
            logger.info(f"✓ Dataset: {desc.get('Name', 'Unknown')}")
            logger.info(f"  BIDSVersion: {desc.get('BIDSVersion', 'Unknown')}")
        except Exception as e:
            results["warnings"].append(f"Could not parse dataset_description.json: {e}")

    # Find all subjects
    subject_dirs = sorted([d for d in bids_root.iterdir() if d.is_dir() and d.name.startswith("sub-")])

    if not subject_dirs:
        results["valid"] = False
        results["errors"].append("No subject directories found")
        return results

    logger.info(f"\n✓ Found {len(subject_dirs)} subjects")

    # Verify each subject
    for sub_dir in subject_dirs:
        subject_id = sub_dir.name.replace("sub-", "")
        results["subjects"].append(subject_id)

        # Check for EEG data
        eeg_dir = sub_dir / "eeg"
        if eeg_dir.exists():
            eeg_files = list(eeg_dir.glob("*.set")) + list(eeg_dir.glob("*.fif"))
            logger.info(f"  - {subject_id}: {len(eeg_files)} EEG files")

            # Extract tasks from filenames
            for eeg_file in eeg_files:
                # Parse task from filename (e.g., sub-XXX_task-RestingState_eeg.set)
                parts = eeg_file.stem.split("_")
                for part in parts:
                    if part.startswith("task-"):
                        task = part.replace("task-", "")
                        results["tasks"].add(task)
        else:
            results["warnings"].append(f"Subject {subject_id} has no eeg directory")

    # Check for participants.tsv
    participants_file = bids_root / "participants.tsv"
    if participants_file.exists():
        logger.info(f"\n✓ Found participants.tsv")
        try:
            with open(participants_file) as f:
                lines = f.readlines()
            logger.info(f"  Contains {len(lines) - 1} participants (header + data)")
        except Exception as e:
            results["warnings"].append(f"Could not read participants.tsv: {e}")
    else:
        results["warnings"].append("Missing participants.tsv (recommended)")

    # Summary
    logger.info(f"\n📊 Summary:")
    logger.info(f"  Subjects: {len(results['subjects'])}")
    logger.info(f"  Tasks: {len(results['tasks'])} - {', '.join(sorted(results['tasks']))}")
    logger.info(f"  Errors: {len(results['errors'])}")
    logger.info(f"  Warnings: {len(results['warnings'])}")

    if results["errors"]:
        logger.error("\n❌ BIDS validation FAILED")
        for error in results["errors"]:
            logger.error(f"  - {error}")
    else:
        logger.info("\n✅ BIDS structure is valid!")

    if results["warnings"]:
        logger.warning("\n⚠️  Warnings:")
        for warning in results["warnings"]:
            logger.warning(f"  - {warning}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify BIDS data structure")
    parser.add_argument(
        "--bids_root",
        type=Path,
        default="data/raw/hbn",
        help="Path to BIDS dataset root",
    )
    args = parser.parse_args()

    logger.info(f"🔍 Verifying BIDS structure at: {args.bids_root}\n")
    results = verify_bids_structure(args.bids_root)

    # Return exit code based on validation
    return 0 if results["valid"] else 1


if __name__ == "__main__":
    exit(main())
