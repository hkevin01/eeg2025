#!/usr/bin/env python3
"""
Data Integrity Verification Script
Validates HBN EEG dataset structure and quality
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataIntegrityChecker:
    """Comprehensive data integrity validation."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.report = {
            "total_subjects": 0,
            "valid_subjects": 0,
            "invalid_subjects": [],
            "task_coverage": {},
            "issues": [],
        }

    def check_bids_structure(self) -> bool:
        """Validate BIDS directory structure."""
        logger.info("Checking BIDS structure...")

        required_files = ["dataset_description.json", "participants.tsv"]
        missing_files = [
            f for f in required_files if not (self.data_dir / f).exists()
        ]

        if missing_files:
            self.report["issues"].append(f"Missing BIDS files: {missing_files}")
            return False

        logger.info("✓ BIDS structure valid")
        return True

    def check_subject_data(self, subject_id: str) -> Dict:
        """Check data quality for a single subject."""
        subject_dir = self.data_dir / subject_id
        subject_report = {"valid": True, "tasks": {}, "issues": []}

        if not subject_dir.exists():
            subject_report["valid"] = False
            subject_report["issues"].append("Directory not found")
            return subject_report

        # Check for EEG data files
        eeg_files = list(subject_dir.rglob("*_eeg.fif"))

        if not eeg_files:
            subject_report["valid"] = False
            subject_report["issues"].append("No EEG data files found")
            return subject_report

        for eeg_file in eeg_files:
            task = self._extract_task_name(eeg_file.name)
            subject_report["tasks"][task] = self._validate_eeg_file(eeg_file)

        return subject_report

    def _extract_task_name(self, filename: str) -> str:
        """Extract task name from filename."""
        if "task-sus" in filename:
            return "sus"
        elif "task-ccd" in filename:
            return "ccd"
        elif "task-rest" in filename:
            return "rest"
        return "unknown"

    def _validate_eeg_file(self, eeg_path: Path) -> Dict:
        """Validate individual EEG file."""
        validation = {
            "exists": eeg_path.exists(),
            "size_mb": 0,
            "has_events": False,
            "has_channels": False,
        }

        if validation["exists"]:
            validation["size_mb"] = eeg_path.stat().st_size / (1024 * 1024)

            # Check for companion files
            base = eeg_path.stem.replace("_eeg", "")
            parent = eeg_path.parent

            validation["has_events"] = (parent / f"{base}_events.tsv").exists()
            validation["has_channels"] = (parent / f"{base}_channels.tsv").exists()

        return validation

    def scan_all_subjects(self) -> None:
        """Scan all subjects in the dataset."""
        logger.info(f"Scanning dataset: {self.data_dir}")

        # Find all subject directories
        subject_dirs = sorted([d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith("sub-")])

        self.report["total_subjects"] = len(subject_dirs)
        logger.info(f"Found {len(subject_dirs)} subjects")

        task_counts = {"sus": 0, "ccd": 0, "rest": 0}

        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            logger.debug(f"Checking {subject_id}...")

            subject_report = self.check_subject_data(subject_id)

            if subject_report["valid"]:
                self.report["valid_subjects"] += 1

                # Count task coverage
                for task in subject_report["tasks"].keys():
                    if task in task_counts:
                        task_counts[task] += 1
            else:
                self.report["invalid_subjects"].append(
                    {
                        "subject": subject_id,
                        "issues": subject_report["issues"],
                    }
                )

        self.report["task_coverage"] = task_counts

    def generate_report(self, output_path: Path) -> None:
        """Generate comprehensive validation report."""
        logger.info("\n" + "=" * 60)
        logger.info("DATA INTEGRITY REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Subjects: {self.report['total_subjects']}")
        logger.info(f"Valid Subjects: {self.report['valid_subjects']}")
        logger.info(
            f"Invalid Subjects: {len(self.report['invalid_subjects'])}"
        )
        logger.info("\nTask Coverage:")
        for task, count in self.report["task_coverage"].items():
            logger.info(f"  {task.upper()}: {count} subjects")

        if self.report["issues"]:
            logger.warning("\nIssues Found:")
            for issue in self.report["issues"]:
                logger.warning(f"  - {issue}")

        if self.report["invalid_subjects"]:
            logger.warning("\nInvalid Subjects:")
            for entry in self.report["invalid_subjects"][:10]:  # Show first 10
                logger.warning(f"  {entry['subject']}: {entry['issues']}")
            if len(self.report["invalid_subjects"]) > 10:
                logger.warning(
                    f"  ... and {len(self.report['invalid_subjects']) - 10} more"
                )

        # Save to JSON
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.report, f, indent=2)

        logger.info(f"\n✓ Report saved to: {output_path}")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Verify HBN data integrity")

    parser.add_argument(
        "--data_dir",
        type=Path,
        default="data/raw/hbn",
        help="Path to HBN dataset directory",
    )

    parser.add_argument(
        "--output_report",
        type=Path,
        default="data/verification_report.json",
        help="Output path for verification report",
    )

    args = parser.parse_args()

    # Check if data directory exists
    if not args.data_dir.exists():
        logger.error(f"Data directory not found: {args.data_dir}")
        logger.info("Please download the dataset first using:")
        logger.info("  python scripts/download_hbn_data.py")
        return 1

    # Run integrity check
    checker = DataIntegrityChecker(args.data_dir)

    # Validate BIDS structure
    checker.check_bids_structure()

    # Scan all subjects
    checker.scan_all_subjects()

    # Generate report
    checker.generate_report(args.output_report)

    # Return error code if validation failed
    if checker.report["invalid_subjects"] or checker.report["issues"]:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
