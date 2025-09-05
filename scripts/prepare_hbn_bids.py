#!/usr/bin/env python3
"""
Prepare HBN-EEG BIDS dataset by creating symlinks and validating structure.

This script helps set up the HBN dataset for the EEG Foundation Challenge
by creating symlinks to avoid data duplication and validating BIDS compliance.
"""

import argparse
import logging
import shutil
from pathlib import Path
from typing import Optional

import pandas as pd
from mne_bids import BIDSPath


def setup_logging(level: str = "INFO") -> None:
    """Set up logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def validate_source_directory(source_path: Path) -> bool:
    """
    Validate that source directory contains valid BIDS structure.

    Args:
        source_path: Path to source BIDS directory

    Returns:
        True if valid BIDS structure found
    """
    logger = logging.getLogger(__name__)

    # Check for required BIDS files
    required_files = ["participants.tsv", "dataset_description.json"]
    for required_file in required_files:
        if not (source_path / required_file).exists():
            logger.error(f"Required BIDS file not found: {required_file}")
            return False

    # Check for subject directories
    subject_dirs = list(source_path.glob("sub-*"))
    if not subject_dirs:
        logger.error("No subject directories found")
        return False

    logger.info(f"Found {len(subject_dirs)} subject directories")

    # Check for EEG data
    eeg_files = list(source_path.glob("**/eeg/*.edf")) + list(source_path.glob("**/eeg/*.bdf"))
    if not eeg_files:
        logger.warning("No EDF/BDF EEG files found")
    else:
        logger.info(f"Found {len(eeg_files)} EEG files")

    return True


def create_symlinks(source_path: Path, target_path: Path, force: bool = False) -> None:
    """
    Create symlinks from source to target directory.

    Args:
        source_path: Source BIDS directory
        target_path: Target directory for symlinks
        force: Whether to overwrite existing target directory
    """
    logger = logging.getLogger(__name__)

    # Handle existing target directory
    if target_path.exists():
        if force:
            logger.info(f"Removing existing target directory: {target_path}")
            shutil.rmtree(target_path)
        else:
            logger.error(f"Target directory already exists: {target_path}")
            logger.error("Use --force to overwrite")
            return

    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Created target directory: {target_path}")

    # Create symlinks for all contents
    for item in source_path.iterdir():
        if item.name.startswith('.'):
            continue  # Skip hidden files

        target_item = target_path / item.name
        try:
            target_item.symlink_to(item.resolve())
            logger.debug(f"Created symlink: {item.name}")
        except OSError as e:
            logger.error(f"Failed to create symlink for {item.name}: {e}")

    logger.info("Symlink creation completed")


def validate_symlinks(target_path: Path) -> bool:
    """
    Validate that all symlinks are working correctly.

    Args:
        target_path: Target directory with symlinks

    Returns:
        True if all symlinks are valid
    """
    logger = logging.getLogger(__name__)

    broken_links = []

    for item in target_path.rglob("*"):
        if item.is_symlink() and not item.exists():
            broken_links.append(item)

    if broken_links:
        logger.error(f"Found {len(broken_links)} broken symlinks:")
        for link in broken_links[:10]:  # Show first 10
            logger.error(f"  {link}")
        if len(broken_links) > 10:
            logger.error(f"  ... and {len(broken_links) - 10} more")
        return False

    logger.info("All symlinks are valid")
    return True


def generate_summary_report(target_path: Path) -> None:
    """Generate a summary report of the dataset."""
    logger = logging.getLogger(__name__)

    # Count subjects
    subject_dirs = list(target_path.glob("sub-*"))
    n_subjects = len(subject_dirs)

    # Count sessions
    session_dirs = list(target_path.glob("**/ses-*"))
    n_sessions = len(session_dirs)

    # Count EEG files by task
    task_counts = {}
    for eeg_file in target_path.glob("**/eeg/*_eeg.*"):
        # Extract task from filename
        filename = eeg_file.name
        if "_task-" in filename:
            task = filename.split("_task-")[1].split("_")[0]
            task_counts[task] = task_counts.get(task, 0) + 1

    # Generate report
    logger.info("Dataset Summary:")
    logger.info(f"  Subjects: {n_subjects}")
    logger.info(f"  Sessions: {n_sessions}")
    logger.info(f"  Task distribution:")
    for task, count in sorted(task_counts.items()):
        logger.info(f"    {task}: {count} files")

    # Save summary to file
    summary_file = target_path / "dataset_summary.txt"
    with open(summary_file, "w") as f:
        f.write("HBN-EEG Dataset Summary\n")
        f.write("=" * 30 + "\n")
        f.write(f"Subjects: {n_subjects}\n")
        f.write(f"Sessions: {n_sessions}\n")
        f.write("Task distribution:\n")
        for task, count in sorted(task_counts.items()):
            f.write(f"  {task}: {count} files\n")

    logger.info(f"Summary saved to: {summary_file}")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Prepare HBN-EEG BIDS dataset with symlinks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Source BIDS directory path",
    )

    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Target directory for symlinks",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing target directory",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)

    logger.info("Starting HBN-EEG BIDS preparation")
    logger.info(f"Source: {args.source}")
    logger.info(f"Target: {args.target}")

    # Validate source directory
    if not args.source.exists():
        logger.error(f"Source directory does not exist: {args.source}")
        return

    if not validate_source_directory(args.source):
        logger.error("Source directory validation failed")
        return

    # Create symlinks
    create_symlinks(args.source, args.target, args.force)

    # Validate symlinks
    if not validate_symlinks(args.target):
        logger.error("Symlink validation failed")
        return

    # Generate summary report
    generate_summary_report(args.target)

    logger.info("HBN-EEG BIDS preparation completed successfully")


if __name__ == "__main__":
    main()
