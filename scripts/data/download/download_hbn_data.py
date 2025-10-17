#!/usr/bin/env python3
"""
HBN Dataset Download Script
Downloads EEG data from Healthy Brain Network dataset
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HBNDownloader:
    """Download HBN EEG data from various sources."""

    def __init__(self, output_dir: Path, resume: bool = True):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume = resume

        # HBN data URLs (these are examples - update with actual URLs)
        self.base_urls = {
            "s3": "s3://fcp-indi/data/Projects/HBN/EEG/",
            "http": "https://fcp-indi.s3.amazonaws.com/data/Projects/HBN/EEG/",
        }

        # Track download progress
        self.progress_file = self.output_dir / "download_progress.json"
        self.progress = self._load_progress()

    def _load_progress(self) -> dict:
        """Load download progress from file."""
        if self.progress_file.exists() and self.resume:
            with open(self.progress_file) as f:
                return json.load(f)
        return {"downloaded": [], "failed": [], "total": 0}

    def _save_progress(self):
        """Save download progress to file."""
        with open(self.progress_file, "w") as f:
            json.dump(self.progress, f, indent=2)

    def download_file(self, url: str, output_path: Path) -> bool:
        """Download a single file with resume support."""
        try:
            # Check if already downloaded
            if output_path.exists() and self.resume:
                logger.debug(f"Skipping {output_path.name} (already exists)")
                return True

            # Create parent directory
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Download with progress bar
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()

            total_size = int(response.headers.get("content-length", 0))

            with open(output_path, "wb") as f:
                with tqdm(
                    total=total_size, unit="B", unit_scale=True, desc=output_path.name
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))

            logger.info(f"✓ Downloaded {output_path.name}")
            return True

        except Exception as e:
            logger.error(f"✗ Failed to download {url}: {e}")
            if output_path.exists():
                output_path.unlink()  # Remove partial download
            return False

    def download_subject(
        self, subject_id: str, tasks: List[str] = ["sus", "ccd", "rest"]
    ) -> bool:
        """Download all EEG data for a specific subject."""
        logger.info(f"Downloading subject {subject_id}...")

        success = True
        for task in tasks:
            files_to_download = [
                f"{subject_id}_task-{task}_eeg.fif",
                f"{subject_id}_task-{task}_events.tsv",
                f"{subject_id}_task-{task}_channels.tsv",
                f"{subject_id}_task-{task}_eeg.json",
            ]

            for filename in files_to_download:
                url = urljoin(self.base_urls["http"], f"{subject_id}/{filename}")
                output_path = self.output_dir / subject_id / filename

                if not self.download_file(url, output_path):
                    success = False

        if success:
            self.progress["downloaded"].append(subject_id)
        else:
            self.progress["failed"].append(subject_id)

        self._save_progress()
        return success

    def download_subjects(
        self,
        subjects: Optional[List[str]] = None,
        max_subjects: Optional[int] = None,
        tasks: List[str] = ["sus", "ccd", "rest"],
    ):
        """Download multiple subjects."""
        if subjects is None:
            # Get subject list from server or manifest
            subjects = self._get_subject_list()

        if max_subjects:
            subjects = subjects[:max_subjects]

        logger.info(f"Starting download of {len(subjects)} subjects...")
        logger.info(f"Tasks: {', '.join(tasks)}")
        logger.info(f"Output directory: {self.output_dir}")

        total = len(subjects)
        success_count = 0
        fail_count = 0

        for i, subject_id in enumerate(subjects, 1):
            logger.info(f"\n[{i}/{total}] Processing {subject_id}")

            # Skip if already downloaded
            if subject_id in self.progress["downloaded"]:
                logger.info(f"Skipping {subject_id} (already downloaded)")
                success_count += 1
                continue

            if self.download_subject(subject_id, tasks):
                success_count += 1
            else:
                fail_count += 1

        logger.info("\n" + "=" * 60)
        logger.info("Download Summary:")
        logger.info(f"✓ Successfully downloaded: {success_count}/{total}")
        logger.info(f"✗ Failed: {fail_count}/{total}")
        logger.info(f"Progress saved to: {self.progress_file}")
        logger.info("=" * 60)

    def _get_subject_list(self) -> List[str]:
        """
        Get list of available subjects from server.
        This is a placeholder - implement based on actual data source.
        """
        # Option 1: Load from manifest file
        manifest_path = Path("data/hbn_subjects_manifest.txt")
        if manifest_path.exists():
            with open(manifest_path) as f:
                return [line.strip() for line in f if line.strip()]

        # Option 2: Generate sample list (for testing)
        logger.warning("No subject manifest found, using sample subjects")
        return [f"sub-NDARINV{str(i).zfill(8)}" for i in range(1, 101)]

    def verify_downloads(self) -> dict:
        """Verify integrity of downloaded files."""
        logger.info("Verifying downloaded files...")

        verification = {"valid": [], "invalid": [], "missing": []}

        for subject_id in self.progress["downloaded"]:
            subject_dir = self.output_dir / subject_id

            if not subject_dir.exists():
                verification["missing"].append(subject_id)
                continue

            # Check for required files
            required_files = ["_eeg.fif", "_events.tsv", "_channels.tsv"]
            files_exist = all(
                any(subject_dir.glob(f"*{pattern}")) for pattern in required_files
            )

            if files_exist:
                verification["valid"].append(subject_id)
            else:
                verification["invalid"].append(subject_id)

        logger.info(f"✓ Valid: {len(verification['valid'])}")
        logger.info(f"✗ Invalid: {len(verification['invalid'])}")
        logger.info(f"? Missing: {len(verification['missing'])}")

        return verification


def main():
    parser = argparse.ArgumentParser(description="Download HBN EEG dataset")

    parser.add_argument(
        "--output_dir",
        type=Path,
        default="data/raw/hbn",
        help="Output directory for downloaded data",
    )

    parser.add_argument(
        "--subjects",
        nargs="+",
        help="Specific subjects to download (e.g., sub-NDARINV001)",
    )

    parser.add_argument(
        "--max_subjects", type=int, help="Maximum number of subjects to download"
    )

    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["sus", "ccd", "rest"],
        help="Tasks to download",
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume interrupted downloads",
    )

    parser.add_argument(
        "--no-resume", dest="resume", action="store_false", help="Start fresh download"
    )

    parser.add_argument(
        "--verify", action="store_true", help="Verify downloaded files"
    )

    args = parser.parse_args()

    # Create downloader
    downloader = HBNDownloader(output_dir=args.output_dir, resume=args.resume)

    if args.verify:
        # Just verify existing downloads
        results = downloader.verify_downloads()
        sys.exit(0 if not results["invalid"] else 1)
    else:
        # Download data
        downloader.download_subjects(
            subjects=args.subjects, max_subjects=args.max_subjects, tasks=args.tasks
        )


if __name__ == "__main__":
    main()
