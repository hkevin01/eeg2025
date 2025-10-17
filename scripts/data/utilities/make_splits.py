#!/usr/bin/env python3
"""
Official Train/Val/Test splits generator for EEG Foundation Challenge 2025.

This script creates challenge-compliant data splits with strict subject-level
isolation, ensuring no data leakage between folds. Implements the official
splitting strategy described in Figure 1 and subsection 1.2 of the challenge.

Features:
- Subject-level isolation (recordings never cross folds)
- Stratified splitting by age and sex
- Reproducible with version control and checksums
- Validation of split integrity
- Support for both challenges (CCD and CBCL)
"""

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataio.starter_kit import StarterKitDataLoader

logger = logging.getLogger(__name__)


class OfficialSplitGenerator:
    """
    Generate official Train/Val/Test splits for EEG Foundation Challenge 2025.

    Implements challenge-compliant splitting with strict subject-level isolation
    and stratification by demographic variables.
    """

    def __init__(
        self,
        bids_root: Path,
        splits_dir: Path,
        random_seed: int = 42,
        val_size: float = 0.15,
        test_size: float = 0.15
    ):
        """
        Initialize split generator.

        Args:
            bids_root: Path to BIDS dataset root
            splits_dir: Directory to save split files
            random_seed: Random seed for reproducibility
            val_size: Proportion of data for validation
            test_size: Proportion of data for test
        """
        self.bids_root = Path(bids_root)
        self.splits_dir = Path(splits_dir)
        self.random_seed = random_seed
        self.val_size = val_size
        self.test_size = test_size

        # Ensure splits directory exists
        self.splits_dir.mkdir(parents=True, exist_ok=True)

        # Initialize data loader
        self.loader = StarterKitDataLoader(
            bids_root=self.bids_root,
            enable_caching=True
        )

        logger.info(f"Initialized split generator for {self.bids_root}")
        logger.info(f"Split sizes: train={1-val_size-test_size:.1%}, val={val_size:.1%}, test={test_size:.1%}")

    def load_participants_with_data(self) -> pd.DataFrame:
        """
        Load participants with available EEG data and phenotype information.

        Returns:
            DataFrame with participants and their metadata
        """
        try:
            # Load participants data
            participants_df = self.loader._load_participants_data()

            if participants_df is None or participants_df.empty:
                raise ValueError("No participants data found")

            logger.info(f"Loaded {len(participants_df)} participants from participants.tsv")

            # Check for subjects with actual EEG data
            subjects_with_eeg = self._find_subjects_with_eeg_data()
            logger.info(f"Found {len(subjects_with_eeg)} subjects with EEG data")

            # Filter participants to only those with EEG data
            participants_df = participants_df[
                participants_df['participant_id'].isin(subjects_with_eeg)
            ].copy()

            logger.info(f"Filtered to {len(participants_df)} participants with EEG data")

            # Load and merge phenotype data if available
            try:
                phenotype_df = self.loader._load_phenotype_data()
                if phenotype_df is not None and not phenotype_df.empty:
                    # Merge phenotype data
                    participants_df = participants_df.merge(
                        phenotype_df,
                        on='participant_id',
                        how='left'
                    )
                    logger.info(f"Merged phenotype data for {len(participants_df)} participants")

            except Exception as e:
                logger.warning(f"Could not load phenotype data: {e}")

            return participants_df

        except Exception as e:
            logger.error(f"Error loading participants data: {e}")
            raise

    def _find_subjects_with_eeg_data(self) -> Set[str]:
        """Find all subjects that have EEG data files."""
        subjects_with_data = set()

        # Look for subject directories
        for subject_dir in self.bids_root.glob("sub-*"):
            if not subject_dir.is_dir():
                continue

            subject_id = subject_dir.name

            # Check if subject has any EEG sessions
            has_eeg = False
            for session_dir in subject_dir.glob("ses-*"):
                eeg_dir = session_dir / "eeg"
                if eeg_dir.exists():
                    # Check for any EEG files
                    eeg_files = list(eeg_dir.glob("*.edf")) + list(eeg_dir.glob("*.bdf"))
                    if eeg_files:
                        has_eeg = True
                        break

            if has_eeg:
                subjects_with_data.add(subject_id)

        return subjects_with_data

    def create_stratified_splits(
        self,
        participants_df: pd.DataFrame
    ) -> Dict[str, List[str]]:
        """
        Create stratified train/val/test splits with subject-level isolation.

        Args:
            participants_df: DataFrame with participant information

        Returns:
            Dictionary with 'train', 'val', 'test' keys mapping to subject lists
        """
        logger.info("Creating stratified splits...")

        # Set random seed for reproducibility
        np.random.seed(self.random_seed)

        # Create stratification variables
        participants_df = self._prepare_stratification_variables(participants_df)

        # Get subjects and stratification labels
        subjects = participants_df['participant_id'].tolist()

        # Create stratification labels (combining multiple variables)
        strat_labels = self._create_stratification_labels(participants_df)

        logger.info(f"Splitting {len(subjects)} subjects")
        logger.info(f"Stratification distribution: {np.unique(strat_labels, return_counts=True)}")

        # First split: train+val vs test
        train_val_subjects, test_subjects, train_val_labels, _ = train_test_split(
            subjects,
            strat_labels,
            test_size=self.test_size,
            random_state=self.random_seed,
            stratify=strat_labels
        )

        # Second split: train vs val
        val_size_adjusted = self.val_size / (1 - self.test_size)
        train_subjects, val_subjects = train_test_split(
            train_val_subjects,
            test_size=val_size_adjusted,
            random_state=self.random_seed,
            stratify=train_val_labels
        )

        splits = {
            'train': sorted(train_subjects),
            'val': sorted(val_subjects),
            'test': sorted(test_subjects)
        }

        # Log split statistics
        for split_name, split_subjects in splits.items():
            logger.info(f"{split_name}: {len(split_subjects)} subjects")

        # Validate splits
        self._validate_splits(splits, participants_df)

        return splits

    def _prepare_stratification_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare variables for stratified splitting."""
        df = df.copy()

        # Age binning for stratification
        if 'age' in df.columns:
            age_series = pd.to_numeric(df['age'], errors='coerce')
            df['age_bin'] = pd.cut(
                age_series,
                bins=[0, 10, 13, 16, 100],
                labels=['child', 'pre_teen', 'teen', 'adult'],
                include_lowest=True
            )
        else:
            df['age_bin'] = 'unknown'

        # Sex normalization
        if 'sex' in df.columns:
            df['sex_norm'] = df['sex'].fillna('unknown').str.lower()
            df['sex_norm'] = df['sex_norm'].map({'m': 'male', 'f': 'female'}).fillna('unknown')
        else:
            df['sex_norm'] = 'unknown'

        return df

    def _create_stratification_labels(self, df: pd.DataFrame) -> np.ndarray:
        """Create combined stratification labels."""
        # Combine age_bin and sex_norm for stratification
        strat_labels = df['age_bin'].astype(str) + '_' + df['sex_norm'].astype(str)

        # Handle cases with too few samples by grouping
        label_counts = strat_labels.value_counts()
        min_samples_per_group = 6  # Need at least 6 for train/val/test splits

        # Group rare categories
        rare_labels = label_counts[label_counts < min_samples_per_group].index
        strat_labels = strat_labels.replace(rare_labels, 'other')

        return strat_labels.values

    def _validate_splits(self, splits: Dict[str, List[str]], participants_df: pd.DataFrame):
        """Validate that splits meet challenge requirements."""
        logger.info("Validating splits...")

        # Check no overlap between splits
        all_subjects = set()
        for split_name, split_subjects in splits.items():
            split_set = set(split_subjects)

            # Check no overlap with previous splits
            overlap = all_subjects.intersection(split_set)
            if overlap:
                raise ValueError(f"Subject overlap detected in {split_name}: {overlap}")

            all_subjects.update(split_set)

        # Check all subjects are accounted for
        expected_subjects = set(participants_df['participant_id'].tolist())
        if all_subjects != expected_subjects:
            missing = expected_subjects - all_subjects
            extra = all_subjects - expected_subjects
            if missing:
                logger.warning(f"Missing subjects from splits: {missing}")
            if extra:
                logger.warning(f"Extra subjects in splits: {extra}")

        # Validate demographic distribution
        self._validate_demographic_distribution(splits, participants_df)

        logger.info("✅ Split validation passed")

    def _validate_demographic_distribution(self, splits: Dict[str, List[str]], participants_df: pd.DataFrame):
        """Validate demographic distribution across splits."""
        for split_name, split_subjects in splits.items():
            split_df = participants_df[participants_df['participant_id'].isin(split_subjects)]

            if 'age' in split_df.columns:
                ages = pd.to_numeric(split_df['age'], errors='coerce').dropna()
                if len(ages) > 0:
                    logger.info(f"{split_name} age: mean={ages.mean():.1f}, std={ages.std():.1f}")

            if 'sex' in split_df.columns:
                sex_counts = split_df['sex'].value_counts()
                logger.info(f"{split_name} sex distribution: {dict(sex_counts)}")

    def save_splits(self, splits: Dict[str, List[str]], version: str = "v1.0") -> Path:
        """
        Save splits to disk with versioning and checksums.

        Args:
            splits: Dictionary with split assignments
            version: Version string for the splits

        Returns:
            Path to the saved splits file
        """
        # Create versioned filename
        splits_filename = f"official_splits_{version}.json"
        splits_file = self.splits_dir / splits_filename

        # Add metadata
        splits_data = {
            "version": version,
            "created_date": pd.Timestamp.now().isoformat(),
            "random_seed": self.random_seed,
            "val_size": self.val_size,
            "test_size": self.test_size,
            "total_subjects": sum(len(subjects) for subjects in splits.values()),
            "splits": splits
        }

        # Save splits
        with open(splits_file, 'w') as f:
            json.dump(splits_data, f, indent=2, sort_keys=True)

        logger.info(f"Saved splits to {splits_file}")

        # Create checksum
        self._create_checksum(splits_file)

        # Save human-readable summary
        self._save_splits_summary(splits, version)

        return splits_file

    def _create_checksum(self, splits_file: Path):
        """Create MD5 checksum for splits file."""
        with open(splits_file, 'rb') as f:
            checksum = hashlib.md5(f.read()).hexdigest()

        checksum_file = splits_file.with_suffix('.md5')
        with open(checksum_file, 'w') as f:
            f.write(f"{checksum}  {splits_file.name}\n")

        logger.info(f"Created checksum: {checksum}")

    def _save_splits_summary(self, splits: Dict[str, List[str]], version: str):
        """Save human-readable splits summary."""
        summary_file = self.splits_dir / f"splits_summary_{version}.txt"

        with open(summary_file, 'w') as f:
            f.write(f"EEG Foundation Challenge 2025 - Official Splits {version}\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"Random seed: {self.random_seed}\n")
            f.write(f"Validation size: {self.val_size:.1%}\n")
            f.write(f"Test size: {self.test_size:.1%}\n\n")

            total_subjects = sum(len(subjects) for subjects in splits.values())

            for split_name, split_subjects in splits.items():
                f.write(f"{split_name.upper()} ({len(split_subjects)} subjects, "
                       f"{len(split_subjects)/total_subjects:.1%}):\n")
                f.write("-" * 40 + "\n")

                for subject in split_subjects[:10]:  # Show first 10
                    f.write(f"  {subject}\n")

                if len(split_subjects) > 10:
                    f.write(f"  ... and {len(split_subjects) - 10} more\n")

                f.write("\n")

        logger.info(f"Saved summary to {summary_file}")

    def load_existing_splits(self, version: str = "v1.0") -> Optional[Dict[str, List[str]]]:
        """Load existing splits if available."""
        splits_file = self.splits_dir / f"official_splits_{version}.json"

        if not splits_file.exists():
            return None

        try:
            with open(splits_file, 'r') as f:
                splits_data = json.load(f)

            # Verify checksum if available
            checksum_file = splits_file.with_suffix('.md5')
            if checksum_file.exists():
                self._verify_checksum(splits_file, checksum_file)

            logger.info(f"Loaded existing splits from {splits_file}")
            return splits_data['splits']

        except Exception as e:
            logger.error(f"Error loading existing splits: {e}")
            return None

    def _verify_checksum(self, splits_file: Path, checksum_file: Path):
        """Verify file integrity using checksum."""
        with open(splits_file, 'rb') as f:
            actual_checksum = hashlib.md5(f.read()).hexdigest()

        with open(checksum_file, 'r') as f:
            expected_checksum = f.read().strip().split()[0]

        if actual_checksum != expected_checksum:
            raise ValueError(f"Checksum mismatch for {splits_file}")

        logger.debug("Checksum verification passed")


def main():
    """Main function to generate official splits."""
    parser = argparse.ArgumentParser(
        description="Generate official Train/Val/Test splits for EEG Foundation Challenge 2025"
    )
    parser.add_argument(
        "--bids-root",
        type=Path,
        required=True,
        help="Path to BIDS dataset root directory"
    )
    parser.add_argument(
        "--splits-dir",
        type=Path,
        default="data/splits",
        help="Directory to save split files"
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1.0",
        help="Version string for the splits"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--val-size",
        type=float,
        default=0.15,
        help="Proportion of data for validation"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.15,
        help="Proportion of data for test"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if splits exist"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Initialize generator
        generator = OfficialSplitGenerator(
            bids_root=args.bids_root,
            splits_dir=args.splits_dir,
            random_seed=args.random_seed,
            val_size=args.val_size,
            test_size=args.test_size
        )

        # Check for existing splits
        if not args.force:
            existing_splits = generator.load_existing_splits(args.version)
            if existing_splits is not None:
                logger.info(f"Splits {args.version} already exist. Use --force to regenerate.")
                return

        # Load participants data
        logger.info("Loading participants data...")
        participants_df = generator.load_participants_with_data()

        if participants_df.empty:
            logger.error("No participants data found")
            return

        # Generate splits
        logger.info("Generating stratified splits...")
        splits = generator.create_stratified_splits(participants_df)

        # Save splits
        logger.info("Saving splits...")
        splits_file = generator.save_splits(splits, args.version)

        logger.info("✅ Official splits generation completed successfully!")
        logger.info(f"Splits saved to: {splits_file}")

        # Summary
        total_subjects = sum(len(subjects) for subjects in splits.values())
        logger.info(f"Total subjects: {total_subjects}")
        for split_name, split_subjects in splits.items():
            logger.info(f"  {split_name}: {len(split_subjects)} ({len(split_subjects)/total_subjects:.1%})")

    except Exception as e:
        logger.error(f"Failed to generate splits: {e}")
        raise


if __name__ == "__main__":
    main()
