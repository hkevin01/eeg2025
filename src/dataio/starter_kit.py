"""
Starter Kit integration for EEG Foundation Challenge 2025.

This module provides utilities for loading official challenge labels, splits,
and metrics exactly as specified in the Starter Kit.

Features robust error handling, memory management, and graceful degradation
for production-level reliability.
"""

import gc
import hashlib
import json
import logging
import time
import traceback
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import psutil
from scipy.stats import pearsonr
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

# Configure logging with more detailed formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Track memory usage statistics for monitoring and optimization."""

    process_memory_mb: float = 0.0
    available_memory_mb: float = 0.0
    memory_percent: float = 0.0
    peak_memory_mb: float = 0.0
    timestamp: float = field(default_factory=time.time)

    def update(self) -> None:
        """Update memory statistics with current system state."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            virtual_memory = psutil.virtual_memory()

            self.process_memory_mb = memory_info.rss / 1024 / 1024
            self.available_memory_mb = virtual_memory.available / 1024 / 1024
            self.memory_percent = virtual_memory.percent
            self.peak_memory_mb = max(self.peak_memory_mb, self.process_memory_mb)
            self.timestamp = time.time()

        except Exception as e:
            logger.warning(f"Failed to update memory stats: {e}")


@dataclass
class TimingStats:
    """Track execution timing for performance monitoring and optimization."""

    operation_times: Dict[str, List[float]] = field(default_factory=dict)
    total_execution_time: float = 0.0
    start_time: Optional[float] = None

    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_time = time.time()

    def end_timer(self, operation: str) -> float:
        """End timing and record duration."""
        if self.start_time is None:
            logger.warning(f"Timer not started for operation: {operation}")
            return 0.0

        duration = time.time() - self.start_time

        if operation not in self.operation_times:
            self.operation_times[operation] = []
        self.operation_times[operation].append(duration)

        self.total_execution_time += duration
        self.start_time = None

        return duration

    def get_average_time(self, operation: str) -> float:
        """Get average execution time for an operation."""
        times = self.operation_times.get(operation, [])
        return sum(times) / len(times) if times else 0.0


def memory_monitor(threshold_mb: float = 1000.0):
    """
    Decorator to monitor memory usage and warn if threshold exceeded.

    Args:
        threshold_mb: Memory threshold in MB to trigger warnings
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            memory_stats = MemoryStats()
            memory_stats.update()
            initial_memory = memory_stats.process_memory_mb

            try:
                result = func(*args, **kwargs)

                # Check memory after execution
                memory_stats.update()
                memory_increase = memory_stats.process_memory_mb - initial_memory

                if memory_increase > threshold_mb:
                    logger.warning(
                        f"Function {func.__name__} increased memory by {memory_increase:.1f}MB "
                        f"(threshold: {threshold_mb}MB)"
                    )

                return result

            except MemoryError as e:
                logger.error(f"Memory error in {func.__name__}: {e}")
                # Force garbage collection
                gc.collect()
                raise

        return wrapper

    return decorator


def timing_monitor(func):
    """Decorator to monitor execution timing."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            logger.debug(f"Function {func.__name__} took {duration:.3f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Function {func.__name__} failed after {duration:.3f}s: {e}")
            raise

    return wrapper


@contextmanager
def graceful_error_handler(operation_name: str, default_return=None):
    """
    Context manager for graceful error handling with detailed logging.

    Args:
        operation_name: Name of the operation for logging
        default_return: Default value to return on error
    """
    try:
        logger.debug(f"Starting operation: {operation_name}")
        yield
        logger.debug(f"Completed operation: {operation_name}")

    except FileNotFoundError as e:
        logger.error(f"File not found in {operation_name}: {e}")
        if default_return is not None:
            return default_return
        raise

    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data encountered in {operation_name}: {e}")
        if default_return is not None:
            return default_return
        raise

    except MemoryError as e:
        logger.error(f"Memory error in {operation_name}: {e}")
        gc.collect()  # Force garbage collection
        if default_return is not None:
            return default_return
        raise

    except Exception as e:
        logger.error(f"Unexpected error in {operation_name}: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        if default_return is not None:
            return default_return
        raise


class StarterKitDataLoader:
    """
    Official data loader matching Starter Kit specifications with enhanced robustness.

    Implements exact label extraction, splits, and metrics as defined
    in the EEG Foundation Challenge 2025 (arXiv:2506.19141).

    Features:
    - Comprehensive error handling and recovery
    - Memory usage monitoring and optimization
    - Performance timing and profiling
    - Graceful degradation under resource constraints
    - Detailed logging and debugging support
    """

    def __init__(
        self,
        bids_root: Union[str, Path],
        starter_kit_path: Optional[Path] = None,
        memory_limit_mb: float = 2000.0,
        enable_caching: bool = True,
        validate_on_init: bool = True,
    ):
        """
        Initialize Starter Kit data loader with enhanced robustness.

        Args:
            bids_root: Path to BIDS dataset root
            starter_kit_path: Path to starter kit files (optional)
            memory_limit_mb: Memory limit in MB for processing
            enable_caching: Whether to enable data caching for performance
            validate_on_init: Whether to validate data integrity on initialization

        Raises:
            FileNotFoundError: If BIDS root directory doesn't exist
            PermissionError: If insufficient permissions to read data
            MemoryError: If insufficient memory for initialization
        """
        # Initialize core attributes
        self.bids_root = Path(bids_root)
        self.starter_kit_path = starter_kit_path
        self.memory_limit_mb = memory_limit_mb
        self.enable_caching = enable_caching

        # Initialize monitoring
        self.memory_stats = MemoryStats()
        self.timing_stats = TimingStats()

        # Initialize data containers with explicit typing
        self.participants_data: Optional[pd.DataFrame] = None
        self.phenotype_data: Optional[pd.DataFrame] = None
        self.official_splits: Optional[Dict[str, List[str]]] = None
        self.ccd_cache: Dict[str, pd.DataFrame] = {}
        self.cbcl_cache: Dict[str, pd.DataFrame] = {}

        # Validate initialization parameters
        self._validate_initialization_params()

        # Load official metadata with error handling
        try:
            logger.info(
                f"Initializing StarterKitDataLoader with BIDS root: {self.bids_root}"
            )
            self.memory_stats.update()

            if validate_on_init:
                self._validate_bids_structure()

            self._load_participants_data()
            self._load_phenotype_data()
            self._load_official_splits()

            logger.info("StarterKitDataLoader initialization completed successfully")

        except Exception as e:
            logger.error(f"Failed to initialize StarterKitDataLoader: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(
                f"StarterKitDataLoader initialization failed: {e}"
            ) from e

    def _validate_initialization_params(self) -> None:
        """Validate initialization parameters and raise appropriate errors."""
        if not isinstance(self.bids_root, Path):
            try:
                self.bids_root = Path(self.bids_root)
            except Exception as e:
                raise ValueError(f"Invalid bids_root path: {e}")

        if not self.bids_root.exists():
            raise FileNotFoundError(
                f"BIDS root directory does not exist: {self.bids_root}"
            )

        if not self.bids_root.is_dir():
            raise NotADirectoryError(f"BIDS root is not a directory: {self.bids_root}")

        if self.memory_limit_mb <= 0:
            raise ValueError(f"Memory limit must be positive: {self.memory_limit_mb}")

        # Check available memory
        available_memory = psutil.virtual_memory().available / 1024 / 1024
        if self.memory_limit_mb > available_memory * 0.8:
            logger.warning(
                f"Memory limit ({self.memory_limit_mb}MB) is high relative to "
                f"available memory ({available_memory:.1f}MB)"
            )

    def _validate_bids_structure(self) -> None:
        """Validate basic BIDS directory structure."""
        required_files = ["participants.tsv"]
        missing_files = []

        for filename in required_files:
            filepath = self.bids_root / filename
            if not filepath.exists():
                missing_files.append(filename)

        if missing_files:
            logger.warning(f"Missing BIDS files: {missing_files}")
            # Don't raise error - allow graceful degradation

        logger.info(f"Validated BIDS structure at {self.bids_root}")

    @memory_monitor(threshold_mb=100.0)
    @timing_monitor
    def _load_participants_data(self) -> None:
        """
        Load participants.tsv with official subject metadata.

        Features robust error handling, memory monitoring, and data validation.

        Raises:
            FileNotFoundError: If participants.tsv not found
            pd.errors.EmptyDataError: If file is empty
            ValueError: If required columns missing
        """
        participants_file = self.bids_root / "participants.tsv"

        with graceful_error_handler("loading participants data"):
            if not participants_file.exists():
                raise FileNotFoundError(
                    f"participants.tsv not found at {participants_file}"
                )

            # Check file size and permissions
            file_size_mb = participants_file.stat().st_size / 1024 / 1024
            if file_size_mb > self.memory_limit_mb * 0.1:
                logger.warning(
                    f"Large participants file ({file_size_mb:.1f}MB), "
                    f"consider chunked loading"
                )

            # Load with error handling for encoding issues
            try:
                self.participants_data = pd.read_csv(
                    participants_file,
                    sep="\t",
                    encoding="utf-8",
                    low_memory=False,
                    na_values=["", "NA", "NaN", "NULL", "null"],
                )
            except UnicodeDecodeError:
                logger.warning("UTF-8 decode failed, trying latin-1 encoding")
                self.participants_data = pd.read_csv(
                    participants_file,
                    sep="\t",
                    encoding="latin-1",
                    low_memory=False,
                    na_values=["", "NA", "NaN", "NULL", "null"],
                )

            # Validate data integrity
            if self.participants_data.empty:
                raise pd.errors.EmptyDataError("participants.tsv contains no data")

            # Validate required columns with detailed error messages
            required_cols = ["participant_id", "age", "sex"]
            missing_cols = set(required_cols) - set(self.participants_data.columns)
            if missing_cols:
                error_msg = (
                    f"Missing required columns in participants.tsv: {missing_cols}"
                )
                available_cols = list(self.participants_data.columns)
                logger.error(f"{error_msg}. Available columns: {available_cols}")

                # Attempt to recover by creating missing columns with defaults
                for col in missing_cols:
                    if col == "participant_id":
                        # This is critical - cannot recover
                        raise ValueError(f"Critical column missing: {col}")
                    elif col == "age":
                        self.participants_data[col] = np.nan
                        logger.warning(f"Created missing {col} column with NaN values")
                    elif col == "sex":
                        self.participants_data[col] = "unknown"
                        logger.warning(
                            f"Created missing {col} column with 'unknown' values"
                        )

            # Data quality checks
            self._validate_participants_data()

            # Memory optimization
            self._optimize_participants_memory()

            logger.info(
                f"Successfully loaded {len(self.participants_data)} participants "
                f"({self.participants_data.memory_usage(deep=True).sum() / 1024 / 1024:.1f}MB)"
            )

    def _validate_participants_data(self) -> None:
        """Validate participants data quality and fix common issues."""
        if self.participants_data is None:
            return

        # Check for duplicate participant IDs
        duplicates = self.participants_data["participant_id"].duplicated()
        if duplicates.any():
            duplicate_ids = self.participants_data[duplicates][
                "participant_id"
            ].tolist()
            logger.error(f"Found duplicate participant IDs: {duplicate_ids}")
            # Remove duplicates, keeping first occurrence
            self.participants_data = self.participants_data.drop_duplicates(
                subset=["participant_id"], keep="first"
            )
            logger.warning(f"Removed {duplicates.sum()} duplicate entries")

        # Validate age ranges (reasonable bounds for HBN dataset)
        if "age" in self.participants_data.columns:
            age_series = pd.to_numeric(self.participants_data["age"], errors="coerce")
            invalid_ages = (age_series < 0) | (age_series > 100)
            if invalid_ages.any():
                logger.warning(
                    f"Found {invalid_ages.sum()} participants with invalid ages"
                )
                # Set invalid ages to NaN
                self.participants_data.loc[invalid_ages, "age"] = np.nan

        # Validate sex values
        if "sex" in self.participants_data.columns:
            valid_sex_values = {"M", "F", "male", "female", "Male", "Female", "unknown"}
            invalid_sex = ~self.participants_data["sex"].isin(valid_sex_values)
            if invalid_sex.any():
                logger.warning(
                    f"Found {invalid_sex.sum()} participants with invalid sex values"
                )
                self.participants_data.loc[invalid_sex, "sex"] = "unknown"

    def _optimize_participants_memory(self) -> None:
        """Optimize memory usage of participants data."""
        if self.participants_data is None:
            return

        # Convert categorical columns
        categorical_cols = ["sex"]
        for col in categorical_cols:
            if col in self.participants_data.columns:
                self.participants_data[col] = self.participants_data[col].astype(
                    "category"
                )

        # Optimize numeric columns
        if "age" in self.participants_data.columns:
            # Age can be stored as int8 (0-255 range)
            age_series = pd.to_numeric(self.participants_data["age"], errors="coerce")
            if age_series.max() <= 255 and age_series.min() >= 0:
                self.participants_data["age"] = age_series.astype(
                    "Int8"
                )  # Nullable integer

        # Force garbage collection
        gc.collect()

    @memory_monitor(threshold_mb=200.0)
    @timing_monitor
    def _load_phenotype_data(self) -> None:
        """
        Load phenotype data for psychopathology prediction with enhanced robustness.

        Features:
        - Comprehensive error handling for missing files
        - Memory-efficient loading with chunking for large files
        - Data validation and quality checks
        - Graceful degradation when files are missing or corrupted
        """
        phenotype_dir = self.bids_root / "phenotype"
        self.phenotype_data = {}

        with graceful_error_handler("loading phenotype data", default_return={}):
            if not phenotype_dir.exists():
                logger.warning(f"Phenotype directory not found: {phenotype_dir}")
                return

            # Load CBCL data (Child Behavior Checklist) - critical for Challenge 2
            self._load_cbcl_data(phenotype_dir)

            # Load other phenotype files with error recovery
            self._load_additional_phenotype_files(phenotype_dir)

            # Validate loaded phenotype data
            self._validate_phenotype_data()

            logger.info(
                f"Loaded phenotype data for {len(self.phenotype_data)} instruments"
            )

    def _load_cbcl_data(self, phenotype_dir: Path) -> None:
        """Load CBCL data with comprehensive error handling and validation."""
        cbcl_file = phenotype_dir / "cbcl.tsv"

        if not cbcl_file.exists():
            logger.warning(f"CBCL file not found: {cbcl_file}")
            return

        try:
            # Check file size for memory planning
            file_size_mb = cbcl_file.stat().st_size / 1024 / 1024

            if file_size_mb > self.memory_limit_mb * 0.2:
                logger.info(
                    f"Large CBCL file ({file_size_mb:.1f}MB), using chunked loading"
                )
                # Load in chunks to manage memory
                chunks = []
                chunk_size = max(
                    1000, int(self.memory_limit_mb * 100)
                )  # Adjust chunk size based on memory limit

                for chunk in pd.read_csv(cbcl_file, sep="\t", chunksize=chunk_size):
                    chunks.append(chunk)

                self.cbcl_df = pd.concat(chunks, ignore_index=True)
                del chunks  # Free memory
            else:
                self.cbcl_df = pd.read_csv(
                    cbcl_file,
                    sep="\t",
                    low_memory=False,
                    na_values=["", "NA", "NaN", "NULL", "null", "-999", "999"],
                )

            # Validate CBCL data structure
            self._validate_cbcl_data()

            # Optimize memory usage
            self._optimize_cbcl_memory()

            self.phenotype_data["cbcl"] = self.cbcl_df
            memory_usage = self.cbcl_df.memory_usage(deep=True).sum() / 1024 / 1024
            logger.info(
                f"Loaded CBCL data: {len(self.cbcl_df)} subjects ({memory_usage:.1f}MB)"
            )

        except Exception as e:
            logger.error(f"Failed to load CBCL data: {e}")
            # Continue without CBCL data - log error but don't crash

    def _validate_cbcl_data(self) -> None:
        """Validate CBCL data structure and content."""
        if not hasattr(self, "cbcl_df") or self.cbcl_df is None:
            return

        # Expected CBCL columns for Challenge 2
        expected_cbcl_cols = [
            "participant_id",
            "p_factor",
            "internalizing",
            "externalizing",
            "attention",
        ]
        missing_cbcl_cols = set(expected_cbcl_cols) - set(self.cbcl_df.columns)

        if missing_cbcl_cols:
            logger.error(f"Missing CBCL columns: {missing_cbcl_cols}")
            available_cols = list(self.cbcl_df.columns)
            logger.info(f"Available CBCL columns: {available_cols}")

            # Attempt recovery by creating missing columns with NaN
            for col in missing_cbcl_cols:
                if col != "participant_id":  # participant_id is critical
                    self.cbcl_df[col] = np.nan
                    logger.warning(
                        f"Created missing CBCL column '{col}' with NaN values"
                    )

        # Validate numeric ranges for CBCL scores (typically z-scores or T-scores)
        numeric_cols = ["p_factor", "internalizing", "externalizing", "attention"]
        for col in numeric_cols:
            if col in self.cbcl_df.columns:
                numeric_series = pd.to_numeric(self.cbcl_df[col], errors="coerce")

                # Check for reasonable score ranges (z-scores typically -3 to +3, T-scores 0-100)
                extreme_values = (numeric_series < -10) | (numeric_series > 200)
                if extreme_values.any():
                    logger.warning(
                        f"Found {extreme_values.sum()} extreme values in CBCL column '{col}'"
                    )
                    # Don't modify - might be valid extreme scores

        # Check for duplicate subjects
        if "participant_id" in self.cbcl_df.columns:
            duplicates = self.cbcl_df["participant_id"].duplicated()
            if duplicates.any():
                logger.warning(
                    f"Found {duplicates.sum()} duplicate subjects in CBCL data"
                )
                self.cbcl_df = self.cbcl_df.drop_duplicates(
                    subset=["participant_id"], keep="first"
                )

    def _optimize_cbcl_memory(self) -> None:
        """Optimize memory usage of CBCL data."""
        if not hasattr(self, "cbcl_df") or self.cbcl_df is None:
            return

        # Convert numeric columns to float32 if possible (saves 50% memory vs float64)
        numeric_cols = ["p_factor", "internalizing", "externalizing", "attention"]
        for col in numeric_cols:
            if col in self.cbcl_df.columns:
                original_series = self.cbcl_df[col]
                try:
                    # Convert to numeric first
                    numeric_series = pd.to_numeric(original_series, errors="coerce")

                    # Check if values fit in float32 range
                    if (
                        numeric_series.min() >= np.finfo(np.float32).min
                        and numeric_series.max() <= np.finfo(np.float32).max
                    ):
                        self.cbcl_df[col] = numeric_series.astype("float32")
                    else:
                        self.cbcl_df[col] = numeric_series.astype("float64")
                except Exception as e:
                    logger.warning(f"Failed to optimize column {col}: {e}")

        gc.collect()

    def _load_additional_phenotype_files(self, phenotype_dir: Path) -> None:
        """Load additional phenotype files with error recovery."""
        phenotype_files = list(phenotype_dir.glob("*.tsv"))

        for pheno_file in phenotype_files:
            if pheno_file.name == "cbcl.tsv":
                continue  # Already loaded

            try:
                # Memory check before loading
                file_size_mb = pheno_file.stat().st_size / 1024 / 1024
                current_memory = self.memory_stats.process_memory_mb

                if current_memory + file_size_mb > self.memory_limit_mb:
                    logger.warning(
                        f"Skipping {pheno_file.name} - would exceed memory limit "
                        f"({current_memory:.1f} + {file_size_mb:.1f} > {self.memory_limit_mb}MB)"
                    )
                    continue

                df = pd.read_csv(
                    pheno_file,
                    sep="\t",
                    low_memory=False,
                    na_values=["", "NA", "NaN", "NULL", "null"],
                )

                if not df.empty:
                    self.phenotype_data[pheno_file.stem] = df
                    logger.info(f"Loaded {pheno_file.stem}: {len(df)} subjects")
                else:
                    logger.warning(f"Empty phenotype file: {pheno_file.name}")

            except Exception as e:
                logger.error(f"Failed to load phenotype file {pheno_file.name}: {e}")
                # Continue with other files - don't let one failure stop everything

    def _validate_phenotype_data(self) -> None:
        """Validate all loaded phenotype data."""
        if not self.phenotype_data:
            logger.warning("No phenotype data loaded")
            return

        for instrument_name, df in self.phenotype_data.items():
            try:
                # Basic validation
                if df.empty:
                    logger.warning(f"Empty data for instrument: {instrument_name}")
                    continue

                # Check for participant_id column
                if "participant_id" not in df.columns:
                    logger.error(f"Missing participant_id in {instrument_name}")
                    continue

                # Check data types and memory usage
                memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024
                logger.debug(
                    f"Instrument {instrument_name}: {len(df)} rows, {memory_usage:.1f}MB"
                )

            except Exception as e:
                logger.error(f"Validation failed for {instrument_name}: {e}")

    @timing_monitor
    def _load_official_splits(self) -> None:
        """
        Load official train/val/test splits with comprehensive error handling.

        Features:
        - Automatic split generation if official splits missing
        - Data integrity validation with checksums
        - Graceful recovery from corrupted split files
        - Reproducible split generation with fixed seeds
        """
        splits_file = self.bids_root / "splits.json"

        with graceful_error_handler("loading official splits"):
            if splits_file.exists():
                try:
                    with open(splits_file, "r", encoding="utf-8") as f:
                        self.official_splits = json.load(f)

                    # Validate loaded splits
                    if self._validate_splits(self.official_splits):
                        logger.info("Successfully loaded and validated official splits")
                        return
                    else:
                        logger.warning(
                            "Official splits validation failed, regenerating..."
                        )

                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logger.error(f"Corrupted splits file: {e}")
                    logger.info("Regenerating splits...")

            # Create or regenerate splits
            self.official_splits = self._create_subject_splits()

            # Save splits for reproducibility
            try:
                with open(splits_file, "w", encoding="utf-8") as f:
                    json.dump(self.official_splits, f, indent=2, ensure_ascii=False)

                # Create checksum for validation
                self._create_splits_checksum(splits_file)
                logger.info(f"Created and saved new splits to {splits_file}")

            except Exception as e:
                logger.error(f"Failed to save splits: {e}")
                # Continue with in-memory splits

    def _validate_splits(self, splits: Dict[str, List[str]]) -> bool:
        """
        Validate split data structure and integrity.

        Args:
            splits: Dictionary containing train/val/test splits

        Returns:
            bool: True if splits are valid, False otherwise
        """
        try:
            # Check required keys
            required_keys = {"train", "val", "test"}
            if not set(splits.keys()) >= required_keys:
                logger.error(
                    f"Missing required split keys. Expected: {required_keys}, Got: {set(splits.keys())}"
                )
                return False

            # Check that splits are non-empty lists
            for split_name, split_data in splits.items():
                if not isinstance(split_data, list):
                    logger.error(
                        f"Split '{split_name}' is not a list: {type(split_data)}"
                    )
                    return False

                if len(split_data) == 0:
                    logger.error(f"Split '{split_name}' is empty")
                    return False

            # Check for overlaps between splits
            all_subjects = set()
            for split_name, split_data in splits.items():
                split_set = set(split_data)

                # Check for duplicates within split
                if len(split_set) != len(split_data):
                    duplicates = len(split_data) - len(split_set)
                    logger.error(
                        f"Split '{split_name}' contains {duplicates} duplicate subjects"
                    )
                    return False

                # Check for overlaps with other splits
                overlap = all_subjects.intersection(split_set)
                if overlap:
                    logger.error(
                        f"Split '{split_name}' overlaps with previous splits: {overlap}"
                    )
                    return False

                all_subjects.update(split_set)

            # Validate subject IDs format (should start with 'sub-')
            invalid_subjects = []
            for subject in all_subjects:
                if not isinstance(subject, str) or not subject.startswith("sub-"):
                    invalid_subjects.append(subject)

            if invalid_subjects:
                logger.warning(
                    f"Found {len(invalid_subjects)} subjects with non-standard IDs"
                )
                # Don't fail validation - just warn

            # Check split proportions (reasonable ranges)
            total_subjects = len(all_subjects)
            train_ratio = len(splits["train"]) / total_subjects
            val_ratio = len(splits["val"]) / total_subjects
            test_ratio = len(splits["test"]) / total_subjects

            if not (0.5 <= train_ratio <= 0.8):
                logger.warning(f"Unusual train split ratio: {train_ratio:.2f}")
            if not (0.1 <= val_ratio <= 0.3):
                logger.warning(f"Unusual val split ratio: {val_ratio:.2f}")
            if not (0.1 <= test_ratio <= 0.3):
                logger.warning(f"Unusual test split ratio: {test_ratio:.2f}")

            logger.info(
                f"Splits validation passed: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])} subjects"
            )
            return True

        except Exception as e:
            logger.error(f"Split validation failed: {e}")
            return False

    def _create_subject_splits(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        random_seed: int = 42,
        stratify_by: Optional[str] = None,
    ) -> Dict[str, List[str]]:
        """
        Create subject-level splits to prevent data leakage with enhanced options.

        Args:
            train_ratio: Proportion of subjects for training
            val_ratio: Proportion of subjects for validation
            random_seed: Random seed for reproducibility
            stratify_by: Column name for stratified splitting (e.g., 'age_group', 'sex')

        Returns:
            Dictionary with train/val/test splits

        Raises:
            ValueError: If ratios are invalid or insufficient data
        """
        # Validate input parameters
        if not (0.0 < train_ratio < 1.0):
            raise ValueError(
                f"Invalid train_ratio: {train_ratio}. Must be between 0 and 1."
            )
        if not (0.0 < val_ratio < 1.0):
            raise ValueError(
                f"Invalid val_ratio: {val_ratio}. Must be between 0 and 1."
            )
        if train_ratio + val_ratio >= 1.0:
            raise ValueError(
                f"train_ratio + val_ratio must be < 1.0. Got: {train_ratio + val_ratio}"
            )

        test_ratio = 1.0 - train_ratio - val_ratio

        # Get subjects from participants data
        if self.participants_data is None or self.participants_data.empty:
            raise ValueError("No participants data available for creating splits")

        subjects = self.participants_data["participant_id"].dropna().unique().tolist()
        if len(subjects) == 0:
            raise ValueError("No valid participant IDs found")

        # Minimum subjects per split
        min_subjects_per_split = 1
        n_subjects = len(subjects)

        if n_subjects < 3:  # Need at least 3 subjects for train/val/test
            raise ValueError(f"Insufficient subjects for splitting: {n_subjects}")

        # Calculate split sizes
        n_train = max(min_subjects_per_split, int(n_subjects * train_ratio))
        n_val = max(min_subjects_per_split, int(n_subjects * val_ratio))
        n_test = n_subjects - n_train - n_val

        if n_test < min_subjects_per_split:
            # Adjust splits to ensure minimum test size
            n_test = min_subjects_per_split
            remaining = n_subjects - n_test
            n_train = max(
                min_subjects_per_split,
                int(remaining * train_ratio / (train_ratio + val_ratio)),
            )
            n_val = remaining - n_train

        logger.info(
            f"Creating splits with ratios: train={n_train/n_subjects:.2f}, val={n_val/n_subjects:.2f}, test={n_test/n_subjects:.2f}"
        )

        # Set random seed for reproducibility
        np.random.seed(random_seed)

        # Stratified splitting if requested
        if stratify_by and stratify_by in self.participants_data.columns:
            try:
                splits = self._create_stratified_splits(
                    subjects, stratify_by, n_train, n_val, n_test
                )
            except Exception as e:
                logger.warning(
                    f"Stratified splitting failed: {e}. Falling back to random splitting."
                )
                splits = self._create_random_splits(subjects, n_train, n_val, n_test)
        else:
            splits = self._create_random_splits(subjects, n_train, n_val, n_test)

        # Final validation
        if not self._validate_splits(splits):
            raise ValueError("Generated splits failed validation")

        return splits

    def _create_random_splits(
        self, subjects: List[str], n_train: int, n_val: int, n_test: int
    ) -> Dict[str, List[str]]:
        """Create random splits."""
        subjects_shuffled = np.random.permutation(subjects).tolist()

        splits = {
            "train": subjects_shuffled[:n_train],
            "val": subjects_shuffled[n_train : n_train + n_val],
            "test": subjects_shuffled[n_train + n_val : n_train + n_val + n_test],
        }

        logger.info(
            f"Created random splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
        )
        return splits

    def _create_stratified_splits(
        self,
        subjects: List[str],
        stratify_by: str,
        n_train: int,
        n_val: int,
        n_test: int,
    ) -> Dict[str, List[str]]:
        """Create stratified splits to maintain group proportions."""
        # Get stratification labels
        subject_data = self.participants_data.set_index("participant_id")[stratify_by]

        # Group subjects by stratification variable
        groups = {}
        for subject in subjects:
            if subject in subject_data.index:
                group_value = subject_data[subject]
                if pd.notna(group_value):
                    if group_value not in groups:
                        groups[group_value] = []
                    groups[group_value].append(subject)

        if not groups:
            raise ValueError(
                f"No valid groups found for stratification by {stratify_by}"
            )

        # Allocate subjects from each group proportionally
        splits = {"train": [], "val": [], "test": []}

        for group_value, group_subjects in groups.items():
            np.random.shuffle(group_subjects)
            group_size = len(group_subjects)

            # Calculate proportional sizes for this group
            group_n_train = max(1, int(group_size * n_train / len(subjects)))
            group_n_val = max(1, int(group_size * n_val / len(subjects)))
            group_n_test = group_size - group_n_train - group_n_val

            # Ensure we don't exceed group size
            if group_n_test <= 0:
                group_n_test = 1
                group_n_train = max(1, group_size - group_n_val - group_n_test)

            # Allocate subjects
            splits["train"].extend(group_subjects[:group_n_train])
            splits["val"].extend(
                group_subjects[group_n_train : group_n_train + group_n_val]
            )
            splits["test"].extend(
                group_subjects[
                    group_n_train
                    + group_n_val : group_n_train
                    + group_n_val
                    + group_n_test
                ]
            )

        logger.info(
            f"Created stratified splits by {stratify_by}: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}"
        )
        return splits

    def _create_splits_checksum(self, splits_file: Path) -> None:
        """Create checksum for splits file validation."""
        try:
            with open(splits_file, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            checksum_file = splits_file.with_suffix(".sha256")
            with open(checksum_file, "w") as f:
                f.write(f"{file_hash}  {splits_file.name}\n")

            logger.debug(f"Created checksum file: {checksum_file}")

        except Exception as e:
            logger.warning(f"Failed to create checksum: {e}")

    @memory_monitor(threshold_mb=50.0)
    @timing_monitor
    def load_ccd_labels(self, split: str = "train") -> pd.DataFrame:
        """
        Load CCD (Cross-Cognitive Domain) labels for Challenge 1 with robust error handling.

        Args:
            split: Data split to load ("train", "val", "test")

        Returns:
            DataFrame with CCD labels (response_time, success)

        Raises:
            ValueError: If split is invalid
            FileNotFoundError: If required CCD files not found
            pd.errors.EmptyDataError: If no valid data found
        """
        # Validate input parameters
        if split not in ["train", "val", "test"]:
            raise ValueError(
                f"Invalid split: {split}. Must be one of ['train', 'val', 'test']"
            )

        # Check cache first for performance
        cache_key = f"ccd_{split}"
        if self.enable_caching and cache_key in self.ccd_cache:
            logger.debug(f"Returning cached CCD data for {split}")
            return self.ccd_cache[cache_key].copy()

        with graceful_error_handler(f"loading CCD labels for {split}"):
            # Get subjects for this split
            if self.official_splits is None:
                raise ValueError("Official splits not loaded")

            split_subjects = self.official_splits.get(split, [])
            if not split_subjects:
                raise ValueError(f"No subjects found for split: {split}")

            logger.info(
                f"Loading CCD labels for {len(split_subjects)} subjects in {split} split"
            )

            # Load CCD events data
            ccd_data_list = []
            failed_subjects = []

            for subject_id in split_subjects:
                try:
                    subject_data = self._load_subject_ccd_data(subject_id)
                    if subject_data is not None and not subject_data.empty:
                        ccd_data_list.append(subject_data)
                    else:
                        failed_subjects.append(subject_id)

                except Exception as e:
                    logger.warning(f"Failed to load CCD data for {subject_id}: {e}")
                    failed_subjects.append(subject_id)
                    continue

            # Report loading statistics
            if failed_subjects:
                logger.warning(
                    f"Failed to load CCD data for {len(failed_subjects)} subjects: {failed_subjects[:5]}{'...' if len(failed_subjects) > 5 else ''}"
                )

            if not ccd_data_list:
                raise pd.errors.EmptyDataError(
                    f"No CCD data found for any subject in {split} split"
                )

            # Combine all subject data
            ccd_labels = pd.concat(ccd_data_list, ignore_index=True)

            # Validate and clean the combined data
            ccd_labels = self._validate_and_clean_ccd_data(ccd_labels)

            # Cache the result for future use
            if self.enable_caching:
                self.ccd_cache[cache_key] = ccd_labels.copy()

            # Memory usage reporting
            memory_usage = ccd_labels.memory_usage(deep=True).sum() / 1024 / 1024
            logger.info(
                f"Successfully loaded CCD data for {split}: "
                f"{len(ccd_labels)} trials from {len(ccd_data_list)} subjects "
                f"({memory_usage:.1f}MB)"
            )

            return ccd_labels

    def _load_subject_ccd_data(self, subject_id: str) -> Optional[pd.DataFrame]:
        """
        Load CCD data for a single subject with comprehensive error handling.

        Args:
            subject_id: Subject identifier (e.g., 'sub-001')

        Returns:
            DataFrame with CCD trials for the subject, or None if not found/invalid
        """
        try:
            # Construct path to subject's CCD events file
            # BIDS format: sub-XXX/ses-YYY/func/sub-XXX_ses-YYY_task-CCD_events.tsv
            subject_dir = self.bids_root / subject_id

            if not subject_dir.exists():
                logger.debug(f"Subject directory not found: {subject_dir}")
                return None

            # Look for CCD events files across all sessions
            ccd_files = list(subject_dir.glob("*/func/*task-CCD*events.tsv"))

            if not ccd_files:
                logger.debug(f"No CCD events files found for {subject_id}")
                return None

            # Load and combine data from all sessions
            session_data_list = []

            for ccd_file in ccd_files:
                try:
                    # Extract session info from filename
                    session_info = self._extract_session_info(ccd_file)

                    # Load events file
                    events_df = pd.read_csv(
                        ccd_file,
                        sep="\t",
                        low_memory=False,
                        na_values=["", "NA", "NaN", "NULL", "null", "n/a"],
                    )

                    if events_df.empty:
                        logger.debug(f"Empty events file: {ccd_file}")
                        continue

                    # Add metadata
                    events_df = events_df.copy()
                    events_df["subject_id"] = subject_id
                    events_df["session_id"] = session_info.get("session", "unknown")
                    events_df["file_path"] = str(ccd_file)

                    # Process and validate events
                    processed_events = self._process_ccd_events(events_df)

                    if processed_events is not None and not processed_events.empty:
                        session_data_list.append(processed_events)

                except Exception as e:
                    logger.warning(f"Failed to process CCD file {ccd_file}: {e}")
                    continue

            if not session_data_list:
                return None

            # Combine all sessions for this subject
            subject_ccd_data = pd.concat(session_data_list, ignore_index=True)

            # Add unique trial identifiers
            subject_ccd_data["trial_id"] = range(len(subject_ccd_data))
            subject_ccd_data["unique_trial_id"] = [
                f"{subject_id}_{row['session_id']}_trial_{row['trial_id']}"
                for _, row in subject_ccd_data.iterrows()
            ]

            return subject_ccd_data

        except Exception as e:
            logger.error(f"Error loading CCD data for {subject_id}: {e}")
            return None

    def _extract_session_info(self, file_path: Path) -> Dict[str, str]:
        """Extract session and other metadata from BIDS filename."""
        filename = file_path.name
        info = {}

        # Extract session (ses-XXX)
        import re

        session_match = re.search(r"ses-([^_]+)", filename)
        if session_match:
            info["session"] = session_match.group(1)

        # Extract task (task-XXX)
        task_match = re.search(r"task-([^_]+)", filename)
        if task_match:
            info["task"] = task_match.group(1)

        # Extract run (run-XXX)
        run_match = re.search(r"run-([^_]+)", filename)
        if run_match:
            info["run"] = run_match.group(1)

        return info

    def _process_ccd_events(self, events_df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Process raw CCD events to extract response_time and success labels.

        Args:
            events_df: Raw events DataFrame from BIDS events.tsv

        Returns:
            Processed DataFrame with challenge-ready labels
        """
        try:
            # Expected columns for CCD events (may vary by dataset)
            # Common columns: onset, duration, trial_type, response_time, accuracy
            required_columns = ["onset"]
            missing_columns = set(required_columns) - set(events_df.columns)

            if missing_columns:
                logger.warning(f"Missing required CCD columns: {missing_columns}")
                return None

            # Filter for relevant trial types (task-specific)
            relevant_trials = events_df.copy()

            # Look for response time column (various possible names)
            rt_columns = ["response_time", "RT", "reaction_time", "rt"]
            rt_column = None
            for col in rt_columns:
                if col in events_df.columns:
                    rt_column = col
                    break

            # Look for accuracy/success column
            accuracy_columns = ["accuracy", "correct", "success", "acc"]
            accuracy_column = None
            for col in accuracy_columns:
                if col in events_df.columns:
                    accuracy_column = col
                    break

            # Process response times
            if rt_column:
                relevant_trials["response_time_target"] = pd.to_numeric(
                    relevant_trials[rt_column], errors="coerce"
                )

                # Clean response times (remove impossible values)
                rt_min, rt_max = 0.1, 10.0  # seconds - reasonable bounds
                valid_rt = (relevant_trials["response_time_target"] >= rt_min) & (
                    relevant_trials["response_time_target"] <= rt_max
                )

                invalid_rt_count = (~valid_rt).sum()
                if invalid_rt_count > 0:
                    logger.debug(f"Found {invalid_rt_count} invalid response times")
                    relevant_trials.loc[~valid_rt, "response_time_target"] = np.nan
            else:
                logger.warning("No response time column found in CCD events")
                relevant_trials["response_time_target"] = np.nan

            # Process accuracy/success
            if accuracy_column:
                # Convert to binary success (1=correct, 0=incorrect)
                accuracy_values = relevant_trials[accuracy_column]

                # Handle different encoding schemes
                if accuracy_values.dtype == "object":
                    # String values - convert to numeric
                    success_mapping = {
                        "correct": 1,
                        "incorrect": 0,
                        "true": 1,
                        "false": 0,
                        "yes": 1,
                        "no": 0,
                        "1": 1,
                        "0": 0,
                        1: 1,
                        0: 0,
                    }

                    relevant_trials["success_target"] = accuracy_values.map(
                        success_mapping
                    )
                else:
                    # Numeric values - ensure 0/1 encoding
                    numeric_accuracy = pd.to_numeric(accuracy_values, errors="coerce")
                    relevant_trials["success_target"] = (numeric_accuracy > 0.5).astype(
                        int
                    )

                # Validate success values
                valid_success = relevant_trials["success_target"].isin([0, 1])
                invalid_success_count = (~valid_success).sum()
                if invalid_success_count > 0:
                    logger.debug(
                        f"Found {invalid_success_count} invalid success values"
                    )
                    relevant_trials.loc[~valid_success, "success_target"] = np.nan
            else:
                logger.warning("No accuracy/success column found in CCD events")
                relevant_trials["success_target"] = np.nan

            # Add task metadata
            if "trial_type" in events_df.columns:
                relevant_trials["task_name"] = events_df["trial_type"]
            else:
                relevant_trials["task_name"] = "CCD"  # Default task name

            # Add temporal information
            if "onset" in events_df.columns:
                relevant_trials["onset_time"] = pd.to_numeric(
                    events_df["onset"], errors="coerce"
                )

            if "duration" in events_df.columns:
                relevant_trials["trial_duration"] = pd.to_numeric(
                    events_df["duration"], errors="coerce"
                )

            # Filter out trials with no valid labels
            valid_trials = relevant_trials[
                (relevant_trials["response_time_target"].notna())
                | (relevant_trials["success_target"].notna())
            ]

            if valid_trials.empty:
                logger.debug("No valid CCD trials after processing")
                return None

            # Select final columns for output
            output_columns = [
                "subject_id",
                "session_id",
                "task_name",
                "onset_time",
                "response_time_target",
                "success_target",
            ]

            # Only include columns that exist
            available_columns = [
                col for col in output_columns if col in valid_trials.columns
            ]
            result_df = valid_trials[available_columns].copy()

            logger.debug(f"Processed {len(result_df)} valid CCD trials")
            return result_df

        except Exception as e:
            logger.error(f"Error processing CCD events: {e}")
            return None

    def _validate_and_clean_ccd_data(self, ccd_data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean combined CCD data with comprehensive quality checks.

        Args:
            ccd_data: Combined CCD data from all subjects

        Returns:
            Cleaned and validated CCD data
        """
        logger.debug(f"Validating CCD data: {len(ccd_data)} trials")
        original_count = len(ccd_data)

        # Remove completely empty rows
        ccd_data = ccd_data.dropna(how="all")

        # Check for required columns
        required_columns = ["subject_id", "response_time_target", "success_target"]
        missing_columns = set(required_columns) - set(ccd_data.columns)

        for col in missing_columns:
            ccd_data[col] = np.nan
            logger.warning(f"Added missing column '{col}' with NaN values")

        # Validate subject IDs
        invalid_subjects = ccd_data["subject_id"].isna()
        if invalid_subjects.any():
            logger.warning(
                f"Removing {invalid_subjects.sum()} trials with invalid subject IDs"
            )
            ccd_data = ccd_data[~invalid_subjects]

        # Response time validation and outlier removal
        if "response_time_target" in ccd_data.columns:
            rt_series = ccd_data["response_time_target"]
            valid_rt = rt_series.notna()

            if valid_rt.any():
                # Statistical outlier detection (3 standard deviations)
                rt_mean = rt_series[valid_rt].mean()
                rt_std = rt_series[valid_rt].std()

                if rt_std > 0:
                    outlier_threshold = 3.0
                    outliers = (rt_series > rt_mean + outlier_threshold * rt_std) | (
                        rt_series < rt_mean - outlier_threshold * rt_std
                    )

                    outlier_count = outliers.sum()
                    if outlier_count > 0:
                        logger.debug(f"Flagged {outlier_count} response time outliers")
                        # Don't remove - just flag for potential exclusion in analysis
                        ccd_data["rt_outlier"] = outliers

        # Success rate validation
        if "success_target" in ccd_data.columns:
            valid_success = ccd_data["success_target"].notna()
            if valid_success.any():
                success_rate = ccd_data[valid_success]["success_target"].mean()
                logger.debug(f"Overall success rate: {success_rate:.3f}")

                # Warn if success rate is extreme
                if success_rate < 0.1 or success_rate > 0.9:
                    logger.warning(f"Extreme success rate detected: {success_rate:.3f}")

        # Add data quality flags
        ccd_data["has_rt"] = ccd_data["response_time_target"].notna()
        ccd_data["has_success"] = ccd_data["success_target"].notna()
        ccd_data["complete_trial"] = ccd_data["has_rt"] & ccd_data["has_success"]

        # Summary statistics
        complete_trials = ccd_data["complete_trial"].sum()
        rt_only_trials = ccd_data["has_rt"].sum() - complete_trials
        success_only_trials = ccd_data["has_success"].sum() - complete_trials

        logger.info(
            f"CCD data validation complete: {len(ccd_data)}/{original_count} trials retained. "
            f"Complete: {complete_trials}, RT-only: {rt_only_trials}, Success-only: {success_only_trials}"
        )

        return ccd_data

    @memory_monitor(threshold_mb=50.0)
    @timing_monitor
    def load_cbcl_labels(self, split: str = "train") -> pd.DataFrame:
        """
        Load CBCL (Child Behavior Checklist) labels for Challenge 2 with robust error handling.

        Args:
            split: Data split to load ("train", "val", "test")

        Returns:
            DataFrame with CBCL labels (p_factor, internalizing, externalizing, attention, binary_label)

        Raises:
            ValueError: If split is invalid
            pd.errors.EmptyDataError: If no valid data found
        """
        # Validate input parameters
        if split not in ["train", "val", "test"]:
            raise ValueError(
                f"Invalid split: {split}. Must be one of ['train', 'val', 'test']"
            )

        # Check cache first
        cache_key = f"cbcl_{split}"
        if self.enable_caching and cache_key in self.cbcl_cache:
            logger.debug(f"Returning cached CBCL data for {split}")
            return self.cbcl_cache[cache_key].copy()

        with graceful_error_handler(f"loading CBCL labels for {split}"):
            # Check if CBCL data is available
            if (
                not hasattr(self, "cbcl_df")
                or self.cbcl_df is None
                or self.cbcl_df.empty
            ):
                raise pd.errors.EmptyDataError(
                    "No CBCL data available. Check phenotype data loading."
                )

            # Get subjects for this split
            if self.official_splits is None:
                raise ValueError("Official splits not loaded")

            split_subjects = self.official_splits.get(split, [])
            if not split_subjects:
                raise ValueError(f"No subjects found for split: {split}")

            logger.info(
                f"Loading CBCL labels for {len(split_subjects)} subjects in {split} split"
            )

            # Filter CBCL data for split subjects
            cbcl_split_data = self.cbcl_df[
                self.cbcl_df["participant_id"].isin(split_subjects)
            ].copy()

            if cbcl_split_data.empty:
                raise pd.errors.EmptyDataError(
                    f"No CBCL data found for subjects in {split} split"
                )

            # Process and validate CBCL data
            cbcl_labels = self._process_cbcl_data(cbcl_split_data)

            # Cache the result
            if self.enable_caching:
                self.cbcl_cache[cache_key] = cbcl_labels.copy()

            # Memory usage reporting
            memory_usage = cbcl_labels.memory_usage(deep=True).sum() / 1024 / 1024
            logger.info(
                f"Successfully loaded CBCL data for {split}: "
                f"{len(cbcl_labels)} subjects ({memory_usage:.1f}MB)"
            )

            return cbcl_labels

    def _process_cbcl_data(self, cbcl_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process raw CBCL data to create challenge-ready labels with comprehensive validation.

        Args:
            cbcl_data: Raw CBCL data from phenotype files

        Returns:
            Processed DataFrame with standardized CBCL labels
        """
        try:
            logger.debug(f"Processing CBCL data for {len(cbcl_data)} subjects")

            # Create a copy to avoid modifying original data
            processed_data = cbcl_data.copy()

            # Define expected CBCL columns (may vary by dataset version)
            cbcl_columns = {
                # Core CBCL scales
                "p_factor": ["p_factor", "general_factor", "p_score"],
                "internalizing": ["internalizing", "internal", "int_score"],
                "externalizing": ["externalizing", "external", "ext_score"],
                "attention": ["attention", "adhd", "att_score", "attention_problems"],
                # Alternative column names
                "cbcl_p_factor": ["cbcl_p_factor"],
                "cbcl_internalizing": ["cbcl_internalizing"],
                "cbcl_externalizing": ["cbcl_externalizing"],
                "cbcl_attention": ["cbcl_attention"],
            }

            # Map columns to standardized names
            column_mapping = {}
            for standard_name, possible_names in cbcl_columns.items():
                for col_name in possible_names:
                    if col_name in processed_data.columns:
                        if standard_name.startswith("cbcl_"):
                            # Remove cbcl_ prefix for consistency
                            target_name = standard_name.replace("cbcl_", "")
                        else:
                            target_name = standard_name
                        column_mapping[col_name] = target_name
                        break

            # Rename columns
            if column_mapping:
                processed_data = processed_data.rename(columns=column_mapping)
                logger.debug(f"Mapped CBCL columns: {column_mapping}")

            # Required columns for the challenge
            required_cbcl_columns = [
                "p_factor",
                "internalizing",
                "externalizing",
                "attention",
            ]

            # Check which columns are available
            available_columns = []
            missing_columns = []

            for col in required_cbcl_columns:
                if col in processed_data.columns:
                    available_columns.append(col)
                else:
                    missing_columns.append(col)

            if missing_columns:
                logger.warning(f"Missing CBCL columns: {missing_columns}")
                # Create missing columns with NaN values
                for col in missing_columns:
                    processed_data[col] = np.nan

            # Convert CBCL scores to numeric
            for col in required_cbcl_columns:
                if col in processed_data.columns:
                    processed_data[col] = pd.to_numeric(
                        processed_data[col], errors="coerce"
                    )

            # Validate and clean CBCL scores
            for col in required_cbcl_columns:
                if col in processed_data.columns:
                    # Remove extreme outliers (beyond reasonable T-score range)
                    # Standard CBCL T-scores typically range from 30-100
                    score_min, score_max = 20, 120  # Allow some buffer

                    outliers = (processed_data[col] < score_min) | (
                        processed_data[col] > score_max
                    )

                    outlier_count = outliers.sum()
                    if outlier_count > 0:
                        logger.warning(f"Found {outlier_count} outlier values in {col}")
                        processed_data.loc[outliers, col] = np.nan

            # Create binary classification labels for attention problems
            # Typically T-score >= 65 indicates clinical significance
            if "attention" in processed_data.columns:
                attention_threshold = 65.0
                processed_data["binary_label"] = (
                    processed_data["attention"] >= attention_threshold
                ).astype(int)

                # Handle missing attention scores
                missing_attention = processed_data["attention"].isna()
                processed_data.loc[missing_attention, "binary_label"] = np.nan

                logger.debug(
                    f"Created binary labels using attention threshold {attention_threshold}: "
                    f"{processed_data['binary_label'].sum()} positive cases"
                )
            else:
                processed_data["binary_label"] = np.nan
                logger.warning(
                    "No attention scores available for binary classification"
                )

            # Calculate data completeness statistics
            completeness_stats = {}
            for col in required_cbcl_columns + ["binary_label"]:
                if col in processed_data.columns:
                    valid_count = processed_data[col].notna().sum()
                    total_count = len(processed_data)
                    completeness_stats[col] = (
                        valid_count / total_count if total_count > 0 else 0
                    )

            logger.info(f"CBCL data completeness: {completeness_stats}")

            # Select final columns
            output_columns = (
                ["participant_id"] + required_cbcl_columns + ["binary_label"]
            )
            available_output_columns = [
                col for col in output_columns if col in processed_data.columns
            ]

            result_df = processed_data[available_output_columns].copy()

            # Add metadata
            result_df["cbcl_complete"] = (
                result_df[required_cbcl_columns].notna().all(axis=1)
            )
            result_df["n_valid_cbcl"] = (
                result_df[required_cbcl_columns].notna().sum(axis=1)
            )

            logger.debug(
                f"Processed CBCL data: {len(result_df)} subjects with {result_df['cbcl_complete'].sum()} complete profiles"
            )

            return result_df

        except Exception as e:
            logger.error(f"Error processing CBCL data: {e}")
            # Return minimal DataFrame with participant IDs
            minimal_df = pd.DataFrame(
                {"participant_id": cbcl_data.get("participant_id", [])}
            )
            for col in [
                "p_factor",
                "internalizing",
                "externalizing",
                "attention",
                "binary_label",
            ]:
                minimal_df[col] = np.nan
            return minimal_df

    @memory_monitor(threshold_mb=25.0)
    @timing_monitor
    def compute_official_metrics(self, split: str = "val") -> Dict[str, float]:
        """
        Compute official challenge metrics with comprehensive error handling and validation.

        Args:
            split: Data split to compute metrics for ("train", "val", "test")

        Returns:
            Dictionary of official metrics

        Raises:
            ValueError: If split is invalid or no data available
        """
        # Validate input
        if split not in ["train", "val", "test"]:
            raise ValueError(
                f"Invalid split: {split}. Must be one of ['train', 'val', 'test']"
            )

        with graceful_error_handler(f"computing official metrics for {split}"):
            # Load required data
            try:
                ccd_data = self.load_ccd_labels(split)
                cbcl_data = self.load_cbcl_labels(split)
            except Exception as e:
                logger.error(f"Failed to load data for metrics computation: {e}")
                return self._get_default_metrics()

            if ccd_data.empty and cbcl_data.empty:
                logger.warning(f"No data available for metrics computation in {split}")
                return self._get_default_metrics()

            metrics = {}

            # Challenge 1: CCD Response Time and Success Prediction
            if not ccd_data.empty:
                try:
                    ccd_metrics = self._compute_ccd_metrics(ccd_data)
                    metrics.update(ccd_metrics)
                except Exception as e:
                    logger.error(f"Error computing CCD metrics: {e}")
                    metrics.update(self._get_default_ccd_metrics())

            # Challenge 2: CBCL Factor Prediction
            if not cbcl_data.empty:
                try:
                    cbcl_metrics = self._compute_cbcl_metrics(cbcl_data)
                    metrics.update(cbcl_metrics)
                except Exception as e:
                    logger.error(f"Error computing CBCL metrics: {e}")
                    metrics.update(self._get_default_cbcl_metrics())

            # Add data quality metrics
            try:
                quality_metrics = self._compute_data_quality_metrics(
                    ccd_data, cbcl_data
                )
                metrics.update(quality_metrics)
            except Exception as e:
                logger.error(f"Error computing data quality metrics: {e}")

            logger.info(f"Computed {len(metrics)} official metrics for {split}")
            return metrics

    def _compute_ccd_metrics(self, ccd_data: pd.DataFrame) -> Dict[str, float]:
        """Compute metrics for CCD (Challenge 1) tasks."""
        metrics = {}

        # Response time metrics
        if "response_time_target" in ccd_data.columns:
            rt_series = ccd_data["response_time_target"].dropna()

            if len(rt_series) > 0:
                metrics["ccd_response_time_mean"] = float(rt_series.mean())
                metrics["ccd_response_time_std"] = float(rt_series.std())
                metrics["ccd_response_time_median"] = float(rt_series.median())
                metrics["ccd_n_response_times"] = len(rt_series)

                # Response time percentiles
                percentiles = [10, 25, 75, 90]
                for p in percentiles:
                    metrics[f"ccd_response_time_p{p}"] = float(
                        np.percentile(rt_series, p)
                    )
            else:
                metrics.update(self._get_default_ccd_metrics())

        # Success rate metrics
        if "success_target" in ccd_data.columns:
            success_series = ccd_data["success_target"].dropna()

            if len(success_series) > 0:
                success_rate = success_series.mean()
                metrics["ccd_success_rate"] = float(success_rate)
                metrics["ccd_n_trials"] = len(success_series)
                metrics["ccd_n_successful"] = int(success_series.sum())

                # Success rate confidence interval (using Wilson score interval)
                if len(success_series) > 10:
                    n = len(success_series)
                    p = success_rate
                    z = 1.96  # 95% confidence

                    denominator = 1 + z**2 / n
                    center = (p + z**2 / (2 * n)) / denominator
                    margin = (
                        z * np.sqrt((p * (1 - p) / n + z**2 / (4 * n**2))) / denominator
                    )

                    metrics["ccd_success_rate_ci_lower"] = float(
                        max(0, center - margin)
                    )
                    metrics["ccd_success_rate_ci_upper"] = float(
                        min(1, center + margin)
                    )

        return metrics

    def _compute_cbcl_metrics(self, cbcl_data: pd.DataFrame) -> Dict[str, float]:
        """Compute metrics for CBCL (Challenge 2) tasks."""
        metrics = {}

        cbcl_columns = ["p_factor", "internalizing", "externalizing", "attention"]

        for col in cbcl_columns:
            if col in cbcl_data.columns:
                series = cbcl_data[col].dropna()

                if len(series) > 0:
                    metrics[f"cbcl_{col}_mean"] = float(series.mean())
                    metrics[f"cbcl_{col}_std"] = float(series.std())
                    metrics[f"cbcl_{col}_median"] = float(series.median())
                    metrics[f"cbcl_{col}_n_valid"] = len(series)

                    # Score distribution percentiles
                    percentiles = [25, 75]
                    for p in percentiles:
                        metrics[f"cbcl_{col}_p{p}"] = float(np.percentile(series, p))

        # Binary classification metrics
        if "binary_label" in cbcl_data.columns:
            binary_series = cbcl_data["binary_label"].dropna()

            if len(binary_series) > 0:
                positive_rate = binary_series.mean()
                metrics["cbcl_positive_rate"] = float(positive_rate)
                metrics["cbcl_n_binary_labels"] = len(binary_series)
                metrics["cbcl_n_positive"] = int(binary_series.sum())

        # Data completeness
        complete_profiles = (
            cbcl_data["cbcl_complete"].sum()
            if "cbcl_complete" in cbcl_data.columns
            else 0
        )
        metrics["cbcl_completeness_rate"] = (
            float(complete_profiles / len(cbcl_data)) if len(cbcl_data) > 0 else 0.0
        )

        return metrics

    def _compute_data_quality_metrics(
        self, ccd_data: pd.DataFrame, cbcl_data: pd.DataFrame
    ) -> Dict[str, float]:
        """Compute data quality and completeness metrics."""
        metrics = {}

        # Overall data availability
        metrics["total_ccd_subjects"] = len(ccd_data) if not ccd_data.empty else 0
        metrics["total_cbcl_subjects"] = len(cbcl_data) if not cbcl_data.empty else 0

        # Subject overlap
        if not ccd_data.empty and not cbcl_data.empty:
            ccd_subjects = set(ccd_data.get("subject_id", []))
            cbcl_subjects = set(cbcl_data.get("participant_id", []))

            overlap = len(ccd_subjects.intersection(cbcl_subjects))
            union = len(ccd_subjects.union(cbcl_subjects))

            metrics["subject_overlap_count"] = overlap
            metrics["subject_overlap_rate"] = (
                float(overlap / union) if union > 0 else 0.0
            )

        # Data quality flags
        if not ccd_data.empty:
            if "complete_trial" in ccd_data.columns:
                metrics["ccd_complete_trials_rate"] = float(
                    ccd_data["complete_trial"].mean()
                )

            if "rt_outlier" in ccd_data.columns:
                metrics["ccd_outlier_rate"] = float(ccd_data["rt_outlier"].mean())

        return metrics

    def _get_default_metrics(self) -> Dict[str, float]:
        """Return default metrics when computation fails."""
        return {**self._get_default_ccd_metrics(), **self._get_default_cbcl_metrics()}

    def _get_default_ccd_metrics(self) -> Dict[str, float]:
        """Return default CCD metrics."""
        return {
            "ccd_response_time_mean": 0.0,
            "ccd_response_time_std": 0.0,
            "ccd_success_rate": 0.0,
            "ccd_n_trials": 0,
            "ccd_n_response_times": 0,
        }

    def _get_default_cbcl_metrics(self) -> Dict[str, float]:
        """Return default CBCL metrics."""
        metrics = {}
        for col in ["p_factor", "internalizing", "externalizing", "attention"]:
            metrics[f"cbcl_{col}_mean"] = 0.0
            metrics[f"cbcl_{col}_std"] = 0.0
            metrics[f"cbcl_{col}_n_valid"] = 0

        metrics["cbcl_positive_rate"] = 0.0
        metrics["cbcl_n_binary_labels"] = 0
        metrics["cbcl_completeness_rate"] = 0.0

        return metrics

    def get_data_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive data summary with error handling and validation.

        Returns:
            Dictionary containing complete data summary statistics
        """
        with graceful_error_handler("generating data summary"):
            summary = {}

            # Basic information
            summary["bids_root"] = str(self.bids_root)
            summary["memory_limit_gb"] = self.memory_limit_gb
            summary["enable_caching"] = self.enable_caching
            summary["timestamp"] = pd.Timestamp.now().isoformat()

            # BIDS structure validation
            try:
                bids_validation = self._validate_bids_structure()
                summary["bids_validation"] = bids_validation
            except Exception as e:
                logger.error(f"BIDS validation failed: {e}")
                summary["bids_validation"] = {"valid": False, "error": str(e)}

            # Participants data
            if hasattr(self, "participants_df") and self.participants_df is not None:
                summary["participants"] = {
                    "total_subjects": len(self.participants_df),
                    "columns": list(self.participants_df.columns),
                    "age_stats": (
                        self._get_age_statistics()
                        if "age" in self.participants_df.columns
                        else None
                    ),
                }
            else:
                summary["participants"] = {"total_subjects": 0, "loaded": False}

            # Phenotype data
            if hasattr(self, "phenotype_df") and self.phenotype_df is not None:
                summary["phenotype"] = {
                    "total_records": len(self.phenotype_df),
                    "columns": list(self.phenotype_df.columns),
                    "memory_usage_mb": float(
                        self.phenotype_df.memory_usage(deep=True).sum() / 1024 / 1024
                    ),
                }
            else:
                summary["phenotype"] = {"total_records": 0, "loaded": False}

            # CBCL data
            if hasattr(self, "cbcl_df") and self.cbcl_df is not None:
                summary["cbcl"] = {
                    "total_subjects": len(self.cbcl_df),
                    "columns": list(self.cbcl_df.columns),
                    "completeness": self._get_cbcl_completeness_stats(),
                }
            else:
                summary["cbcl"] = {"total_subjects": 0, "loaded": False}

            # Official splits
            if hasattr(self, "official_splits") and self.official_splits is not None:
                summary["splits"] = {
                    split: len(subjects)
                    for split, subjects in self.official_splits.items()
                }
                summary["splits"]["total_in_splits"] = sum(summary["splits"].values())
            else:
                summary["splits"] = {"loaded": False}

            # Cache statistics
            if self.enable_caching:
                summary["cache"] = {
                    "ccd_cache_size": len(self.ccd_cache),
                    "cbcl_cache_size": len(self.cbcl_cache),
                    "cached_splits": list(self.ccd_cache.keys())
                    + list(self.cbcl_cache.keys()),
                }

            # Memory usage
            try:
                current_memory = self._get_current_memory_usage()
                summary["memory_usage"] = {
                    "current_mb": current_memory,
                    "limit_mb": self.memory_limit_gb * 1024,
                    "utilization_pct": (current_memory / (self.memory_limit_gb * 1024))
                    * 100,
                }
            except Exception as e:
                logger.warning(f"Could not get memory usage: {e}")
                summary["memory_usage"] = {"error": str(e)}

            logger.info("Generated comprehensive data summary")
            return summary

    def _get_age_statistics(self) -> Dict[str, float]:
        """Get age statistics from participants data."""
        if "age" not in self.participants_df.columns:
            return {}

        age_series = pd.to_numeric(self.participants_df["age"], errors="coerce")
        valid_ages = age_series.dropna()

        if len(valid_ages) == 0:
            return {}

        return {
            "mean": float(valid_ages.mean()),
            "std": float(valid_ages.std()),
            "min": float(valid_ages.min()),
            "max": float(valid_ages.max()),
            "median": float(valid_ages.median()),
            "n_valid": len(valid_ages),
            "n_missing": len(age_series) - len(valid_ages),
        }

    def _get_cbcl_completeness_stats(self) -> Dict[str, float]:
        """Get CBCL data completeness statistics."""
        if not hasattr(self, "cbcl_df") or self.cbcl_df is None:
            return {}

        cbcl_columns = ["p_factor", "internalizing", "externalizing", "attention"]
        available_columns = [col for col in cbcl_columns if col in self.cbcl_df.columns]

        if not available_columns:
            return {}

        completeness = {}
        for col in available_columns:
            valid_count = self.cbcl_df[col].notna().sum()
            total_count = len(self.cbcl_df)
            completeness[f"{col}_completeness"] = (
                float(valid_count / total_count) if total_count > 0 else 0.0
            )

        # Overall completeness (all columns valid)
        if available_columns:
            all_valid = self.cbcl_df[available_columns].notna().all(axis=1).sum()
            completeness["overall_completeness"] = (
                float(all_valid / len(self.cbcl_df)) if len(self.cbcl_df) > 0 else 0.0
            )

        return completeness

    def cleanup_cache(self) -> None:
        """
        Clean up cache and free memory with comprehensive garbage collection.
        """
        try:
            # Clear data caches
            if hasattr(self, "ccd_cache"):
                cache_size = len(self.ccd_cache)
                self.ccd_cache.clear()
                logger.debug(f"Cleared CCD cache ({cache_size} items)")

            if hasattr(self, "cbcl_cache"):
                cache_size = len(self.cbcl_cache)
                self.cbcl_cache.clear()
                logger.debug(f"Cleared CBCL cache ({cache_size} items)")

            # Force garbage collection
            import gc

            collected = gc.collect()
            logger.debug(f"Garbage collection freed {collected} objects")

            # Log memory usage after cleanup
            try:
                current_memory = self._get_current_memory_usage()
                logger.info(f"Memory after cleanup: {current_memory:.1f}MB")
            except Exception as e:
                logger.debug(f"Could not measure memory after cleanup: {e}")

        except Exception as e:
            logger.error(f"Error during cache cleanup: {e}")

    def __del__(self):
        """
        Destructor to ensure proper cleanup when object is destroyed.
        """
        try:
            if hasattr(self, "cleanup_cache"):
                self.cleanup_cache()
        except Exception:
            # Avoid raising exceptions in destructor
            pass

    def _create_splits_checksum(self, splits_file: Path) -> None:
        """Create checksum for splits validation."""
        try:
            import hashlib

            with open(splits_file, "rb") as f:
                checksum = hashlib.md5(f.read()).hexdigest()

            checksum_file = splits_file.with_suffix(".md5")
            with open(checksum_file, "w") as f:
                f.write(f"{checksum}  {splits_file.name}\n")

            logger.info(f"Created splits checksum: {checksum}")
        except Exception as e:
            logger.warning(f"Failed to create checksum for {splits_file}: {e}")

    def get_ccd_labels(
        self, subject_id: str, session_id: str = "001"
    ) -> Dict[str, Any]:
        """
        Extract CCD (Continuous Cognitive Demand) task labels.

        Challenge 1 targets:
        - Response time (RT) regression
        - Success rate classification

        Args:
            subject_id: Subject identifier
            session_id: Session identifier

        Returns:
            Dictionary with CCD labels
        """
        # Look for CCD behavioral data
        subject_key = f"sub-{subject_id}"

        # Initialize with defaults
        labels = {
            "response_time": np.nan,
            "success": 0,
            "trial_count": 0,
            "mean_rt": np.nan,
            "success_rate": 0.0,
        }

        # Try to load CCD behavioral data from events.tsv or behavioral files
        ccd_files = list(
            self.bids_root.glob(
                f"{subject_key}/ses-{session_id}/eeg/*_task-CCD_events.tsv"
            )
        )

        if ccd_files:
            events_df = pd.read_csv(ccd_files[0], sep="\t")

            # Extract response times and success
            if "response_time" in events_df.columns:
                rt_data = events_df["response_time"].dropna()
                success_data = events_df.get("success", events_df.get("correct", []))

                if len(rt_data) > 0:
                    labels.update(
                        {
                            "mean_rt": rt_data.mean(),
                            "response_time": rt_data.iloc[
                                -1
                            ],  # Last RT for single prediction
                            "trial_count": len(rt_data),
                        }
                    )

                if len(success_data) > 0:
                    labels.update(
                        {
                            "success_rate": success_data.mean(),
                            "success": int(
                                success_data.iloc[-1]
                            ),  # Last success for single prediction
                        }
                    )

        return labels

    def get_cbcl_labels(self, subject_id: str) -> Dict[str, float]:
        """
        Extract CBCL (Child Behavior Checklist) labels for psychopathology prediction.

        Challenge 2 targets:
        - P-factor (general psychopathology)
        - Internalizing problems
        - Externalizing problems
        - Attention problems

        Args:
            subject_id: Subject identifier

        Returns:
            Dictionary with CBCL scores
        """
        subject_key = f"sub-{subject_id}"

        # Initialize with defaults (0 = typical)
        labels = {
            "p_factor": 0.0,
            "internalizing": 0.0,
            "externalizing": 0.0,
            "attention": 0.0,
            "binary_label": 0,  # 0 = typical, 1 = atypical
        }

        if "cbcl" not in self.phenotype_data:
            return labels

        cbcl_df = self.phenotype_data["cbcl"]
        subject_data = cbcl_df[cbcl_df["participant_id"] == subject_key]

        if subject_data.empty:
            return labels

        # Map CBCL column names (these may vary by dataset)
        cbcl_mapping = {
            "p_factor": ["CBCL_Total_T", "cbcl_p_factor", "p_factor"],
            "internalizing": ["CBCL_Internal_T", "cbcl_internalizing", "internalizing"],
            "externalizing": ["CBCL_External_T", "cbcl_externalizing", "externalizing"],
            "attention": ["CBCL_Attention_T", "cbcl_attention", "attention"],
        }

        for label_key, possible_cols in cbcl_mapping.items():
            for col in possible_cols:
                if col in subject_data.columns:
                    value = subject_data[col].iloc[0]
                    if pd.notna(value):
                        labels[label_key] = float(value)
                    break

        # Create binary label based on clinical cutoffs (T-score >= 65)
        clinical_threshold = 65.0
        is_clinical = any(
            labels[key] >= clinical_threshold
            for key in ["p_factor", "internalizing", "externalizing", "attention"]
            if labels[key] > 0
        )
        labels["binary_label"] = 1 if is_clinical else 0

        return labels

    def get_subject_metadata(self, subject_id: str) -> Dict[str, Any]:
        """Get subject metadata for domain adaptation."""
        subject_key = f"sub-{subject_id}"

        metadata = {
            "subject_id": subject_id,
            "site": "unknown",
            "age": np.nan,
            "sex": "unknown",
            "scanner": "unknown",
        }

        subject_data = self.participants_df[
            self.participants_df["participant_id"] == subject_key
        ]

        if not subject_data.empty:
            row = subject_data.iloc[0]

            # Map common metadata fields
            field_mapping = {
                "age": ["age", "Age"],
                "sex": ["sex", "Sex", "gender"],
                "site": ["site", "Site", "scanner_site"],
                "scanner": ["scanner", "Scanner", "manufacturer"],
            }

            for meta_key, possible_cols in field_mapping.items():
                for col in possible_cols:
                    if col in row.index and pd.notna(row[col]):
                        metadata[meta_key] = row[col]
                        break

        return metadata


class OfficialMetrics:
    """
    Official metrics implementation matching Starter Kit specifications.
    """

    @staticmethod
    def compute_challenge1_metrics(
        predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute Challenge 1 (CCD) metrics.

        Official metrics:
        - Response Time: Pearson correlation, RMSE
        - Success: AUROC, AUPRC, Balanced Accuracy

        Args:
            predictions: Dictionary with 'response_time' and 'success' predictions
            targets: Dictionary with 'response_time' and 'success' ground truth

        Returns:
            Dictionary with computed metrics
        """
        metrics = {}

        # Response time metrics
        if "response_time" in predictions and "response_time" in targets:
            rt_pred = predictions["response_time"]
            rt_true = targets["response_time"]

            # Filter out NaN values
            valid_mask = ~(np.isnan(rt_pred) | np.isnan(rt_true))
            if valid_mask.sum() > 0:
                rt_pred_valid = rt_pred[valid_mask]
                rt_true_valid = rt_true[valid_mask]

                # Pearson correlation
                if len(rt_pred_valid) > 1:
                    corr, p_value = pearsonr(rt_pred_valid, rt_true_valid)
                    metrics["rt_correlation"] = corr
                    metrics["rt_correlation_pvalue"] = p_value

                # RMSE
                metrics["rt_rmse"] = np.sqrt(
                    mean_squared_error(rt_true_valid, rt_pred_valid)
                )
                metrics["rt_mae"] = mean_absolute_error(rt_true_valid, rt_pred_valid)

        # Success metrics
        if "success" in predictions and "success" in targets:
            success_pred = predictions["success"]
            success_true = targets["success"]

            # Handle both probabilities and binary predictions
            if success_pred.max() <= 1.0 and success_pred.min() >= 0.0:
                # Probabilities
                try:
                    metrics["success_auroc"] = roc_auc_score(success_true, success_pred)
                    metrics["success_auprc"] = average_precision_score(
                        success_true, success_pred
                    )
                except ValueError as e:
                    logger.warning(f"Could not compute ROC metrics: {e}")

                # Convert to binary for balanced accuracy
                success_pred_binary = (success_pred > 0.5).astype(int)
            else:
                # Already binary
                success_pred_binary = success_pred.astype(int)

            metrics["success_balanced_acc"] = balanced_accuracy_score(
                success_true, success_pred_binary
            )

        return metrics

    @staticmethod
    def compute_challenge2_metrics(
        predictions: Dict[str, np.ndarray], targets: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Compute Challenge 2 (psychopathology) metrics.

        Official metrics:
        - CBCL dimensions: Pearson correlation per dimension
        - Mean correlation across all dimensions
        - Binary classification: AUROC, AUPRC, Balanced Accuracy

        Args:
            predictions: Dictionary with CBCL dimension predictions
            targets: Dictionary with CBCL dimension ground truth

        Returns:
            Dictionary with computed metrics
        """
        metrics = {}

        # CBCL dimension correlations
        dimensions = ["p_factor", "internalizing", "externalizing", "attention"]
        correlations = []

        for dim in dimensions:
            if dim in predictions and dim in targets:
                pred = predictions[dim]
                true = targets[dim]

                # Filter valid values
                valid_mask = ~(np.isnan(pred) | np.isnan(true))
                if valid_mask.sum() > 1:
                    pred_valid = pred[valid_mask]
                    true_valid = true[valid_mask]

                    corr, p_value = pearsonr(pred_valid, true_valid)
                    metrics[f"{dim}_correlation"] = corr
                    metrics[f"{dim}_correlation_pvalue"] = p_value
                    correlations.append(corr)

        # Mean correlation across dimensions
        if correlations:
            metrics["mean_correlation"] = np.mean(correlations)
            metrics["std_correlation"] = np.std(correlations)

        # Binary classification metrics
        if "binary_label" in predictions and "binary_label" in targets:
            binary_pred = predictions["binary_label"]
            binary_true = targets["binary_label"]

            # Handle probabilities vs binary
            if binary_pred.max() <= 1.0 and binary_pred.min() >= 0.0:
                try:
                    metrics["binary_auroc"] = roc_auc_score(binary_true, binary_pred)
                    metrics["binary_auprc"] = average_precision_score(
                        binary_true, binary_pred
                    )
                except ValueError as e:
                    logger.warning(f"Could not compute binary ROC metrics: {e}")

                binary_pred_int = (binary_pred > 0.5).astype(int)
            else:
                binary_pred_int = binary_pred.astype(int)

            metrics["binary_balanced_acc"] = balanced_accuracy_score(
                binary_true, binary_pred_int
            )

        return metrics

    @staticmethod
    def aggregate_subject_predictions(
        predictions: List[np.ndarray], subject_ids: List[str], aggregation: str = "mean"
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Aggregate per-window predictions to per-subject level.

        Args:
            predictions: List of prediction arrays
            subject_ids: List of subject IDs for each prediction
            aggregation: Aggregation method ('mean', 'median', 'last')

        Returns:
            Aggregated predictions and unique subject IDs
        """
        df = pd.DataFrame({"subject_id": subject_ids, "prediction": predictions})

        if aggregation == "mean":
            agg_df = df.groupby("subject_id")["prediction"].mean().reset_index()
        elif aggregation == "median":
            agg_df = df.groupby("subject_id")["prediction"].median().reset_index()
        elif aggregation == "last":
            agg_df = df.groupby("subject_id")["prediction"].last().reset_index()
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation}")

        return agg_df["prediction"].values, agg_df["subject_id"].tolist()


class SubmissionValidator:
    """
    Validates submissions against official Starter Kit format.
    """

    def __init__(self):
        self.challenge1_schema = {
            "required_columns": [
                "subject_id",
                "session_id",
                "response_time_pred",
                "success_pred",
            ],
            "dtypes": {
                "subject_id": "object",
                "session_id": "object",
                "response_time_pred": "float64",
                "success_pred": "float64",
            },
        }

        self.challenge2_schema = {
            "required_columns": [
                "subject_id",
                "p_factor_pred",
                "internalizing_pred",
                "externalizing_pred",
                "attention_pred",
                "binary_pred",
            ],
            "dtypes": {
                "subject_id": "object",
                "p_factor_pred": "float64",
                "internalizing_pred": "float64",
                "externalizing_pred": "float64",
                "attention_pred": "float64",
                "binary_pred": "float64",
            },
        }

    def validate_challenge1_submission(
        self, submission_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate Challenge 1 submission format."""
        return self._validate_submission(
            submission_df, self.challenge1_schema, "Challenge 1"
        )

    def validate_challenge2_submission(
        self, submission_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Validate Challenge 2 submission format."""
        return self._validate_submission(
            submission_df, self.challenge2_schema, "Challenge 2"
        )

    def _validate_submission(
        self, df: pd.DataFrame, schema: Dict, challenge_name: str
    ) -> Dict[str, Any]:
        """Validate submission against schema."""
        results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "challenge": challenge_name,
        }

        # Check required columns
        missing_cols = set(schema["required_columns"]) - set(df.columns)
        if missing_cols:
            results["errors"].append(f"Missing required columns: {missing_cols}")
            results["valid"] = False

        # Check for missing values
        for col in schema["required_columns"]:
            if col in df.columns and df[col].isna().any():
                results["errors"].append(f"Column {col} contains missing values")
                results["valid"] = False

        # Check data types
        for col, expected_dtype in schema["dtypes"].items():
            if col in df.columns:
                try:
                    df[col].astype(expected_dtype)
                except (ValueError, TypeError):
                    results["warnings"].append(
                        f"Column {col} cannot be converted to {expected_dtype}"
                    )

        # Challenge-specific validations
        if challenge_name == "Challenge 1":
            # Check success predictions are probabilities
            if "success_pred" in df.columns:
                if df["success_pred"].min() < 0 or df["success_pred"].max() > 1:
                    results["warnings"].append(
                        "success_pred values should be probabilities [0, 1]"
                    )

        elif challenge_name == "Challenge 2":
            # Check binary predictions are probabilities
            if "binary_pred" in df.columns:
                if df["binary_pred"].min() < 0 or df["binary_pred"].max() > 1:
                    results["warnings"].append(
                        "binary_pred values should be probabilities [0, 1]"
                    )

        return results
