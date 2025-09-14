"""
Submission packaging for EEG2025 Challenge.

This module handles:
1. CSV export for cross-task transfer predictions (cross_task_submission.csv)
2. CSV export for psychopathology predictions (psychopathology_submission.csv)
3. Schema validation and compliance checking
4. Submission manifest generation
5. Archive creation for final submission
"""

import csv
import hashlib
import json
import logging
import os
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SubmissionMetadata:
    """Metadata for submission package."""

    team_name: str
    submission_id: str
    timestamp: str
    challenge_track: str  # "cross_task" or "psychopathology" or "both"
    model_description: str
    training_duration_hours: float
    num_parameters: int
    cross_validation_folds: int
    best_validation_score: Optional[float] = None
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class CrossTaskPrediction:
    """Single cross-task prediction entry."""

    subject_id: str
    session_id: str
    task: str  # Target task name
    prediction: float  # Predicted score/performance
    confidence: Optional[float] = None  # Model confidence

    def validate(self) -> bool:
        """Validate prediction entry."""
        # Check required fields
        if not all([self.subject_id, self.session_id, self.task]):
            return False

        # Check prediction is numeric
        if not isinstance(self.prediction, (int, float)) or np.isnan(self.prediction):
            return False

        # Check confidence if provided
        if self.confidence is not None:
            if not isinstance(self.confidence, (int, float)) or np.isnan(
                self.confidence
            ):
                return False
            if not (0 <= self.confidence <= 1):
                return False

        return True


@dataclass
class PsychopathologyPrediction:
    """Single psychopathology prediction entry."""

    subject_id: str
    p_factor: float
    internalizing: float
    externalizing: float
    attention: float
    confidence_p_factor: Optional[float] = None
    confidence_internalizing: Optional[float] = None
    confidence_externalizing: Optional[float] = None
    confidence_attention: Optional[float] = None

    def validate(self) -> bool:
        """Validate prediction entry."""
        # Check required fields
        if not self.subject_id:
            return False

        # Check all predictions are numeric
        predictions = [
            self.p_factor,
            self.internalizing,
            self.externalizing,
            self.attention,
        ]
        if not all(
            isinstance(p, (int, float)) and not np.isnan(p) for p in predictions
        ):
            return False

        # Check confidences if provided
        confidences = [
            self.confidence_p_factor,
            self.confidence_internalizing,
            self.confidence_externalizing,
            self.confidence_attention,
        ]
        for conf in confidences:
            if conf is not None:
                if not isinstance(conf, (int, float)) or np.isnan(conf):
                    return False
                if not (0 <= conf <= 1):
                    return False

        return True


class SubmissionValidator:
    """Validates submission files against challenge schema."""

    # Expected column schemas
    CROSS_TASK_SCHEMA = {
        "required": ["subject_id", "session_id", "task", "prediction"],
        "optional": ["confidence"],
        "types": {
            "subject_id": str,
            "session_id": str,
            "task": str,
            "prediction": float,
            "confidence": float,
        },
    }

    PSYCH_SCHEMA = {
        "required": [
            "subject_id",
            "p_factor",
            "internalizing",
            "externalizing",
            "attention",
        ],
        "optional": [
            "confidence_p_factor",
            "confidence_internalizing",
            "confidence_externalizing",
            "confidence_attention",
        ],
        "types": {
            "subject_id": str,
            "p_factor": float,
            "internalizing": float,
            "externalizing": float,
            "attention": float,
            "confidence_p_factor": float,
            "confidence_internalizing": float,
            "confidence_externalizing": float,
            "confidence_attention": float,
        },
    }

    def __init__(self):
        """Initialize validator."""
        self.validation_errors = []
        self.validation_warnings = []

    def validate_cross_task_csv(self, csv_path: Path) -> bool:
        """Validate cross-task submission CSV."""
        self.validation_errors = []
        self.validation_warnings = []

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            self.validation_errors.append(f"Failed to read CSV: {e}")
            return False

        # Check required columns
        missing_cols = set(self.CROSS_TASK_SCHEMA["required"]) - set(df.columns)
        if missing_cols:
            self.validation_errors.append(f"Missing required columns: {missing_cols}")

        # Check for duplicate entries
        duplicate_mask = df.duplicated(subset=["subject_id", "session_id", "task"])
        if duplicate_mask.any():
            duplicates = df[duplicate_mask]
            self.validation_errors.append(f"Found {len(duplicates)} duplicate entries")

        # Validate each row
        for idx, row in df.iterrows():
            try:
                pred = CrossTaskPrediction(
                    subject_id=str(row["subject_id"]),
                    session_id=str(row["session_id"]),
                    task=str(row["task"]),
                    prediction=float(row["prediction"]),
                    confidence=(
                        float(row["confidence"])
                        if "confidence" in row and pd.notna(row["confidence"])
                        else None
                    ),
                )

                if not pred.validate():
                    self.validation_errors.append(f"Invalid prediction at row {idx}")

            except Exception as e:
                self.validation_errors.append(f"Error validating row {idx}: {e}")

        # Check prediction ranges (warnings)
        if "prediction" in df.columns:
            pred_min, pred_max = df["prediction"].min(), df["prediction"].max()
            if pred_min < -5 or pred_max > 5:
                self.validation_warnings.append(
                    f"Prediction range [{pred_min:.2f}, {pred_max:.2f}] seems unusual"
                )

        return len(self.validation_errors) == 0

    def validate_psychopathology_csv(self, csv_path: Path) -> bool:
        """Validate psychopathology submission CSV."""
        self.validation_errors = []
        self.validation_warnings = []

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            self.validation_errors.append(f"Failed to read CSV: {e}")
            return False

        # Check required columns
        missing_cols = set(self.PSYCH_SCHEMA["required"]) - set(df.columns)
        if missing_cols:
            self.validation_errors.append(f"Missing required columns: {missing_cols}")

        # Check for duplicate subject IDs
        if df["subject_id"].duplicated().any():
            duplicates = df[df["subject_id"].duplicated()]
            self.validation_errors.append(
                f"Found duplicate subject IDs: {duplicates['subject_id'].tolist()}"
            )

        # Validate each row
        for idx, row in df.iterrows():
            try:
                pred = PsychopathologyPrediction(
                    subject_id=str(row["subject_id"]),
                    p_factor=float(row["p_factor"]),
                    internalizing=float(row["internalizing"]),
                    externalizing=float(row["externalizing"]),
                    attention=float(row["attention"]),
                    confidence_p_factor=(
                        float(row["confidence_p_factor"])
                        if "confidence_p_factor" in row
                        and pd.notna(row["confidence_p_factor"])
                        else None
                    ),
                    confidence_internalizing=(
                        float(row["confidence_internalizing"])
                        if "confidence_internalizing" in row
                        and pd.notna(row["confidence_internalizing"])
                        else None
                    ),
                    confidence_externalizing=(
                        float(row["confidence_externalizing"])
                        if "confidence_externalizing" in row
                        and pd.notna(row["confidence_externalizing"])
                        else None
                    ),
                    confidence_attention=(
                        float(row["confidence_attention"])
                        if "confidence_attention" in row
                        and pd.notna(row["confidence_attention"])
                        else None
                    ),
                )

                if not pred.validate():
                    self.validation_errors.append(f"Invalid prediction at row {idx}")

            except Exception as e:
                self.validation_errors.append(f"Error validating row {idx}: {e}")

        # Check score ranges (warnings)
        score_cols = ["p_factor", "internalizing", "externalizing", "attention"]
        for col in score_cols:
            if col in df.columns:
                col_min, col_max = df[col].min(), df[col].max()
                # CBCL scores typically range roughly -3 to 3 (standardized)
                if col_min < -5 or col_max > 5:
                    self.validation_warnings.append(
                        f"{col} range [{col_min:.2f}, {col_max:.2f}] seems unusual for CBCL scores"
                    )

        return len(self.validation_errors) == 0

    def get_validation_report(self) -> Dict[str, List[str]]:
        """Get validation report."""
        return {"errors": self.validation_errors, "warnings": self.validation_warnings}


class SubmissionPackager:
    """Packages model predictions into challenge submission format."""

    def __init__(self, output_dir: Path, team_name: str = "team_unknown"):
        """Initialize submission packager."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.team_name = team_name
        self.validator = SubmissionValidator()

        # Generate submission ID
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.submission_id = f"{team_name}_{timestamp}"

        logger.info(
            f"Initialized submission packager for {team_name}, ID: {self.submission_id}"
        )

    def export_cross_task_predictions(
        self, predictions: List[CrossTaskPrediction], filename: Optional[str] = None
    ) -> Path:
        """Export cross-task predictions to CSV."""
        if filename is None:
            filename = "cross_task_submission.csv"

        csv_path = self.output_dir / filename

        logger.info(
            f"Exporting {len(predictions)} cross-task predictions to {csv_path}"
        )

        # Validate all predictions first
        invalid_predictions = []
        for i, pred in enumerate(predictions):
            if not pred.validate():
                invalid_predictions.append(i)

        if invalid_predictions:
            raise ValueError(f"Invalid predictions at indices: {invalid_predictions}")

        # Write CSV
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = [
                "subject_id",
                "session_id",
                "task",
                "prediction",
                "confidence",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for pred in predictions:
                row = {
                    "subject_id": pred.subject_id,
                    "session_id": pred.session_id,
                    "task": pred.task,
                    "prediction": pred.prediction,
                    "confidence": (
                        pred.confidence if pred.confidence is not None else ""
                    ),
                }
                writer.writerow(row)

        # Validate the written CSV
        if not self.validator.validate_cross_task_csv(csv_path):
            report = self.validator.get_validation_report()
            raise ValueError(f"Generated CSV failed validation: {report['errors']}")

        logger.info(f"Successfully exported cross-task predictions to {csv_path}")
        return csv_path

    def export_psychopathology_predictions(
        self,
        predictions: List[PsychopathologyPrediction],
        filename: Optional[str] = None,
    ) -> Path:
        """Export psychopathology predictions to CSV."""
        if filename is None:
            filename = "psychopathology_submission.csv"

        csv_path = self.output_dir / filename

        logger.info(
            f"Exporting {len(predictions)} psychopathology predictions to {csv_path}"
        )

        # Validate all predictions first
        invalid_predictions = []
        for i, pred in enumerate(predictions):
            if not pred.validate():
                invalid_predictions.append(i)

        if invalid_predictions:
            raise ValueError(f"Invalid predictions at indices: {invalid_predictions}")

        # Write CSV
        with open(csv_path, "w", newline="") as csvfile:
            fieldnames = [
                "subject_id",
                "p_factor",
                "internalizing",
                "externalizing",
                "attention",
                "confidence_p_factor",
                "confidence_internalizing",
                "confidence_externalizing",
                "confidence_attention",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for pred in predictions:
                row = {
                    "subject_id": pred.subject_id,
                    "p_factor": pred.p_factor,
                    "internalizing": pred.internalizing,
                    "externalizing": pred.externalizing,
                    "attention": pred.attention,
                    "confidence_p_factor": (
                        pred.confidence_p_factor
                        if pred.confidence_p_factor is not None
                        else ""
                    ),
                    "confidence_internalizing": (
                        pred.confidence_internalizing
                        if pred.confidence_internalizing is not None
                        else ""
                    ),
                    "confidence_externalizing": (
                        pred.confidence_externalizing
                        if pred.confidence_externalizing is not None
                        else ""
                    ),
                    "confidence_attention": (
                        pred.confidence_attention
                        if pred.confidence_attention is not None
                        else ""
                    ),
                }
                writer.writerow(row)

        # Validate the written CSV
        if not self.validator.validate_psychopathology_csv(csv_path):
            report = self.validator.get_validation_report()
            raise ValueError(f"Generated CSV failed validation: {report['errors']}")

        logger.info(f"Successfully exported psychopathology predictions to {csv_path}")
        return csv_path

    def create_submission_manifest(
        self,
        metadata: SubmissionMetadata,
        files: List[Path],
        filename: Optional[str] = None,
    ) -> Path:
        """Create submission manifest JSON."""
        if filename is None:
            filename = "submission_manifest.json"

        manifest_path = self.output_dir / filename

        # Calculate file hashes
        file_info = []
        for file_path in files:
            if file_path.exists():
                with open(file_path, "rb") as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()

                file_info.append(
                    {
                        "filename": file_path.name,
                        "size_bytes": file_path.stat().st_size,
                        "sha256": file_hash,
                        "created": datetime.fromtimestamp(
                            file_path.stat().st_ctime, timezone.utc
                        ).isoformat(),
                    }
                )

        # Create manifest
        manifest = {
            "metadata": metadata.to_dict(),
            "files": file_info,
            "submission_created": datetime.now(timezone.utc).isoformat(),
            "validation_status": "pending",
        }

        # Write manifest
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(f"Created submission manifest: {manifest_path}")
        return manifest_path

    def create_submission_archive(
        self, files: List[Path], archive_name: Optional[str] = None
    ) -> Path:
        """Create final submission archive."""
        if archive_name is None:
            archive_name = f"{self.submission_id}_submission.zip"

        archive_path = self.output_dir / archive_name

        logger.info(f"Creating submission archive: {archive_path}")

        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for file_path in files:
                if file_path.exists():
                    zipf.write(file_path, file_path.name)
                    logger.debug(f"Added {file_path.name} to archive")

        logger.info(
            f"Created submission archive: {archive_path} ({archive_path.stat().st_size} bytes)"
        )
        return archive_path

    def validate_submission_files(self, csv_files: List[Path]) -> Dict[str, Any]:
        """Validate all submission CSV files."""
        validation_results = {}

        for csv_file in csv_files:
            filename = csv_file.name

            if "cross_task" in filename.lower():
                is_valid = self.validator.validate_cross_task_csv(csv_file)
                validation_results[filename] = {
                    "valid": is_valid,
                    "type": "cross_task",
                    "report": self.validator.get_validation_report(),
                }
            elif "psychopathology" in filename.lower() or "psych" in filename.lower():
                is_valid = self.validator.validate_psychopathology_csv(csv_file)
                validation_results[filename] = {
                    "valid": is_valid,
                    "type": "psychopathology",
                    "report": self.validator.get_validation_report(),
                }
            else:
                validation_results[filename] = {
                    "valid": False,
                    "type": "unknown",
                    "report": {"errors": ["Unknown file type"], "warnings": []},
                }

        return validation_results

    def package_full_submission(
        self,
        cross_task_predictions: Optional[List[CrossTaskPrediction]] = None,
        psychopathology_predictions: Optional[List[PsychopathologyPrediction]] = None,
        metadata: Optional[SubmissionMetadata] = None,
        additional_files: Optional[List[Path]] = None,
    ) -> Tuple[Path, Dict[str, Any]]:
        """Package complete submission with all components."""
        logger.info("Packaging full submission...")

        if not cross_task_predictions and not psychopathology_predictions:
            raise ValueError("At least one prediction type must be provided")

        # Determine challenge track
        if cross_task_predictions and psychopathology_predictions:
            track = "both"
        elif cross_task_predictions:
            track = "cross_task"
        else:
            track = "psychopathology"

        # Create default metadata if not provided
        if metadata is None:
            metadata = SubmissionMetadata(
                team_name=self.team_name,
                submission_id=self.submission_id,
                timestamp=datetime.now(timezone.utc).isoformat(),
                challenge_track=track,
                model_description="EEG2025 submission",
                training_duration_hours=0.0,
                num_parameters=0,
                cross_validation_folds=5,
            )

        # Export prediction files
        exported_files = []

        if cross_task_predictions:
            ct_file = self.export_cross_task_predictions(cross_task_predictions)
            exported_files.append(ct_file)

        if psychopathology_predictions:
            psych_file = self.export_psychopathology_predictions(
                psychopathology_predictions
            )
            exported_files.append(psych_file)

        # Add additional files
        if additional_files:
            for file_path in additional_files:
                if file_path.exists():
                    exported_files.append(file_path)

        # Validate all files
        validation_results = self.validate_submission_files(exported_files)

        # Create manifest
        manifest_file = self.create_submission_manifest(metadata, exported_files)
        exported_files.append(manifest_file)

        # Create archive
        archive_file = self.create_submission_archive(exported_files)

        logger.info(f"Full submission packaged: {archive_file}")

        return archive_file, validation_results


def create_sample_predictions(
    num_subjects: int = 100,
) -> Tuple[List[CrossTaskPrediction], List[PsychopathologyPrediction]]:
    """Create sample predictions for testing."""
    np.random.seed(42)  # For reproducible samples

    # Generate sample cross-task predictions
    cross_task_predictions = []
    tasks = ["sternberg", "n_back", "rest", "flanker", "go_nogo"]

    for i in range(num_subjects):
        subject_id = f"sub-{i+1:03d}"

        for task in tasks:
            # Simulate some sessions having multiple recordings
            num_sessions = np.random.choice([1, 2], p=[0.7, 0.3])

            for session in range(num_sessions):
                session_id = f"ses-{session+1}"

                # Generate realistic predictions (roughly normal around 0)
                prediction = np.random.normal(0, 1.5)
                confidence = np.random.uniform(0.3, 0.95)

                cross_task_predictions.append(
                    CrossTaskPrediction(
                        subject_id=subject_id,
                        session_id=session_id,
                        task=task,
                        prediction=prediction,
                        confidence=confidence,
                    )
                )

    # Generate sample psychopathology predictions
    psych_predictions = []

    for i in range(num_subjects):
        subject_id = f"sub-{i+1:03d}"

        # Generate correlated CBCL scores (p_factor influences others)
        p_factor = np.random.normal(0, 1)
        internalizing = p_factor * 0.6 + np.random.normal(0, 0.8)
        externalizing = p_factor * 0.5 + np.random.normal(0, 0.9)
        attention = p_factor * 0.4 + np.random.normal(0, 0.9)

        # Generate confidences
        conf_p = np.random.uniform(0.4, 0.9)
        conf_int = np.random.uniform(0.3, 0.85)
        conf_ext = np.random.uniform(0.3, 0.85)
        conf_att = np.random.uniform(0.3, 0.85)

        psych_predictions.append(
            PsychopathologyPrediction(
                subject_id=subject_id,
                p_factor=p_factor,
                internalizing=internalizing,
                externalizing=externalizing,
                attention=attention,
                confidence_p_factor=conf_p,
                confidence_internalizing=conf_int,
                confidence_externalizing=conf_ext,
                confidence_attention=conf_att,
            )
        )

    return cross_task_predictions, psych_predictions


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    ct_preds, psych_preds = create_sample_predictions(50)

    # Create submission packager
    packager = SubmissionPackager(
        output_dir=Path("test_submission"), team_name="example_team"
    )

    # Package submission
    metadata = SubmissionMetadata(
        team_name="example_team",
        submission_id=packager.submission_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        challenge_track="both",
        model_description="Example EEG model with DANN and SSL",
        training_duration_hours=12.5,
        num_parameters=2_500_000,
        cross_validation_folds=5,
        best_validation_score=0.73,
        notes="Used 3-layer CNN with domain adversarial training",
    )

    archive_path, validation_results = packager.package_full_submission(
        cross_task_predictions=ct_preds,
        psychopathology_predictions=psych_preds,
        metadata=metadata,
    )

    print(f"Created submission archive: {archive_path}")
    print("\nValidation results:")
    for filename, result in validation_results.items():
        print(f"  {filename}: {'✓' if result['valid'] else '✗'}")
        if result["report"]["errors"]:
            print(f"    Errors: {result['report']['errors']}")
        if result["report"]["warnings"]:
            print(f"    Warnings: {result['report']['warnings']}")
