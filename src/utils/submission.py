"""
Submission utilities for EEG Foundation Challenge 2025.

This module provides utilities for creating and validating competition submissions
according to the official format requirements.
"""

import hashlib
import json
import logging
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SubmissionFormatter:
    """
    Formatter for creating competition submissions.
    """

    def __init__(self, team_name: str, submission_name: str):
        """
        Initialize submission formatter.

        Args:
            team_name: Name of the team
            submission_name: Name of the submission
        """
        self.team_name = team_name
        self.submission_name = submission_name
        self.required_columns = {
            "cross_task": ["subject_id", "session_id", "task_name", "prediction"],
            "psychopathology": [
                "subject_id",
                "session_id",
                "prediction_binary",
                "prediction_score",
            ],
        }

    def format_cross_task_predictions(
        self,
        predictions: np.ndarray,
        subject_ids: List[str],
        session_ids: List[str],
        task_names: List[str],
        output_path: Union[str, Path],
    ) -> pd.DataFrame:
        """
        Format cross-task predictions for submission.

        Args:
            predictions: Prediction array [n_samples, n_classes] or [n_samples]
            subject_ids: List of subject IDs
            session_ids: List of session IDs
            task_names: List of task names
            output_path: Path to save the CSV file

        Returns:
            Formatted DataFrame
        """
        # Ensure predictions are 1D (class predictions)
        if predictions.ndim > 1:
            predictions = np.argmax(predictions, axis=1)

        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {
                "subject_id": subject_ids,
                "session_id": session_ids,
                "task_name": task_names,
                "prediction": predictions,
            }
        )

        # Validate format
        self._validate_submission(submission_df, "cross_task")

        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        logger.info(f"Saved cross-task submission to {output_path}")

        return submission_df

    def format_psychopathology_predictions(
        self,
        binary_predictions: np.ndarray,
        prediction_scores: np.ndarray,
        subject_ids: List[str],
        session_ids: List[str],
        output_path: Union[str, Path],
    ) -> pd.DataFrame:
        """
        Format psychopathology predictions for submission.

        Args:
            binary_predictions: Binary predictions [n_samples]
            prediction_scores: Prediction scores [n_samples]
            subject_ids: List of subject IDs
            session_ids: List of session IDs
            output_path: Path to save the CSV file

        Returns:
            Formatted DataFrame
        """
        # Create submission DataFrame
        submission_df = pd.DataFrame(
            {
                "subject_id": subject_ids,
                "session_id": session_ids,
                "prediction_binary": binary_predictions.astype(int),
                "prediction_score": prediction_scores,
            }
        )

        # Validate format
        self._validate_submission(submission_df, "psychopathology")

        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        logger.info(f"Saved psychopathology submission to {output_path}")

        return submission_df

    def _validate_submission(self, df: pd.DataFrame, task_type: str):
        """
        Validate submission format.

        Args:
            df: Submission DataFrame
            task_type: Type of task ('cross_task' or 'psychopathology')
        """
        required_cols = self.required_columns[task_type]

        # Check required columns
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns for {task_type}: {missing_cols}"
            )

        # Check for missing values
        for col in required_cols:
            if df[col].isna().any():
                raise ValueError(f"Column {col} contains missing values")

        # Task-specific validations
        if task_type == "cross_task":
            # Check prediction values are valid integers
            if not df["prediction"].dtype.kind in "ui":
                logger.warning("Cross-task predictions should be integers")

            # Check prediction range (assuming 0-10 tasks)
            if df["prediction"].min() < 0 or df["prediction"].max() > 10:
                logger.warning("Cross-task predictions outside expected range [0, 10]")

        elif task_type == "psychopathology":
            # Check binary predictions are 0 or 1
            if not set(df["prediction_binary"].unique()).issubset({0, 1}):
                raise ValueError("Binary predictions must be 0 or 1")

            # Check scores are in reasonable range
            if df["prediction_score"].min() < 0 or df["prediction_score"].max() > 1:
                logger.warning("Prediction scores outside [0, 1] range")

        logger.info(f"Submission validation passed for {task_type}")

    def create_submission_package(
        self,
        predictions_files: Dict[str, Path],
        model_description: str,
        output_dir: Union[str, Path],
        include_code: bool = False,
        code_dir: Optional[Path] = None,
    ) -> Path:
        """
        Create a complete submission package.

        Args:
            predictions_files: Dictionary mapping task names to prediction CSV files
            model_description: Description of the model and approach
            output_dir: Directory to save the submission package
            include_code: Whether to include source code
            code_dir: Directory containing source code (if include_code=True)

        Returns:
            Path to the created submission ZIP file
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create submission metadata
        metadata = {
            "team_name": self.team_name,
            "submission_name": self.submission_name,
            "model_description": model_description,
            "predictions_files": {k: str(v) for k, v in predictions_files.items()},
            "submission_time": pd.Timestamp.now().isoformat(),
        }

        # Save metadata
        metadata_file = output_dir / "submission_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Create submission ZIP
        submission_zip = output_dir / f"{self.team_name}_{self.submission_name}.zip"

        with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add metadata
            zipf.write(metadata_file, "submission_metadata.json")

            # Add prediction files
            for task_name, pred_file in predictions_files.items():
                zipf.write(pred_file, f"predictions_{task_name}.csv")

            # Add code if requested
            if include_code and code_dir is not None:
                code_dir = Path(code_dir)
                for code_file in code_dir.rglob("*.py"):
                    # Skip __pycache__ and .git directories
                    if "__pycache__" not in str(code_file) and ".git" not in str(
                        code_file
                    ):
                        arcname = f"code/{code_file.relative_to(code_dir)}"
                        zipf.write(code_file, arcname)

        logger.info(f"Created submission package: {submission_zip}")
        return submission_zip

    def validate_submission_package(
        self, submission_path: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Validate a submission package.

        Args:
            submission_path: Path to the submission ZIP file

        Returns:
            Validation results dictionary
        """
        submission_path = Path(submission_path)

        if not submission_path.exists():
            raise FileNotFoundError(f"Submission file not found: {submission_path}")

        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "file_checksums": {},
        }

        try:
            with zipfile.ZipFile(submission_path, "r") as zipf:
                file_list = zipf.namelist()

                # Check for required files
                if "submission_metadata.json" not in file_list:
                    validation_results["errors"].append(
                        "Missing submission_metadata.json"
                    )
                    validation_results["valid"] = False

                # Check for prediction files
                prediction_files = [
                    f for f in file_list if f.startswith("predictions_")
                ]
                if not prediction_files:
                    validation_results["errors"].append("No prediction files found")
                    validation_results["valid"] = False

                # Validate each prediction file
                for pred_file in prediction_files:
                    try:
                        with zipf.open(pred_file) as f:
                            df = pd.read_csv(f)

                            # Extract task type from filename
                            task_type = pred_file.replace("predictions_", "").replace(
                                ".csv", ""
                            )

                            if task_type in self.required_columns:
                                self._validate_submission(df, task_type)
                            else:
                                validation_results["warnings"].append(
                                    f"Unknown task type: {task_type}"
                                )

                    except Exception as e:
                        validation_results["errors"].append(
                            f"Error validating {pred_file}: {str(e)}"
                        )
                        validation_results["valid"] = False

                # Compute file checksums
                for file_name in file_list:
                    with zipf.open(file_name) as f:
                        content = f.read()
                        checksum = hashlib.md5(content).hexdigest()
                        validation_results["file_checksums"][file_name] = checksum

        except Exception as e:
            validation_results["errors"].append(
                f"Error reading submission package: {str(e)}"
            )
            validation_results["valid"] = False

        return validation_results


class PredictionAggregator:
    """
    Utility for aggregating predictions from multiple models or folds.
    """

    def __init__(self, aggregation_method: str = "mean"):
        """
        Initialize prediction aggregator.

        Args:
            aggregation_method: Method for aggregation ('mean', 'median', 'vote')
        """
        self.aggregation_method = aggregation_method

    def aggregate_predictions(
        self, predictions_list: List[np.ndarray], weights: Optional[List[float]] = None
    ) -> np.ndarray:
        """
        Aggregate predictions from multiple models.

        Args:
            predictions_list: List of prediction arrays
            weights: Optional weights for each model

        Returns:
            Aggregated predictions
        """
        if not predictions_list:
            raise ValueError("No predictions provided")

        # Stack predictions
        predictions_array = np.stack(predictions_list, axis=0)

        if weights is not None:
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            predictions_array = predictions_array * weights[:, np.newaxis]

        # Aggregate based on method
        if self.aggregation_method == "mean":
            aggregated = np.mean(predictions_array, axis=0)
        elif self.aggregation_method == "median":
            aggregated = np.median(predictions_array, axis=0)
        elif self.aggregation_method == "vote":
            # For classification: majority vote
            if predictions_array.ndim == 2:  # [n_models, n_samples]
                aggregated = np.array(
                    [
                        np.bincount(predictions_array[:, i]).argmax()
                        for i in range(predictions_array.shape[1])
                    ]
                )
            else:  # [n_models, n_samples, n_classes]
                # Average probabilities then take argmax
                mean_probs = np.mean(predictions_array, axis=0)
                aggregated = np.argmax(mean_probs, axis=1)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        return aggregated

    def aggregate_cross_validation_predictions(
        self, cv_predictions: Dict[int, np.ndarray], cv_indices: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        Aggregate cross-validation predictions.

        Args:
            cv_predictions: Dictionary mapping fold to predictions
            cv_indices: Dictionary mapping fold to test indices

        Returns:
            Aggregated predictions for all samples
        """
        # Determine total number of samples
        total_samples = sum(len(indices) for indices in cv_indices.values())

        # Initialize aggregated predictions
        first_pred = list(cv_predictions.values())[0]
        if first_pred.ndim == 1:
            aggregated = np.zeros(total_samples)
        else:
            aggregated = np.zeros((total_samples, first_pred.shape[1]))

        # Fill in predictions for each fold
        for fold, predictions in cv_predictions.items():
            indices = cv_indices[fold]
            aggregated[indices] = predictions

        return aggregated


def create_starter_kit_submission(
    model_predictions: Dict[str, np.ndarray],
    metadata: Dict[str, List],
    output_dir: Union[str, Path],
    team_name: str = "baseline_team",
    submission_name: str = "starter_kit_submission",
) -> Path:
    """
    Create a submission using the starter kit format.

    Args:
        model_predictions: Dictionary with model predictions for each task
        metadata: Dictionary with metadata (subject_ids, session_ids, etc.)
        output_dir: Output directory for submission files
        team_name: Team name
        submission_name: Submission name

    Returns:
        Path to submission package
    """
    formatter = SubmissionFormatter(team_name, submission_name)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prediction_files = {}

    # Create cross-task submission
    if "cross_task" in model_predictions:
        cross_task_file = output_dir / "cross_task_predictions.csv"
        formatter.format_cross_task_predictions(
            predictions=model_predictions["cross_task"],
            subject_ids=metadata["subject_ids"],
            session_ids=metadata["session_ids"],
            task_names=metadata["task_names"],
            output_path=cross_task_file,
        )
        prediction_files["cross_task"] = cross_task_file

    # Create psychopathology submission
    if "psychopathology_binary" in model_predictions:
        psych_file = output_dir / "psychopathology_predictions.csv"
        formatter.format_psychopathology_predictions(
            binary_predictions=model_predictions["psychopathology_binary"],
            prediction_scores=model_predictions.get(
                "psychopathology_scores", model_predictions["psychopathology_binary"]
            ),
            subject_ids=metadata["subject_ids"],
            session_ids=metadata["session_ids"],
            output_path=psych_file,
        )
        prediction_files["psychopathology"] = psych_file

    # Create submission package
    submission_package = formatter.create_submission_package(
        predictions_files=prediction_files,
        model_description="Baseline submission using temporal CNN architecture",
        output_dir=output_dir,
        include_code=False,
    )

    return submission_package
