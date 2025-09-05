#!/usr/bin/env python3
"""
Evaluation script for EEG Foundation Challenge 2025.

This script generates predictions in the official Starter Kit format,
aggregates them according to challenge rules, and computes official metrics.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import argparse

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
import hydra

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from dataio.hbn_dataset import HBNDataset, create_hbn_datasets
from dataio.starter_kit import StarterKitDataLoader, OfficialMetrics, SubmissionValidator
from scripts.train_enhanced import EEGFoundationModel
from utils.submission import create_starter_kit_submission

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EvaluationPipeline:
    """
    Comprehensive evaluation pipeline for the EEG Foundation Challenge.

    Handles prediction generation, aggregation, and official metrics computation
    following Starter Kit specifications.
    """

    def __init__(self, config: DictConfig, checkpoint_path: str):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.task_type = config.task.name

        # Load model
        self.model = self._load_model()
        self.model.eval()

        # Initialize Starter Kit components
        self.starter_kit_loader = StarterKitDataLoader(config.data.bids_root)
        self.submission_validator = SubmissionValidator()

        # Official metrics
        self.official_metrics = OfficialMetrics()

    def _load_model(self) -> EEGFoundationModel:
        """Load trained model from checkpoint."""
        logger.info(f"Loading model from {self.checkpoint_path}")

        model = EEGFoundationModel.load_from_checkpoint(
            self.checkpoint_path,
            config=self.config,
            strict=False
        )

        return model

    def generate_predictions(self, split: str = "test") -> pd.DataFrame:
        """
        Generate predictions for the specified split.

        Args:
            split: Data split to evaluate ("test", "val", "train")

        Returns:
            DataFrame with predictions in Starter Kit format
        """
        logger.info(f"Generating predictions for {split} split...")

        # Create datasets
        datasets = create_hbn_datasets(
            bids_root=self.config.data.bids_root,
            task_type=self.task_type,
            window_length=self.config.data.windows.get(f"{self.task_type}_len_s", 2.0),
            overlap=self.config.data.overlap,
            sample_rate=self.config.data.sample_rate,
            use_official_splits=True,
            enable_compression_aug=False  # No augmentation during evaluation
        )

        if split not in datasets:
            raise ValueError(f"Split {split} not found in datasets")

        dataset = datasets[split]
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.num_workers,
            pin_memory=True
        )

        # Generate predictions
        all_predictions = []
        all_metadata = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx % 100 == 0:
                    logger.info(f"Processing batch {batch_idx}/{len(dataloader)}")

                # Move to device
                eeg = batch["eeg"].to(self.model.device)

                # Forward pass
                outputs = self.model(eeg)

                # Extract predictions based on task type
                batch_predictions = self._extract_predictions(outputs, batch)
                all_predictions.extend(batch_predictions)

                # Extract metadata
                batch_metadata = self._extract_metadata(batch)
                all_metadata.extend(batch_metadata)

        # Create predictions DataFrame
        predictions_df = pd.DataFrame(all_predictions)
        metadata_df = pd.DataFrame(all_metadata)

        # Combine predictions with metadata
        result_df = pd.concat([metadata_df, predictions_df], axis=1)

        logger.info(f"Generated {len(result_df)} predictions for {split} split")
        return result_df

    def _extract_predictions(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> List[Dict]:
        """Extract predictions from model outputs."""
        batch_size = list(outputs.values())[0].size(0)
        predictions = []

        for i in range(batch_size):
            pred_dict = {}

            if self.task_type == "cross_task":
                # Challenge 1 predictions
                if "response_time" in outputs:
                    pred_dict["response_time"] = float(outputs["response_time"][i].cpu().numpy())

                if "success" in outputs:
                    success_probs = torch.softmax(outputs["success"][i], dim=0)
                    pred_dict["success"] = float(success_probs[1].cpu().numpy())

            elif self.task_type == "psychopathology":
                # Challenge 2 predictions
                cbcl_dims = ["p_factor", "internalizing", "externalizing", "attention"]

                for dim in cbcl_dims:
                    if dim in outputs:
                        pred_dict[dim] = float(outputs[dim][i].cpu().numpy())

                if "binary" in outputs:
                    binary_probs = torch.softmax(outputs["binary"][i], dim=0)
                    pred_dict["binary_label"] = float(binary_probs[1].cpu().numpy())

            predictions.append(pred_dict)

        return predictions

    def _extract_metadata(self, batch: Dict[str, torch.Tensor]) -> List[Dict]:
        """Extract metadata from batch."""
        batch_size = batch["eeg"].size(0)
        metadata = []

        for i in range(batch_size):
            meta_dict = {}

            # Standard metadata
            if "subject_id" in batch:
                meta_dict["subject"] = batch["subject_id"][i].item()
            if "session_id" in batch:
                meta_dict["session"] = batch["session_id"][i].item()
            if "window_start" in batch:
                meta_dict["window_start"] = float(batch["window_start"][i].cpu().numpy())
            if "window_end" in batch:
                meta_dict["window_end"] = float(batch["window_end"][i].cpu().numpy())

            # Task-specific metadata
            if self.task_type == "cross_task":
                if "task_name" in batch:
                    meta_dict["task"] = batch["task_name"][i] if isinstance(batch["task_name"], list) else str(batch["task_name"][i])
                if "trial_id" in batch:
                    meta_dict["trial"] = batch["trial_id"][i].item()

            metadata.append(meta_dict)

        return metadata

    def aggregate_predictions(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate predictions according to official challenge rules.

        Args:
            predictions_df: Raw predictions DataFrame

        Returns:
            Aggregated predictions DataFrame
        """
        logger.info("Aggregating predictions according to challenge rules...")

        if self.task_type == "cross_task":
            return self._aggregate_challenge1_predictions(predictions_df)
        elif self.task_type == "psychopathology":
            return self._aggregate_challenge2_predictions(predictions_df)
        else:
            return predictions_df

    def _aggregate_challenge1_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate Challenge 1 predictions per trial."""
        # Group by subject, session, task, and trial
        groupby_cols = ["subject", "session", "task", "trial"]
        available_cols = [col for col in groupby_cols if col in df.columns]

        if not available_cols:
            logger.warning("No grouping columns found for Challenge 1 aggregation")
            return df

        aggregated = df.groupby(available_cols).agg({
            "response_time": "mean",  # Average response time across windows
            "success": "mean",        # Average success probability
        }).reset_index()

        logger.info(f"Aggregated {len(df)} windows to {len(aggregated)} trials")
        return aggregated

    def _aggregate_challenge2_predictions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate Challenge 2 predictions per subject."""
        # Group by subject and session
        groupby_cols = ["subject", "session"]
        available_cols = [col for col in groupby_cols if col in df.columns]

        if not available_cols:
            logger.warning("No grouping columns found for Challenge 2 aggregation")
            return df

        # Aggregate continuous variables
        agg_dict = {}
        cbcl_dims = ["p_factor", "internalizing", "externalizing", "attention"]

        for dim in cbcl_dims:
            if dim in df.columns:
                agg_dict[dim] = "mean"

        if "binary_label" in df.columns:
            agg_dict["binary_label"] = "mean"

        aggregated = df.groupby(available_cols).agg(agg_dict).reset_index()

        logger.info(f"Aggregated {len(df)} windows to {len(aggregated)} subjects")
        return aggregated

    def load_ground_truth(self, split: str = "test") -> pd.DataFrame:
        """Load ground truth labels for the specified split."""
        logger.info(f"Loading ground truth for {split} split...")

        if self.task_type == "cross_task":
            labels_df = self.starter_kit_loader.load_ccd_labels(split)
        elif self.task_type == "psychopathology":
            labels_df = self.starter_kit_loader.load_cbcl_labels(split)
        else:
            raise ValueError(f"Unknown task type: {self.task_type}")

        logger.info(f"Loaded {len(labels_df)} ground truth samples")
        return labels_df

    def compute_official_metrics(self, predictions_df: pd.DataFrame, labels_df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute official challenge metrics.

        Args:
            predictions_df: Aggregated predictions
            labels_df: Ground truth labels

        Returns:
            Dictionary of official metrics
        """
        logger.info("Computing official metrics...")

        # Merge predictions with labels
        if self.task_type == "cross_task":
            merge_cols = ["subject", "session", "task", "trial"]
        elif self.task_type == "psychopathology":
            merge_cols = ["subject", "session"]
        else:
            merge_cols = ["subject"]

        # Only use available columns
        available_merge_cols = [col for col in merge_cols if col in predictions_df.columns and col in labels_df.columns]

        if not available_merge_cols:
            logger.error("No common columns for merging predictions and labels")
            return {}

        merged_df = pd.merge(predictions_df, labels_df, on=available_merge_cols, how="inner")

        if merged_df.empty:
            logger.error("No matching samples found between predictions and labels")
            return {}

        logger.info(f"Merged {len(merged_df)} samples for metric computation")

        # Extract predictions and targets
        predictions = {}
        targets = {}

        if self.task_type == "cross_task":
            if "response_time" in merged_df.columns and "response_time_target" in merged_df.columns:
                predictions["response_time"] = merged_df["response_time"].values
                targets["response_time"] = merged_df["response_time_target"].values

            if "success" in merged_df.columns and "success_target" in merged_df.columns:
                predictions["success"] = merged_df["success"].values
                targets["success"] = merged_df["success_target"].values

        elif self.task_type == "psychopathology":
            cbcl_dims = ["p_factor", "internalizing", "externalizing", "attention"]

            for dim in cbcl_dims:
                if dim in merged_df.columns and f"{dim}_target" in merged_df.columns:
                    predictions[dim] = merged_df[dim].values
                    targets[dim] = merged_df[f"{dim}_target"].values

            if "binary_label" in merged_df.columns and "binary_target" in merged_df.columns:
                predictions["binary_label"] = merged_df["binary_label"].values
                targets["binary_label"] = merged_df["binary_target"].values

        # Compute metrics
        if self.task_type == "cross_task":
            metrics = self.official_metrics.compute_challenge1_metrics(predictions, targets)
        elif self.task_type == "psychopathology":
            metrics = self.official_metrics.compute_challenge2_metrics(predictions, targets)
        else:
            metrics = {}

        return metrics

    def create_submission_files(self, predictions_df: pd.DataFrame, output_dir: str):
        """Create submission files in official format."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Creating submission files in {output_path}")

        # Create submission
        submission_path = create_starter_kit_submission(
            predictions_df=predictions_df,
            task_type=self.task_type,
            output_dir=str(output_path)
        )

        # Validate submission
        try:
            self.submission_validator.validate_submission(submission_path, self.task_type)
            logger.info("✅ Submission validation passed!")
        except Exception as e:
            logger.error(f"❌ Submission validation failed: {e}")

        return submission_path

    def run_full_evaluation(self, splits: List[str] = None, output_dir: str = None) -> Dict[str, Dict]:
        """
        Run complete evaluation pipeline.

        Args:
            splits: List of splits to evaluate
            output_dir: Directory to save results

        Returns:
            Dictionary of results for each split
        """
        if splits is None:
            splits = ["val", "test"]

        if output_dir is None:
            output_dir = f"results/{self.task_type}"

        results = {}

        for split in splits:
            logger.info(f"\n{'='*50}")
            logger.info(f"Evaluating {split} split")
            logger.info(f"{'='*50}")

            try:
                # Generate predictions
                predictions_df = self.generate_predictions(split)

                # Aggregate predictions
                aggregated_df = self.aggregate_predictions(predictions_df)

                # Save raw predictions
                split_output_dir = Path(output_dir) / split
                split_output_dir.mkdir(parents=True, exist_ok=True)

                predictions_path = split_output_dir / "raw_predictions.csv"
                predictions_df.to_csv(predictions_path, index=False)
                logger.info(f"Saved raw predictions to {predictions_path}")

                aggregated_path = split_output_dir / "aggregated_predictions.csv"
                aggregated_df.to_csv(aggregated_path, index=False)
                logger.info(f"Saved aggregated predictions to {aggregated_path}")

                # Create submission files
                submission_path = self.create_submission_files(aggregated_df, str(split_output_dir))

                # Compute metrics if labels are available
                metrics = {}
                try:
                    labels_df = self.load_ground_truth(split)
                    metrics = self.compute_official_metrics(aggregated_df, labels_df)

                    # Save metrics
                    metrics_path = split_output_dir / "metrics.json"
                    import json
                    with open(metrics_path, 'w') as f:
                        json.dump(metrics, f, indent=2)

                    logger.info(f"Official metrics for {split}:")
                    for metric_name, metric_value in metrics.items():
                        logger.info(f"  {metric_name}: {metric_value:.4f}")

                except Exception as e:
                    logger.warning(f"Could not compute metrics for {split}: {e}")

                results[split] = {
                    "predictions_path": str(predictions_path),
                    "aggregated_path": str(aggregated_path),
                    "submission_path": str(submission_path),
                    "metrics": metrics,
                    "num_predictions": len(predictions_df),
                    "num_aggregated": len(aggregated_df)
                }

            except Exception as e:
                logger.error(f"Error evaluating {split}: {e}")
                results[split] = {"error": str(e)}

        return results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate EEG Foundation Challenge model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], help="Splits to evaluate")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")

    args = parser.parse_args()

    # Load config
    config = OmegaConf.load(args.config)

    # Create evaluation pipeline
    evaluator = EvaluationPipeline(config, args.checkpoint)

    # Run evaluation
    results = evaluator.run_full_evaluation(
        splits=args.splits,
        output_dir=args.output_dir
    )

    # Print summary
    print("\n" + "="*70)
    print("EVALUATION SUMMARY")
    print("="*70)

    for split, result in results.items():
        print(f"\n{split.upper()} Split:")
        if "error" in result:
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  ✅ Predictions: {result['num_predictions']} → {result['num_aggregated']} (aggregated)")
            print(f"  📁 Files: {result['submission_path']}")

            if result["metrics"]:
                print(f"  📊 Metrics:")
                for metric_name, metric_value in result["metrics"].items():
                    print(f"    {metric_name}: {metric_value:.4f}")


if __name__ == "__main__":
    main()
