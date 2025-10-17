#!/usr/bin/env python3
"""
Competition Evaluation and Submission Script
============================================

Generates predictions for both challenges and creates submission files.
"""

import argparse
import logging
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from datetime import datetime

from src.training.challenge1_trainer import Challenge1Model, Challenge1Config
from src.training.challenge2_trainer import Challenge2Model, Challenge2Config
from src.dataio.hbn_dataset import create_hbn_datasets
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Competition Evaluation & Submission")

    # Challenge selection
    parser.add_argument("--challenge", type=str, required=True,
                       choices=["challenge1", "challenge2", "both"],
                       help="Which challenge to evaluate")

    # Model paths
    parser.add_argument("--challenge1_model", type=str, default=None,
                       help="Path to Challenge 1 model checkpoint")
    parser.add_argument("--challenge2_model", type=str, default=None,
                       help="Path to Challenge 2 model checkpoint")

    # Data configuration
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to HBN BIDS dataset")
    parser.add_argument("--window_length", type=float, default=2.0,
                       help="Window length in seconds")
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Window overlap ratio")

    # Evaluation configuration
    parser.add_argument("--split", type=str, default="test",
                       choices=["train", "val", "test"],
                       help="Dataset split to evaluate")
    parser.add_argument("--batch_size", type=int, default=64,
                       help="Batch size for evaluation")

    # Ensemble configuration
    parser.add_argument("--use_ensemble", action="store_true",
                       help="Use ensemble of multiple models")
    parser.add_argument("--ensemble_models", type=str, default=None,
                       help="Comma-separated paths to ensemble models")
    parser.add_argument("--ensemble_weights", type=str, default=None,
                       help="Comma-separated ensemble weights")

    # Output configuration
    parser.add_argument("--output_dir", type=str, default="submissions",
                       help="Output directory for submissions")
    parser.add_argument("--submission_name", type=str, default=None,
                       help="Name for submission files")

    # Hardware
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda, cpu, or auto)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")

    return parser.parse_args()


def load_model_and_config(checkpoint_path, model_class, config_class, device):
    """Load model and configuration from checkpoint."""

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load configuration
    config_path = checkpoint_path.parent / "config.yaml"
    if config_path.exists():
        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
        config = config_class(**config_dict)
    else:
        # Try to extract config from checkpoint
        if "config" in checkpoint:
            config = checkpoint["config"]
        else:
            raise FileNotFoundError(f"Configuration not found for {checkpoint_path}")

    # Create model
    model = model_class(config, n_channels=128)

    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model, config


def evaluate_challenge1(model, data_loader, device, use_temperature_scaling=True):
    """Evaluate Challenge 1 model and generate predictions."""

    predictions = {
        "subject_id": [],
        "session_id": [],
        "trial_id": [],
        "rt_prediction": [],
        "success_prediction": [],
        "confidence": []
    }

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(device)

            # Forward pass
            outputs = model(batch)

            # Extract predictions
            rt_pred = outputs["rt_predictions"].cpu().numpy()
            success_logits = outputs["success_logits"].cpu().numpy()

            # Apply temperature scaling if available
            if use_temperature_scaling and hasattr(model, "temperature"):
                success_logits = success_logits / model.temperature.cpu().numpy()

            # Convert to probabilities
            success_probs = torch.softmax(torch.tensor(success_logits), dim=-1).numpy()
            success_pred = np.argmax(success_probs, axis=-1)
            confidence = np.max(success_probs, axis=-1)

            # Store predictions
            batch_size = rt_pred.shape[0]
            for i in range(batch_size):
                predictions["subject_id"].append(batch["subject_id"][i])
                predictions["session_id"].append(batch["session_id"][i])
                predictions["trial_id"].append(batch["trial_id"][i])
                predictions["rt_prediction"].append(rt_pred[i])
                predictions["success_prediction"].append(success_pred[i])
                predictions["confidence"].append(confidence[i])

    return pd.DataFrame(predictions)


def evaluate_challenge2(model, data_loader, device):
    """Evaluate Challenge 2 model and generate predictions."""

    predictions = {
        "subject_id": [],
        "session_id": []
    }

    # Add columns for each target factor
    target_factors = model.config.target_factors
    for factor in target_factors:
        predictions[f"{factor}_prediction"] = []
        predictions[f"{factor}_uncertainty"] = []

    model.eval()
    with torch.no_grad():
        for batch in data_loader:
            # Move batch to device
            for key, value in batch.items():
                if torch.is_tensor(value):
                    batch[key] = value.to(device)

            # Forward pass
            outputs = model(batch)

            # Extract predictions
            factor_predictions = outputs["factor_predictions"].cpu().numpy()  # [batch, n_factors]
            uncertainties = outputs.get("uncertainties", None)

            if uncertainties is not None:
                uncertainties = uncertainties.cpu().numpy()
            else:
                uncertainties = np.zeros_like(factor_predictions)

            # Store predictions
            batch_size = factor_predictions.shape[0]
            for i in range(batch_size):
                predictions["subject_id"].append(batch["subject_id"][i])
                predictions["session_id"].append(batch["session_id"][i])

                for j, factor in enumerate(target_factors):
                    predictions[f"{factor}_prediction"].append(factor_predictions[i, j])
                    predictions[f"{factor}_uncertainty"].append(uncertainties[i, j])

    return pd.DataFrame(predictions)


def ensemble_predictions(predictions_list, weights=None):
    """Ensemble multiple prediction DataFrames."""

    if weights is None:
        weights = [1.0 / len(predictions_list)] * len(predictions_list)

    if len(weights) != len(predictions_list):
        raise ValueError("Number of weights must match number of prediction sets")

    # Initialize ensemble with first prediction set
    ensemble = predictions_list[0].copy()

    # Identify prediction columns (exclude metadata)
    meta_columns = ["subject_id", "session_id", "trial_id"]
    pred_columns = [col for col in ensemble.columns if col not in meta_columns]

    # Weight the first predictions
    for col in pred_columns:
        if col in ensemble.columns:
            ensemble[col] *= weights[0]

    # Add weighted predictions from other models
    for i, (preds, weight) in enumerate(zip(predictions_list[1:], weights[1:]), 1):
        for col in pred_columns:
            if col in preds.columns:
                ensemble[col] += weight * preds[col]

    return ensemble


def create_submission_files(predictions, challenge, output_dir):
    """Create competition submission files."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if challenge == "challenge1":
        # Challenge 1: Response time and success predictions
        submission = predictions[["subject_id", "session_id", "trial_id",
                                "rt_prediction", "success_prediction"]].copy()

        # Ensure proper formatting
        submission["rt_prediction"] = submission["rt_prediction"].round(4)
        submission["success_prediction"] = submission["success_prediction"].astype(int)

        submission_path = output_dir / "challenge1_submission.csv"
        submission.to_csv(submission_path, index=False)

        logger.info(f"Challenge 1 submission saved to {submission_path}")
        logger.info(f"  {len(submission)} predictions")
        logger.info(f"  RT range: {submission['rt_prediction'].min():.3f} - {submission['rt_prediction'].max():.3f}")
        logger.info(f"  Success rate: {submission['success_prediction'].mean():.3f}")

    elif challenge == "challenge2":
        # Challenge 2: Psychopathology factor predictions
        factor_columns = [col for col in predictions.columns if col.endswith("_prediction")]
        meta_columns = ["subject_id", "session_id"]

        submission = predictions[meta_columns + factor_columns].copy()

        # Round predictions
        for col in factor_columns:
            submission[col] = submission[col].round(4)

        submission_path = output_dir / "challenge2_submission.csv"
        submission.to_csv(submission_path, index=False)

        logger.info(f"Challenge 2 submission saved to {submission_path}")
        logger.info(f"  {len(submission)} predictions")
        logger.info(f"  Factors: {len(factor_columns)}")

        # Log factor statistics
        for col in factor_columns:
            factor_name = col.replace("_prediction", "")
            mean_val = submission[col].mean()
            std_val = submission[col].std()
            logger.info(f"  {factor_name}: {mean_val:.3f} ± {std_val:.3f}")

    return submission_path


def main():
    """Main evaluation function."""

    # Parse arguments
    args = parse_args()

    # Setup logging
    if args.submission_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.submission_name = f"submission_{timestamp}"

    output_dir = Path(args.output_dir) / args.submission_name
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        level=logging.INFO,
        log_file=output_dir / "evaluation.log"
    )

    logger.info(f"Starting evaluation: {args.submission_name}")
    logger.info(f"Challenge: {args.challenge}")

    # Device configuration
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Evaluate challenges
    if args.challenge in ["challenge1", "both"]:
        logger.info("Evaluating Challenge 1...")

        if args.challenge1_model is None:
            raise ValueError("Challenge 1 model path required")

        # Load model
        if args.use_ensemble and args.ensemble_models:
            # Ensemble evaluation
            model_paths = args.ensemble_models.split(",")
            weights = None
            if args.ensemble_weights:
                weights = [float(w) for w in args.ensemble_weights.split(",")]

            predictions_list = []
            for model_path in model_paths:
                model, config = load_model_and_config(
                    model_path.strip(), Challenge1Model, Challenge1Config, device
                )

                # Create data loader
                datasets = create_hbn_datasets(
                    bids_root=args.data_root,
                    window_length=args.window_length,
                    overlap=args.overlap,
                    task_type="cross_task",
                    use_official_splits=True
                )

                from torch.utils.data import DataLoader
                data_loader = DataLoader(
                    datasets[args.split],
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=args.num_workers,
                    pin_memory=True
                )

                # Generate predictions
                preds = evaluate_challenge1(model, data_loader, device)
                predictions_list.append(preds)

            # Ensemble predictions
            challenge1_predictions = ensemble_predictions(predictions_list, weights)

        else:
            # Single model evaluation
            model, config = load_model_and_config(
                args.challenge1_model, Challenge1Model, Challenge1Config, device
            )

            # Create data loader
            datasets = create_hbn_datasets(
                bids_root=args.data_root,
                window_length=args.window_length,
                overlap=args.overlap,
                task_type="cross_task",
                use_official_splits=True
            )

            from torch.utils.data import DataLoader
            data_loader = DataLoader(
                datasets[args.split],
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )

            # Generate predictions
            challenge1_predictions = evaluate_challenge1(model, data_loader, device)

        # Create submission
        create_submission_files(challenge1_predictions, "challenge1", output_dir)

    if args.challenge in ["challenge2", "both"]:
        logger.info("Evaluating Challenge 2...")

        if args.challenge2_model is None:
            raise ValueError("Challenge 2 model path required")

        # Single model evaluation (ensemble can be added similarly)
        model, config = load_model_and_config(
            args.challenge2_model, Challenge2Model, Challenge2Config, device
        )

        # Create data loader
        datasets = create_hbn_datasets(
            bids_root=args.data_root,
            window_length=args.window_length,
            overlap=args.overlap,
            task_type="psychopathology",
            use_official_splits=True
        )

        from torch.utils.data import DataLoader
        data_loader = DataLoader(
            datasets[args.split],
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )

        # Generate predictions
        challenge2_predictions = evaluate_challenge2(model, data_loader, device)

        # Create submission
        create_submission_files(challenge2_predictions, "challenge2", output_dir)

    logger.info(f"Evaluation completed. Submissions saved to {output_dir}")


if __name__ == "__main__":
    main()
