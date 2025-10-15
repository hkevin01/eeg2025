#!/usr/bin/env python3
"""
Training Script for Challenge 2: Psychopathology Factor Prediction
===================================================================

Trains models for CBCL factor prediction with subject invariance.
"""

import argparse
import logging
import torch
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime

from src.training.challenge2_trainer import Challenge2Trainer, Challenge2Model, Challenge2Config
from src.dataio.hbn_dataset import create_hbn_datasets
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Challenge 2 Training")

    # Data configuration
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to HBN BIDS dataset")
    parser.add_argument("--window_length", type=float, default=2.0,
                       help="Window length in seconds")
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Window overlap ratio")

    # Target factors
    parser.add_argument("--target_factors", type=str,
                       default="p_factor,internalizing,externalizing,attention",
                       help="Comma-separated list of CBCL factors to predict")
    parser.add_argument("--use_age_normalization", action="store_true",
                       help="Use age-based normalization")
    parser.add_argument("--use_demographic_features", action="store_true",
                       help="Include demographic features")

    # Model configuration
    parser.add_argument("--backbone_type", type=str, default="transformer",
                       choices=["transformer", "cnn", "hybrid"])
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)

    # Multi-task learning
    parser.add_argument("--tasks", type=str, default="RS,SuS,MW,CCD",
                       help="Comma-separated list of tasks to include")
    parser.add_argument("--task_weights", type=str, default=None,
                       help="Comma-separated task weights (same order as tasks)")

    # Subject invariance
    parser.add_argument("--use_subject_invariance", action="store_true",
                       help="Use subject-level invariance")
    parser.add_argument("--use_irm_penalty", action="store_true",
                       help="Use IRM penalty for invariance")
    parser.add_argument("--irm_weight", type=float, default=0.1)

    # Clinical data processing
    parser.add_argument("--score_normalization", type=str, default="robust",
                       choices=["standard", "robust", "quantile"])
    parser.add_argument("--missing_data_strategy", type=str, default="median",
                       choices=["mean", "median", "zero"])
    parser.add_argument("--outlier_threshold", type=float, default=3.0)

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Loss weights
    parser.add_argument("--correlation_loss_weight", type=float, default=1.0)
    parser.add_argument("--mse_loss_weight", type=float, default=0.5)
    parser.add_argument("--invariance_loss_weight", type=float, default=0.1)

    # Cross-validation
    parser.add_argument("--use_cross_validation", action="store_true",
                       help="Use cross-validation")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--stratify_by_age", action="store_true")

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="runs/challenge2",
                       help="Output directory for logs and checkpoints")
    parser.add_argument("--experiment_name", type=str, default=None,
                       help="Experiment name for logging")
    parser.add_argument("--log_level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    # Hardware
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda, cpu, or auto)")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of data loader workers")

    return parser.parse_args()


def create_config(args) -> Challenge2Config:
    """Create training configuration from arguments."""

    # Parse target factors
    target_factors = [f.strip() for f in args.target_factors.split(",")]

    # Parse tasks
    tasks = [t.strip() for t in args.tasks.split(",")]

    # Parse task weights if provided
    task_weights = None
    if args.task_weights:
        weights = [float(w) for w in args.task_weights.split(",")]
        if len(weights) == len(tasks):
            task_weights = {task: weight for task, weight in zip(tasks, weights)}

    # Create configuration
    config = Challenge2Config(
        # Model architecture
        backbone_type=args.backbone_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,

        # Psychopathology factors
        target_factors=target_factors,
        use_age_normalization=args.use_age_normalization,
        use_demographic_features=args.use_demographic_features,

        # Multi-task learning
        tasks=tasks,
        task_weights=task_weights,

        # Subject invariance
        use_subject_invariance=args.use_subject_invariance,
        use_irm_penalty=args.use_irm_penalty,
        irm_weight=args.irm_weight,

        # Clinical data processing
        score_normalization=args.score_normalization,
        missing_data_strategy=args.missing_data_strategy,
        outlier_threshold=args.outlier_threshold,

        # Loss configuration
        correlation_loss_weight=args.correlation_loss_weight,
        mse_loss_weight=args.mse_loss_weight,
        invariance_loss_weight=args.invariance_loss_weight,

        # Training configuration
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,

        # Cross-validation
        use_cross_validation=args.use_cross_validation,
        cv_folds=args.cv_folds,
        stratify_by_age=args.stratify_by_age
    )

    return config


def create_data_loaders(args):
    """Create data loaders for training and validation."""

    # Create datasets with official splits
    datasets = create_hbn_datasets(
        bids_root=args.data_root,
        window_length=args.window_length,
        overlap=args.overlap,
        task_type="psychopathology",  # Multi-task for psychopathology
        use_official_splits=True
    )

    # Create data loaders
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        datasets["train"],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Test loader for final evaluation
    test_loader = DataLoader(
        datasets["test"],
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    ) if "test" in datasets else None

    return train_loader, val_loader, test_loader


def run_single_fold(config, train_loader, val_loader, output_dir, device, fold_idx=None):
    """Run training for a single fold."""

    fold_suffix = f"_fold_{fold_idx}" if fold_idx is not None else ""
    fold_output_dir = output_dir / f"fold_{fold_idx}" if fold_idx is not None else output_dir
    fold_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training fold {fold_idx if fold_idx is not None else 'single'}")

    # Create model
    model = Challenge2Model(config, n_channels=128)

    # Create trainer
    trainer = Challenge2Trainer(
        config=config,
        model=model,
        device=device,
        log_dir=fold_output_dir
    )

    # Training
    history = trainer.fit(train_loader, val_loader)

    return trainer, history


def run_cross_validation(config, datasets, args, output_dir, device):
    """Run cross-validation training."""

    from sklearn.model_selection import KFold, StratifiedKFold
    from torch.utils.data import Subset, DataLoader

    # Combine train and val for CV
    full_dataset = torch.utils.data.ConcatDataset([datasets["train"], datasets["val"]])

    # Extract stratification variable (age groups)
    if config.stratify_by_age:
        # This would need to be implemented based on dataset structure
        # For now, use simple KFold
        kfold = KFold(n_splits=config.cv_folds, shuffle=True, random_state=42)
        indices = list(range(len(full_dataset)))
        splits = list(kfold.split(indices))
    else:
        kfold = KFold(n_splits=config.cv_folds, shuffle=True, random_state=42)
        indices = list(range(len(full_dataset)))
        splits = list(kfold.split(indices))

    fold_results = []

    for fold_idx, (train_indices, val_indices) in enumerate(splits):
        logger.info(f"Starting fold {fold_idx + 1}/{config.cv_folds}")

        # Create fold datasets
        train_subset = Subset(full_dataset, train_indices)
        val_subset = Subset(full_dataset, val_indices)

        # Create fold data loaders
        fold_train_loader = DataLoader(
            train_subset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True
        )

        fold_val_loader = DataLoader(
            val_subset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False
        )

        # Run fold training
        trainer, history = run_single_fold(
            config, fold_train_loader, fold_val_loader,
            output_dir, device, fold_idx
        )

        fold_results.append({
            "fold": fold_idx,
            "best_metric": trainer.best_metric,
            "history": history
        })

        logger.info(f"Fold {fold_idx + 1} completed. Best metric: {trainer.best_metric:.4f}")

    # Compute CV statistics
    best_metrics = [result["best_metric"] for result in fold_results]
    cv_mean = np.mean(best_metrics)
    cv_std = np.std(best_metrics)

    logger.info(f"Cross-validation results:")
    logger.info(f"  Mean: {cv_mean:.4f} ± {cv_std:.4f}")
    logger.info(f"  Individual folds: {best_metrics}")

    # Save CV results
    cv_results = {
        "mean_metric": cv_mean,
        "std_metric": cv_std,
        "fold_metrics": best_metrics,
        "detailed_results": fold_results
    }

    cv_results_path = output_dir / "cv_results.yaml"
    with open(cv_results_path, "w") as f:
        yaml.dump(cv_results, f, default_flow_style=False)

    return cv_results


def main():
    """Main training function."""

    # Parse arguments
    args = parse_args()

    # Setup logging
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"challenge2_{timestamp}"

    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        level=getattr(logging, args.log_level),
        log_file=output_dir / "training.log"
    )

    logger.info(f"Starting Challenge 2 training: {args.experiment_name}")
    logger.info(f"Output directory: {output_dir}")

    # Device configuration
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Create configuration
    config = create_config(args)

    # Save configuration
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config.__dict__, f, default_flow_style=False)
    logger.info(f"Configuration saved to {config_path}")

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(args)
    datasets = {"train": train_loader.dataset, "val": val_loader.dataset}
    if test_loader:
        datasets["test"] = test_loader.dataset

    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Training
    try:
        if config.use_cross_validation:
            logger.info(f"Running {config.cv_folds}-fold cross-validation...")
            cv_results = run_cross_validation(config, datasets, args, output_dir, device)
            logger.info("Cross-validation completed successfully!")

        else:
            logger.info("Running single training...")
            trainer, history = run_single_fold(
                config, train_loader, val_loader, output_dir, device
            )

            logger.info("Training completed successfully!")
            logger.info(f"Best validation metric: {trainer.best_metric:.4f}")

            # Save training history
            history_path = output_dir / "training_history.yaml"
            with open(history_path, "w") as f:
                yaml.dump(history, f, default_flow_style=False)

        # Final evaluation on test set if available
        if test_loader and not config.use_cross_validation:
            logger.info("Evaluating on test set...")

            # Load best model
            best_checkpoint = output_dir / "best_model.ckpt"
            if best_checkpoint.exists():
                checkpoint = torch.load(best_checkpoint, map_location=device)
                trainer.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded best model for test evaluation")

            # Test evaluation
            test_predictions = []
            test_targets = []

            trainer.model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    # Move batch to device
                    for key, value in batch.items():
                        if torch.is_tensor(value):
                            batch[key] = value.to(device)

                    predictions, targets = trainer.validate_step(batch)
                    test_predictions.append(predictions)
                    test_targets.append(targets)

            # Compute test metrics
            test_metrics = trainer.compute_official_metrics(test_predictions, test_targets)

            logger.info("Test Results:")
            for metric_name, value in test_metrics.items():
                logger.info(f"  {metric_name}: {value:.4f}")

            # Save test results
            test_results_path = output_dir / "test_results.yaml"
            with open(test_results_path, "w") as f:
                yaml.dump(test_metrics, f, default_flow_style=False)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info(f"Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
