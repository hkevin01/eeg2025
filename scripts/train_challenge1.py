#!/usr/bin/env python3
"""
Training Script for Challenge 1: Cross-Task Transfer Learning
=============================================================

Trains enhanced models for SuS → CCD transfer with official metrics.
"""

import argparse
import logging
import torch
import yaml
from pathlib import Path
from datetime import datetime

from src.training.challenge1_trainer import Challenge1Trainer, Challenge1Model, Challenge1Config
from src.dataio.hbn_dataset import create_hbn_datasets
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Challenge 1 Training")

    # Data configuration
    parser.add_argument("--data_root", type=str, required=True,
                       help="Path to HBN BIDS dataset")
    parser.add_argument("--window_length", type=float, default=2.0,
                       help="Window length in seconds")
    parser.add_argument("--overlap", type=float, default=0.5,
                       help="Window overlap ratio")

    # Model configuration
    parser.add_argument("--backbone_type", type=str, default="transformer",
                       choices=["transformer", "cnn", "hybrid"])
    parser.add_argument("--hidden_dim", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)

    # Transfer learning
    parser.add_argument("--ssl_checkpoint", type=str, default=None,
                       help="Path to SSL pretrained checkpoint")
    parser.add_argument("--progressive_unfreezing", action="store_true",
                       help="Use progressive unfreezing")
    parser.add_argument("--unfreeze_schedule", type=str, default="0.25,0.5,0.75,1.0",
                       help="Comma-separated unfreezing schedule")

    # Domain adaptation
    parser.add_argument("--use_subject_adaptation", action="store_true",
                       help="Use subject-level domain adaptation")
    parser.add_argument("--use_site_adaptation", action="store_true",
                       help="Use site-level domain adaptation")
    parser.add_argument("--domain_adaptation_weight", type=float, default=0.1)

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    # Loss weights
    parser.add_argument("--rt_loss_weight", type=float, default=1.0)
    parser.add_argument("--success_loss_weight", type=float, default=1.0)
    parser.add_argument("--domain_loss_weight", type=float, default=0.1)

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="runs/challenge1",
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


def create_config(args) -> Challenge1Config:
    """Create training configuration from arguments."""

    # Parse unfreeze schedule
    unfreeze_schedule = [float(x) for x in args.unfreeze_schedule.split(",")]

    # Create configuration
    config = Challenge1Config(
        # Model architecture
        backbone_type=args.backbone_type,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,

        # Transfer learning
        ssl_checkpoint=args.ssl_checkpoint,
        progressive_unfreezing=args.progressive_unfreezing,
        unfreeze_schedule=unfreeze_schedule,

        # Domain adaptation
        use_subject_adaptation=args.use_subject_adaptation,
        use_site_adaptation=args.use_site_adaptation,
        domain_adaptation_weight=args.domain_adaptation_weight,

        # Loss configuration
        rt_loss_weight=args.rt_loss_weight,
        success_loss_weight=args.success_loss_weight,
        domain_loss_weight=args.domain_loss_weight,

        # Training configuration
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay
    )

    return config


def create_data_loaders(args):
    """Create data loaders for training and validation."""

    # Create datasets with official splits
    datasets = create_hbn_datasets(
        bids_root=args.data_root,
        window_length=args.window_length,
        overlap=args.overlap,
        task_type="cross_task",  # SuS → CCD transfer
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


def main():
    """Main training function."""

    # Parse arguments
    args = parse_args()

    # Setup logging
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"challenge1_{timestamp}"

    output_dir = Path(args.output_dir) / args.experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(
        level=getattr(logging, args.log_level),
        log_file=output_dir / "training.log"
    )

    logger.info(f"Starting Challenge 1 training: {args.experiment_name}")
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

    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    if test_loader:
        logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    logger.info("Creating Challenge 1 model...")
    model = Challenge1Model(config, n_channels=128)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create trainer
    trainer = Challenge1Trainer(
        config=config,
        model=model,
        device=device,
        log_dir=output_dir
    )

    # Training loop
    logger.info("Starting training...")

    try:
        history = trainer.fit(train_loader, val_loader)

        logger.info("Training completed successfully!")
        logger.info(f"Best validation metric: {trainer.best_metric:.4f}")

        # Final evaluation on test set if available
        if test_loader:
            logger.info("Evaluating on test set...")

            # Load best model
            best_checkpoint = output_dir / "best_model.ckpt"
            if best_checkpoint.exists():
                checkpoint = torch.load(best_checkpoint, map_location=device)
                model.load_state_dict(checkpoint["model_state_dict"])
                logger.info("Loaded best model for test evaluation")

            # Test evaluation
            test_predictions = []
            test_targets = []

            model.eval()
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

        # Save training history
        history_path = output_dir / "training_history.yaml"
        with open(history_path, "w") as f:
            yaml.dump(history, f, default_flow_style=False)

    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

    logger.info(f"Training completed. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
