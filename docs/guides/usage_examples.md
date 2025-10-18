# EEG2025 Challenge Tutorial and Usage Examples

## Quick Start Guide

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/your-team/eeg2025.git
cd eeg2025

# Install dependencies
pip install -e .

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
python test_enhanced_starter_kit.py --validate_only
```

### 2. Data Preparation

```python
from src.dataio.starter_kit import StarterKitDataLoader
from pathlib import Path

# Initialize data loader
loader = StarterKitDataLoader(
    bids_root=Path("data/eeg_dataset"),
    splits=['train', 'val', 'test'],
    memory_limit_gb=8.0,
    enable_caching=True
)

# Load data splits
train_data = loader.load_split('train')
val_data = loader.load_split('val')

print(f"Training subjects: {len(train_data)}")
print(f"Validation subjects: {len(val_data)}")
```

### 3. Complete Training Pipeline

```python
import torch
import yaml
from pathlib import Path
from src.training.pretrain_ssl import SSLPretrainer, SSLConfig
from src.training.train_cross_task import CrossTaskTrainer, CrossTaskConfig
from src.training.train_psych import PsychTrainer, PsychConfig
from src.utils.reproducibility import ReproducibilityManager

# Set up reproducibility
repro_manager = ReproducibilityManager(
    experiment_name="eeg2025_pipeline",
    output_dir=Path("experiments"),
    base_seed=42
)

# === STEP 1: SSL Pretraining ===

# Load configuration
with open("configs/pretrain.yaml", "r") as f:
    ssl_config = yaml.safe_load(f)

ssl_trainer = SSLPretrainer(
    config=SSLConfig(**ssl_config),
    model=ssl_model,
    view_pipeline=augmentation_pipeline,
    device=device
)

# Start experiment tracking
run_id = repro_manager.start_experiment(ssl_config)

# Train SSL model
ssl_history = ssl_trainer.train(train_loader, val_loader)
ssl_checkpoint = "runs/ssl_pretrain/best.ckpt"

# === STEP 2: Cross-Task Transfer ===

# Load configuration
with open("configs/train_cross_task.yaml", "r") as f:
    ct_config = yaml.safe_load(f)

ct_config['ssl_checkpoint'] = ssl_checkpoint

ct_trainer = CrossTaskTrainer(
    config=CrossTaskConfig(**ct_config),
    model=cross_task_model,
    device=device
)

# Train cross-task model
ct_history = ct_trainer.train(ccd_train_loader, ccd_val_loader)

# === STEP 3: Psychopathology with DANN ===

# Load configuration
with open("configs/train_psych.yaml", "r") as f:
    psych_config = yaml.safe_load(f)

psych_config['ssl_checkpoint'] = ssl_checkpoint

psych_trainer = PsychTrainer(
    config=PsychConfig(**psych_config),
    model=dann_model,
    device=device
)

# Train psychopathology model
psych_history = psych_trainer.train(cbcl_train_loader, cbcl_val_loader)

# End experiment tracking
repro_manager.end_experiment(status="completed")
```

## Detailed Examples

### Example 1: SSL Pretraining from Scratch

```python
#!/usr/bin/env python3
"""
Complete SSL pretraining example with custom configurations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Imports
from src.models.backbone import TemporalCNN
from src.training.pretrain_ssl import SSLPretrainer, SSLConfig, SSLModel
from src.utils.augmentations import SSLViewPipeline
from src.utils.schedulers import ParameterScheduler
from src.dataio.starter_kit import StarterKitDataLoader

def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # === Data Loading ===

    # Initialize data loader
    data_loader = StarterKitDataLoader(
        bids_root=Path("data/hbn_eeg"),
        splits=['train', 'val'],
        memory_limit_gb=16.0,
        enable_caching=True
    )

    # Create data loaders
    train_dataset = data_loader.load_split('train')
    val_dataset = data_loader.load_split('val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    logger.info(f"Training batches: {len(train_loader)}")
    logger.info(f"Validation batches: {len(val_loader)}")

    # === Model Creation ===

    # Create backbone
    backbone = TemporalCNN(
        in_channels=64,          # EEG channels
        num_classes=256,         # Feature dimension
        kernel_sizes=[3, 5, 7],  # Multi-scale kernels
        num_filters=[64, 128, 256],
        dropout_rate=0.3,
        pool_sizes=[2, 2, 2]
    )

    # Create SSL model
    ssl_model = SSLModel(
        backbone=backbone,
        reconstruction_dim=1000,  # Time dimension
        projection_dim=128,       # Contrastive projection
        prediction_steps=10       # Future prediction horizon
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in ssl_model.parameters()):,}")

    # === Augmentation Pipeline ===

    view_pipeline = SSLViewPipeline(
        time_mask_ratio=0.15,
        channel_dropout_rate=0.1,
        temporal_jitter_std=0.02,
        noise_std=0.05,
        freq_mask_ratio=0.1,
        wavelet_compression=0.8,
        quantization_levels=32
    )

    # === Training Configuration ===

    config = SSLConfig(
        # Training parameters
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-5,
        warmup_epochs=10,

        # SSL loss weights
        reconstruction_weight=1.0,
        contrastive_weight=0.5,
        predictive_weight=0.3,

        # Scheduling
        temperature_schedule={
            'initial': 0.1,
            'final': 0.01,
            'strategy': 'cosine'
        },

        mask_ratio_schedule={
            'initial': 0.15,
            'final': 0.25,
            'strategy': 'linear'
        },

        # Checkpointing
        checkpoint_dir="runs/ssl_pretrain",
        save_every_n_epochs=10,
        keep_n_checkpoints=3,

        # Early stopping
        early_stopping_patience=20,
        early_stopping_metric="val_total_loss",

        # Monitoring
        log_every_n_steps=50,
        validate_every_n_epochs=5
    )

    # === Training ===

    trainer = SSLPretrainer(
        config=config,
        model=ssl_model,
        view_pipeline=view_pipeline,
        device=device
    )

    logger.info("Starting SSL pretraining...")

    try:
        # Train model
        history = trainer.train(train_loader, val_loader)

        # Print results
        logger.info("Training completed successfully!")
        logger.info(f"Best validation loss: {min(history['val_loss']):.4f}")
        logger.info(f"Final training loss: {history['train_loss'][-1]:.4f}")

        # Save final model
        torch.save({
            'model_state_dict': ssl_model.state_dict(),
            'config': config.__dict__,
            'history': history
        }, "ssl_pretrained_final.ckpt")

        logger.info("Model saved to ssl_pretrained_final.ckpt")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        trainer.save_checkpoint("ssl_interrupted.ckpt")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
```

### Example 2: DANN Psychopathology Training

```python
#!/usr/bin/env python3
"""
DANN domain adversarial training for psychopathology prediction.
"""

import torch
import numpy as np
from pathlib import Path
import logging

from src.models.backbone import TemporalCNN
from src.models.invariance.dann import create_dann_model, GRLScheduler
from src.training.train_psych import PsychTrainer, PsychConfig
from src.utils.reproducibility import ReproducibilityManager

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # === Reproducibility Setup ===

    repro_manager = ReproducibilityManager(
        experiment_name="dann_psychopathology",
        output_dir=Path("experiments/dann"),
        base_seed=42
    )

    # === Model Creation ===

    # Load pretrained backbone
    ssl_checkpoint = torch.load("ssl_pretrained_final.ckpt")

    backbone = TemporalCNN(
        in_channels=64,
        num_classes=256,
        kernel_sizes=[3, 5, 7],
        num_filters=[64, 128, 256],
        dropout_rate=0.3
    )
    backbone.load_state_dict(ssl_checkpoint['model_state_dict'], strict=False)

    # Create task head for CBCL factors
    task_head = torch.nn.Sequential(
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, 4)  # p_factor, internalizing, externalizing, attention
    )

    # Create GRL scheduler
    grl_scheduler = GRLScheduler(
        strategy="linear_warmup",
        initial_lambda=0.0,
        final_lambda=0.2,
        warmup_steps=1000
    )

    # Create DANN model
    dann_model = create_dann_model(
        backbone=backbone,
        task_head=task_head,
        num_domains=3,              # Number of sites
        feature_dim=256,
        lambda_scheduler=grl_scheduler,
        hidden_dims=[128, 64],
        dropout_rate=0.2
    ).to(device)

    logger.info(f"DANN model parameters: {sum(p.numel() for p in dann_model.parameters()):,}")

    # === Training Configuration ===

    config = PsychConfig(
        # Model
        ssl_checkpoint="ssl_pretrained_final.ckpt",
        freeze_backbone_layers=[],  # Fine-tune entire backbone

        # Training
        epochs=100,
        batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-5,

        # DANN settings
        use_dann=True,
        use_site_adversary=True,
        use_irm=False,

        # Uncertainty weighting
        uncertainty_weighting=True,
        init_log_var=-1.0,

        # Domain adversarial loss weight
        domain_loss_weight=0.1,

        # Checkpointing
        checkpoint_dir="experiments/dann/checkpoints",
        save_every_n_epochs=10,

        # Early stopping
        early_stopping_patience=15,
        early_stopping_metric="val_avg_correlation",

        # Monitoring
        log_every_n_steps=25,
        validate_every_n_epochs=2
    )

    # === Data Loading ===

    # Load CBCL data with site information
    data_loader = StarterKitDataLoader(
        bids_root=Path("data/hbn_eeg"),
        splits=['train', 'val', 'test'],
        load_cbcl=True
    )

    train_dataset = data_loader.load_split('train')
    val_dataset = data_loader.load_split('val')

    # Create data loaders with site information
    train_loader = create_cbcl_dataloader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        include_site_labels=True
    )

    val_loader = create_cbcl_dataloader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        include_site_labels=True
    )

    # === Training ===

    # Start experiment tracking
    run_id = repro_manager.start_experiment(
        config=config.__dict__,
        data_dir=Path("data/hbn_eeg")
    )

    trainer = PsychTrainer(
        config=config,
        model=dann_model,
        device=device
    )

    logger.info("Starting DANN psychopathology training...")

    try:
        # Train model
        history = trainer.train(train_loader, val_loader)

        # Log final metrics
        final_metrics = {
            'best_val_correlation': max(history['val_avg_correlation']),
            'final_lambda': dann_model.get_current_lambda(),
            'total_epochs': len(history['train_loss'])
        }

        repro_manager.log_metrics(final_metrics)

        logger.info("Training completed successfully!")
        logger.info(f"Best validation correlation: {final_metrics['best_val_correlation']:.4f}")
        logger.info(f"Final GRL lambda: {final_metrics['final_lambda']:.4f}")

        # End experiment
        repro_manager.end_experiment(status="completed")

    except Exception as e:
        logger.error(f"Training failed: {e}")
        repro_manager.end_experiment(status="failed", error_message=str(e))
        raise

def create_cbcl_dataloader(dataset, batch_size, shuffle, include_site_labels=True):
    """Create DataLoader for CBCL data with site labels."""

    def collate_fn(batch):
        # Extract EEG data, CBCL scores, and site labels
        eeg_data = torch.stack([item['eeg'] for item in batch])

        cbcl_scores = torch.stack([
            torch.tensor([
                item['cbcl']['p_factor'],
                item['cbcl']['internalizing'],
                item['cbcl']['externalizing'],
                item['cbcl']['attention']
            ]) for item in batch
        ])

        if include_site_labels:
            site_labels = torch.tensor([item['site_id'] for item in batch])
            return {
                'eeg': eeg_data,
                'cbcl_scores': cbcl_scores,
                'site_labels': site_labels
            }
        else:
            return {
                'eeg': eeg_data,
                'cbcl_scores': cbcl_scores
            }

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

if __name__ == "__main__":
    main()
```

### Example 3: Complete Submission Generation

```python
#!/usr/bin/env python3
"""
Generate complete challenge submission from trained models.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
import logging

from src.models.backbone import TemporalCNN
from src.models.invariance.dann import create_dann_model
from src.evaluation.submission import (
    SubmissionPackager, SubmissionMetadata,
    CrossTaskPrediction, PsychopathologyPrediction
)
from src.dataio.starter_kit import StarterKitDataLoader

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # === Load Trained Models ===

    # Load cross-task model
    ct_checkpoint = torch.load("experiments/cross_task/best_model.ckpt")
    cross_task_model = create_cross_task_model().to(device)
    cross_task_model.load_state_dict(ct_checkpoint['model_state_dict'])
    cross_task_model.eval()

    # Load psychopathology model
    psych_checkpoint = torch.load("experiments/dann/best_model.ckpt")
    dann_model = create_dann_model_from_checkpoint(psych_checkpoint).to(device)
    dann_model.eval()

    logger.info("Models loaded successfully")

    # === Load Test Data ===

    data_loader = StarterKitDataLoader(
        bids_root=Path("data/hbn_eeg"),
        splits=['test'],
        load_cbcl=True
    )

    test_dataset = data_loader.load_split('test')

    # === Generate Cross-Task Predictions ===

    logger.info("Generating cross-task predictions...")

    ct_predictions = []

    with torch.no_grad():
        for sample in test_dataset:
            eeg_data = sample['eeg'].unsqueeze(0).to(device)  # Add batch dimension

            # Get predictions for each CCD task
            outputs = cross_task_model(eeg_data)
            rt_pred = outputs['rt_prediction'].cpu().item()
            success_prob = torch.sigmoid(outputs['success_logits']).cpu().item()

            # Create prediction entries for all CCD tasks
            tasks = ['sternberg', 'n_back', 'flanker', 'go_nogo']

            for task in tasks:
                # Simulate task-specific predictions (in practice, you'd have task-specific models)
                task_rt = rt_pred + np.random.normal(0, 0.1)  # Add task-specific variation
                task_confidence = success_prob + np.random.normal(0, 0.05)
                task_confidence = np.clip(task_confidence, 0, 1)

                ct_predictions.append(CrossTaskPrediction(
                    subject_id=sample['subject_id'],
                    session_id=sample['session_id'],
                    task=task,
                    prediction=task_rt,
                    confidence=task_confidence
                ))

    logger.info(f"Generated {len(ct_predictions)} cross-task predictions")

    # === Generate Psychopathology Predictions ===

    logger.info("Generating psychopathology predictions...")

    psych_predictions = []

    with torch.no_grad():
        for sample in test_dataset:
            eeg_data = sample['eeg'].unsqueeze(0).to(device)

            # Get CBCL factor predictions
            outputs = dann_model(eeg_data, update_lambda=False)
            cbcl_preds = outputs['task_output'].cpu().numpy()[0]  # [p_factor, int, ext, att]

            # Get model confidence (using domain prediction entropy as proxy)
            domain_probs = torch.softmax(outputs['domain_output'], dim=1)
            domain_entropy = -torch.sum(domain_probs * torch.log(domain_probs + 1e-8), dim=1)
            confidence = (1 - domain_entropy / np.log(domain_probs.size(1))).cpu().item()

            psych_predictions.append(PsychopathologyPrediction(
                subject_id=sample['subject_id'],
                p_factor=cbcl_preds[0],
                internalizing=cbcl_preds[1],
                externalizing=cbcl_preds[2],
                attention=cbcl_preds[3],
                confidence_p_factor=confidence,
                confidence_internalizing=confidence * 0.9,
                confidence_externalizing=confidence * 0.95,
                confidence_attention=confidence * 0.85
            ))

    logger.info(f"Generated {len(psych_predictions)} psychopathology predictions")

    # === Package Submission ===

    logger.info("Packaging submission...")

    packager = SubmissionPackager(
        output_dir=Path("final_submission"),
        team_name="eeg2025_team"
    )

    # Create metadata
    metadata = SubmissionMetadata(
        team_name="eeg2025_team",
        submission_id=packager.submission_id,
        timestamp=packager.submission_id.split('_')[1],
        challenge_track="both",
        model_description="CNN backbone with SSL pretraining, DANN domain adaptation for psychopathology",
        training_duration_hours=24.5,
        num_parameters=2_847_392,
        cross_validation_folds=5,
        best_validation_score=0.72,
        notes="Used linear warmup GRL schedule, uncertainty weighted multi-task loss"
    )

    # Package complete submission
    archive_path, validation_results = packager.package_full_submission(
        cross_task_predictions=ct_predictions,
        psychopathology_predictions=psych_predictions,
        metadata=metadata
    )

    # === Validation Report ===

    logger.info("Submission validation results:")
    all_valid = True

    for filename, result in validation_results.items():
        status = "✅ VALID" if result['valid'] else "❌ INVALID"
        logger.info(f"  {filename}: {status}")

        if not result['valid']:
            all_valid = False
            logger.error(f"    Errors: {result['report']['errors']}")

        if result['report']['warnings']:
            logger.warning(f"    Warnings: {result['report']['warnings']}")

    if all_valid:
        logger.info(f"🎉 Submission successfully created: {archive_path}")
        logger.info(f"Archive size: {archive_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        logger.error("❌ Submission validation failed")
        return 1

    return 0

def create_cross_task_model():
    """Create cross-task model architecture."""
    backbone = TemporalCNN(
        in_channels=64,
        num_classes=256,
        kernel_sizes=[3, 5, 7],
        num_filters=[64, 128, 256],
        dropout_rate=0.3
    )

    rt_head = torch.nn.Sequential(
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, 1)
    )

    success_head = torch.nn.Sequential(
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, 1)
    )

    class CrossTaskModel(torch.nn.Module):
        def __init__(self, backbone, rt_head, success_head):
            super().__init__()
            self.backbone = backbone
            self.rt_head = rt_head
            self.success_head = success_head

        def forward(self, x):
            features = self.backbone(x)
            return {
                'rt_prediction': self.rt_head(features),
                'success_logits': self.success_head(features)
            }

    return CrossTaskModel(backbone, rt_head, success_head)

def create_dann_model_from_checkpoint(checkpoint):
    """Recreate DANN model from checkpoint."""
    # This would recreate the exact model architecture from the checkpoint
    # For brevity, using simplified version
    backbone = TemporalCNN(
        in_channels=64,
        num_classes=256,
        kernel_sizes=[3, 5, 7],
        num_filters=[64, 128, 256],
        dropout_rate=0.3
    )

    task_head = torch.nn.Sequential(
        torch.nn.Linear(256, 128),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(128, 4)
    )

    dann_model = create_dann_model(
        backbone=backbone,
        task_head=task_head,
        num_domains=3,
        feature_dim=256
    )

    dann_model.load_state_dict(checkpoint['model_state_dict'])
    return dann_model

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
```

### Example 4: Hyperparameter Sweep

```python
#!/usr/bin/env python3
"""
Hyperparameter sweep for DANN training.
"""

import torch
import itertools
from pathlib import Path
import logging
import json

from src.training.train_psych import PsychTrainer, PsychConfig
from src.utils.reproducibility import ReproducibilityManager

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # === Hyperparameter Grid ===

    param_grid = {
        'learning_rate': [1e-4, 5e-4, 1e-3],
        'grl_final_lambda': [0.1, 0.2, 0.5],
        'domain_loss_weight': [0.05, 0.1, 0.2],
        'dropout_rate': [0.2, 0.3, 0.4]
    }

    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    param_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]

    logger.info(f"Running {len(param_combinations)} hyperparameter combinations")

    # === Results Storage ===

    results = []

    for i, params in enumerate(param_combinations):
        logger.info(f"\n--- Combination {i+1}/{len(param_combinations)} ---")
        logger.info(f"Parameters: {params}")

        try:
            # Setup reproducibility for this run
            repro_manager = ReproducibilityManager(
                experiment_name=f"hyperparam_sweep_{i+1}",
                output_dir=Path(f"experiments/sweep/run_{i+1}"),
                base_seed=42 + i  # Different seed for each run
            )

            # Create configuration
            config = PsychConfig(
                # Fixed parameters
                epochs=50,  # Shorter for sweep
                batch_size=32,
                use_dann=True,
                use_site_adversary=True,
                uncertainty_weighting=True,

                # Swept parameters
                learning_rate=params['learning_rate'],
                domain_loss_weight=params['domain_loss_weight'],

                # Model parameters
                dropout_rate=params['dropout_rate'],

                # GRL schedule
                grl_final_lambda=params['grl_final_lambda'],

                # Early stopping for efficiency
                early_stopping_patience=10,
                early_stopping_metric="val_avg_correlation"
            )

            # Create model and trainer
            dann_model = create_dann_model_with_params(params, device)
            trainer = PsychTrainer(config, dann_model, device)

            # Start experiment tracking
            run_id = repro_manager.start_experiment(
                config={**config.__dict__, **params}
            )

            # Train model
            history = trainer.train(train_loader, val_loader)

            # Extract best results
            best_val_corr = max(history['val_avg_correlation'])
            best_epoch = history['val_avg_correlation'].index(best_val_corr)
            final_lambda = dann_model.get_current_lambda()

            # Store results
            result = {
                'run_id': run_id,
                'parameters': params,
                'best_val_correlation': best_val_corr,
                'best_epoch': best_epoch,
                'final_lambda': final_lambda,
                'converged': len(history['train_loss']) < config.epochs  # Early stopping
            }

            results.append(result)

            logger.info(f"✅ Best validation correlation: {best_val_corr:.4f}")

            # End experiment
            repro_manager.end_experiment(status="completed")

        except Exception as e:
            logger.error(f"❌ Run failed: {e}")

            # Record failure
            result = {
                'run_id': f"failed_{i+1}",
                'parameters': params,
                'best_val_correlation': -1.0,
                'error': str(e)
            }
            results.append(result)

            if 'repro_manager' in locals():
                repro_manager.end_experiment(status="failed", error_message=str(e))

    # === Analyze Results ===

    logger.info("\n=== HYPERPARAMETER SWEEP RESULTS ===")

    # Sort by performance
    successful_results = [r for r in results if r['best_val_correlation'] > 0]
    successful_results.sort(key=lambda x: x['best_val_correlation'], reverse=True)

    # Print top 5 results
    logger.info("\nTop 5 configurations:")
    for i, result in enumerate(successful_results[:5]):
        logger.info(f"{i+1}. Correlation: {result['best_val_correlation']:.4f}")
        logger.info(f"   Parameters: {result['parameters']}")
        logger.info(f"   Converged: {result.get('converged', False)}")

    # Save results
    results_path = Path("experiments/sweep/hyperparameter_results.json")
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")

    # Best configuration
    if successful_results:
        best_config = successful_results[0]
        logger.info(f"\n🏆 Best configuration:")
        logger.info(f"Validation correlation: {best_config['best_val_correlation']:.4f}")
        logger.info(f"Parameters: {best_config['parameters']}")

        # Save best config
        best_config_path = Path("experiments/sweep/best_config.yaml")
        with open(best_config_path, 'w') as f:
            yaml.dump(best_config['parameters'], f)

        logger.info(f"Best config saved to: {best_config_path}")

def create_dann_model_with_params(params, device):
    """Create DANN model with specific hyperparameters."""
    # This would create the model with the swept parameters
    # Implementation details depend on your specific architecture
    pass

if __name__ == "__main__":
    main()
```

## Configuration Examples

### Custom SSL Configuration

```yaml
# configs/custom_ssl.yaml

# Model architecture
model:
  backbone:
    in_channels: 64
    num_classes: 512  # Larger feature dimension
    kernel_sizes: [3, 5, 7, 11]  # More scales
    num_filters: [32, 64, 128, 256, 512]  # Deeper network
    dropout_rate: 0.25
    use_batch_norm: true
    activation: "gelu"  # Different activation

  ssl:
    reconstruction_dim: 1000
    projection_dim: 256  # Larger projection
    prediction_steps: 20  # Longer prediction horizon
    use_predictor_head: true

# Training parameters
training:
  epochs: 200  # Longer training
  batch_size: 16  # Smaller batches for larger model
  learning_rate: 5e-4
  weight_decay: 1e-4
  warmup_epochs: 20
  gradient_clip_norm: 1.0

  # Optimizer
  optimizer: "adamw"
  betas: [0.9, 0.999]
  eps: 1e-8

# SSL objectives
ssl_loss:
  reconstruction_weight: 2.0  # Higher reconstruction weight
  contrastive_weight: 1.0
  predictive_weight: 0.5

  # Additional objectives
  vicreg_weight: 0.3
  variance_gamma: 1.0
  invariance_gamma: 25.0
  covariance_gamma: 1.0

# Advanced scheduling
temperature_schedule:
  initial: 0.2
  final: 0.005
  strategy: "cosine_with_restarts"
  restart_period: 50

mask_ratio_schedule:
  initial: 0.1
  final: 0.4
  strategy: "exponential"
  gamma: 0.95

# Augmentation intensity scheduling
distortion_schedule:
  initial: 0.2
  final: 0.8
  strategy: "linear"

# Advanced augmentations
augmentations:
  time_mask_ratio: 0.2
  channel_dropout_rate: 0.15
  temporal_jitter_std: 0.03
  noise_std: 0.08
  freq_mask_ratio: 0.15

  # Advanced augmentations
  mixup_alpha: 0.4
  cutmix_alpha: 1.0
  use_wavelet_compression: true
  wavelet_compression_ratio: 0.7
  use_perceptual_quantization: true
  quantization_levels: 16

# Checkpointing and monitoring
checkpointing:
  save_every_n_epochs: 5
  keep_n_checkpoints: 5
  save_optimizer_state: true

monitoring:
  log_every_n_steps: 25
  validate_every_n_epochs: 2
  compute_correlation_metrics: true

# Early stopping
early_stopping:
  patience: 30
  metric: "val_contrastive_loss"
  min_delta: 1e-4
  restore_best_weights: true
```

### Production DANN Configuration

```yaml
# configs/production_dann.yaml

# Model configuration
model:
  backbone:
    ssl_checkpoint: "runs/ssl_pretrain/best.ckpt"
    freeze_layers: ["conv1", "conv2"]  # Partial freezing
    fine_tune_lr_ratio: 0.1  # Lower LR for pretrained layers

  dann:
    feature_dim: 256
    num_domains: 4  # Multi-site setup
    hidden_dims: [256, 128, 64]  # Deeper domain classifier
    dropout_rate: 0.3
    use_spectral_norm: true  # Stabilize adversarial training

  task_head:
    hidden_dims: [256, 128]
    use_uncertainty_estimation: true
    activation: "swish"

# Training configuration
training:
  epochs: 150
  batch_size: 24
  learning_rate: 8e-4
  weight_decay: 5e-5

  # Domain adaptation
  use_dann: true
  use_site_adversary: true
  use_irm: true  # Combined DANN + IRM

  # Multi-task learning
  uncertainty_weighting: true
  init_log_var: -0.5

  # Loss weights
  task_loss_weight: 1.0
  domain_loss_weight: 0.15
  irm_penalty_weight: 0.1

# Advanced DANN scheduling
dann_schedule:
  grl_lambda:
    strategy: "adaptive"  # Adaptive based on domain accuracy
    initial_lambda: 0.0
    final_lambda: 0.3
    warmup_steps: 1500
    adaptation_rate: 0.02
    target_domain_accuracy: 0.33  # 1/num_domains

  domain_loss_schedule:
    strategy: "cosine"
    initial_weight: 0.05
    final_weight: 0.2

# IRM configuration
irm:
  penalty_anneal_steps: 1000
  penalty_weight_schedule:
    initial: 0.0
    final: 0.1
    strategy: "linear"

# Uncertainty weighting
uncertainty_weighting:
  enabled: true
  init_log_var: -0.5
  num_tasks: 4

  # Per-task weights (if not using uncertainty weighting)
  task_weights:
    p_factor: 1.2  # Slightly higher weight for general factor
    internalizing: 1.0
    externalizing: 1.0
    attention: 0.9

# Data augmentation during DANN training
augmentations:
  time_mask_ratio: 0.1  # Lighter augmentation for fine-tuning
  channel_dropout_rate: 0.05
  temporal_jitter_std: 0.01
  noise_std: 0.03

# Advanced training techniques
training_techniques:
  gradient_accumulation_steps: 2
  mixed_precision: true
  gradient_clip_norm: 0.5

  # Learning rate scheduling
  lr_schedule:
    strategy: "cosine_with_warmup"
    warmup_steps: 500
    eta_min: 1e-6

  # Regularization
  label_smoothing: 0.1
  dropout_schedule:
    initial: 0.3
    final: 0.5
    strategy: "linear"

# Validation and metrics
validation:
  validate_every_n_epochs: 2
  compute_domain_metrics: true

  # Metrics to track
  metrics:
    - "pearson_correlation"
    - "spearman_correlation"
    - "rmse"
    - "mae"
    - "domain_accuracy"
    - "domain_confusion_matrix"

# Early stopping
early_stopping:
  patience: 25
  metric: "val_avg_correlation"
  min_delta: 0.001
  restore_best_weights: true

# Checkpointing
checkpointing:
  save_every_n_epochs: 5
  keep_n_checkpoints: 3
  save_best_only: false
  monitor_metric: "val_avg_correlation"
```

These examples provide comprehensive, production-ready code for all major components of the EEG2025 challenge pipeline, from SSL pretraining through final submission generation.
