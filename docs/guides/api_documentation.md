# EEG2025 Challenge API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Core Models](#core-models)
3. [Training Infrastructure](#training-infrastructure)
4. [DANN Domain Adaptation](#dann-domain-adaptation)
5. [Submission System](#submission-system)
6. [Reproducibility](#reproducibility)
7. [Utilities](#utilities)
8. [Configuration](#configuration)

## Overview

The EEG2025 Challenge implementation provides a complete pipeline for EEG-based machine learning including:

- **Self-Supervised Learning (SSL)** pretraining
- **Cross-Task Transfer** learning
- **Domain Adversarial Neural Networks (DANN)** for psychopathology prediction
- **Submission packaging** and validation
- **Reproducibility** infrastructure

## Core Models

### TemporalCNN Backbone

```python
from src.models.backbone import TemporalCNN

model = TemporalCNN(
    in_channels=64,          # EEG channels
    num_classes=256,         # Feature dimensions
    kernel_sizes=[3, 5, 7],  # Multi-scale temporal kernels
    num_filters=[64, 128, 256], # Filter progression
    dropout_rate=0.3,        # Dropout for regularization
    pool_sizes=[2, 2, 2]     # Pooling progression
)

# Forward pass
features = model(eeg_data)  # [batch, channels, time] -> [batch, features]
```

### DANN Model

```python
from src.models.invariance.dann import create_dann_model, GRLScheduler

# Create scheduler
scheduler = GRLScheduler(
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
    feature_dim=128,            # Feature dimensions
    lambda_scheduler=scheduler,
    hidden_dims=[64, 32],       # Domain classifier layers
    dropout_rate=0.2
)

# Forward pass with domain adversarial training
outputs = dann_model(x, update_lambda=True)
# Returns: {
#   'task_output': tensor,    # CBCL factor predictions
#   'domain_output': tensor,  # Site predictions
#   'lambda': float,          # Current GRL lambda
#   'features': tensor        # If return_features=True
# }
```

### SSL Model

```python
from src.training.pretrain_ssl import SSLModel
from src.models.losses.ssl_losses import CombinedSSLLoss

# Create SSL model
ssl_model = SSLModel(
    backbone=backbone,
    reconstruction_dim=1000,    # Time dimension for reconstruction
    projection_dim=128,         # Contrastive projection dimension
    prediction_steps=10         # Future prediction horizon
)

# Combined SSL loss
ssl_loss = CombinedSSLLoss(
    reconstruction_weight=1.0,
    contrastive_weight=0.5,
    predictive_weight=0.3,
    temperature=0.1
)

# Training step
view1, view2, masks = augmentation_pipeline(eeg_data)
ssl_outputs = ssl_model(view1, view2, masks)
loss = ssl_loss(ssl_outputs, view1, view2, masks)
```

## Training Infrastructure

### SSL Pretraining

```python
from src.training.pretrain_ssl import SSLPretrainer, SSLConfig

# Configuration
config = SSLConfig(
    epochs=100,
    batch_size=32,
    learning_rate=1e-3,
    mask_ratio=0.15,
    temperature=0.1,
    checkpoint_dir="runs/ssl_pretrain"
)

# Create trainer
trainer = SSLPretrainer(
    config=config,
    model=ssl_model,
    view_pipeline=augmentation_pipeline,
    device=device
)

# Train model
history = trainer.train(train_loader, val_loader)
```

### Cross-Task Transfer

```python
from src.training.train_cross_task import CrossTaskTrainer, CrossTaskConfig

# Configuration with SSL checkpoint
config = CrossTaskConfig(
    ssl_checkpoint="runs/ssl_pretrain/best.ckpt",
    freeze_backbone_ratio=0.8,  # Freeze 80% of backbone layers
    epochs=50,
    learning_rate=1e-4,
    use_film_adapter=True,       # Feature-wise Linear Modulation
    use_mmd_alignment=True       # Maximum Mean Discrepancy
)

# Create trainer
trainer = CrossTaskTrainer(
    config=config,
    model=cross_task_model,
    device=device
)

# Train with official metrics
history = trainer.train(train_loader, val_loader)
```

### Psychopathology Training with DANN

```python
from src.training.train_psych import PsychTrainer, PsychConfig

# Configuration
config = PsychConfig(
    ssl_checkpoint="runs/ssl_pretrain/best.ckpt",
    epochs=100,
    learning_rate=1e-3,
    use_dann=True,
    use_site_adversary=True,
    use_irm=False,
    uncertainty_weighting=True
)

# Create trainer
trainer = PsychTrainer(
    config=config,
    model=dann_model,
    device=device
)

# Train with domain adversarial objective
history = trainer.train(train_loader, val_loader)
```

## DANN Domain Adaptation

### Gradient Reversal Layer

```python
from src.models.invariance.dann import GradientReversalLayer

# Create GRL
grl = GradientReversalLayer(lambda_val=0.1)

# Forward pass (identity)
x_reversed = grl(x)  # Same as x in forward

# Backward pass reverses gradients by -lambda_val
loss = domain_classifier(x_reversed).sum()
loss.backward()  # Gradients are reversed

# Update lambda during training
grl.set_lambda(new_lambda)
```

### GRL Scheduling Strategies

```python
from src.models.invariance.dann import GRLScheduler

# Linear warmup: 0 -> 0.2 over 1000 steps
linear_scheduler = GRLScheduler(
    strategy="linear_warmup",
    initial_lambda=0.0,
    final_lambda=0.2,
    warmup_steps=1000
)

# Exponential approach to final value
exp_scheduler = GRLScheduler(
    strategy="exponential",
    initial_lambda=0.0,
    final_lambda=0.2,
    total_steps=5000,
    gamma=0.001
)

# Cosine annealing with warmup
cosine_scheduler = GRLScheduler(
    strategy="cosine",
    initial_lambda=0.0,
    final_lambda=0.2,
    warmup_steps=500,
    total_steps=2000
)

# Adaptive based on domain accuracy
adaptive_scheduler = GRLScheduler(
    strategy="adaptive",
    initial_lambda=0.0,
    final_lambda=0.2,
    warmup_steps=1000,
    adaptation_rate=0.01
)

# Usage
for step in range(training_steps):
    current_lambda = scheduler.step(domain_accuracy=domain_acc)
    grl.set_lambda(current_lambda)
```

### Uncertainty Weighted Multi-Task Loss

```python
from src.training.train_psych import UncertaintyWeightedLoss

# Create loss with learned uncertainties
loss_fn = UncertaintyWeightedLoss(
    num_tasks=4,  # p_factor, internalizing, externalizing, attention
    init_log_var=-1.0  # Initial log variance
)

# Compute weighted loss
predictions = {
    'p_factor': pred_p,
    'internalizing': pred_int,
    'externalizing': pred_ext,
    'attention': pred_att
}

targets = {
    'p_factor': target_p,
    'internalizing': target_int,
    'externalizing': target_ext,
    'attention': target_att
}

total_loss, task_losses, uncertainties = loss_fn(predictions, targets)
```

### IRM Penalty

```python
from src.models.invariance.dann import IRMPenalty

# Create IRM penalty
irm_penalty = IRMPenalty(penalty_weight=1.0)

# Compute penalty across domains
penalty = irm_penalty.compute_penalty(
    features=features,           # [batch, feature_dim]
    targets=targets,            # [batch]
    domain_ids=domain_labels,   # [batch] domain indices
    classifier=task_head        # PyTorch module
)

# Add to total loss
total_loss = task_loss + irm_penalty_weight * penalty
```

## Submission System

### Creating Predictions

```python
from src.evaluation.submission import (
    CrossTaskPrediction, PsychopathologyPrediction
)

# Cross-task predictions
ct_predictions = [
    CrossTaskPrediction(
        subject_id="sub-001",
        session_id="ses-1",
        task="sternberg",
        prediction=0.85,    # RT prediction
        confidence=0.92     # Model confidence
    ),
    # ... more predictions
]

# Psychopathology predictions
psych_predictions = [
    PsychopathologyPrediction(
        subject_id="sub-001",
        p_factor=0.2,
        internalizing=-0.1,
        externalizing=0.4,
        attention=-0.3,
        confidence_p_factor=0.89,
        confidence_internalizing=0.82,
        confidence_externalizing=0.91,
        confidence_attention=0.78
    ),
    # ... more predictions
]
```

### Submission Packaging

```python
from src.evaluation.submission import SubmissionPackager, SubmissionMetadata
from datetime import datetime

# Create packager
packager = SubmissionPackager(
    output_dir=Path("submission_output"),
    team_name="my_team"
)

# Create metadata
metadata = SubmissionMetadata(
    team_name="my_team",
    submission_id="sub_001",
    timestamp=datetime.now().isoformat(),
    challenge_track="both",  # "cross_task", "psychopathology", or "both"
    model_description="CNN with DANN domain adaptation",
    training_duration_hours=12.5,
    num_parameters=2_500_000,
    cross_validation_folds=5,
    best_validation_score=0.73
)

# Package complete submission
archive_path, validation_results = packager.package_full_submission(
    cross_task_predictions=ct_predictions,
    psychopathology_predictions=psych_predictions,
    metadata=metadata
)

print(f"Submission archive: {archive_path}")
for filename, result in validation_results.items():
    print(f"{filename}: {'✓' if result['valid'] else '✗'}")
```

### Validation

```python
from src.evaluation.submission import SubmissionValidator

# Validate submission files
validator = SubmissionValidator()

# Validate cross-task CSV
is_valid = validator.validate_cross_task_csv("cross_task_submission.csv")
if not is_valid:
    report = validator.get_validation_report()
    print("Errors:", report['errors'])
    print("Warnings:", report['warnings'])

# Validate psychopathology CSV
is_valid = validator.validate_psychopathology_csv("psychopathology_submission.csv")
```

## Reproducibility

### Seed Management

```python
from src.utils.reproducibility import SeedManager

# Create seed manager
seed_manager = SeedManager(base_seed=42)

# Set all seeds for reproducibility
seed_manager.set_all_seeds()

# Get specific component seeds
data_seed = seed_manager.get_seed('data_loader')
model_seed = seed_manager.get_seed('model_init')

# Create PyTorch generators
train_generator = seed_manager.create_generator('training')
val_generator = seed_manager.create_generator('evaluation')

# Use in DataLoader
train_loader = DataLoader(
    dataset,
    batch_size=32,
    generator=train_generator,
    worker_init_fn=lambda worker_id: np.random.seed(data_seed + worker_id)
)
```

### Experiment Tracking

```python
from src.utils.reproducibility import ReproducibilityManager

# Create manager
repro_manager = ReproducibilityManager(
    experiment_name="dann_psychopathology",
    output_dir=Path("experiments"),
    base_seed=42
)

# Start experiment with config
config = {
    'model': {'type': 'dann', 'lambda_schedule': 'linear_warmup'},
    'training': {'epochs': 100, 'lr': 1e-3}
}

run_id = repro_manager.start_experiment(
    config=config,
    data_dir=Path("data/eeg_dataset")
)

# Log metrics during training
repro_manager.log_metrics({
    'epoch_10': {'train_loss': 0.45, 'val_loss': 0.52}
})

# Save checkpoints with tracking
repro_manager.save_checkpoint(
    model=dann_model,
    checkpoint_path=Path(f"checkpoints/{run_id}_epoch_10.ckpt"),
    metrics={'val_correlation': 0.68}
)

# End experiment
repro_manager.end_experiment(status="completed")
```

### Environment Capture

```python
from src.utils.reproducibility import EnvironmentCapture

# Capture current environment
env_capture = EnvironmentCapture()
env_info = env_capture.capture_environment()

print(f"Python: {env_info.python_version}")
print(f"PyTorch: {env_info.pytorch_version}")
print(f"CUDA: {env_info.cuda_version}")
print(f"Platform: {env_info.platform}")
print(f"Git commit: {env_info.git_info['commit_hash_short']}")

# Save environment info
with open("environment.json", "w") as f:
    json.dump(env_info.to_dict(), f, indent=2)
```

## Utilities

### Parameter Scheduling

```python
from src.utils.schedulers import ParameterScheduler

# Temperature scheduling for contrastive learning
temp_scheduler = ParameterScheduler(
    initial_value=0.1,
    final_value=0.01,
    total_steps=1000,
    strategy="cosine"  # "linear", "exponential", "cosine"
)

# Mask ratio scheduling for SSL
mask_scheduler = ParameterScheduler(
    initial_value=0.15,
    final_value=0.25,
    total_steps=5000,
    strategy="linear"
)

# Usage in training loop
for step in range(total_steps):
    current_temp = temp_scheduler.step()
    current_mask_ratio = mask_scheduler.step()

    # Use in loss computation
    contrastive_loss = InfoNCE(temperature=current_temp)
    masked_data = apply_masking(data, ratio=current_mask_ratio)
```

### SSL Augmentations

```python
from src.utils.augmentations import SSLViewPipeline

# Create augmentation pipeline
view_pipeline = SSLViewPipeline(
    time_mask_ratio=0.15,       # Fraction of time to mask
    channel_dropout_rate=0.1,   # Channel dropout probability
    temporal_jitter_std=0.02,   # Temporal jitter standard deviation
    noise_std=0.05,             # Gaussian noise standard deviation
    freq_mask_ratio=0.1,        # Frequency masking ratio
    wavelet_compression=0.8,    # Wavelet compression ratio
    quantization_levels=32      # Perceptual quantization levels
)

# Generate augmented views
view1, view2, masks = view_pipeline(eeg_data)

# Schedulable augmentation intensities
for epoch in range(num_epochs):
    # Increase augmentation intensity over time
    current_intensity = min(1.0, epoch / 50)
    view_pipeline.update_intensity(current_intensity)
```

### Loss Functions

```python
from src.models.losses.corr_mse import CorrMSELoss
from src.models.losses.ssl_losses import ContrastiveLoss

# Correlation + MSE loss for regression
corr_mse_loss = CorrMSELoss(
    mse_weight=1.0,
    corr_weight=0.5,
    correlation_type="pearson"  # "pearson" or "spearman"
)

loss = corr_mse_loss(predictions, targets)

# Contrastive loss for SSL
contrastive_loss = ContrastiveLoss(
    temperature=0.1,
    projection_dim=128
)

# Compute loss between augmented views
loss = contrastive_loss(features1, features2)
```

## Configuration

### SSL Pretraining Config (`configs/pretrain.yaml`)

```yaml
# Model configuration
model:
  backbone:
    in_channels: 64
    num_classes: 256
    kernel_sizes: [3, 5, 7]
    num_filters: [64, 128, 256]
    dropout_rate: 0.3

  ssl:
    reconstruction_dim: 1000
    projection_dim: 128
    prediction_steps: 10

# Training configuration
training:
  epochs: 100
  batch_size: 32
  learning_rate: 1e-3
  weight_decay: 1e-5
  warmup_epochs: 10

# SSL loss weights
ssl_loss:
  reconstruction_weight: 1.0
  contrastive_weight: 0.5
  predictive_weight: 0.3

# Parameter scheduling
temperature_schedule:
  initial: 0.1
  final: 0.01
  strategy: "cosine"

mask_ratio_schedule:
  initial: 0.15
  final: 0.25
  strategy: "linear"

# Augmentation configuration
augmentations:
  time_mask_ratio: 0.15
  channel_dropout_rate: 0.1
  temporal_jitter_std: 0.02
  noise_std: 0.05
  freq_mask_ratio: 0.1
```

### DANN Training Config (`configs/train_psych.yaml`)

```yaml
# Model configuration
model:
  backbone:
    ssl_checkpoint: "runs/ssl_pretrain/best.ckpt"
    freeze_layers: []

  dann:
    feature_dim: 128
    num_domains: 3
    hidden_dims: [64, 32]
    dropout_rate: 0.2

# Training configuration
training:
  epochs: 100
  batch_size: 32
  learning_rate: 1e-3
  weight_decay: 1e-5
  use_dann: true
  use_site_adversary: true
  use_irm: false

# DANN scheduling
dann_schedule:
  grl_lambda:
    strategy: "linear_warmup"
    initial_lambda: 0.0
    final_lambda: 0.2
    warmup_steps: 1000

# Uncertainty weighting
uncertainty_weighting:
  enabled: true
  init_log_var: -1.0
  num_tasks: 4

# Multi-task weights
task_weights:
  p_factor: 1.0
  internalizing: 1.0
  externalizing: 1.0
  attention: 1.0
```

## Error Handling

All components include comprehensive error handling:

```python
try:
    # Model training
    history = trainer.train(train_loader, val_loader)
except RuntimeError as e:
    logger.error(f"Training failed: {e}")
    # Automatic checkpoint saving
    trainer.save_checkpoint("emergency_checkpoint.ckpt")
except KeyboardInterrupt:
    logger.info("Training interrupted by user")
    trainer.save_checkpoint("interrupted_checkpoint.ckpt")
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    # Full traceback logging
    logger.error(traceback.format_exc())
```

## Performance Optimization

### Memory Management

```python
# Automatic memory monitoring
@memory_monitor(threshold_mb=1000)
def training_step(batch):
    # Training logic here
    pass

# Manual memory optimization
torch.cuda.empty_cache()
gc.collect()

# Gradient accumulation for large batches
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()

    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Multi-GPU Training

```python
# DataParallel
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

# DistributedDataParallel
if distributed:
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank]
    )
```

This comprehensive API documentation provides detailed usage examples and explanations for all major components of the EEG2025 challenge implementation.
