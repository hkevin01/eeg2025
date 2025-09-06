"""
Integration of task-aware models with enhanced training pipeline.

This module provides optimized training loops, performance benchmarking,
and comprehensive evaluation for the enhanced EEG foundation model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from pathlib import Path
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import wandb
from tqdm import tqdm
import gc

# Import our components
from ..models.task_aware import (
    TaskAwareTemporalCNN, MultiTaskHead, HBNTask
)
from ..models.invariance.dann import MultiAdversaryDANN, AdversaryType
from ..utils.gpu_optimization import OptimizedModel, ModelBenchmarker
from ..data.enhanced_pipeline import (
    create_enhanced_dataloader, RealLabelManager
)
from ..utils.augmentations import (
    TimeMasking, CompressionDistortion, SchedulableAugmentation
)


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for enhanced training."""
    # Model parameters
    input_channels: int = 19
    hidden_dim: int = 128
    num_layers: int = 6
    dropout: float = 0.1
    use_task_tokens: bool = True
    adaptation_method: str = "film"  # "film", "lora", "linear"

    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_epochs: int = 5

    # Multi-adversary DANN
    use_multi_adversary: bool = True
    adversary_types: List[str] = None
    adversary_lambda: float = 1.0
    adversary_schedule: str = "linear"  # "constant", "linear", "exp"

    # Optimization
    use_amp: bool = True
    compile_mode: str = "max-autotune"
    gradient_clip: float = 1.0

    # Data parameters
    sequence_length: int = 1000
    num_workers: int = 4

    # Augmentation schedule
    use_schedulable_augs: bool = True
    mask_ratio_schedule: Tuple[float, float] = (0.1, 0.3)
    compression_schedule: Tuple[float, float] = (0.0, 0.1)

    # Evaluation
    eval_frequency: int = 5
    save_frequency: int = 10
    benchmark_frequency: int = 20

    def __post_init__(self):
        if self.adversary_types is None:
            self.adversary_types = ["site", "subject", "session"]


class EnhancedTrainer:
    """Enhanced trainer with task-aware models and multi-adversary training."""

    def __init__(self,
                 config: TrainingConfig,
                 data_root: Union[str, Path],
                 output_dir: Union[str, Path],
                 device: str = "auto"):
        """
        Initialize enhanced trainer.

        Args:
            config: Training configuration
            data_root: Root data directory
            output_dir: Output directory for checkpoints
            device: Training device
        """
        self.config = config
        self.data_root = Path(data_root)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize components
        self.label_manager = RealLabelManager(data_root)
        self.benchmarker = ModelBenchmarker(str(self.device))

        # Initialize models
        self._init_models()

        # Initialize optimizers
        self._init_optimizers()

        # Initialize data loaders
        self._init_data_loaders()

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # Metrics tracking
        self.train_metrics = []
        self.val_metrics = []
        self.benchmark_results = []

    def _init_models(self):
        """Initialize task-aware model with multi-adversary training."""
        # Task-aware backbone
        self.backbone = TaskAwareTemporalCNN(
            input_channels=self.config.input_channels,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
            use_task_tokens=self.config.use_task_tokens,
            adaptation_method=self.config.adaptation_method
        )

        # Multi-task head
        self.multi_task_head = MultiTaskHead(
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout
        )

        # Multi-adversary DANN
        if self.config.use_multi_adversary:
            adversary_types = [AdversaryType(t) for t in self.config.adversary_types]
            self.dann = MultiAdversaryDANN(
                feature_dim=self.config.hidden_dim,
                adversary_types=adversary_types,
                hidden_dim=self.config.hidden_dim
            )
        else:
            self.dann = None

        # Combine into optimized model
        self.model = OptimizedModel(
            model=nn.ModuleDict({
                'backbone': self.backbone,
                'multi_task_head': self.multi_task_head,
                'dann': self.dann
            }) if self.dann else nn.ModuleDict({
                'backbone': self.backbone,
                'multi_task_head': self.multi_task_head
            }),
            use_amp=self.config.use_amp,
            compile_mode=self.config.compile_mode,
            device=str(self.device)
        )

        logger.info(f"Initialized models on {self.device}")
        logger.info(f"Model optimization: {self.model.get_optimization_info()}")

    def _init_optimizers(self):
        """Initialize optimizers with different schedules."""
        # Main optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.num_epochs // 4,
            T_mult=2,
            eta_min=self.config.learning_rate * 0.01
        )

        # Warmup scheduler
        if self.config.warmup_epochs > 0:
            self.warmup_scheduler = optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=0.1,
                total_iters=self.config.warmup_epochs
            )
        else:
            self.warmup_scheduler = None

    def _init_data_loaders(self):
        """Initialize enhanced data loaders for all tasks."""
        self.train_loaders = {}
        self.val_loaders = {}

        # Create augmentations
        augmentations = []
        if self.config.use_schedulable_augs:
            augmentations = [
                TimeMasking(
                    mask_ratio=self.config.mask_ratio_schedule[0],
                    schedulable=True
                ),
                CompressionDistortion(
                    distortion_percentage=self.config.compression_schedule[0],
                    schedulable=True
                )
            ]

        # Create loaders for each task
        for task in HBNTask:
            try:
                # Training loader
                train_loader = create_enhanced_dataloader(
                    data_root=self.data_root,
                    task=task,
                    split="train",
                    batch_size=self.config.batch_size,
                    sequence_length=self.config.sequence_length,
                    num_workers=self.config.num_workers,
                    label_manager=self.label_manager,
                    augmentations=augmentations.copy()
                )

                # Validation loader
                val_loader = create_enhanced_dataloader(
                    data_root=self.data_root,
                    task=task,
                    split="val",
                    batch_size=self.config.batch_size,
                    sequence_length=self.config.sequence_length,
                    num_workers=self.config.num_workers,
                    label_manager=self.label_manager,
                    augmentations=[]  # No augmentations for validation
                )

                if len(train_loader.dataset) > 0:
                    self.train_loaders[task] = train_loader
                    logger.info(f"Created {task.value} train loader: {len(train_loader.dataset)} samples")

                if len(val_loader.dataset) > 0:
                    self.val_loaders[task] = val_loader
                    logger.info(f"Created {task.value} val loader: {len(val_loader.dataset)} samples")

            except Exception as e:
                logger.warning(f"Failed to create loaders for {task.value}: {e}")

        logger.info(f"Initialized {len(self.train_loaders)} training loaders")
        logger.info(f"Initialized {len(self.val_loaders)} validation loaders")

    def _update_augmentation_schedule(self, epoch: int):
        """Update schedulable augmentation parameters."""
        if not self.config.use_schedulable_augs:
            return

        # Calculate schedule progress
        progress = min(1.0, epoch / self.config.num_epochs)

        # Update mask ratio
        mask_start, mask_end = self.config.mask_ratio_schedule
        current_mask_ratio = mask_start + progress * (mask_end - mask_start)

        # Update compression distortion
        comp_start, comp_end = self.config.compression_schedule
        current_compression = comp_start + progress * (comp_end - comp_start)

        # Update augmentations in all train loaders
        for task, loader in self.train_loaders.items():
            dataset = loader.dataset
            for aug in dataset.augmentations:
                if isinstance(aug, SchedulableAugmentation):
                    if hasattr(aug, 'mask_ratio'):
                        aug.mask_ratio = current_mask_ratio
                    if hasattr(aug, 'distortion_percentage'):
                        aug.distortion_percentage = current_compression

        logger.debug(f"Updated augmentations: mask_ratio={current_mask_ratio:.3f}, compression={current_compression:.3f}")

    def _compute_losses(self, batch: Dict[str, torch.Tensor], task: HBNTask) -> Dict[str, torch.Tensor]:
        """Compute all losses for a batch."""
        # Extract input
        eeg_data = batch["eeg"].to(self.device)
        batch_size = eeg_data.shape[0]

        # Forward pass through backbone
        features = self.model.model['backbone'](eeg_data, task)

        # Multi-task predictions
        predictions = self.model.model['multi_task_head'](features)

        losses = {}

        # Task-specific losses
        if task == HBNTask.CCD:
            # CCD regression losses
            if 'ccd_rt_mean' in batch:
                rt_target = batch['ccd_rt_mean'].to(self.device)
                rt_pred = predictions['ccd_rt']
                losses['ccd_rt'] = nn.MSELoss()(rt_pred.squeeze(), rt_target)

            if 'ccd_accuracy' in batch:
                acc_target = batch['ccd_accuracy'].to(self.device)
                acc_pred = predictions['ccd_success']
                losses['ccd_accuracy'] = nn.MSELoss()(acc_pred.squeeze(), acc_target)

        # CBCL factor losses (for all tasks)
        cbcl_targets = []
        cbcl_preds = []

        for factor in ['internalizing', 'externalizing', 'attention', 'total']:
            key = f'cbcl_{factor}'
            if key in batch:
                cbcl_targets.append(batch[key].to(self.device))
                cbcl_preds.append(predictions[f'cbcl_{factor}'])

        if cbcl_targets:
            cbcl_target = torch.stack(cbcl_targets, dim=1)
            cbcl_pred = torch.stack(cbcl_preds, dim=1)
            losses['cbcl'] = nn.MSELoss()(cbcl_pred, cbcl_target)

        # Multi-adversary losses
        if self.dann is not None:
            adversary_losses = {}

            # Site adversary
            if 'site' in batch:
                site_labels = self._encode_categorical(batch['site'], 'site')
                if site_labels is not None:
                    adversary_losses[AdversaryType.SITE] = site_labels.to(self.device)

            # Subject adversary (use subject_id hash)
            if 'subject_id' in batch:
                subject_labels = torch.tensor([
                    hash(sid) % 100 for sid in batch['subject_id']
                ], dtype=torch.long).to(self.device)
                adversary_losses[AdversaryType.SUBJECT] = subject_labels

            # Age adversary
            if 'age' in batch:
                age_labels = (batch['age'].to(self.device) // 2).long()  # Bin by 2-year groups
                age_labels = torch.clamp(age_labels, 0, 10)  # Cap at 10 groups
                adversary_losses[AdversaryType.AGE_GROUP] = age_labels

            # Task adversary
            task_labels = torch.full((batch_size,), task.value, dtype=torch.long).to(self.device)
            adversary_losses[AdversaryType.TASK] = task_labels

            # Compute adversarial loss
            if adversary_losses:
                # Calculate lambda based on schedule
                adversary_lambda = self._get_adversary_lambda(self.current_epoch)

                adv_loss = self.model.model['dann'].compute_adversarial_loss(
                    features, adversary_losses, adversary_lambda
                )
                losses['adversarial'] = adv_loss

        return losses, predictions

    def _encode_categorical(self, values: List[str], category: str) -> Optional[torch.Tensor]:
        """Encode categorical values to integer labels."""
        if category == 'site':
            # Common HBN sites
            site_mapping = {
                'RU': 0, 'CBIC': 1, 'CUNY': 2, 'NYU': 3, 'SI': 4,
                'Unknown': 5
            }
            labels = [site_mapping.get(site, 5) for site in values]
            return torch.tensor(labels, dtype=torch.long)

        return None

    def _get_adversary_lambda(self, epoch: int) -> float:
        """Get adversary lambda based on schedule."""
        if self.config.adversary_schedule == "constant":
            return self.config.adversary_lambda
        elif self.config.adversary_schedule == "linear":
            progress = min(1.0, epoch / self.config.num_epochs)
            return self.config.adversary_lambda * progress
        elif self.config.adversary_schedule == "exp":
            progress = min(1.0, epoch / self.config.num_epochs)
            return self.config.adversary_lambda * (1 - np.exp(-5 * progress))
        else:
            return self.config.adversary_lambda

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch across all tasks."""
        self.model.train()

        # Update augmentation schedule
        self._update_augmentation_schedule(self.current_epoch)

        epoch_losses = {}
        epoch_metrics = {}

        # Iterate through all tasks
        for task, train_loader in self.train_loaders.items():
            task_losses = []

            pbar = tqdm(train_loader, desc=f"Training {task.value}")
            for batch_idx, batch in enumerate(pbar):
                try:
                    # Compute losses
                    losses, predictions = self._compute_losses(batch, task)

                    # Total loss
                    total_loss = sum(losses.values())

                    # Backward pass with optimization
                    if self.config.use_amp:
                        with autocast():
                            total_loss_scaled = total_loss

                        self.model.scaler.scale(total_loss_scaled).backward()

                        if self.config.gradient_clip > 0:
                            self.model.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.gradient_clip
                            )

                        self.model.scaler.step(self.optimizer)
                        self.model.scaler.update()
                    else:
                        total_loss.backward()

                        if self.config.gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.config.gradient_clip
                            )

                        self.optimizer.step()

                    self.optimizer.zero_grad()

                    # Track losses
                    loss_dict = {f"{task.value}_{k}": v.item() for k, v in losses.items()}
                    loss_dict[f"{task.value}_total"] = total_loss.item()
                    task_losses.append(loss_dict)

                    # Update progress bar
                    pbar.set_postfix(loss=total_loss.item())

                    self.global_step += 1

                except Exception as e:
                    logger.error(f"Training error for {task.value}: {e}")
                    continue

            # Aggregate task losses
            if task_losses:
                for key in task_losses[0].keys():
                    values = [loss[key] for loss in task_losses if key in loss]
                    if values:
                        epoch_losses[key] = np.mean(values)

        # Update learning rate
        if self.current_epoch < self.config.warmup_epochs and self.warmup_scheduler:
            self.warmup_scheduler.step()
        else:
            self.scheduler.step()

        return epoch_losses

    def validate(self) -> Dict[str, float]:
        """Validate on all tasks."""
        self.model.eval()

        val_losses = {}

        with torch.no_grad():
            for task, val_loader in self.val_loaders.items():
                task_losses = []

                for batch in val_loader:
                    try:
                        losses, predictions = self._compute_losses(batch, task)
                        total_loss = sum(losses.values())

                        loss_dict = {f"{task.value}_{k}": v.item() for k, v in losses.items()}
                        loss_dict[f"{task.value}_total"] = total_loss.item()
                        task_losses.append(loss_dict)

                    except Exception as e:
                        logger.error(f"Validation error for {task.value}: {e}")
                        continue

                # Aggregate task losses
                if task_losses:
                    for key in task_losses[0].keys():
                        values = [loss[key] for loss in task_losses if key in loss]
                        if values:
                            val_losses[key] = np.mean(values)

        return val_losses

    def benchmark_performance(self):
        """Run performance benchmark."""
        logger.info("Running performance benchmark...")

        # Test different input shapes
        input_shapes = [
            (1, self.config.input_channels, self.config.sequence_length),
            (8, self.config.input_channels, self.config.sequence_length),
            (32, self.config.input_channels, self.config.sequence_length),
        ]

        # Benchmark the backbone model
        results = self.benchmarker.benchmark_comprehensive(
            model=self.backbone,
            input_shapes=input_shapes,
            save_path=self.output_dir / f"benchmark_epoch_{self.current_epoch}.json"
        )

        self.benchmarker.print_summary(results)
        self.benchmark_results.extend(results)

    def save_checkpoint(self, is_best: bool = False):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': asdict(self.config),
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics,
            'best_val_loss': self.best_val_loss
        }

        # Save regular checkpoint
        checkpoint_path = self.output_dir / f"checkpoint_epoch_{self.current_epoch}.pt"
        torch.save(checkpoint, checkpoint_path)

        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model with val_loss: {self.best_val_loss:.4f}")

    def train(self):
        """Main training loop."""
        logger.info("Starting enhanced training...")
        logger.info(f"Training configuration: {asdict(self.config)}")

        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch

            # Training
            train_losses = self.train_epoch()
            self.train_metrics.append({'epoch': epoch, **train_losses})

            # Validation
            if epoch % self.config.eval_frequency == 0:
                val_losses = self.validate()
                self.val_metrics.append({'epoch': epoch, **val_losses})

                # Check for best model
                avg_val_loss = np.mean([v for k, v in val_losses.items() if 'total' in k])
                is_best = avg_val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = avg_val_loss

                logger.info(f"Epoch {epoch}: train_loss={np.mean(list(train_losses.values())):.4f}, "
                           f"val_loss={avg_val_loss:.4f}")

            # Save checkpoint
            if epoch % self.config.save_frequency == 0:
                self.save_checkpoint(is_best=is_best if epoch % self.config.eval_frequency == 0 else False)

            # Performance benchmark
            if epoch % self.config.benchmark_frequency == 0 and epoch > 0:
                self.benchmark_performance()

            # Memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Final benchmark
        self.benchmark_performance()

        # Save final metrics
        metrics_path = self.output_dir / "training_metrics.json"
        with open(metrics_path, 'w') as f:
            json.dump({
                'train_metrics': self.train_metrics,
                'val_metrics': self.val_metrics,
                'benchmark_results': self.benchmark_results,
                'config': asdict(self.config)
            }, f, indent=2)

        logger.info("Training completed!")


if __name__ == "__main__":
    # Example training configuration
    config = TrainingConfig(
        input_channels=19,
        hidden_dim=128,
        num_layers=6,
        batch_size=32,
        learning_rate=1e-4,
        num_epochs=100,
        use_task_tokens=True,
        adaptation_method="film",
        use_multi_adversary=True,
        adversary_types=["site", "subject", "task"],
        use_amp=True,
        compile_mode="max-autotune"
    )

    # Initialize trainer
    data_root = Path("/path/to/hbn/data")  # Update with actual path
    output_dir = Path("./experiments/enhanced_training")

    if data_root.exists():
        trainer = EnhancedTrainer(
            config=config,
            data_root=data_root,
            output_dir=output_dir
        )

        # Start training
        trainer.train()
    else:
        print("Data root not found - please update path")
        print("Training configuration created successfully!")

    print("✅ Enhanced training integration completed!")
