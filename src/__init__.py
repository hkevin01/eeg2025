"""
EEG2025: Foundation Model for EEG Analysis
==========================================

A comprehensive foundation model for electroencephalography (EEG) analysis
designed for the 2025 EEG Challenge. This package provides state-of-the-art
transformer architectures, domain adaptation techniques, and self-supervised
learning methods for EEG signal processing.

Key Features:
- Multi-adversarial Domain Adaptation Network (DANN)
- Task-aware adapters with FiLM and LoRA
- Compression-augmented self-supervised learning
- GPU-optimized operations with Triton kernels
- Comprehensive benchmarking and evaluation tools

Usage:
------
```python
from eeg2025.models.backbone import EEGTransformer
from eeg2025.models.adapters import TaskAwareAdapter
from eeg2025.training.trainers import SSLTrainer

# Initialize model
model = EEGTransformer(
    num_channels=64,
    sequence_length=1000,
    d_model=768,
    num_layers=12
)

# Add task-specific adaptation
adapter = TaskAwareAdapter(
    d_model=768,
    num_tasks=6,
    task_names=['rest', 'task', 'smt', 'video', 'movie', 'dmn']
)

# Train with SSL
trainer = SSLTrainer(model, adapter)
```

For more examples, see the documentation at https://eeg2025.readthedocs.io/
"""

__version__ = "0.1.0"
__author__ = "EEG Challenge Team"
__email__ = "team@eeg2025.org"

# Core imports for easy access
try:
    # Original components
    from src.dataio.bids_loader import HBNDataLoader
    from src.models.adapters.task_aware import TaskAwareAdapter

    # New foundation model components
    from src.models.backbone.eeg_transformer import EEGTransformer
    from src.models.backbones.temporal_cnn import TemporalCNN
    from src.models.backbones.transformer_tiny import TransformerTiny
    from src.models.compression_ssl.augmentation import CompressionAugmentation
    from src.models.heads.temporal_regression import TemporalRegressionHead
    from src.training.trainers.ssl_trainer import SSLTrainer

    __all__ = [
        # Original components
        "HBNDataLoader",
        "TemporalCNN",
        "TransformerTiny",
        # New foundation model components
        "EEGTransformer",
        "TaskAwareAdapter",
        "TemporalRegressionHead",
        "CompressionAugmentation",
        "SSLTrainer",
    ]
except ImportError as e:
    # Graceful fallback for missing dependencies
    import warnings

    warnings.warn(f"Some components could not be imported: {e}", ImportWarning)
    __all__ = []
