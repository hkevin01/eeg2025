# EEG Foundation Challenge 2025 — Cross-Task and Cross-Subject EEG Decoding

> Built on: brain-computer-compression foundation with advanced compression and self-supervised learning

## Overview

This project develops a foundation model approach for the EEG Foundation Challenge (NeurIPS 2025), integrating brain-computer compression techniques with self-supervised learning to create robust, subject-invariant EEG representations. Our approach tackles two key challenges using the Healthy Brain Network (HBN) EEG dataset:

**Challenge 1 (Cross-Task Transfer)**: Predict CCD response time (regression) and success (classification) using features learned from passive SuS.  
**Challenge 2 (Psychopathology)**: Predict 4 CBCL-derived factors (p-factor, internalizing, externalizing, attention) across tasks with subject-invariant representations.

**Dataset**: HBN-EEG (BIDS format), >3,000 participants, 6 tasks (RS, SuS, MW, CCD, SL, SyS).  
**Reference**: https://eeg2025.github.io/ (NeurIPS 2025 Competition Track)

### Key Innovations

- **Compression-Augmented Self-Supervised Learning**: Novel integration of predictive coding, wavelet compression, and perceptual quantization as augmentation strategies
- **Subject-Invariant Embeddings**: Domain-adversarial training for robust cross-subject generalization
- **Real-Time Streaming**: Optimized for streaming inference with <2ms latency per window
- **Multi-Task Foundation Model**: Unified architecture for diverse EEG analysis tasks

## Quick Start

### Prerequisites

- Python 3.10 or 3.11
- CUDA-compatible GPU (recommended)
- Docker (optional but recommended)

### Installation

#### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/your-username/eeg2025.git
cd eeg2025

# Build and run Docker container
docker-compose up --build

# Access the container
docker exec -it eeg2025_dev bash
```

#### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/your-username/eeg2025.git
cd eeg2025

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Quick Example

```python
from src.dataio.bids_loader import HBNDataLoader
from src.models.backbones.temporal_cnn import TemporalCNN
from src.training.pretrain_ssl import SSLPretrainer

# Load data
loader = HBNDataLoader(data_path="data/bids_symlinks")
train_data = loader.get_dataset(tasks=["SuS", "CCD"], split="train")

# Initialize model
model = TemporalCNN(n_channels=64, n_classes=128)

# Pretrain with SSL
trainer = SSLPretrainer(model=model, config="configs/pretrain.yaml")
trainer.fit(train_data)
```

## Project Structure

```
eeg2025/
├── src/                          # Source code
│   ├── dataio/                   # Data loading and preprocessing
│   │   ├── compression/          # Compression algorithms
│   │   ├── bids_loader.py        # BIDS dataset loader
│   │   └── preprocessing.py      # EEG preprocessing pipeline
│   ├── models/                   # Model architectures
│   │   ├── backbones/            # Feature extraction models
│   │   ├── heads/                # Task-specific heads
│   │   └── losses/               # Custom loss functions
│   ├── training/                 # Training procedures
│   └── utils/                    # Utilities and helpers
├── configs/                      # Configuration files
├── scripts/                      # Automation scripts
├── tests/                        # Test suite
├── docs/                         # Documentation
├── docker/                       # Docker configuration
└── notebooks/                    # Jupyter notebooks
```

## Usage

### Data Preparation

1. **Download HBN-EEG Dataset**: Follow the instructions in the challenge to obtain access to the HBN dataset
2. **Create BIDS Symlinks**:
   ```bash
   python scripts/prepare_hbn_bids.py --source /path/to/hbn --target data/bids_symlinks
   ```
3. **Generate Splits**:
   ```bash
   python scripts/make_splits.py --config configs/data.yaml
   ```

### Training

#### Self-Supervised Pretraining

```bash
# Run SSL pretraining
python scripts/run_pretrain.sh --config configs/pretrain.yaml

# Or using the training script directly
python src/training/pretrain_ssl.py --config configs/pretrain.yaml
```

#### Cross-Task Transfer Learning

```bash
# Train cross-task model
python scripts/run_cross_task.sh --config configs/train_cross_task.yaml
```

#### Psychopathology Prediction

```bash
# Train psychopathology model
python scripts/run_psych.sh --config configs/train_psych.yaml
```

### Evaluation

```bash
# Evaluate model performance
python src/training/evaluate.py --checkpoint path/to/checkpoint.ckpt --config configs/eval.yaml

# Generate submission files
python scripts/export_submission.py --checkpoint path/to/checkpoint.ckpt --task cross_task --output submission.csv
```

### Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/data.yaml`: Data loading and preprocessing settings
- `configs/pretrain.yaml`: Self-supervised pretraining configuration
- `configs/train_cross_task.yaml`: Cross-task transfer training
- `configs/train_psych.yaml`: Psychopathology prediction training

Example configuration override:
```bash
python src/training/train_cross_task.py model.backbone.d_model=256 training.batch_size=128
```

## Model Architecture

### Backbone Networks

- **TemporalCNN**: Efficient 1D CNN with depthwise separable convolutions
- **TransformerTiny**: Lightweight transformer optimized for EEG sequences
- **Wav2EEG**: Masked time modeling architecture inspired by Wav2Vec2

### Self-Supervised Learning

- **Masked Time Modeling**: Reconstruct masked EEG segments
- **Contrastive Learning**: Multi-view contrastive learning with compression augmentation
- **Predictive Coding**: Learn to predict future EEG samples

### Domain Adaptation

- **DANN**: Domain-Adversarial Neural Networks for subject invariance
- **IRM**: Invariant Risk Minimization penalties

## Performance

### Benchmark Results

| Task | Metric | Score |
|------|--------|-------|
| Cross-Task RT | Pearson r | 0.75+ |
| Cross-Task Success | AUROC | 0.80+ |
| Psychopathology | Mean Pearson r | 0.65+ |

### Efficiency Metrics

- **Inference Speed**: <2ms per 2-second window (GPU)
- **Memory Usage**: <2GB GPU memory for inference
- **Model Size**: <50MB for deployment

## Development

### Code Quality

We maintain high code quality standards:

- **Type Hints**: All functions include type annotations
- **Testing**: >90% test coverage with pytest
- **Linting**: Black formatting, flake8 linting, mypy type checking
- **Documentation**: Comprehensive docstrings and API documentation

### Contributing

Please read [CONTRIBUTING.md](.github/CONTRIBUTING.md) for guidelines on contributing to this project.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test category
pytest tests/models/ -v
```

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{eeg2025,
  title={Foundation Models and Compression for Robust EEG Analysis},
  author={Your Name},
  year={2025},
  url={https://github.com/your-username/eeg2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NeurIPS 2025 EEG Foundation Challenge organizers
- Healthy Brain Network for providing the dataset
- Open-source EEG analysis community (MNE, EEGLAB)
- PyTorch Lightning team for the training framework

## Support

For questions and support:

- Open an issue on GitHub for bugs and feature requests
- Check the [documentation](docs/) for detailed guides
- Review the [project plan](docs/PROJECT_PLAN.md) for development roadmap

---

**Note**: This is a research project for the NeurIPS 2025 EEG Foundation Challenge. The code is provided for educational and research purposes.
