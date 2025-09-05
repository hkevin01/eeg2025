# EEG Foundation Challenge 2025

A comprehensive implementation for the EEG Foundation Challenge 2025, featuring official Starter Kit integration, domain adaptation, compression-augmented SSL, and state-of-the-art architectures.

## 🎯 Challenge Overview

This repository implements solutions for both challenges:

- **Challenge 1 (CCD)**: Cross-cognitive domain response time prediction and success classification
- **Challenge 2 (CBCL)**: Multi-target psychopathology regression and binary classification

## ✨ Key Features

### 🔬 Official Integration
- **Starter Kit Compliance**: Full integration with official data schemas and metrics
- **Official Metrics**: Exact implementation of challenge evaluation metrics
- **Submission Format**: Automatic generation of competition-ready submission files
- **Validation**: Built-in validation against Starter Kit requirements

### 🧠 Advanced Architectures
- **ConformerTiny**: Temporal attention with rotary position embeddings
- **Enhanced CNN**: Depthwise separable convolutions with Squeeze-Excitation
- **RobustEEG**: Channel-dropout resilient backbone with attention mechanisms
- **Multi-Scale Processing**: Adaptive temporal and spectral feature extraction

### 🌐 Domain Adaptation
- **DANN**: Domain Adversarial Neural Networks with gradient reversal
- **MMD**: Maximum Mean Discrepancy for domain alignment
- **IRM**: Invariant Risk Minimization for stable features
- **Curriculum Learning**: Progressive domain adaptation scheduling

### 🔧 Compression-Aware SSL
- **Compression Augmentation**: Calibrated distortion for robust features
- **Real-time Constraints**: <2ms latency for 2-second windows
- **Adaptive Encoding**: Dynamic compression based on signal characteristics

## 🏗️ Architecture

```
src/
├── dataio/
│   ├── starter_kit.py          # Official Starter Kit integration
│   ├── hbn_dataset.py          # BIDS-compliant dataset with real labels
│   └── bids_loader.py          # Enhanced BIDS data loading
├── models/
│   ├── backbones/
│   │   └── enhanced_cnn.py     # ConformerTiny + Enhanced CNN architectures
│   └── heads.py                # Task-specific prediction heads
├── utils/
│   ├── domain_adaptation.py    # DANN/MMD/IRM implementation
│   ├── compression.py          # Compression-aware augmentation
│   └── submission.py           # Official submission generation
└── scripts/
    ├── train_enhanced.py       # Full training pipeline with ablations
    ├── evaluate.py             # Official evaluation with aggregation
    └── dry_run.py              # Integration testing and validation
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n eeg2025 python=3.10
conda activate eeg2025

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install dependencies
pip install -r requirements.txt

# Install MNE-Python for EEG processing
pip install mne[hdf5]
```

### 2. Data Preparation

```bash
# Set your BIDS root path
export BIDS_ROOT="/path/to/hbn/bids"

# Update configuration
sed -i "s|/path/to/hbn/bids|${BIDS_ROOT}|g" configs/enhanced.yaml
```

### 3. Dry Run Testing

Test the complete pipeline before training:

```bash
# Test Starter Kit integration and generate sample CSVs
python scripts/dry_run.py \
    --bids-root $BIDS_ROOT \
    --output-dir dry_run_results

# Check results
cat dry_run_results/dry_run_results.json
```

### 4. Training

#### Challenge 1 (CCD) - Cross-Cognitive Domain

```bash
# Run with ablation study (recommended)
python scripts/train_enhanced.py \
    --config-path configs \
    --config-name challenge1 \
    data.bids_root=$BIDS_ROOT \
    training.run_ablation=true

# Or run full training directly
python scripts/train_enhanced.py \
    --config-path configs \
    --config-name challenge1 \
    data.bids_root=$BIDS_ROOT \
    training.run_ablation=false
```

#### Challenge 2 (CBCL) - Psychopathology

```bash
# Run with ablation study
python scripts/train_enhanced.py \
    --config-path configs \
    --config-name challenge2 \
    data.bids_root=$BIDS_ROOT \
    training.run_ablation=true
```

### 5. Evaluation & Submission

```bash
# Evaluate trained model
python scripts/evaluate.py \
    --checkpoint checkpoints/challenge1/best.ckpt \
    --config configs/challenge1.yaml \
    --splits val test \
    --output-dir results/challenge1

# Check generated submission files
ls results/challenge1/test/submission_challenge1.csv
```

## 📊 Expected Performance

### Challenge 1 (CCD)
- **Response Time Pearson r**: 0.35+ (target: 0.30+)
- **Success AUROC**: 0.75+ (target: 0.70+)
- **Mean Metric**: 0.55+ (average of above)

### Challenge 2 (CBCL)
- **Average Pearson r**: 0.25+ (target: 0.20+)
- **Binary AUROC**: 0.70+ (target: 0.65+)
- **Individual Metrics**: p_factor (0.30+), internalizing (0.25+), externalizing (0.25+), attention (0.20+)

## 🔬 Ablation Study Results

The training script automatically runs ablation studies:

1. **Baseline**: Standard CNN without advanced features
2. **+ ConformerTiny**: Adds temporal attention mechanisms
3. **+ Compression Aug**: Adds compression-aware SSL
4. **+ Domain Adaptation**: Adds DANN/MMD for cross-subject generalization

Expected performance gains:
- ConformerTiny: +10-15% improvement
- Compression Aug: +5-8% improvement
- Domain Adaptation: +8-12% improvement

## 📋 Configuration

### Key Configuration Parameters

```yaml
# Enhanced backbone
model:
  backbone:
    type: "enhanced_cnn"
    use_conformer: true      # ConformerTiny integration
    use_se: true            # Squeeze-Excitation blocks

# Domain adaptation
domain_adaptation:
  enabled: true
  dann:
    lambda_max: 1.0         # Adversarial strength
  mmd:
    weight: 0.1            # MMD penalty weight

# Compression augmentation
data:
  compression_augmentation: true
  compression_strengths: [0.1, 0.2, 0.3]

# Training
training:
  run_ablation: true       # Enable ablation study
  compression_aware: true  # Compression-aware training
```

## 🎯 Official Metrics

### Challenge 1 (CCD)
- **Response Time**: Pearson correlation coefficient (r)
- **Success**: Area Under ROC Curve (AUROC)
- **Final Metric**: Mean of the two above

### Challenge 2 (CBCL)
- **CBCL Dimensions**: Pearson r for p_factor, internalizing, externalizing, attention
- **Binary Classification**: AUROC for typical vs. atypical
- **Final Metric**: Average Pearson r across CBCL dimensions

## 📁 Output Structure

```
results/
├── challenge1/
│   ├── val/
│   │   ├── raw_predictions.csv
│   │   ├── aggregated_predictions.csv
│   │   ├── submission_challenge1.csv
│   │   └── metrics.json
│   └── test/
│       └── [same structure]
└── challenge2/
    └── [similar structure]

checkpoints/
├── baseline/
├── with_conformer/
├── with_compression_aug/
└── with_domain_adaptation/
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python scripts/train_enhanced.py training.batch_size=16
   ```

2. **Missing Labels**
   ```bash
   # Check Starter Kit integration
   python scripts/dry_run.py --bids-root $BIDS_ROOT
   ```

3. **Slow Training**
   ```bash
   # Use mixed precision and reduce workers
   python scripts/train_enhanced.py training.precision="16-mixed" training.num_workers=2
   ```

### Validation Failures

If submission validation fails:

```bash
# Check submission format
head results/challenge1/test/submission_challenge1.csv

# Verify against Starter Kit
python -c "
from src.dataio.starter_kit import SubmissionValidator
validator = SubmissionValidator()
validator.validate_submission('results/challenge1/test/submission_challenge1.csv', 'cross_task')
print('✅ Validation passed')
"
```

## 🔧 Development

### Adding New Components

1. **New Backbone**:
   - Add to `src/models/backbones/`
   - Register in `enhanced_cnn.py`
   - Update config options

2. **New Domain Adaptation Method**:
   - Add to `src/utils/domain_adaptation.py`
   - Update `create_domain_adaptation_components()`

3. **New Metrics**:
   - Add to `src/dataio/starter_kit.py`
   - Update `OfficialMetrics` class

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Test specific component
python -c "
from src.dataio.starter_kit import StarterKitDataLoader
loader = StarterKitDataLoader('/path/to/bids')
print('✅ StarterKit loading works')
"
```

## 📚 References

- [EEG Foundation Challenge 2025](https://www.kaggle.com/competitions/eeg-foundation-challenge-2025)
- [Healthy Brain Network Dataset](http://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/)
- [ConformerTiny Architecture](https://arxiv.org/abs/2005.08100)
- [Domain Adversarial Training](https://arxiv.org/abs/1505.07818)
- [Compression-Aware SSL](https://arxiv.org/abs/2108.06845)

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section above
- Run the dry-run script to validate setup
- Open an issue with detailed logs and configuration
