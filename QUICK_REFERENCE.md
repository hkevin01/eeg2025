# EEG2025 Quick Reference Guide

**One-Page Summary for Daily Use**

---

## 🎯 Current Status (October 13, 2025)

✅ **COMPLETED**
- CI/CD pipeline (all 3 jobs passing)
- Core model architectures (EEGTransformer, task adapters, DANN)
- Training infrastructure (Challenge 1 & 2 trainers)
- Data loading pipeline (HBNDataset, official splits)
- Augmentation suite (8 techniques)
- GPU optimization (Triton kernels, mixed precision)

⚠️ **CRITICAL GAPS**
- No actual HBN data yet
- Test coverage: 1% (15/1417 files)
- Inference speed not validated
- Artifact handling minimal
- Cross-site validation not done

---

## 📋 Today's Todo (Update Daily)

```markdown
### Current Sprint: Week 0 (Pre-competition setup)
- [ ] Task 1: _______________________
- [ ] Task 2: _______________________
- [ ] Task 3: _______________________

### Blockers
- None

### Tomorrow's Plan
1. _______________________
2. _______________________
3. _______________________
```

---

## 🚀 Top 5 Immediate Actions

1. **Get Data** - Download HBN-EEG dataset (~500GB)
   ```bash
   # Register at: https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/
   ```

2. **Write Tests** - Achieve 30% test coverage minimum
   ```bash
   # Priority tests: data_loading, model_forward, challenge1_metrics,
   #                 challenge2_metrics, inference_speed
   ```

3. **Validate Pipeline** - Run end-to-end on real data
   ```bash
   python scripts/validate_pipeline.py --data_root data/hbn --quick_test
   ```

4. **Measure Baseline** - Train simple model for comparison
   ```bash
   python scripts/train_challenge1.py --quick_baseline --max_epochs 5
   ```

5. **Check Inference Speed** - Verify <50ms requirement
   ```bash
   python tests/test_inference_speed.py --model checkpoints/baseline.pt
   ```

---

## 📊 Competition Targets

### Challenge 1 (Cross-Task Transfer: SuS → CCD)
| Metric | MVP | Competitive | Winning |
|--------|-----|-------------|---------|
| RT Correlation | 0.40 | 0.60 | 0.70 |
| Success AUROC | 0.65 | 0.75 | 0.85 |
| **Combined Score** | **>0.50** | **>0.60** | **>0.70** |

### Challenge 2 (Psychopathology Prediction)
| Metric | MVP | Competitive | Winning |
|--------|-----|-------------|---------|
| Avg Correlation | 0.20 | 0.35 | 0.40 |
| Binary AUROC | 0.65 | 0.75 | 0.80 |

---

## 🔧 Common Commands

### Training
```bash
# Challenge 1 (SuS → CCD transfer)
python scripts/train_challenge1.py \
    --data_root data/hbn \
    --output_dir checkpoints/c1 \
    --max_epochs 50

# Challenge 2 (Psychopathology)
python scripts/train_challenge2.py \
    --data_root data/hbn \
    --output_dir checkpoints/c2 \
    --use_age_norm \
    --irm_penalty 0.1
```

### Testing
```bash
# Run all tests
pytest tests/ -v --cov=src

# Run specific test
pytest tests/test_data_loading.py -v

# Check test coverage
pytest --cov=src --cov-report=html
```

### Evaluation
```bash
# Evaluate on validation set
python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --split val \
    --output results/val_predictions.csv

# Generate submission file
python scripts/generate_submission.py \
    --checkpoint checkpoints/best.pt \
    --output submissions/challenge1_submission.csv
```

### Debugging
```bash
# Check for errors
python -m src.utils.get_errors

# Profile training speed
python scripts/profile_training.py --model_config config/challenge1.yaml

# Validate data quality
python scripts/data_quality_report.py --data_root data/hbn
```

---

## 🐛 Common Issues & Fixes

### Issue: "CUDA out of memory"
**Fix**: Reduce batch size or enable gradient checkpointing
```python
# In config
batch_size: 16  # Reduce from 32
gradient_checkpointing: true
```

### Issue: "Test coverage too low"
**Fix**: Add more unit tests for critical functions
```bash
pytest --cov=src --cov-report=term-missing  # See what's missing
```

### Issue: "Inference too slow (>50ms)"
**Fix**: Apply quantization and optimize preprocessing
```bash
python scripts/quantize_model.py --model checkpoints/best.pt --dtype int8
```

### Issue: "Poor cross-site generalization"
**Fix**: Increase domain adaptation weight
```python
# In config
domain_loss_weight: 0.5  # Increase from 0.1
```

### Issue: "Data loading bottleneck"
**Fix**: Increase num_workers and use pinned memory
```python
# In data loader
num_workers: 8  # Increase from 4
pin_memory: true
persistent_workers: true
```

---

## 📁 Key File Locations

### Models
- `src/models/backbone/eeg_transformer.py` - Core transformer
- `src/models/foundation_model.py` - Main model
- `src/models/challenge1_model.py` - Challenge 1 specific
- `src/models/challenge2_model.py` - Challenge 2 specific

### Training
- `src/training/challenge1_trainer.py` - Challenge 1 trainer
- `src/training/challenge2_trainer.py` - Challenge 2 trainer
- `scripts/train_challenge1.py` - CLI for Challenge 1
- `scripts/train_challenge2.py` - CLI for Challenge 2

### Data
- `src/dataio/hbn_dataset.py` - HBN dataset loader
- `src/dataio/preprocessing.py` - EEG preprocessing
- `src/utils/augmentations.py` - Data augmentation

### Evaluation
- `src/evaluation/challenge1_metrics.py` - Official metrics C1
- `src/evaluation/challenge2_metrics.py` - Official metrics C2
- `scripts/evaluate.py` - Evaluation script
- `scripts/generate_submission.py` - Submission generator

### Configuration
- `config/challenge1.yaml` - Challenge 1 config
- `config/challenge2.yaml` - Challenge 2 config
- `config/ssl_pretraining.yaml` - SSL config

---

## 📚 Key Architecture Details

### Model Architecture
```
Input: (batch, 128 channels, 1000 timesteps)  # 2 seconds at 500Hz
  ↓
Channel Projection: (128 → 768)
  ↓
Positional Encoding: Sinusoidal
  ↓
Transformer: 12 layers × 12 heads × 768d
  ↓
Task-Aware Adapters: FiLM + LoRA
  ↓
Domain Adaptation: Multi-adversary DANN
  ↓
Prediction Heads: Task-specific
  ↓
Output: Task predictions + domain predictions
```

### Augmentation Pipeline (SSL)
1. **TimeMasking** - Mask random time segments
2. **CompressionDistortion** - Perceptual compression
3. **ChannelDropout** - Drop random channels
4. **WaveletDistortion** - Wavelet-domain noise
5. **FrequencyMasking** - Mask frequency bands
6. **TemporalJitter** - Small time shifts
7. **PerceptualQuantization** - Psychoacoustic quantization
8. **GaussianNoise** - Additive white noise

---

## 🎯 Daily Workflow

### Morning (Planning)
1. Review yesterday's progress
2. Update IMPLEMENTATION_ROADMAP.md
3. Identify today's top 3 priorities
4. Check for blockers

### Midday (Execution)
5. Work on priority tasks
6. Run tests frequently
7. Commit code regularly
8. Monitor training if running

### Evening (Review)
9. Review today's accomplishments
10. Update progress tracking
11. Plan tomorrow's tasks
12. Backup important checkpoints

---

## 📞 Emergency Contacts & Resources

### Documentation
- Main README: `README.md`
- Analysis: `CHALLENGE_ANALYSIS_AND_RECOMMENDATIONS.md`
- Roadmap: `IMPLEMENTATION_ROADMAP.md`

### External Resources
- HBN Dataset: https://fcon_1000.projects.nitrc.org/indi/cmi_healthy_brain_network/
- MNE Documentation: https://mne.tools/stable/index.html
- PyTorch Docs: https://pytorch.org/docs/stable/index.html
- Competition Platform: [Add link when available]

### Useful Commands
```bash
# Quick validation of entire pipeline
python scripts/validate_installation.py

# Check CI/CD status
gh workflow view

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f logs/training.log

# Backup checkpoints
rsync -av checkpoints/ backup/checkpoints/
```

---

## 💡 Pro Tips

1. **Always test on small subset first** before full training
   ```bash
   --quick_test --max_samples 100
   ```

2. **Use wandb for experiment tracking**
   ```bash
   wandb login
   wandb init --project eeg2025
   ```

3. **Save checkpoints frequently** (every epoch during development)
   ```python
   save_every_n_epochs: 1  # Change to 5 for production
   ```

4. **Monitor validation metrics closely** to detect overfitting early

5. **Keep a training log** with key decisions and observations

6. **Run inference tests** after every significant model change

7. **Use tmux/screen** for long training runs
   ```bash
   tmux new -s training
   # Detach: Ctrl+b, then d
   # Reattach: tmux attach -t training
   ```

8. **Automate repetitive tasks** with bash scripts

9. **Document unexpected behaviors** immediately

10. **Celebrate small wins** - building ML systems is hard!

---

**Last Updated**: October 13, 2025
**Next Review**: Daily

---

*Keep this document open during development for quick reference!*
