# Guide: Train on Full Dataset

**Goal:** Use all 38,506 EEG windows for maximum performance

## Quick Start

```bash
# 1. Copy script
cp scripts/train_minimal.py scripts/train_full.py

# 2. Edit line ~125: Change max_samples
# From: max_samples=5000
# To:   max_samples=None  # Use all data

# 3. Increase model capacity (optional)
# hidden_dim: 64 → 128
# n_heads: 4 → 8
# n_layers: 2 → 4

# 4. Run (3-5 hours)
tmux new -s training
python3 scripts/train_full.py
# Press Ctrl+B then D to detach
```

## Expected Results

- **Time:** 3-5 hours (vs 28 minutes)
- **Val accuracy:** 60-70% (vs 50%)
- **Transfer learning:** Much better performance on challenges

## Monitoring

```bash
# Watch progress
tail -f logs/full_*.log

# Check resource usage
htop

# Reattach to tmux
tmux attach -t training
```

## After Training

```bash
# Use full model for challenges
sed -i 's/minimal_best.pth/full_best.pth/g' scripts/train_challenge1_simple.py
python3 scripts/train_challenge1_simple.py

# Expected improvement: 2-3x better Pearson r
```

See `TRAINING_COMPLETE.md` for more details.
