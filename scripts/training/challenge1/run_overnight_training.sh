#!/bin/bash
# Automated overnight training pipeline
# Waits for preprocessing, then trains baseline and hybrid models

set -e  # Exit on error

PROJECT_ROOT="/home/kevin/Projects/eeg2025"
cd "$PROJECT_ROOT"

echo "=============================================="
echo "🌙 OVERNIGHT TRAINING PIPELINE"
echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo

# Stage 1: Wait for feature preprocessing to complete
echo "Stage 1: Waiting for feature preprocessing..."
echo "=============================================="

while ps aux | grep -q "[a]dd_neuro_features"; do
    echo "  Preprocessing still running... ($(date '+%H:%M:%S'))"
    python3 /tmp/check_h5_progress.py | tail -2
    sleep 60
done

echo "✅ Preprocessing complete!"
echo

# Verify features were created successfully
echo "Verifying features..."
python3 /tmp/check_h5_progress.py
echo

# Stage 2: Train baseline CNN
echo
echo "Stage 2: Training baseline CNN..."
echo "=============================================="
echo "Started: $(date '+%H:%M:%S')"
echo

nohup python3 scripts/training/challenge1/train_baseline_fast.py > logs/baseline_training_$(date +%Y%m%d_%H%M%S).log 2>&1

echo "✅ Baseline training complete!"
echo "Finished: $(date '+%H:%M:%S')"
echo

# Stage 3: Train hybrid model
echo
echo "Stage 3: Training hybrid model..."
echo "=============================================="
echo "Started: $(date '+%H:%M:%S')"
echo

nohup python3 scripts/training/challenge1/train_hybrid_fast.py > logs/hybrid_training_$(date +%Y%m%d_%H%M%S).log 2>&1

echo "✅ Hybrid training complete!"
echo "Finished: $(date '+%H:%M:%S')"
echo

# Stage 4: Compare results
echo
echo "Stage 4: Final comparison..."
echo "=============================================="

python3 << 'PYEOF'
import torch
from pathlib import Path

PROJECT_ROOT = Path('/home/kevin/Projects/eeg2025')

print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80 + "\n")

# Load baseline
baseline_path = PROJECT_ROOT / 'checkpoints' / 'baseline_best.pth'
if baseline_path.exists():
    baseline = torch.load(baseline_path, map_location='cpu')
    print(f"Baseline CNN:")
    print(f"  Validation NRMSE: {baseline['val_nrmse']:.4f}")
    print(f"  Validation Loss:  {baseline['val_loss']:.4f}")
    print(f"  Epoch: {baseline['epoch']}")
else:
    print("⚠️  Baseline checkpoint not found!")
    baseline = None

print()

# Load hybrid
hybrid_path = PROJECT_ROOT / 'checkpoints' / 'hybrid_best.pth'
if hybrid_path.exists():
    hybrid = torch.load(hybrid_path, map_location='cpu')
    print(f"Hybrid Model:")
    print(f"  Validation NRMSE: {hybrid['val_nrmse']:.4f}")
    print(f"  Validation Loss:  {hybrid['val_loss']:.4f}")
    print(f"  Epoch: {hybrid['epoch']}")
else:
    print("⚠️  Hybrid checkpoint not found!")
    hybrid = None

print("\n" + "="*80)

# Compare
if baseline and hybrid:
    baseline_nrmse = baseline['val_nrmse']
    hybrid_nrmse = hybrid['val_nrmse']
    
    if hybrid_nrmse < baseline_nrmse:
        improvement = ((baseline_nrmse - hybrid_nrmse) / baseline_nrmse) * 100
        print(f"🎉 WINNER: HYBRID MODEL")
        print(f"   Improvement: {improvement:.2f}%")
        print(f"   NRMSE: {baseline_nrmse:.4f} → {hybrid_nrmse:.4f}")
        print()
        print("✅ Neuroscience features helped!")
        print("   Consider updating submission.py with hybrid model")
    else:
        degradation = ((hybrid_nrmse - baseline_nrmse) / baseline_nrmse) * 100
        print(f"🏆 WINNER: BASELINE CNN")
        print(f"   Better by: {degradation:.2f}%")
        print(f"   NRMSE: {hybrid_nrmse:.4f} → {baseline_nrmse:.4f}")
        print()
        print("⚠️  Features didn't improve performance")
        print("   Keep current submission model")
        
print("="*80 + "\n")
PYEOF

echo
echo "=============================================="
echo "🌅 OVERNIGHT TRAINING COMPLETE"
echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="
echo
echo "Summary saved to logs/"
echo "Checkpoints saved to checkpoints/"
echo

