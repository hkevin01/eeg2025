#!/bin/bash
# Restart training with Hybrid approach: CPU data loading + GPU training

echo "🔄 Restarting training with Hybrid CPU/GPU approach..."
echo "   Data loading: CPU (no ROCm issues)"
echo "   Training: GPU (4-5x faster)"
echo ""

# Activate virtual environment
source venv/bin/activate

# Kill any existing training processes
pkill -f "train_challenge.*_robust_gpu"
sleep 2

# Clear old logs
mkdir -p logs
> logs/train_c1_robust_hybrid.log
> logs/train_c2_robust_hybrid.log

# Start Challenge 1 training
echo "Starting Challenge 1 training..."
nohup python scripts/train_challenge1_robust_gpu.py > logs/train_c1_robust_hybrid.log 2>&1 &
C1_PID=$!
echo "  PID: $C1_PID"
echo "  Log: logs/train_c1_robust_hybrid.log"

# Wait a bit before starting Challenge 2
sleep 5

# Start Challenge 2 training
echo "Starting Challenge 2 training..."
nohup python scripts/train_challenge2_robust_gpu.py > logs/train_c2_robust_hybrid.log 2>&1 &
C2_PID=$!
echo "  PID: $C2_PID"
echo "  Log: logs/train_c2_robust_hybrid.log"

echo ""
echo "✅ Training started!"
echo ""
echo "Monitor with:"
echo "  tail -f logs/train_c1_robust_hybrid.log"
echo "  tail -f logs/train_c2_robust_hybrid.log"
echo "  bash monitor_training_enhanced.sh"
echo ""
echo "Check GPU usage:"
echo "  rocm-smi"
echo "  watch -n 2 rocm-smi"
