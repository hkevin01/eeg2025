#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║     🧠 EEG Challenge Training - Robust TMUX Launcher                ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Kill any existing sessions
tmux kill-session -t eeg_c1_train 2>/dev/null || true
tmux kill-session -t eeg_c2_train 2>/dev/null || true

# Create logs directory
mkdir -p logs/training_20251026

# Get absolute paths
VENV_PATH="/home/kevin/Projects/eeg2025/venv_cpu"
PROJECT_PATH="/home/kevin/Projects/eeg2025"

echo "🚀 Starting Challenge 1 in tmux..."
tmux new-session -d -s eeg_c1_train -c "$PROJECT_PATH" \
  "source $VENV_PATH/bin/activate && \
   export OMP_NUM_THREADS=6 && \
   export MKL_NUM_THREADS=6 && \
   echo 'Challenge 1 Training Started' && \
   python training/train_c1_sam_simple.py \
     --device cpu \
     --epochs 50 \
     --batch-size 32 \
     --lr 0.001 \
     --rho 0.05 \
     --exp-name sam_c1_final_$(date +%H%M) \
   2>&1 | tee logs/training_20251026/c1_final.log; \
   echo ''; \
   echo '========================================'; \
   echo 'Challenge 1 COMPLETE - Check log above'; \
   echo '========================================'; \
   exec bash"

sleep 2

echo "🚀 Starting Challenge 2 in tmux..."
tmux new-session -d -s eeg_c2_train -c "$PROJECT_PATH" \
  "source $VENV_PATH/bin/activate && \
   export OMP_NUM_THREADS=6 && \
   export MKL_NUM_THREADS=6 && \
   pip install eegdash --quiet && \
   echo 'Challenge 2 Training Started' && \
   python training/train_c2_sam_real_data.py \
     --device cpu \
     --epochs 20 \
     --batch-size 8 \
     --lr 0.001 \
     --rho 0.05 \
     --exp-name sam_c2_final_$(date +%H%M) \
   2>&1 | tee logs/training_20251026/c2_final.log; \
   echo ''; \
   echo '========================================'; \
   echo 'Challenge 2 COMPLETE - Check log above'; \
   echo '========================================'; \
   exec bash"

sleep 2

echo ""
echo "✅ Both training sessions launched in tmux!"
echo ""
echo "📊 Sessions created:"
tmux ls

echo ""
echo "🔍 View training:"
echo "   Challenge 1: tmux attach -t eeg_c1_train"
echo "   Challenge 2: tmux attach -t eeg_c2_train"
echo "   (Press Ctrl+B then D to detach)"
echo ""
echo "📝 View logs:"
echo "   tail -f logs/training_20251026/c1_final.log"
echo "   tail -f logs/training_20251026/c2_final.log"
echo ""
echo "⏱  Monitoring script: ./monitor_tmux.sh"
echo ""
