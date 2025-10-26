#!/bin/bash
set -e

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║       🧠 EEG Challenge 2025 - TMUX Training Session                 ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""

# Check if tmux is installed
if ! command -v tmux &> /dev/null; then
    echo "❌ tmux not found. Installing..."
    sudo apt-get update && sudo apt-get install -y tmux
fi

# Kill any existing training sessions
tmux kill-session -t eeg_c1_train 2>/dev/null || true
tmux kill-session -t eeg_c2_train 2>/dev/null || true

# Create logs directory
mkdir -p logs/training_$(date +%Y%m%d)

echo "🚀 Starting Challenge 1 Training in tmux..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start Challenge 1 in tmux
tmux new-session -d -s eeg_c1_train -n "C1_Training" \
    "cd /home/kevin/Projects/eeg2025 && \
     source venv_cpu/bin/activate && \
     export OMP_NUM_THREADS=12 && \
     export MKL_NUM_THREADS=12 && \
     python training/train_c1_sam_simple.py \
         --device cpu \
         --epochs 50 \
         --batch-size 32 \
         --lr 0.001 \
         --rho 0.05 \
         --exp-name sam_c1_tmux_$(date +%Y%m%d) \
         2>&1 | tee logs/training_$(date +%Y%m%d)/c1_tmux.log; \
     echo ''; \
     echo '✅ Challenge 1 Complete! Press ENTER to exit or wait for C2...'; \
     read"

echo "✅ Challenge 1 started in tmux session: eeg_c1_train"
echo ""

echo "🚀 Starting Challenge 2 Training in tmux..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Start Challenge 2 in tmux
tmux new-session -d -s eeg_c2_train -n "C2_Training" \
    "cd /home/kevin/Projects/eeg2025 && \
     source venv_cpu/bin/activate && \
     export OMP_NUM_THREADS=12 && \
     export MKL_NUM_THREADS=12 && \
     python training/train_c2_sam_real_data.py \
         --device cpu \
         --epochs 20 \
         --batch-size 8 \
         --lr 0.001 \
         --rho 0.05 \
         --exp-name sam_c2_tmux_$(date +%Y%m%d) \
         2>&1 | tee logs/training_$(date +%Y%m%d)/c2_tmux.log; \
     echo ''; \
     echo '✅ Challenge 2 Complete! Press ENTER to exit...'; \
     read"

echo "✅ Challenge 2 started in tmux session: eeg_c2_train"
echo ""

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║                   🎉 Both Trainings Started!                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "📺 View Training Sessions:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Challenge 1:  tmux attach -t eeg_c1_train"
echo "  Challenge 2:  tmux attach -t eeg_c2_train"
echo ""
echo "  Detach: Ctrl+B, then D"
echo "  Kill: tmux kill-session -t <session_name>"
echo ""
echo "📊 Check Status:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  List sessions:   tmux ls"
echo "  View C1 log:     tail -f logs/training_$(date +%Y%m%d)/c1_tmux.log"
echo "  View C2 log:     tail -f logs/training_$(date +%Y%m%d)/c2_tmux.log"
echo ""
echo "💡 Training persists even if VSCode crashes or terminal closes!"
echo ""
