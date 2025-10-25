#!/bin/bash

echo "🔄 Restarting SAM Training - Both Challenges"
echo "="*60

# Kill any existing sessions
tmux kill-session -t sam_c1 2>/dev/null
tmux kill-session -t sam_c2 2>/dev/null

# Clean old logs
mv training_sam_c1.log training_sam_c1_old.log 2>/dev/null
mv training_sam_c2.log training_sam_c2_old.log 2>/dev/null

echo ""
echo "🧠 Starting Challenge 1 (CompactCNN) on GPU..."
tmux new-session -d -s sam_c1 "cd /home/kevin/Projects/eeg2025 && export PYTORCH_ROCM_ARCH=gfx1010 && export HSA_OVERRIDE_GFX_VERSION=10.1.0 && python -u train_c1_sam_simple.py --data-dir data/ds005506-bdf data/ds005507-bdf --epochs 50 --batch-size 32 --lr 1e-3 --device cuda --max-subjects 50 2>&1 | tee training_sam_c1.log"

echo "✅ C1 started in tmux session: sam_c1"
echo "   Using LIMITED dataset (50 subjects) for faster testing"
echo ""

sleep 3

echo "📊 Checking C1 startup..."
tmux capture-pane -t sam_c1 -p | tail -10

echo ""
echo "Commands:"
echo "  Monitor C1: tmux attach -t sam_c1"
echo "  View log: tail -f training_sam_c1.log"
echo "  Detach: Ctrl+B, then D"
echo ""

