#!/bin/bash
# Quick restart script for Phase 1 training

cd /home/kevin/Projects/eeg2025

# Activate virtual environment
source venv/bin/activate

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║        🚀 RESTARTING PHASE 1 ROBUST TRAINING                    ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Kill any existing processes
pkill -f "train_challenge1_robust_gpu.py" 2>/dev/null
pkill -f "train_challenge2_robust_gpu.py" 2>/dev/null
sleep 2

# Start Challenge 1
echo "Starting Challenge 1 (Response Time Prediction)..."
nohup python scripts/train_challenge1_robust_gpu.py > logs/train_c1_robust_final.log 2>&1 &
C1_PID=$!
echo "  ✓ Challenge 1 PID: $C1_PID"

sleep 2

# Start Challenge 2
echo "Starting Challenge 2 (Externalizing Behavior Prediction)..."
nohup python scripts/train_challenge2_robust_gpu.py > logs/train_c2_robust_final.log 2>&1 &
C2_PID=$!
echo "  ✓ Challenge 2 PID: $C2_PID"

sleep 3

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    ✅ TRAINING STARTED                           ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""
echo "Challenge 1 PID: $C1_PID"
echo "Challenge 2 PID: $C2_PID"
echo ""
echo "📊 Monitor progress with:"
echo "   bash scripts/monitor_training.sh"
echo ""
echo "📝 Check logs:"
echo "   tail -f logs/train_c1_robust_final.log"
echo "   tail -f logs/train_c2_robust_final.log"
echo ""
echo "⏱️  Expected completion: ~2-3 hours"
echo ""
