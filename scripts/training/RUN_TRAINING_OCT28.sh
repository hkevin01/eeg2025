#!/bin/bash
# Training Script - ALL R-Sets with Random Split
# Run in background with nohup

cd /home/kevin/Projects/eeg2025
source venv_cpu/bin/activate

echo "Starting training at $(date)"
echo "Logs will be saved to: logs/train_c1_all_rsets_$(date +%Y%m%d_%H%M%S).log"

nohup python scripts/experiments/train_c1_all_rsets.py > logs/train_all_rsets_$(date +%Y%m%d_%H%M%S).log 2>&1 &

PID=$!
echo "Training started with PID: $PID"
echo "Monitor with: tail -f logs/train_all_rsets_*.log"
echo "Or: ps aux | grep $PID"
