#!/bin/bash
cd /home/kevin/Projects/eeg2025
source activate_sdk.sh

echo "Starting Challenge 1 training..."
python train_challenge1_enhanced.py > training_challenge1_real.log 2>&1 &
PID=$!
echo "Started with PID: $PID"
echo "Monitor with: tail -f training_challenge1_real.log"
