#!/bin/bash

# Restart robust training on CPU (ROCm compatibility mode)
echo "🔄 Restarting robust training (CPU mode)..."

# Activate venv
source venv/bin/activate

# Kill existing training processes
pkill -f "train_challenge.*_robust" || true
sleep 2

# Clear old logs
rm -f logs/train_c1_robust_cpu.log logs/train_c2_robust_cpu.log

# Start both challenges in background
echo "Starting Challenge 1 (Response Time)..."
nohup python scripts/train_challenge1_robust_cpu.py > logs/train_c1_robust_cpu.log 2>&1 &
C1_PID=$!

echo "Starting Challenge 2 (Externalizing)..."
nohup python scripts/train_challenge2_robust_cpu.py > logs/train_c2_robust_cpu.log 2>&1 &
C2_PID=$!

echo ""
echo "✅ Training started!"
echo "   Challenge 1 PID: $C1_PID"
echo "   Challenge 2 PID: $C2_PID"
echo ""
echo "📊 Monitor progress:"
echo "   bash monitor_training_enhanced.sh"
echo ""
echo "📝 View logs:"
echo "   tail -f logs/train_c1_robust_cpu.log"
echo "   tail -f logs/train_c2_robust_cpu.log"
