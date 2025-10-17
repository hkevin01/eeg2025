#!/bin/bash
cd /home/kevin/Projects/eeg2025

# Kill any existing screen sessions
screen -S challenge1 -X quit 2>/dev/null
screen -S challenge2 -X quit 2>/dev/null
sleep 2

echo "🚀 Starting training in screen sessions (VSCode-independent)"

# Start Challenge 1 in detached screen
screen -dmS challenge1 bash -c "
  source venv/bin/activate
  python scripts/train_challenge1_robust_gpu.py > logs/train_c1_robust_hybrid.log 2>&1
"

# Start Challenge 2 in detached screen  
screen -dmS challenge2 bash -c "
  source venv/bin/activate
  python scripts/train_challenge2_robust_gpu.py > logs/train_c2_robust_hybrid.log 2>&1
"

sleep 3

echo "✅ Training started in screen sessions"
echo ""
echo "Sessions:"
screen -ls
echo ""
echo "Monitor:"
echo "  tail -f logs/train_c1_robust_hybrid.log"
echo "  tail -f logs/train_c2_robust_hybrid.log"
echo ""
echo "Attach to session:"
echo "  screen -r challenge1"
echo "  screen -r challenge2"
echo ""
echo "These will survive:"
echo "  ✓ VSCode crashes"
echo "  ✓ Terminal closes"
echo "  ✓ SSH disconnects"
