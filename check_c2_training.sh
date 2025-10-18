#!/bin/bash
# Quick training progress checker

LOG_FILE=$(ls -t logs/train_c2*.log 2>/dev/null | head -1)

if [ -z "$LOG_FILE" ]; then
    echo "❌ No training log found"
    exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "   🧠 CHALLENGE 2 TCN TRAINING PROGRESS"
echo "════════════════════════════════════════════════════════════════"
echo
echo "📋 Log: $LOG_FILE"
echo "📊 Size: $(du -h "$LOG_FILE" | cut -f1)"
echo

# Check if training is running
if tmux has-session -t eeg_both_challenges 2>/dev/null; then
    echo "✅ Tmux session: ACTIVE"
else
    echo "⚠️  Tmux session: NOT FOUND"
fi
echo

# Get latest progress
echo "📈 Latest Progress:"
tail -3 "$LOG_FILE"
echo

# Check for epoch completion
EPOCHS=$(grep -c "^Epoch" "$LOG_FILE")
echo "🔄 Epochs Started: $EPOCHS"

# Check for validation results
VAL_RESULTS=$(grep -c "Val Loss:" "$LOG_FILE")
echo "✅ Validations Complete: $VAL_RESULTS"

# Get best loss if available
BEST_LOSS=$(grep "Best model" "$LOG_FILE" | tail -1)
if [ -n "$BEST_LOSS" ]; then
    echo "🏆 $BEST_LOSS"
fi

echo
echo "════════════════════════════════════════════════════════════════"
echo "💡 Commands:"
echo "   Watch live: tail -f $LOG_FILE"
echo "   Attach:     tmux attach -t eeg_both_challenges"
echo "════════════════════════════════════════════════════════════════"
