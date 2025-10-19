#!/bin/bash
# Comprehensive Training Dashboard

clear

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║        🧠 EEG2025 Challenge - Training Dashboard              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📅 $(date)"
echo "⏰ Deadline: November 2, 2025 ($(( ($(date -d '2025-11-02' +%s) - $(date +%s)) / 86400 )) days remaining)"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 CACHE STATUS (Challenge 2)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

CACHE_COUNT=0
CACHE_SIZE=0

for release in R1 R2 R3 R4 R5; do
    FILE="data/cached/challenge2_${release}_windows.h5"
    if [ -f "$FILE" ]; then
        SIZE=$(du -h "$FILE" | cut -f1)
        echo "  ✅ $release: $FILE ($SIZE)"
        CACHE_COUNT=$((CACHE_COUNT + 1))
        SIZE_BYTES=$(stat -c%s "$FILE" 2>/dev/null || stat -f%z "$FILE" 2>/dev/null)
        CACHE_SIZE=$((CACHE_SIZE + SIZE_BYTES))
    else
        echo "  ⏳ $release: Not yet created"
    fi
done

if [ $CACHE_COUNT -gt 0 ]; then
    CACHE_SIZE_GB=$(echo "scale=2; $CACHE_SIZE / 1024 / 1024 / 1024" | bc)
    echo ""
    echo "  Total: $CACHE_COUNT/5 files ($CACHE_SIZE_GB GB)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 ACTIVE PROCESSES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Check tmux sessions
if command -v tmux &> /dev/null; then
    SESSIONS=$(tmux ls 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "$SESSIONS" | while read line; do
            echo "  📺 $line"
        done
    else
        echo "  No tmux sessions running"
    fi
else
    echo "  ⚠️  tmux not installed"
fi

echo ""

# Check training processes
TRAIN_PID=$(ps aux | grep train_challenge2_fast.py | grep -v grep | awk '{print $2}')
if [ ! -z "$TRAIN_PID" ]; then
    echo "  🚀 Training: PID $TRAIN_PID"
    ps aux | grep $TRAIN_PID | grep -v grep | awk '{print "     CPU: " $3 "%, MEM: " $4 "%"}'
else
    echo "  ⏸️  Training: Not running"
fi

# Check cache creation
CACHE_PID=$(ps aux | grep continue_cache.py | grep -v grep | awk '{print $2}')
if [ ! -z "$CACHE_PID" ]; then
    echo "  📦 Cache creation: PID $CACHE_PID"
    ps aux | grep $CACHE_PID | grep -v grep | awk '{print "     CPU: " $3 "%, MEM: " $4 "%"}'
else
    echo "  📦 Cache creation: Not running"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🗄️  DATABASE STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -f data/metadata.db ]; then
    DB_SIZE=$(du -h data/metadata.db | cut -f1)
    echo "  ✅ Database: data/metadata.db ($DB_SIZE)"
    
    # Check for training runs
    RUN_COUNT=$(sqlite3 data/metadata.db "SELECT COUNT(*) FROM training_runs;" 2>/dev/null)
    if [ $? -eq 0 ]; then
        echo "  Training runs: $RUN_COUNT"
        
        if [ "$RUN_COUNT" -gt 0 ]; then
            echo ""
            echo "  Latest run:"
            sqlite3 data/metadata.db "SELECT '    ID: ' || run_id || ', Status: ' || status || ', Epochs: ' || COALESCE(total_epochs, 0) || ', Best Val Loss: ' || COALESCE(printf('%.6f', best_val_loss), 'N/A') FROM training_runs ORDER BY run_id DESC LIMIT 1;" 2>/dev/null
        fi
    fi
else
    echo "  ❌ Database not found"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 CHALLENGE STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "  Challenge 1:"
if [ -f "checkpoints/challenge1_tcn_competition_best.pth" ]; then
    SIZE=$(du -h "checkpoints/challenge1_tcn_competition_best.pth" | cut -f1)
    echo "    ✅ Ready ($SIZE)"
else
    echo "    ⚠️  Model not found"
fi

echo ""
echo "  Challenge 2:"
if [ $CACHE_COUNT -eq 5 ]; then
    echo "    ✅ Cache complete (5/5 files)"
    if [ ! -z "$TRAIN_PID" ]; then
        echo "    🔄 Training in progress"
    else
        echo "    ⏳ Ready to train"
    fi
else
    echo "    🔄 Cache creating ($CACHE_COUNT/5 files)"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📋 QUICK ACTIONS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ $CACHE_COUNT -lt 5 ]; then
    echo "  Cache creation:"
    echo "    tmux attach -t cache_creation    # Watch cache creation"
    echo "    tail -f logs/cache_creation_continued.log"
    echo ""
fi

if [ $CACHE_COUNT -eq 5 ] && [ -z "$TRAIN_PID" ]; then
    echo "  ⭐ Ready to start training:"
    echo "    ./launch_training_tmux.sh"
    echo ""
fi

if [ ! -z "$TRAIN_PID" ]; then
    echo "  Training monitoring:"
    echo "    tmux attach -t eeg_challenge2_training"
    echo "    tail -f logs/training_challenge2_tmux.log"
    echo "    sqlite3 data/metadata.db 'SELECT * FROM epoch_history;'"
    echo ""
fi

echo "  Database queries:"
echo "    sqlite3 data/metadata.db 'SELECT * FROM training_runs;'"
echo "    sqlite3 data/metadata.db 'SELECT * FROM best_models;'"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 TIP: Run './dashboard.sh' anytime to see current status"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

