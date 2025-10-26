#!/bin/bash

echo "🎯 Training Monitor"
echo "=" "=" "=" "=" "=" | tr -d ' '

# Check if training is running
if ps aux | grep "train_c1_cached" | grep -v grep > /dev/null; then
    PID=$(ps aux | grep "train_c1_cached" | grep -v grep | awk '{print $2}')
    CPU=$(ps aux | grep "train_c1_cached" | grep -v grep | awk '{print $3}')
    MEM=$(ps aux | grep "train_c1_cached" | grep -v grep | awk '{print $4}')
    RSS=$(ps aux | grep "train_c1_cached" | grep -v grep | awk '{print $6/1024}')
    
    echo "✅ Training ACTIVE"
    echo "   PID: $PID"
    echo "   CPU: ${CPU}%"
    echo "   MEM: ${MEM}% (${RSS}MB)"
    echo ""
    echo "📋 Latest output:"
    echo "---"
    tail -30 logs/c1_cached_training.log 2>/dev/null || echo "Log not ready yet..."
else
    echo "❌ Training NOT running"
    echo ""
    echo "📋 Last log output:"
    echo "---"
    tail -50 logs/c1_cached_training.log 2>/dev/null || echo "No log file found"
fi

