#!/bin/bash

echo "🧠 EEG Training Monitor - Auto-refresh every 30 seconds"
echo "Press Ctrl+C to stop"
echo ""

while true; do
    clear
    ./monitor_tmux.sh
    echo ""
    echo "⏰ Auto-refresh in 30 seconds... (Ctrl+C to stop)"
    sleep 30
done
