#!/bin/bash
while true; do
  clear
  echo "=== Training Status at $(date) ==="
  echo
  echo "Challenge 1:"
  systemctl --user status challenge1-training.service 2>/dev/null | grep -E '(Active|Memory|CPU|Main PID)' || echo "Not running"
  echo "Log size: $(wc -l logs/train_c1_robust_hybrid.log 2>/dev/null | cut -d' ' -f1) lines"
  tail -3 logs/train_c1_robust_hybrid.log 2>/dev/null | grep -E '(Preprocessing|Creating|Extracting|trials|epoch|loss)' || tail -1 logs/train_c1_robust_hybrid.log 2>/dev/null
  echo
  echo "Challenge 2:"
  systemctl --user status challenge2-training.service 2>/dev/null | grep -E '(Active|Memory|CPU|Main PID)' || echo "Not running"
  echo "Log size: $(wc -l logs/train_c2_robust_hybrid.log 2>/dev/null | cut -d' ' -f1) lines"
  tail -3 logs/train_c2_robust_hybrid.log 2>/dev/null | grep -E '(Preprocessing|Creating|Extracting|trials|epoch|loss)' || tail -1 logs/train_c2_robust_hybrid.log 2>/dev/null
  echo
  echo "Press Ctrl+C to exit"
  sleep 10
done
