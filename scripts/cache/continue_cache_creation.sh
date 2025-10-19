#!/bin/bash
# Continue cache creation for R3, R4, R5 in tmux

SESSION_NAME="cache_creation"
LOG_FILE="logs/cache_creation_continued.log"

echo "════════════════════════════════════════════════════════════════"
echo "📦 Continuing Cache Creation (R3, R4, R5)"
echo "════════════════════════════════════════════════════════════════"

# Check existing files
echo "Checking existing cache files..."
ls -lh data/cached/challenge2_*.h5 2>/dev/null

echo ""
echo "Missing releases: R3, R4, R5"
echo ""

# Kill existing cache session if any
tmux kill-session -t $SESSION_NAME 2>/dev/null

# Create Python script to continue
cat > continue_cache.py << 'PYTHON_EOF'
#!/usr/bin/env python3
"""Continue cache creation for remaining releases"""

import sys
sys.path.append('.')

from create_challenge2_cache import create_cache_for_release

print("Starting cache creation for R3, R4, R5...")
print("=" * 80)

releases = ['R3', 'R4', 'R5']

for release in releases:
    print(f"\n{'='*80}")
    print(f"Processing {release}")
    print(f"{'='*80}\n")
    
    try:
        create_cache_for_release(release, is_validation=(release == 'R5'))
        print(f"\n✅ {release} complete!")
    except Exception as e:
        print(f"\n❌ {release} failed: {e}")
        import traceback
        traceback.print_exc()
        continue

print("\n" + "="*80)
print("✅ Cache creation complete!")
print("="*80)
PYTHON_EOF

chmod +x continue_cache.py

# Launch in tmux
echo "Launching in tmux session '$SESSION_NAME'..."
tmux new-session -d -s $SESSION_NAME
tmux send-keys -t $SESSION_NAME "cd $(pwd)" C-m
tmux send-keys -t $SESSION_NAME "python3 continue_cache.py 2>&1 | tee $LOG_FILE" C-m

echo ""
echo "✅ Cache creation started in tmux!"
echo ""
echo "Monitor with:"
echo "  tmux attach -t $SESSION_NAME"
echo "  tail -f $LOG_FILE"
echo ""
echo "Check progress:"
echo "  ls -lh data/cached/challenge2_*.h5"
echo ""

