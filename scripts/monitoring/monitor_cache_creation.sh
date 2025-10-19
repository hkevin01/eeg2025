#!/bin/bash
# Monitor Cache Creation Progress

watch -n 5 'clear; echo "════════════════════════════════════════════════════════════════"; echo "📦 CACHE CREATION PROGRESS"; echo "════════════════════════════════════════════════════════════════"; echo ""; echo "⏱️  $(date)"; echo ""; ps aux | grep -i create_challenge2_cache | grep -v grep | head -1 || echo "❌ Cache creation not running"; echo ""; echo "📊 Cache Files Created:"; ls -lh data/cached/challenge2_*.h5 2>/dev/null | awk "{print \"  ✅ \" \$9 \" (\" \$5 \")\"}"; echo ""; echo "📝 Last 10 Log Lines:"; tail -10 logs/cache_creation.log | grep -v "^$"; echo ""; echo "════════════════════════════════════════════════════════════════"; echo "Press Ctrl+C to exit"'
