#!/bin/bash
# Comprehensive Infrastructure Status Check

clear
echo "════════════════════════════════════════════════════════════════"
echo "🧠 EEG2025 INFRASTRUCTURE STATUS"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📅 $(date)"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 ACTIVE PROCESSES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
CACHE_PID=$(ps aux | grep create_challenge2_cache | grep -v grep | awk '{print $2}')
if [ ! -z "$CACHE_PID" ]; then
    echo "  ✅ Cache Creation: PID $CACHE_PID (RUNNING)"
    ps aux | grep $CACHE_PID | grep -v grep | awk '{print "     CPU: " $3 "%, MEM: " $4 "%"}'
else
    echo "  ❌ Cache Creation: Not running"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 CACHE FILES STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Challenge 1 Cache (Existing):"
ls -lh data/cached/challenge1_*.h5 2>/dev/null | awk '{print "  ✅ " $9 " (" $5 ")"}'
TOTAL_C1=$(du -sh data/cached/challenge1_*.h5 2>/dev/null | tail -1 | awk '{print $1}')
[ ! -z "$TOTAL_C1" ] && echo "  Total: $TOTAL_C1"
echo ""
echo "Challenge 2 Cache (Creating):"
if [ -f data/cached/challenge2_R1_windows.h5 ]; then
    ls -lh data/cached/challenge2_*.h5 | awk '{print "  ✅ " $9 " (" $5 ")"}'
    TOTAL_C2=$(du -sh data/cached/challenge2_*.h5 | tail -1 | awk '{print $1}')
    echo "  Total: $TOTAL_C2 (In Progress)"
else
    echo "  ⏳ Not yet created (R1 loading)"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🗄️  METADATA DATABASE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f data/metadata.db ]; then
    SIZE=$(ls -lh data/metadata.db | awk '{print $5}')
    echo "  ✅ Database: data/metadata.db ($SIZE)"
    echo ""
    echo "  Tables:"
    sqlite3 data/metadata.db "SELECT name FROM sqlite_master WHERE type='table';" | awk '{print "    • " $0}'
    echo ""
    echo "  Views:"
    sqlite3 data/metadata.db "SELECT name FROM sqlite_master WHERE type='view';" | awk '{print "    • " $0}'
else
    echo "  ❌ Database not created"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 CACHE CREATION LOG (Last 15 Lines)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f logs/cache_creation.log ]; then
    tail -15 logs/cache_creation.log | grep -v "^$"
else
    echo "  ❌ No log file yet"
fi
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 INFRASTRUCTURE SCRIPTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
[ -f create_challenge2_cache.py ] && echo "  ✅ create_challenge2_cache.py" || echo "  ❌ create_challenge2_cache.py"
[ -f create_metadata_database.py ] && echo "  ✅ create_metadata_database.py" || echo "  ❌ create_metadata_database.py"
[ -f train_challenge2_fast.py ] && echo "  ✅ train_challenge2_fast.py" || echo "  ❌ train_challenge2_fast.py"
echo ""

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 EXPECTED vs ACTUAL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Expected Cache Files (Challenge 2):"
echo "  • challenge2_R1_windows.h5 (~800MB)"
echo "  • challenge2_R2_windows.h5 (~800MB)"
echo "  • challenge2_R3_windows.h5 (~1.0GB)"
echo "  • challenge2_R4_windows.h5 (~1.6GB)"
echo "  • challenge2_R5_windows.h5 (~400MB)"
echo "  Total Expected: ~4.6GB"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "🎯 NEXT ACTIONS"
echo "════════════════════════════════════════════════════════════════"
echo ""
if [ ! -z "$CACHE_PID" ]; then
    echo "✓ Cache creation in progress (PID $CACHE_PID)"
    echo "✓ Monitor: ./monitor_cache_creation.sh"
    echo "✓ Wait for completion (~20-40 min total)"
else
    if [ -f data/cached/challenge2_R5_windows.h5 ]; then
        echo "✅ ALL CACHE FILES READY!"
        echo "🚀 Next: python3 train_challenge2_fast.py"
    else
        echo "⚠️  Cache creation not running and files incomplete"
        echo "🔄 Restart: nohup python3 create_challenge2_cache.py > logs/cache_creation.log 2>&1 &"
    fi
fi
echo ""
echo "════════════════════════════════════════════════════════════════"
