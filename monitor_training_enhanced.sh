#!/bin/bash
# Enhanced Training Progress Monitor with detailed status tracking

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

REFRESH_INTERVAL=15  # seconds

show_header() {
    clear
    echo -e "${BOLD}================================================================${NC}"
    echo -e "${BOLD}🔍 EEG Training Progress Monitor${NC}"
    echo -e "${BOLD}================================================================${NC}"
    echo -e "Last update: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
}

get_release_progress() {
    local logfile=$1
    local challenge_name=$2
    
    # Count releases loaded/completed
    local releases_info=$(grep -E "Loading release R[0-9]" "$logfile" 2>/dev/null | tail -5)
    local current_release=$(echo "$releases_info" | tail -1 | grep -oP 'R\d+' | head -1)
    local completed_releases=$(grep -E "Complete - [0-9]+ valid" "$logfile" 2>/dev/null | grep -oP 'R\d+' | wc -l)
    
    # Get corruption stats
    local total_corrupted=$(grep -E "Corrupted file" "$logfile" 2>/dev/null | wc -l)
    
    # Check if training has started
    local training_started=$(grep -q "Training Multi-Release Model" "$logfile" 2>/dev/null && echo "yes" || echo "no")
    
    # Get latest epoch info if training
    local epoch_info=""
    local nrmse_info=""
    if [ "$training_started" == "yes" ]; then
        epoch_info=$(grep -E "Epoch [0-9]+/[0-9]+" "$logfile" 2>/dev/null | tail -1 | grep -oP 'Epoch \d+/\d+')
        nrmse_info=$(grep "Val NRMSE:" "$logfile" 2>/dev/null | tail -1)
    fi
    
    # Build status string
    echo "$current_release|$completed_releases|$total_corrupted|$training_started|$epoch_info|$nrmse_info"
}

show_challenge_status() {
    local challenge_num=$1
    local pid=$2
    local logfile=$3
    
    echo -e "${BOLD}📊 CHALLENGE $challenge_num${NC}"
    echo -e "   PID: ${pid:-${RED}Stopped${NC}}"
    
    if [ -z "$pid" ]; then
        # Check if it crashed
        if [ -f "$logfile" ]; then
            local last_error=$(grep -E "FATAL ERROR|Traceback|Error" "$logfile" 2>/dev/null | tail -1)
            if [ -n "$last_error" ]; then
                echo -e "   ${RED}Status: CRASHED${NC}"
                echo -e "   ${RED}Error: ${last_error:0:80}${NC}"
            else
                echo -e "   Status: Stopped"
            fi
        else
            echo -e "   Status: Not started"
        fi
        return
    fi
    
    # Get progress info
    IFS='|' read -r current_release completed_releases corrupted training_started epoch_info nrmse_info <<< "$(get_release_progress "$logfile" "$challenge_num")"
    
    if [ "$training_started" == "yes" ]; then
        echo -e "   ${GREEN}Status: ✅ TRAINING${NC}"
        [ -n "$epoch_info" ] && echo -e "   Progress: $epoch_info"
        [ -n "$nrmse_info" ] && echo -e "   $nrmse_info"
    else
        if [ -n "$current_release" ]; then
            echo -e "   ${YELLOW}Status: 📥 Loading data${NC}"
            echo -e "   Current: $current_release (Release ${completed_releases}/4 completed)"
            echo -e "   Corrupted files skipped: $corrupted"
            
            # Show checking progress
            local checking_progress=$(tail -20 "$logfile" 2>/dev/null | grep "Checked.*files" | tail -1)
            [ -n "$checking_progress" ] && echo -e "   ${CYAN}$checking_progress${NC}"
        else
            echo -e "   ${YELLOW}Status: 🔄 Initializing...${NC}"
        fi
    fi
    
    # Show CPU and memory usage
    if [ -n "$pid" ]; then
        local cpu_mem=$(ps -p $pid -o %cpu,%mem --no-headers 2>/dev/null)
        if [ -n "$cpu_mem" ]; then
            echo -e "   Resources: CPU ${cpu_mem}%"
        fi
    fi
}

# Main monitoring loop
C1_TRAINING_STARTED=0
C2_TRAINING_STARTED=0

while true; do
    # Get PIDs
    C1_PID=$(ps aux | grep "[p]ython3 scripts/train_challenge1_multi_release.py" | awk '{print $2}')
    C2_PID=$(ps aux | grep "[p]ython3 scripts/train_challenge2_multi_release.py" | awk '{print $2}')
    
    # Find most recent log files
    C1_LOG=$(ls -t logs/challenge1_training*.log 2>/dev/null | head -1)
    C2_LOG=$(ls -t logs/challenge2_training*.log 2>/dev/null | head -1)
    
    # Check if both stopped
    if [ -z "$C1_PID" ] && [ -z "$C2_PID" ]; then
        show_header
        echo -e "${YELLOW}⚠️  Both training processes have stopped!${NC}"
        echo ""
        
        # Check for completion
        if grep -q "Training completed successfully" "$C1_LOG" 2>/dev/null && \
           grep -q "Training completed successfully" "$C2_LOG" 2>/dev/null; then
            echo -e "${GREEN}${BOLD}✅ TRAINING COMPLETE!${NC}"
            echo ""
            echo "Results:"
            grep "Best.*NRMSE" "$C1_LOG" 2>/dev/null | tail -1 | sed 's/^/  Challenge 1: /'
            grep "Best.*NRMSE" "$C2_LOG" 2>/dev/null | tail -1 | sed 's/^/  Challenge 2: /'
            echo ""
            echo "Weight files:"
            ls -lh weights_challenge_*_multi_release.pt 2>/dev/null | awk '{print "  "$9" ("$5")"}'
        else
            echo -e "${RED}Check crash logs for details:${NC}"
            ls -t logs/challenge*_crash_*.log 2>/dev/null | head -2 | sed 's/^/  /'
        fi
        break
    fi
    
    # Show status
    show_header
    show_challenge_status "1" "$C1_PID" "$C1_LOG"
    echo ""
    show_challenge_status "2" "$C2_PID" "$C2_LOG"
    echo ""
    echo -e "${BOLD}================================================================${NC}"
    
    # Check if both transitioned to training
    if [ -n "$C1_PID" ] && [ -n "$C2_PID" ]; then
        if grep -q "Training Multi-Release Model" "$C1_LOG" 2>/dev/null && \
           grep -q "Training Multi-Release Model" "$C2_LOG" 2>/dev/null; then
            if [ $C1_TRAINING_STARTED -eq 0 ] && [ $C2_TRAINING_STARTED -eq 0 ]; then
                C1_TRAINING_STARTED=1
                C2_TRAINING_STARTED=1
                echo ""
                echo -e "${GREEN}${BOLD}✅ DATA DOWNLOAD COMPLETE!${NC}"
                echo -e "${GREEN}🎯 Both challenges now TRAINING${NC}"
                echo ""
                echo "Estimated completion: ~14 hours (tomorrow morning)"
                echo "You can close this monitor - training will continue in background"
                echo ""
                echo "To check progress later:"
                echo "  ./monitor_training_enhanced.sh"
                echo "  tail -f $C1_LOG"
                echo "  tail -f $C2_LOG"
                echo ""
            fi
        fi
    fi
    
    echo -e "Refreshing in ${REFRESH_INTERVAL}s... (Ctrl+C to stop)"
    echo -e "${CYAN}Tip: Training runs in background - safe to close this monitor${NC}"
    sleep $REFRESH_INTERVAL
done

echo ""
echo "Monitor stopped at: $(date)"
