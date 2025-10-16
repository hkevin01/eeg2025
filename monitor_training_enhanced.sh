#!/bin/bash

# Colors for better visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Configuration
REFRESH_INTERVAL=10
C1_LOG="logs/challenge1_training_v7_R4val_fixed.log"
C2_LOG="logs/challenge2_training_v9_R4val_fixed.log"

# Function to format time
format_time() {
    local seconds=$1
    if [ $seconds -lt 60 ]; then
        echo "${seconds}s"
    elif [ $seconds -lt 3600 ]; then
        printf "%dm %ds" $((seconds/60)) $((seconds%60))
    else
        printf "%dh %dm" $((seconds/3600)) $(((seconds%3600)/60))
    fi
}

# Function to get process info
get_process_info() {
    local script=$1
    local info=$(ps aux | grep "[p]ython3 scripts/$script" | head -1)
    if [ -z "$info" ]; then
        echo "STOPPED|0|0|0"
    else
        local pid=$(echo "$info" | awk '{print $2}')
        local cpu=$(echo "$info" | awk '{print $3}')
        local mem=$(echo "$info" | awk '{print $4}')
        local time=$(echo "$info" | awk '{print $10}')
        echo "$pid|$cpu|$mem|$time"
    fi
}

# Function to parse NRMSE values
parse_nrmse() {
    local value=$1
    if [ -z "$value" ]; then
        echo "N/A"
    else
        # Extract just the number
        local num=$(echo "$value" | grep -oP '\d+\.\d+' | head -1)
        if [ -z "$num" ]; then
            echo "N/A"
        else
            # Color code based on value
            if (( $(echo "$num < 1.0" | bc -l) )); then
                echo -e "${GREEN}$num${NC}"
            elif (( $(echo "$num < 2.0" | bc -l) )); then
                echo -e "${YELLOW}$num${NC}"
            else
                echo -e "${RED}$num${NC}"
            fi
        fi
    fi
}

# Function to get training progress
get_training_progress() {
    local log_file=$1
    local challenge_name=$2
    
    if [ ! -f "$log_file" ]; then
        echo -e "${RED}❌ Log file not found${NC}"
        return
    fi
    
    # Check if training started
    if grep -q "Epoch 1/50" "$log_file" 2>/dev/null; then
        # Get current epoch
        local current_epoch=$(grep -oP "Epoch \K\d+(?=/50)" "$log_file" | tail -1)
        local total_epochs=50
        
        if [ -z "$current_epoch" ]; then
            echo -e "${YELLOW}🔄 Starting Epoch 1...${NC}"
            return
        fi
        
        # Calculate progress percentage
        local progress=$((current_epoch * 100 / total_epochs))
        local bar_length=30
        local filled=$((progress * bar_length / 100))
        local empty=$((bar_length - filled))
        
        # Create progress bar
        local bar="["
        for ((i=0; i<filled; i++)); do bar+="█"; done
        for ((i=0; i<empty; i++)); do bar+="░"; done
        bar+="]"
        
        # Get NRMSE values
        local train_nrmse=$(grep "Train NRMSE:" "$log_file" | tail -1 | grep -oP '\d+\.\d+')
        local val_nrmse=$(grep "Val NRMSE:" "$log_file" | tail -1 | grep -oP '\d+\.\d+')
        
        # Get best validation NRMSE
        local best_val=$(grep "Best Val NRMSE" "$log_file" | tail -1 | grep -oP '\d+\.\d+')
        
        echo -e "${GREEN}✅ TRAINING${NC}"
        echo -e "   Epoch: ${BOLD}$current_epoch/$total_epochs${NC} ${bar} ${progress}%"
        
        if [ ! -z "$train_nrmse" ]; then
            echo -e "   Train NRMSE: $(parse_nrmse $train_nrmse)"
        fi
        
        if [ ! -z "$val_nrmse" ]; then
            echo -e "   Val NRMSE:   $(parse_nrmse $val_nrmse)"
        fi
        
        if [ ! -z "$best_val" ]; then
            echo -e "   Best Val:    $(parse_nrmse $best_val) ⭐"
        fi
        
        # Estimate time remaining
        if [ $current_epoch -gt 0 ]; then
            local start_time=$(stat -c %Y "$log_file")
            local current_time=$(date +%s)
            local elapsed=$((current_time - start_time))
            local time_per_epoch=$((elapsed / current_epoch))
            local remaining_epochs=$((total_epochs - current_epoch))
            local eta=$((time_per_epoch * remaining_epochs))
            echo -e "   ETA: ${CYAN}$(format_time $eta)${NC}"
        fi
        
    else
        # Still loading data
        local latest=$(tail -5 "$log_file" 2>/dev/null | grep -E "Loading|Creating|Checking|Windows|Total|Std:" | tail -1 | sed 's/^[[:space:]]*//' | cut -c1-80)
        if [ ! -z "$latest" ]; then
            echo -e "${YELLOW}🔄 LOADING DATA${NC}"
            echo -e "   ${latest}..."
            
            # Check for validation statistics (Challenge 2)
            if grep -q "Range:" "$log_file" 2>/dev/null; then
                local train_std=$(grep "Training.*Std:" "$log_file" | tail -1 | grep -oP 'Std: \K\d+\.\d+')
                local val_std=$(grep "Validation.*Std:" "$log_file" | tail -1 | grep -oP 'Std: \K\d+\.\d+')
                
                if [ ! -z "$train_std" ]; then
                    echo -e "   Train Std: ${GREEN}$train_std${NC} ✓"
                fi
                if [ ! -z "$val_std" ]; then
                    if (( $(echo "$val_std > 0.1" | bc -l) )); then
                        echo -e "   Val Std:   ${GREEN}$val_std${NC} ✓ (R4 has variance!)"
                    else
                        echo -e "   Val Std:   ${RED}$val_std${NC} ⚠️  (Zero variance!)"
                    fi
                fi
            fi
        else
            echo -e "${YELLOW}🔄 INITIALIZING...${NC}"
        fi
    fi
}

# Main monitoring loop
while true; do
    clear
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}   🧠 EEG MULTI-RELEASE TRAINING MONITOR${NC}"
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "   $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    
    # Challenge 1
    echo -e "${BOLD}${MAGENTA}📊 CHALLENGE 1: Response Time Prediction${NC}"
    echo -e "${BLUE}   Training: R1, R2, R3 | Validation: R4${NC}"
    
    C1_INFO=$(get_process_info "train_challenge1_multi_release.py")
    IFS='|' read -r c1_pid c1_cpu c1_mem c1_time <<< "$C1_INFO"
    
    if [ "$c1_pid" == "STOPPED" ]; then
        echo -e "   ${RED}❌ NOT RUNNING${NC}"
    else
        echo -e "   PID: ${c1_pid} | CPU: ${c1_cpu}% | MEM: ${c1_mem}% | Runtime: ${c1_time}"
        get_training_progress "$C1_LOG" "Challenge 1"
    fi
    
    echo ""
    
    # Challenge 2
    echo -e "${BOLD}${MAGENTA}📊 CHALLENGE 2: Externalizing Prediction${NC}"
    echo -e "${BLUE}   Training: R1, R2, R3 | Validation: R4 (FIXED!)${NC}"
    
    C2_INFO=$(get_process_info "train_challenge2_multi_release.py")
    IFS='|' read -r c2_pid c2_cpu c2_mem c2_time <<< "$C2_INFO"
    
    if [ "$c2_pid" == "STOPPED" ]; then
        echo -e "   ${RED}❌ NOT RUNNING${NC}"
    else
        echo -e "   PID: ${c2_pid} | CPU: ${c2_cpu}% | MEM: ${c2_mem}% | Runtime: ${c2_time}"
        get_training_progress "$C2_LOG" "Challenge 2"
    fi
    
    echo ""
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}   Refreshing in ${REFRESH_INTERVAL}s | Press Ctrl+C to stop${NC}"
    echo -e "${CYAN}   Tip: Training runs in background - safe to close this monitor${NC}"
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════════${NC}"
    
    sleep $REFRESH_INTERVAL
done
