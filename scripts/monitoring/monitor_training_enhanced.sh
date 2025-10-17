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

# Auto-detect latest log files
detect_latest_logs() {
    # Find the most recent Challenge 1 log (try multiple patterns)
    C1_LOG=$(ls -t logs/train_c1*.log logs/challenge1*.log 2>/dev/null | head -1)
    if [ -z "$C1_LOG" ]; then
        C1_LOG="logs/challenge1_fresh_start.log"
    fi

    # Find the most recent Challenge 2 log (try multiple patterns)
    C2_LOG=$(ls -t logs/train_c2*.log logs/challenge2*.log 2>/dev/null | head -1)
    if [ -z "$C2_LOG" ]; then
        C2_LOG="logs/challenge2_fresh_start.log"
    fi
}

# Detect logs on startup
detect_latest_logs

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
    # Try exact match first, then fallback to pattern
    local info=$(ps aux | grep "[p]ython.*scripts/$script" | head -1)
    if [ -z "$info" ]; then
        # Try broader patterns for robust/gpu/cpu variants
        info=$(ps aux | grep "[p]ython.*train_challenge.*\.py" | grep -v grep | head -1)
    fi
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

# Function to get detailed GPU info
get_gpu_info() {
    # Try rocm-smi for AMD (prioritize since we're using AMD GPU)
    if command -v rocm-smi &> /dev/null; then
        # Get comprehensive GPU information
        local gpu_output=$(rocm-smi 2>/dev/null)

        # Parse GPU name
        local gpu_name=$(echo "$gpu_output" | grep -oP 'Radeon [^\s]+(\s+[^\s]+)*' | head -1 | sed 's/[[:space:]]*$//')
        if [ -z "$gpu_name" ]; then
            gpu_name="AMD GPU (ROCm)"
        fi

        # Parse GPU utilization (percentage)
        local gpu_util=$(echo "$gpu_output" | grep -oP 'GPU%\s+\n.*?\s+\K\d+(?=%)' | head -1)
        if [ -z "$gpu_util" ]; then
            gpu_util="0"
        fi

        # Parse VRAM utilization
        local vram_used=$(echo "$gpu_output" | grep "VRAM%" | grep -oP '\d+(?=%)')
        if [ -z "$vram_used" ]; then
            vram_used="0"
        fi

        # Get temperature
        local temp=$(echo "$gpu_output" | grep "°C" | grep -oP '\d+\.?\d*(?=°C)' | head -1)
        if [ -z "$temp" ]; then
            temp="N/A"
        fi

        # Get power draw
        local power=$(echo "$gpu_output" | grep "W" | grep -oP '\d+\.?\d*(?=W)' | head -1)
        if [ -z "$power" ]; then
            power="N/A"
        fi

        # Get clock speeds
        local sclk=$(echo "$gpu_output" | grep "SCLK" | grep -oP '\d+Mhz' | head -1)
        local mclk=$(echo "$gpu_output" | grep "MCLK" | grep -oP '\d+Mhz' | head -1)

        echo "AMD ROCm|$gpu_name|$gpu_util|$vram_used|$temp|$power|$sclk|$mclk"
        return
    fi

    # Try nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        local gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        local gpu_mem=$(nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits 2>/dev/null | head -1)
        local gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        local temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
        local power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ ! -z "$gpu_util" ]; then
            echo "NVIDIA|$gpu_name|$gpu_util|$gpu_mem|$temp|$power|N/A|N/A"
            return
        fi
    fi

    # Check if PyTorch can see CUDA
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        local gpu_name=$(python3 -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'Unknown')" 2>/dev/null)
        echo "CUDA|$gpu_name|Active|N/A|N/A|N/A|N/A|N/A"
        return
    fi

    echo "NONE|CPU Only|0|0|N/A|N/A|N/A|N/A"
}

# Function to check if training is using GPU from logs
check_gpu_usage_from_logs() {
    local log_file=$1

    if [ ! -f "$log_file" ]; then
        echo "UNKNOWN"
        return
    fi

    # Check for GPU detection messages
    if grep -q "GPU detected:" "$log_file" 2>/dev/null; then
        local gpu_name=$(grep "GPU detected:" "$log_file" | tail -1 | sed 's/.*GPU detected: //' | cut -d' ' -f1-4)
        echo "GPU|$gpu_name"
    elif grep -q "Device: cuda" "$log_file" 2>/dev/null; then
        echo "GPU|CUDA Device"
    elif grep -q "Mixed Precision (AMP): True" "$log_file" 2>/dev/null; then
        echo "GPU|Mixed Precision Active"
    elif grep -q "No GPU detected" "$log_file" 2>/dev/null || grep -q "Device: cpu" "$log_file" 2>/dev/null; then
        echo "CPU|No GPU"
    else
        echo "UNKNOWN"
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

    # Check if training completed
    if grep -q "✅ TRAINING COMPLETE" "$log_file" 2>/dev/null; then
        local best_val=$(grep "Best validation NRMSE:" "$log_file" | tail -1 | grep -oP '\d+\.\d+')
        local total_time=$(grep "Time:" "$log_file" | tail -1 | awk '{print $2, $3}')
        local model_file=$(grep "Model saved:" "$log_file" | tail -1 | awk '{print $3}')

        echo -e "${GREEN}✅ TRAINING COMPLETE${NC}"
        if [ ! -z "$best_val" ]; then
            echo -e "   ${BOLD}Best Val NRMSE: $(parse_nrmse $best_val)${NC} ⭐"
        fi
        if [ ! -z "$total_time" ]; then
            echo -e "   Training Time: ${CYAN}$total_time${NC}"
        fi
        if [ ! -z "$model_file" ]; then
            echo -e "   Model: ${GREEN}$model_file${NC}"
        fi

        # Check for early stopping
        if grep -q "Early stopping" "$log_file" 2>/dev/null; then
            echo -e "   ${YELLOW}⏹️  Early stopping triggered${NC}"
        fi

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
        echo -e "${YELLOW}🔄 LOADING DATA${NC}"

        # Check for release loading progress
        local loading_r1=$(grep -c "Loading R1" "$log_file" 2>/dev/null)
        local loading_r2=$(grep -c "Loading R2" "$log_file" 2>/dev/null)
        local loading_r3=$(grep -c "Loading R3" "$log_file" 2>/dev/null)
        local total_loaded=$(grep "✅ Total:" "$log_file" 2>/dev/null | tail -1)

        # Show which releases are being loaded
        if [ $loading_r1 -gt 0 ]; then
            if grep -q "Loading R1.*✓" "$log_file" 2>/dev/null; then
                echo -e "   ${GREEN}✓${NC} R1 loaded"
            else
                echo -e "   ${YELLOW}⏳${NC} Loading R1..."
            fi
        fi

        if [ $loading_r2 -gt 0 ]; then
            if grep -q "Loading R2.*✓" "$log_file" 2>/dev/null; then
                echo -e "   ${GREEN}✓${NC} R2 loaded"
            else
                echo -e "   ${YELLOW}⏳${NC} Loading R2..."
            fi
        fi

        if [ $loading_r3 -gt 0 ]; then
            if grep -q "Loading R3.*✓" "$log_file" 2>/dev/null; then
                echo -e "   ${GREEN}✓${NC} R3 loaded"
            else
                echo -e "   ${YELLOW}⏳${NC} Loading R3..."
            fi
        fi

        # Show total if available
        if [ ! -z "$total_loaded" ]; then
            local trial_count=$(echo "$total_loaded" | grep -oP '\d+(?= trials)')
            if [ ! -z "$trial_count" ]; then
                echo -e "   ${CYAN}📊 Total: ${BOLD}$trial_count${NC}${CYAN} trials${NC}"
            fi
        fi

        # Check for splitting message
        if grep -q "Splitting dataset" "$log_file" 2>/dev/null; then
            local train_samples=$(grep "Train:" "$log_file" | tail -1 | grep -oP '\d+(?= samples)')
            local val_samples=$(grep "Val:" "$log_file" | tail -1 | grep -oP '\d+(?= samples)')
            if [ ! -z "$train_samples" ]; then
                echo -e "   ${CYAN}📊 Train: ${train_samples} | Val: ${val_samples}${NC}"
            fi
            echo -e "   ${YELLOW}⏳${NC} Preparing DataLoaders..."
        fi

        # Show latest activity (last non-empty line with relevant info)
        local latest=$(tail -10 "$log_file" 2>/dev/null | grep -E "Loading|Creating|Checking|Windows|Preprocessing|Applying" | tail -1 | sed 's/^[[:space:]]*//' | cut -c1-70)
        if [ ! -z "$latest" ]; then
            echo -e "   ${DIM}${latest}...${NC}"
        fi

        # Check for data statistics
        if grep -q "Range:" "$log_file" 2>/dev/null; then
            # Get the most recent range/std info
            local data_range=$(grep "Range:" "$log_file" | tail -1 | grep -oP 'Range: \[\K[^\]]+')
            local data_std=$(grep "Std:" "$log_file" | tail -1 | grep -oP 'Std: \K\d+\.\d+')
            local data_mean=$(grep "Mean:" "$log_file" | tail -1 | grep -oP 'Mean: \K[-\d.]+')

            if [ ! -z "$data_range" ]; then
                # Check if range indicates zero variance (same min/max)
                local min=$(echo "$data_range" | cut -d',' -f1 | xargs)
                local max=$(echo "$data_range" | cut -d',' -f2 | xargs)

                if [ "$min" == "$max" ]; then
                    echo -e "   ${RED}⚠️  ZERO VARIANCE DETECTED!${NC}"
                    echo -e "   Range: [$data_range] - All values identical!"
                else
                    echo -e "   ${GREEN}✓ Variance OK${NC} - Range: [$data_range]"
                    if [ ! -z "$data_std" ]; then
                        echo -e "   Mean: $data_mean, Std: ${GREEN}$data_std${NC}"
                    fi
                fi
            fi
        fi

        # Check for R1+R2 combined split info (Challenge 2 fresh)
        if grep -q "Splitting R1+R2" "$log_file" 2>/dev/null; then
            local total=$(grep "Total:" "$log_file" | tail -1 | grep -oP '\d+')
            local train=$(grep "Train:" "$log_file" | tail -1 | grep -oP '\d+')
            local val=$(grep "Val:" "$log_file" | tail -1 | grep -oP '\d+')

            if [ ! -z "$total" ]; then
                echo -e "   ${GREEN}✓ R1+R2 Split${NC}: $total → ${CYAN}$train train${NC} / ${MAGENTA}$val val${NC}"
            fi
        fi
    fi  # Close the "else" block from line 269
}

# Trap Ctrl+C for clean exit
trap 'echo -e "\n${CYAN}Monitor stopped by user. Training continues in background.${NC}"; exit 0' INT

# Main monitoring loop
while true; do
    clear
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}   🧠 EEG MULTI-RELEASE TRAINING MONITOR${NC}"
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "   $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""

    # Re-detect logs in case they changed
    detect_latest_logs

    # Challenge 1
    echo -e "${BOLD}${MAGENTA}📊 CHALLENGE 1: Response Time Prediction${NC}"

    # Detect training strategy from log
    if grep -q "Training: R1, R2" "$C1_LOG" 2>/dev/null; then
        echo -e "${BLUE}   Training: R1+R2 | Validation: R3 ${GREEN}(FRESH)${NC}"
    else
        echo -e "${BLUE}   Training: R1, R2, R3 | Validation: R4${NC}"
    fi
    echo -e "${CYAN}   Log: $C1_LOG${NC}"

    C1_INFO=$(get_process_info "train_challenge1_multi_release.py")
    IFS='|' read -r c1_pid c1_cpu c1_mem c1_time <<< "$C1_INFO"

    if [ "$c1_pid" == "STOPPED" ]; then
        # Check if it completed or just not running
        if grep -q "✅ TRAINING COMPLETE" "$C1_LOG" 2>/dev/null; then
            echo -e "   ${GREEN}✅ COMPLETED${NC}"
        else
            echo -e "   ${RED}❌ NOT RUNNING${NC}"
        fi
    else
        echo -e "   PID: ${c1_pid} | CPU: ${c1_cpu}% | MEM: ${c1_mem}% | Runtime: ${c1_time}"

        # Check what device training is actually using from logs
        IFS='|' read -r log_device log_detail <<< "$(check_gpu_usage_from_logs "$C1_LOG")"

        # Show GPU/CPU device being used
        IFS='|' read -r gpu_type gpu_name gpu_util gpu_mem temp power sclk mclk <<< "$(get_gpu_info)"

        if [ "$log_device" == "GPU" ]; then
            echo -e "   ${GREEN}✅ TRAINING ON GPU:${NC} ${BOLD}$log_detail${NC}"

            if [ "$gpu_type" != "NONE" ]; then
                echo -e "   ${MAGENTA}🎮 $gpu_name${NC}"
                echo -e "   ${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

                # Color code GPU utilization
                if [ ! -z "$gpu_util" ] && [ "$gpu_util" != "N/A" ]; then
                    if [ "$gpu_util" -gt 70 ]; then
                        echo -e "   ${GREEN}GPU Utilization: ${BOLD}${gpu_util}%${NC} ${GREEN}⚡ (High - Good!)${NC}"
                    elif [ "$gpu_util" -gt 30 ]; then
                        echo -e "   ${YELLOW}GPU Utilization: ${BOLD}${gpu_util}%${NC} ${YELLOW}⚡ (Moderate)${NC}"
                    else
                        echo -e "   ${RED}GPU Utilization: ${BOLD}${gpu_util}%${NC} ${RED}⚠️  (Low - Check logs)${NC}"
                    fi
                fi

                # Color code VRAM usage
                if [ ! -z "$gpu_mem" ] && [ "$gpu_mem" != "N/A" ]; then
                    if [ "$gpu_mem" -gt 80 ]; then
                        echo -e "   ${RED}VRAM Usage: ${BOLD}${gpu_mem}%${NC} ${RED}⚠️  (High)${NC}"
                    elif [ "$gpu_mem" -gt 50 ]; then
                        echo -e "   ${YELLOW}VRAM Usage: ${BOLD}${gpu_mem}%${NC} ${YELLOW}(Moderate)${NC}"
                    else
                        echo -e "   ${GREEN}VRAM Usage: ${BOLD}${gpu_mem}%${NC} ${GREEN}(Good)${NC}"
                    fi
                fi

                # Show temperature with color coding
                if [ ! -z "$temp" ] && [ "$temp" != "N/A" ]; then
                    if [ $(echo "$temp > 80" | bc -l 2>/dev/null || echo 0) -eq 1 ]; then
                        echo -e "   ${RED}Temperature: ${BOLD}${temp}°C${NC} ${RED}🔥 (Hot!)${NC}"
                    elif [ $(echo "$temp > 65" | bc -l 2>/dev/null || echo 0) -eq 1 ]; then
                        echo -e "   ${YELLOW}Temperature: ${BOLD}${temp}°C${NC} ${YELLOW}🌡️  (Warm)${NC}"
                    else
                        echo -e "   ${GREEN}Temperature: ${BOLD}${temp}°C${NC} ${GREEN}❄️  (Cool)${NC}"
                    fi
                fi

                # Show power and clocks
                if [ ! -z "$power" ] && [ "$power" != "N/A" ]; then
                    echo -e "   ${CYAN}Power Draw: ${power}W${NC}"
                fi

                if [ ! -z "$sclk" ] && [ "$sclk" != "N/A" ]; then
                    echo -e "   ${CYAN}GPU Clock: $sclk | Memory Clock: $mclk${NC}"
                fi

                echo -e "   ${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            fi
        elif [ "$log_device" == "CPU" ]; then
            echo -e "   ${YELLOW}⚠️  TRAINING ON CPU${NC} (GPU available but not used)"
            if [ "$gpu_type" != "NONE" ]; then
                echo -e "   ${YELLOW}   Available GPU: $gpu_name (idle)${NC}"
            fi
        else
            # Unknown - show system GPU info
            if [ "$gpu_type" != "NONE" ]; then
                echo -e "   ${CYAN}🎮 GPU Available:${NC} $gpu_name"
                echo -e "   ${CYAN}   Util: ${gpu_util}% | VRAM: ${gpu_mem}%${NC}"
            else
                echo -e "   ${YELLOW}💻 Device: CPU Only${NC}"
            fi
        fi

        get_training_progress "$C1_LOG" "Challenge 1"
    fi

    echo ""

    # Challenge 2
    echo -e "${BOLD}${MAGENTA}📊 CHALLENGE 2: Externalizing Prediction${NC}"

    # Detect training strategy from log
    if grep -q "Splitting R1+R2 dataset" "$C2_LOG" 2>/dev/null; then
        echo -e "${BLUE}   Training: R1+R2 80/20 split ${GREEN}(FRESH - Variance Fixed!)${NC}"
    else
        echo -e "${BLUE}   Training: R1, R2, R3 | Validation: R4${NC}"
    fi
    echo -e "${CYAN}   Log: $C2_LOG${NC}"

    C2_INFO=$(get_process_info "train_challenge2_multi_release.py")
    IFS='|' read -r c2_pid c2_cpu c2_mem c2_time <<< "$C2_INFO"

    if [ "$c2_pid" == "STOPPED" ]; then
        # Check if it completed or just not running
        if grep -q "✅ TRAINING COMPLETE" "$C2_LOG" 2>/dev/null; then
            echo -e "   ${GREEN}✅ COMPLETED${NC}"
        else
            echo -e "   ${RED}❌ NOT RUNNING${NC}"
        fi
    else
        echo -e "   PID: ${c2_pid} | CPU: ${c2_cpu}% | MEM: ${c2_mem}% | Runtime: ${c2_time}"

        # Check what device training is actually using from logs
        IFS='|' read -r log_device log_detail <<< "$(check_gpu_usage_from_logs "$C2_LOG")"

        # Show GPU/CPU device being used
        IFS='|' read -r gpu_type gpu_name gpu_util gpu_mem temp power sclk mclk <<< "$(get_gpu_info)"

        if [ "$log_device" == "GPU" ]; then
            echo -e "   ${GREEN}✅ TRAINING ON GPU:${NC} ${BOLD}$log_detail${NC}"

            if [ "$gpu_type" != "NONE" ]; then
                echo -e "   ${MAGENTA}🎮 $gpu_name${NC}"
                echo -e "   ${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

                # Color code GPU utilization
                if [ ! -z "$gpu_util" ] && [ "$gpu_util" != "N/A" ]; then
                    if [ "$gpu_util" -gt 70 ]; then
                        echo -e "   ${GREEN}GPU Utilization: ${BOLD}${gpu_util}%${NC} ${GREEN}⚡ (High - Good!)${NC}"
                    elif [ "$gpu_util" -gt 30 ]; then
                        echo -e "   ${YELLOW}GPU Utilization: ${BOLD}${gpu_util}%${NC} ${YELLOW}⚡ (Moderate)${NC}"
                    else
                        echo -e "   ${RED}GPU Utilization: ${BOLD}${gpu_util}%${NC} ${RED}⚠️  (Low - Check logs)${NC}"
                    fi
                fi

                # Color code VRAM usage
                if [ ! -z "$gpu_mem" ] && [ "$gpu_mem" != "N/A" ]; then
                    if [ "$gpu_mem" -gt 80 ]; then
                        echo -e "   ${RED}VRAM Usage: ${BOLD}${gpu_mem}%${NC} ${RED}⚠️  (High)${NC}"
                    elif [ "$gpu_mem" -gt 50 ]; then
                        echo -e "   ${YELLOW}VRAM Usage: ${BOLD}${gpu_mem}%${NC} ${YELLOW}(Moderate)${NC}"
                    else
                        echo -e "   ${GREEN}VRAM Usage: ${BOLD}${gpu_mem}%${NC} ${GREEN}(Good)${NC}"
                    fi
                fi

                # Show temperature with color coding
                if [ ! -z "$temp" ] && [ "$temp" != "N/A" ]; then
                    if [ $(echo "$temp > 80" | bc -l 2>/dev/null || echo 0) -eq 1 ]; then
                        echo -e "   ${RED}Temperature: ${BOLD}${temp}°C${NC} ${RED}🔥 (Hot!)${NC}"
                    elif [ $(echo "$temp > 65" | bc -l 2>/dev/null || echo 0) -eq 1 ]; then
                        echo -e "   ${YELLOW}Temperature: ${BOLD}${temp}°C${NC} ${YELLOW}🌡️  (Warm)${NC}"
                    else
                        echo -e "   ${GREEN}Temperature: ${BOLD}${temp}°C${NC} ${GREEN}❄️  (Cool)${NC}"
                    fi
                fi

                # Show power and clocks
                if [ ! -z "$power" ] && [ "$power" != "N/A" ]; then
                    echo -e "   ${CYAN}Power Draw: ${power}W${NC}"
                fi

                if [ ! -z "$sclk" ] && [ "$sclk" != "N/A" ]; then
                    echo -e "   ${CYAN}GPU Clock: $sclk | Memory Clock: $mclk${NC}"
                fi

                echo -e "   ${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
            fi
        elif [ "$log_device" == "CPU" ]; then
            echo -e "   ${YELLOW}⚠️  TRAINING ON CPU${NC} (GPU available but not used)"
            if [ "$gpu_type" != "NONE" ]; then
                echo -e "   ${YELLOW}   Available GPU: $gpu_name (idle)${NC}"
            fi
        else
            # Unknown - show system GPU info
            if [ "$gpu_type" != "NONE" ]; then
                echo -e "   ${CYAN}🎮 GPU Available:${NC} $gpu_name"
                echo -e "   ${CYAN}   Util: ${gpu_util}% | VRAM: ${gpu_mem}%${NC}"
            else
                echo -e "   ${YELLOW}💻 Device: CPU Only${NC}"
            fi
        fi

        get_training_progress "$C2_LOG" "Challenge 2"
    fi

    echo ""

    # Show overall summary if both completed
    if grep -q "✅ TRAINING COMPLETE" "$C1_LOG" 2>/dev/null && grep -q "✅ TRAINING COMPLETE" "$C2_LOG" 2>/dev/null; then
        echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════════════${NC}"
        echo -e "${BOLD}${GREEN}   🎉 BOTH CHALLENGES COMPLETE!${NC}"
        echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════════════${NC}"

        c1_score=$(grep "Best validation NRMSE:" "$C1_LOG" | tail -1 | grep -oP '\d+\.\d+')
        c2_score=$(grep "Best validation NRMSE:" "$C2_LOG" | tail -1 | grep -oP '\d+\.\d+')

        if [ ! -z "$c1_score" ] && [ ! -z "$c2_score" ]; then
            echo -e "   ${CYAN}Challenge 1 NRMSE:${NC} $(parse_nrmse $c1_score)"
            echo -e "   ${CYAN}Challenge 2 NRMSE:${NC} $(parse_nrmse $c2_score)"

            # Calculate overall score
            overall=$(echo "scale=4; ($c1_score + $c2_score) / 2" | bc)
            echo -e "   ${BOLD}${MAGENTA}Overall Score:${NC} ${BOLD}$overall${NC}"

            # Provide recommendation
            echo ""
            if (( $(echo "$overall < 0.70" | bc -l) )); then
                echo -e "   ${GREEN}✅ EXCELLENT! Submit Phase 1 now!${NC}"
                echo -e "   ${CYAN}   Score < 0.70 is competitive for top rankings${NC}"
            elif (( $(echo "$overall < 0.80" | bc -l) )); then
                echo -e "   ${YELLOW}⚠️  BORDERLINE - Consider Phase 2 improvements${NC}"
                echo -e "   ${CYAN}   Phase 2 could push you into top 3${NC}"
            else
                echo -e "   ${RED}⚠️  NEEDS IMPROVEMENT - Phase 2 recommended${NC}"
                echo -e "   ${CYAN}   Focus on Challenge $([ $(echo "$c1_score > $c2_score" | bc -l) -eq 1 ] && echo "1" || echo "2")${NC}"
            fi

            echo ""
            echo -e "   ${CYAN}Next steps:${NC}"
            echo -e "   ${CYAN}  1. Review PHASE1_RESULTS.md${NC}"
            echo -e "   ${CYAN}  2. Test: python submission.py${NC}"
            echo -e "   ${CYAN}  3. Create: zip submission.zip ...${NC}"
        fi

        echo -e "${BOLD}${GREEN}════════════════════════════════════════════════════════════════════${NC}"
    fi

    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}   Refreshing in ${REFRESH_INTERVAL}s | Press Ctrl+C to stop${NC}"
    echo -e "${CYAN}   Tip: Training runs in background - safe to close this monitor${NC}"
    echo -e "${BOLD}${CYAN}════════════════════════════════════════════════════════════════════${NC}"

    sleep $REFRESH_INTERVAL
done
