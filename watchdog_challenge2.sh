#!/bin/bash

# Watchdog script for Challenge 2 training
# Monitors for crashes, freezing, and errors

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m'
BOLD='\033[1m'

LOG_FILE="logs/challenge2_correct_training.log"
SCRIPT_NAME="train_challenge2_correct.py"
WATCHDOG_LOG="logs/watchdog.log"
CHECK_INTERVAL=60  # Check every 60 seconds
FREEZE_THRESHOLD=300  # Consider frozen if no new logs for 5 minutes

# Create watchdog log
mkdir -p logs
echo "========================================" >> "$WATCHDOG_LOG"
echo "Watchdog started: $(date)" >> "$WATCHDOG_LOG"
echo "========================================" >> "$WATCHDOG_LOG"

# Function to log and display
log_message() {
    local level=$1
    local message=$2
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "[$timestamp] $level: $message" >> "$WATCHDOG_LOG"
    
    case $level in
        "ERROR")
            echo -e "${BOLD}${RED}🚨 ERROR: $message${NC}" ;;
        "WARNING")
            echo -e "${BOLD}${YELLOW}⚠️  WARNING: $message${NC}" ;;
        "INFO")
            echo -e "${CYAN}ℹ️  INFO: $message${NC}" ;;
        "SUCCESS")
            echo -e "${GREEN}✅ SUCCESS: $message${NC}" ;;
    esac
}

# Function to send alert (visual + audio beep)
send_alert() {
    local alert_type=$1
    local message=$2
    
    # Visual alert
    clear
    echo -e "${BOLD}${RED}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                    ║"
    echo "║                    🚨 TRAINING ALERT! 🚨                          ║"
    echo "║                                                                    ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
    echo ""
    echo -e "${BOLD}${RED}Alert Type: $alert_type${NC}"
    echo -e "${BOLD}${RED}Message: $message${NC}"
    echo ""
    echo -e "${BOLD}${RED}Time: $(date)${NC}"
    echo ""
    echo -e "${YELLOW}Check the logs for details:${NC}"
    echo -e "  tail -100 $LOG_FILE"
    echo -e "  cat $WATCHDOG_LOG"
    echo ""
    
    # Try to beep (if available)
    for i in {1..5}; do
        echo -ne '\007' 2>/dev/null
        sleep 0.3
    done
    
    log_message "ERROR" "$alert_type: $message"
}

# Track last log file size
last_size=0
last_change_time=$(date +%s)
freeze_count=0
crash_check_count=0

echo ""
log_message "INFO" "Watchdog monitoring started"
log_message "INFO" "Target: $SCRIPT_NAME"
log_message "INFO" "Log file: $LOG_FILE"
log_message "INFO" "Check interval: ${CHECK_INTERVAL}s"
log_message "INFO" "Freeze threshold: ${FREEZE_THRESHOLD}s"
echo ""

# Main monitoring loop
while true; do
    current_time=$(date +%s)
    
    # Check if process is still running
    if ! pgrep -f "$SCRIPT_NAME" > /dev/null; then
        # Check if training completed normally
        if grep -q "Training complete" "$LOG_FILE" 2>/dev/null; then
            log_message "SUCCESS" "Training completed successfully!"
            echo ""
            echo -e "${BOLD}${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
            echo -e "${BOLD}${GREEN}║                                                                    ║${NC}"
            echo -e "${BOLD}${GREEN}║              ✅ TRAINING COMPLETED SUCCESSFULLY! ✅               ║${NC}"
            echo -e "${BOLD}${GREEN}║                                                                    ║${NC}"
            echo -e "${BOLD}${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
            echo ""
            echo -e "${CYAN}Next steps:${NC}"
            echo -e "  1. cp weights_challenge_2_correct.pt weights_challenge_2.pt"
            echo -e "  2. python test_submission_verbose.py"
            echo ""
            exit 0
        else
            # Process died unexpectedly
            send_alert "CRASH DETECTED" "Training process is not running!"
            echo ""
            echo -e "${RED}Checking for errors in log file...${NC}"
            echo ""
            
            # Show last 20 lines
            echo -e "${YELLOW}Last 20 lines of log:${NC}"
            tail -20 "$LOG_FILE" | sed 's/^/  /'
            echo ""
            
            # Check for common errors
            if grep -qi "out of memory\|cuda out of memory\|killed" "$LOG_FILE"; then
                echo -e "${RED}💥 Detected: OUT OF MEMORY error${NC}"
                echo -e "${YELLOW}Solution: Reduce batch size or use CPU mode${NC}"
            elif grep -qi "cuda error\|gpu error" "$LOG_FILE"; then
                echo -e "${RED}💥 Detected: CUDA/GPU error${NC}"
                echo -e "${YELLOW}Solution: Check GPU availability or switch to CPU${NC}"
            elif grep -qi "file not found\|no such file" "$LOG_FILE"; then
                echo -e "${RED}💥 Detected: File not found error${NC}"
                echo -e "${YELLOW}Solution: Check data directory and file paths${NC}"
            elif grep -qi "permission denied" "$LOG_FILE"; then
                echo -e "${RED}💥 Detected: Permission error${NC}"
                echo -e "${YELLOW}Solution: Check file/directory permissions${NC}"
            else
                echo -e "${RED}💥 Process crashed unexpectedly${NC}"
                echo -e "${YELLOW}Check the full log for details: tail -100 $LOG_FILE${NC}"
            fi
            
            echo ""
            echo -e "${CYAN}Watchdog stopped. Fix the issue and restart training.${NC}"
            exit 1
        fi
    fi
    
    # Check if log file exists
    if [ ! -f "$LOG_FILE" ]; then
        log_message "WARNING" "Log file not found: $LOG_FILE"
        sleep $CHECK_INTERVAL
        continue
    fi
    
    # Check for new content (freeze detection)
    current_size=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null)
    
    if [ "$current_size" -gt "$last_size" ]; then
        # New content added - training is progressing
        last_size=$current_size
        last_change_time=$current_time
        freeze_count=0
        
        # Extract latest progress
        latest_line=$(tail -1 "$LOG_FILE")
        if echo "$latest_line" | grep -q "Epoch"; then
            epoch=$(echo "$latest_line" | grep -oP "Epoch \K[0-9]+/[0-9]+")
            batch=$(echo "$latest_line" | grep -oP "Batch \K[0-9]+/[0-9]+")
            loss=$(echo "$latest_line" | grep -oP "Loss: \K[0-9.]+")
            
            log_message "INFO" "Progress: Epoch $epoch, Batch $batch, Loss $loss"
        fi
        
        # Check for errors in recent logs
        if tail -10 "$LOG_FILE" | grep -qi "error\|exception\|failed"; then
            log_message "WARNING" "Error detected in recent logs!"
            echo -e "${YELLOW}Recent errors found:${NC}"
            tail -10 "$LOG_FILE" | grep -i "error\|exception\|failed" | sed 's/^/  /'
        fi
    else
        # No new content
        time_since_change=$((current_time - last_change_time))
        
        if [ $time_since_change -gt $FREEZE_THRESHOLD ]; then
            freeze_count=$((freeze_count + 1))
            
            if [ $freeze_count -eq 1 ]; then
                # First freeze detection
                send_alert "FREEZE DETECTED" "No log updates for ${time_since_change}s (threshold: ${FREEZE_THRESHOLD}s)"
                echo ""
                echo -e "${YELLOW}Process is still running but no progress...${NC}"
                echo ""
                
                # Get process info
                PID=$(pgrep -f "$SCRIPT_NAME")
                if [ ! -z "$PID" ]; then
                    echo -e "${CYAN}Process info:${NC}"
                    ps -p $PID -o pid,ppid,%cpu,%mem,etime,cmd | sed 's/^/  /'
                    echo ""
                fi
                
                # Show last 20 lines of log
                echo -e "${YELLOW}Last 20 lines of log:${NC}"
                tail -20 "$LOG_FILE" | sed 's/^/  /'
                echo ""
                
                echo -e "${CYAN}Continuing to monitor... (alert will repeat every check)${NC}"
            else
                log_message "WARNING" "Still frozen: ${time_since_change}s since last update"
            fi
        else
            # Within threshold, but no update yet
            log_message "INFO" "Waiting for update... (${time_since_change}s since last change)"
        fi
    fi
    
    # Periodic health check (every 10 checks)
    crash_check_count=$((crash_check_count + 1))
    if [ $((crash_check_count % 10)) -eq 0 ]; then
        PID=$(pgrep -f "$SCRIPT_NAME")
        CPU=$(ps -p $PID -o %cpu --no-headers 2>/dev/null || echo "N/A")
        MEM=$(ps -p $PID -o %mem --no-headers 2>/dev/null || echo "N/A")
        RUNTIME=$(ps -p $PID -o etime --no-headers 2>/dev/null || echo "N/A")
        
        log_message "INFO" "Health check: CPU=${CPU}%, MEM=${MEM}%, Runtime=${RUNTIME}"
        
        # Check for extremely high memory usage
        if [ "$MEM" != "N/A" ]; then
            MEM_INT=$(echo $MEM | cut -d'.' -f1)
            if [ "$MEM_INT" -gt 90 ]; then
                log_message "WARNING" "Very high memory usage: ${MEM}%"
                echo -e "${YELLOW}⚠️  Memory usage is very high (${MEM}%), monitoring closely...${NC}"
            fi
        fi
    fi
    
    # Display status bar
    echo -ne "\r${GREEN}✓${NC} Monitoring... Process: ${GREEN}RUNNING${NC} | Last update: ${time_since_change}s ago | Checks: $crash_check_count"
    
    # Sleep before next check
    sleep $CHECK_INTERVAL
done
