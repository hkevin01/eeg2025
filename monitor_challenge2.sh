#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color
BOLD='\033[1m'

LOG_FILE="logs/challenge2_correct_training.log"
SCRIPT_NAME="train_challenge2_correct.py"

clear

echo -e "${BOLD}${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${CYAN}║                                                                    ║${NC}"
echo -e "${BOLD}${CYAN}║           🧠 Challenge 2 Training Monitor - Real-Time 🧠          ║${NC}"
echo -e "${BOLD}${CYAN}║                                                                    ║${NC}"
echo -e "${BOLD}${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check if process is running
check_process() {
    if pgrep -f "$SCRIPT_NAME" > /dev/null; then
        echo -e "${GREEN}✅ Training process is RUNNING${NC}"
        PID=$(pgrep -f "$SCRIPT_NAME")
        echo -e "${CYAN}   PID: $PID${NC}"
        
        # Get process info
        CPU=$(ps -p $PID -o %cpu --no-headers 2>/dev/null || echo "N/A")
        MEM=$(ps -p $PID -o %mem --no-headers 2>/dev/null || echo "N/A")
        TIME=$(ps -p $PID -o etime --no-headers 2>/dev/null || echo "N/A")
        
        echo -e "${CYAN}   CPU: ${CPU}%  |  Memory: ${MEM}%  |  Runtime: ${TIME}${NC}"
        return 0
    else
        echo -e "${RED}❌ Training process is NOT RUNNING${NC}"
        return 1
    fi
}

# Function to get training phase
get_training_phase() {
    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${RED}⚠️  Log file not found: $LOG_FILE${NC}"
        return 1
    fi
    
    echo -e "\n${BOLD}${YELLOW}📊 Training Progress:${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Check for different phases
    if grep -q "Loading datasets..." "$LOG_FILE"; then
        echo -e "${CYAN}Phase: Data Loading${NC}"
        
        # Show which releases are loaded
        for release in R1 R2 R3 R4 R5; do
            if grep -q "$release.*loaded" "$LOG_FILE"; then
                echo -e "  ${GREEN}✓${NC} $release loaded"
            else
                echo -e "  ${YELLOW}⏳${NC} $release loading..."
            fi
        done
    fi
    
    if grep -q "Combining and filtering datasets" "$LOG_FILE"; then
        echo -e "${CYAN}Phase: Dataset Filtering${NC}"
        
        # Show dataset sizes
        grep -E "Total recordings|After filtering" "$LOG_FILE" | tail -5 | while read line; do
            echo -e "  ${CYAN}→${NC} $line"
        done
    fi
    
    if grep -q "Creating windows" "$LOG_FILE"; then
        echo -e "${CYAN}Phase: Window Creation${NC}"
        
        # Show window counts
        grep -E "Creating windows|Total windows" "$LOG_FILE" | tail -5 | while read line; do
            echo -e "  ${CYAN}→${NC} $line"
        done
    fi
    
    if grep -q "Creating DataLoaders" "$LOG_FILE"; then
        echo -e "${CYAN}Phase: DataLoader Setup${NC}"
    fi
    
    if grep -q "Starting training" "$LOG_FILE"; then
        echo -e "${GREEN}Phase: Training Started!${NC}"
        
        # Show training progress
        echo -e "\n${BOLD}${MAGENTA}📈 Epoch Progress:${NC}"
        echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        # Extract epoch information
        grep -E "Epoch [0-9]+/[0-9]+" "$LOG_FILE" | tail -10 | while read line; do
            if echo "$line" | grep -q "Val Loss"; then
                epoch=$(echo "$line" | grep -oP "Epoch \K[0-9]+")
                train_loss=$(echo "$line" | grep -oP "Train Loss: \K[0-9.]+")
                val_loss=$(echo "$line" | grep -oP "Val Loss: \K[0-9.]+")
                
                echo -e "  ${GREEN}Epoch $epoch${NC} | Train: ${CYAN}$train_loss${NC} | Val: ${YELLOW}$val_loss${NC}"
            else
                echo -e "  ${CYAN}→${NC} $line"
            fi
        done
        
        # Check for best model
        if grep -q "New best model" "$LOG_FILE"; then
            best_epoch=$(grep "New best model" "$LOG_FILE" | tail -1 | grep -oP "epoch \K[0-9]+")
            best_loss=$(grep "New best model" "$LOG_FILE" | tail -1 | grep -oP "loss: \K[0-9.]+")
            echo -e "\n  ${GREEN}⭐ Best Model: Epoch $best_epoch (Val Loss: $best_loss)${NC}"
        fi
        
        # Check for early stopping
        if grep -q "Early stopping" "$LOG_FILE"; then
            echo -e "\n  ${YELLOW}⚠️  Early stopping triggered${NC}"
        fi
    fi
    
    if grep -q "Training complete" "$LOG_FILE"; then
        echo -e "\n${BOLD}${GREEN}✅ TRAINING COMPLETE!${NC}"
        
        # Show final results
        echo -e "\n${BOLD}${GREEN}🎉 Final Results:${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        grep -E "Best val loss|Model saved to" "$LOG_FILE" | tail -5 | while read line; do
            echo -e "  ${GREEN}✓${NC} $line"
        done
    fi
}

# Function to show recent log entries
show_recent_logs() {
    if [ ! -f "$LOG_FILE" ]; then
        return 1
    fi
    
    echo -e "\n${BOLD}${BLUE}📝 Recent Log Entries (last 15 lines):${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    tail -15 "$LOG_FILE" | while read line; do
        # Color code different types of messages
        if echo "$line" | grep -qi "error\|exception\|failed"; then
            echo -e "${RED}$line${NC}"
        elif echo "$line" | grep -qi "warning"; then
            echo -e "${YELLOW}$line${NC}"
        elif echo "$line" | grep -qi "epoch\|loss"; then
            echo -e "${GREEN}$line${NC}"
        elif echo "$line" | grep -qi "loading\|creating\|starting"; then
            echo -e "${CYAN}$line${NC}"
        else
            echo -e "$line"
        fi
    done
}

# Function to show statistics
show_statistics() {
    if [ ! -f "$LOG_FILE" ]; then
        return 1
    fi
    
    echo -e "\n${BOLD}${MAGENTA}📊 Training Statistics:${NC}"
    echo -e "${MAGENTA}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Log file size
    LOG_SIZE=$(du -h "$LOG_FILE" | cut -f1)
    echo -e "  ${CYAN}Log Size:${NC} $LOG_SIZE"
    
    # Number of lines
    LINE_COUNT=$(wc -l < "$LOG_FILE")
    echo -e "  ${CYAN}Log Lines:${NC} $LINE_COUNT"
    
    # Time info
    if [ -f "$LOG_FILE" ]; then
        START_TIME=$(stat -c %y "$LOG_FILE" | cut -d'.' -f1)
        echo -e "  ${CYAN}Started:${NC} $START_TIME"
    fi
    
    # Count epochs completed
    EPOCHS_DONE=$(grep -c "Epoch [0-9]*/[0-9]*" "$LOG_FILE" || echo "0")
    echo -e "  ${CYAN}Epochs Logged:${NC} $EPOCHS_DONE"
    
    # Check for errors
    ERROR_COUNT=$(grep -ci "error\|exception" "$LOG_FILE" || echo "0")
    if [ "$ERROR_COUNT" -gt 0 ]; then
        echo -e "  ${RED}⚠️  Errors Found:${NC} $ERROR_COUNT"
    else
        echo -e "  ${GREEN}✓ No errors detected${NC}"
    fi
    
    # GPU info if available
    if command -v nvidia-smi &> /dev/null; then
        echo -e "\n  ${CYAN}GPU Status:${NC}"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1 | awk -F',' '{
            printf "    GPU %s: %s\n", $1, $2
            printf "    Utilization: %s%% | Memory: %s/%s MB\n", $3, $4, $5
        }'
    fi
}

# Function to show next steps
show_next_steps() {
    if grep -q "Training complete" "$LOG_FILE" 2>/dev/null; then
        echo -e "\n${BOLD}${GREEN}✅ Next Steps:${NC}"
        echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "  ${GREEN}1.${NC} Copy weights: ${CYAN}cp weights_challenge_2_correct.pt weights_challenge_2.pt${NC}"
        echo -e "  ${GREEN}2.${NC} Test submission: ${CYAN}python test_submission_verbose.py${NC}"
        echo -e "  ${GREEN}3.${NC} Create package: ${CYAN}zip -j submission.zip submission.py weights_*.pt${NC}"
        echo -e "  ${GREEN}4.${NC} Submit to competition platform"
    else
        echo -e "\n${BOLD}${YELLOW}⏳ Training in progress...${NC}"
        echo -e "  ${CYAN}→${NC} Monitor with: ${CYAN}tail -f $LOG_FILE${NC}"
        echo -e "  ${CYAN}→${NC} Or run this script again for updates"
    fi
}

# Main monitoring loop
echo ""
check_process
PROCESS_RUNNING=$?

echo ""
get_training_phase

show_recent_logs

show_statistics

show_next_steps

echo -e "\n${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${CYAN}Commands:${NC}"
echo -e "  ${CYAN}Live tail:${NC} tail -f $LOG_FILE"
echo -e "  ${CYAN}Re-run monitor:${NC} ./monitor_challenge2.sh"
echo -e "  ${CYAN}Check process:${NC} ps aux | grep $SCRIPT_NAME"
echo -e "  ${CYAN}Kill if needed:${NC} pkill -f $SCRIPT_NAME"
echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Offer to watch continuously
if [ "$PROCESS_RUNNING" -eq 0 ]; then
    echo -e "${YELLOW}Press Ctrl+C to exit, or wait 30 seconds for auto-refresh...${NC}"
    echo ""
    sleep 30
    exec "$0"  # Re-run this script
fi
