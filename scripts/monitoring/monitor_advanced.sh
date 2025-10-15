#!/bin/bash
# Advanced Training Monitor with Live Updates

REFRESH_INTERVAL=5  # seconds

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

clear

echo -e "${CYAN}╔════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║       EEG Training Monitor - Live Updates                         ║${NC}"
echo -e "${CYAN}╚════════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to exit${NC}"
echo ""

while true; do
    # Move cursor to home position (preserve header)
    tput cup 5 0
    
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}📊 System Resources${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # CPU Usage
    CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')
    echo -e "  CPU Usage: ${GREEN}${CPU_USAGE}${NC}"
    
    # Memory Usage
    MEM_INFO=$(free -h | awk '/^Mem:/ {printf "%.1f GB / %.1f GB (%.0f%%)", $3, $2, ($3/$2)*100}')
    echo -e "  Memory:    ${GREEN}${MEM_INFO}${NC}"
    
    # Disk Usage
    DISK_USAGE=$(df -h . | awk 'NR==2 {print $3 " / " $2 " (" $5 ")"}')
    echo -e "  Disk:      ${GREEN}${DISK_USAGE}${NC}"
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}🏃 Training Processes${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Find training processes
    TRAIN_PROCS=$(ps aux | grep python | grep train | grep -v grep)
    
    if [ -z "$TRAIN_PROCS" ]; then
        echo -e "  ${RED}❌ No training processes running${NC}"
    else
        echo "$TRAIN_PROCS" | while IFS= read -r line; do
            PID=$(echo $line | awk '{print $2}')
            CPU=$(echo $line | awk '{print $3}')
            MEM=$(echo $line | awk '{print $4}')
            TIME=$(echo $line | awk '{print $10}')
            CMD=$(echo $line | awk '{for(i=11;i<=NF;i++) printf $i" "; print ""}')
            
            echo -e "  ${GREEN}✓${NC} PID: ${YELLOW}${PID}${NC} | CPU: ${GREEN}${CPU}%${NC} | MEM: ${GREEN}${MEM}%${NC} | Time: ${CYAN}${TIME}${NC}"
            echo -e "    ${CMD}"
        done
    fi
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}📝 Recent Training Output${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # Get latest log file
    LATEST_LOG=$(ls -t logs/*.log 2>/dev/null | head -1)
    
    if [ -n "$LATEST_LOG" ]; then
        echo -e "  Log: ${CYAN}${LATEST_LOG}${NC}"
        echo ""
        tail -15 "$LATEST_LOG" | sed 's/^/  /'
    else
        echo -e "  ${YELLOW}No log files found${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}💾 Checkpoints${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    # List checkpoints
    CHECKPOINTS=$(ls -lh checkpoints/*.pth 2>/dev/null | tail -5)
    
    if [ -n "$CHECKPOINTS" ]; then
        echo "$CHECKPOINTS" | awk '{printf "  %s  %s  %s\n", $9, $5, $6" "$7" "$8}'
    else
        echo -e "  ${YELLOW}No checkpoints yet${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}Last updated: $(date '+%Y-%m-%d %H:%M:%S')${NC}"
    echo -e "${YELLOW}Refreshing in ${REFRESH_INTERVAL}s... (Ctrl+C to exit)${NC}"
    
    sleep $REFRESH_INTERVAL
done
