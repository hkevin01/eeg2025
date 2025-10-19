#!/bin/bash

# Watchdog management script

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'
BOLD='\033[1m'

case "${1:-status}" in
    start)
        if pgrep -f "watchdog_challenge2.sh" > /dev/null; then
            echo -e "${YELLOW}⚠️  Watchdog is already running${NC}"
            PID=$(pgrep -f "watchdog_challenge2.sh")
            echo -e "   PID: $PID"
        else
            echo -e "${GREEN}🐕 Starting watchdog...${NC}"
            nohup ./watchdog_challenge2.sh > logs/watchdog_output.log 2>&1 &
            sleep 2
            if pgrep -f "watchdog_challenge2.sh" > /dev/null; then
                PID=$(pgrep -f "watchdog_challenge2.sh")
                echo -e "${GREEN}✅ Watchdog started (PID: $PID)${NC}"
                echo -e "${CYAN}   Monitoring: train_challenge2_correct.py${NC}"
                echo -e "${CYAN}   Log: logs/watchdog.log${NC}"
                echo -e "${CYAN}   Output: logs/watchdog_output.log${NC}"
            else
                echo -e "${RED}❌ Failed to start watchdog${NC}"
            fi
        fi
        ;;
        
    stop)
        if pgrep -f "watchdog_challenge2.sh" > /dev/null; then
            echo -e "${YELLOW}🛑 Stopping watchdog...${NC}"
            pkill -f "watchdog_challenge2.sh"
            sleep 1
            if pgrep -f "watchdog_challenge2.sh" > /dev/null; then
                echo -e "${RED}❌ Failed to stop watchdog (trying force kill)${NC}"
                pkill -9 -f "watchdog_challenge2.sh"
            else
                echo -e "${GREEN}✅ Watchdog stopped${NC}"
            fi
        else
            echo -e "${YELLOW}⚠️  Watchdog is not running${NC}"
        fi
        ;;
        
    restart)
        echo -e "${CYAN}🔄 Restarting watchdog...${NC}"
        $0 stop
        sleep 2
        $0 start
        ;;
        
    status)
        echo -e "${BOLD}${CYAN}Watchdog Status${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        
        if pgrep -f "watchdog_challenge2.sh" > /dev/null; then
            PID=$(pgrep -f "watchdog_challenge2.sh")
            RUNTIME=$(ps -p $PID -o etime --no-headers 2>/dev/null)
            echo -e "${GREEN}✅ Watchdog: RUNNING${NC}"
            echo -e "   PID: $PID"
            echo -e "   Runtime: $RUNTIME"
        else
            echo -e "${RED}❌ Watchdog: NOT RUNNING${NC}"
        fi
        
        echo ""
        
        if pgrep -f "train_challenge2_correct.py" > /dev/null; then
            PID=$(pgrep -f "train_challenge2_correct.py")
            RUNTIME=$(ps -p $PID -o etime --no-headers 2>/dev/null)
            CPU=$(ps -p $PID -o %cpu --no-headers 2>/dev/null)
            MEM=$(ps -p $PID -o %mem --no-headers 2>/dev/null)
            echo -e "${GREEN}✅ Training: RUNNING${NC}"
            echo -e "   PID: $PID"
            echo -e "   Runtime: $RUNTIME"
            echo -e "   CPU: ${CPU}%"
            echo -e "   Memory: ${MEM}%"
        else
            echo -e "${RED}❌ Training: NOT RUNNING${NC}"
        fi
        
        echo ""
        echo -e "${CYAN}Logs:${NC}"
        if [ -f "logs/watchdog.log" ]; then
            LINES=$(wc -l < logs/watchdog.log)
            LAST=$(tail -1 logs/watchdog.log)
            echo -e "   Watchdog log: $LINES lines"
            echo -e "   Last entry: $LAST"
        else
            echo -e "   Watchdog log: Not found"
        fi
        
        if [ -f "logs/challenge2_correct_training.log" ]; then
            LINES=$(wc -l < logs/challenge2_correct_training.log)
            echo -e "   Training log: $LINES lines"
        else
            echo -e "   Training log: Not found"
        fi
        ;;
        
    logs)
        echo -e "${BOLD}${CYAN}Watchdog Logs (last 20 lines)${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        if [ -f "logs/watchdog.log" ]; then
            tail -20 logs/watchdog.log
        else
            echo -e "${YELLOW}No logs found${NC}"
        fi
        ;;
        
    follow)
        echo -e "${CYAN}Following watchdog output (Ctrl+C to stop)...${NC}"
        echo ""
        tail -f logs/watchdog_output.log 2>/dev/null || echo "Log file not found"
        ;;
        
    *)
        echo -e "${BOLD}${CYAN}Watchdog Management${NC}"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs|follow}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the watchdog"
        echo "  stop    - Stop the watchdog"
        echo "  restart - Restart the watchdog"
        echo "  status  - Show current status"
        echo "  logs    - Show recent watchdog logs"
        echo "  follow  - Follow watchdog output in real-time"
        echo ""
        echo "The watchdog monitors training for:"
        echo "  • Process crashes"
        echo "  • Freezing (no log updates for 5+ minutes)"
        echo "  • Errors in logs"
        echo "  • High memory usage"
        ;;
esac
