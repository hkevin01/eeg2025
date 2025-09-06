#!/bin/bash
# EEG GPU Demo Management Script
# Provides easy commands to manage the interactive demo

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_error() { echo -e "${RED}❌ $1${NC}"; }

# Check dependencies
check_deps() {
    log_info "Checking dependencies..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check NVIDIA Docker (optional)
    if ! docker info 2>/dev/null | grep -q nvidia; then
        log_warn "NVIDIA Docker runtime not detected - GPU features may be limited"
    fi
    
    log_success "Dependencies check complete"
}

# Start demo services
start_demo() {
    log_info "Starting EEG GPU Demo..."
    
    check_deps
    
    # Build and start services
    docker-compose -f docker/docker-compose.demo.yml up --build -d
    
    # Wait for services to be ready
    log_info "Waiting for services to start..."
    sleep 5
    
    # Check health
    if curl -s http://localhost:8000/health &> /dev/null; then
        log_success "Demo server is running!"
        echo ""
        echo "🌐 Demo Interface: http://localhost:8080/demo/"
        echo "🔧 API Health: http://localhost:8000/health"
        echo "📊 Direct Demo: http://localhost:8000/web/demo.html"
        echo ""
        echo "Use '$0 logs' to view logs"
        echo "Use '$0 stop' to stop services"
    else
        log_error "Failed to start demo server"
        docker-compose -f docker/docker-compose.demo.yml logs
        exit 1
    fi
}

# Stop demo services
stop_demo() {
    log_info "Stopping EEG GPU Demo..."
    docker-compose -f docker/docker-compose.demo.yml down
    log_success "Demo stopped"
}

# Show logs
show_logs() {
    docker-compose -f docker/docker-compose.demo.yml logs -f "${1:-}"
}

# Restart services
restart_demo() {
    log_info "Restarting EEG GPU Demo..."
    stop_demo
    start_demo
}

# Clean up
cleanup() {
    log_info "Cleaning up demo resources..."
    docker-compose -f docker/docker-compose.demo.yml down -v --remove-orphans
    docker system prune -f
    log_success "Cleanup complete"
}

# Check status
status() {
    log_info "Demo Status:"
    docker-compose -f docker/docker-compose.demo.yml ps
    
    echo ""
    log_info "Service Health:"
    
    # Check demo server
    if curl -s http://localhost:8000/health &> /dev/null; then
        log_success "Demo server: Online"
    else
        log_error "Demo server: Offline"
    fi
    
    # Check nginx proxy
    if curl -s http://localhost:8080 &> /dev/null; then
        log_success "Nginx proxy: Online"
    else
        log_error "Nginx proxy: Offline"
    fi
}

# Development mode (local Python server)
dev_server() {
    log_info "Starting development server..."
    
    # Check Python dependencies
    if ! python3 -c "import fastapi, uvicorn" &> /dev/null; then
        log_error "Missing dependencies. Install with: pip install fastapi uvicorn[standard]"
        exit 1
    fi
    
    # Start development server
    export PYTHONPATH="$PROJECT_ROOT"
    python3 scripts/launch_demo.py --reload --host 0.0.0.0 --port 8000
}

# Show help
show_help() {
    echo "EEG GPU Demo Management Script"
    echo ""
    echo "Usage: $0 COMMAND"
    echo ""
    echo "Commands:"
    echo "  start       Start demo services with Docker"
    echo "  stop        Stop demo services"
    echo "  restart     Restart demo services"
    echo "  status      Show service status"
    echo "  logs [svc]  Show logs (optionally for specific service)"
    echo "  cleanup     Clean up all demo resources"
    echo "  dev         Start development server (local Python)"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start                  # Start full demo environment"
    echo "  $0 logs eeg-demo-server   # Show backend logs"
    echo "  $0 dev                    # Development mode"
    echo ""
    echo "URLs:"
    echo "  http://localhost:8080/demo/     - Full demo interface (with proxy)"
    echo "  http://localhost:8000/web/demo.html - Direct demo interface"
    echo "  http://localhost:8000/health    - API health check"
}

# Main command dispatcher
case "${1:-help}" in
    start)
        start_demo
        ;;
    stop)
        stop_demo
        ;;
    restart)
        restart_demo
        ;;
    status)
        status
        ;;
    logs)
        show_logs "$2"
        ;;
    cleanup)
        cleanup
        ;;
    dev)
        dev_server
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac
