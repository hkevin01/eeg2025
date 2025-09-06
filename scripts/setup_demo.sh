#!/bin/bash
# Interactive GPU Demo Setup Script
# Sets up all dependencies for the EEG GPU demonstration

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
log_success() { echo -e "${GREEN}✅ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }

echo "🚀 EEG GPU Demo Setup"
echo "===================="

# Install Python dependencies
log_info "Installing Python dependencies..."
pip install fastapi uvicorn[standard] pydantic requests scipy

# Install PyTorch (CPU version for basic functionality)
log_info "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install GPU libraries (optional)
log_info "Installing GPU libraries (optional)..."
pip install triton cupy-cuda12x || log_warn "GPU libraries installation failed (continuing with CPU-only)"

# Install additional dependencies
log_info "Installing additional dependencies..."
pip install numpy matplotlib pillow

log_success "Setup complete!"

echo ""
echo "🎮 Quick Start Commands:"
echo "  ./scripts/demo.sh dev          # Start development server"
echo "  ./scripts/demo.sh start        # Start full Docker stack"
echo "  python tests/test_demo_integration.py  # Run integration tests"
echo ""
echo "📱 Demo URLs:"
echo "  http://localhost:8000/web/demo.html    # Direct demo"
echo "  http://localhost:8000/health           # API health check"
