#!/usr/bin/env python3
"""
Demo Server Launcher
==================

Convenient launcher for the EEG GPU demo server with automatic dependency checking.
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required = ['fastapi', 'uvicorn', 'torch', 'numpy', 'scipy']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing dependencies: {', '.join(missing)}")
        print("Install with: pip install fastapi uvicorn[standard] torch numpy scipy")
        return False
    
    print("✅ All dependencies found")
    return True

def check_gpu_components():
    """Check GPU component availability."""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"🔧 CUDA Available: {cuda_available}")
        
        # Check GPU components
        gpu_components = []
        try:
            import triton
            gpu_components.append("Triton")
        except ImportError:
            pass
        
        try:
            import cupy
            gpu_components.append("CuPy")
        except ImportError:
            pass
        
        if gpu_components:
            print(f"⚡ GPU Components: {', '.join(gpu_components)}")
        else:
            print("⚠️  No GPU acceleration libraries found (Triton, CuPy)")
        
        return True
    except ImportError:
        print("❌ PyTorch not found")
        return False

def main():
    parser = argparse.ArgumentParser(description="Launch EEG GPU Demo Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--skip-checks", action="store_true", help="Skip dependency checks")
    args = parser.parse_args()
    
    if not args.skip_checks:
        print("🔍 Checking dependencies...")
        if not check_dependencies():
            sys.exit(1)
        
        print("🔧 Checking GPU components...")
        check_gpu_components()
    
    # Launch server
    print(f"\n🚀 Starting demo server on {args.host}:{args.port}")
    print(f"📱 Demo URL: http://localhost:{args.port}/web/demo.html")
    print(f"🔗 API Health: http://localhost:{args.port}/health")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        cmd = [
            sys.executable, "-m", "uvicorn",
            "backend.demo_server:app",
            "--host", args.host,
            "--port", str(args.port),
            "--log-level", "info"
        ]
        
        if args.reload:
            cmd.append("--reload")
        
        subprocess.run(cmd, cwd=Path(__file__).parent)
    except KeyboardInterrupt:
        print("\n👋 Demo server stopped")

if __name__ == "__main__":
    main()
