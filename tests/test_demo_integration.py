#!/usr/bin/env python3
"""
Demo Integration Test
====================

Tests the interactive GPU demo setup and validates all components.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import requests


def test_file_structure():
    """Test that all demo files are in place."""
    print("🔍 Testing file structure...")

    required_files = [
        "backend/demo_server.py",
        "web/demo.html",
        "web/README.md",
        "scripts/demo.sh",
        "scripts/launch_demo.py",
        "docker/docker-compose.demo.yml",
        "docker/Dockerfile.demo",
        "docker/nginx.conf",
    ]

    missing_files = []
    project_root = Path(__file__).parent.parent

    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)

    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False

    print("✅ All demo files present")
    return True


def test_backend_imports():
    """Test that backend can import required modules."""
    print("🔍 Testing backend imports...")

    try:
        # Test basic imports
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from backend.demo_server import app, health

        print("✅ Backend imports successful")
        return True
    except ImportError as e:
        print(f"❌ Backend import failed: {e}")
        return False


def test_gpu_detection():
    """Test GPU component detection."""
    print("🔍 Testing GPU detection...")

    try:
        import torch

        cuda_available = torch.cuda.is_available()
        print(f"  CUDA: {'✅' if cuda_available else '❌'}")

        # Test Triton
        try:
            import triton

            print("  Triton: ✅")
        except ImportError:
            print("  Triton: ❌ (optional)")

        # Test CuPy
        try:
            import cupy

            print("  CuPy: ✅")
        except ImportError:
            print("  CuPy: ❌ (optional)")

        return True
    except ImportError as e:
        print(f"❌ PyTorch not available: {e}")
        return False


def test_demo_server_start():
    """Test that demo server can start."""
    print("🔍 Testing demo server startup...")

    project_root = Path(__file__).parent.parent

    try:
        # Start server in background
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.demo_server:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8001",  # Use different port to avoid conflicts
                "--log-level",
                "error",
            ],
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for startup
        time.sleep(3)

        # Test health endpoint
        try:
            response = requests.get("http://127.0.0.1:8001/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                print(f"✅ Server started successfully")
                print(f"  Status: {health_data.get('status')}")
                print(f"  Device: {health_data.get('device')}")
                print(f"  CUDA: {health_data.get('cuda')}")
                success = True
            else:
                print(f"❌ Health check failed: {response.status_code}")
                success = False
        except requests.RequestException as e:
            print(f"❌ Failed to connect to server: {e}")
            success = False

        # Clean up
        proc.terminate()
        proc.wait(timeout=5)

        return success

    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False


def test_inference_endpoint():
    """Test inference endpoint functionality."""
    print("🔍 Testing inference endpoint...")

    project_root = Path(__file__).parent.parent

    try:
        # Start server
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "backend.demo_server:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8002",
                "--log-level",
                "error",
            ],
            cwd=project_root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        time.sleep(3)

        # Test inference
        payload = {
            "channels": 128,
            "sfreq": 500,
            "window_s": 2.0,
            "use_fused_preproc": True,
            "bandpass": [0.1, 40.0],
            "notch": 60.0,
            "use_rmsnorm": True,
            "use_perceptual_quant": False,
            "simulate": True,
        }

        try:
            response = requests.post(
                "http://127.0.0.1:8002/infer_once", json=payload, timeout=10
            )

            if response.status_code == 200:
                result = response.json()
                latency = result.get("latency_ms", {})
                outputs = result.get("outputs", {})

                print("✅ Inference endpoint working")
                print(f"  Total latency: {latency.get('total', 0):.2f}ms")
                print(f"  Response time: {outputs.get('response_time_ms', 0):.0f}ms")
                success = True
            else:
                print(f"❌ Inference failed: {response.status_code}")
                success = False

        except requests.RequestException as e:
            print(f"❌ Inference request failed: {e}")
            success = False

        # Clean up
        proc.terminate()
        proc.wait(timeout=5)

        return success

    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False


def test_html_content():
    """Test that HTML demo contains required elements."""
    print("🔍 Testing HTML demo content...")

    html_path = Path(__file__).parent.parent / "web" / "demo.html"

    try:
        with open(html_path, "r") as f:
            content = f.read()

        required_elements = [
            'id="use_fused"',  # GPU controls
            'id="run"',  # Run button
            'id="ts"',  # Time series canvas
            'id="spec"',  # Spectrum canvas
            "/health",  # Health check
            "/infer_once",  # Inference endpoint
        ]

        missing_elements = []
        for element in required_elements:
            if element not in content:
                missing_elements.append(element)

        if missing_elements:
            print(f"❌ Missing HTML elements: {missing_elements}")
            return False

        print("✅ HTML demo content validated")
        return True

    except Exception as e:
        print(f"❌ HTML validation failed: {e}")
        return False


def main():
    """Run all demo tests."""
    print("🧪 EEG GPU Demo Integration Test")
    print("=" * 50)

    tests = [
        test_file_structure,
        test_backend_imports,
        test_gpu_detection,
        test_html_content,
        test_demo_server_start,
        test_inference_endpoint,
    ]

    results = []
    for test in tests:
        print()
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)

    print("\n" + "=" * 50)
    print("📊 Test Summary")

    passed = sum(results)
    total = len(results)

    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Demo is ready to launch.")
        print("\nTo start the demo:")
        print("  ./scripts/demo.sh start     # Full Docker stack")
        print("  ./scripts/demo.sh dev       # Development server")
    else:
        print("⚠️  Some tests failed. Check the output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
