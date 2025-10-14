"""
Test inference speed to ensure <50ms latency requirement
"""

import time
from pathlib import Path

import numpy as np
import pytest
import torch

from src.models.advanced_foundation_model import (
    AdvancedEEGFoundationModel,
    FoundationModelConfig,
)


@pytest.fixture
def model():
    """Create a foundation model for testing."""
    config = FoundationModelConfig(
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        use_domain_adaptation=False,  # Disable for speed test
        use_compression_ssl=False,  # Disable for speed test
        use_gpu_optimization=False,  # Test CPU first
    )
    model = AdvancedEEGFoundationModel(config)
    model.eval()
    return model


@pytest.fixture
def sample_input():
    """Create sample input (batch_size=1, n_channels=128, seq_len=1000)."""
    # 2-second window at 500 Hz = 1000 samples
    batch_size = 1
    n_channels = 128
    seq_len = 1000
    x = torch.randn(batch_size, n_channels, seq_len)
    task_ids = torch.zeros(batch_size, dtype=torch.long)  # Task 0
    return {"x": x, "task_ids": task_ids}


def measure_inference_time(model, input_data, n_runs=100):
    """Measure inference time over multiple runs."""
    times = []

    # Extract inputs
    if isinstance(input_data, dict):
        x = input_data["x"]
        task_ids = input_data["task_ids"]
    else:
        x = input_data
        task_ids = torch.zeros(x.shape[0], dtype=torch.long)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x, task_ids)

    # Actual timing
    with torch.no_grad():
        for _ in range(n_runs):
            start = time.perf_counter()
            _ = model(x, task_ids)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # Convert to ms

    return times


def test_average_inference_latency(model, sample_input):
    """Test that average inference latency is < 50ms."""
    times = measure_inference_time(model, sample_input, n_runs=100)

    avg_time = np.mean(times)
    std_time = np.std(times)

    print(f"\n⏱️  Inference Speed Statistics:")
    print(f"   Average: {avg_time:.2f} ms")
    print(f"   Std Dev: {std_time:.2f} ms")
    print(f"   Min: {np.min(times):.2f} ms")
    print(f"   Max: {np.max(times):.2f} ms")

    # Critical requirement: average must be < 50ms
    assert avg_time < 50.0, f"Average inference time {avg_time:.2f}ms exceeds 50ms requirement"


def test_p95_inference_latency(model, sample_input):
    """Test that P95 latency is < 75ms."""
    times = measure_inference_time(model, sample_input, n_runs=100)

    p95_time = np.percentile(times, 95)

    print(f"\n⏱️  P95 Latency: {p95_time:.2f} ms")

    # P95 should be reasonable (e.g., < 75ms)
    assert p95_time < 75.0, f"P95 latency {p95_time:.2f}ms exceeds 75ms threshold"


def test_p99_inference_latency(model, sample_input):
    """Test that P99 latency is < 100ms."""
    times = measure_inference_time(model, sample_input, n_runs=100)

    p99_time = np.percentile(times, 99)

    print(f"\n⏱️  P99 Latency: {p99_time:.2f} ms")

    # P99 should still be reasonable
    assert p99_time < 100.0, f"P99 latency {p99_time:.2f}ms exceeds 100ms threshold"


def test_inference_consistency(model, sample_input):
    """Test that inference time is consistent (low variance)."""
    times = measure_inference_time(model, sample_input, n_runs=100)

    std_time = np.std(times)
    avg_time = np.mean(times)
    cv = std_time / avg_time  # Coefficient of variation

    print(f"\n⏱️  Inference Consistency:")
    print(f"   Coefficient of Variation: {cv:.3f}")

    # Check that variance is reasonable (CV < 0.5 means std < 50% of mean)
    assert cv < 0.5, f"Inference time too variable (CV={cv:.3f})"


def test_batch_inference_speed(model):
    """Test inference speed with different batch sizes."""
    batch_sizes = [1, 4, 8, 16]
    n_channels = 128
    seq_len = 1000

    print(f"\n⏱️  Batch Inference Speed:")

    for batch_size in batch_sizes:
        x = torch.randn(batch_size, n_channels, seq_len)
        task_ids = torch.zeros(batch_size, dtype=torch.long)
        input_data = {"x": x, "task_ids": task_ids}
        times = measure_inference_time(model, input_data, n_runs=50)
        avg_time = np.mean(times)
        time_per_sample = avg_time / batch_size

        print(f"   Batch {batch_size:2d}: {avg_time:6.2f} ms total, {time_per_sample:5.2f} ms/sample")

        # For batch_size=1, should still be < 50ms
        if batch_size == 1:
            assert avg_time < 50.0, f"Single-sample inference {avg_time:.2f}ms exceeds 50ms"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_gpu_inference_speed(model, sample_input):
    """Test inference speed on GPU."""
    device = torch.device("cuda")
    model = model.to(device)
    input_data = {
        "x": sample_input["x"].to(device),
        "task_ids": sample_input["task_ids"].to(device),
    }

    # Warmup and timing
    times = measure_inference_time(model, input_data, n_runs=100)

    avg_time = np.mean(times)

    print(f"\n⏱️  GPU Inference Speed:")
    print(f"   Average: {avg_time:.2f} ms")

    # GPU should be faster than CPU requirement
    assert avg_time < 50.0, f"GPU inference time {avg_time:.2f}ms exceeds 50ms requirement"


def test_model_size():
    """Test that model size is reasonable for production deployment."""
    config = FoundationModelConfig(
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        use_domain_adaptation=False,
        use_compression_ssl=False,
        use_gpu_optimization=False,
    )
    model = AdvancedEEGFoundationModel(config)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    n_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n📊 Model Size:")
    print(f"   Total parameters: {n_params:,}")
    print(f"   Trainable parameters: {n_params_trainable:,}")
    print(f"   Model size (MB): {n_params * 4 / 1024 / 1024:.2f}")  # Assuming float32

    # Check that model isn't too large (< 100M parameters is reasonable)
    assert n_params < 100_000_000, f"Model too large: {n_params:,} parameters"


if __name__ == "__main__":
    # Run benchmarks directly
    config = FoundationModelConfig(
        hidden_dim=256,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        use_domain_adaptation=False,
        use_compression_ssl=False,
        use_gpu_optimization=False,
    )
    model = AdvancedEEGFoundationModel(config)
    model.eval()

    x = torch.randn(1, 128, 1000)
    task_ids = torch.zeros(1, dtype=torch.long)
    sample_input = {"x": x, "task_ids": task_ids}

    print("=" * 60)
    print("EEG Foundation Model - Inference Speed Benchmark")
    print("=" * 60)

    times = measure_inference_time(model, sample_input, n_runs=100)
    avg_time = np.mean(times)
    p95_time = np.percentile(times, 95)
    p99_time = np.percentile(times, 99)

    print(f"\n⏱️  Results:")
    print(f"   Average: {avg_time:.2f} ms {'✅' if avg_time < 50 else '❌'}")
    print(f"   P95: {p95_time:.2f} ms {'✅' if p95_time < 75 else '❌'}")
    print(f"   P99: {p99_time:.2f} ms {'✅' if p99_time < 100 else '❌'}")
    print(f"   Min: {np.min(times):.2f} ms")
    print(f"   Max: {np.max(times):.2f} ms")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model: {n_params:,} parameters ({n_params * 4 / 1024 / 1024:.2f} MB)")

    if avg_time < 50:
        print("\n✅ PASS: Model meets <50ms inference requirement!")
    else:
        print("\n❌ FAIL: Model does NOT meet <50ms inference requirement")
        print("   Consider: model quantization, pruning, or architecture optimization")
