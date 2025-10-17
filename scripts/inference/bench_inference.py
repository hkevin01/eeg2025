#!/usr/bin/env python3
"""
Inference Benchmark Script
==========================

Measures latency, throughput, and memory usage for EEG models.
Generates JSON reports for performance validation.
"""

import argparse
import json
import time
import gc
import psutil
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

# Import model components
try:
    from src.models.advanced_foundation_model import AdvancedEEGFoundationModel, FoundationModelConfig
    from src.models.heads.regression import TemporalRegressionHead
    from src.models.heads.classification import CalibratedClassificationHead
    from src.models.heads.psychopathology import PsychopathologyHead
    HAS_MODELS = True
except ImportError:
    HAS_MODELS = False
    print("Warning: Model imports failed, using dummy models for benchmarking")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""
    batch_sizes: List[int]
    sequence_lengths: List[int]
    n_channels: int
    n_warmup: int
    n_iterations: int
    device: str
    enable_amp: bool
    enable_compile: bool


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    batch_size: int
    sequence_length: int
    n_channels: int

    # Latency metrics (milliseconds)
    p50_latency: float
    p95_latency: float
    p99_latency: float
    mean_latency: float

    # Throughput metrics
    samples_per_second: float

    # Memory metrics (MB)
    peak_memory_mb: float
    memory_utilization: float

    # Model info
    model_name: str
    device: str
    amp_enabled: bool
    compile_enabled: bool


class DummyEEGModel(torch.nn.Module):
    """Dummy EEG model for benchmarking when real models unavailable."""

    def __init__(self, n_channels: int = 128, d_model: int = 256):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(n_channels, d_model, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.head = torch.nn.Linear(d_model, 4)  # 4 outputs for demo

    def forward(self, x):
        # x: (batch, channels, time)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x).squeeze(-1)
        return self.head(x)


class ModelBenchmark:
    """Benchmark runner for EEG models."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.device = torch.device(config.device)

        # Setup memory tracking
        if torch.cuda.is_available() and "cuda" in config.device:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def create_model(self, model_name: str, n_channels: int) -> torch.nn.Module:
        """Create model for benchmarking."""
        if not HAS_MODELS or model_name == "dummy":
            return DummyEEGModel(n_channels=n_channels)

        try:
            if model_name == "foundation":
                config = FoundationModelConfig(
                    n_channels=n_channels,
                    d_model=512,
                    n_layers=6,
                    n_heads=8
                )
                return AdvancedEEGFoundationModel(config)

            elif model_name == "regression_head":
                return TemporalRegressionHead(
                    input_dim=512,
                    hidden_dim=256
                )

            elif model_name == "classification_head":
                return CalibratedClassificationHead(
                    input_dim=512,
                    num_classes=2
                )

            elif model_name == "psychopathology_head":
                return PsychopathologyHead(
                    input_dim=512,
                    target_factors=["p_factor", "internalizing", "externalizing", "attention"]
                )

            else:
                return DummyEEGModel(n_channels=n_channels)

        except Exception as e:
            print(f"Failed to create {model_name}: {e}, using dummy model")
            return DummyEEGModel(n_channels=n_channels)

    def create_synthetic_data(self, batch_size: int, n_channels: int, seq_length: int) -> torch.Tensor:
        """Create synthetic EEG data for benchmarking."""
        return torch.randn(
            batch_size, n_channels, seq_length,
            device=self.device,
            dtype=torch.float32
        )

    def measure_memory(self) -> Tuple[float, float]:
        """Measure current memory usage."""
        if torch.cuda.is_available() and "cuda" in self.config.device:
            # GPU memory
            peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
            total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2  # MB
            utilization = peak_memory / total_memory * 100
            return peak_memory, utilization
        else:
            # CPU memory
            process = psutil.Process()
            memory_info = process.memory_info()
            peak_memory = memory_info.rss / 1024**2  # MB
            total_memory = psutil.virtual_memory().total / 1024**2  # MB
            utilization = peak_memory / total_memory * 100
            return peak_memory, utilization

    def benchmark_model(
        self,
        model: torch.nn.Module,
        batch_size: int,
        sequence_length: int,
        model_name: str
    ) -> BenchmarkResult:
        """Benchmark a single model configuration."""
        model = model.to(self.device)
        model.eval()

        # Apply optimizations
        if self.config.enable_compile:
            try:
                model = torch.compile(model, mode="max-autotune")
            except Exception as e:
                print(f"Compilation failed: {e}")

        # Create synthetic data
        data = self.create_synthetic_data(batch_size, self.config.n_channels, sequence_length)

        # Warmup
        print(f"  Warming up... ({self.config.n_warmup} iterations)")
        with torch.no_grad():
            for _ in range(self.config.n_warmup):
                if self.config.enable_amp:
                    with torch.autocast(device_type=self.device.type):
                        _ = model(data)
                else:
                    _ = model(data)

        # Clear memory stats after warmup
        if torch.cuda.is_available() and "cuda" in self.config.device:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        # Benchmark
        print(f"  Benchmarking... ({self.config.n_iterations} iterations)")
        latencies = []

        with torch.no_grad():
            for i in range(self.config.n_iterations):
                # Sync before timing
                if torch.cuda.is_available() and "cuda" in self.config.device:
                    torch.cuda.synchronize()

                start_time = time.perf_counter()

                if self.config.enable_amp:
                    with torch.autocast(device_type=self.device.type):
                        output = model(data)
                else:
                    output = model(data)

                # Sync after inference
                if torch.cuda.is_available() and "cuda" in self.config.device:
                    torch.cuda.synchronize()

                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                if i % 10 == 0:
                    print(f"    Iteration {i+1}/{self.config.n_iterations}: {latency_ms:.2f}ms")

        # Calculate metrics
        latencies = np.array(latencies)
        p50_latency = np.percentile(latencies, 50)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        mean_latency = np.mean(latencies)

        samples_per_second = batch_size * 1000 / mean_latency  # Convert ms to s

        peak_memory, memory_util = self.measure_memory()

        return BenchmarkResult(
            batch_size=batch_size,
            sequence_length=sequence_length,
            n_channels=self.config.n_channels,
            p50_latency=p50_latency,
            p95_latency=p95_latency,
            p99_latency=p99_latency,
            mean_latency=mean_latency,
            samples_per_second=samples_per_second,
            peak_memory_mb=peak_memory,
            memory_utilization=memory_util,
            model_name=model_name,
            device=self.config.device,
            amp_enabled=self.config.enable_amp,
            compile_enabled=self.config.enable_compile
        )

    def run_full_benchmark(self, model_names: List[str]) -> Dict[str, List[BenchmarkResult]]:
        """Run comprehensive benchmark across all configurations."""
        results = {}

        for model_name in model_names:
            print(f"\nBenchmarking {model_name}...")
            model_results = []

            for batch_size in self.config.batch_sizes:
                for seq_length in self.config.sequence_lengths:
                    print(f"  Config: batch_size={batch_size}, seq_length={seq_length}")

                    try:
                        # Create fresh model for each test
                        model = self.create_model(model_name, self.config.n_channels)

                        result = self.benchmark_model(
                            model, batch_size, seq_length, model_name
                        )
                        model_results.append(result)

                        print(f"    P95 latency: {result.p95_latency:.2f}ms")
                        print(f"    Throughput: {result.samples_per_second:.1f} samples/s")
                        print(f"    Peak memory: {result.peak_memory_mb:.1f}MB")

                        # Cleanup
                        del model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    except Exception as e:
                        print(f"    Failed: {e}")
                        continue

            results[model_name] = model_results

        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="EEG Model Inference Benchmark")

    parser.add_argument("--models", type=str, nargs="+",
                       default=["dummy", "foundation", "regression_head"],
                       help="Models to benchmark")
    parser.add_argument("--batch_sizes", type=int, nargs="+",
                       default=[1, 8, 16, 32],
                       help="Batch sizes to test")
    parser.add_argument("--sequence_lengths", type=int, nargs="+",
                       default=[500, 1000, 2000],
                       help="Sequence lengths to test")
    parser.add_argument("--n_channels", type=int, default=128,
                       help="Number of EEG channels")
    parser.add_argument("--n_warmup", type=int, default=5,
                       help="Number of warmup iterations")
    parser.add_argument("--n_iterations", type=int, default=50,
                       help="Number of benchmark iterations")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use (cuda, cpu, auto)")
    parser.add_argument("--enable_amp", action="store_true",
                       help="Enable automatic mixed precision")
    parser.add_argument("--enable_compile", action="store_true",
                       help="Enable torch.compile optimization")
    parser.add_argument("--output_dir", type=str, default="outputs/benchmarks",
                       help="Output directory for results")

    return parser.parse_args()


def save_results(results: Dict[str, List[BenchmarkResult]], output_dir: Path):
    """Save benchmark results to JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert results to JSON-serializable format
    json_results = {}
    for model_name, model_results in results.items():
        json_results[model_name] = [asdict(result) for result in model_results]

    # Save detailed results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    detailed_file = output_dir / f"detailed_benchmark_{timestamp}.json"

    with open(detailed_file, "w") as f:
        json.dump(json_results, f, indent=2)

    print(f"\nDetailed results saved to: {detailed_file}")

    # Save summary results
    summary = generate_summary(results)
    summary_file = output_dir / f"benchmark_summary_{timestamp}.json"

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary results saved to: {summary_file}")

    # Save latest results (for CI/automation)
    latest_file = output_dir / "latest_benchmark.json"
    with open(latest_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Latest results saved to: {latest_file}")


def generate_summary(results: Dict[str, List[BenchmarkResult]]) -> Dict:
    """Generate summary statistics from benchmark results."""
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {}
    }

    for model_name, model_results in results.items():
        if not model_results:
            continue

        # Calculate aggregate metrics
        latencies = [r.p95_latency for r in model_results]
        throughputs = [r.samples_per_second for r in model_results]
        memories = [r.peak_memory_mb for r in model_results]

        summary["models"][model_name] = {
            "num_configs": len(model_results),
            "p95_latency": {
                "min": min(latencies),
                "max": max(latencies),
                "mean": np.mean(latencies)
            },
            "throughput": {
                "min": min(throughputs),
                "max": max(throughputs),
                "mean": np.mean(throughputs)
            },
            "peak_memory_mb": {
                "min": min(memories),
                "max": max(memories),
                "mean": np.mean(memories)
            },
            "best_config": {
                "lowest_latency": asdict(min(model_results, key=lambda x: x.p95_latency)),
                "highest_throughput": asdict(max(model_results, key=lambda x: x.samples_per_second))
            }
        }

    return summary


def print_summary(results: Dict[str, List[BenchmarkResult]]):
    """Print benchmark summary to console."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    for model_name, model_results in results.items():
        if not model_results:
            continue

        print(f"\n{model_name.upper()}:")
        print("-" * 40)

        # Best performance config
        best_latency = min(model_results, key=lambda x: x.p95_latency)
        best_throughput = max(model_results, key=lambda x: x.samples_per_second)

        print(f"Best P95 Latency: {best_latency.p95_latency:.2f}ms")
        print(f"  Config: batch={best_latency.batch_size}, seq_len={best_latency.sequence_length}")
        print(f"  Memory: {best_latency.peak_memory_mb:.1f}MB")

        print(f"Best Throughput: {best_throughput.samples_per_second:.1f} samples/s")
        print(f"  Config: batch={best_throughput.batch_size}, seq_len={best_throughput.sequence_length}")
        print(f"  Latency: {best_throughput.p95_latency:.2f}ms")

        # Check performance targets from README
        if best_latency.p95_latency < 50:
            print("  ✅ Meets <50ms P95 latency target")
        else:
            print("  ❌ Does not meet <50ms P95 latency target")


def main():
    """Main benchmark function."""
    args = parse_args()

    # Configure device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Running benchmarks on {device}")
    print(f"Models: {args.models}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Sequence lengths: {args.sequence_lengths}")
    print(f"AMP enabled: {args.enable_amp}")
    print(f"Compile enabled: {args.enable_compile}")

    # Create benchmark config
    config = BenchmarkConfig(
        batch_sizes=args.batch_sizes,
        sequence_lengths=args.sequence_lengths,
        n_channels=args.n_channels,
        n_warmup=args.n_warmup,
        n_iterations=args.n_iterations,
        device=device,
        enable_amp=args.enable_amp,
        enable_compile=args.enable_compile
    )

    # Run benchmarks
    benchmark = ModelBenchmark(config)
    results = benchmark.run_full_benchmark(args.models)

    # Save and display results
    output_dir = Path(args.output_dir)
    save_results(results, output_dir)
    print_summary(results)

    print(f"\nBenchmark completed successfully!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
