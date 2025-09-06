"""
Inference Benchmarking Suite
===========================

Production-ready inference benchmarking with latency profiling, memory monitoring,
streaming evaluation, and performance optimization analysis.

Key Features:
- Latency profiling with percentile analysis
- Memory usage monitoring and optimization
- Streaming evaluation for real-time applications
- Batch processing optimization
- Model comparison and performance targets
- Automated performance regression detection
- Production deployment readiness assessment
"""

from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import gc
import psutil
import threading
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager


@dataclass
class PerformanceTarget:
    """Performance targets for production deployment."""
    max_latency_ms: float = 100.0        # Maximum acceptable latency
    p95_latency_ms: float = 50.0         # 95th percentile latency target
    p99_latency_ms: float = 80.0         # 99th percentile latency target
    max_memory_mb: float = 2048.0        # Maximum memory usage
    min_throughput_qps: float = 10.0     # Minimum queries per second
    max_warmup_time_s: float = 30.0      # Maximum warmup time
    target_accuracy: float = 0.85        # Minimum accuracy requirement


@dataclass
class InferenceBenchmarkConfig:
    """Configuration for inference benchmarking."""
    # Test parameters
    warmup_iterations: int = 50
    measure_iterations: int = 200
    batch_sizes: List[int] = None
    sequence_lengths: List[int] = None

    # Profiling options
    profile_memory: bool = True
    profile_cpu: bool = True
    profile_gpu: bool = True
    memory_tracking_interval: float = 0.01  # seconds

    # Streaming evaluation
    enable_streaming: bool = True
    streaming_window_size: int = 100
    streaming_latency_target_ms: float = 50.0

    # Output options
    save_results: bool = True
    output_dir: str = "./benchmark_results"
    generate_plots: bool = True

    # Performance targets
    performance_targets: PerformanceTarget = None

    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 2, 4, 8, 16, 32]
        if self.sequence_lengths is None:
            self.sequence_lengths = [512, 1024, 2048, 4096]
        if self.performance_targets is None:
            self.performance_targets = PerformanceTarget()


@dataclass
class LatencyMetrics:
    """Latency measurement results."""
    mean: float
    std: float
    min: float
    max: float
    p50: float
    p95: float
    p99: float
    p999: float

    @classmethod
    def from_measurements(cls, latencies: List[float]) -> 'LatencyMetrics':
        """Create metrics from list of latency measurements."""
        latencies_np = np.array(latencies)
        return cls(
            mean=float(np.mean(latencies_np)),
            std=float(np.std(latencies_np)),
            min=float(np.min(latencies_np)),
            max=float(np.max(latencies_np)),
            p50=float(np.percentile(latencies_np, 50)),
            p95=float(np.percentile(latencies_np, 95)),
            p99=float(np.percentile(latencies_np, 99)),
            p999=float(np.percentile(latencies_np, 99.9))
        )


@dataclass
class MemoryMetrics:
    """Memory usage metrics."""
    peak_cpu_mb: float
    peak_gpu_mb: float
    avg_cpu_mb: float
    avg_gpu_mb: float
    cpu_samples: List[float]
    gpu_samples: List[float]

    @classmethod
    def from_samples(cls, cpu_samples: List[float], gpu_samples: List[float]) -> 'MemoryMetrics':
        """Create metrics from memory samples."""
        return cls(
            peak_cpu_mb=max(cpu_samples) if cpu_samples else 0.0,
            peak_gpu_mb=max(gpu_samples) if gpu_samples else 0.0,
            avg_cpu_mb=np.mean(cpu_samples) if cpu_samples else 0.0,
            avg_gpu_mb=np.mean(gpu_samples) if gpu_samples else 0.0,
            cpu_samples=cpu_samples,
            gpu_samples=gpu_samples
        )


@dataclass
class BenchmarkResult:
    """Complete benchmark result for a single configuration."""
    batch_size: int
    sequence_length: int
    latency_metrics: LatencyMetrics
    memory_metrics: MemoryMetrics
    throughput_qps: float
    meets_targets: bool
    target_violations: List[str]


class MemoryMonitor:
    """Real-time memory monitoring during inference."""

    def __init__(self, interval: float = 0.01):
        self.interval = interval
        self.cpu_samples = []
        self.gpu_samples = []
        self.running = False
        self.thread = None

    def start(self):
        """Start memory monitoring."""
        self.running = True
        self.cpu_samples.clear()
        self.gpu_samples.clear()
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()

    def stop(self):
        """Stop memory monitoring."""
        self.running = False
        if self.thread:
            self.thread.join()

    def _monitor_loop(self):
        """Memory monitoring loop."""
        while self.running:
            # CPU memory
            process = psutil.Process()
            cpu_memory_mb = process.memory_info().rss / 1024 / 1024
            self.cpu_samples.append(cpu_memory_mb)

            # GPU memory
            if torch.cuda.is_available():
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                self.gpu_samples.append(gpu_memory_mb)
            else:
                self.gpu_samples.append(0.0)

            time.sleep(self.interval)

    def get_metrics(self) -> MemoryMetrics:
        """Get memory metrics from monitoring."""
        return MemoryMetrics.from_samples(self.cpu_samples, self.gpu_samples)


@contextmanager
def cuda_timer():
    """Context manager for precise CUDA timing."""
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        torch.cuda.synchronize()
        start_event.record()

        yield lambda: start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds

        end_event.record()
        torch.cuda.synchronize()
    else:
        start_time = time.time()
        yield lambda: time.time() - start_time


class StreamingEvaluator:
    """Evaluates streaming inference performance."""

    def __init__(self, window_size: int = 100, latency_target_ms: float = 50.0):
        self.window_size = window_size
        self.latency_target_ms = latency_target_ms
        self.latency_window = deque(maxlen=window_size)
        self.accuracy_window = deque(maxlen=window_size)
        self.violation_count = 0
        self.total_inferences = 0

    def add_result(self, latency_ms: float, is_correct: bool):
        """Add a streaming inference result."""
        self.latency_window.append(latency_ms)
        self.accuracy_window.append(1.0 if is_correct else 0.0)
        self.total_inferences += 1

        if latency_ms > self.latency_target_ms:
            self.violation_count += 1

    def get_current_metrics(self) -> Dict[str, float]:
        """Get current streaming metrics."""
        if not self.latency_window:
            return {}

        return {
            'current_latency_p95': np.percentile(list(self.latency_window), 95),
            'current_accuracy': np.mean(list(self.accuracy_window)),
            'violation_rate': self.violation_count / self.total_inferences,
            'avg_latency': np.mean(list(self.latency_window))
        }

    def is_meeting_targets(self) -> bool:
        """Check if currently meeting streaming targets."""
        metrics = self.get_current_metrics()
        if not metrics:
            return True

        return (metrics['current_latency_p95'] <= self.latency_target_ms and
                metrics['violation_rate'] <= 0.05)  # Allow 5% violations


class InferenceBenchmark:
    """
    Comprehensive inference benchmarking suite.
    """

    def __init__(self, config: InferenceBenchmarkConfig):
        self.config = config
        self.results = []
        self.memory_monitor = MemoryMonitor(config.memory_tracking_interval)

        # Create output directory
        if config.save_results:
            Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    def benchmark_model(
        self,
        model: nn.Module,
        input_generator: Callable[[int, int], torch.Tensor],
        target_generator: Optional[Callable[[int, int], torch.Tensor]] = None,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """
        Comprehensive model benchmarking.

        Args:
            model: PyTorch model to benchmark
            input_generator: Function that generates inputs (batch_size, seq_len) -> tensor
            target_generator: Optional function for accuracy evaluation
            model_name: Name for the model being benchmarked

        Returns:
            Complete benchmark results
        """
        print(f"Benchmarking {model_name}...")

        model.eval()
        torch.cuda.empty_cache()

        all_results = []

        for batch_size in self.config.batch_sizes:
            for seq_len in self.config.sequence_lengths:
                print(f"  Testing batch_size={batch_size}, seq_len={seq_len}")

                result = self._benchmark_configuration(
                    model, input_generator, target_generator,
                    batch_size, seq_len
                )
                all_results.append(result)

        # Aggregate results
        aggregated = self._aggregate_results(all_results, model_name)

        # Save results
        if self.config.save_results:
            self._save_results(aggregated, model_name)

        # Generate plots
        if self.config.generate_plots:
            self._generate_plots(all_results, model_name)

        return aggregated

    def _benchmark_configuration(
        self,
        model: nn.Module,
        input_generator: Callable,
        target_generator: Optional[Callable],
        batch_size: int,
        seq_len: int
    ) -> BenchmarkResult:
        """Benchmark a specific configuration."""

        # Generate test data
        sample_input = input_generator(batch_size, seq_len)
        if torch.cuda.is_available():
            sample_input = sample_input.cuda()

        # Warmup
        with torch.no_grad():
            for _ in range(self.config.warmup_iterations):
                _ = model(sample_input)

        torch.cuda.synchronize()
        gc.collect()

        # Start memory monitoring
        if self.config.profile_memory:
            self.memory_monitor.start()

        # Measure latency
        latencies = []
        accuracies = []

        with torch.no_grad():
            for i in range(self.config.measure_iterations):
                # Generate fresh input for each iteration
                test_input = input_generator(batch_size, seq_len)
                if torch.cuda.is_available():
                    test_input = test_input.cuda()

                # Time inference
                with cuda_timer() as timer:
                    output = model(test_input)

                latency = timer() * 1000  # Convert to milliseconds
                latencies.append(latency)

                # Measure accuracy if targets provided
                if target_generator is not None:
                    target = target_generator(batch_size, seq_len)
                    if torch.cuda.is_available():
                        target = target.cuda()

                    # Simple accuracy for demonstration
                    pred = torch.argmax(output, dim=-1) if output.dim() > 1 else output.round()
                    correct = (pred == target).float().mean().item()
                    accuracies.append(correct)

        # Stop memory monitoring
        if self.config.profile_memory:
            self.memory_monitor.stop()
            memory_metrics = self.memory_monitor.get_metrics()
        else:
            memory_metrics = MemoryMetrics.from_samples([], [])

        # Calculate metrics
        latency_metrics = LatencyMetrics.from_measurements(latencies)
        throughput = batch_size / (latency_metrics.mean / 1000)  # QPS

        # Check performance targets
        meets_targets, violations = self._check_performance_targets(
            latency_metrics, memory_metrics, throughput
        )

        return BenchmarkResult(
            batch_size=batch_size,
            sequence_length=seq_len,
            latency_metrics=latency_metrics,
            memory_metrics=memory_metrics,
            throughput_qps=throughput,
            meets_targets=meets_targets,
            target_violations=violations
        )

    def _check_performance_targets(
        self,
        latency_metrics: LatencyMetrics,
        memory_metrics: MemoryMetrics,
        throughput: float
    ) -> Tuple[bool, List[str]]:
        """Check if performance meets targets."""
        targets = self.config.performance_targets
        violations = []

        if latency_metrics.mean > targets.max_latency_ms:
            violations.append(f"Mean latency {latency_metrics.mean:.2f}ms > {targets.max_latency_ms}ms")

        if latency_metrics.p95 > targets.p95_latency_ms:
            violations.append(f"P95 latency {latency_metrics.p95:.2f}ms > {targets.p95_latency_ms}ms")

        if latency_metrics.p99 > targets.p99_latency_ms:
            violations.append(f"P99 latency {latency_metrics.p99:.2f}ms > {targets.p99_latency_ms}ms")

        if memory_metrics.peak_gpu_mb > targets.max_memory_mb:
            violations.append(f"GPU memory {memory_metrics.peak_gpu_mb:.2f}MB > {targets.max_memory_mb}MB")

        if throughput < targets.min_throughput_qps:
            violations.append(f"Throughput {throughput:.2f} QPS < {targets.min_throughput_qps} QPS")

        return len(violations) == 0, violations

    def _aggregate_results(self, results: List[BenchmarkResult], model_name: str) -> Dict[str, Any]:
        """Aggregate benchmark results."""

        # Overall performance summary
        all_latencies = []
        all_throughputs = []
        total_violations = []

        for result in results:
            all_latencies.extend([
                result.latency_metrics.mean,
                result.latency_metrics.p95,
                result.latency_metrics.p99
            ])
            all_throughputs.append(result.throughput_qps)
            total_violations.extend(result.target_violations)

        # Best configurations
        best_latency = min(results, key=lambda r: r.latency_metrics.p95)
        best_throughput = max(results, key=lambda r: r.throughput_qps)

        # Performance grade
        passing_configs = sum(1 for r in results if r.meets_targets)
        performance_grade = passing_configs / len(results)

        return {
            'model_name': model_name,
            'summary': {
                'total_configurations': len(results),
                'passing_configurations': passing_configs,
                'performance_grade': performance_grade,
                'avg_latency_ms': np.mean(all_latencies),
                'avg_throughput_qps': np.mean(all_throughputs),
                'total_violations': len(total_violations)
            },
            'best_configurations': {
                'lowest_latency': asdict(best_latency),
                'highest_throughput': asdict(best_throughput)
            },
            'all_results': [asdict(r) for r in results],
            'violations': total_violations
        }

    def _save_results(self, results: Dict[str, Any], model_name: str):
        """Save benchmark results to file."""
        output_file = Path(self.config.output_dir) / f"{model_name}_benchmark.json"

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"Results saved to {output_file}")

    def _generate_plots(self, results: List[BenchmarkResult], model_name: str):
        """Generate performance visualization plots."""
        try:
            # Set up plotting style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'Performance Analysis: {model_name}', fontsize=16)

            # Extract data for plotting
            batch_sizes = [r.batch_size for r in results]
            latencies_p95 = [r.latency_metrics.p95 for r in results]
            throughputs = [r.throughput_qps for r in results]
            memory_usage = [r.memory_metrics.peak_gpu_mb for r in results]

            # Plot 1: Latency vs Batch Size
            axes[0, 0].scatter(batch_sizes, latencies_p95, alpha=0.7)
            axes[0, 0].axhline(y=self.config.performance_targets.p95_latency_ms,
                              color='r', linestyle='--', label='P95 Target')
            axes[0, 0].set_xlabel('Batch Size')
            axes[0, 0].set_ylabel('P95 Latency (ms)')
            axes[0, 0].set_title('Latency vs Batch Size')
            axes[0, 0].legend()

            # Plot 2: Throughput vs Batch Size
            axes[0, 1].scatter(batch_sizes, throughputs, alpha=0.7, color='green')
            axes[0, 1].axhline(y=self.config.performance_targets.min_throughput_qps,
                              color='r', linestyle='--', label='Throughput Target')
            axes[0, 1].set_xlabel('Batch Size')
            axes[0, 1].set_ylabel('Throughput (QPS)')
            axes[0, 1].set_title('Throughput vs Batch Size')
            axes[0, 1].legend()

            # Plot 3: Memory Usage vs Batch Size
            axes[1, 0].scatter(batch_sizes, memory_usage, alpha=0.7, color='orange')
            axes[1, 0].axhline(y=self.config.performance_targets.max_memory_mb,
                              color='r', linestyle='--', label='Memory Target')
            axes[1, 0].set_xlabel('Batch Size')
            axes[1, 0].set_ylabel('Peak GPU Memory (MB)')
            axes[1, 0].set_title('Memory Usage vs Batch Size')
            axes[1, 0].legend()

            # Plot 4: Latency Distribution
            all_latencies = []
            for r in results:
                all_latencies.extend([r.latency_metrics.mean, r.latency_metrics.p95, r.latency_metrics.p99])

            axes[1, 1].hist(all_latencies, bins=20, alpha=0.7, color='purple')
            axes[1, 1].axvline(x=self.config.performance_targets.max_latency_ms,
                              color='r', linestyle='--', label='Max Latency Target')
            axes[1, 1].set_xlabel('Latency (ms)')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Latency Distribution')
            axes[1, 1].legend()

            plt.tight_layout()

            # Save plot
            plot_file = Path(self.config.output_dir) / f"{model_name}_performance_plots.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Performance plots saved to {plot_file}")

        except Exception as e:
            print(f"Failed to generate plots: {e}")

    def streaming_benchmark(
        self,
        model: nn.Module,
        input_generator: Callable,
        target_generator: Optional[Callable] = None,
        duration_seconds: int = 60
    ) -> Dict[str, Any]:
        """
        Benchmark streaming inference performance.

        Args:
            model: Model to benchmark
            input_generator: Function to generate inputs
            target_generator: Optional function to generate targets for accuracy
            duration_seconds: How long to run the streaming test

        Returns:
            Streaming performance results
        """
        if not self.config.enable_streaming:
            return {}

        print(f"Running streaming benchmark for {duration_seconds} seconds...")

        model.eval()
        evaluator = StreamingEvaluator(
            self.config.streaming_window_size,
            self.config.streaming_latency_target_ms
        )

        start_time = time.time()
        inference_count = 0

        with torch.no_grad():
            while time.time() - start_time < duration_seconds:
                # Generate single sample
                test_input = input_generator(1, 1024)  # Single sample, fixed length
                if torch.cuda.is_available():
                    test_input = test_input.cuda()

                # Time inference
                with cuda_timer() as timer:
                    output = model(test_input)

                latency_ms = timer() * 1000

                # Evaluate accuracy if possible
                is_correct = True  # Default to True if no target generator
                if target_generator is not None:
                    target = target_generator(1, 1024)
                    if torch.cuda.is_available():
                        target = target.cuda()

                    pred = torch.argmax(output, dim=-1) if output.dim() > 1 else output.round()
                    is_correct = (pred == target).all().item()

                # Add result to evaluator
                evaluator.add_result(latency_ms, is_correct)
                inference_count += 1

                # Optional: Add small delay to simulate real-world conditions
                # time.sleep(0.001)

        total_time = time.time() - start_time
        final_metrics = evaluator.get_current_metrics()

        return {
            'duration_seconds': total_time,
            'total_inferences': inference_count,
            'inferences_per_second': inference_count / total_time,
            'meeting_targets': evaluator.is_meeting_targets(),
            'metrics': final_metrics,
            'target_latency_ms': self.config.streaming_latency_target_ms
        }


# Example usage and testing
if __name__ == "__main__":
    # Test configuration
    config = InferenceBenchmarkConfig(
        warmup_iterations=10,
        measure_iterations=50,
        batch_sizes=[1, 4, 8],
        sequence_lengths=[512, 1024],
        save_results=False,
        generate_plots=False
    )

    benchmark = InferenceBenchmark(config)

    # Create dummy model and data generators
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv1d(64, 128, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(128, 10)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x).squeeze(-1)
            return self.fc(x)

    def input_generator(batch_size: int, seq_len: int) -> torch.Tensor:
        return torch.randn(batch_size, 64, seq_len)

    def target_generator(batch_size: int, seq_len: int) -> torch.Tensor:
        return torch.randint(0, 10, (batch_size,))

    model = DummyModel()
    if torch.cuda.is_available():
        model = model.cuda()

    # Run benchmark
    print("Running inference benchmark...")
    results = benchmark.benchmark_model(
        model, input_generator, target_generator, "dummy_model"
    )

    print(f"Benchmark completed. Performance grade: {results['summary']['performance_grade']:.2%}")
    print(f"Average latency: {results['summary']['avg_latency_ms']:.2f}ms")
    print(f"Average throughput: {results['summary']['avg_throughput_qps']:.2f} QPS")

    # Test streaming benchmark
    print("\nRunning streaming benchmark...")
    streaming_results = benchmark.streaming_benchmark(
        model, input_generator, target_generator, duration_seconds=10
    )

    if streaming_results:
        print(f"Streaming performance: {streaming_results['inferences_per_second']:.1f} inferences/sec")
        print(f"Meeting targets: {streaming_results['meeting_targets']}")

    print("\nAll tests completed!")
