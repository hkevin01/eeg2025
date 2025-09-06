"""
GPU optimization utilities for efficient training and inference.

This module provides mixed precision training, model compilation, memory profiling,
and performance benchmarking capabilities for maximum GPU utilization.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
import time
import psutil
import GPUtil
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np
import json
from dataclasses import dataclass, asdict
from pathlib import Path
import gc
import warnings
from contextlib import contextmanager
import functools


@dataclass
class PerformanceMetrics:
    """Performance metrics for model benchmarking."""
    avg_inference_time_ms: float
    p50_inference_time_ms: float
    p95_inference_time_ms: float
    p99_inference_time_ms: float
    throughput_samples_per_sec: float
    peak_memory_mb: float
    avg_memory_mb: float
    gpu_utilization_pct: float
    batch_size: int
    sequence_length: int
    device: str
    model_params: int
    model_flops: Optional[int] = None


class MemoryProfiler:
    """GPU memory profiler for tracking memory usage during training/inference."""

    def __init__(self, device: str = "auto"):
        """
        Initialize memory profiler.

        Args:
            device: Device to profile ('cuda', 'cpu', or 'auto')
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.reset()

    def reset(self):
        """Reset memory tracking."""
        self.memory_history = []
        self.peak_memory = 0.0

        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def snapshot(self) -> float:
        """Take a memory snapshot and return current usage in MB."""
        if self.device == "cuda" and torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated() / 1024 / 1024  # MB
            peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
        else:
            # CPU memory (less precise)
            process = psutil.Process()
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            peak_memory = current_memory

        self.memory_history.append(current_memory)
        self.peak_memory = max(self.peak_memory, peak_memory)

        return current_memory

    def get_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        if not self.memory_history:
            return {"avg_memory_mb": 0.0, "peak_memory_mb": 0.0}

        return {
            "avg_memory_mb": np.mean(self.memory_history),
            "peak_memory_mb": self.peak_memory,
            "min_memory_mb": np.min(self.memory_history),
            "max_memory_mb": np.max(self.memory_history)
        }


class LatencyProfiler:
    """Latency profiler for measuring inference performance."""

    def __init__(self, warmup_runs: int = 10, measurement_runs: int = 100):
        """
        Initialize latency profiler.

        Args:
            warmup_runs: Number of warmup runs
            measurement_runs: Number of measurement runs
        """
        self.warmup_runs = warmup_runs
        self.measurement_runs = measurement_runs
        self.latencies = []

    def reset(self):
        """Reset latency measurements."""
        self.latencies = []

    @contextmanager
    def measure(self):
        """Context manager for measuring latency."""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        start_time = time.perf_counter()
        yield

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000
        self.latencies.append(latency_ms)

    def benchmark_model(self,
                       model: nn.Module,
                       input_shape: Tuple[int, ...],
                       device: str = "cuda",
                       use_amp: bool = False) -> Dict[str, float]:
        """
        Benchmark model inference latency.

        Args:
            model: Model to benchmark
            input_shape: Input tensor shape
            device: Device to run on
            use_amp: Use mixed precision

        Returns:
            Latency statistics
        """
        model = model.to(device)
        model.eval()
        self.reset()

        # Create dummy input
        dummy_input = torch.randn(input_shape, device=device)

        # Warmup runs
        with torch.no_grad():
            for _ in range(self.warmup_runs):
                if use_amp and device == "cuda":
                    with autocast():
                        _ = model(dummy_input)
                else:
                    _ = model(dummy_input)

        # Measurement runs
        with torch.no_grad():
            for _ in range(self.measurement_runs):
                with self.measure():
                    if use_amp and device == "cuda":
                        with autocast():
                            _ = model(dummy_input)
                    else:
                        _ = model(dummy_input)

        # Calculate statistics
        latencies = np.array(self.latencies)
        return {
            "avg_inference_time_ms": np.mean(latencies),
            "p50_inference_time_ms": np.percentile(latencies, 50),
            "p95_inference_time_ms": np.percentile(latencies, 95),
            "p99_inference_time_ms": np.percentile(latencies, 99),
            "min_inference_time_ms": np.min(latencies),
            "max_inference_time_ms": np.max(latencies),
            "std_inference_time_ms": np.std(latencies),
            "throughput_samples_per_sec": input_shape[0] / (np.mean(latencies) / 1000)
        }


class OptimizedModel(nn.Module):
    """
    Wrapper for model optimization with AMP and compilation.
    """

    def __init__(self,
                 model: nn.Module,
                 use_amp: bool = True,
                 compile_mode: Optional[str] = "max-autotune",
                 device: str = "auto"):
        """
        Initialize optimized model wrapper.

        Args:
            model: Base model to optimize
            use_amp: Enable mixed precision training
            compile_mode: PyTorch 2.0 compilation mode ("max-autotune", "reduce-overhead", None)
            device: Device to run on
        """
        super().__init__()

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.base_model = model.to(self.device)
        self.use_amp = use_amp and (self.device.type == "cuda")
        self.compile_mode = compile_mode

        # Setup mixed precision
        if self.use_amp:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        # Compile model if requested and available
        if compile_mode and hasattr(torch, 'compile'):
            try:
                print(f"Compiling model with mode: {compile_mode}")
                self.model = torch.compile(self.base_model, mode=compile_mode)
                self.is_compiled = True
            except Exception as e:
                print(f"Model compilation failed: {e}")
                self.model = self.base_model
                self.is_compiled = False
        else:
            self.model = self.base_model
            self.is_compiled = False

    def forward(self, *args, **kwargs):
        """Forward pass with optional mixed precision."""
        if self.use_amp:
            with autocast():
                return self.model(*args, **kwargs)
        else:
            return self.model(*args, **kwargs)

    def training_step(self,
                     loss_fn: Callable,
                     optimizer: torch.optim.Optimizer,
                     *args, **kwargs) -> torch.Tensor:
        """
        Perform optimized training step with AMP.

        Args:
            loss_fn: Loss function
            optimizer: Optimizer
            *args, **kwargs: Arguments for forward pass

        Returns:
            Loss value
        """
        if self.use_amp:
            with autocast():
                loss = loss_fn(self(*args, **kwargs))

            # Scaled backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            loss = loss_fn(self(*args, **kwargs))
            loss.backward()
            optimizer.step()

        optimizer.zero_grad()
        return loss

    def get_optimization_info(self) -> Dict[str, Any]:
        """Get information about applied optimizations."""
        return {
            "use_amp": self.use_amp,
            "compile_mode": self.compile_mode,
            "is_compiled": self.is_compiled,
            "device": str(self.device),
            "scaler_enabled": self.scaler is not None
        }


class ModelBenchmarker:
    """Comprehensive model benchmarking suite."""

    def __init__(self, device: str = "auto"):
        """
        Initialize model benchmarker.

        Args:
            device: Device for benchmarking
        """
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.memory_profiler = MemoryProfiler(self.device)
        self.latency_profiler = LatencyProfiler()

    def benchmark_comprehensive(self,
                              model: nn.Module,
                              input_shapes: List[Tuple[int, ...]],
                              optimization_configs: List[Dict[str, Any]] = None,
                              save_path: Optional[str] = None) -> List[PerformanceMetrics]:
        """
        Run comprehensive benchmark across multiple configurations.

        Args:
            model: Model to benchmark
            input_shapes: List of input shapes to test
            optimization_configs: List of optimization configurations
            save_path: Path to save results

        Returns:
            List of performance metrics
        """
        if optimization_configs is None:
            optimization_configs = [
                {"use_amp": False, "compile_mode": None},
                {"use_amp": True, "compile_mode": None},
                {"use_amp": True, "compile_mode": "reduce-overhead"},
                {"use_amp": True, "compile_mode": "max-autotune"}
            ]

        results = []

        for input_shape in input_shapes:
            for config in optimization_configs:
                print(f"Benchmarking {input_shape} with config {config}")

                try:
                    # Create optimized model
                    opt_model = OptimizedModel(
                        model=model,
                        device=self.device,
                        **config
                    )

                    # Benchmark latency
                    latency_stats = self.latency_profiler.benchmark_model(
                        opt_model, input_shape, self.device, config.get("use_amp", False)
                    )

                    # Memory profiling
                    self.memory_profiler.reset()
                    self.memory_profiler.snapshot()  # Initial

                    # Forward pass for memory measurement
                    dummy_input = torch.randn(input_shape, device=self.device)
                    with torch.no_grad():
                        if config.get("use_amp", False) and self.device == "cuda":
                            with autocast():
                                _ = opt_model(dummy_input)
                        else:
                            _ = opt_model(dummy_input)

                    self.memory_profiler.snapshot()  # After inference
                    memory_stats = self.memory_profiler.get_stats()

                    # GPU utilization (if available)
                    gpu_util = 0.0
                    if self.device == "cuda" and torch.cuda.is_available():
                        try:
                            gpus = GPUtil.getGPUs()
                            if gpus:
                                gpu_util = gpus[0].load * 100
                        except:
                            pass

                    # Count parameters
                    model_params = sum(p.numel() for p in model.parameters())

                    # Create performance metrics
                    metrics = PerformanceMetrics(
                        avg_inference_time_ms=latency_stats["avg_inference_time_ms"],
                        p50_inference_time_ms=latency_stats["p50_inference_time_ms"],
                        p95_inference_time_ms=latency_stats["p95_inference_time_ms"],
                        p99_inference_time_ms=latency_stats["p99_inference_time_ms"],
                        throughput_samples_per_sec=latency_stats["throughput_samples_per_sec"],
                        peak_memory_mb=memory_stats["peak_memory_mb"],
                        avg_memory_mb=memory_stats["avg_memory_mb"],
                        gpu_utilization_pct=gpu_util,
                        batch_size=input_shape[0],
                        sequence_length=input_shape[-1] if len(input_shape) > 2 else 0,
                        device=self.device,
                        model_params=model_params
                    )

                    # Add optimization info
                    metrics_dict = asdict(metrics)
                    metrics_dict.update(config)
                    metrics_dict.update(opt_model.get_optimization_info())

                    results.append(metrics_dict)

                    print(f"  Latency: {latency_stats['avg_inference_time_ms']:.2f}ms")
                    print(f"  Throughput: {latency_stats['throughput_samples_per_sec']:.1f} samples/sec")
                    print(f"  Memory: {memory_stats['peak_memory_mb']:.1f}MB")

                except Exception as e:
                    print(f"  Benchmark failed: {e}")
                    continue

                finally:
                    # Cleanup
                    del opt_model
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

        # Save results if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)

            print(f"Benchmark results saved to {save_path}")

        return results

    def print_summary(self, results: List[Dict[str, Any]]):
        """Print benchmark summary table."""
        print("\n" + "="*100)
        print("BENCHMARK SUMMARY")
        print("="*100)

        # Headers
        headers = [
            "Config", "Batch", "Latency (ms)", "Throughput", "Memory (MB)",
            "P95 Latency", "GPU Util %"
        ]

        # Print header
        print(f"{'Config':<20} {'Batch':<8} {'Latency':<12} {'Throughput':<12} "
              f"{'Memory':<12} {'P95':<12} {'GPU %':<8}")
        print("-" * 100)

        # Print results
        for result in results:
            config_str = f"AMP:{result.get('use_amp', False)} "
            compile_mode = result.get('compile_mode', 'None')
            if compile_mode:
                config_str += f"Compile:{compile_mode[:8]}"
            else:
                config_str += "Compile:None"

            print(f"{config_str:<20} {result['batch_size']:<8} "
                  f"{result['avg_inference_time_ms']:<12.2f} "
                  f"{result['throughput_samples_per_sec']:<12.1f} "
                  f"{result['peak_memory_mb']:<12.1f} "
                  f"{result['p95_inference_time_ms']:<12.2f} "
                  f"{result['gpu_utilization_pct']:<8.1f}")


def create_fused_layernorm(dim: int, eps: float = 1e-5) -> nn.Module:
    """
    Create fused LayerNorm if available, otherwise fallback to standard.

    Args:
        dim: Layer dimension
        eps: Epsilon for numerical stability

    Returns:
        LayerNorm module (fused if available)
    """
    try:
        # Try to use Apex FusedLayerNorm
        from apex.normalization import FusedLayerNorm
        return FusedLayerNorm(dim, eps=eps)
    except ImportError:
        # Fallback to PyTorch native
        return nn.LayerNorm(dim, eps=eps)


def optimize_for_inference(model: nn.Module,
                          example_input: torch.Tensor,
                          device: str = "cuda") -> nn.Module:
    """
    Optimize model for inference with various techniques.

    Args:
        model: Model to optimize
        example_input: Example input for tracing/compilation
        device: Target device

    Returns:
        Optimized model
    """
    model = model.to(device)
    model.eval()

    # Apply optimizations
    optimized_model = model

    # 1. Try TorchScript tracing
    try:
        with torch.no_grad():
            traced_model = torch.jit.trace(model, example_input)
            traced_model = torch.jit.optimize_for_inference(traced_model)
            optimized_model = traced_model
            print("✅ Applied TorchScript optimization")
    except Exception as e:
        print(f"❌ TorchScript optimization failed: {e}")

    # 2. Try PyTorch 2.0 compilation
    if hasattr(torch, 'compile'):
        try:
            compiled_model = torch.compile(optimized_model, mode="max-autotune")
            optimized_model = compiled_model
            print("✅ Applied PyTorch 2.0 compilation")
        except Exception as e:
            print(f"❌ Compilation failed: {e}")

    return optimized_model


if __name__ == "__main__":
    # Example usage and testing
    from ..models.backbone import TemporalCNN

    # Create test model
    model = TemporalCNN(input_channels=19, num_layers=4)

    # Benchmark configurations
    input_shapes = [
        (1, 19, 1000),   # Single sample
        (8, 19, 1000),   # Small batch
        (32, 19, 1000),  # Medium batch
    ]

    # Run benchmark
    benchmarker = ModelBenchmarker()
    results = benchmarker.benchmark_comprehensive(model, input_shapes)
    benchmarker.print_summary(results)

    print("✅ GPU optimization testing completed!")
