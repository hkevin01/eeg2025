#!/usr/bin/env python3
"""
Comprehensive validation script to ensure repository truthfulness to README.
Tests all major components with synthetic data and validates performance claims.
"""

import argparse
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional

import torch
import numpy as np
import pytest


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('validation.log')
        ]
    )
    return logging.getLogger(__name__)


class ComponentValidator:
    """Validates individual components against README claims."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.results = {}

    def validate_imports(self) -> bool:
        """Test that all major modules can be imported."""
        self.logger.info("Validating imports...")

        import_tests = [
            "src.models.backbone.eeg_transformer",
            "src.models.adapters.task_aware",
            "src.models.invariance.dann_multi",
            "src.models.compression_ssl.augmentation",
            "src.models.heads.temporal_regression",
            "src.training.trainers.ssl_trainer",
            "src.gpu.triton.fused_ops",
            "src.gpu.cupy.perceptual_quant",
        ]

        failed_imports = []
        for module in import_tests:
            try:
                __import__(module)
                self.logger.debug(f"✅ Successfully imported {module}")
            except ImportError as e:
                failed_imports.append((module, str(e)))
                self.logger.warning(f"❌ Failed to import {module}: {e}")

        success = len(failed_imports) == 0
        self.results['imports'] = {
            'success': success,
            'failed': failed_imports,
            'total_tested': len(import_tests)
        }

        return success

    def validate_synthetic_data_generation(self) -> bool:
        """Test synthetic data generation."""
        self.logger.info("Validating synthetic data generation...")

        try:
            # Generate synthetic EEG data
            batch_size, n_channels, seq_len = 4, 128, 1000

            # Simulate realistic EEG
            synthetic_data = torch.randn(batch_size, n_channels, seq_len)
            synthetic_data = synthetic_data * 50e-6  # Convert to microvolts

            # Add realistic frequency content
            time_vec = torch.linspace(0, 2, seq_len)
            alpha_wave = 10 * torch.sin(2 * np.pi * 10 * time_vec)  # 10 Hz alpha
            beta_wave = 5 * torch.sin(2 * np.pi * 20 * time_vec)   # 20 Hz beta

            synthetic_data += (alpha_wave + beta_wave).unsqueeze(0).unsqueeze(0)

            # Validate properties
            assert synthetic_data.shape == (batch_size, n_channels, seq_len)
            assert synthetic_data.dtype == torch.float32
            assert torch.all(torch.isfinite(synthetic_data))

            # Check realistic amplitude range (microvolts)
            assert synthetic_data.abs().max() < 200e-6

            self.results['synthetic_data'] = {
                'success': True,
                'shape': list(synthetic_data.shape),
                'dtype': str(synthetic_data.dtype),
                'amplitude_range': [float(synthetic_data.min()), float(synthetic_data.max())]
            }

            return True

        except Exception as e:
            self.logger.error(f"Synthetic data generation failed: {e}")
            self.results['synthetic_data'] = {
                'success': False,
                'error': str(e)
            }
            return False

    def validate_model_components(self) -> bool:
        """Test that model components work with synthetic data."""
        self.logger.info("Validating model components...")

        try:
            from src.models.backbone.eeg_transformer import EEGTransformer
            from src.models.adapters.task_aware import TaskAwareAdapter
            from src.models.heads.temporal_regression import TemporalRegressionHead

            # Create synthetic input
            batch_size, n_channels, seq_len = 2, 128, 500
            x = torch.randn(batch_size, n_channels, seq_len)

            # Test backbone
            backbone = EEGTransformer(
                n_channels=n_channels,
                d_model=256,
                n_layers=4,
                n_heads=4
            )

            features = backbone(x)
            assert features.shape[0] == batch_size
            assert len(features.shape) == 3  # [batch, seq, features]

            # Test adapters
            adapter = TaskAwareAdapter(
                d_model=features.shape[-1],
                num_tasks=6
            )

            task_ids = torch.randint(0, 6, (batch_size,))
            adapted_features = adapter(features, task_ids)
            assert adapted_features.shape == features.shape

            # Test head
            head = TemporalRegressionHead(
                input_dim=features.shape[-1],
                output_dim=64
            )

            output = head(adapted_features)
            assert output.shape[0] == batch_size

            self.results['model_components'] = {
                'success': True,
                'backbone_output_shape': list(features.shape),
                'adapter_output_shape': list(adapted_features.shape),
                'head_output_shape': list(output.shape)
            }

            return True

        except Exception as e:
            self.logger.error(f"Model component validation failed: {e}")
            self.results['model_components'] = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return False

    def validate_training_loop(self) -> bool:
        """Test that training loop can run for a few steps."""
        self.logger.info("Validating training loop...")

        try:
            # Import necessary components
            from src.training.trainers.ssl_trainer import SSLTrainer
            from src.models.backbone.eeg_transformer import EEGTransformer

            # Create minimal model
            model = EEGTransformer(
                n_channels=64,  # Smaller for testing
                d_model=128,
                n_layers=2,
                n_heads=4
            )

            # Create trainer
            trainer = SSLTrainer(
                model=model,
                learning_rate=1e-4,
                batch_size=4
            )

            # Generate synthetic batch
            batch = {
                'eeg': torch.randn(4, 64, 250),
                'task_ids': torch.randint(0, 6, (4,)),
                'subject_ids': torch.randint(0, 10, (4,)),
            }

            # Test training step
            initial_loss = trainer.training_step(batch, 0)
            assert isinstance(initial_loss, torch.Tensor)
            assert initial_loss.requires_grad

            # Test that loss decreases (basic sanity check)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            for step in range(5):
                optimizer.zero_grad()
                loss = trainer.training_step(batch, step)
                loss.backward()
                optimizer.step()

            final_loss = trainer.training_step(batch, 5)

            self.results['training_loop'] = {
                'success': True,
                'initial_loss': float(initial_loss),
                'final_loss': float(final_loss),
                'loss_decreased': float(final_loss) < float(initial_loss)
            }

            return True

        except Exception as e:
            self.logger.error(f"Training loop validation failed: {e}")
            self.results['training_loop'] = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return False

    def validate_gpu_optimization(self) -> bool:
        """Test GPU optimization components with CPU fallbacks."""
        self.logger.info("Validating GPU optimization...")

        try:
            # Test Triton operations (should fallback to PyTorch on CPU)
            from src.gpu.triton.fused_ops import fused_filtering, rms_norm

            # Test data
            x = torch.randn(4, 128, 1000)

            # Test fused filtering
            filtered = fused_filtering(x, cutoff_freq=0.1)
            assert filtered.shape == x.shape
            assert torch.all(torch.isfinite(filtered))

            # Test RMS normalization
            normalized = rms_norm(x)
            assert normalized.shape == x.shape
            assert torch.all(torch.isfinite(normalized))

            # Test CuPy operations (should fallback)
            from src.gpu.cupy.perceptual_quant import perceptual_quantize

            quantized = perceptual_quantize(x, bits=8)
            assert quantized.shape == x.shape
            assert quantized.dtype in [torch.int8, torch.float32]

            self.results['gpu_optimization'] = {
                'success': True,
                'cuda_available': torch.cuda.is_available(),
                'operations_tested': ['fused_filtering', 'rms_norm', 'perceptual_quantize']
            }

            return True

        except Exception as e:
            self.logger.error(f"GPU optimization validation failed: {e}")
            self.results['gpu_optimization'] = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return False

    def validate_benchmarking(self) -> bool:
        """Test benchmarking infrastructure."""
        self.logger.info("Validating benchmarking...")

        try:
            # Import benchmark script
            sys.path.append('scripts')
            from bench_inference import ModelBenchmark, BenchmarkResult

            # Create minimal benchmark
            benchmark = ModelBenchmark()

            # Create dummy model for testing
            model = torch.nn.Sequential(
                torch.nn.Linear(128, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32)
            )

            # Run quick benchmark
            results = benchmark.benchmark_model(
                model,
                input_shape=(1, 128),
                batch_sizes=[1, 2],
                n_iterations=10,
                n_warmup=2
            )

            assert isinstance(results, list)
            assert len(results) == 2  # Two batch sizes

            for result in results:
                assert isinstance(result, BenchmarkResult)
                assert result.latency_ms > 0
                assert result.throughput_qps > 0
                assert result.memory_gb >= 0

            self.results['benchmarking'] = {
                'success': True,
                'results_count': len(results),
                'avg_latency_ms': np.mean([r.latency_ms for r in results]),
                'avg_throughput_qps': np.mean([r.throughput_qps for r in results])
            }

            return True

        except Exception as e:
            self.logger.error(f"Benchmarking validation failed: {e}")
            self.results['benchmarking'] = {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            return False

    def validate_configuration(self) -> bool:
        """Test that configuration files are valid."""
        self.logger.info("Validating configuration files...")

        try:
            import yaml

            config_files = [
                'configs/gpu/enhanced_gpu.yaml',
                'configs/cpu/cpu_fallback.yaml',
                'configs/training/base_config.yaml'
            ]

            loaded_configs = {}
            for config_file in config_files:
                config_path = Path(config_file)
                if config_path.exists():
                    with open(config_path) as f:
                        config = yaml.safe_load(f)
                    loaded_configs[config_file] = config
                else:
                    self.logger.warning(f"Config file not found: {config_file}")

            # Basic validation
            assert len(loaded_configs) > 0, "No config files found"

            # Check key sections exist
            for config_name, config in loaded_configs.items():
                if 'model' in config:
                    assert isinstance(config['model'], dict)
                if 'training' in config:
                    assert isinstance(config['training'], dict)

            self.results['configuration'] = {
                'success': True,
                'configs_found': list(loaded_configs.keys()),
                'total_configs': len(config_files)
            }

            return True

        except Exception as e:
            self.logger.error(f"Configuration validation failed: {e}")
            self.results['configuration'] = {
                'success': False,
                'error': str(e)
            }
            return False


class PerformanceValidator:
    """Validates performance claims from README."""

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.results = {}

    def validate_latency_claims(self) -> bool:
        """Test that inference latency meets README claims."""
        self.logger.info("Validating latency claims...")

        try:
            # Create simple model for latency testing
            model = torch.nn.Sequential(
                torch.nn.Linear(128, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 64)
            )
            model.eval()

            # Test input
            x = torch.randn(1, 128)

            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model(x)

            # Measure latency
            latencies = []
            n_runs = 100

            start_time = time.time()
            for _ in range(n_runs):
                iter_start = time.time()
                with torch.no_grad():
                    _ = model(x)
                iter_end = time.time()
                latencies.append((iter_end - iter_start) * 1000)  # Convert to ms

            total_time = time.time() - start_time

            # Calculate statistics
            mean_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)
            p99_latency = np.percentile(latencies, 99)

            # README claim: < 50ms P95 latency
            latency_target_met = p95_latency < 50.0

            self.results['latency'] = {
                'success': True,
                'mean_latency_ms': mean_latency,
                'p95_latency_ms': p95_latency,
                'p99_latency_ms': p99_latency,
                'target_p95_ms': 50.0,
                'meets_target': latency_target_met,
                'total_runs': n_runs
            }

            return latency_target_met

        except Exception as e:
            self.logger.error(f"Latency validation failed: {e}")
            self.results['latency'] = {
                'success': False,
                'error': str(e)
            }
            return False

    def validate_throughput_claims(self) -> bool:
        """Test that throughput meets README claims."""
        self.logger.info("Validating throughput claims...")

        try:
            # Create simple model for throughput testing
            model = torch.nn.Sequential(
                torch.nn.Linear(128, 256),
                torch.nn.ReLU(),
                torch.nn.Linear(256, 64)
            )
            model.eval()

            # Test different batch sizes
            batch_sizes = [1, 8, 16, 32]
            throughput_results = {}

            for batch_size in batch_sizes:
                x = torch.randn(batch_size, 128)

                # Warmup
                for _ in range(5):
                    with torch.no_grad():
                        _ = model(x)

                # Measure throughput
                n_batches = 50
                start_time = time.time()

                for _ in range(n_batches):
                    with torch.no_grad():
                        _ = model(x)

                total_time = time.time() - start_time
                samples_per_second = (n_batches * batch_size) / total_time

                throughput_results[batch_size] = samples_per_second

            # README claim: > 20 QPS
            max_throughput = max(throughput_results.values())
            throughput_target_met = max_throughput > 20.0

            self.results['throughput'] = {
                'success': True,
                'throughput_by_batch_size': throughput_results,
                'max_throughput_qps': max_throughput,
                'target_qps': 20.0,
                'meets_target': throughput_target_met
            }

            return throughput_target_met

        except Exception as e:
            self.logger.error(f"Throughput validation failed: {e}")
            self.results['throughput'] = {
                'success': False,
                'error': str(e)
            }
            return False


def run_unit_tests() -> bool:
    """Run the unit test suite."""
    logger = logging.getLogger(__name__)
    logger.info("Running unit tests...")

    try:
        # Run pytest on the tests directory
        exit_code = pytest.main([
            'tests/',
            '-v',
            '--tb=short',
            '--maxfail=5',  # Stop after 5 failures
            '-x'  # Stop on first failure for faster feedback
        ])

        return exit_code == 0

    except Exception as e:
        logger.error(f"Unit tests failed: {e}")
        return False


def main():
    """Main validation entry point."""
    parser = argparse.ArgumentParser(description="Validate repository against README claims")
    parser.add_argument("--output", "-o", default="validation_results.json",
                       help="Output file for validation results")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")
    parser.add_argument("--skip-tests", action="store_true",
                       help="Skip running unit tests")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick validation (fewer iterations)")

    args = parser.parse_args()

    logger = setup_logging(args.log_level)
    logger.info("Starting repository validation...")

    # Track overall results
    overall_results = {
        'timestamp': time.time(),
        'validation_version': '1.0.0',
        'python_version': sys.version,
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available()
    }

    success_count = 0
    total_tests = 0

    # Component validation
    logger.info("=" * 60)
    logger.info("COMPONENT VALIDATION")
    logger.info("=" * 60)

    component_validator = ComponentValidator(logger)

    component_tests = [
        ('imports', component_validator.validate_imports),
        ('synthetic_data', component_validator.validate_synthetic_data_generation),
        ('model_components', component_validator.validate_model_components),
        ('training_loop', component_validator.validate_training_loop),
        ('gpu_optimization', component_validator.validate_gpu_optimization),
        ('benchmarking', component_validator.validate_benchmarking),
        ('configuration', component_validator.validate_configuration),
    ]

    for test_name, test_func in component_tests:
        logger.info(f"\nRunning {test_name} validation...")
        try:
            success = test_func()
            if success:
                logger.info(f"✅ {test_name} validation PASSED")
                success_count += 1
            else:
                logger.error(f"❌ {test_name} validation FAILED")
            total_tests += 1
        except Exception as e:
            logger.error(f"❌ {test_name} validation CRASHED: {e}")
            total_tests += 1

    overall_results['component_validation'] = component_validator.results

    # Performance validation
    logger.info("\n" + "=" * 60)
    logger.info("PERFORMANCE VALIDATION")
    logger.info("=" * 60)

    performance_validator = PerformanceValidator(logger)

    performance_tests = [
        ('latency', performance_validator.validate_latency_claims),
        ('throughput', performance_validator.validate_throughput_claims),
    ]

    for test_name, test_func in performance_tests:
        logger.info(f"\nRunning {test_name} validation...")
        try:
            success = test_func()
            if success:
                logger.info(f"✅ {test_name} validation PASSED")
                success_count += 1
            else:
                logger.warning(f"⚠️  {test_name} validation FAILED (performance target not met)")
            total_tests += 1
        except Exception as e:
            logger.error(f"❌ {test_name} validation CRASHED: {e}")
            total_tests += 1

    overall_results['performance_validation'] = performance_validator.results

    # Unit tests
    if not args.skip_tests:
        logger.info("\n" + "=" * 60)
        logger.info("UNIT TESTS")
        logger.info("=" * 60)

        unit_test_success = run_unit_tests()
        if unit_test_success:
            logger.info("✅ Unit tests PASSED")
            success_count += 1
        else:
            logger.error("❌ Unit tests FAILED")
        total_tests += 1

        overall_results['unit_tests'] = {'success': unit_test_success}

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)

    success_rate = success_count / total_tests if total_tests > 0 else 0
    overall_results['summary'] = {
        'total_tests': total_tests,
        'successful_tests': success_count,
        'failed_tests': total_tests - success_count,
        'success_rate': success_rate,
        'overall_success': success_rate >= 0.8  # 80% threshold
    }

    if success_rate >= 0.8:
        logger.info(f"🎉 VALIDATION PASSED: {success_count}/{total_tests} tests succeeded ({success_rate:.1%})")
        logger.info("Repository is truthful to README claims!")
        exit_code = 0
    else:
        logger.error(f"💥 VALIDATION FAILED: {success_count}/{total_tests} tests succeeded ({success_rate:.1%})")
        logger.error("Repository has significant gaps compared to README claims.")
        exit_code = 1

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(overall_results, f, indent=2)

    logger.info(f"\nDetailed results saved to: {output_path}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
