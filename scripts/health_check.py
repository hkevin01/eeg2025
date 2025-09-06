#!/usr/bin/env python3
"""
Project Health Check Script
===========================

Quick health check to ensure the repository is in a good state.
Runs essential validations and reports overall project health.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import yaml


class HealthChecker:
    """Performs quick health checks on the project."""

    def __init__(self):
        self.checks = []
        self.results = {}

    def add_check(self, name: str, description: str, check_func) -> None:
        """Add a health check."""
        self.checks.append((name, description, check_func))

    def run_all_checks(self) -> Dict[str, any]:
        """Run all registered health checks."""
        print("🏥 Running Project Health Check...")
        print("=" * 50)

        passed = 0
        total = len(self.checks)

        for name, description, check_func in self.checks:
            print(f"\n📋 {description}")

            try:
                start_time = time.time()
                success, details = check_func()
                elapsed = time.time() - start_time

                if success:
                    print(f"✅ PASS ({elapsed:.2f}s)")
                    passed += 1
                    status = "PASS"
                else:
                    print(f"❌ FAIL ({elapsed:.2f}s)")
                    status = "FAIL"

                self.results[name] = {
                    'status': status,
                    'elapsed_time': elapsed,
                    'details': details
                }

            except Exception as e:
                print(f"💥 ERROR: {e}")
                self.results[name] = {
                    'status': 'ERROR',
                    'error': str(e)
                }

        # Summary
        success_rate = passed / total
        print(f"\n{'='*50}")
        print(f"📊 HEALTH CHECK SUMMARY")
        print(f"{'='*50}")
        print(f"Passed: {passed}/{total} ({success_rate:.1%})")

        if success_rate >= 0.8:
            print("🎉 Project is HEALTHY!")
            overall_status = "HEALTHY"
        elif success_rate >= 0.6:
            print("⚠️  Project has MINOR ISSUES")
            overall_status = "MINOR_ISSUES"
        else:
            print("🚨 Project has MAJOR ISSUES")
            overall_status = "MAJOR_ISSUES"

        self.results['summary'] = {
            'overall_status': overall_status,
            'passed': passed,
            'total': total,
            'success_rate': success_rate
        }

        return self.results


def check_python_environment() -> Tuple[bool, Dict]:
    """Check Python environment and dependencies."""
    details = {
        'python_version': sys.version,
        'pytorch_version': torch.__version__ if torch else None,
        'cuda_available': torch.cuda.is_available() if torch else False
    }

    # Check Python version
    if sys.version_info < (3, 8):
        return False, {**details, 'error': 'Python 3.8+ required'}

    # Check PyTorch
    try:
        import torch
        details['pytorch_version'] = torch.__version__
        details['cuda_available'] = torch.cuda.is_available()
    except ImportError:
        return False, {**details, 'error': 'PyTorch not installed'}

    return True, details


def check_project_structure() -> Tuple[bool, Dict]:
    """Check that essential project files and directories exist."""
    required_paths = [
        'src/',
        'tests/',
        'configs/',
        'scripts/',
        'requirements.txt',
        'README.md',
        'setup.py',
    ]

    missing = []
    existing = []

    for path_str in required_paths:
        path = Path(path_str)
        if path.exists():
            existing.append(path_str)
        else:
            missing.append(path_str)

    details = {
        'existing': existing,
        'missing': missing,
        'total_required': len(required_paths)
    }

    success = len(missing) == 0
    if not success:
        details['error'] = f"Missing required paths: {missing}"

    return success, details


def check_core_imports() -> Tuple[bool, Dict]:
    """Check that core modules can be imported."""
    core_modules = [
        'src.models.backbone.eeg_transformer',
        'src.models.adapters.task_aware',
        'src.models.heads.temporal_regression',
        'src.training.trainers.ssl_trainer',
    ]

    import_results = {}
    failed_imports = []

    for module in core_modules:
        try:
            __import__(module)
            import_results[module] = 'SUCCESS'
        except ImportError as e:
            import_results[module] = f'FAILED: {e}'
            failed_imports.append(module)

    details = {
        'import_results': import_results,
        'failed_imports': failed_imports,
        'total_modules': len(core_modules)
    }

    success = len(failed_imports) == 0
    return success, details


def check_configurations() -> Tuple[bool, Dict]:
    """Check that configuration files are valid YAML."""
    config_files = [
        'configs/gpu/enhanced_gpu.yaml',
        'configs/cpu/cpu_fallback.yaml',
        'configs/training/base_config.yaml'
    ]

    config_status = {}
    invalid_configs = []

    for config_file in config_files:
        config_path = Path(config_file)

        if not config_path.exists():
            config_status[config_file] = 'MISSING'
            invalid_configs.append(config_file)
            continue

        try:
            with open(config_path) as f:
                config = yaml.safe_load(f)

            # Basic validation
            if not isinstance(config, dict):
                raise ValueError("Config must be a dictionary")

            config_status[config_file] = 'VALID'

        except Exception as e:
            config_status[config_file] = f'INVALID: {e}'
            invalid_configs.append(config_file)

    details = {
        'config_status': config_status,
        'invalid_configs': invalid_configs,
        'total_configs': len(config_files)
    }

    success = len(invalid_configs) == 0
    return success, details


def check_basic_functionality() -> Tuple[bool, Dict]:
    """Check that basic model functionality works."""
    try:
        from src.models.backbone.eeg_transformer import EEGTransformer

        # Create a small model
        model = EEGTransformer(
            n_channels=64,
            d_model=128,
            n_layers=2,
            n_heads=4
        )

        # Test forward pass
        batch_size, seq_len = 2, 250
        x = torch.randn(batch_size, 64, seq_len)

        with torch.no_grad():
            output = model(x)

        # Validate output
        assert output.shape[0] == batch_size
        assert len(output.shape) == 3
        assert torch.all(torch.isfinite(output))

        details = {
            'model_type': type(model).__name__,
            'input_shape': list(x.shape),
            'output_shape': list(output.shape),
            'parameters': sum(p.numel() for p in model.parameters())
        }

        return True, details

    except Exception as e:
        return False, {'error': str(e)}


def check_test_suite() -> Tuple[bool, Dict]:
    """Check that test files exist and are structured correctly."""
    test_files = [
        'tests/test_dann_multi.py',
        'tests/test_adapters.py',
        'tests/test_compression_ssl.py',
        'tests/test_gpu_ops.py',
        'tests/test_heads.py',
    ]

    test_status = {}
    missing_tests = []

    for test_file in test_files:
        test_path = Path(test_file)

        if not test_path.exists():
            test_status[test_file] = 'MISSING'
            missing_tests.append(test_file)
            continue

        # Check if file contains test functions
        try:
            with open(test_path) as f:
                content = f.read()

            if 'def test_' in content or 'class Test' in content:
                test_status[test_file] = 'EXISTS'
            else:
                test_status[test_file] = 'NO_TESTS'
                missing_tests.append(test_file)

        except Exception as e:
            test_status[test_file] = f'ERROR: {e}'
            missing_tests.append(test_file)

    details = {
        'test_status': test_status,
        'missing_tests': missing_tests,
        'total_test_files': len(test_files)
    }

    success = len(missing_tests) == 0
    return success, details


def check_benchmarking() -> Tuple[bool, Dict]:
    """Check that benchmarking infrastructure exists."""
    benchmark_files = [
        'scripts/bench_inference.py',
    ]

    benchmark_status = {}
    missing_benchmarks = []

    for benchmark_file in benchmark_files:
        benchmark_path = Path(benchmark_file)

        if not benchmark_path.exists():
            benchmark_status[benchmark_file] = 'MISSING'
            missing_benchmarks.append(benchmark_file)
            continue

        try:
            with open(benchmark_path) as f:
                content = f.read()

            # Check for key components
            has_benchmark_class = 'class' in content and 'Benchmark' in content
            has_main = 'if __name__' in content

            if has_benchmark_class and has_main:
                benchmark_status[benchmark_file] = 'COMPLETE'
            else:
                benchmark_status[benchmark_file] = 'INCOMPLETE'

        except Exception as e:
            benchmark_status[benchmark_file] = f'ERROR: {e}'
            missing_benchmarks.append(benchmark_file)

    details = {
        'benchmark_status': benchmark_status,
        'missing_benchmarks': missing_benchmarks,
        'total_benchmark_files': len(benchmark_files)
    }

    success = len(missing_benchmarks) == 0
    return success, details


def main():
    """Main health check entry point."""
    parser = argparse.ArgumentParser(description="Run project health check")
    parser.add_argument("--output", "-o", default="health_check.json",
                       help="Output file for health check results")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")

    args = parser.parse_args()

    # Create health checker
    checker = HealthChecker()

    # Register health checks
    checker.add_check("python_env", "Checking Python environment", check_python_environment)
    checker.add_check("project_structure", "Checking project structure", check_project_structure)
    checker.add_check("core_imports", "Checking core module imports", check_core_imports)
    checker.add_check("configurations", "Checking configuration files", check_configurations)
    checker.add_check("basic_functionality", "Checking basic model functionality", check_basic_functionality)
    checker.add_check("test_suite", "Checking test suite", check_test_suite)
    checker.add_check("benchmarking", "Checking benchmarking infrastructure", check_benchmarking)

    # Run all checks
    results = checker.run_all_checks()

    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n📄 Detailed results saved to: {output_path}")

    # Determine exit code
    overall_status = results['summary']['overall_status']
    if overall_status == "HEALTHY":
        exit_code = 0
    elif overall_status == "MINOR_ISSUES":
        exit_code = 1
    else:
        exit_code = 2

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
