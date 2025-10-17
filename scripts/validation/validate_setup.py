#!/usr/bin/env python3
"""
Validation script for EEG Foundation Challenge 2025 project setup.

This script validates the project structure, dependencies, and data setup
to ensure everything is ready for training.
"""

import sys
import logging
import importlib
from pathlib import Path
from typing import Dict, List, Any

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_python_version() -> Dict[str, Any]:
    """Check Python version compatibility."""
    import sys

    required_version = (3, 10)
    current_version = sys.version_info[:2]

    result = {
        'name': 'Python Version',
        'status': 'PASS' if current_version >= required_version else 'FAIL',
        'current': f"{current_version[0]}.{current_version[1]}",
        'required': f"{required_version[0]}.{required_version[1]}+",
        'details': []
    }

    if current_version < required_version:
        result['details'].append(f"Python {required_version[0]}.{required_version[1]}+ required")

    return result


def check_dependencies() -> Dict[str, Any]:
    """Check if required dependencies are installed."""
    required_packages = [
        'torch',
        'pytorch_lightning',
        'mne',
        'mne_bids',
        'numpy',
        'pandas',
        'scikit-learn',
        'hydra-core',
        'omegaconf',
        'tqdm',
        'matplotlib',
        'seaborn'
    ]

    optional_packages = [
        'wandb',
        'tensorboard',
        'polars',
        'blosc2',
        'lz4',
        'zstandard',
        'pywavelets',
        'librosa'
    ]

    result = {
        'name': 'Dependencies',
        'status': 'PASS',
        'installed': [],
        'missing': [],
        'optional_missing': [],
        'details': []
    }

    # Check required packages
    for package in required_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            result['installed'].append(package)
        except ImportError:
            result['missing'].append(package)
            result['status'] = 'FAIL'

    # Check optional packages
    for package in optional_packages:
        try:
            importlib.import_module(package.replace('-', '_'))
            result['installed'].append(package)
        except ImportError:
            result['optional_missing'].append(package)

    if result['missing']:
        result['details'].append(f"Missing required packages: {', '.join(result['missing'])}")

    if result['optional_missing']:
        result['details'].append(f"Missing optional packages: {', '.join(result['optional_missing'])}")

    return result


def check_pytorch_setup() -> Dict[str, Any]:
    """Check PyTorch and CUDA setup."""
    result = {
        'name': 'PyTorch Setup',
        'status': 'PASS',
        'details': []
    }

    try:
        import torch

        # Check PyTorch version
        torch_version = torch.__version__
        result['torch_version'] = torch_version

        # Check CUDA availability
        cuda_available = torch.cuda.is_available()
        result['cuda_available'] = cuda_available

        if cuda_available:
            cuda_version = torch.version.cuda
            device_count = torch.cuda.device_count()
            result['cuda_version'] = cuda_version
            result['gpu_count'] = device_count
            result['details'].append(f"CUDA {cuda_version} with {device_count} GPU(s)")
        else:
            result['details'].append("CUDA not available - will use CPU")
            if torch.backends.mps.is_available():
                result['details'].append("MPS (Apple Silicon) available")

    except ImportError:
        result['status'] = 'FAIL'
        result['details'].append("PyTorch not installed")

    return result


def check_project_structure() -> Dict[str, Any]:
    """Check project directory structure."""
    project_root = Path(__file__).parent.parent

    required_dirs = [
        'src',
        'configs',
        'scripts',
        'tests',
        'docs',
        'docker'
    ]

    required_files = [
        'requirements.txt',
        'pyproject.toml',
        'README.md',
        'Makefile'
    ]

    result = {
        'name': 'Project Structure',
        'status': 'PASS',
        'missing_dirs': [],
        'missing_files': [],
        'details': []
    }

    # Check directories
    for dirname in required_dirs:
        dir_path = project_root / dirname
        if not dir_path.exists():
            result['missing_dirs'].append(dirname)
            result['status'] = 'FAIL'

    # Check files
    for filename in required_files:
        file_path = project_root / filename
        if not file_path.exists():
            result['missing_files'].append(filename)
            result['status'] = 'FAIL'

    if result['missing_dirs']:
        result['details'].append(f"Missing directories: {', '.join(result['missing_dirs'])}")

    if result['missing_files']:
        result['details'].append(f"Missing files: {', '.join(result['missing_files'])}")

    return result


def check_source_modules() -> Dict[str, Any]:
    """Check if source modules can be imported."""
    modules_to_check = [
        'dataio.bids_loader',
        'dataio.hbn_dataset',
        'models.backbones.temporal_cnn',
        'models.heads',
        'utils.compression',
        'utils.submission'
    ]

    result = {
        'name': 'Source Modules',
        'status': 'PASS',
        'importable': [],
        'failed': [],
        'details': []
    }

    for module in modules_to_check:
        try:
            importlib.import_module(module)
            result['importable'].append(module)
        except ImportError as e:
            result['failed'].append(module)
            result['status'] = 'FAIL'
            result['details'].append(f"Failed to import {module}: {str(e)}")

    return result


def check_config_files() -> Dict[str, Any]:
    """Check configuration files."""
    project_root = Path(__file__).parent.parent
    config_dir = project_root / 'configs'

    required_configs = [
        'base.yaml',
        'data.yaml',
        'model.yaml',
        'training.yaml'
    ]

    result = {
        'name': 'Configuration Files',
        'status': 'PASS',
        'found': [],
        'missing': [],
        'details': []
    }

    for config_file in required_configs:
        config_path = config_dir / config_file
        if config_path.exists():
            result['found'].append(config_file)
        else:
            result['missing'].append(config_file)
            result['status'] = 'FAIL'

    if result['missing']:
        result['details'].append(f"Missing config files: {', '.join(result['missing'])}")

    return result


def check_data_setup() -> Dict[str, Any]:
    """Check data directory setup."""
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'

    result = {
        'name': 'Data Setup',
        'status': 'PASS',
        'details': []
    }

    if not data_dir.exists():
        result['status'] = 'WARN'
        result['details'].append("Data directory not found - you'll need to set up your BIDS dataset")
    else:
        # Check for BIDS structure
        bids_files = ['participants.tsv', 'dataset_description.json']
        found_bids = any((data_dir / f).exists() for f in bids_files)

        if found_bids:
            result['details'].append("BIDS dataset structure detected")
        else:
            result['status'] = 'WARN'
            result['details'].append("No BIDS dataset found - set up your HBN data")

    return result


def check_docker_setup() -> Dict[str, Any]:
    """Check Docker setup."""
    project_root = Path(__file__).parent.parent
    docker_dir = project_root / 'docker'

    result = {
        'name': 'Docker Setup',
        'status': 'PASS',
        'details': []
    }

    dockerfile = docker_dir / 'Dockerfile'
    if dockerfile.exists():
        result['details'].append("Dockerfile found")
    else:
        result['status'] = 'WARN'
        result['details'].append("Dockerfile not found")

    compose_file = project_root / 'docker-compose.yml'
    if compose_file.exists():
        result['details'].append("docker-compose.yml found")
    else:
        result['status'] = 'WARN'
        result['details'].append("docker-compose.yml not found")

    return result


def run_validation() -> List[Dict[str, Any]]:
    """Run all validation checks."""
    checks = [
        check_python_version,
        check_dependencies,
        check_pytorch_setup,
        check_project_structure,
        check_source_modules,
        check_config_files,
        check_data_setup,
        check_docker_setup
    ]

    results = []
    for check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            result = {
                'name': check_func.__name__,
                'status': 'ERROR',
                'details': [f"Check failed with error: {str(e)}"]
            }
            results.append(result)

    return results


def print_validation_report(results: List[Dict[str, Any]]):
    """Print validation report."""
    print("\n" + "="*60)
    print("EEG Foundation Challenge 2025 - Project Validation Report")
    print("="*60)

    total_checks = len(results)
    passed = sum(1 for r in results if r['status'] == 'PASS')
    warnings = sum(1 for r in results if r['status'] == 'WARN')
    failed = sum(1 for r in results if r['status'] == 'FAIL')
    errors = sum(1 for r in results if r['status'] == 'ERROR')

    print(f"\nSummary: {passed}/{total_checks} checks passed")
    if warnings > 0:
        print(f"Warnings: {warnings}")
    if failed > 0:
        print(f"Failed: {failed}")
    if errors > 0:
        print(f"Errors: {errors}")

    print("\nDetailed Results:")
    print("-" * 40)

    for result in results:
        status_symbol = {
            'PASS': '✅',
            'WARN': '⚠️',
            'FAIL': '❌',
            'ERROR': '💥'
        }.get(result['status'], '❓')

        print(f"{status_symbol} {result['name']}: {result['status']}")

        if result.get('details'):
            for detail in result['details']:
                print(f"   → {detail}")

        # Print additional info for some checks
        if result['name'] == 'Dependencies' and result.get('installed'):
            print(f"   → Installed: {len(result['installed'])} packages")

        if result['name'] == 'PyTorch Setup':
            if result.get('torch_version'):
                print(f"   → PyTorch version: {result['torch_version']}")
            if result.get('cuda_available'):
                print(f"   → CUDA available: {result['cuda_available']}")

        print()

    # Recommendations
    print("Recommendations:")
    print("-" * 20)

    if failed > 0 or errors > 0:
        print("🔧 Fix failed checks before proceeding with training")

    if any(r['status'] == 'FAIL' and 'Dependencies' in r['name'] for r in results):
        print("📦 Install missing dependencies: pip install -r requirements.txt")

    if any('BIDS' in str(r.get('details', [])) for r in results):
        print("📁 Set up your BIDS dataset in the data/ directory")

    if any('CUDA' in str(r.get('details', [])) for r in results):
        print("🚀 Consider setting up CUDA for GPU acceleration")

    print("\n" + "="*60)


if __name__ == "__main__":
    print("Running EEG Foundation Challenge 2025 project validation...")
    results = run_validation()
    print_validation_report(results)
