"""
Reproducibility infrastructure for EEG2025 Challenge.

This module provides:
1. Seed management and deterministic operations
2. Run manifest generation and tracking
3. Environment capture and versioning
4. Experiment logging and provenance
5. Configuration snapshot management
"""

import os
import sys
import json
import hashlib
import pickle
import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, asdict
import warnings

import numpy as np
import torch
import yaml

try:
    import git
    GIT_AVAILABLE = True
except ImportError:
    GIT_AVAILABLE = False
    warnings.warn("GitPython not available - git tracking disabled")

logger = logging.getLogger(__name__)


@dataclass
class EnvironmentInfo:
    """Environment information for reproducibility."""
    python_version: str
    pytorch_version: str
    cuda_version: Optional[str]
    cudnn_version: Optional[str]
    platform: str
    hostname: str
    cpu_count: int
    memory_gb: float
    gpu_info: List[Dict[str, Any]]
    pip_packages: Dict[str, str]
    git_info: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class RunManifest:
    """Manifest for a training/evaluation run."""
    run_id: str
    timestamp: str
    experiment_name: str
    config_snapshot: Dict[str, Any]
    environment: EnvironmentInfo
    seeds: Dict[str, int]
    command_line: List[str]
    working_directory: str
    input_data_hash: Optional[str] = None
    model_checkpoints: List[str] = None
    output_files: List[str] = None
    metrics: Dict[str, Any] = None
    duration_seconds: Optional[float] = None
    status: str = "running"  # running, completed, failed
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class SeedManager:
    """Manages seeds for reproducible experiments."""

    def __init__(self, base_seed: int = 42):
        """Initialize seed manager."""
        self.base_seed = base_seed
        self.seeds = {}

        # Generate deterministic seeds for different components
        self._generate_component_seeds()

        logger.info(f"Initialized seed manager with base seed: {base_seed}")

    def _generate_component_seeds(self):
        """Generate deterministic seeds for different components."""
        # Use numpy's random number generator to create deterministic seeds
        rng = np.random.RandomState(self.base_seed)

        self.seeds = {
            'python': self.base_seed,
            'numpy': rng.randint(0, 2**31),
            'torch': rng.randint(0, 2**31),
            'torch_cuda': rng.randint(0, 2**31),
            'data_loader': rng.randint(0, 2**31),
            'model_init': rng.randint(0, 2**31),
            'training': rng.randint(0, 2**31),
            'evaluation': rng.randint(0, 2**31),
            'cross_validation': rng.randint(0, 2**31)
        }

    def set_all_seeds(self):
        """Set all seeds for reproducibility."""
        # Python hash seed (needs to be set before Python starts)
        os.environ['PYTHONHASHSEED'] = str(self.seeds['python'])

        # NumPy seed
        np.random.seed(self.seeds['numpy'])

        # PyTorch seeds
        torch.manual_seed(self.seeds['torch'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seeds['torch_cuda'])
            torch.cuda.manual_seed_all(self.seeds['torch_cuda'])

        # Additional PyTorch deterministic settings
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Enable deterministic algorithms (may impact performance)
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)

        logger.info("All seeds set for reproducibility")

    def get_seed(self, component: str) -> int:
        """Get seed for specific component."""
        if component not in self.seeds:
            raise ValueError(f"Unknown component: {component}. Available: {list(self.seeds.keys())}")
        return self.seeds[component]

    def create_generator(self, component: str) -> torch.Generator:
        """Create PyTorch generator with component seed."""
        generator = torch.Generator()
        generator.manual_seed(self.get_seed(component))
        return generator

    def get_all_seeds(self) -> Dict[str, int]:
        """Get all component seeds."""
        return self.seeds.copy()


class EnvironmentCapture:
    """Captures environment information for reproducibility."""

    def __init__(self):
        """Initialize environment capture."""
        self.environment_info = None

    def capture_environment(self) -> EnvironmentInfo:
        """Capture current environment information."""
        logger.info("Capturing environment information...")

        # Python and PyTorch versions
        python_version = platform.python_version()
        pytorch_version = torch.__version__

        # CUDA information
        cuda_version = None
        cudnn_version = None
        if torch.cuda.is_available():
            cuda_version = torch.version.cuda
            cudnn_version = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None

        # Platform information
        platform_info = platform.platform()
        hostname = platform.node()
        cpu_count = os.cpu_count()

        # Memory information (approximate)
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
        except ImportError:
            memory_gb = 0.0

        # GPU information
        gpu_info = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                gpu_props = torch.cuda.get_device_properties(i)
                gpu_info.append({
                    'id': i,
                    'name': gpu_props.name,
                    'memory_gb': gpu_props.total_memory / (1024**3),
                    'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
                })

        # Package versions
        pip_packages = self._get_package_versions()

        # Git information
        git_info = self._get_git_info() if GIT_AVAILABLE else None

        self.environment_info = EnvironmentInfo(
            python_version=python_version,
            pytorch_version=pytorch_version,
            cuda_version=cuda_version,
            cudnn_version=cudnn_version,
            platform=platform_info,
            hostname=hostname,
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            gpu_info=gpu_info,
            pip_packages=pip_packages,
            git_info=git_info
        )

        logger.info("Environment captured successfully")
        return self.environment_info

    def _get_package_versions(self) -> Dict[str, str]:
        """Get versions of key packages."""
        packages = {}

        # Core packages
        core_packages = [
            'torch', 'numpy', 'pandas', 'scipy', 'scikit-learn',
            'matplotlib', 'seaborn', 'mne', 'h5py', 'tqdm'
        ]

        for package in core_packages:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                packages[package] = version
            except ImportError:
                packages[package] = 'not_installed'

        # Try to get pip list output
        try:
            result = subprocess.run(['pip', 'list', '--format=json'],
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                pip_list = json.loads(result.stdout)
                for package_info in pip_list:
                    packages[package_info['name']] = package_info['version']
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, json.JSONDecodeError):
            logger.warning("Could not get complete pip list")

        return packages

    def _get_git_info(self) -> Optional[Dict[str, str]]:
        """Get git repository information."""
        try:
            # Find git repository
            repo_path = Path.cwd()
            while repo_path != repo_path.parent:
                if (repo_path / '.git').exists():
                    break
                repo_path = repo_path.parent
            else:
                return None

            repo = git.Repo(repo_path)

            # Get current commit info
            commit = repo.head.commit

            git_info = {
                'repository_path': str(repo_path),
                'commit_hash': commit.hexsha,
                'commit_hash_short': commit.hexsha[:8],
                'branch': repo.active_branch.name if not repo.head.is_detached else 'detached',
                'commit_message': commit.message.strip(),
                'commit_author': str(commit.author),
                'commit_date': commit.committed_datetime.isoformat(),
                'is_dirty': repo.is_dirty(),
                'untracked_files': repo.untracked_files,
                'remote_url': next(iter(repo.remotes.origin.urls), None) if repo.remotes else None
            }

            # Get diff if repository is dirty
            if repo.is_dirty():
                git_info['diff'] = repo.git.diff()

            return git_info

        except Exception as e:
            logger.warning(f"Could not get git information: {e}")
            return None


class RunTracker:
    """Tracks experiment runs for reproducibility."""

    def __init__(self, experiment_name: str, output_dir: Path, seed_manager: Optional[SeedManager] = None):
        """Initialize run tracker."""
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.seed_manager = seed_manager or SeedManager()
        self.environment_capture = EnvironmentCapture()

        # Generate unique run ID
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        self.run_id = f"{experiment_name}_{timestamp}"

        self.manifest = None
        self.start_time = None

        logger.info(f"Initialized run tracker for {experiment_name}, run ID: {self.run_id}")

    def start_run(self, config: Dict[str, Any], command_line: Optional[List[str]] = None) -> str:
        """Start tracking a new run."""
        logger.info(f"Starting run: {self.run_id}")

        self.start_time = datetime.now(timezone.utc)

        # Set seeds for reproducibility
        self.seed_manager.set_all_seeds()

        # Capture environment
        environment = self.environment_capture.capture_environment()

        # Create manifest
        self.manifest = RunManifest(
            run_id=self.run_id,
            timestamp=self.start_time.isoformat(),
            experiment_name=self.experiment_name,
            config_snapshot=self._create_config_snapshot(config),
            environment=environment,
            seeds=self.seed_manager.get_all_seeds(),
            command_line=command_line or sys.argv,
            working_directory=str(Path.cwd()),
            model_checkpoints=[],
            output_files=[],
            metrics={}
        )

        # Save initial manifest
        self._save_manifest()

        return self.run_id

    def update_metrics(self, metrics: Dict[str, Any]):
        """Update run metrics."""
        if self.manifest is None:
            raise RuntimeError("Run not started - call start_run() first")

        self.manifest.metrics.update(metrics)
        self._save_manifest()

    def add_checkpoint(self, checkpoint_path: str):
        """Add model checkpoint to tracking."""
        if self.manifest is None:
            raise RuntimeError("Run not started - call start_run() first")

        self.manifest.model_checkpoints.append(checkpoint_path)
        self._save_manifest()

    def add_output_file(self, file_path: str):
        """Add output file to tracking."""
        if self.manifest is None:
            raise RuntimeError("Run not started - call start_run() first")

        self.manifest.output_files.append(file_path)
        self._save_manifest()

    def set_input_data_hash(self, data_hash: str):
        """Set hash of input data."""
        if self.manifest is None:
            raise RuntimeError("Run not started - call start_run() first")

        self.manifest.input_data_hash = data_hash
        self._save_manifest()

    def end_run(self, status: str = "completed", error_message: Optional[str] = None):
        """End the current run."""
        if self.manifest is None:
            raise RuntimeError("Run not started - call start_run() first")

        end_time = datetime.now(timezone.utc)
        duration = (end_time - self.start_time).total_seconds()

        self.manifest.duration_seconds = duration
        self.manifest.status = status
        self.manifest.error_message = error_message

        self._save_manifest()

        logger.info(f"Run {self.run_id} ended with status: {status}")
        logger.info(f"Duration: {duration:.1f} seconds")

    def _create_config_snapshot(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create a snapshot of the configuration."""
        # Deep copy to avoid modifying original
        snapshot = json.loads(json.dumps(config, default=str))

        # Add timestamp
        snapshot['_snapshot_timestamp'] = datetime.now(timezone.utc).isoformat()

        return snapshot

    def _save_manifest(self):
        """Save manifest to file."""
        manifest_path = self.output_dir / f"{self.run_id}_manifest.json"

        with open(manifest_path, 'w') as f:
            json.dump(self.manifest.to_dict(), f, indent=2, default=str)

        logger.debug(f"Manifest saved to {manifest_path}")

    def get_manifest_path(self) -> Path:
        """Get path to manifest file."""
        return self.output_dir / f"{self.run_id}_manifest.json"


class DataHasher:
    """Computes hashes of datasets for provenance tracking."""

    @staticmethod
    def hash_file(file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)

        return sha256_hash.hexdigest()

    @staticmethod
    def hash_directory(dir_path: Path, include_patterns: Optional[List[str]] = None) -> str:
        """Compute hash of directory contents."""
        if include_patterns is None:
            include_patterns = ['*.csv', '*.json', '*.yaml', '*.yml']

        file_hashes = []

        for pattern in include_patterns:
            for file_path in dir_path.rglob(pattern):
                if file_path.is_file():
                    file_hash = DataHasher.hash_file(file_path)
                    rel_path = file_path.relative_to(dir_path)
                    file_hashes.append(f"{rel_path}:{file_hash}")

        # Sort for consistent ordering
        file_hashes.sort()

        # Hash the concatenated file hashes
        combined = '\n'.join(file_hashes)
        return hashlib.sha256(combined.encode()).hexdigest()

    @staticmethod
    def hash_array(array: np.ndarray) -> str:
        """Compute hash of numpy array."""
        # Ensure consistent byte order
        if array.dtype.byteorder not in ('=', '|'):
            array = array.astype(array.dtype.newbyteorder('='))

        return hashlib.sha256(array.tobytes()).hexdigest()

    @staticmethod
    def hash_config(config: Dict[str, Any]) -> str:
        """Compute hash of configuration dictionary."""
        # Convert to JSON string with sorted keys for consistency
        config_str = json.dumps(config, sort_keys=True, default=str)
        return hashlib.sha256(config_str.encode()).hexdigest()


class ReproducibilityManager:
    """High-level manager for reproducibility features."""

    def __init__(self, experiment_name: str, output_dir: Path, base_seed: int = 42):
        """Initialize reproducibility manager."""
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.base_seed = base_seed

        # Initialize components
        self.seed_manager = SeedManager(base_seed)
        self.run_tracker = RunTracker(experiment_name, output_dir, self.seed_manager)
        self.data_hasher = DataHasher()

        logger.info(f"Initialized reproducibility manager for {experiment_name}")

    def start_experiment(self, config: Dict[str, Any], data_dir: Optional[Path] = None) -> str:
        """Start a reproducible experiment."""
        logger.info("Starting reproducible experiment...")

        # Compute data hash if data directory provided
        data_hash = None
        if data_dir and data_dir.exists():
            logger.info("Computing data hash...")
            data_hash = self.data_hasher.hash_directory(data_dir)
            logger.info(f"Data hash: {data_hash[:16]}...")

        # Start run tracking
        run_id = self.run_tracker.start_run(config)

        if data_hash:
            self.run_tracker.set_input_data_hash(data_hash)

        # Save configuration separately
        config_path = self.output_dir / f"{run_id}_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        self.run_tracker.add_output_file(str(config_path))

        logger.info(f"Experiment started with run ID: {run_id}")
        return run_id

    def log_metrics(self, metrics: Dict[str, Any]):
        """Log experiment metrics."""
        self.run_tracker.update_metrics(metrics)

    def save_checkpoint(self, model: torch.nn.Module, checkpoint_path: Path,
                       metrics: Optional[Dict[str, Any]] = None):
        """Save model checkpoint with tracking."""
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'run_id': self.run_tracker.run_id,
            'seed_info': self.seed_manager.get_all_seeds(),
            'metrics': metrics or {},
            'timestamp': datetime.now(timezone.utc).isoformat()
        }, checkpoint_path)

        # Track checkpoint
        self.run_tracker.add_checkpoint(str(checkpoint_path))

        if metrics:
            self.log_metrics(metrics)

        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def end_experiment(self, status: str = "completed", error_message: Optional[str] = None):
        """End the experiment."""
        self.run_tracker.end_run(status, error_message)

        # Generate summary report
        self._generate_summary_report()

        logger.info("Experiment ended")

    def _generate_summary_report(self):
        """Generate experiment summary report."""
        manifest = self.run_tracker.manifest
        if not manifest:
            return

        report_path = self.output_dir / f"{manifest.run_id}_summary.md"

        report = f"""# Experiment Summary

## Basic Information
- **Run ID**: {manifest.run_id}
- **Experiment**: {manifest.experiment_name}
- **Status**: {manifest.status}
- **Duration**: {manifest.duration_seconds:.1f} seconds
- **Timestamp**: {manifest.timestamp}

## Environment
- **Python**: {manifest.environment.python_version}
- **PyTorch**: {manifest.environment.pytorch_version}
- **CUDA**: {manifest.environment.cuda_version or 'N/A'}
- **Platform**: {manifest.environment.platform}
- **Hostname**: {manifest.environment.hostname}

## Reproducibility
- **Base Seed**: {self.base_seed}
- **Component Seeds**: {manifest.seeds}
- **Git Commit**: {manifest.environment.git_info.get('commit_hash_short', 'N/A') if manifest.environment.git_info else 'N/A'}
- **Data Hash**: {manifest.input_data_hash[:16] + '...' if manifest.input_data_hash else 'N/A'}

## Outputs
- **Checkpoints**: {len(manifest.model_checkpoints)}
- **Output Files**: {len(manifest.output_files)}

## Configuration
```yaml
{yaml.dump(manifest.config_snapshot, default_flow_style=False)}
```

## Metrics
```json
{json.dumps(manifest.metrics, indent=2)}
```
"""

        with open(report_path, 'w') as f:
            f.write(report)

        logger.info(f"Summary report saved: {report_path}")


def create_reproducible_dataloader(dataset, batch_size: int, seed: int, **kwargs) -> torch.utils.data.DataLoader:
    """Create a reproducible DataLoader."""
    # Create generator with specific seed
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Set worker init function for multiprocessing
    def worker_init_fn(worker_id):
        np.random.seed(seed + worker_id)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        generator=generator,
        worker_init_fn=worker_init_fn,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Create reproducibility manager
    manager = ReproducibilityManager(
        experiment_name="test_experiment",
        output_dir=Path("test_reproducibility"),
        base_seed=42
    )

    # Example configuration
    config = {
        'model': {
            'type': 'cnn',
            'layers': 3,
            'filters': [64, 128, 256]
        },
        'training': {
            'epochs': 10,
            'lr': 0.001,
            'batch_size': 32
        },
        'dann_schedule': {
            'strategy': 'linear_warmup',
            'warmup_steps': 1000
        }
    }

    # Start experiment
    run_id = manager.start_experiment(config)

    # Simulate training with metrics
    for epoch in range(3):
        metrics = {
            f'epoch_{epoch}': {
                'train_loss': np.random.random(),
                'val_loss': np.random.random(),
                'train_acc': np.random.random(),
                'val_acc': np.random.random()
            }
        }
        manager.log_metrics(metrics)

    # End experiment
    manager.end_experiment()

    print(f"Test experiment completed with run ID: {run_id}")
