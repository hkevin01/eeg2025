"""
Performance benchmarking infrastructure for EEG2025 challenge models.

This module provides comprehensive benchmarking tools to systematically evaluate
and compare different model configurations, training strategies, and ablation studies.
"""

import json
import os
import time
import warnings
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import GPUtil
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
import torch.nn as nn

from ..data.loaders import EEGDataLoader
from ..models.backbone import TemporalCNN
from ..models.invariance.dann import DANNModel
from ..training.train_psych import PsychTrainer, UncertaintyWeightedLoss
from ..training.train_ssl import SSLTrainer
from ..utils.reproducibility import EnvironmentCapture, SeedManager


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark experiments."""

    name: str
    description: str
    model_config: Dict[str, Any]
    training_config: Dict[str, Any]
    data_config: Dict[str, Any]
    ssl_enabled: bool = True
    dann_enabled: bool = True
    irm_enabled: bool = False
    num_seeds: int = 3
    max_epochs: int = 100
    early_stopping_patience: int = 10
    compute_budget_hours: Optional[float] = None


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""

    config_name: str
    seed: int
    cross_task_correlation: float
    psychopathology_correlation: float
    p_factor_correlation: float
    internalizing_correlation: float
    externalizing_correlation: float
    attention_correlation: float
    training_time_hours: float
    peak_memory_gb: float
    convergence_epoch: int
    final_loss: float
    domain_accuracy: Optional[float] = None
    validation_correlations: Optional[Dict[str, float]] = None
    test_correlations: Optional[Dict[str, float]] = None
    config_dict: Optional[Dict[str, Any]] = None


class PerformanceProfiler:
    """Profiles GPU memory usage and training time."""

    def __init__(self):
        self.start_time = None
        self.peak_memory = 0.0
        self.memory_history = []
        self.time_history = []

    def start_profiling(self):
        """Start performance profiling."""
        self.start_time = time.time()
        self.peak_memory = 0.0
        self.memory_history = []
        self.time_history = []

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    def update(self):
        """Update profiling metrics."""
        if self.start_time is None:
            return

        current_time = time.time() - self.start_time

        # Get memory usage
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / 1e9
            self.peak_memory = max(self.peak_memory, memory_gb)
        else:
            # Use CPU memory as fallback
            memory_gb = psutil.virtual_memory().used / 1e9
            self.peak_memory = max(self.peak_memory, memory_gb)

        self.memory_history.append(memory_gb)
        self.time_history.append(current_time)

    def get_results(self) -> Tuple[float, float]:
        """Get profiling results (time_hours, peak_memory_gb)."""
        if self.start_time is None:
            return 0.0, 0.0

        total_time = time.time() - self.start_time
        return total_time / 3600.0, self.peak_memory


class ModelBenchmarker:
    """Comprehensive model benchmarking framework."""

    def __init__(self, results_dir: str = "benchmark_results", device: str = "auto"):
        """
        Initialize benchmarker.

        Args:
            results_dir: Directory to save benchmark results
            device: Computing device ('auto', 'cuda', 'cpu')
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)

        # Setup device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Benchmarking on device: {self.device}")

        # Results storage
        self.results: List[BenchmarkResult] = []
        self.config_registry: Dict[str, BenchmarkConfig] = {}

        # Setup environment capture
        self.env_capture = EnvironmentCapture()

    def register_config(self, config: BenchmarkConfig):
        """Register a benchmark configuration."""
        self.config_registry[config.name] = config
        print(f"Registered benchmark config: {config.name}")

    def register_default_configs(self):
        """Register default benchmark configurations for comprehensive comparison."""

        # Baseline CNN without SSL or DANN
        baseline_config = BenchmarkConfig(
            name="baseline_cnn",
            description="Baseline TemporalCNN without SSL pretraining or domain adaptation",
            model_config={
                "input_channels": 19,
                "num_classes": 4,  # CBCL factors
                "temporal_kernel_size": 25,
                "num_layers": 4,
                "hidden_channels": [32, 64, 128, 256],
                "dropout": 0.3,
            },
            training_config={
                "batch_size": 32,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "optimizer": "adamw",
            },
            data_config={"sequence_length": 1000, "overlap": 0.5, "augmentation": True},
            ssl_enabled=False,
            dann_enabled=False,
            irm_enabled=False,
        )

        # SSL-only configuration
        ssl_only_config = BenchmarkConfig(
            name="ssl_only",
            description="TemporalCNN with SSL pretraining but no domain adaptation",
            model_config={
                "input_channels": 19,
                "num_classes": 4,
                "temporal_kernel_size": 25,
                "num_layers": 4,
                "hidden_channels": [32, 64, 128, 256],
                "dropout": 0.3,
            },
            training_config={
                "batch_size": 32,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "optimizer": "adamw",
                "ssl_epochs": 50,
            },
            data_config={"sequence_length": 1000, "overlap": 0.5, "augmentation": True},
            ssl_enabled=True,
            dann_enabled=False,
            irm_enabled=False,
        )

        # DANN-only configuration
        dann_only_config = BenchmarkConfig(
            name="dann_only",
            description="TemporalCNN with DANN but no SSL pretraining",
            model_config={
                "input_channels": 19,
                "num_classes": 4,
                "temporal_kernel_size": 25,
                "num_layers": 4,
                "hidden_channels": [32, 64, 128, 256],
                "dropout": 0.3,
            },
            training_config={
                "batch_size": 32,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "optimizer": "adamw",
                "dann_lambda_max": 0.2,
                "dann_schedule": "linear_warmup",
            },
            data_config={"sequence_length": 1000, "overlap": 0.5, "augmentation": True},
            ssl_enabled=False,
            dann_enabled=True,
            irm_enabled=False,
        )

        # Full pipeline configuration
        full_pipeline_config = BenchmarkConfig(
            name="full_pipeline",
            description="Complete pipeline with SSL pretraining and DANN domain adaptation",
            model_config={
                "input_channels": 19,
                "num_classes": 4,
                "temporal_kernel_size": 25,
                "num_layers": 5,  # Slightly deeper for best performance
                "hidden_channels": [32, 64, 128, 256, 512],
                "dropout": 0.3,
            },
            training_config={
                "batch_size": 32,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "optimizer": "adamw",
                "ssl_epochs": 50,
                "dann_lambda_max": 0.2,
                "dann_schedule": "linear_warmup",
                "uncertainty_weighting": True,
            },
            data_config={"sequence_length": 1000, "overlap": 0.5, "augmentation": True},
            ssl_enabled=True,
            dann_enabled=True,
            irm_enabled=False,
        )

        # IRM comparison configuration
        irm_config = BenchmarkConfig(
            name="ssl_irm",
            description="SSL pretraining with IRM instead of DANN",
            model_config={
                "input_channels": 19,
                "num_classes": 4,
                "temporal_kernel_size": 25,
                "num_layers": 5,
                "hidden_channels": [32, 64, 128, 256, 512],
                "dropout": 0.3,
            },
            training_config={
                "batch_size": 32,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "optimizer": "adamw",
                "ssl_epochs": 50,
                "irm_penalty": 1e-3,
                "uncertainty_weighting": True,
            },
            data_config={"sequence_length": 1000, "overlap": 0.5, "augmentation": True},
            ssl_enabled=True,
            dann_enabled=False,
            irm_enabled=True,
        )

        # Combined DANN+IRM configuration
        combined_config = BenchmarkConfig(
            name="dann_irm_combined",
            description="Full pipeline with both DANN and IRM for maximum robustness",
            model_config={
                "input_channels": 19,
                "num_classes": 4,
                "temporal_kernel_size": 25,
                "num_layers": 5,
                "hidden_channels": [32, 64, 128, 256, 512],
                "dropout": 0.3,
            },
            training_config={
                "batch_size": 32,
                "learning_rate": 1e-3,
                "weight_decay": 1e-4,
                "optimizer": "adamw",
                "ssl_epochs": 50,
                "dann_lambda_max": 0.2,
                "dann_schedule": "linear_warmup",
                "irm_penalty": 1e-3,
                "uncertainty_weighting": True,
            },
            data_config={"sequence_length": 1000, "overlap": 0.5, "augmentation": True},
            ssl_enabled=True,
            dann_enabled=True,
            irm_enabled=True,
        )

        # Register all configurations
        configs = [
            baseline_config,
            ssl_only_config,
            dann_only_config,
            full_pipeline_config,
            irm_config,
            combined_config,
        ]

        for config in configs:
            self.register_config(config)

        print(f"Registered {len(configs)} default benchmark configurations")

    def run_single_benchmark(
        self, config: BenchmarkConfig, seed: int, data_loader: EEGDataLoader
    ) -> BenchmarkResult:
        """
        Run a single benchmark experiment.

        Args:
            config: Benchmark configuration
            seed: Random seed for reproducibility
            data_loader: EEG data loader

        Returns:
            Benchmark result
        """
        print(f"\nRunning benchmark: {config.name} (seed={seed})")

        # Setup reproducibility
        seed_manager = SeedManager(seed)
        seed_manager.seed_everything()

        # Initialize profiler
        profiler = PerformanceProfiler()
        profiler.start_profiling()

        try:
            # Create model
            backbone = TemporalCNN(
                input_channels=config.model_config["input_channels"],
                temporal_kernel_size=config.model_config["temporal_kernel_size"],
                num_layers=config.model_config["num_layers"],
                hidden_channels=config.model_config["hidden_channels"],
                dropout=config.model_config["dropout"],
            )

            # SSL pretraining if enabled
            if config.ssl_enabled:
                print("  Starting SSL pretraining...")
                ssl_trainer = SSLTrainer(
                    model=backbone, device=self.device, **config.training_config
                )

                ssl_trainer.train(
                    data_loader=data_loader,
                    num_epochs=config.training_config.get("ssl_epochs", 50),
                    save_checkpoints=False,
                )

                profiler.update()
                print("  SSL pretraining completed")

            # Setup psychopathology model
            if config.dann_enabled:
                # Create DANN model
                model = DANNModel(
                    backbone=backbone,
                    num_classes=config.model_config["num_classes"],
                    num_domains=3,  # Assume 3 sites
                    dann_lambda_max=config.training_config.get("dann_lambda_max", 0.2),
                    schedule_strategy=config.training_config.get(
                        "dann_schedule", "linear_warmup"
                    ),
                )
            else:
                # Use regular backbone with classification head
                model = nn.Sequential(
                    backbone,
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(
                        backbone.get_output_dim(), config.model_config["num_classes"]
                    ),
                )

            model = model.to(self.device)

            # Setup trainer
            if config.training_config.get("uncertainty_weighting", False):
                criterion = UncertaintyWeightedLoss(num_tasks=4)
            else:
                criterion = nn.MSELoss()

            trainer = PsychTrainer(
                model=model,
                criterion=criterion,
                device=self.device,
                irm_penalty=(
                    config.training_config.get("irm_penalty", 0.0)
                    if config.irm_enabled
                    else 0.0
                ),
                **{
                    k: v
                    for k, v in config.training_config.items()
                    if k
                    not in [
                        "ssl_epochs",
                        "dann_lambda_max",
                        "dann_schedule",
                        "irm_penalty",
                        "uncertainty_weighting",
                    ]
                },
            )

            # Train psychopathology model
            print("  Starting psychopathology training...")
            training_history = trainer.train(
                train_loader=data_loader.get_train_loader(),
                val_loader=data_loader.get_val_loader(),
                num_epochs=config.max_epochs,
                early_stopping_patience=config.early_stopping_patience,
                save_checkpoints=False,
            )

            profiler.update()
            print("  Psychopathology training completed")

            # Evaluation
            print("  Evaluating model...")
            eval_results = trainer.evaluate(data_loader.get_test_loader())

            # Extract results
            time_hours, peak_memory = profiler.get_results()

            # Find convergence epoch (best validation performance)
            val_losses = [epoch["val_loss"] for epoch in training_history]
            convergence_epoch = np.argmin(val_losses) + 1
            final_loss = val_losses[-1]

            # Domain accuracy for DANN models
            domain_accuracy = None
            if config.dann_enabled and hasattr(trainer, "last_domain_accuracy"):
                domain_accuracy = trainer.last_domain_accuracy

            # Create result
            result = BenchmarkResult(
                config_name=config.name,
                seed=seed,
                cross_task_correlation=eval_results.get("cross_task_correlation", 0.0),
                psychopathology_correlation=eval_results.get(
                    "psychopathology_correlation", 0.0
                ),
                p_factor_correlation=eval_results.get("p_factor_correlation", 0.0),
                internalizing_correlation=eval_results.get(
                    "internalizing_correlation", 0.0
                ),
                externalizing_correlation=eval_results.get(
                    "externalizing_correlation", 0.0
                ),
                attention_correlation=eval_results.get("attention_correlation", 0.0),
                training_time_hours=time_hours,
                peak_memory_gb=peak_memory,
                convergence_epoch=convergence_epoch,
                final_loss=final_loss,
                domain_accuracy=domain_accuracy,
                config_dict=asdict(config),
            )

            print(
                f"  Results: Psych r={result.psychopathology_correlation:.3f}, "
                f"Time={result.training_time_hours:.1f}h, "
                f"Memory={result.peak_memory_gb:.1f}GB"
            )

            return result

        except Exception as e:
            print(f"  Benchmark failed: {str(e)}")
            # Return failed result
            time_hours, peak_memory = profiler.get_results()
            return BenchmarkResult(
                config_name=config.name,
                seed=seed,
                cross_task_correlation=0.0,
                psychopathology_correlation=0.0,
                p_factor_correlation=0.0,
                internalizing_correlation=0.0,
                externalizing_correlation=0.0,
                attention_correlation=0.0,
                training_time_hours=time_hours,
                peak_memory_gb=peak_memory,
                convergence_epoch=0,
                final_loss=float("inf"),
                config_dict=asdict(config),
            )

    def run_benchmark_suite(
        self,
        config_names: Optional[List[str]] = None,
        data_loader: Optional[EEGDataLoader] = None,
    ) -> pd.DataFrame:
        """
        Run a complete benchmark suite.

        Args:
            config_names: List of config names to benchmark (None for all)
            data_loader: EEG data loader (will create mock if None)

        Returns:
            DataFrame with benchmark results
        """
        print("Starting benchmark suite...")

        # Use all configs if none specified
        if config_names is None:
            config_names = list(self.config_registry.keys())

        # Create mock data loader if none provided
        if data_loader is None:
            print("Warning: Using mock data loader for benchmarking")
            data_loader = self._create_mock_dataloader()

        # Save environment info
        env_info = self.env_capture.capture_environment()
        env_path = (
            self.results_dir
            / f"environment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(env_path, "w") as f:
            json.dump(env_info, f, indent=2)

        # Run benchmarks
        all_results = []
        total_runs = sum(self.config_registry[name].num_seeds for name in config_names)
        current_run = 0

        for config_name in config_names:
            if config_name not in self.config_registry:
                print(f"Warning: Unknown config '{config_name}', skipping")
                continue

            config = self.config_registry[config_name]

            for seed in range(config.num_seeds):
                current_run += 1
                print(f"\nProgress: {current_run}/{total_runs}")

                # Check compute budget
                if config.compute_budget_hours is not None:
                    estimated_time = self._estimate_runtime(config)
                    if estimated_time > config.compute_budget_hours:
                        print(
                            f"  Skipping {config_name} seed {seed}: exceeds compute budget"
                        )
                        continue

                result = self.run_single_benchmark(config, seed, data_loader)
                all_results.append(result)
                self.results.extend([result])

        # Convert to DataFrame
        results_df = pd.DataFrame([asdict(r) for r in all_results])

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = self.results_dir / f"benchmark_results_{timestamp}.csv"
        results_df.to_csv(results_path, index=False)

        print(f"\nBenchmark suite completed. Results saved to {results_path}")
        return results_df

    def _create_mock_dataloader(self) -> EEGDataLoader:
        """Create a mock data loader for testing."""
        # This would normally load real EEG data
        # For benchmarking, we create synthetic data
        print("Creating mock EEG data loader...")

        class MockDataLoader:
            def __init__(self):
                self.batch_size = 32
                self.sequence_length = 1000
                self.num_channels = 19

            def get_train_loader(self):
                return self._create_mock_loader(num_batches=50)

            def get_val_loader(self):
                return self._create_mock_loader(num_batches=10)

            def get_test_loader(self):
                return self._create_mock_loader(num_batches=10)

            def _create_mock_loader(self, num_batches):
                for _ in range(num_batches):
                    x = torch.randn(
                        self.batch_size, self.num_channels, self.sequence_length
                    )
                    y = torch.randn(self.batch_size, 4)  # CBCL factors
                    domain = torch.randint(0, 3, (self.batch_size,))  # 3 domains
                    yield {"eeg": x, "cbcl": y, "domain": domain}

        return MockDataLoader()

    def _estimate_runtime(self, config: BenchmarkConfig) -> float:
        """Estimate runtime for a configuration in hours."""
        # Simple heuristic based on model complexity and training settings
        base_time = 1.0  # Base time in hours

        # Adjust for SSL
        if config.ssl_enabled:
            base_time += config.training_config.get("ssl_epochs", 50) * 0.02

        # Adjust for model complexity
        num_layers = config.model_config.get("num_layers", 4)
        base_time *= num_layers / 4.0

        # Adjust for DANN
        if config.dann_enabled:
            base_time *= 1.2

        return base_time

    def generate_summary_report(self, results_df: pd.DataFrame) -> str:
        """
        Generate a comprehensive summary report.

        Args:
            results_df: DataFrame with benchmark results

        Returns:
            Formatted summary report
        """
        report = []
        report.append("# EEG2025 Benchmark Results Summary\n")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Aggregate results by configuration
        summary_stats = (
            results_df.groupby("config_name")
            .agg(
                {
                    "psychopathology_correlation": ["mean", "std", "max"],
                    "cross_task_correlation": ["mean", "std", "max"],
                    "training_time_hours": ["mean", "std"],
                    "peak_memory_gb": ["mean", "std"],
                    "convergence_epoch": ["mean", "std"],
                }
            )
            .round(3)
        )

        report.append("## Performance Summary\n")
        report.append(
            "| Configuration | Psych r (mean±std) | Cross-task r (mean±std) | Time (h) | Memory (GB) | Convergence |"
        )
        report.append(
            "|---------------|-------------------|-------------------------|----------|-------------|-------------|"
        )

        for config_name in summary_stats.index:
            psych_mean = summary_stats.loc[
                config_name, ("psychopathology_correlation", "mean")
            ]
            psych_std = summary_stats.loc[
                config_name, ("psychopathology_correlation", "std")
            ]
            cross_mean = summary_stats.loc[
                config_name, ("cross_task_correlation", "mean")
            ]
            cross_std = summary_stats.loc[
                config_name, ("cross_task_correlation", "std")
            ]
            time_mean = summary_stats.loc[config_name, ("training_time_hours", "mean")]
            memory_mean = summary_stats.loc[config_name, ("peak_memory_gb", "mean")]
            conv_mean = summary_stats.loc[config_name, ("convergence_epoch", "mean")]

            report.append(
                f"| {config_name} | {psych_mean:.3f}±{psych_std:.3f} | "
                f"{cross_mean:.3f}±{cross_std:.3f} | {time_mean:.1f} | "
                f"{memory_mean:.1f} | {conv_mean:.0f} |"
            )

        # Best performing configurations
        report.append("\n## Top Performing Configurations\n")

        best_psych = results_df.loc[results_df["psychopathology_correlation"].idxmax()]
        best_cross = results_df.loc[results_df["cross_task_correlation"].idxmax()]

        report.append(
            f"**Best Psychopathology Performance:** {best_psych['config_name']} "
            f"(r={best_psych['psychopathology_correlation']:.3f})"
        )
        report.append(
            f"**Best Cross-Task Performance:** {best_cross['config_name']} "
            f"(r={best_cross['cross_task_correlation']:.3f})"
        )

        # Efficiency analysis
        report.append("\n## Efficiency Analysis\n")

        # Performance per hour
        results_df["psych_per_hour"] = (
            results_df["psychopathology_correlation"]
            / results_df["training_time_hours"]
        )
        most_efficient = results_df.loc[results_df["psych_per_hour"].idxmax()]

        report.append(
            f"**Most Efficient Configuration:** {most_efficient['config_name']} "
            f"({most_efficient['psych_per_hour']:.3f} correlation/hour)"
        )

        # Statistical significance tests
        report.append("\n## Statistical Analysis\n")

        configs = results_df["config_name"].unique()
        if len(configs) > 1:
            from scipy import stats

            baseline_results = results_df[results_df["config_name"] == "baseline_cnn"][
                "psychopathology_correlation"
            ]

            report.append("**Significance tests vs baseline (p-values):**")
            for config in configs:
                if config == "baseline_cnn":
                    continue

                config_results = results_df[results_df["config_name"] == config][
                    "psychopathology_correlation"
                ]
                if len(config_results) > 1 and len(baseline_results) > 1:
                    _, p_value = stats.ttest_ind(config_results, baseline_results)
                    significance = (
                        "***"
                        if p_value < 0.001
                        else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    )
                    report.append(f"- {config}: p={p_value:.4f} {significance}")

        return "\n".join(report)

    def create_performance_plots(
        self, results_df: pd.DataFrame, save_dir: Optional[str] = None
    ):
        """
        Create comprehensive performance visualization plots.

        Args:
            results_df: DataFrame with benchmark results
            save_dir: Directory to save plots (uses results_dir if None)
        """
        if save_dir is None:
            save_dir = self.results_dir
        else:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # 1. Performance comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Psychopathology correlation
        sns.boxplot(
            data=results_df,
            x="config_name",
            y="psychopathology_correlation",
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Psychopathology Correlation by Configuration")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # Cross-task correlation
        sns.boxplot(
            data=results_df, x="config_name", y="cross_task_correlation", ax=axes[0, 1]
        )
        axes[0, 1].set_title("Cross-Task Correlation by Configuration")
        axes[0, 1].tick_params(axis="x", rotation=45)

        # Training time
        sns.boxplot(
            data=results_df, x="config_name", y="training_time_hours", ax=axes[1, 0]
        )
        axes[1, 0].set_title("Training Time by Configuration")
        axes[1, 0].tick_params(axis="x", rotation=45)

        # Memory usage
        sns.boxplot(data=results_df, x="config_name", y="peak_memory_gb", ax=axes[1, 1])
        axes[1, 1].set_title("Peak Memory Usage by Configuration")
        axes[1, 1].tick_params(axis="x", rotation=45)

        plt.tight_layout()
        plt.savefig(
            save_dir / "performance_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        # 2. Performance vs efficiency scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))

        for config in results_df["config_name"].unique():
            config_data = results_df[results_df["config_name"] == config]
            ax.scatter(
                config_data["training_time_hours"],
                config_data["psychopathology_correlation"],
                label=config,
                s=100,
                alpha=0.7,
            )

        ax.set_xlabel("Training Time (hours)")
        ax.set_ylabel("Psychopathology Correlation")
        ax.set_title("Performance vs Training Time Trade-off")
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_dir / "performance_vs_time.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 3. Detailed factor correlation heatmap
        factor_cols = [
            "p_factor_correlation",
            "internalizing_correlation",
            "externalizing_correlation",
            "attention_correlation",
        ]

        # Aggregate by configuration
        factor_means = results_df.groupby("config_name")[factor_cols].mean()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(factor_means.T, annot=True, cmap="viridis", ax=ax, fmt=".3f")
        ax.set_title("CBCL Factor Correlations by Configuration")
        ax.set_ylabel("CBCL Factors")

        plt.tight_layout()
        plt.savefig(
            save_dir / "factor_correlations_heatmap.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

        print(f"Performance plots saved to {save_dir}")


def run_comprehensive_benchmark():
    """Run a comprehensive benchmark suite with all default configurations."""

    # Initialize benchmarker
    benchmarker = ModelBenchmarker(results_dir="benchmark_results")

    # Register default configurations
    benchmarker.register_default_configs()

    # Run benchmark suite
    print("Starting comprehensive benchmark suite...")
    results_df = benchmarker.run_benchmark_suite()

    # Generate summary report
    report = benchmarker.generate_summary_report(results_df)

    # Save report
    report_path = benchmarker.results_dir / "benchmark_summary.md"
    with open(report_path, "w") as f:
        f.write(report)

    # Create performance plots
    benchmarker.create_performance_plots(results_df)

    print(f"\nBenchmark completed! Results available in {benchmarker.results_dir}")
    print(f"Summary report: {report_path}")

    return results_df, benchmarker


if __name__ == "__main__":
    # Run comprehensive benchmark
    results_df, benchmarker = run_comprehensive_benchmark()

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY")
    print("=" * 80)

    summary_stats = (
        results_df.groupby("config_name")
        .agg(
            {
                "psychopathology_correlation": ["mean", "std"],
                "training_time_hours": "mean",
            }
        )
        .round(3)
    )

    print(summary_stats)
