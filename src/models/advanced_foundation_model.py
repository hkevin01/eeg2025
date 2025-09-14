"""
Advanced EEG Foundation Model Integration
========================================

Complete integration of all advanced components for the EEG Foundation Challenge:
- Multi-adversary DANN for domain adaptation
- Task-aware architecture with adapters
- Compression-augmented SSL
- GPU optimization
- Inference benchmarking

This module provides a unified interface for training and deploying
state-of-the-art EEG foundation models with competitive performance.
"""

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapters import (
    TASK_NAMES,
    TaskAdapterConfig,
    TaskAwareBackbone,
    TaskSpecificHead,
    TaskTokenEmbedding,
    create_task_aware_model,
)
from .compression_ssl import (
    CompressionSSLConfig,
    CompressionSSLFramework,
    ParameterScheduler,
)
from .gpu_optimization import GPUOptimConfig, GPUOptimizer, PerformanceBenchmark
from .inference_benchmark import (
    InferenceBenchmark,
    InferenceBenchmarkConfig,
    PerformanceTarget,
)

# Import our advanced components
from .invariance.dann_multi import (
    DANNMultiConfig,
    GradientReversalLayer,
    MultiAdversaryDANN,
)


@dataclass
class FoundationModelConfig:
    """Unified configuration for advanced EEG foundation model."""

    # Model architecture
    backbone_type: str = "transformer"  # "transformer", "cnn", "hybrid"
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    dropout: float = 0.1

    # Task adaptation
    task_adapter_config: TaskAdapterConfig = None
    num_tasks: int = 6

    # Domain adaptation
    dann_config: DANNMultiConfig = None
    use_domain_adaptation: bool = True

    # SSL training
    ssl_config: CompressionSSLConfig = None
    use_compression_ssl: bool = True

    # GPU optimization
    gpu_config: GPUOptimConfig = None
    use_gpu_optimization: bool = True

    # Benchmarking
    benchmark_config: InferenceBenchmarkConfig = None

    def __post_init__(self):
        if self.task_adapter_config is None:
            self.task_adapter_config = TaskAdapterConfig(
                adapter_type="both",
                task_emb_dim=64,
                hidden_dim=self.hidden_dim // 4,
                use_task_attention=True,
            )

        if self.dann_config is None:
            self.dann_config = DANNMultiConfig(
                domains=["subject", "site"],
                feature_dim=self.hidden_dim,
                hidden_dims=[512, 256],
                use_lambda_schedule=True,
            )

        if self.ssl_config is None:
            self.ssl_config = CompressionSSLConfig(
                reconstruction_weight=1.0,
                compression_consistency_weight=0.5,
                contrastive_weight=0.3,
            )

        if self.gpu_config is None:
            self.gpu_config = GPUOptimConfig(
                use_mixed_precision=True,
                use_torch_compile=True,
                use_gradient_checkpointing=True,
            )

        if self.benchmark_config is None:
            self.benchmark_config = InferenceBenchmarkConfig(
                performance_targets=PerformanceTarget(
                    max_latency_ms=50.0, p95_latency_ms=30.0, min_throughput_qps=20.0
                )
            )


class SimpleTransformerBackbone(nn.Module):
    """Simple transformer backbone for demonstration."""

    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim

        # Input projection
        self.input_proj = nn.Conv1d(19, config.hidden_dim, kernel_size=3, padding=1)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, config.hidden_dim, 2048))

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        # Global pooling
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input EEG tensor (B, C, T)

        Returns:
            Encoded features (B, hidden_dim)
        """
        B, C, T = x.shape

        # Project to hidden dimension
        x = self.input_proj(x)  # (B, hidden_dim, T)

        # Add positional encoding
        pos_enc = self.pos_encoding[:, :, :T]
        x = x + pos_enc

        # Transpose for transformer (B, T, hidden_dim)
        x = x.transpose(1, 2)

        # Apply transformer
        x = self.transformer(x)

        # Global pooling (B, T, hidden_dim) -> (B, hidden_dim)
        x = x.transpose(1, 2)  # (B, hidden_dim, T)
        x = self.pool(x).squeeze(-1)  # (B, hidden_dim)

        return x


class SimpleDecoder(nn.Module):
    """Simple decoder for SSL reconstruction."""

    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        self.config = config

        # Upsampling layers
        self.decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim * 2, 19 * 512),  # Reconstruct to 512 timepoints
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode features back to EEG signal.

        Args:
            x: Encoded features (B, hidden_dim)

        Returns:
            Reconstructed EEG (B, C, T)
        """
        x = self.decoder(x)
        return x.view(-1, 19, 512)


class AdvancedEEGFoundationModel(nn.Module):
    """
    Advanced EEG Foundation Model with all enhancements.

    Combines:
    - Task-aware backbone with adapters
    - Multi-adversary domain adaptation
    - Compression-augmented SSL
    - GPU optimization
    - Production-ready inference
    """

    def __init__(self, config: FoundationModelConfig):
        super().__init__()
        self.config = config

        # Create backbone
        if config.backbone_type == "transformer":
            backbone = SimpleTransformerBackbone(config)
        else:
            raise ValueError(f"Unsupported backbone type: {config.backbone_type}")

        # Wrap with task-aware components
        self.task_backbone, self.task_heads = create_task_aware_model(
            backbone=backbone,
            backbone_output_dim=config.hidden_dim,
            config=config.task_adapter_config,
            num_tasks=config.num_tasks,
            output_dims={
                "regression": 1,  # RT prediction
                "classification": 1,  # Success prediction
                "psychopathology": 4,  # CBCL factors
            },
        )

        # Domain adaptation
        if config.use_domain_adaptation:
            self.domain_adapter = MultiAdversaryDANN(config.dann_config)
        else:
            self.domain_adapter = None

        # SSL framework
        if config.use_compression_ssl:
            decoder = SimpleDecoder(config)
            self.ssl_framework = CompressionSSLFramework(
                encoder=self.task_backbone.backbone,
                decoder=decoder,
                config=config.ssl_config,
            )
        else:
            self.ssl_framework = None

        # GPU optimization
        if config.use_gpu_optimization:
            self.gpu_optimizer = GPUOptimizer(config.gpu_config)
            self.optimized = False
        else:
            self.gpu_optimizer = None
            self.optimized = True

    def optimize_for_inference(self):
        """Apply GPU optimizations for inference."""
        if self.gpu_optimizer and not self.optimized:
            self.task_backbone = self.gpu_optimizer.optimize_model(self.task_backbone)
            for head_name, head in self.task_heads.items():
                self.task_heads[head_name] = self.gpu_optimizer.optimize_model(head)
            self.optimized = True

    def forward(
        self,
        x: torch.Tensor,
        task_ids: torch.Tensor,
        domain_ids: Optional[Dict[str, torch.Tensor]] = None,
        mode: str = "inference",
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multiple modes.

        Args:
            x: Input EEG tensor (B, C, T)
            task_ids: Task IDs (B,)
            domain_ids: Domain IDs for adaptation
            mode: "inference", "ssl", or "training"

        Returns:
            Dictionary of outputs depending on mode
        """
        if mode == "ssl" and self.ssl_framework is not None:
            return self.ssl_framework(x)

        # Standard forward pass
        features, task_emb = self.task_backbone(x, task_ids, return_features=True)

        # Apply domain adaptation if training
        if (
            mode == "training"
            and self.domain_adapter is not None
            and domain_ids is not None
        ):
            adapted_features, domain_losses = self.domain_adapter(features, domain_ids)
            features = adapted_features
        else:
            domain_losses = {}

        # Generate predictions from all heads
        outputs = {}
        for head_name, head in self.task_heads.items():
            outputs[head_name] = head(features, task_emb)

        # Add features and domain losses
        outputs["features"] = features
        outputs["task_embeddings"] = task_emb
        outputs.update(domain_losses)

        return outputs

    def ssl_pretrain(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_epochs: int = 10,
        device: str = "cuda",
    ) -> Dict[str, List[float]]:
        """
        Self-supervised pretraining with compression augmentation.

        Args:
            dataloader: Training data loader
            num_epochs: Number of training epochs
            device: Device for training

        Returns:
            Training history
        """
        if self.ssl_framework is None:
            raise ValueError("SSL framework not initialized")

        self.train()
        self.to(device)

        # Create optimizer
        if self.gpu_optimizer:
            optimizer = self.gpu_optimizer.create_optimizer(self.ssl_framework)
        else:
            optimizer = torch.optim.AdamW(self.ssl_framework.parameters(), lr=1e-4)

        history = {
            "total_loss": [],
            "reconstruction_loss": [],
            "consistency_loss": [],
            "contrastive_loss": [],
        }

        print(f"Starting SSL pretraining for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            epoch_losses = {
                "total_loss": [],
                "reconstruction_loss": [],
                "consistency_loss": [],
                "contrastive_loss": [],
            }

            for batch_idx, batch in enumerate(dataloader):
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(device)
                else:
                    x = batch.to(device)

                # SSL training step
                if self.gpu_optimizer:
                    step_results = self.gpu_optimizer.training_step(
                        self.ssl_framework,
                        lambda outputs, inputs: outputs["total_loss"],
                        optimizer,
                        {"x": x},
                        batch_idx,
                    )

                    # Get detailed losses
                    with torch.no_grad():
                        ssl_outputs = self.ssl_framework(x)
                        for key in epoch_losses:
                            if key in ssl_outputs:
                                epoch_losses[key].append(ssl_outputs[key].item())
                else:
                    # Standard training step
                    optimizer.zero_grad()
                    ssl_outputs = self.ssl_framework(x)
                    loss = ssl_outputs["total_loss"]
                    loss.backward()
                    optimizer.step()

                    for key in epoch_losses:
                        if key in ssl_outputs:
                            epoch_losses[key].append(ssl_outputs[key].item())

                if batch_idx % 100 == 0:
                    print(
                        f"Epoch {epoch}, Batch {batch_idx}, "
                        f"Loss: {epoch_losses['total_loss'][-1]:.4f}"
                    )

            # Record epoch averages
            for key in history:
                if epoch_losses[key]:
                    history[key].append(sum(epoch_losses[key]) / len(epoch_losses[key]))

            print(f"Epoch {epoch} completed. Avg loss: {history['total_loss'][-1]:.4f}")

        print("SSL pretraining completed!")
        return history

    def benchmark_performance(
        self, input_generator: callable, model_name: str = "advanced_eeg_model"
    ) -> Dict[str, Any]:
        """
        Comprehensive performance benchmarking.

        Args:
            input_generator: Function to generate test inputs
            model_name: Name for benchmark results

        Returns:
            Benchmark results
        """
        if not self.optimized:
            self.optimize_for_inference()

        benchmark = InferenceBenchmark(self.config.benchmark_config)

        def wrapped_input_generator(
            batch_size: int, seq_len: int
        ) -> Dict[str, torch.Tensor]:
            """Wrapper to provide task IDs."""
            x = input_generator(batch_size, seq_len)
            task_ids = torch.zeros(batch_size, dtype=torch.long)  # Default to task 0
            return {"x": x, "task_ids": task_ids}

        def forward_wrapper(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
            """Wrapper for benchmark compatibility."""
            outputs = self.forward(inputs["x"], inputs["task_ids"], mode="inference")
            return outputs["regression"]  # Return one head output

        # Temporarily replace forward method for benchmarking
        original_forward = self.forward
        self.forward = forward_wrapper

        try:
            results = benchmark.benchmark_model(
                self, wrapped_input_generator, model_name=model_name
            )
        finally:
            # Restore original forward method
            self.forward = original_forward

        return results

    def save_model(self, save_dir: str):
        """Save complete model with configuration."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(self.state_dict(), save_path / "model.pt")

        # Save configuration
        with open(save_path / "config.json", "w") as f:
            # Convert config to dict for serialization
            config_dict = asdict(self.config)
            json.dump(config_dict, f, indent=2)

        print(f"Model saved to {save_path}")

    @classmethod
    def load_model(cls, save_dir: str) -> "AdvancedEEGFoundationModel":
        """Load model from saved directory."""
        save_path = Path(save_dir)

        # Load configuration
        with open(save_path / "config.json", "r") as f:
            config_dict = json.load(f)

        # Reconstruct config object (simplified)
        config = FoundationModelConfig(**config_dict)

        # Create model
        model = cls(config)

        # Load state dict
        model.load_state_dict(torch.load(save_path / "model.pt"))

        print(f"Model loaded from {save_path}")
        return model


def create_sample_data_generator():
    """Create sample data generator for testing."""

    def generate_batch(batch_size: int, seq_len: int) -> torch.Tensor:
        # Generate realistic EEG-like signals
        t = torch.linspace(0, 10, seq_len)
        signals = []

        for _ in range(batch_size):
            # Multiple frequency components
            signal = torch.zeros(19, seq_len)
            for ch in range(19):
                # Alpha (8-13 Hz), Beta (13-30 Hz), Gamma (30-100 Hz)
                alpha = torch.sin(2 * torch.pi * (8 + torch.rand(1) * 5) * t)
                beta = 0.5 * torch.sin(2 * torch.pi * (13 + torch.rand(1) * 17) * t)
                gamma = 0.2 * torch.sin(2 * torch.pi * (30 + torch.rand(1) * 70) * t)
                noise = 0.1 * torch.randn(seq_len)

                signal[ch] = alpha + beta + gamma + noise

            signals.append(signal)

        return torch.stack(signals)

    return generate_batch


# Example usage and testing
if __name__ == "__main__":
    print("Testing Advanced EEG Foundation Model...")

    # Create configuration
    config = FoundationModelConfig(
        hidden_dim=256, num_layers=4, num_heads=8  # Smaller for testing
    )

    # Create model
    model = AdvancedEEGFoundationModel(config)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Test forward pass
    batch_size = 4
    seq_len = 1000
    x = torch.randn(batch_size, 19, seq_len)
    task_ids = torch.randint(0, 6, (batch_size,))

    print("\nTesting inference mode...")
    with torch.no_grad():
        outputs = model(x, task_ids, mode="inference")
        print(f"Outputs: {list(outputs.keys())}")
        print(f"Regression output shape: {outputs['regression'].shape}")
        print(f"Features shape: {outputs['features'].shape}")

    print("\nTesting SSL mode...")
    if model.ssl_framework:
        with torch.no_grad():
            ssl_outputs = model(x, task_ids, mode="ssl")
            print(f"SSL outputs: {list(ssl_outputs.keys())}")
            print(f"Total loss: {ssl_outputs['total_loss'].item():.4f}")

    # Test optimization
    print("\nTesting GPU optimization...")
    model.optimize_for_inference()
    print("Model optimized for inference")

    # Test benchmarking
    print("\nTesting performance benchmarking...")
    data_generator = create_sample_data_generator()

    # Quick benchmark with limited configurations
    quick_config = InferenceBenchmarkConfig(
        warmup_iterations=5,
        measure_iterations=10,
        batch_sizes=[1, 4],
        sequence_lengths=[512],
        save_results=False,
        generate_plots=False,
    )
    model.config.benchmark_config = quick_config

    benchmark_results = model.benchmark_performance(data_generator, "test_model")
    print(
        f"Benchmark completed. Performance grade: {benchmark_results['summary']['performance_grade']:.2%}"
    )

    # Test save/load
    print("\nTesting save/load...")
    save_dir = "/tmp/test_model"
    model.save_model(save_dir)

    loaded_model = AdvancedEEGFoundationModel.load_model(save_dir)
    print("Model successfully saved and loaded")

    print("\nAll tests completed successfully!")
