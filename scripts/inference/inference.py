"""
Inference Script
================

End-to-end inference pipeline for EEG foundation model.
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
import numpy as np
import yaml

# Try to import project modules with fallbacks
try:
    from src.models.backbone.eeg_transformer import EEGTransformer
    from src.models.adapters.task_aware import TaskAwareAdapter
    from src.models.heads.temporal_regression import TemporalRegressionHead
except ImportError as e:
    print(f"Warning: Could not import project modules: {e}")
    EEGTransformer = None


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """Create model from configuration."""
    if EEGTransformer is None:
        # Fallback model for testing
        return create_dummy_model(config)

    model_config = config.get('model', {})

    # Create backbone
    backbone = EEGTransformer(
        n_channels=model_config.get('n_channels', 128),
        d_model=model_config.get('d_model', 768),
        n_layers=model_config.get('n_layers', 12),
        n_heads=model_config.get('n_heads', 12),
        dropout=model_config.get('dropout', 0.1)
    )

    # Add task adapter if enabled
    if model_config.get('use_task_adapters', False):
        adapter = TaskAwareAdapter(
            d_model=model_config.get('d_model', 768),
            num_tasks=model_config.get('num_tasks', 6)
        )
        backbone = nn.Sequential(backbone, adapter)

    return backbone


def create_dummy_model(config: Dict[str, Any]) -> nn.Module:
    """Create dummy model for testing when imports fail."""
    model_config = config.get('model', {})

    n_channels = model_config.get('n_channels', 128)
    d_model = model_config.get('d_model', 768)

    return nn.Sequential(
        nn.Linear(n_channels, d_model),
        nn.ReLU(),
        nn.Linear(d_model, d_model // 2),
        nn.ReLU(),
        nn.Linear(d_model // 2, 64)
    )


def generate_synthetic_eeg(
    batch_size: int = 1,
    n_channels: int = 128,
    seq_len: int = 1000,
    sampling_rate: int = 500
) -> torch.Tensor:
    """
    Generate synthetic EEG data for testing.

    Args:
        batch_size: Number of samples
        n_channels: Number of EEG channels
        seq_len: Sequence length
        sampling_rate: Sampling rate in Hz

    Returns:
        Synthetic EEG tensor
    """
    # Time vector
    duration = seq_len / sampling_rate
    t = torch.linspace(0, duration, seq_len)

    # Generate realistic EEG-like signals
    eeg_data = torch.zeros(batch_size, n_channels, seq_len)

    for batch in range(batch_size):
        for ch in range(n_channels):
            # Base noise
            signal = 0.1 * torch.randn(seq_len)

            # Add frequency components typical of EEG
            # Alpha waves (8-12 Hz)
            alpha_freq = 8 + 4 * torch.rand(1)
            alpha_amp = 0.05 + 0.03 * torch.rand(1)
            signal += alpha_amp * torch.sin(2 * np.pi * alpha_freq * t)

            # Beta waves (13-30 Hz)
            beta_freq = 13 + 17 * torch.rand(1)
            beta_amp = 0.02 + 0.02 * torch.rand(1)
            signal += beta_amp * torch.sin(2 * np.pi * beta_freq * t)

            # Theta waves (4-8 Hz)
            theta_freq = 4 + 4 * torch.rand(1)
            theta_amp = 0.03 + 0.02 * torch.rand(1)
            signal += theta_amp * torch.sin(2 * np.pi * theta_freq * t)

            # Scale to microvolts
            signal *= 50e-6

            eeg_data[batch, ch] = signal

    return eeg_data


class EEGInferencePipeline:
    """Complete inference pipeline for EEG processing."""

    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, Any],
        device: str = "auto"
    ):
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.config = config

        self.logger = logging.getLogger(__name__)

        # Preprocessing parameters
        inference_config = config.get('inference', {})
        self.use_streaming = inference_config.get('enable_streaming', True)
        self.buffer_size = inference_config.get('buffer_size', 4096)

        # Performance tracking
        self.inference_times = []
        self.batch_sizes = []

    def preprocess(self, eeg_data: torch.Tensor) -> torch.Tensor:
        """
        Preprocess EEG data for inference.

        Args:
            eeg_data: Raw EEG data [batch_size, n_channels, seq_len]

        Returns:
            Preprocessed EEG data
        """
        # Move to device
        eeg_data = eeg_data.to(self.device)

        # Normalization (z-score)
        mean = eeg_data.mean(dim=-1, keepdim=True)
        std = eeg_data.std(dim=-1, keepdim=True) + 1e-8
        eeg_data = (eeg_data - mean) / std

        # Additional preprocessing can be added here
        # - Filtering
        # - Artifact removal
        # - Channel selection

        return eeg_data

    def inference_single(self, eeg_data: torch.Tensor) -> Dict[str, Any]:
        """
        Run inference on a single batch.

        Args:
            eeg_data: EEG data [batch_size, n_channels, seq_len]

        Returns:
            Inference results
        """
        start_time = time.time()

        # Preprocess
        processed_data = self.preprocess(eeg_data)

        # Model inference
        with torch.no_grad():
            if hasattr(self.model, 'forward'):
                outputs = self.model(processed_data)
            else:
                # Fallback for simple models
                if len(processed_data.shape) == 3:
                    # Global average pooling
                    processed_data = processed_data.mean(dim=-1)
                outputs = self.model(processed_data)

        inference_time = time.time() - start_time

        # Track performance
        self.inference_times.append(inference_time)
        self.batch_sizes.append(eeg_data.shape[0])

        # Format outputs
        if isinstance(outputs, torch.Tensor):
            outputs = outputs.cpu().numpy()
        elif isinstance(outputs, dict):
            outputs = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                      for k, v in outputs.items()}

        return {
            'predictions': outputs,
            'inference_time_ms': inference_time * 1000,
            'batch_size': eeg_data.shape[0],
            'input_shape': list(eeg_data.shape)
        }

    def inference_streaming(
        self,
        eeg_stream: torch.Tensor,
        window_size: int = 1000,
        overlap: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Run streaming inference on continuous EEG data.

        Args:
            eeg_stream: Continuous EEG data [n_channels, total_length]
            window_size: Window size for processing
            overlap: Overlap between windows (0.0-1.0)

        Returns:
            List of inference results
        """
        n_channels, total_length = eeg_stream.shape
        step_size = int(window_size * (1 - overlap))

        results = []

        for start_idx in range(0, total_length - window_size + 1, step_size):
            end_idx = start_idx + window_size

            # Extract window
            window_data = eeg_stream[:, start_idx:end_idx]
            batch_data = window_data.unsqueeze(0)  # Add batch dimension

            # Run inference
            result = self.inference_single(batch_data)
            result['window_start'] = start_idx
            result['window_end'] = end_idx

            results.append(result)

        return results

    def get_performance_stats(self) -> Dict[str, float]:
        """Get inference performance statistics."""
        if not self.inference_times:
            return {}

        inference_times_ms = [t * 1000 for t in self.inference_times]

        stats = {
            'mean_latency_ms': np.mean(inference_times_ms),
            'p50_latency_ms': np.percentile(inference_times_ms, 50),
            'p95_latency_ms': np.percentile(inference_times_ms, 95),
            'p99_latency_ms': np.percentile(inference_times_ms, 99),
            'min_latency_ms': np.min(inference_times_ms),
            'max_latency_ms': np.max(inference_times_ms),
            'total_inferences': len(self.inference_times),
            'total_samples': sum(self.batch_sizes)
        }

        # Calculate throughput
        total_time = sum(self.inference_times)
        if total_time > 0:
            stats['throughput_samples_per_sec'] = stats['total_samples'] / total_time
            stats['throughput_batches_per_sec'] = len(self.inference_times) / total_time

        return stats


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description="EEG Foundation Model Inference")
    parser.add_argument("--config", "-c", required=True,
                       help="Path to configuration file")
    parser.add_argument("--model-path", "-m",
                       help="Path to trained model checkpoint")
    parser.add_argument("--input-data", "-i",
                       help="Path to input EEG data")
    parser.add_argument("--output", "-o", default="inference_results.json",
                       help="Output file for results")
    parser.add_argument("--batch-size", "-b", type=int, default=1,
                       help="Batch size for inference")
    parser.add_argument("--test-mode", action="store_true",
                       help="Run in test mode with synthetic data")
    parser.add_argument("--streaming", action="store_true",
                       help="Use streaming inference")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting EEG inference pipeline...")

    try:
        # Load configuration
        config = load_config(args.config)
        logger.info(f"Loaded config from {args.config}")

        # Create model
        model = create_model_from_config(config)
        logger.info(f"Created model with {sum(p.numel() for p in model.parameters()):,} parameters")

        # Load checkpoint if provided
        if args.model_path and Path(args.model_path).exists():
            checkpoint = torch.load(args.model_path, map_location='cpu')
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            logger.info(f"Loaded model weights from {args.model_path}")

        # Create inference pipeline
        pipeline = EEGInferencePipeline(model, config)
        logger.info(f"Inference pipeline created on device: {pipeline.device}")

        # Prepare input data
        if args.test_mode or not args.input_data:
            # Generate synthetic data
            logger.info("Generating synthetic EEG data for testing...")
            eeg_data = generate_synthetic_eeg(
                batch_size=args.batch_size,
                n_channels=config.get('model', {}).get('n_channels', 128),
                seq_len=1000
            )
        else:
            # Load real data (implementation depends on data format)
            logger.info(f"Loading EEG data from {args.input_data}")
            eeg_data = torch.load(args.input_data)

        logger.info(f"Input data shape: {eeg_data.shape}")

        # Run inference
        if args.streaming and len(eeg_data.shape) == 2:
            logger.info("Running streaming inference...")
            results = pipeline.inference_streaming(eeg_data)
        else:
            logger.info("Running batch inference...")
            results = pipeline.inference_single(eeg_data)

        # Get performance statistics
        perf_stats = pipeline.get_performance_stats()
        logger.info("Performance statistics:")
        for key, value in perf_stats.items():
            logger.info(f"  {key}: {value:.4f}")

        # Save results
        output_data = {
            'results': results,
            'performance': perf_stats,
            'config': config,
            'metadata': {
                'model_parameters': sum(p.numel() for p in model.parameters()),
                'device': str(pipeline.device),
                'input_shape': list(eeg_data.shape),
                'test_mode': args.test_mode
            }
        }

        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)

        logger.info(f"Results saved to {args.output}")
        logger.info("Inference completed successfully!")

        return 0

    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
