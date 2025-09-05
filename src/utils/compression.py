"""
Compression metrics and utilities for EEG Foundation Challenge.

This module provides various compression algorithms and metrics for evaluating
the compression performance of learned representations.
"""

import logging
import time
from typing import Dict, List, Optional, Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import lz4.frame
import zstandard as zstd
import blosc2
from scipy import signal
import pywavelets as pywt

logger = logging.getLogger(__name__)


class CompressionMetrics:
    """
    Comprehensive compression metrics for EEG data and learned representations.
    """

    def __init__(self):
        """Initialize compression metrics."""
        self.algorithms = {
            'lz4': self._compress_lz4,
            'zstd': self._compress_zstd,
            'blosc2': self._compress_blosc2,
            'wavelet': self._compress_wavelet,
            'fft': self._compress_fft
        }

        self.metrics_history = []

    def compute_compression_metrics(
        self,
        data: Union[np.ndarray, torch.Tensor],
        algorithms: Optional[List[str]] = None,
        precision: str = 'float32'
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute compression metrics for given data.

        Args:
            data: Input data to compress
            algorithms: List of algorithms to use (if None, use all)
            precision: Data precision ('float32', 'float16', 'int16')

        Returns:
            Dictionary with compression metrics for each algorithm
        """
        if algorithms is None:
            algorithms = list(self.algorithms.keys())

        # Convert to numpy if tensor
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()

        # Convert precision
        data = self._convert_precision(data, precision)

        results = {}
        original_size = data.nbytes

        for algorithm in algorithms:
            if algorithm in self.algorithms:
                try:
                    compressed_data, compression_time = self._time_compression(
                        self.algorithms[algorithm], data
                    )

                    compressed_size = len(compressed_data) if isinstance(compressed_data, bytes) else compressed_data.nbytes

                    results[algorithm] = {
                        'compression_ratio': original_size / compressed_size,
                        'compression_size': compressed_size,
                        'compression_time': compression_time,
                        'bits_per_sample': (compressed_size * 8) / data.size
                    }

                except Exception as e:
                    logger.warning(f"Compression failed for {algorithm}: {e}")
                    results[algorithm] = {
                        'compression_ratio': 1.0,
                        'compression_size': original_size,
                        'compression_time': 0.0,
                        'bits_per_sample': 32.0
                    }

        # Store metrics
        self.metrics_history.append({
            'timestamp': time.time(),
            'original_size': original_size,
            'data_shape': data.shape,
            'precision': precision,
            'results': results
        })

        return results

    def _convert_precision(self, data: np.ndarray, precision: str) -> np.ndarray:
        """Convert data to specified precision."""
        if precision == 'float32':
            return data.astype(np.float32)
        elif precision == 'float16':
            return data.astype(np.float16)
        elif precision == 'int16':
            # Scale and convert to int16
            data_scaled = (data / np.abs(data).max() * 32767).astype(np.int16)
            return data_scaled
        else:
            raise ValueError(f"Unsupported precision: {precision}")

    def _time_compression(self, compress_func, data):
        """Time a compression function."""
        start_time = time.time()
        compressed = compress_func(data)
        compression_time = time.time() - start_time
        return compressed, compression_time

    def _compress_lz4(self, data: np.ndarray) -> bytes:
        """Compress using LZ4."""
        return lz4.frame.compress(data.tobytes())

    def _compress_zstd(self, data: np.ndarray) -> bytes:
        """Compress using Zstandard."""
        cctx = zstd.ZstdCompressor(level=3)
        return cctx.compress(data.tobytes())

    def _compress_blosc2(self, data: np.ndarray) -> bytes:
        """Compress using Blosc2."""
        return blosc2.compress(data.tobytes(), codec='lz4', clevel=5)

    def _compress_wavelet(self, data: np.ndarray) -> np.ndarray:
        """Compress using wavelet transform with thresholding."""
        if data.ndim == 1:
            # 1D wavelet transform
            coeffs = pywt.wavedec(data, 'db4', level=4)

            # Threshold coefficients (keep top 10%)
            threshold = np.percentile(np.abs(np.concatenate(coeffs)), 90)
            coeffs_thresh = [
                pywt.threshold(c, threshold, mode='soft') for c in coeffs
            ]

            return np.concatenate(coeffs_thresh)

        else:
            # 2D wavelet transform (channel-wise)
            compressed = []
            for channel in range(data.shape[0]):
                coeffs = pywt.wavedec(data[channel], 'db4', level=4)
                threshold = np.percentile(np.abs(np.concatenate(coeffs)), 90)
                coeffs_thresh = [
                    pywt.threshold(c, threshold, mode='soft') for c in coeffs
                ]
                compressed.append(np.concatenate(coeffs_thresh))

            return np.array(compressed)

    def _compress_fft(self, data: np.ndarray) -> np.ndarray:
        """Compress using FFT with frequency domain thresholding."""
        if data.ndim == 1:
            # 1D FFT
            fft_coeffs = np.fft.rfft(data)

            # Keep top 50% of frequencies by magnitude
            magnitudes = np.abs(fft_coeffs)
            threshold = np.percentile(magnitudes, 50)
            mask = magnitudes > threshold

            compressed_coeffs = fft_coeffs * mask
            return np.concatenate([compressed_coeffs.real, compressed_coeffs.imag])

        else:
            # 2D FFT (channel-wise)
            compressed = []
            for channel in range(data.shape[0]):
                fft_coeffs = np.fft.rfft(data[channel])
                magnitudes = np.abs(fft_coeffs)
                threshold = np.percentile(magnitudes, 50)
                mask = magnitudes > threshold
                compressed_coeffs = fft_coeffs * mask
                compressed.append(np.concatenate([compressed_coeffs.real, compressed_coeffs.imag]))

            return np.array(compressed)

    def compute_feature_compression(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Compute compression metrics for learned features.

        Args:
            features: Learned feature representations
            labels: Optional labels for supervised compression

        Returns:
            Dictionary with compression metrics
        """
        features_np = features.detach().cpu().numpy()

        # Basic compression metrics
        compression_results = self.compute_compression_metrics(features_np)

        # Feature-specific metrics
        feature_metrics = {
            'feature_dim': features.shape[-1],
            'feature_sparsity': (features == 0).float().mean().item(),
            'feature_entropy': self._compute_entropy(features_np),
            'feature_variance': features.var(dim=0).mean().item()
        }

        # Add best compression ratio
        best_ratio = max([r['compression_ratio'] for r in compression_results.values()])
        feature_metrics['best_compression_ratio'] = best_ratio

        return feature_metrics

    def _compute_entropy(self, data: np.ndarray) -> float:
        """Compute Shannon entropy of data."""
        # Quantize data for entropy calculation
        data_quantized = np.round(data * 1000).astype(int)
        _, counts = np.unique(data_quantized, return_counts=True)

        # Compute probabilities
        probs = counts / counts.sum()

        # Compute entropy
        entropy = -np.sum(probs * np.log2(probs + 1e-12))
        return entropy

    def get_compression_summary(self) -> Dict[str, float]:
        """Get summary statistics of compression performance."""
        if not self.metrics_history:
            return {}

        # Aggregate metrics across all measurements
        all_ratios = []
        all_times = []
        all_sizes = []

        for entry in self.metrics_history:
            for algorithm, metrics in entry['results'].items():
                all_ratios.append(metrics['compression_ratio'])
                all_times.append(metrics['compression_time'])
                all_sizes.append(metrics['compression_size'])

        return {
            'mean_compression_ratio': np.mean(all_ratios),
            'std_compression_ratio': np.std(all_ratios),
            'mean_compression_time': np.mean(all_times),
            'total_compressed_size': sum(all_sizes),
            'num_measurements': len(self.metrics_history)
        }


class AdaptiveQuantizer(nn.Module):
    """
    Learnable quantization for neural network compression.
    """

    def __init__(
        self,
        num_bits: int = 8,
        signed: bool = True,
        learnable_scale: bool = True
    ):
        """
        Initialize adaptive quantizer.

        Args:
            num_bits: Number of quantization bits
            signed: Whether to use signed quantization
            learnable_scale: Whether to learn quantization scale
        """
        super().__init__()
        self.num_bits = num_bits
        self.signed = signed

        # Quantization levels
        if signed:
            self.qmin = -(2 ** (num_bits - 1))
            self.qmax = 2 ** (num_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** num_bits - 1

        # Learnable scale parameter
        if learnable_scale:
            self.scale = nn.Parameter(torch.ones(1))
        else:
            self.register_buffer('scale', torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantize input tensor.

        Args:
            x: Input tensor

        Returns:
            Quantized tensor
        """
        # Compute scale if not learnable
        if not hasattr(self, 'scale') or not self.scale.requires_grad:
            scale = x.abs().max() / self.qmax
        else:
            scale = self.scale

        # Quantize
        x_scaled = x / scale
        x_quantized = torch.clamp(torch.round(x_scaled), self.qmin, self.qmax)
        x_dequantized = x_quantized * scale

        return x_dequantized

    def get_compression_ratio(self, input_precision: int = 32) -> float:
        """Get theoretical compression ratio."""
        return input_precision / self.num_bits


class CompressionAwareLoss(nn.Module):
    """
    Loss function that incorporates compression metrics.
    """

    def __init__(
        self,
        base_loss: nn.Module,
        compression_weight: float = 0.1,
        target_compression_ratio: float = 10.0
    ):
        """
        Initialize compression-aware loss.

        Args:
            base_loss: Base loss function (e.g., CrossEntropyLoss)
            compression_weight: Weight for compression term
            target_compression_ratio: Target compression ratio
        """
        super().__init__()
        self.base_loss = base_loss
        self.compression_weight = compression_weight
        self.target_compression_ratio = target_compression_ratio
        self.compression_metrics = CompressionMetrics()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        features: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute compression-aware loss.

        Args:
            predictions: Model predictions
            targets: Ground truth targets
            features: Intermediate features for compression evaluation

        Returns:
            Dictionary with loss components
        """
        # Base loss
        base_loss = self.base_loss(predictions, targets)

        # Compression loss
        compression_loss = torch.tensor(0.0, device=predictions.device)

        if features is not None:
            # Compute compression metrics
            feature_metrics = self.compression_metrics.compute_feature_compression(features)
            current_ratio = feature_metrics.get('best_compression_ratio', 1.0)

            # Compression penalty (encourage compression)
            compression_penalty = torch.relu(
                torch.tensor(self.target_compression_ratio) - torch.tensor(current_ratio)
            )
            compression_loss = compression_penalty ** 2

        # Total loss
        total_loss = base_loss + self.compression_weight * compression_loss

        return {
            'total_loss': total_loss,
            'base_loss': base_loss,
            'compression_loss': compression_loss
        }


def evaluate_model_compression(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    device: str = 'cuda'
) -> Dict[str, float]:
    """
    Evaluate compression performance of a trained model.

    Args:
        model: Trained model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on

    Returns:
        Dictionary with compression metrics
    """
    model.eval()
    compression_metrics = CompressionMetrics()

    all_features = []
    all_predictions = []

    with torch.no_grad():
        for batch in data_loader:
            x = batch['eeg'].to(device)

            # Forward pass
            if hasattr(model, 'backbone'):
                features = model.backbone(x)
                outputs = model(x)
            else:
                outputs = model(x)
                features = outputs  # Assume outputs are features

            all_features.append(features.cpu())
            all_predictions.append(outputs.cpu())

    # Concatenate all features
    all_features = torch.cat(all_features, dim=0)
    all_predictions = torch.cat(all_predictions, dim=0)

    # Compute compression metrics
    feature_metrics = compression_metrics.compute_feature_compression(all_features)
    prediction_metrics = compression_metrics.compute_compression_metrics(all_predictions.numpy())

    # Combine results
    results = {
        'feature_compression_ratio': feature_metrics.get('best_compression_ratio', 1.0),
        'feature_sparsity': feature_metrics.get('feature_sparsity', 0.0),
        'feature_entropy': feature_metrics.get('feature_entropy', 0.0),
        'prediction_compression_ratio': max([m['compression_ratio'] for m in prediction_metrics.values()]),
        'model_parameters': sum(p.numel() for p in model.parameters()),
        'model_size_mb': sum(p.numel() * 4 for p in model.parameters()) / (1024 ** 2)  # Assuming float32
    }

    return results
