#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: scripts/stream_infer_demo.py
"""
Streaming EEG inference demo with GPU optimization.
Demonstrates real-time processing with CUDA streams, pinned memory,
and KV caching for transformer models.
"""
from __future__ import annotations
import argparse
import time
from collections import deque
from typing import Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn

# GPU acceleration imports (with fallbacks)
try:
    from src.gpu.triton import fused_bandpass_notch_car, make_biquad_coeffs, check_triton_availability
    TRITON_AVAILABLE = check_triton_availability()
except ImportError:
    TRITON_AVAILABLE = False

try:
    from src.gpu.cupy import compression_augmentation_suite
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class StreamingEEGModel(nn.Module):
    """
    Example streaming EEG model with KV caching support.
    
    This is a placeholder that demonstrates the interface.
    Replace with your actual backbone model.
    """
    
    def __init__(
        self, 
        num_channels: int, 
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_seq_len: int = 1000,
        dropout: float = 0.1
    ):
        super().__init__()
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        
        # Input projection
        self.input_proj = nn.Conv1d(num_channels, embed_dim, kernel_size=5, padding=2)
        
        # Transformer layers (simplified)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Output heads
        self.challenge1_head = nn.Linear(embed_dim, 2)  # response_time, success_prob
        self.challenge2_head = nn.Linear(embed_dim, 5)  # CBCL factors
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, max_seq_len, embed_dim) * 0.02
        )
        
        self.dropout = nn.Dropout(dropout)
        
    @torch.no_grad()
    def forward(
        self, 
        x: torch.Tensor, 
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False,
        return_features: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """
        Forward pass with optional KV caching for streaming.
        
        Args:
            x: (B, C, T) EEG input
            kv_cache: Optional cached keys/values from previous windows
            use_cache: Whether to use and update cache
            return_features: Whether to return intermediate features
            
        Returns:
            predictions: Dict with challenge1 and challenge2 outputs
            updated_cache: Updated KV cache for next window
        """
        B, C, T = x.shape
        
        # Input projection: (B, C, T) -> (B, embed_dim, T) -> (B, T, embed_dim)
        h = self.input_proj(x).transpose(1, 2)
        
        # Add positional encoding (truncate if needed)
        pos_len = min(T, self.max_seq_len)
        h[:, :pos_len, :] += self.pos_encoding[:, :pos_len, :]
        h = self.dropout(h)
        
        # Transformer layers
        # Note: For true KV caching, need custom attention implementation
        # This is simplified demo - PyTorch TransformerEncoder doesn't support KV cache
        for i, layer in enumerate(self.layers):
            h = layer(h)
        
        # Global average pooling over time
        features = torch.mean(h, dim=1)  # (B, embed_dim)
        
        # Prediction heads
        challenge1_out = self.challenge1_head(features)  # (B, 2)
        challenge2_out = self.challenge2_head(features)  # (B, 5)
        
        predictions = {
            'challenge1': {
                'response_time': challenge1_out[:, 0],
                'success_probability': torch.sigmoid(challenge1_out[:, 1])
            },
            'challenge2': {
                'p_factor': challenge2_out[:, 0],
                'internalizing': challenge2_out[:, 1], 
                'externalizing': challenge2_out[:, 2],
                'attention_problems': challenge2_out[:, 3],
                'diagnostic_probability': torch.sigmoid(challenge2_out[:, 4])
            }
        }
        
        if return_features:
            predictions['features'] = features
        
        # Placeholder cache update (implement proper KV caching in real model)
        updated_cache = kv_cache if use_cache else None
        
        return predictions, updated_cache


class StreamingEEGProcessor:
    """
    High-performance streaming EEG processor with GPU optimization.
    """
    
    def __init__(
        self,
        model: StreamingEEGModel,
        device: torch.device,
        window_size_s: float = 2.0,
        stride_s: float = 0.5,
        sampling_rate: int = 500,
        batch_size: int = 32,
        enable_gpu_preprocess: bool = True,
        enable_compression_augment: bool = False,
        filter_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize streaming processor.
        
        Args:
            model: EEG model for inference
            device: Torch device (cuda/cpu)
            window_size_s: Window size in seconds
            stride_s: Stride between windows in seconds
            sampling_rate: EEG sampling rate in Hz
            batch_size: Batch size for processing
            enable_gpu_preprocess: Use GPU preprocessing if available
            enable_compression_augment: Apply compression augmentation
            filter_config: Configuration for filtering
        """
        self.model = model.to(device).eval()
        self.device = device
        self.use_cuda = device.type == 'cuda'
        
        # Window parameters
        self.window_size_samples = int(window_size_s * sampling_rate)
        self.stride_samples = int(stride_s * sampling_rate)
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        
        # GPU optimization flags
        self.enable_gpu_preprocess = enable_gpu_preprocess and TRITON_AVAILABLE and self.use_cuda
        self.enable_compression_augment = enable_compression_augment and CUPY_AVAILABLE and self.use_cuda
        
        # Compile model if possible
        try:
            if self.use_cuda:
                self.model = torch.compile(self.model, mode="max-autotune")
        except Exception:
            pass  # Compilation failed, continue without
        
        # Setup filtering
        self.filter_config = filter_config or {
            'bandpass_low': 0.1,
            'bandpass_high': 40.0,
            'notch_freq': 60.0,
            'notch_q': 30.0
        }
        
        if self.enable_gpu_preprocess:
            self._setup_gpu_filtering()
        
        # Setup CUDA streams for async processing
        if self.use_cuda:
            self.copy_stream = torch.cuda.Stream()
            self.compute_stream = torch.cuda.Stream()
        else:
            self.copy_stream = None
            self.compute_stream = None
        
        # Buffers for double buffering
        self._setup_buffers()
        
        # Performance tracking
        self.latencies = deque(maxlen=1000)
        self.throughput_counter = 0
        self.total_samples_processed = 0
        
    def _setup_gpu_filtering(self):
        """Setup GPU filter coefficients."""
        try:
            self.biquad_bp1, self.biquad_bp2, self.biquad_notch = make_biquad_coeffs(
                sfreq=self.sampling_rate,
                bp_lo=self.filter_config['bandpass_low'],
                bp_hi=self.filter_config['bandpass_high'],
                notch=self.filter_config['notch_freq'],
                Q=self.filter_config['notch_q'],
                device=str(self.device)
            )
        except Exception as e:
            print(f"Warning: GPU filter setup failed: {e}")
            self.enable_gpu_preprocess = False
    
    def _setup_buffers(self):
        """Setup pinned memory buffers for efficient transfers."""
        buffer_shape = (self.batch_size, self.model.num_channels, self.window_size_samples)
        
        if self.use_cuda:
            # Host buffers (pinned memory)
            self.host_buffer_a = torch.empty(
                buffer_shape, dtype=torch.float32, pin_memory=True
            )
            self.host_buffer_b = torch.empty(
                buffer_shape, dtype=torch.float32, pin_memory=True
            )
            
            # Device buffers
            self.device_buffer_a = torch.empty(
                buffer_shape, dtype=torch.float32, device=self.device
            )
            self.device_buffer_b = torch.empty(
                buffer_shape, dtype=torch.float32, device=self.device
            )
        else:
            # CPU-only buffers
            self.host_buffer_a = torch.empty(buffer_shape, dtype=torch.float32)
            self.device_buffer_a = self.host_buffer_a
        
        # Buffer state
        self.current_buffer = 'a'
        
    def preprocess_eeg(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply EEG preprocessing pipeline.
        
        Args:
            x: (B, C, T) raw EEG data
            
        Returns:
            preprocessed: (B, C, T) filtered and normalized data
        """
        if self.enable_gpu_preprocess and x.is_cuda:
            # GPU preprocessing
            try:
                # Fused filtering
                filtered = fused_bandpass_notch_car(
                    x, self.biquad_bp1, self.biquad_bp2, self.biquad_notch
                )
                
                # RMSNorm (if available)
                try:
                    from src.gpu.triton import rmsnorm_time
                    normalized = rmsnorm_time(filtered)
                except ImportError:
                    # Fallback to torch operations
                    normalized = self._torch_normalize(filtered)
                
                return normalized
            except Exception as e:
                print(f"Warning: GPU preprocessing failed: {e}")
                # Fallback to CPU preprocessing
                return self._cpu_preprocess(x)
        else:
            # CPU preprocessing
            return self._cpu_preprocess(x)
    
    def _torch_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-based normalization fallback."""
        # Per-channel z-score normalization over time
        mean = torch.mean(x, dim=2, keepdim=True)
        std = torch.std(x, dim=2, keepdim=True) + 1e-6
        return (x - mean) / std
    
    def _cpu_preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """CPU preprocessing fallback."""
        # Simple bandpass filter using torch operations
        # This is a placeholder - implement proper filtering
        
        # High-pass filter (remove DC drift)
        x_hp = x - torch.mean(x, dim=2, keepdim=True)
        
        # Normalization
        normalized = self._torch_normalize(x_hp)
        
        return normalized
    
    def process_batch(
        self, 
        batch_data: torch.Tensor,
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]], float]:
        """
        Process a batch of EEG windows.
        
        Args:
            batch_data: (B, C, T) batch of EEG windows
            kv_cache: Optional KV cache from previous batch
            use_cache: Whether to use KV caching
            
        Returns:
            predictions: Model predictions
            updated_cache: Updated KV cache
            latency_ms: Processing latency in milliseconds
        """
        start_time = time.perf_counter()
        
        with torch.no_grad():
            # Preprocessing
            preprocessed = self.preprocess_eeg(batch_data)
            
            # Compression augmentation (if enabled)
            if self.enable_compression_augment:
                try:
                    preprocessed = compression_augmentation_suite(preprocessed)
                except Exception as e:
                    print(f"Warning: Compression augmentation failed: {e}")
            
            # Model inference
            if self.use_cuda:
                with torch.cuda.amp.autocast(enabled=True):
                    predictions, updated_cache = self.model(
                        preprocessed, kv_cache=kv_cache, use_cache=use_cache
                    )
            else:
                predictions, updated_cache = self.model(
                    preprocessed, kv_cache=kv_cache, use_cache=use_cache
                )
        
        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000.0
        
        # Update performance tracking
        self.latencies.append(latency_ms)
        self.throughput_counter += batch_data.shape[0]
        self.total_samples_processed += batch_data.shape[0]
        
        return predictions, updated_cache, latency_ms
    
    def stream_process(
        self, 
        data_generator,
        max_iterations: Optional[int] = None,
        use_kv_cache: bool = False,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Main streaming processing loop.
        
        Args:
            data_generator: Generator yielding (B, C, T) EEG batches
            max_iterations: Maximum number of iterations to run
            use_kv_cache: Whether to use KV caching across batches
            verbose: Print progress information
            
        Returns:
            performance_stats: Dictionary with performance metrics
        """
        kv_cache = None
        iteration = 0
        start_time = time.time()
        
        try:
            for batch_data in data_generator:
                if max_iterations and iteration >= max_iterations:
                    break
                
                # Ensure data is on correct device
                if isinstance(batch_data, np.ndarray):
                    batch_data = torch.from_numpy(batch_data).float()
                
                if self.use_cuda and not batch_data.is_cuda:
                    batch_data = batch_data.to(self.device, non_blocking=True)
                
                # Process batch
                predictions, kv_cache, latency = self.process_batch(
                    batch_data, kv_cache=kv_cache, use_cache=use_kv_cache
                )
                
                # Progress reporting
                if verbose and iteration % 50 == 0:
                    avg_latency = np.mean(list(self.latencies)[-50:]) if self.latencies else 0
                    print(f"Iteration {iteration}: {avg_latency:.2f}ms avg latency")
                
                iteration += 1
            
        except KeyboardInterrupt:
            print("Streaming interrupted by user")
        
        # Compute final statistics
        total_time = time.time() - start_time
        performance_stats = self._compute_performance_stats(total_time)
        
        return performance_stats
    
    def _compute_performance_stats(self, total_time: float) -> Dict[str, Any]:
        """Compute performance statistics."""
        if not self.latencies:
            return {'error': 'No latency data collected'}
        
        latencies_array = np.array(list(self.latencies))
        
        stats = {
            'latency_ms': {
                'mean': float(np.mean(latencies_array)),
                'median': float(np.median(latencies_array)),
                'p95': float(np.percentile(latencies_array, 95)),
                'p99': float(np.percentile(latencies_array, 99)),
                'min': float(np.min(latencies_array)),
                'max': float(np.max(latencies_array)),
                'std': float(np.std(latencies_array))
            },
            'throughput': {
                'total_samples': self.total_samples_processed,
                'total_time_s': total_time,
                'samples_per_second': self.total_samples_processed / total_time if total_time > 0 else 0,
                'batches_per_second': len(self.latencies) / total_time if total_time > 0 else 0
            },
            'gpu_optimization': {
                'gpu_preprocessing': self.enable_gpu_preprocess,
                'compression_augmentation': self.enable_compression_augment,
                'triton_available': TRITON_AVAILABLE,
                'cupy_available': CUPY_AVAILABLE,
                'cuda_available': torch.cuda.is_available(),
                'device': str(self.device)
            }
        }
        
        return stats


def simulate_eeg_stream(
    num_channels: int = 128,
    sampling_rate: int = 500,
    window_size_s: float = 2.0,
    stride_s: float = 0.5,
    batch_size: int = 32,
    num_batches: int = 200
):
    """
    Simulate streaming EEG data for testing.
    
    Yields:
        batches: (batch_size, num_channels, window_samples) tensors
    """
    window_samples = int(window_size_s * sampling_rate)
    
    for i in range(num_batches):
        # Generate realistic EEG-like data
        # Mix of: 1/f noise, alpha rhythm, artifacts
        batch = torch.randn(batch_size, num_channels, window_samples) * 0.5
        
        # Add 1/f noise characteristic
        for ch in range(num_channels):
            freqs = torch.fft.fftfreq(window_samples, 1/sampling_rate)
            freqs[0] = 1  # Avoid division by zero
            noise_spectrum = torch.randn(window_samples, dtype=torch.complex64) / torch.sqrt(torch.abs(freqs))
            noise_signal = torch.fft.ifft(noise_spectrum).real
            batch[:, ch, :] += noise_signal * 0.3
        
        # Add alpha rhythm around 10 Hz
        t = torch.linspace(0, window_size_s, window_samples)
        alpha_signal = 0.2 * torch.sin(2 * np.pi * 10 * t)
        batch += alpha_signal[None, None, :]
        
        yield batch


def main():
    """Main function for streaming demo."""
    parser = argparse.ArgumentParser(description="Streaming EEG Inference Demo")
    parser.add_argument("--channels", type=int, default=128, help="Number of EEG channels")
    parser.add_argument("--sfreq", type=int, default=500, help="Sampling frequency (Hz)")
    parser.add_argument("--window_s", type=float, default=2.0, help="Window size (seconds)")
    parser.add_argument("--stride_s", type=float, default=0.5, help="Stride (seconds)")
    parser.add_argument("--iterations", type=int, default=200, help="Number of iterations")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda", help="Device")
    parser.add_argument("--enable_gpu_preprocess", action="store_true", help="Enable GPU preprocessing")
    parser.add_argument("--enable_compression", action="store_true", help="Enable compression augmentation")
    parser.add_argument("--use_kv_cache", action="store_true", help="Use KV caching")
    parser.add_argument("--model_size", choices=["tiny", "small", "medium"], default="small", help="Model size")
    args = parser.parse_args()
    
    # Setup device
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Model configuration based on size
    model_configs = {
        'tiny': {'embed_dim': 64, 'num_heads': 2, 'num_layers': 1},
        'small': {'embed_dim': 128, 'num_heads': 4, 'num_layers': 2},
        'medium': {'embed_dim': 256, 'num_heads': 8, 'num_layers': 4}
    }
    config = model_configs[args.model_size]
    
    # Initialize model
    model = StreamingEEGModel(
        num_channels=args.channels,
        embed_dim=config['embed_dim'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers']
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Initialize processor
    processor = StreamingEEGProcessor(
        model=model,
        device=device,
        window_size_s=args.window_s,
        stride_s=args.stride_s,
        sampling_rate=args.sfreq,
        batch_size=args.batch_size,
        enable_gpu_preprocess=args.enable_gpu_preprocess,
        enable_compression_augment=args.enable_compression
    )
    
    # Print optimization status
    print(f"GPU preprocessing: {'✓' if processor.enable_gpu_preprocess else '✗'}")
    print(f"Compression augmentation: {'✓' if processor.enable_compression_augment else '✗'}")
    print(f"Triton available: {'✓' if TRITON_AVAILABLE else '✗'}")
    print(f"CuPy available: {'✓' if CUPY_AVAILABLE else '✗'}")
    
    # Generate simulated data
    data_stream = simulate_eeg_stream(
        num_channels=args.channels,
        sampling_rate=args.sfreq,
        window_size_s=args.window_s,
        stride_s=args.stride_s,
        batch_size=args.batch_size,
        num_batches=args.iterations
    )
    
    print(f"\nStarting streaming inference ({args.iterations} batches)...")
    
    # Run streaming processing
    performance_stats = processor.stream_process(
        data_generator=data_stream,
        max_iterations=args.iterations,
        use_kv_cache=args.use_kv_cache,
        verbose=True
    )
    
    # Print results
    print("\n" + "="*60)
    print("PERFORMANCE RESULTS")
    print("="*60)
    
    latency = performance_stats['latency_ms']
    throughput = performance_stats['throughput']
    
    print(f"Latency (ms):")
    print(f"  Mean: {latency['mean']:.2f}")
    print(f"  Median: {latency['median']:.2f}")
    print(f"  P95: {latency['p95']:.2f}")
    print(f"  P99: {latency['p99']:.2f}")
    print(f"  Range: {latency['min']:.2f} - {latency['max']:.2f}")
    
    print(f"\nThroughput:")
    print(f"  Samples/sec: {throughput['samples_per_second']:.1f}")
    print(f"  Batches/sec: {throughput['batches_per_second']:.1f}")
    print(f"  Total samples: {throughput['total_samples']:,}")
    
    print(f"\nOptimizations:")
    gpu_opt = performance_stats['gpu_optimization']
    for key, value in gpu_opt.items():
        print(f"  {key}: {value}")
    
    # Performance targets
    print(f"\nPerformance Targets:")
    target_latency = 50.0  # ms
    target_p95 = 100.0    # ms
    
    latency_pass = "✓" if latency['mean'] < target_latency else "✗"
    p95_pass = "✓" if latency['p95'] < target_p95 else "✗"
    
    print(f"  Mean latency < {target_latency}ms: {latency_pass} ({latency['mean']:.2f}ms)")
    print(f"  P95 latency < {target_p95}ms: {p95_pass} ({latency['p95']:.2f}ms)")


if __name__ == "__main__":
    main()
