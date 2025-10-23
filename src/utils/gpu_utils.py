"""
GPU Detection and Configuration Utility
Supports both NVIDIA CUDA and AMD ROCm
"""

import os
import sys
import torch
from typing import Tuple, Optional


class GPUConfig:
    """GPU configuration and detection"""
    
    def __init__(self):
        self.backend = self._detect_backend()
        self.available = torch.cuda.is_available()
        self.device_name = None
        self.device_count = 0
        self.architecture = None
        
        if self.available:
            self.device_count = torch.cuda.device_count()
            self.device_name = torch.cuda.get_device_name(0)
            if self.backend == "rocm":
                props = torch.cuda.get_device_properties(0)
                self.architecture = props.gcnArchName
    
    def _detect_backend(self) -> str:
        """Detect if using CUDA (NVIDIA) or ROCm (AMD)"""
        if not torch.cuda.is_available():
            return "none"
        
        # Check for ROCm
        if hasattr(torch.version, 'hip') and torch.version.hip is not None:
            return "rocm"
        
        # Check for CUDA
        if hasattr(torch.version, 'cuda') and torch.version.cuda is not None:
            return "cuda"
        
        return "unknown"
    
    def get_device(self, gpu_id: Optional[int] = None) -> torch.device:
        """Get appropriate device (cuda:0, cuda:1, or cpu)"""
        if not self.available:
            return torch.device('cpu')
        
        if gpu_id is None:
            return torch.device('cuda')
        
        if gpu_id >= self.device_count:
            print(f"Warning: GPU {gpu_id} not available. Using GPU 0.")
            return torch.device('cuda:0')
        
        return torch.device(f'cuda:{gpu_id}')
    
    def setup_environment(self, force_sdk: bool = False):
        """
        Setup environment for optimal GPU performance
        
        Args:
            force_sdk: If True and ROCm detected, use custom SDK path
        """
        if self.backend == "rocm":
            # AMD ROCm specific settings
            if force_sdk and os.path.exists("/opt/rocm_sdk_612"):
                self._setup_rocm_sdk()
            
            # Unset HSA override if present
            if 'HSA_OVERRIDE_GFX_VERSION' in os.environ:
                print("⚠️  Removing HSA_OVERRIDE_GFX_VERSION (not needed with proper build)")
                del os.environ['HSA_OVERRIDE_GFX_VERSION']
            
            # ROCm optimization flags
            os.environ.setdefault('HSA_TOOLS_LIB', '')
            os.environ.setdefault('ROCM_PATH', '/opt/rocm')
        
        elif self.backend == "cuda":
            # NVIDIA CUDA specific settings
            os.environ.setdefault('CUDA_LAUNCH_BLOCKING', '0')
            # Enable TF32 for faster training on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    def _setup_rocm_sdk(self):
        """Setup custom ROCm SDK environment"""
        sdk_path = "/opt/rocm_sdk_612"
        
        # Update paths
        os.environ['ROCM_SDK_PATH'] = sdk_path
        
        pythonpath = os.environ.get('PYTHONPATH', '')
        sdk_site_packages = f"{sdk_path}/lib/python3.11/site-packages"
        if sdk_site_packages not in pythonpath:
            os.environ['PYTHONPATH'] = f"{sdk_site_packages}:{pythonpath}"
        
        ld_library_path = os.environ.get('LD_LIBRARY_PATH', '')
        sdk_libs = f"{sdk_path}/lib:{sdk_path}/lib64"
        if sdk_path not in ld_library_path:
            os.environ['LD_LIBRARY_PATH'] = f"{sdk_libs}:{ld_library_path}"
        
        print(f"✅ Using custom ROCm SDK: {sdk_path}")
    
    def print_info(self):
        """Print GPU configuration information"""
        print("=" * 60)
        print("GPU Configuration")
        print("=" * 60)
        print(f"Backend: {self.backend.upper()}")
        print(f"Available: {self.available}")
        
        if self.available:
            print(f"Device Count: {self.device_count}")
            print(f"Device Name: {self.device_name}")
            
            if self.backend == "rocm":
                print(f"Architecture: {self.architecture}")
                print(f"ROCm Version: {torch.version.hip}")
            elif self.backend == "cuda":
                print(f"CUDA Version: {torch.version.cuda}")
            
            # Memory info
            for i in range(self.device_count):
                props = torch.cuda.get_device_properties(i)
                total_memory = props.total_memory / 1024**3  # Convert to GB
                print(f"GPU {i} Memory: {total_memory:.2f} GB")
        else:
            print("⚠️  No GPU available - using CPU")
        
        print("=" * 60)
    
    def get_optimal_batch_size(self, default: int = 32) -> int:
        """
        Get optimal batch size based on GPU memory
        
        Args:
            default: Default batch size if GPU not available
        
        Returns:
            Recommended batch size
        """
        if not self.available:
            return default
        
        props = torch.cuda.get_device_properties(0)
        total_memory_gb = props.total_memory / 1024**3
        
        # Heuristic: ~2GB per batch of 32
        if total_memory_gb < 4:
            return 16
        elif total_memory_gb < 8:
            return 32
        elif total_memory_gb < 12:
            return 64
        else:
            return 128
    
    def optimize_for_competition(self):
        """Apply optimizations for competition environment"""
        if not self.available:
            return
        
        # Enable cuDNN benchmarking for faster training
        torch.backends.cudnn.benchmark = True
        
        # Disable cudnn determinism for speed (can re-enable if needed)
        torch.backends.cudnn.deterministic = False
        
        if self.backend == "cuda":
            # NVIDIA specific optimizations
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        
        print("✅ Applied competition optimizations")


def get_gpu_config(verbose: bool = True) -> GPUConfig:
    """
    Get GPU configuration
    
    Args:
        verbose: Print configuration info
    
    Returns:
        GPUConfig object
    """
    config = GPUConfig()
    
    if verbose:
        config.print_info()
    
    return config


def setup_device(gpu_id: Optional[int] = None, 
                 force_sdk: bool = False,
                 optimize: bool = True) -> Tuple[torch.device, GPUConfig]:
    """
    Setup device for training
    
    Args:
        gpu_id: GPU ID to use (None = auto-select)
        force_sdk: Force use of custom ROCm SDK
        optimize: Apply performance optimizations
    
    Returns:
        Tuple of (device, gpu_config)
    """
    config = get_gpu_config(verbose=True)
    
    # Setup environment
    config.setup_environment(force_sdk=force_sdk)
    
    # Apply optimizations
    if optimize:
        config.optimize_for_competition()
    
    # Get device
    device = config.get_device(gpu_id)
    
    print(f"\n🎯 Using device: {device}")
    if config.available and config.backend == "rocm":
        print(f"   AMD GPU: {config.device_name} ({config.architecture})")
    elif config.available and config.backend == "cuda":
        print(f"   NVIDIA GPU: {config.device_name}")
    
    return device, config


if __name__ == "__main__":
    # Test GPU detection
    device, config = setup_device(optimize=True)
    
    # Test GPU operation
    if config.available:
        print("\n🧪 Testing GPU operations...")
        try:
            x = torch.randn(100, 100, device=device)
            y = torch.randn(100, 100, device=device)
            z = torch.matmul(x, y)
            print(f"✅ GPU computation successful!")
            print(f"   Result shape: {z.shape}")
            print(f"   Result device: {z.device}")
        except Exception as e:
            print(f"❌ GPU test failed: {e}")
    
    print(f"\n💡 Recommended batch size: {config.get_optimal_batch_size()}")
