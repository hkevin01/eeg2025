"""Test package initialization."""

# Import test modules for easier access
try:
    from . import test_dann_multi
    from . import test_adapters
    from . import test_compression_ssl
    from . import test_gpu_ops
    from . import test_heads
except ImportError:
    # Handle missing dependencies gracefully
    pass

__all__ = [
    'test_dann_multi',
    'test_adapters',
    'test_compression_ssl',
    'test_gpu_ops',
    'test_heads'
]
