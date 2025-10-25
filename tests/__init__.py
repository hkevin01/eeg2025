"""Test package initialization."""

# Import test modules for easier access
try:
    from . import (
        test_adapters,
        test_compression_ssl,
        test_dann_multi,
        test_gpu_ops,
        test_heads,
    )
except ImportError:
    # Handle missing dependencies gracefully
    pass

__all__ = [
    "test_dann_multi",
    "test_adapters",
    "test_compression_ssl",
    "test_gpu_ops",
    "test_heads",
]
