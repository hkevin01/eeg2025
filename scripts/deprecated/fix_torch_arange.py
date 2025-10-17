"""
Temporary fix for torch.arange ROCm bug
This needs to be imported before any braindecode operations
"""
import torch

# Save original arange
_original_arange = torch.arange

def fixed_arange(*args, **kwargs):
    """
    Fixed arange that always uses CPU for ROCm compatibility.
    The data will be moved to GPU later during training.
    """
    # Remove device argument if present (discard the value)
    kwargs.pop('device', None)

    # Force CPU for all arange operations
    result = _original_arange(*args, device='cpu', **kwargs)

    # If original requested cuda/gpu, we'll still return CPU
    # (data will be moved to GPU during training, not data loading)
    return result

# Monkey-patch globally
torch.arange = fixed_arange

# Also patch torch.Tensor.arange if it exists
if hasattr(torch.Tensor, 'arange'):
    torch.Tensor.arange = staticmethod(fixed_arange)

print("✅ Applied torch.arange ROCm fix (CPU-only)")
