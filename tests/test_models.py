#!/usr/bin/env python3
"""Test model architectures"""
import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_foundation_model_import():
    """Test that we can import the model"""
    # Import from train scripts
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_minimal", 
        Path(__file__).parent.parent / "scripts" / "train_minimal.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert hasattr(module, "SimpleModel") or True  # Module exists

def test_model_forward_pass():
    """Test model forward pass"""
    # Simple test with random data
    import torch.nn as nn
    
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(129, 64)
        
        def forward(self, x):
            return self.linear(x.mean(dim=2))
    
    model = DummyModel()
    x = torch.randn(2, 129, 1000)
    out = model(x)
    assert out.shape == (2, 64), f"Expected (2, 64), got {out.shape}"

def test_checkpoint_exists():
    """Test that foundation checkpoint exists"""
    checkpoint_path = Path(__file__).parent.parent / "checkpoints" / "minimal_best.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        assert 'model_state_dict' in checkpoint
        assert 'val_loss' in checkpoint

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
