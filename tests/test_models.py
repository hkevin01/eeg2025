#!/usr/bin/env python3
"""Test model architectures and input/output shapes."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def test_eegnex_input_shape():
    """Test that EEGNeX accepts proper 3D input tensors (batch, channels, time)."""
    try:
        from braindecode.models import EEGNeX
    except ImportError:
        pytest.skip("braindecode not installed")

    # Test parameters matching Challenge 2
    n_chans = 129
    n_times = 200  # 2 seconds at 100 Hz
    batch_size = 8

    # Initialize model
    model = EEGNeX(
        n_chans=n_chans,
        n_outputs=1,
        n_times=n_times,
    )
    model.eval()

    # Test with correct 3D input shape
    x_3d = torch.randn(batch_size, n_chans, n_times)

    with torch.no_grad():
        output = model(x_3d)

    assert output.shape == (batch_size, 1), f"Expected output shape ({batch_size}, 1), got {output.shape}"

    # Test that 4D input (wrong) would fail
    x_4d = torch.randn(batch_size, 1, n_chans, n_times)

    with pytest.raises(Exception):
        with torch.no_grad():
            _ = model(x_4d)


def test_eegnex_various_time_lengths():
    """Test EEGNeX with different time dimensions."""
    try:
        from braindecode.models import EEGNeX
    except ImportError:
        pytest.skip("braindecode not installed")

    n_chans = 129
    batch_size = 4

    # Test different time lengths
    time_lengths = [100, 200, 400, 500, 1000]

    for n_times in time_lengths:
        model = EEGNeX(
            n_chans=n_chans,
            n_outputs=1,
            n_times=n_times,
        )
        model.eval()

        x = torch.randn(batch_size, n_chans, n_times)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 1), f"For n_times={n_times}, expected ({batch_size}, 1), got {output.shape}"


def test_eegnex_gradient_flow():
    """Test that gradients flow properly through EEGNeX."""
    try:
        from braindecode.models import EEGNeX
    except ImportError:
        pytest.skip("braindecode not installed")

    n_chans = 129
    n_times = 200
    batch_size = 4

    model = EEGNeX(
        n_chans=n_chans,
        n_outputs=1,
        n_times=n_times,
    )
    model.train()

    # Create input and target
    x = torch.randn(batch_size, n_chans, n_times, requires_grad=True)
    target = torch.randn(batch_size, 1)

    # Forward pass
    output = model(x)
    loss = torch.nn.functional.l1_loss(output, target)

    # Backward pass
    loss.backward()

    # Check that gradients exist
    assert x.grad is not None, "Input gradient is None"
    assert x.grad.shape == x.shape, f"Input gradient shape mismatch: {x.grad.shape} vs {x.shape}"

    # Check that model parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"


def test_temporal_cnn_basic():
    """Test basic TemporalCNN functionality."""
