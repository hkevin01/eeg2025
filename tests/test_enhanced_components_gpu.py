#!/usr/bin/env python3
"""GPU validation for enhanced EEG components."""

import pytest
import torch

from src.models.enhanced_components import (
    EnhancedEEGNeX,
    MultiScaleFeaturesExtractor,
    TemporalAttention,
    mixup_data,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU required for enhanced component tests"
)


def _get_device() -> torch.device:
    return torch.device("cuda")


def test_temporal_attention_forward_on_gpu():
    device = _get_device()
    module = TemporalAttention(embed_dim=96, num_heads=4, dropout=0.1).to(device)
    x = torch.randn(4, 96, 200, device=device, requires_grad=True)

    output = module(x)
    assert output.shape == x.shape
    assert output.device.type == "cuda"

    loss = output.sum()
    loss.backward()
    assert x.grad is not None


def test_multiscale_features_forward_on_gpu():
    device = _get_device()
    module = MultiScaleFeaturesExtractor(in_channels=129, out_channels=16).to(device)
    x = torch.randn(2, 129, 200, device=device, requires_grad=True)

    output = module(x)
    assert output.shape == (2, 48, 200)
    assert output.device.type == "cuda"

    output.sum().backward()
    assert x.grad is not None


def test_enhanced_eegnex_forward_on_gpu():
    device = _get_device()
    model = EnhancedEEGNeX(n_channels=129, n_times=200, n_outputs=1).to(device)
    x = torch.randn(2, 129, 200, device=device)

    output = model(x)
    assert output.shape == (2, 1)
    assert torch.isfinite(output).all()


def test_mixup_data_gpu_behavior():
    device = _get_device()
    x = torch.randn(8, 129, 200, device=device)
    y = torch.randn(8, 1, device=device)

    mixed_x, y_a, y_b, lam = mixup_data(x, y, alpha=0.4)

    assert mixed_x.shape == x.shape
    assert y_a.shape == y.shape
    assert y_b.shape == y.shape
    assert 0.0 < lam < 1.0
    assert mixed_x.device.type == "cuda"
    assert y_a.device.type == "cuda"
    assert y_b.device.type == "cuda"
