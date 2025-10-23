"""
ROCm-optimized EEGNeX variant for AMD gfx1030 GPUs.

Replaces the problematic depthwise spatial convolution with a custom
einsum-based implementation that compiles to basic HIP kernels supported on
gfx10-class devices.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils.parametrize import register_parametrization

try:  # Optional dependency from braindecode
    from braindecode.modules.parametrization import MaxNormParametrize
except Exception:  # pragma: no cover - fallback if braindecode not available
    MaxNormParametrize = None


class DepthwiseSpatialConvROCm(nn.Module):
    """Depthwise spatial convolution implemented via einsum for ROCm GPUs.

    Matches the parameter shape of ``nn.Conv2d`` with ``groups=in_channels`` and
    kernel size ``(n_chans, 1)``. This avoids relying on precompiled HIP depthwise
    kernels that are broken on AMD gfx1030 devices.
    """

    def __init__(
        self,
        in_channels: int,
        depth_multiplier: int,
        kernel_size: tuple[int, int],
        bias: bool = False,
        max_norm: float | None = 1.0,
    ) -> None:
        super().__init__()
        if bias:
            raise ValueError("DepthwiseSpatialConvROCm does not support bias.")
        if kernel_size[1] != 1:
            raise ValueError("Expected kernel width of 1 for spatial convolution.")

        self.in_channels = in_channels
        self.depth_multiplier = depth_multiplier
        self.kernel_size = kernel_size
        self.out_channels = in_channels * depth_multiplier

        weight_shape = (self.out_channels, 1, kernel_size[0], kernel_size[1])
        self.weight = nn.Parameter(torch.empty(weight_shape))
        nn.init.xavier_uniform_(self.weight, gain=1.0)

        if max_norm is not None and MaxNormParametrize is not None:
            register_parametrization(self, "weight", MaxNormParametrize(max_norm))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply spatial depthwise convolution via einsum."""

        if x.ndim != 4:
            raise ValueError(f"Expected 4D input, got shape {tuple(x.shape)}")

        batch, in_channels, n_chans, time = x.shape
        if in_channels != self.in_channels:
            raise ValueError(
                f"Input channels ({in_channels}) do not match configured value "
                f"({self.in_channels})."
            )
        if n_chans != self.kernel_size[0]:
            raise ValueError(
                f"Input spatial dimension ({n_chans}) does not match kernel height "
                f"({self.kernel_size[0]})."
            )

        # Reshape weights to (in_channels, depth_multiplier, n_chans)
        weight = self.weight.view(
            self.in_channels, self.depth_multiplier, self.kernel_size[0]
        )

        # Compute dot product across channel dimension using einsum.
        # x: (batch, in_channels, n_chans, time)
        # weight: (in_channels, depth_multiplier, n_chans)
        out = torch.einsum("bict,idc->bidt", x, weight)

        # Reshape to convolutional output shape: (batch, out_channels, 1, time)
        out = out.view(batch, self.out_channels, 1, time)
        return out


class EEGNeXROCm(nn.Module):
    """EEGNeX architecture with ROCm-friendly spatial convolution."""

    def __init__(
        self,
        n_chans: int,
        n_outputs: int,
        n_times: int,
        sfreq: int,
        filter_1: int = 8,
        filter_2: int = 32,
        depth_multiplier: int = 2,
        kernel_block_1_2: int = 64,
        kernel_block_4: int = 16,
        kernel_block_5: int = 16,
        dilation_block_4: int = 2,
        dilation_block_5: int = 4,
        avg_pool_block3: tuple[int, int] = (1, 4),
        avg_pool_block5: tuple[int, int] = (1, 8),
        drop_prob: float = 0.5,
        spatial_max_norm: float | None = 1.0,
    ) -> None:
        super().__init__()

        filter_3 = filter_2 * depth_multiplier

        self.block_1 = nn.Sequential(
            Rearrange("b c t -> b 1 c t"),
            nn.Conv2d(1, filter_1, kernel_size=(1, kernel_block_1_2), padding="same", bias=False),
            nn.BatchNorm2d(filter_1),
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(filter_1, filter_2, kernel_size=(1, kernel_block_1_2), padding="same", bias=False),
            nn.BatchNorm2d(filter_2),
        )

        self.block_3 = nn.Sequential(
            DepthwiseSpatialConvROCm(
                in_channels=filter_2,
                depth_multiplier=depth_multiplier,
                kernel_size=(n_chans, 1),
                bias=False,
                max_norm=spatial_max_norm,
            ),
            nn.BatchNorm2d(filter_3),
            nn.ELU(),
            nn.AvgPool2d(avg_pool_block3, padding=(0, avg_pool_block3[1] // 2)),
            nn.Dropout(drop_prob),
        )

        self.block_4 = nn.Sequential(
            nn.Conv2d(
                filter_3,
                filter_2,
                kernel_size=(1, kernel_block_4),
                dilation=(1, dilation_block_4),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(filter_2),
        )

        self.block_5 = nn.Sequential(
            nn.Conv2d(
                filter_2,
                filter_1,
                kernel_size=(1, kernel_block_5),
                dilation=(1, dilation_block_5),
                padding="same",
                bias=False,
            ),
            nn.BatchNorm2d(filter_1),
            nn.ELU(),
            nn.AvgPool2d(avg_pool_block5, padding=(0, avg_pool_block5[1] // 2)),
            nn.Dropout(drop_prob),
            nn.Flatten(),
        )

        # Compute flattened size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, n_chans, n_times)
            features = self._forward_features(dummy)
            final_size = features.shape[-1]

        self.final_layer = nn.Linear(final_size, n_outputs)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = self.final_layer(x)
        return x


__all__ = ["EEGNeXROCm", "DepthwiseSpatialConvROCm"]
