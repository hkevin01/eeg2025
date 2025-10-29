"""EEGNet-style backbone for regression on EEG windows.

Based on "EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer
Interfaces" (Lawhern et al., 2018). Depthwise and separable convolutions make
the architecture parameter efficient while capturing spatial-temporal structure.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGNetBackbone(nn.Module):
    """Compact EEGNet variant tailored for (channels, time) inputs."""

    def __init__(
        self,
        n_channels: int = 129,
        n_times: int = 200,
        temporal_kernel: int = 64,
        F1: int = 8,
        D: int = 2,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()

        F2 = F1 * D
        padding = temporal_kernel // 2

        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, temporal_kernel), padding=(0, padding), bias=False),
            nn.BatchNorm2d(F1),
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F2, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        self.separable = nn.Sequential(
            nn.Conv2d(F2, F2, (1, 16), groups=F2, padding=(0, 8), bias=False),
            nn.Conv2d(F2, F2, 1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )

        # Determine flattened dimension with a dummy forward pass
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_times)
            feat = self.separable(self.depthwise(self.firstconv(dummy)))
            self._n_features = feat.shape[1] * feat.shape[3]

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._n_features, F2 * 4),
            nn.ELU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(F2 * 4, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (B, 1, channels, time)
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        return self.head(x)


__all__ = ["EEGNetBackbone"]