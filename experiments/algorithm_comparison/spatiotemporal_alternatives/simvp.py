"""SimVP-style convolutional video prediction alternative."""
from __future__ import annotations

import torch
from torch import nn

from ._common import ForecastHeads, validate_inputs


class SimVP(nn.Module):
    def __init__(self, input_channels: int = 13, baseline_channels: int = 6, horizons: int = 5, hidden_channels: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.baseline = nn.Conv2d(baseline_channels, hidden_channels, 1)
        self.heads = ForecastHeads(hidden_channels, horizons)

    def forward(self, sequence: torch.Tensor, baseline: torch.Tensor | None = None) -> torch.Tensor:
        validate_inputs(sequence, baseline, self.encoder[0].in_channels)
        encoded = self.encoder(sequence.mean(dim=1))
        if baseline is not None:
            encoded = encoded + self.baseline(baseline)
        return self.heads(encoded)

