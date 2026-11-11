"""Multi-task attention network sequence model."""
from __future__ import annotations

import torch
from torch import nn

from ._common import TASKS, TaskHeads, validate_sequence


class MTAN(nn.Module):
    def __init__(self, input_channels: int = 13, horizons: int = 5, hidden_channels: int = 16):
        super().__init__()
        self.horizons = horizons
        self.encoder = nn.Sequential(
            nn.Conv3d(input_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1),
            nn.GELU(),
        )
        self.attention = nn.ModuleList(nn.Conv3d(hidden_channels, 1, 1) for _ in TASKS)
        self.heads = TaskHeads(hidden_channels, horizons)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        validate_sequence(inputs, self.encoder[0].in_channels)
        encoded = self.encoder(inputs.permute(0, 2, 1, 3, 4))
        features = []
        for attention in self.attention:
            weights = torch.sigmoid(attention(encoded))
            features.append((encoded * weights).mean(dim=2))
        outputs = self.heads(features, self.horizons)
        return outputs if mask is None else outputs * mask

