"""Earthformer-style local space-time attention alternative."""
from __future__ import annotations

import torch
from torch import nn

from ._common import ForecastHeads, validate_inputs


class Earthformer(nn.Module):
    def __init__(self, input_channels: int = 13, baseline_channels: int = 6, horizons: int = 5, hidden_channels: int = 16, heads: int = 4):
        super().__init__()
        self.input_projection = nn.Conv3d(input_channels, hidden_channels, 3, padding=1)
        self.baseline_projection = nn.Conv2d(baseline_channels, hidden_channels, 1)
        self.attention = nn.MultiheadAttention(hidden_channels, heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_channels)
        self.heads = ForecastHeads(hidden_channels, horizons)

    def forward(self, sequence: torch.Tensor, baseline: torch.Tensor | None = None) -> torch.Tensor:
        batch, time, _, height, width = validate_inputs(sequence, baseline, self.input_projection.in_channels)
        encoded = self.input_projection(sequence.permute(0, 2, 1, 3, 4))
        tokens = encoded.permute(0, 2, 3, 4, 1).reshape(batch, time * height * width, -1)
        attended, _ = self.attention(tokens, tokens, tokens, need_weights=False)
        fused = self.norm(tokens + attended).reshape(batch, time, height, width, -1).mean(dim=1).permute(0, 3, 1, 2)
        if baseline is not None:
            fused = fused + self.baseline_projection(baseline)
        return self.heads(fused)

