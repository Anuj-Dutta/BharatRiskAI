"""Shared validation and output helpers for spatiotemporal alternatives."""
from __future__ import annotations

import torch
from torch import nn


def validate_inputs(sequence: torch.Tensor, baseline: torch.Tensor | None, channels: int = 13) -> tuple[int, int, int, int, int]:
    if sequence.ndim != 5:
        raise ValueError("sequence must have shape [batch, time, channels, height, width]")
    shape = tuple(sequence.shape)
    if shape[1:] != (7, channels, 16, 16):
        raise ValueError(f"Expected sequence [B, 7, {channels}, 16, 16], got {shape}")
    if baseline is not None and tuple(baseline.shape[1:]) != (6, 16, 16):
        raise ValueError(f"Expected baseline [B, 6, 16, 16], got {tuple(baseline.shape)}")
    return shape


class ForecastHeads(nn.Module):
    def __init__(self, hidden: int, horizons: int = 5):
        super().__init__()
        self.heads = nn.ModuleList(nn.Conv2d(hidden, horizons, 1) for _ in range(3))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.stack([head(features) for head in self.heads], dim=2).sigmoid()

