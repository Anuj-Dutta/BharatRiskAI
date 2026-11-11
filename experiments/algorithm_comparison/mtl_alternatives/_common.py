"""Small shared building blocks for experimental multi-task networks."""
from __future__ import annotations

import torch
from torch import nn


TASKS = ("thunderstorms", "cloudbursts", "flash_floods")


def validate_sequence(inputs: torch.Tensor, channels: int = 13, time_steps: int = 7) -> tuple[int, int, int, int, int]:
    if inputs.ndim != 5:
        raise ValueError("inputs must have shape [batch, time, channels, height, width]")
    shape = tuple(inputs.shape)
    if shape[1] != time_steps or shape[2] != channels:
        raise ValueError(f"Expected [B, {time_steps}, {channels}, H, W], got {shape}")
    return shape


def output_heads(features: list[torch.Tensor], horizons: int) -> torch.Tensor:
    return torch.stack([head for head in features], dim=2).sigmoid()


class TaskHeads(nn.Module):
    def __init__(self, hidden: int, horizons: int):
        super().__init__()
        self.heads = nn.ModuleList(nn.Conv2d(hidden, horizons, 1) for _ in TASKS)

    def forward(self, feature_maps: list[torch.Tensor], horizons: int) -> torch.Tensor:
        del horizons
        return output_heads([head(feature_maps[index]) for index, head in enumerate(self.heads)], self.heads[0].out_channels)
