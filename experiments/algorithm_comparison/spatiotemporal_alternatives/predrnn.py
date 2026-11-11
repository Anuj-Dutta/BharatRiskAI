"""Compact PredRNN-style recurrent spatiotemporal alternative."""
from __future__ import annotations

import torch
from torch import nn

from ._common import ForecastHeads, validate_inputs


class PredRNN(nn.Module):
    def __init__(self, input_channels: int = 13, baseline_channels: int = 6, horizons: int = 5, hidden_channels: int = 16):
        super().__init__()
        self.input_channels = input_channels
        self.cell = nn.Conv2d(input_channels + hidden_channels, hidden_channels, 3, padding=1)
        self.baseline = nn.Conv2d(baseline_channels, hidden_channels, 1)
        self.heads = ForecastHeads(hidden_channels, horizons)

    def forward(self, sequence: torch.Tensor, baseline: torch.Tensor | None = None) -> torch.Tensor:
        batch, _, _, height, width = validate_inputs(sequence, baseline, self.input_channels)
        state = sequence.new_zeros((batch, self.cell.out_channels, height, width))
        for frame in sequence.unbind(dim=1):
            state = torch.tanh(self.cell(torch.cat((frame, state), dim=1)))
        if baseline is not None:
            state = state + self.baseline(baseline)
        return self.heads(state)

