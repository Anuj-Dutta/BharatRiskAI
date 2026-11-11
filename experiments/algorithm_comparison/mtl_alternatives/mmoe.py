"""Multi-gate mixture-of-experts sequence model."""
from __future__ import annotations

import torch
from torch import nn

from ._common import TASKS, TaskHeads, validate_sequence


class MMoE(nn.Module):
    def __init__(self, input_channels: int = 13, horizons: int = 5, hidden_channels: int = 16, experts: int = 4):
        super().__init__()
        self.horizons = horizons
        self.shared = nn.Conv3d(input_channels, hidden_channels, 3, padding=1)
        self.experts = nn.ModuleList(nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1) for _ in range(experts))
        self.gates = nn.ModuleList(nn.Conv3d(hidden_channels, experts, 1) for _ in TASKS)
        self.heads = TaskHeads(hidden_channels, horizons)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        validate_sequence(inputs, self.shared.in_channels)
        shared = torch.relu(self.shared(inputs.permute(0, 2, 1, 3, 4)))
        expert_maps = torch.stack([torch.relu(expert(shared)) for expert in self.experts], dim=1)
        features = []
        for gate in self.gates:
            weights = torch.softmax(gate(shared), dim=1).unsqueeze(2)
            features.append((expert_maps * weights).sum(dim=1).mean(dim=2))
        outputs = self.heads(features, self.horizons)
        return outputs if mask is None else outputs * mask

