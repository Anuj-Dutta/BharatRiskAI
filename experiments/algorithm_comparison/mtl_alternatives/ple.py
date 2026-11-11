"""Progressive layered extraction sequence model."""
from __future__ import annotations

import torch
from torch import nn

from ._common import TASKS, TaskHeads, validate_sequence


class PLE(nn.Module):
    def __init__(self, input_channels: int = 13, horizons: int = 5, hidden_channels: int = 16):
        super().__init__()
        self.horizons = horizons
        self.shared = nn.Conv3d(input_channels, hidden_channels, 3, padding=1)
        self.shared_expert = nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1)
        self.task_experts = nn.ModuleList(nn.Conv3d(hidden_channels, hidden_channels, 3, padding=1) for _ in TASKS)
        self.gates = nn.ModuleList(nn.Conv3d(hidden_channels, len(TASKS) + 1, 1) for _ in TASKS)
        self.heads = TaskHeads(hidden_channels, horizons)

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        validate_sequence(inputs, self.shared.in_channels)
        shared = torch.relu(self.shared(inputs.permute(0, 2, 1, 3, 4)))
        shared_expert = torch.relu(self.shared_expert(shared))
        task_features = []
        for index, expert in enumerate(self.task_experts):
            task_expert = torch.relu(expert(shared))
            candidates = torch.stack([shared_expert, *[
                torch.relu(other(shared)) for other in self.task_experts
            ]], dim=1)
            weights = torch.softmax(self.gates[index](shared), dim=1).unsqueeze(2)
            task_features.append((candidates * weights).sum(dim=1).mean(dim=2))
        outputs = self.heads(task_features, self.horizons)
        return outputs if mask is None else outputs * mask

