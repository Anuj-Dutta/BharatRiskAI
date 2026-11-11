"""Adapter for the existing inference feature extraction contract."""
from __future__ import annotations

from typing import Any

import numpy as np

from ml.nowcasting.features import build_spatiotemporal_features


def build_inputs(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    sequence, baseline = build_spatiotemporal_features(observation, time_steps=7, grid_size=16)
    if sequence.shape != (7, 13, 16, 16) or baseline.shape != (6, 16, 16):
        raise ValueError(f"Unexpected canonical input shapes: {sequence.shape}, {baseline.shape}")
    return sequence, baseline

