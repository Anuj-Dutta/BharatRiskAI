"""Adapter for canonical sequence assembly and normalization artifacts."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from reports.unified_feature_schema import CHANNEL_ORDER
from scripts.datasets.build_sequences import assemble_sequences


def assemble(feature_dir: str | Path, labels: list[dict[str, Any]], label_root: str | Path | None = None) -> list[dict[str, Any]]:
    return assemble_sequences(feature_dir, labels, label_root=label_root)


def validate_sample(sample: dict[str, Any]) -> None:
    inputs = np.asarray(sample["inputs"])
    targets = np.asarray(sample["targets"])
    if inputs.ndim != 4 or inputs.shape[:2] != (7, len(CHANNEL_ORDER)):
        raise ValueError(f"Expected inputs [7, {len(CHANNEL_ORDER)}, H, W], got {inputs.shape}")
    if targets.ndim != 4 or targets.shape[:2] != (5, 3):
        raise ValueError(f"Expected targets [5, 3, H, W], got {targets.shape}")

