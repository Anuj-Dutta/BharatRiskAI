"""Adapter for the existing tabular flood training pipeline."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml.training.train import load_training_data, time_split

from ..config import RF_FEATURES


def load_split(path: str | Path, target_column: str = "target") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and split exactly as ``ml.training.train`` does."""
    frame = load_training_data(path, RF_FEATURES, target_column=target_column)
    return time_split(frame)


def resolve_target_column(frame: pd.DataFrame, requested: str | None = None) -> str:
    """Use an explicitly requested target, without silently changing its meaning."""
    target = requested or ("risk_score" if "risk_score" in frame.columns else "target")
    if target not in frame.columns:
        raise ValueError(f"Dataset does not contain target column {target!r}")
    return target


def train_test_arrays(train_frame: pd.DataFrame, test_frame: pd.DataFrame, target: str) -> tuple[Any, ...]:
    return train_frame[RF_FEATURES], train_frame[target], test_frame[RF_FEATURES], test_frame[target]

