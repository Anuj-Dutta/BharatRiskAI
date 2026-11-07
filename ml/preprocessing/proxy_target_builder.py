"""Future-observation satellite-derived proxy targets.

These targets are not ground truth and are never generated from issue-time
features. Missing future observations are excluded, not treated as negatives.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProxyTargetConfig:
    horizon_hours: float = 6.0
    threshold_mm: float = 50.0
    window_hours: float = 3.0
    version: str = "qpe-future-accumulation-v1"

    @property
    def definition(self) -> str:
        return f"future QPE accumulation >= {self.threshold_mm:g} mm over {self.window_hours:g} h within {self.horizon_hours:g} h"


def build_future_qpe_targets(
    observations: pd.DataFrame,
    config: ProxyTargetConfig = ProxyTargetConfig(),
) -> pd.DataFrame:
    required = {"issue_time", "qpe_time", "qpe_accumulation_mm"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")
    frame = observations.copy()
    if "grid_id" not in frame.columns:
        # Backward-compatible single-grid records remain spatially unique.
        frame["grid_id"] = "__default__"
    frame["issue_time"] = pd.to_datetime(frame["issue_time"], utc=True)
    frame["qpe_time"] = pd.to_datetime(frame["qpe_time"], utc=True)
    rows = []
    for (issue_time, grid_id), group in frame.groupby(["issue_time", "grid_id"], sort=True):
        future_end = issue_time + pd.Timedelta(hours=config.horizon_hours)
        future = group[(group["qpe_time"] > issue_time) & (group["qpe_time"] <= future_end)]
        valid = future["qpe_accumulation_mm"].notna()
        coverage = float(valid.mean()) if len(future) else 0.0
        if not len(future) or not valid.all():
            target, confidence, status = np.nan, 0.0, "UNUSABLE_MISSING_FUTURE"
        else:
            target = float((future["qpe_accumulation_mm"] >= config.threshold_mm).any())
            confidence, status = coverage, "ACCEPTED"
        rows.append({
            "issue_time": issue_time, "grid_id": grid_id, "target": target, "target_type": "binary_proxy",
            "target_definition": config.definition, "target_source": "INSAT-3DR HEM QPE",
            "target_horizon": f"{config.horizon_hours:g}h", "target_confidence": confidence,
            "target_coverage": coverage, "target_version": config.version, "target_status": status,
        })
    return pd.DataFrame(rows)


def assert_future_only(features: pd.DataFrame, targets: pd.DataFrame) -> None:
    required = {"issue_time", "grid_id"}
    if not required.issubset(features) or not required.issubset(targets):
        raise ValueError("features and targets require issue_time and grid_id")
    if features.duplicated(["issue_time", "grid_id"]).any():
        raise ValueError("duplicate issue-time/grid_id feature rows are not allowed")
    if targets.duplicated(["issue_time", "grid_id"]).any():
        raise ValueError("duplicate issue-time/grid_id target rows are not allowed")
