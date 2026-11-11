"""Configuration shared by the isolated algorithm experiments."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


RF_FEATURES = [
    "rainfall_24h",
    "rainfall_7d",
    "elevation",
    "drainage_score",
    "population_density",
    "historical_flood_count",
    "citizen_report_count",
]
TASK_NAMES = ("thunderstorms", "cloudbursts", "flash_floods")


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    horizons: int = 5
    dynamic_channels: int = 13
    baseline_channels: int = 6
    time_steps: int = 7
    grid_size: int = 16
    hidden_channels: int = 16
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent / "results"
    )

