"""Coverage-aware QPE rate and accumulation calculations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np


@dataclass(frozen=True)
class Accumulation:
    value: np.ndarray
    coverage: np.ndarray
    status: str


def accumulate_rate(
    rates_mm_per_hour: np.ndarray,
    times: list[datetime],
    window_hours: float,
    *,
    minimum_coverage: float = 0.9,
) -> Accumulation:
    """Integrate a rate over actual intervals; insufficient coverage stays NaN.

    Intervals are assigned to the later observation. Duplicate timestamps are
    rejected rather than counted twice.
    """
    rates = np.asarray(rates_mm_per_hour, dtype=np.float64)
    if rates.ndim < 1 or rates.shape[0] != len(times):
        raise ValueError("rates first dimension must match timestamps")
    if window_hours <= 0 or not 0 < minimum_coverage <= 1:
        raise ValueError("window_hours and minimum_coverage must be positive")
    if len(times) < 2:
        shape = rates.shape[1:]
        return Accumulation(np.full(shape, np.nan), np.zeros(shape), "INSUFFICIENT_COVERAGE")
    order = np.argsort([item.timestamp() for item in times])
    sorted_times = [times[int(i)] for i in order]
    if len({item for item in sorted_times}) != len(sorted_times):
        raise ValueError("duplicate QPE timestamps are not valid input")
    sorted_rates = rates[order]
    result = np.full(rates.shape[1:], np.nan)
    coverage = np.zeros(rates.shape[1:], dtype=np.float64)
    for end in range(1, len(sorted_times)):
        elapsed = (sorted_times[end] - sorted_times[end - 1]).total_seconds() / 3600
        if elapsed <= 0:
            raise ValueError("QPE timestamps must be strictly increasing")
        window_start = sorted_times[end] - timedelta(hours=window_hours)
        total = np.zeros(rates.shape[1:], dtype=np.float64)
        covered = np.zeros(rates.shape[1:], dtype=np.float64)
        for index in range(1, end + 1):
            interval = (sorted_times[index] - sorted_times[index - 1]).total_seconds() / 3600
            if sorted_times[index] <= window_start or interval > window_hours:
                continue
            valid = np.isfinite(sorted_rates[index])
            total = np.where(valid, total + np.nan_to_num(sorted_rates[index]) * interval, total)
            covered = np.where(valid, covered + interval, covered)
        required = window_hours * minimum_coverage
        result = np.where(covered >= required, total, np.nan)
        coverage = np.clip(covered / window_hours, 0, 1)
    return Accumulation(result, coverage, "ACCEPTED" if np.any(np.isfinite(result)) else "INSUFFICIENT_COVERAGE")
