from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.clean import clean_frame
from ml.training.train import load_training_data, train


FEATURES = [
    "rainfall_24h",
    "rainfall_7d",
    "elevation",
    "drainage_score",
    "population_density",
    "historical_flood_count",
    "citizen_report_count",
]


def _dataset(rows: int = 12) -> pd.DataFrame:
    frame = pd.DataFrame({feature: np.arange(rows, dtype=float) for feature in FEATURES})
    frame["issue_time"] = pd.date_range("2024-01-01", periods=rows, freq="h", tz="UTC")
    frame["target"] = [index % 2 for index in range(rows)]
    return frame


def test_clean_frame_does_not_replace_missing_observations():
    frame = _dataset(2)
    frame.loc[0, "elevation"] = np.nan
    cleaned = clean_frame(frame)
    assert pd.isna(cleaned.loc[0, "elevation"])


def test_training_rejects_duplicate_issue_times(tmp_path: Path):
    frame = _dataset()
    frame.loc[1, "issue_time"] = frame.loc[0, "issue_time"]
    path = tmp_path / "training.csv"
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate issue-time"):
        load_training_data(path, FEATURES)


def test_training_writes_proxy_metadata_and_model(tmp_path: Path):
    path = tmp_path / "training.csv"
    artifact = tmp_path / "model.joblib"
    _dataset().to_csv(path, index=False)

    result = train(path, FEATURES, "classification", artifact)

    assert artifact.exists()
    assert result["train_rows"] == 9
    assert result["test_rows"] == 3
