"""Train the Flood Random Forest from a user-supplied, time-indexed dataset.

No demo file, synthetic observation, or synthetic target is created here.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, recall_score, roc_auc_score, r2_score

from ml.models.random_forest import build_model
from ml.preprocessing.validation import validate_proxy_training_frame

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "ml" / "artifacts" / "flood_random_forest.joblib"


def time_split(frame: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "issue_time" not in frame:
        raise ValueError("Training data must include issue_time for a time-based split")
    ordered = frame.sort_values("issue_time").reset_index(drop=True)
    cut = max(1, int(len(ordered) * (1 - test_fraction)))
    train, test = ordered.iloc[:cut], ordered.iloc[cut:]
    if test.empty:
        raise ValueError("Training data does not contain a non-empty temporal test period")
    if train["issue_time"].max() >= test["issue_time"].min():
        raise ValueError("Temporal split overlaps")
    return train, test


def load_training_data(path: str | Path, feature_columns: list[str], target_column: str = "target") -> pd.DataFrame:
    frame = pd.read_parquet(path) if Path(path).suffix.lower() == ".parquet" else pd.read_csv(path)
    required = {"issue_time", target_column, *feature_columns}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Training dataset is missing columns: {sorted(missing)}")
    frame["issue_time"] = pd.to_datetime(frame["issue_time"], utc=True, errors="raise")
    if frame.duplicated(subset=["issue_time"]).any():
        raise ValueError("Duplicate issue-time samples are not allowed")
    if target_column == "target":
        validate_proxy_training_frame(frame)
    usable = frame[frame[target_column].notna()].copy()
    if usable.empty:
        raise ValueError("No usable target rows remain; missing future observations are not negatives")
    return usable


def train(path: str | Path, feature_columns: list[str], target_type: str, artifact: str | Path = DEFAULT_ARTIFACT) -> dict:
    frame = load_training_data(path, feature_columns)
    if target_type == "classification" and not frame["target"].isin([0, 1]).all():
        raise ValueError("Classification training requires a binary 0/1 proxy target")
    train_frame, test_frame = time_split(frame)
    if target_type == "classification" and train_frame["target"].nunique() < 2:
        raise ValueError("Classification training requires both proxy classes in the training period")
    if target_type == "classification" and train_frame["target"].nunique() < 2:
        raise ValueError("The temporal training period must contain both proxy-target classes")
    model = build_model(target_type)
    model.fit(train_frame[feature_columns], train_frame["target"])
    predictions = model.predict(test_frame[feature_columns])
    if target_type == "classification":
        metrics = {"precision": precision_score(test_frame["target"], predictions, zero_division=0), "recall": recall_score(test_frame["target"], predictions, zero_division=0), "f1": f1_score(test_frame["target"], predictions, zero_division=0), "class_balance": test_frame["target"].value_counts(normalize=True).to_dict()}
        if len(set(test_frame["target"])) > 1:
            probabilities = model.predict_proba(test_frame[feature_columns])[:, 1]
            metrics.update({"roc_auc": roc_auc_score(test_frame["target"], probabilities), "pr_auc": average_precision_score(test_frame["target"], probabilities)})
    else:
        metrics = {"mae": mean_absolute_error(test_frame["target"], predictions), "rmse": mean_squared_error(test_frame["target"], predictions) ** 0.5, "r2": r2_score(test_frame["target"], predictions)}
    payload = {"model": model, "feature_columns": feature_columns, "target_type": target_type, "target_definition": "future satellite-derived proxy target", "dataset": str(path), "train_period": [train_frame["issue_time"].min().isoformat(), train_frame["issue_time"].max().isoformat()], "test_period": [test_frame["issue_time"].min().isoformat(), test_frame["issue_time"].max().isoformat()], "metrics": metrics}
    destination = Path(artifact)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, destination)
    return {"artifact": str(destination), "metrics": metrics, "train_rows": len(train_frame), "test_rows": len(test_frame)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--features", nargs="+", required=True)
    parser.add_argument("--target-type", choices=["classification", "regression"], required=True)
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    print(json.dumps(train(args.dataset, args.features, args.target_type, args.artifact), indent=2, default=str))


if __name__ == "__main__":
    main()
