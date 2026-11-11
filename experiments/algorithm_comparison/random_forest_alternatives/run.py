"""Run tabular alternatives on the exact RF data/split."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ..adapters.rf_adapter import load_split, resolve_target_column, train_test_arrays
from ..config import RF_FEATURES
from ..evaluation.metrics import regression_metrics
from ..evaluation.result_logger import log_result


def _builders():
    from .extra_trees_model import build_model as extra_trees
    from .lightgbm_model import build_model as lightgbm
    from .xgboost_model import build_model as xgboost
    return {"xgboost": xgboost, "lightgbm": lightgbm, "extra_trees": extra_trees}


def run(dataset: str | Path, target: str | None = None, output: str | Path | None = None, smoke: bool = False) -> list[dict]:
    train_frame, test_frame = load_split(dataset, target_column=target or "target")
    target_name = resolve_target_column(train_frame, target)
    x_train, y_train, x_test, y_test = train_test_arrays(train_frame, test_frame, target_name)
    results = []
    for name, builder in _builders().items():
        started = time.perf_counter()
        result = {"model": name, "status": "ok", "features": RF_FEATURES, "target": target_name}
        try:
            model = builder(n_estimators=2 if smoke else 160)
            model.fit(x_train, y_train)
            predictions = model.predict(x_test)
            result.update(regression_metrics(y_test, predictions), training_seconds=time.perf_counter() - started)
        except RuntimeError as exc:
            result.update(status="skipped", error=str(exc), training_seconds=time.perf_counter() - started)
        except Exception as exc:
            result.update(status="failed", error=f"{type(exc).__name__}: {exc}", training_seconds=time.perf_counter() - started)
        results.append(result)
        if output:
            log_result(output, result)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--target")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.dataset, args.target, args.output, args.smoke), indent=2, default=str))


if __name__ == "__main__":
    main()

