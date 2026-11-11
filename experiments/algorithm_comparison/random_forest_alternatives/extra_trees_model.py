"""Extra Trees alternative using the already-required scikit-learn dependency."""
from __future__ import annotations

def build_model(random_state: int = 42, **kwargs):
    try:
        from sklearn.ensemble import ExtraTreesRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(f"Extra Trees experiment dependencies are unavailable; skipping this experiment: {exc}") from exc
    params = {"n_estimators": 160, "min_samples_leaf": 2, "random_state": random_state, "n_jobs": -1}
    params.update(kwargs)
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", ExtraTreesRegressor(**params))])
