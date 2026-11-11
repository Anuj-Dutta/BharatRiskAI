"""Optional XGBoost alternative."""
from __future__ import annotations

def build_model(random_state: int = 42, **kwargs):
    try:
        from xgboost import XGBRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(f"XGBoost experiment dependencies are unavailable; skipping this experiment: {exc}") from exc
    params = {"n_estimators": 160, "max_depth": 6, "learning_rate": 0.05, "random_state": random_state}
    params.update(kwargs)
    return Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)), ("model", XGBRegressor(**params))])
