from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from ml.models.baseline import heuristic_predict
from ml.models.random_forest import build_model
from ml.preprocessing.clean import FEATURES, clean_frame
from ml.preprocessing.features import split_features_target
from ml.preprocessing.validation import validate_frame
from ml.training.evaluate import evaluate_model
from ml.nowcasting.model import WeatherNowcastModel

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "ml" / "datasets" / "processed" / "flood_demo_training.csv"
ARTIFACT = ROOT / "ml" / "artifacts" / "flood_risk_model.pkl"
NOWCAST_ARTIFACT = ROOT / "ml" / "artifacts" / "weather_nowcast_model.pkl"


def load_dataset() -> pd.DataFrame:
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {DATASET}. Supply a validated historical dataset; "
            "synthetic training data is not generated automatically."
        )
    df = pd.read_csv(DATASET)
    validate_frame(df)
    return clean_frame(df)


def main():
    df = load_dataset()
    x, y = split_features_target(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model = build_model()
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT)
    print({"artifact": str(ARTIFACT), "features": FEATURES, "metrics": metrics, "data": str(DATASET.relative_to(ROOT))})
    
    # Persist the initialized nowcast wrapper until supervised training is available.
    nowcast_model = WeatherNowcastModel()
    joblib.dump(nowcast_model, NOWCAST_ARTIFACT)
    print({"artifact": str(NOWCAST_ARTIFACT), "status": "Weather Nowcasting Model initialized; supervised training pending"})


if __name__ == "__main__":
    main()
