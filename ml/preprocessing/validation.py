from ml.preprocessing.clean import FEATURES


def validate_frame(df):
    missing = [column for column in FEATURES + ["risk_score"] if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")


def validate_proxy_training_frame(df):
    required = {"issue_time", "target"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing proxy-training columns: {sorted(missing)}")
    if df["issue_time"].duplicated().any():
        raise ValueError("Duplicate issue-time samples are not allowed")
    values = df["target"].dropna()
    if not values.isin([0, 1]).all():
        raise ValueError("Binary proxy target must contain only 0, 1, or missing values")
