FEATURES = [
    "rainfall_24h",
    "rainfall_7d",
    "elevation",
    "drainage_score",
    "population_density",
    "historical_flood_count",
    "citizen_report_count",
]


def clean_frame(df):
    """Normalize infinities without inventing values.

    Imputation is performed by the estimator pipeline after the temporal
    split, so validation and training retain missingness information.
    """
    cleaned = df.copy()
    for column in FEATURES:
        cleaned[column] = cleaned[column].replace([float("inf"), -float("inf")], float("nan"))
    return cleaned
