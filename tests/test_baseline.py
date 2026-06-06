import pandas as pd

from src.features.window_features import FEATURE_COLUMNS
from src.models.baseline import train_random_forest_baseline


def test_train_random_forest_baseline() -> None:
    rows = []
    for activity in ["A", "B", "C", "D"]:
        for i in range(50):
            rows.append(
                {
                    "user": 1600,
                    "activity": activity,
                    "window_idx": i,
                    **{col: float(i) for col in FEATURE_COLUMNS},
                }
            )

    features_df = pd.DataFrame(rows)
    model, metrics = train_random_forest_baseline(features_df, FEATURE_COLUMNS)

    assert model.n_features_in_ == len(FEATURE_COLUMNS)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.f1_macro <= 1.0
