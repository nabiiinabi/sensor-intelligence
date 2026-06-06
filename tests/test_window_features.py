import pandas as pd

from src.features.window_features import (
    AVERAGE_FEATURE_COLUMNS,
    BINNED_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    WINDOW_METADATA_COLUMNS,
    build_features_table,
)


def _make_windows_df() -> pd.DataFrame:
    rows = []
    for window_idx in range(2):
        for i in range(200):
            rows.append(
                {
                    "user": 1600,
                    "activity": "A",
                    "window_idx": window_idx,
                    "timestamp": i,
                    "x": float(i % 10),
                    "y": 1.0,
                    "z": 2.0,
                }
            )
    return pd.DataFrame(rows)


def test_build_features_table_shape_and_columns() -> None:
    features_df = build_features_table(_make_windows_df())

    assert list(features_df.columns) == WINDOW_METADATA_COLUMNS + FEATURE_COLUMNS
    assert len(features_df) == 2
    assert features_df.loc[0, "user"] == 1600
    assert features_df.loc[0, "activity"] == "A"
    assert features_df.loc[0, "window_idx"] == 0
    assert abs(features_df.loc[0, BINNED_FEATURE_COLUMNS].sum() - 3.0) < 1e-9
    assert features_df.loc[0, "XAVG"] == 4.5
    assert features_df.loc[0, "YAVG"] == 1.0
    assert features_df.loc[0, "ZAVG"] == 2.0


def test_build_features_table_includes_average_columns() -> None:
    features_df = build_features_table(_make_windows_df())
    assert list(features_df.columns[-3:]) == AVERAGE_FEATURE_COLUMNS
