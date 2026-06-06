import pandas as pd

from src.features.averages import average_features
from src.features.binned_distribution import binned_distribution

BINNED_FEATURE_COLUMNS = (
    [f"X{i}" for i in range(10)]
    + [f"Y{i}" for i in range(10)]
    + [f"Z{i}" for i in range(10)]
)
AVERAGE_FEATURE_COLUMNS = ["XAVG", "YAVG", "ZAVG"]
FEATURE_COLUMNS = BINNED_FEATURE_COLUMNS + AVERAGE_FEATURE_COLUMNS
WINDOW_METADATA_COLUMNS = ["user", "activity", "window_idx"]


def binned_features_for_window(window: pd.DataFrame) -> dict[str, float]:
    """Build X/Y/Z binned distribution features for one window."""
    features: dict[str, float] = {}
    features.update(binned_distribution(window["x"], prefix="X"))
    features.update(binned_distribution(window["y"], prefix="Y"))
    features.update(binned_distribution(window["z"], prefix="Z"))
    return features


def window_features_for_window(window: pd.DataFrame) -> dict[str, float]:
    """Build all engineered features for one window."""
    features = binned_features_for_window(window)
    features.update(average_features(window))
    return features


def build_features_table(windows_df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per window with metadata and engineered feature columns.

    Expects ``windows_df`` from the 200-reading windowing step with columns
    ``user``, ``activity``, ``window_idx``, ``x``, ``y``, ``z``.
    """
    def extract_row(group: pd.DataFrame) -> pd.Series:
        user, activity, window_idx = group.name
        row = window_features_for_window(group)
        row["user"] = user
        row["activity"] = activity
        row["window_idx"] = window_idx
        return pd.Series(row)

    features_df = (
        windows_df.groupby(["user", "activity", "window_idx"], group_keys=False)
        .apply(extract_row, include_groups=False)
        .reset_index(drop=True)
    )
    return features_df[WINDOW_METADATA_COLUMNS + FEATURE_COLUMNS]


def build_binned_features_table(windows_df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible alias; use ``build_features_table`` instead."""
    return build_features_table(windows_df)
