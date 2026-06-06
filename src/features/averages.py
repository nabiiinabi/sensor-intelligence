import pandas as pd


def average_features(window: pd.DataFrame) -> dict[str, float]:
    """Mean accelerometer value per axis (device orientation/position)."""
    return {
        "XAVG": float(window["x"].mean()),
        "YAVG": float(window["y"].mean()),
        "ZAVG": float(window["z"].mean()),
    }
