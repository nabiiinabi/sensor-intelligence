import pandas as pd

from src.features.averages import average_features


def test_average_features() -> None:
    window = pd.DataFrame({"x": [1.0, 3.0], "y": [2.0, 4.0], "z": [0.0, 2.0]})
    result = average_features(window)

    assert result == {"XAVG": 2.0, "YAVG": 3.0, "ZAVG": 1.0}
