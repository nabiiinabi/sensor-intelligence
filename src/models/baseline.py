from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class BaselineMetrics:
    accuracy: float
    f1_macro: float


def train_random_forest_baseline(
    features_df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str = "activity",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[RandomForestClassifier, BaselineMetrics]:
    """Train a stratified RandomForest baseline and return test-set metrics."""
    X = features_df[feature_columns]
    y = features_df[label_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = BaselineMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        f1_macro=float(f1_score(y_test, y_pred, average="macro")),
    )
    return model, metrics
