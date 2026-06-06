from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

WINDOWS_FILENAME = "windows_phone_accel.pkl"
FEATURES_FILENAME = "features_phone_accel.pkl"
MODEL_FILENAME = "random_forest_baseline.joblib"


def processed_dir(project_root: Path) -> Path:
    path = project_root / "data" / "processed"
    path.mkdir(parents=True, exist_ok=True)
    return path


def models_dir(project_root: Path) -> Path:
    path = project_root / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def windows_path(project_root: Path) -> Path:
    return processed_dir(project_root) / WINDOWS_FILENAME


def features_path(project_root: Path) -> Path:
    return processed_dir(project_root) / FEATURES_FILENAME


def model_path(project_root: Path) -> Path:
    return models_dir(project_root) / MODEL_FILENAME


def save_windows_df(windows_df: pd.DataFrame, project_root: Path) -> Path:
    path = windows_path(project_root)
    windows_df.to_pickle(path)
    return path


def save_features_df(features_df: pd.DataFrame, project_root: Path) -> Path:
    path = features_path(project_root)
    features_df.to_pickle(path)
    return path


def save_model(model: RandomForestClassifier, project_root: Path) -> Path:
    path = model_path(project_root)
    joblib.dump(model, path)
    return path


def load_windows_df(project_root: Path) -> pd.DataFrame:
    return pd.read_pickle(windows_path(project_root))


def load_features_df(project_root: Path) -> pd.DataFrame:
    return pd.read_pickle(features_path(project_root))


def load_model(project_root: Path) -> RandomForestClassifier:
    return joblib.load(model_path(project_root))
