"""Data loading, cleaning and the train/test split."""
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from . import config as C


@dataclass
class Dataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list
    raw: pd.DataFrame      # cleaned frame incl. target, for EDA
    n_raw: int


def load_clean() -> pd.DataFrame:
    """Load the raw CSV and apply minimal, defensible cleaning."""
    df = pd.read_csv(C.DATA)
    n_raw = len(df)
    # Drop exact duplicate players (ignoring the surrogate id).
    df = df.drop_duplicates(subset=[c for c in df.columns if c != "PlayerID"])
    df = df.drop(columns="PlayerID")          # identifier: no predictive meaning
    assert df.isna().sum().sum() == 0, "unexpected missing values"
    df.attrs["n_raw"] = n_raw
    return df


def make_dataset(test_size: float = 0.2) -> Dataset:
    df = load_clean()
    y = df[C.TARGET].map({c: i for i, c in enumerate(C.CLASSES)})
    X = pd.get_dummies(df.drop(columns=C.TARGET), dtype=int)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=C.SEED)
    return Dataset(Xtr, Xte, ytr, yte, list(X.columns), df, df.attrs["n_raw"])
