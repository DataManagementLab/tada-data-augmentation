from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

NAME_KEYWORDS = [
    "target",
    "label",
    "class",
    "y",
    "outcome",
    "response",
    "dependent",
    "result",
    "output",
    "diagnosis",
    "category",
    "churn",
    "survived",
    "default",
    "fraud",
    "price",
    "score",
]


def _name_score(col_name: str) -> float:
    name = col_name.strip().lower()
    if name in NAME_KEYWORDS:
        return 1.0
    if any(kw in name for kw in NAME_KEYWORDS):
        return 0.6
    return 0.0


def _uniqueness_ratio(series: pd.Series) -> float:
    n = len(series)
    if n == 0:
        return 1.0
    return series.nunique(dropna=True) / n


def _cardinality_score(series: pd.Series) -> float:
    n = len(series)
    nunique = series.nunique(dropna=True)
    ratio = nunique / max(n, 1)

    # near-unique -> ID-like, bad candidate
    if ratio > 0.9:
        return 0.0
    # classic classification range
    if 2 <= nunique <= 50:
        return 1.0
    # numeric with moderate spread -> plausible regression target
    if pd.api.types.is_numeric_dtype(series) and ratio < 0.5:
        return 0.6
    return 0.2


def _missingness_score(series: pd.Series) -> float:
    frac_missing = series.isna().mean()
    return max(0.0, 1.0 - frac_missing * 2)  # heavy penalty for missing values


def _position_score(col_idx: int, n_cols: int) -> float:
    if col_idx == n_cols - 1 or col_idx == 0:
        return 1.0
    return 0.0


def _predictability_score(
    df: pd.DataFrame, candidate: str, categorical_columns, sample_size: int = 2000
) -> float:
    """
    Train a quick model to predict `candidate` from all other columns.
    Returns a normalized score in [0, 1], higher = more predictable.
    """
    data = df.copy()
    if len(data) > sample_size:
        data = data.sample(sample_size, random_state=0)

    y_raw = data[candidate]
    X = data.drop(columns=[candidate])

    # Drop rows where target is missing
    mask = y_raw.notna()
    y_raw, X = y_raw[mask], X[mask]
    if len(y_raw) < 20 or X.shape[1] == 0:
        return 0.0

    # Encode categorical features simply
    X = X.copy()
    for col in X.columns:
        if col in categorical_columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(X.median(numeric_only=True)).fillna(0)

    is_classification = True if candidate in categorical_columns else False

    try:
        if is_classification:
            y = LabelEncoder().fit_transform(y_raw.astype(str))
            if len(np.unique(y)) < 2:
                return 0.0
            model = RandomForestClassifier(n_estimators=50, random_state=0, n_jobs=-1)
            scores = cross_val_score(model, X, y, cv=3, scoring="accuracy")
            baseline = (
                pd.Series(y).value_counts(normalize=True).max()
            )  # majority class baseline
            score = (scores.mean() - baseline) / max(1 - baseline, 1e-6)
        else:
            model = RandomForestRegressor(n_estimators=50, random_state=0, n_jobs=-1)
            scores = cross_val_score(model, X, y_raw, cv=3, scoring="r2")
            score = scores.mean()
        return float(np.clip(score, 0.0, 1.0))
    except Exception:
        return 0.0


def detect_label_column(
    df: pd.DataFrame,
    categorical_columns: List[str],
    weights: dict = None,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Score every column as a candidate label/target column.

    Returns a DataFrame sorted by total score (descending), with the
    individual heuristic scores as columns for transparency.
    """
    weights = weights or {
        "name": 5.0,
        "cardinality": 1.0,
        "missingness": 0.5,
        "position": 0.5,
        "predictability": 2.5,  # strongest, most reliable signal
    }

    n_cols = df.shape[1]
    rows = []
    for idx, col in enumerate(df.columns):
        series = df[col]
        s_name = _name_score(col)
        s_card = _cardinality_score(series)
        s_miss = _missingness_score(series)
        s_pos = _position_score(idx, n_cols)
        s_pred = _predictability_score(df, col, categorical_columns)

        total = (
            weights["name"] * s_name
            + weights["cardinality"] * s_card
            + weights["missingness"] * s_miss
            + weights["position"] * s_pos
            + weights["predictability"] * s_pred
        )

        rows.append(
            {
                "column": col,
                "name_score": round(s_name, 3),
                "cardinality_score": round(s_card, 3),
                "missingness_score": round(s_miss, 3),
                "position_score": round(s_pos, 3),
                "predictability_score": round(s_pred, 3),
                "total_score": round(total, 3),
            }
        )

    result = (
        pd.DataFrame(rows)
        .sort_values("total_score", ascending=False)
        .reset_index(drop=True)
    )
    return result.head(top_k) if top_k else result
