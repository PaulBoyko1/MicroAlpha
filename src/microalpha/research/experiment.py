"""Leakage-aware baseline evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from microalpha.research.splits import WalkForwardSplit


@dataclass(frozen=True)
class FoldMetrics:
    fold: int
    train_rows: int
    test_rows: int
    majority_accuracy: float
    accuracy: float
    balanced_accuracy: float
    macro_f1: float


def evaluate_logistic_baseline(
    frame: pd.DataFrame,
    *,
    feature_columns: list[str],
    label_column: str,
    splits: list[WalkForwardSplit],
) -> pd.DataFrame:
    """Evaluate regularized logistic regression on validated chronological folds."""
    if not feature_columns:
        raise ValueError("at least one feature column is required")
    if len(set(feature_columns)) != len(feature_columns):
        raise ValueError("feature columns must be unique")
    if label_column in feature_columns:
        raise ValueError("label column cannot be used as a feature")
    missing = set(feature_columns + [label_column]) - set(frame.columns)
    if missing:
        raise ValueError(f"missing experiment columns: {sorted(missing)}")
    if not splits:
        raise ValueError("at least one walk-forward split is required")

    rows: list[FoldMetrics] = []
    for fold_number, split in enumerate(splits, start=1):
        _validate_split(split, n_rows=len(frame), fold_number=fold_number)
        train = frame.iloc[split.train].dropna(subset=feature_columns + [label_column])
        test = frame.iloc[split.test].dropna(subset=feature_columns + [label_column])
        if train.empty or test.empty:
            raise ValueError(f"fold {fold_number} has no valid rows after NaN filtering")

        x_train = _finite_feature_matrix(train, feature_columns, fold_number, "training")
        y_train = _integer_labels(train[label_column], fold_number, "training")
        x_test = _finite_feature_matrix(test, feature_columns, fold_number, "test")
        y_test = _integer_labels(test[label_column], fold_number, "test")
        classes, counts = np.unique(y_train, return_counts=True)
        if len(classes) < 2:
            raise ValueError(f"fold {fold_number} training labels contain fewer than two classes")
        majority_class = int(classes[np.argmax(counts)])
        majority_pred = np.full_like(y_test, majority_class)

        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("model", LogisticRegression(C=1.0, max_iter=2_000)),
            ]
        )
        model.fit(x_train, y_train)
        predicted = model.predict(x_test)
        rows.append(
            FoldMetrics(
                fold=fold_number,
                train_rows=len(train),
                test_rows=len(test),
                majority_accuracy=float(accuracy_score(y_test, majority_pred)),
                accuracy=float(accuracy_score(y_test, predicted)),
                balanced_accuracy=float(balanced_accuracy_score(y_test, predicted)),
                macro_f1=float(f1_score(y_test, predicted, average="macro", zero_division=0)),
            )
        )
    return pd.DataFrame([row.__dict__ for row in rows])


def _validate_split(split: WalkForwardSplit, *, n_rows: int, fold_number: int) -> None:
    _validate_indices(split.train, name="train", n_rows=n_rows, fold_number=fold_number)
    _validate_indices(split.test, name="test", n_rows=n_rows, fold_number=fold_number)
    if split.train[-1] >= split.test[0]:
        raise ValueError(f"fold {fold_number} train indices must precede test indices")


def _validate_indices(
    indices: np.ndarray,
    *,
    name: str,
    n_rows: int,
    fold_number: int,
) -> None:
    if not isinstance(indices, np.ndarray) or indices.ndim != 1 or len(indices) == 0:
        raise ValueError(f"fold {fold_number} {name} indices must be a non-empty 1D array")
    if not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"fold {fold_number} {name} indices must be integers")
    if (indices < 0).any() or (indices >= n_rows).any():
        raise ValueError(f"fold {fold_number} {name} indices are out of bounds")
    if not np.all(np.diff(indices) > 0):
        raise ValueError(f"fold {fold_number} {name} indices must be strictly increasing")


def _finite_feature_matrix(
    frame: pd.DataFrame,
    feature_columns: list[str],
    fold_number: int,
    partition: str,
) -> np.ndarray:
    try:
        values = frame[feature_columns].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fold {fold_number} {partition} features must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"fold {fold_number} {partition} features must be finite")
    return values


def _integer_labels(series: pd.Series, fold_number: int, partition: str) -> np.ndarray:
    try:
        values = series.to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"fold {fold_number} {partition} labels must be numeric") from exc
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"fold {fold_number} {partition} labels must be finite integers")
    return values.astype(int)
