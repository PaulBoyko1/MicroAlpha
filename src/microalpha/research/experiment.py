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
    """Evaluate regularized logistic regression on chronological folds."""
    missing = set(feature_columns + [label_column]) - set(frame.columns)
    if missing:
        raise ValueError(f"missing experiment columns: {sorted(missing)}")
    if not splits:
        raise ValueError("at least one walk-forward split is required")

    rows: list[FoldMetrics] = []
    for fold_number, split in enumerate(splits, start=1):
        train = frame.iloc[split.train].dropna(subset=feature_columns + [label_column])
        test = frame.iloc[split.test].dropna(subset=feature_columns + [label_column])
        if train.empty or test.empty:
            raise ValueError(f"fold {fold_number} has no valid rows after NaN filtering")

        x_train = train[feature_columns].to_numpy(dtype=float)
        y_train = train[label_column].to_numpy(dtype=int)
        x_test = test[feature_columns].to_numpy(dtype=float)
        y_test = test[label_column].to_numpy(dtype=int)
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
