"""Time-ordered expanding walk-forward splits with enforced purge."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WalkForwardSplit:
    """Train/test integer indices for one chronological research fold."""

    train: np.ndarray
    test: np.ndarray


def expanding_walk_forward(
    n_samples: int,
    *,
    min_train_size: int,
    test_size: int,
    purge_size: int,
    label_horizon: int = 0,
    step_size: int | None = None,
) -> list[WalkForwardSplit]:
    """Create expanding folds while preventing target overlap into the test window."""
    if n_samples < 1:
        raise ValueError("n_samples must be positive")
    if min_train_size < 1 or test_size < 1:
        raise ValueError("min_train_size and test_size must be positive")
    if purge_size < 0 or label_horizon < 0:
        raise ValueError("purge_size and label_horizon cannot be negative")
    if purge_size < label_horizon:
        raise ValueError("purge_size must be at least label_horizon")

    step = test_size if step_size is None else step_size
    if step < 1:
        raise ValueError("step_size must be positive")

    splits: list[WalkForwardSplit] = []
    train_end = min_train_size
    while True:
        test_start = train_end + purge_size
        test_end = test_start + test_size
        if test_end > n_samples:
            break
        splits.append(
            WalkForwardSplit(
                train=np.arange(0, train_end, dtype=int),
                test=np.arange(test_start, test_end, dtype=int),
            )
        )
        train_end += step
    return splits
