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
    """Create expanding folds with non-overlapping, purged test windows."""
    sample_count = _positive_int(n_samples, name="n_samples")
    min_train = _positive_int(min_train_size, name="min_train_size")
    test = _positive_int(test_size, name="test_size")
    purge = _nonnegative_int(purge_size, name="purge_size")
    horizon = _nonnegative_int(label_horizon, name="label_horizon")
    if purge < horizon:
        raise ValueError("purge_size must be at least label_horizon")

    step = test if step_size is None else _positive_int(step_size, name="step_size")
    if step < test:
        raise ValueError("step_size must be at least test_size to avoid overlapping test windows")

    splits: list[WalkForwardSplit] = []
    train_end = min_train
    while True:
        test_start = train_end + purge
        test_end = test_start + test
        if test_end > sample_count:
            break
        splits.append(
            WalkForwardSplit(
                train=np.arange(0, train_end, dtype=int),
                test=np.arange(test_start, test_end, dtype=int),
            )
        )
        train_end += step
    if not splits:
        raise ValueError("configuration does not produce a complete walk-forward fold")
    return splits


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value
