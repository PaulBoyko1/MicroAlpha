"""Forward-return labels for event-time research."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def add_forward_midprice_labels(
    frame: pd.DataFrame,
    *,
    horizon_events: int,
    neutral_bps: float = 0.0,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Create forward returns and {-1, 0, +1} direction labels.

    A supplied group column prevents labels from crossing contiguous sessions.
    """
    horizon = _positive_int(horizon_events, name="horizon_events")
    neutral = _finite_nonnegative(neutral_bps, name="neutral_bps")
    if "midprice" not in frame.columns:
        raise ValueError("midprice column is required")
    if group_col is not None and group_col not in frame.columns:
        raise ValueError(f"group column {group_col!r} is missing")

    result = frame.copy()
    try:
        current = result["midprice"].astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("midprice values must be numeric") from exc
    observed = current.dropna().to_numpy(dtype=float)
    if not np.isfinite(observed).all() or (observed <= 0).any():
        raise ValueError("midprice values must be finite and positive when present")

    if group_col is None:
        future = current.shift(-horizon)
    else:
        groups = result[group_col]
        _validate_session_groups(groups, name=group_col)
        future = current.groupby(groups, sort=False).shift(-horizon)

    returns_bps = (future / current - 1.0) * 10_000.0
    result[f"forward_return_{horizon}e_bps"] = returns_bps
    label = np.select(
        [returns_bps > neutral, returns_bps < -neutral],
        [1, -1],
        default=0,
    ).astype(float)
    invalid = future.isna() | current.isna()
    label[invalid.to_numpy()] = np.nan
    result[f"direction_{horizon}e"] = label
    return result


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_nonnegative(value: float, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite non-negative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def _validate_session_groups(groups: pd.Series, *, name: str) -> None:
    if groups.isna().any():
        raise ValueError(f"group column {name!r} must not contain missing values")
    starts = groups.ne(groups.shift())
    if groups.loc[starts].duplicated().any():
        raise ValueError(f"group column {name!r} must form contiguous session blocks")
