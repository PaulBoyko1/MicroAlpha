"""Forward-return labels for event-time research."""

from __future__ import annotations

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

    If ``group_col`` is supplied, labels never cross group/session boundaries.
    """
    if horizon_events < 1:
        raise ValueError("horizon_events must be >= 1")
    if neutral_bps < 0:
        raise ValueError("neutral_bps must be >= 0")
    if "midprice" not in frame.columns:
        raise ValueError("midprice column is required")
    if group_col is not None and group_col not in frame.columns:
        raise ValueError(f"group column {group_col!r} is missing")

    result = frame.copy()
    current = result["midprice"].astype(float)
    if group_col is None:
        future = current.shift(-horizon_events)
    else:
        future = result.groupby(group_col, sort=False)["midprice"].shift(-horizon_events)

    returns_bps = (future / current - 1.0) * 10_000.0
    result[f"forward_return_{horizon_events}e_bps"] = returns_bps
    label = np.select(
        [returns_bps > neutral_bps, returns_bps < -neutral_bps],
        [1, -1],
        default=0,
    ).astype(float)
    invalid = future.isna() | current.isna()
    label[invalid.to_numpy()] = np.nan
    result[f"direction_{horizon_events}e"] = label
    return result
