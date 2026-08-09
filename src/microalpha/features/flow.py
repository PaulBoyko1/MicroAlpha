"""Causal event-flow and top-of-book order-flow imbalance features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_event_flow_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add event-semantic flow features from LOBSTER message fields.

    ``direction`` is the resting limit-order side in LOBSTER. Executions therefore
    use the opposite sign for aggressor flow.
    """
    required = {"event_type", "size", "direction"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required message columns: {sorted(missing)}")

    result = frame.copy()
    event_type = result["event_type"].astype(int)
    size = result["size"].astype(float)
    direction = result["direction"].astype(float)
    book_events = event_type.isin([1, 2, 3, 4, 5])

    result["resting_side_signed_size"] = np.where(book_events, size * direction, np.nan)

    execution = event_type.isin([4, 5])
    result["aggressor_signed_trade_size"] = np.where(
        execution,
        -size * direction,
        np.nan,
    )

    add = event_type == 1
    remove = event_type.isin([2, 3, 4, 5])
    result["signed_liquidity_change"] = np.select(
        [add, remove],
        [size * direction, -size * direction],
        default=np.nan,
    )
    return result


def add_top_of_book_ofi(
    frame: pd.DataFrame,
    *,
    group_col: str | None = None,
) -> pd.DataFrame:
    """Add Cont-style top-of-book OFI, resetting at optional session boundaries."""
    required = {"bid_price_1", "bid_size_1", "ask_price_1", "ask_size_1"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required order-book columns: {sorted(missing)}")
    if group_col is not None and group_col not in frame.columns:
        raise ValueError(f"group column {group_col!r} is missing")

    result = frame.copy()
    bp = result["bid_price_1"].astype(float)
    bs = result["bid_size_1"].astype(float)
    ap = result["ask_price_1"].astype(float)
    ask_size = result["ask_size_1"].astype(float)
    if group_col is None:
        bp0, bs0, ap0, as0 = bp.shift(1), bs.shift(1), ap.shift(1), ask_size.shift(1)
    else:
        groups = result[group_col]
        _validate_session_groups(groups, name=group_col)
        bp0 = bp.groupby(groups, sort=False).shift(1)
        bs0 = bs.groupby(groups, sort=False).shift(1)
        ap0 = ap.groupby(groups, sort=False).shift(1)
        as0 = ask_size.groupby(groups, sort=False).shift(1)

    bid_flow = np.where(bp > bp0, bs, np.where(bp == bp0, bs - bs0, -bs0))
    ask_flow = np.where(ap < ap0, ask_size, np.where(ap == ap0, ask_size - as0, -as0))
    valid = bp.notna() & ap.notna() & bp0.notna() & ap0.notna()
    result["top_of_book_ofi"] = np.where(valid, bid_flow - ask_flow, np.nan)
    return result


def _validate_session_groups(groups: pd.Series, *, name: str) -> None:
    if groups.isna().any():
        raise ValueError(f"group column {name!r} must not contain missing values")
    starts = groups.ne(groups.shift())
    if groups.loc[starts].duplicated().any():
        raise ValueError(f"group column {name!r} must form contiguous session blocks")
