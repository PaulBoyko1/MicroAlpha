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
    regular = event_type != 7

    result["resting_side_signed_size"] = np.where(regular, size * direction, np.nan)

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


def add_top_of_book_ofi(frame: pd.DataFrame) -> pd.DataFrame:
    """Add Cont-style top-of-book order-flow imbalance from consecutive states."""
    required = {"bid_price_1", "bid_size_1", "ask_price_1", "ask_size_1"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required order-book columns: {sorted(missing)}")

    result = frame.copy()
    bp = result["bid_price_1"].astype(float)
    bs = result["bid_size_1"].astype(float)
    ap = result["ask_price_1"].astype(float)
    ask_size = result["ask_size_1"].astype(float)
    bp0, bs0, ap0, as0 = bp.shift(1), bs.shift(1), ap.shift(1), ask_size.shift(1)

    bid_flow = np.where(bp > bp0, bs, np.where(bp == bp0, bs - bs0, -bs0))
    ask_flow = np.where(ap < ap0, ask_size, np.where(ap == ap0, ask_size - as0, -as0))
    valid = bp.notna() & ap.notna() & bp0.notna() & ap0.notna()
    result["top_of_book_ofi"] = np.where(valid, bid_flow - ask_flow, np.nan)
    return result
