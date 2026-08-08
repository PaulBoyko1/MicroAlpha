"""Deterministic, causal order-book features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_book_features(frame: pd.DataFrame, *, levels: int = 1) -> pd.DataFrame:
    """Add features using only the current and earlier rows."""
    if levels < 1:
        raise ValueError("levels must be >= 1")
    required = {"ask_price_1", "ask_size_1", "bid_price_1", "bid_size_1"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"missing required order-book columns: {sorted(missing)}")

    result = frame.copy()
    ask = result["ask_price_1"].astype(float)
    bid = result["bid_price_1"].astype(float)
    ask_size = result["ask_size_1"].astype(float)
    bid_size = result["bid_size_1"].astype(float)
    two_sided = ask.notna() & bid.notna()

    result["midprice"] = np.where(two_sided, (ask + bid) / 2.0, np.nan)
    result["spread"] = np.where(two_sided, ask - bid, np.nan)
    result["is_locked_or_crossed"] = np.where(two_sided, ask <= bid, False)
    result["spread_bps"] = np.where(
        result["midprice"] > 0,
        result["spread"] / result["midprice"] * 10_000.0,
        np.nan,
    )

    top_depth = bid_size + ask_size
    valid_top = two_sided & (top_depth > 0)
    result["queue_imbalance"] = np.where(
        valid_top,
        (bid_size - ask_size) / top_depth,
        np.nan,
    )
    result["microprice"] = np.where(
        valid_top,
        (ask * bid_size + bid * ask_size) / top_depth,
        np.nan,
    )
    result["microprice_edge_bps"] = np.where(
        result["midprice"] > 0,
        (result["microprice"] - result["midprice"]) / result["midprice"] * 10_000.0,
        np.nan,
    )

    bid_depth = _sum_levels(result, "bid_size", levels)
    ask_depth = _sum_levels(result, "ask_size", levels)
    total_depth = bid_depth + ask_depth
    result[f"depth_imbalance_{levels}"] = np.where(
        total_depth > 0,
        (bid_depth - ask_depth) / total_depth,
        np.nan,
    )
    return result


def _sum_levels(frame: pd.DataFrame, prefix: str, levels: int) -> pd.Series:
    columns = [f"{prefix}_{level}" for level in range(1, levels + 1)]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"requested depth level is unavailable: {missing[0]}")
    return frame[columns].astype(float).sum(axis=1)
