"""Strict LOBSTER CSV loading and normalization.

LOBSTER message and order-book files are row-aligned: message row *i* produces
order-book state row *i*. Raw prices are integer units of 1/10,000 dollar.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

MESSAGE_COLUMNS = ["time", "event_type", "order_id", "size", "price", "direction"]
VALID_EVENT_TYPES = frozenset({1, 2, 3, 4, 5, 6, 7})
VALID_DIRECTIONS = frozenset({-1, 1})
PRICE_SCALE = 10_000.0
LOBSTER_EMPTY_ASK = 999_999_999
LOBSTER_EMPTY_BID = -999_999_999


def orderbook_columns(levels: int) -> list[str]:
    """Return canonical LOBSTER order-book column names for ``levels`` depth levels."""
    if levels < 1:
        raise ValueError("levels must be >= 1")
    columns: list[str] = []
    for level in range(1, levels + 1):
        columns.extend(
            [
                f"ask_price_{level}",
                f"ask_size_{level}",
                f"bid_price_{level}",
                f"bid_size_{level}",
            ]
        )
    return columns


def _read_exact_width(path: str | Path, width: int, *, kind: str) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None)
    if frame.empty:
        raise ValueError(f"{kind} file is empty")
    if frame.shape[1] != width:
        raise ValueError(f"expected {width} {kind} columns; received {frame.shape[1]}")
    return frame


def _require_numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        try:
            frame[column] = pd.to_numeric(frame[column], errors="raise")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"column {column!r} must be numeric") from exc
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"column {column!r} must contain only finite values")


def _require_integer_valued(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if not np.equal(values, np.floor(values)).all():
            raise ValueError(f"column {column!r} must be integer-valued")


def load_message_file(path: str | Path) -> pd.DataFrame:
    """Load a message CSV, rejecting malformed widths and coercive enum values."""
    frame = _read_exact_width(path, len(MESSAGE_COLUMNS), kind="message")
    frame.columns = MESSAGE_COLUMNS
    _require_numeric(frame, MESSAGE_COLUMNS)
    _require_integer_valued(
        frame,
        ["event_type", "order_id", "size", "price", "direction"],
    )

    if (frame["time"] < 0).any():
        raise ValueError("message timestamps must be non-negative")
    if not np.all(np.diff(frame["time"].to_numpy(dtype=float)) >= 0):
        raise ValueError("message timestamps must be monotonic non-decreasing")
    if (frame["order_id"] < 0).any():
        raise ValueError("message order IDs must be non-negative")
    if (frame["size"] < 0).any():
        raise ValueError("message sizes must be non-negative")

    event_types = set(frame["event_type"].astype(int).unique())
    invalid_types = event_types - VALID_EVENT_TYPES
    if invalid_types:
        raise ValueError(f"unsupported LOBSTER event types: {sorted(invalid_types)}")

    regular = frame.loc[frame["event_type"] != 7]
    invalid_directions = set(regular["direction"].astype(int).unique()) - VALID_DIRECTIONS
    if invalid_directions:
        raise ValueError(f"invalid order directions: {sorted(invalid_directions)}")
    if (regular["price"] <= 0).any():
        raise ValueError("non-halt message prices must be positive")

    halts = frame.loc[frame["event_type"] == 7]
    if not halts.empty:
        if not (halts["direction"] == -1).all():
            raise ValueError("LOBSTER halt messages must use direction -1")
        if not (halts[["order_id", "size"]] == 0).all().all():
            raise ValueError("LOBSTER halt messages must have zero order ID and size")
        if not halts["price"].isin([-1, 0, 1]).all():
            raise ValueError("LOBSTER halt message prices must be -1, 0, or 1")

    result = frame.copy()
    result["event_type"] = result["event_type"].astype(np.int8)
    result["order_id"] = result["order_id"].astype(np.int64)
    result["size"] = result["size"].astype(np.int64)
    result["direction"] = result["direction"].astype(np.int8)
    result["price"] = result["price"].astype(float) / PRICE_SCALE
    result.loc[result["event_type"] == 7, "price"] = np.nan
    return result


def load_orderbook_file(path: str | Path, *, levels: int) -> pd.DataFrame:
    """Load and normalize a LOBSTER order-book CSV.

    Unoccupied levels are represented by LOBSTER dummy prices and zero volume;
    their normalized price is exposed as NaN rather than a fake extreme price.
    """
    columns = orderbook_columns(levels)
    frame = _read_exact_width(path, len(columns), kind="order-book")
    frame.columns = columns
    _require_numeric(frame, columns)

    size_columns = [c for c in columns if "_size_" in c]
    price_columns = [c for c in columns if "_price_" in c]
    _require_integer_valued(frame, size_columns + price_columns)
    if (frame[size_columns] < 0).any().any():
        raise ValueError("order-book sizes must be non-negative")

    result = frame.copy()
    for level in range(1, levels + 1):
        for side, dummy in (("ask", LOBSTER_EMPTY_ASK), ("bid", LOBSTER_EMPTY_BID)):
            price_col = f"{side}_price_{level}"
            size_col = f"{side}_size_{level}"
            raw_price = result[price_col].astype(float)
            has_dummy_price = raw_price == dummy
            has_zero_size = result[size_col] == 0
            if (has_dummy_price != has_zero_size).any():
                raise ValueError(
                    f"unoccupied {side} levels must pair the dummy price with zero size"
                )
            empty = has_dummy_price
            invalid_occupied = (~empty) & (raw_price <= 0)
            if invalid_occupied.any():
                raise ValueError(f"occupied {side} prices must be positive")
            result[price_col] = raw_price / PRICE_SCALE
            result.loc[empty, price_col] = np.nan
            result[size_col] = result[size_col].astype(np.int64)

    _validate_level_ordering(result, levels)
    return result


def _validate_level_ordering(frame: pd.DataFrame, levels: int) -> None:
    best_ask = frame["ask_price_1"]
    best_bid = frame["bid_price_1"]
    crossed = best_ask.notna() & best_bid.notna() & (best_ask <= best_bid)
    if crossed.any():
        raise ValueError("best ask must be greater than best bid")

    for level in range(2, levels + 1):
        ask_prev = frame[f"ask_price_{level - 1}"]
        ask = frame[f"ask_price_{level}"]
        bid_prev = frame[f"bid_price_{level - 1}"]
        bid = frame[f"bid_price_{level}"]
        ask_gap = ask.notna() & ask_prev.isna()
        bid_gap = bid.notna() & bid_prev.isna()
        ask_bad = ask.notna() & ask_prev.notna() & (ask <= ask_prev)
        bid_bad = bid.notna() & bid_prev.notna() & (bid >= bid_prev)
        if ask_gap.any():
            raise ValueError(f"ask levels must be contiguous from level 1 (level {level})")
        if bid_gap.any():
            raise ValueError(f"bid levels must be contiguous from level 1 (level {level})")
        if ask_bad.any():
            raise ValueError(f"ask prices must increase with depth (level {level})")
        if bid_bad.any():
            raise ValueError(f"bid prices must decrease with depth (level {level})")


def load_lobster_pair(
    message_path: str | Path,
    orderbook_path: str | Path,
    *,
    levels: int,
) -> pd.DataFrame:
    """Load and row-align a message/order-book pair into one chronological frame."""
    messages = load_message_file(message_path)
    book = load_orderbook_file(orderbook_path, levels=levels)
    if len(messages) != len(book):
        raise ValueError(
            "message and order-book files must have identical row counts; "
            f"received {len(messages)} and {len(book)}"
        )
    return pd.concat([messages.reset_index(drop=True), book.reset_index(drop=True)], axis=1)
