from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from microalpha.data.lobster import load_lobster_pair, load_message_file, load_orderbook_file


def _write(path: Path, rows: list[list[float]]) -> None:
    pd.DataFrame(rows).to_csv(path, header=False, index=False)


def test_pair_loader_scales_prices_and_aligns_rows(tmp_path: Path) -> None:
    message, book = tmp_path / "message.csv", tmp_path / "book.csv"
    _write(message, [[34200.1, 1, 101, 100, 1000000, 1], [34200.2, 4, 102, 20, 1000200, -1]])
    _write(book, [[1000200, 200, 1000000, 100], [1000200, 180, 1000000, 100]])
    frame = load_lobster_pair(message, book, levels=1)
    assert frame.loc[0, "price"] == pytest.approx(100.0)
    assert frame.loc[0, "ask_price_1"] == pytest.approx(100.02)
    assert len(frame) == 2


def test_wrong_width_is_rejected_before_pandas_can_reindex(tmp_path: Path) -> None:
    book = tmp_path / "book.csv"
    _write(book, [[1000200, 200, 1000000, 100, 999, 1, 998, 1]])
    with pytest.raises(ValueError, match="expected 4 order-book columns"):
        load_orderbook_file(book, levels=1)


def test_fractional_enum_is_rejected(tmp_path: Path) -> None:
    message = tmp_path / "message.csv"
    _write(message, [[1.0, 1.9, 1, 10, 1000000, 1]])
    with pytest.raises(ValueError, match="event_type.*integer-valued"):
        load_message_file(message)


def test_halt_price_is_nan_and_direction_minus_one_allowed(tmp_path: Path) -> None:
    message = tmp_path / "message.csv"
    _write(message, [[1.0, 7, 0, 0, -1, -1]])
    frame = load_message_file(message)
    assert np.isnan(frame.loc[0, "price"])


def test_dummy_unoccupied_levels_become_nan(tmp_path: Path) -> None:
    book = tmp_path / "book.csv"
    _write(book, [[999999999, 0, -999999999, 0]])
    frame = load_orderbook_file(book, levels=1)
    assert np.isnan(frame.loc[0, "ask_price_1"])
    assert np.isnan(frame.loc[0, "bid_price_1"])


@pytest.mark.parametrize(
    ("row", "match"),
    [
        ([1.0, 1, 1, 10, 1000000.5, 1], "price.*integer-valued"),
        ([1.0, 1, 1, 10, 0, 1], "non-halt message prices"),
        ([1.0, 7, 1, 0, -1, -1], "zero order ID and size"),
        ([1.0, 7, 0, 0, 2, -1], "prices must be -1, 0, or 1"),
    ],
)
def test_message_loader_rejects_noncanonical_event_fields(
    tmp_path: Path,
    row: list[float],
    match: str,
) -> None:
    message = tmp_path / "message.csv"
    _write(message, [row])
    with pytest.raises(ValueError, match=match):
        load_message_file(message)


def test_empty_levels_require_matching_dummy_price_and_zero_size(tmp_path: Path) -> None:
    book = tmp_path / "book.csv"
    _write(book, [[999999999, 10, 1000000, 100]])
    with pytest.raises(ValueError, match="dummy price with zero size"):
        load_orderbook_file(book, levels=1)


def test_order_book_rejects_crossed_best_quotes(tmp_path: Path) -> None:
    book = tmp_path / "book.csv"
    _write(book, [[1000000, 100, 1000000, 100]])
    with pytest.raises(ValueError, match="best ask must be greater"):
        load_orderbook_file(book, levels=1)


def test_order_book_rejects_gapped_depth(tmp_path: Path) -> None:
    book = tmp_path / "book.csv"
    _write(book, [[999999999, 0, -999999999, 0, 1000200, 100, 1000000, 100]])
    with pytest.raises(ValueError, match="ask levels must be contiguous"):
        load_orderbook_file(book, levels=2)
