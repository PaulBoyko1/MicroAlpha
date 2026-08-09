import numpy as np
import pandas as pd
import pytest

from microalpha.features.book import add_book_features
from microalpha.features.flow import add_event_flow_features, add_top_of_book_ofi


def test_top_of_book_features() -> None:
    frame = pd.DataFrame(
        {"ask_price_1": [100.02], "ask_size_1": [100], "bid_price_1": [100.0], "bid_size_1": [300]}
    )
    result = add_book_features(frame, levels=1)
    assert result.loc[0, "midprice"] == pytest.approx(100.01)
    assert result.loc[0, "queue_imbalance"] == pytest.approx(0.5)
    assert result.loc[0, "microprice"] == pytest.approx(100.015)


def test_empty_two_sided_book_is_not_encoded_as_zero_signal() -> None:
    frame = pd.DataFrame(
        {"ask_price_1": [np.nan], "ask_size_1": [0], "bid_price_1": [np.nan], "bid_size_1": [0]}
    )
    result = add_book_features(frame)
    assert np.isnan(result.loc[0, "midprice"])
    assert np.isnan(result.loc[0, "queue_imbalance"])


def test_execution_aggressor_sign_opposes_resting_side() -> None:
    frame = pd.DataFrame({"event_type": [4, 4], "size": [50, 25], "direction": [-1, 1]})
    result = add_event_flow_features(frame)
    assert result["aggressor_signed_trade_size"].tolist() == [50.0, -25.0]


def test_cross_trades_do_not_become_book_flow() -> None:
    frame = pd.DataFrame({"event_type": [6], "size": [50], "direction": [-1]})
    result = add_event_flow_features(frame)
    assert np.isnan(result.loc[0, "resting_side_signed_size"])
    assert np.isnan(result.loc[0, "aggressor_signed_trade_size"])
    assert np.isnan(result.loc[0, "signed_liquidity_change"])


def test_ofi_uses_only_current_and_previous_book_states() -> None:
    frame = pd.DataFrame(
        {
            "bid_price_1": [100.0, 100.0],
            "bid_size_1": [100, 130],
            "ask_price_1": [100.1, 100.1],
            "ask_size_1": [120, 100],
        }
    )
    result = add_top_of_book_ofi(frame)
    assert np.isnan(result.loc[0, "top_of_book_ofi"])
    assert result.loc[1, "top_of_book_ofi"] == pytest.approx(50.0)


def test_ofi_resets_at_session_boundary() -> None:
    frame = pd.DataFrame(
        {
            "session": [1, 1, 2],
            "bid_price_1": [100.0, 100.0, 100.0],
            "bid_size_1": [100, 130, 500],
            "ask_price_1": [100.1, 100.1, 100.1],
            "ask_size_1": [120, 100, 50],
        }
    )
    result = add_top_of_book_ofi(frame, group_col="session")
    assert result.loc[1, "top_of_book_ofi"] == pytest.approx(50.0)
    assert np.isnan(result.loc[2, "top_of_book_ofi"])


def test_ofi_rejects_noncontiguous_sessions() -> None:
    frame = pd.DataFrame(
        {
            "session": [1, 2, 1],
            "bid_price_1": [100.0, 100.0, 100.0],
            "bid_size_1": [100, 100, 100],
            "ask_price_1": [100.1, 100.1, 100.1],
            "ask_size_1": [100, 100, 100],
        }
    )
    with pytest.raises(ValueError, match="contiguous session blocks"):
        add_top_of_book_ofi(frame, group_col="session")
