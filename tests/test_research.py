import numpy as np
import pandas as pd
import pytest

from microalpha.research.experiment import evaluate_logistic_baseline
from microalpha.research.labels import add_forward_midprice_labels
from microalpha.research.splits import WalkForwardSplit, expanding_walk_forward


def test_grouped_forward_label_does_not_cross_session() -> None:
    frame = pd.DataFrame({"midprice": [100.0, 101.0, 200.0, 201.0], "session": [1, 1, 2, 2]})
    result = add_forward_midprice_labels(frame, horizon_events=1, group_col="session")
    assert np.isnan(result.loc[1, "direction_1e"])
    assert result.loc[2, "direction_1e"] == 1


@pytest.mark.parametrize("midprice", [[0.0, 100.0], [100.0, np.inf]])
def test_forward_labels_reject_invalid_midprices(midprice: list[float]) -> None:
    frame = pd.DataFrame({"midprice": midprice})
    with pytest.raises(ValueError, match="finite and positive"):
        add_forward_midprice_labels(frame, horizon_events=1)


def test_forward_labels_reject_nonfinite_neutral_threshold() -> None:
    frame = pd.DataFrame({"midprice": [100.0, 101.0]})
    with pytest.raises(ValueError, match="finite non-negative"):
        add_forward_midprice_labels(frame, horizon_events=1, neutral_bps=float("nan"))


def test_grouped_forward_labels_require_contiguous_sessions() -> None:
    frame = pd.DataFrame({"midprice": [100.0, 101.0, 200.0], "session": [1, 2, 1]})
    with pytest.raises(ValueError, match="contiguous session blocks"):
        add_forward_midprice_labels(frame, horizon_events=1, group_col="session")


def test_purge_must_cover_label_horizon() -> None:
    with pytest.raises(ValueError, match="at least label_horizon"):
        expanding_walk_forward(
            100, min_train_size=40, test_size=10, purge_size=4, label_horizon=5
        )


def test_walk_forward_rejects_overlapping_or_empty_test_windows() -> None:
    with pytest.raises(ValueError, match="at least test_size"):
        expanding_walk_forward(
            100,
            min_train_size=40,
            test_size=10,
            purge_size=10,
            label_horizon=10,
            step_size=5,
        )
    with pytest.raises(ValueError, match="does not produce"):
        expanding_walk_forward(
            20,
            min_train_size=10,
            test_size=10,
            purge_size=1,
            label_horizon=1,
        )


def test_walk_forward_is_ordered_and_purged() -> None:
    splits = expanding_walk_forward(
        30, min_train_size=10, test_size=5, purge_size=2, label_horizon=2
    )
    assert len(splits) == 3
    assert splits[0].train.tolist() == list(range(10))
    assert splits[0].test.tolist() == list(range(12, 17))


def test_logistic_baseline_returns_fold_metrics() -> None:
    x = np.linspace(-2.0, 2.0, 80)
    frame = pd.DataFrame({"x": x, "y": np.where(x >= 0, 1, -1)})
    splits = expanding_walk_forward(
        80, min_train_size=50, test_size=10, purge_size=1, label_horizon=1
    )
    metrics = evaluate_logistic_baseline(
        frame, feature_columns=["x"], label_column="y", splits=splits
    )
    assert list(metrics.columns) == [
        "fold",
        "train_rows",
        "test_rows",
        "majority_accuracy",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
    ]
    assert len(metrics) == 2


def test_logistic_baseline_rejects_label_as_feature() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0], "y": [-1, 1]})
    with pytest.raises(ValueError, match="label column cannot be used"):
        evaluate_logistic_baseline(
            frame,
            feature_columns=["y"],
            label_column="y",
            splits=[],
        )


def test_logistic_baseline_rejects_overlapping_fold_indices() -> None:
    frame = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "y": [-1, 1, -1, 1]})
    split = WalkForwardSplit(train=np.array([0, 1, 2]), test=np.array([2, 3]))
    with pytest.raises(ValueError, match="train indices must precede"):
        evaluate_logistic_baseline(
            frame,
            feature_columns=["x"],
            label_column="y",
            splits=[split],
        )


def test_logistic_baseline_rejects_infinite_features() -> None:
    frame = pd.DataFrame(
        {
            "x": [np.inf, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "y": [-1, -1, -1, 1, 1, 1, -1, 1],
        }
    )
    split = WalkForwardSplit(train=np.arange(0, 6), test=np.arange(6, 8))
    with pytest.raises(ValueError, match="training features must be finite"):
        evaluate_logistic_baseline(
            frame,
            feature_columns=["x"],
            label_column="y",
            splits=[split],
        )
