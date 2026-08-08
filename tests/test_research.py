import numpy as np
import pandas as pd
import pytest

from microalpha.research.experiment import evaluate_logistic_baseline
from microalpha.research.labels import add_forward_midprice_labels
from microalpha.research.splits import expanding_walk_forward


def test_grouped_forward_label_does_not_cross_session() -> None:
    frame = pd.DataFrame({"midprice": [100.0, 101.0, 200.0, 201.0], "session": [1, 1, 2, 2]})
    result = add_forward_midprice_labels(frame, horizon_events=1, group_col="session")
    assert np.isnan(result.loc[1, "direction_1e"])
    assert result.loc[2, "direction_1e"] == 1


def test_purge_must_cover_label_horizon() -> None:
    with pytest.raises(ValueError, match="at least label_horizon"):
        expanding_walk_forward(
            100, min_train_size=40, test_size=10, purge_size=4, label_horizon=5
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
