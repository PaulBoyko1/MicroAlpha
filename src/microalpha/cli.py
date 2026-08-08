"""Command-line interface for inspection and baseline research."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from microalpha.data.lobster import load_lobster_pair
from microalpha.features.book import add_book_features
from microalpha.features.flow import add_event_flow_features, add_top_of_book_ofi
from microalpha.research.experiment import evaluate_logistic_baseline
from microalpha.research.labels import add_forward_midprice_labels
from microalpha.research.splits import expanding_walk_forward


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="microalpha")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="inspect a LOBSTER file pair")
    _add_data_args(inspect_parser)
    inspect_parser.add_argument("--rows", type=int, default=5)

    research_parser = subparsers.add_parser("research", help="run a purged logistic baseline")
    _add_data_args(research_parser)
    research_parser.add_argument("--horizon", type=int, required=True)
    research_parser.add_argument("--neutral-bps", type=float, default=0.0)
    research_parser.add_argument("--min-train", type=int, required=True)
    research_parser.add_argument("--test-size", type=int, required=True)
    research_parser.add_argument("--step-size", type=int)
    return parser


def _add_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("message", type=Path)
    parser.add_argument("orderbook", type=Path)
    parser.add_argument("--levels", type=int, default=10)


def _enrich(message: Path, orderbook: Path, levels: int) -> pd.DataFrame:
    frame = load_lobster_pair(message, orderbook, levels=levels)
    frame = add_book_features(frame, levels=levels)
    frame = add_event_flow_features(frame)
    return add_top_of_book_ofi(frame)


def main() -> None:
    args = build_parser().parse_args()
    frame = _enrich(args.message, args.orderbook, args.levels)
    if args.command == "inspect":
        columns = [
            "time",
            "event_type",
            "midprice",
            "spread_bps",
            "queue_imbalance",
            "microprice_edge_bps",
            f"depth_imbalance_{args.levels}",
            "top_of_book_ofi",
        ]
        print(frame[columns].head(args.rows).to_string(index=False))
        return

    labeled = add_forward_midprice_labels(
        frame,
        horizon_events=args.horizon,
        neutral_bps=args.neutral_bps,
    )
    splits = expanding_walk_forward(
        len(labeled),
        min_train_size=args.min_train,
        test_size=args.test_size,
        step_size=args.step_size,
        purge_size=args.horizon,
        label_horizon=args.horizon,
    )
    features = [
        "spread_bps",
        "queue_imbalance",
        "microprice_edge_bps",
        f"depth_imbalance_{args.levels}",
        "top_of_book_ofi",
        "signed_liquidity_change",
    ]
    metrics = evaluate_logistic_baseline(
        labeled,
        feature_columns=features,
        label_column=f"direction_{args.horizon}e",
        splits=splits,
    )
    print(metrics.to_string(index=False))
    print("\nmean metrics")
    print(metrics.drop(columns=["fold", "train_rows", "test_rows"]).mean().to_string())
