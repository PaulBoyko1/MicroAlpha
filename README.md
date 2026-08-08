# MicroAlpha

**Leakage-aware market microstructure research from raw limit-order-book events to out-of-sample baselines.**

MicroAlpha is a research toolkit for studying whether information visible in a limit order book predicts short-horizon mid-price movement. It starts with LOBSTER-format message/order-book files, validates and normalizes the feed, builds strictly causal microstructure features, creates forward labels, and evaluates simple models with purged chronological walk-forward splits.

The goal is not to ship a retail trading bot. The goal is to make every research claim reproducible, leakage-aware, and increasingly difficult to preserve once spread, fees, latency, and fills are modeled.

## Current capabilities

- strict LOBSTER message and order-book width/type validation
- LOBSTER halt and unoccupied-depth handling
- midprice, spread, queue/depth imbalance, microprice displacement
- event-semantic signed liquidity flow and aggressor trade size
- top-of-book order-flow imbalance (OFI)
- event-horizon forward-return labels with optional session boundaries
- expanding walk-forward evaluation with enforced label purge
- majority and regularized logistic-regression baselines
- CLI commands for inspection and baseline research
- unit tests, Ruff, mypy, coverage, package-build CI

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -e '.[dev]'
pytest
```

Inspect a LOBSTER pair:

```bash
microalpha inspect data/message.csv data/orderbook.csv --levels 10
```

Run a first chronological baseline study:

```bash
microalpha research data/message.csv data/orderbook.csv \
  --levels 10 \
  --horizon 100 \
  --min-train 100000 \
  --test-size 20000 \
  --neutral-bps 0.25
```

MicroAlpha does not bundle market data. LOBSTER data must be obtained separately and used under the provider's terms.

## Research contract

1. Features may use current and past information only.
2. Forward labels are built in a separate stage.
3. The purge between train and test must be at least the label horizon.
4. Hyperparameters are selected before the final held-out period is examined.
5. Predictive metrics and economic metrics are reported separately.
6. Any PnL result must state spread, fees, latency, queue/fill assumptions, and inventory constraints.
7. Negative findings remain part of the research record.

## Architecture

```text
LOBSTER message + order book
            |
            v
 strict schema / market-state validation
            |
            v
 causal feature engineering
            |
            +------> forward labels (separate stage)
            |
            v
 purged walk-forward folds
            |
            v
 naive + statistical/ML baselines
            |
            v
 execution simulator                 [next]
            |
            v
 C++20 event engine + Python API     [planned]
```

## Roadmap

- [x] hardened LOBSTER adapter
- [x] causal top-of-book/depth features
- [x] event-semantic flow features + OFI
- [x] horizon labels + purge enforcement
- [x] majority/logistic walk-forward baseline runner
- [x] unit tests + CI
- [ ] reproducible public LOBSTER sample study with figures
- [ ] confidence intervals and bootstrap stability analysis
- [ ] wall-clock horizon labels
- [ ] regime analysis: spread / volatility / liquidity
- [ ] discrete-event execution simulator
- [ ] C++20 order-book/event feature engine
- [ ] Python bindings + throughput benchmark

## License

Code is MIT licensed. Market data is not included and remains subject to the data provider's terms.
