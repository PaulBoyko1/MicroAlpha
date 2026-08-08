# Research plan

## Phase 1 — data integrity

- Parse LOBSTER message/order-book pairs with exact-width validation.
- Preserve row alignment and chronological ordering.
- Normalize dummy/unoccupied depth to missing state instead of extreme prices.
- Make halt/cross/session policy explicit.
- Reject malformed numeric and enum fields instead of silently coercing them.

## Phase 2 — causal microstructure features

Study information available at event time only:

- spread and spread regime
- queue imbalance
- multi-level depth imbalance
- microprice displacement
- event-semantic signed liquidity flow
- aggressor execution flow
- top-of-book OFI
- rolling submission/cancel/execution intensity

## Phase 3 — labels and evaluation

- event-time horizons first, wall-clock horizons second
- labels never cross sessions
- purge >= label horizon by construction
- majority/heuristic baselines before ML
- regularized logistic regression before boosted trees
- sequence models only after simpler models establish an incremental gap

## Phase 4 — statistical credibility

- per-day and per-regime results
- bootstrap confidence intervals using dependence-aware blocks
- calibration and class-balance diagnostics
- sensitivity to neutral-zone thresholds and horizon choice
- preserve negative/null findings

## Phase 5 — execution realism

- marketable orders pay the spread
- maker/taker fee model
- configurable latency
- queue-position approximation
- partial-fill and fill-uncertainty model
- inventory/risk limits
- gross signal quality and net simulated economics reported separately

## Phase 6 — systems work

- chunked raw ingestion -> typed Parquet partitions
- C++20 event/book feature core
- Python bindings
- golden-vector equivalence tests against Python
- throughput and p50/p95/p99 latency benchmarks
