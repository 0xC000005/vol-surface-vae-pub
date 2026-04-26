# 541a Local Data Expansion Audit

## Context
540a closed the strongest local synthesis of the frontier empirical-score transition flow, factor conditioning, and patch-energy proper scoring below the `8/11` frontier. The next clean route from 539a/540a is not another local fine-tune; it is a larger data/pretraining program or frontier/deployability closure.

This audit checks whether that larger program is executable using data already present in the workspace.

## Local Data Inventory
Primary usable data remains:

- `data/vol_surface_with_ret.npz`: one SPX IV surface panel, shape `(5822, 5, 5)`, plus SPX return/price/slope/skew/level summaries;
- `data/spx_vol_surface_history_full_data_fixed.parquet`: SPX IV dates, `2000-01-03` to `2023-02-24`;
- `data/multi_factor_data.npz`: 13 factor levels and 13 returns/diffs, shape `(5825, 13)` each;
- `data/multi_factor_levels.parquet` and `data/multi_factor_returns.parquet`;
- `data/market_data_ohlcv.csv`: AMZN, MSFT, and S&P 500 OHLCV columns, shape `(5825, 19)`;
- `data/stock_returns_multifeature.npz`: AMZN/MSFT/SP500 return-style features, shape `(5805, 12)`;
- `data/synthetic_rates_D5.npz`: synthetic 5x1 rates-like surface, not an additional real option surface.

Local generated/cached artifacts include:

- `data/percell_scale_cache/`;
- `data/uncertainty_cache/`;
- old results/test outputs under `results/`, `models/`, and `test_spx/`.

These are not new independent training universes.

## Finding
There is no hidden multi-underlying option-surface dataset locally.

The workspace supports:

- one real SPX option-surface history;
- a modest aligned macro/market factor panel;
- a few equity OHLCV/return side features;
- synthetic rates data that does not share the IV-surface target distribution.

It does not support a true scale-based pretraining breakthrough over many option surfaces, names, regions, maturities, or asset classes.

## Mechanism Read
The local failure pattern is now consistent across several families:

- daily-change modeling can become excellent while future level occupancy stays wrong;
- adding the local factor panel can slightly improve level occupancy but not enough;
- horizon-aware patch scoring preserves path geometry but does not break the level/regime bottleneck;
- exact likelihood panel models are far below the `392a`/`510a` structural frontier.

This suggests the missing ingredient is not another local architecture knob. It is either:

- more independent conditional examples, especially more option-surface universes and regimes;
- a different deployability framing that reports the learned base model separately from a policy-calibrated risk layer;
- or accepting the current learned frontier as the honest local result.

## Decision
A true Bitter-Lesson data-scale move is not executable with the current local files alone. Continuing local model search without new data is likely to be brute-force knob search.

Next principled step:

- if external data acquisition is allowed, define and implement a data ingestion/pretraining scaffold for multiple option-surface universes;
- otherwise write a frontier/deployability closure around the `392a`/`510a` `8/11` learned frontier, the `435a` oracle feasibility result, and the failed local routes through 540a.

