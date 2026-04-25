# 534a Multi-Factor Learned-Law Readiness Audit

## Context
533a shifted the only remaining principled route away from local IV-only patches and toward a broader learned financial-panel sequence law:

```text
p(future financial panel | history financial panel)
```

This audit checks whether the current repository has enough aligned data to make that route real rather than another architecture slogan.

## Local Data Inventory
Available committed/local data files include:

- `data/vol_surface_with_ret.npz`
- `data/spx_vol_surface_history_full_data_fixed.parquet`
- `data/multi_factor_data.npz`
- `data/multi_factor_levels.parquet`
- `data/multi_factor_returns.parquet`
- `data/stock_returns_multifeature.npz`
- `data/synthetic_rates_D5.npz`

Primary IV data:

- `surface`: `(5822, 5, 5)`
- `ret`: `(5822,)`
- `price`: `(5822,)`
- `slopes/skews/levels`: `(5822,)` each
- date range from parquet: `2000-01-03` to `2023-02-24`

Primary factor panel:

- factor levels: `(5825, 13)`
- factor returns/diffs: `(5825, 13)`
- columns: `spx`, `usdcad`, `usdjpy`, `dxy`, `copper`, `wheat`, `crude_oil`, `us2y`, `us10y`, `aaa_oas`, `bbb_oas`, `nikkei`, `gold`
- date range: `2000-01-03` to `2023-02-27`

Date alignment:

- IV dates: `5822`
- factor dates: `5825`
- common dates: `5822`
- IV dates missing from factor panel: `0`
- factor-only dates: `2000-09-15`, `2000-09-18`, `2023-02-27`

So the IV/factor panel is cleanly alignable for the current suite framing.

## What This Supports
The repository supports a real local prototype with roughly:

- 25 IV level tokens;
- 13 factor level tokens;
- 13 factor return/diff tokens;
- optional SPX return/price summary tokens already attached to the IV file;
- about 4,010 official training windows and 441 pre-test validation windows under the current 30/30 split.

This is enough to implement a no-lookahead multi-factor probabilistic sequence model and pipe its IV-surface conditional samples into the existing 11-suite.

## What This Does Not Support
This is not enough to honestly claim a foundation-scale Bitter Lesson program:

- only one IV surface universe is present;
- factor panel width is modest, not thousands of assets/curves/options;
- total chronological sample count is still about 5.8k daily observations;
- previous 38-dimensional CSDI-style modeling already showed that simply adding factor channels can degrade the IV scenario law if the target framing is wrong.

Therefore the next model should be framed as a feasibility prototype for a larger learned-law program, not as a full foundation model.

## Clean Next Prototype
If continuing locally, the smallest non-duplicative prototype should be:

```text
panel-token probabilistic pretraining -> IV conditional scenario query
```

Requirements:

- train on the aligned IV + factor panel with no validation-future leakage;
- model the full panel sequence, not generated factor paths bolted onto an IV-only sampler;
- use one probabilistic objective over future panel tokens;
- evaluate only the generated IV surface subpanel with the unchanged 11-suite;
- report it as a feasibility prototype, not a calibrated risk product if it fails.

## Decision
The data is sufficient for a small multi-factor panel prototype, but not sufficient for a true scale-based breakthrough. Continue only if the next implementation is a genuine panel-law prototype. Do not return to local IV-only repairs unless the user explicitly changes the objective.
