# 542a External Option-Surface Data Program

## Context
541a found that the local workspace does not contain a multi-underlying option-surface corpus. The local learned frontier remains `392a`/`510a` at `8/11`, while 535a-540a show that local architecture/loss changes do not break the level/regime bottleneck.

The next Bitter-Lesson-aligned move is data scale, not another local architecture knob.

## Current External Routes
Primary provider routes identified:

- OptionMetrics IvyDB US: historical US equity and index options from January 1996 onward, including implied volatilities, Greeks, underlying data, and standardized constant-maturity volatility surfaces. Source: https://optionmetrics.com/united-states/
- OptionMetrics IvyDB ETF / broader products: ETF options and global/futures option datasets for additional universes. Source: https://optionmetrics.com/data-products/
- ORATS Data API: historical end-of-day US equity options data back to 2007, 5,000+ symbols, strikes, implied monies, volatility summaries, and API access. Sources: https://orats.com/data-api and https://orats.com/docs/historical-data-api
- ThetaData: OPRA options data from June 2012 onward with trade/quote/EOD access, Greeks, and implied volatility calculations. Source: https://www.thetadata.net/options-data

## Practical Read
These sources are not interchangeable:

- OptionMetrics is the cleanest research-grade route because standardized constant-maturity volatility surfaces are already part of the product.
- ORATS is likely the quickest API route for many US tickers and implied-volatility summaries, but its proprietary surface parametrization must be mapped carefully to the repo's fixed grid.
- ThetaData is strong raw options infrastructure, but using it for this project requires a surface-construction pipeline from chains/greeks into a stable tenor/moneyness grid.

None of these data sources is currently present in the workspace, and none can be used without credentials, a subscription, or user-provided exports.

## Required Pretraining Frame
The data program should produce a canonical training table:

```text
(date, universe_id, underlying_features, factor_features, iv_surface_grid)
```

Minimum requirements:

- multiple underlyings or indices, not only SPX;
- stable tenor/moneyness or tenor/delta grid;
- no validation/test future leakage in surface construction;
- corporate-action/symbol continuity handled before model training;
- train/validation/test split by both time and held-out universes where possible;
- SPX holdout evaluation kept exactly comparable to the current 11-suite.

The modeling target should stay simple:

```text
p(future surface path | history surface path, history market/factor panel, universe identity)
```

with a vanilla probabilistic core and no evaluator-specific post-hoc correction.

## Acceptance Gate
A data-scale program is worth running only if the first imported corpus supports:

- at least dozens of liquid option-surface universes;
- at least 8-10 years of daily history per major universe;
- enough metadata to build the same grid consistently;
- a reproducible local build artifact under `data/` that can be audited.

First model acceptance:

- recover the `392a`/`510a` structural passes on SPX;
- exceed `8/11` before any policy calibration;
- improve level KS or regime layer2 without losing conditionality/cointegration.

## Decision
Do not continue local model search as if data scale has happened. The next executable step requires one of:

- user-provided OptionMetrics/ORATS/ThetaData exports;
- credentials and approval to use an external API;
- or a deliberately narrower paper framing that treats `392a`/`510a` as the honest local learned frontier.

