# 516a Observable-State Predictability Audit

## Context

`515a` identified the next first-principles question: maybe `392a`/`510a` are
capped because the failing level/regime quantities are not identifiable from IV
surface history alone. Before adding a broader conditioning branch, `516a`
audits whether observable market state adds usable out-of-sample signal.

## Method

Script: `experiments/backfill/block_ar/analyze_516a_observable_state_predictability.py`.

The audit uses only pre-validation rolling windows, index range `2410..4009`,
with a time-ordered `1200/400` train/holdout split.

Two ridge predictors are compared for each target:

- IV-only summaries from the 30-day IV history.
- IV summaries plus observable state summaries from `ret`, `price`, `slopes`,
  `skews`, and `levels`.

Targets are direct proxies for the repeatedly failing suites:

- future mean level;
- future per-cell mean and h30 level;
- future vol-of-vol;
- future max jump;
- future per-cell q90 jump scale.

## Result

| target | IV-only R2 | IV+state R2 | delta |
|---|---:|---:|---:|
| future mean level scalar | `-0.3143` | `-0.5951` | `-0.2808` |
| future mean level cells | `-0.7351` | `-0.3173` | `+0.4178` |
| future h30 level cells | `-0.9031` | `-0.5040` | `+0.3991` |
| future vol-of-vol scalar | `-0.2720` | `-0.1340` | `+0.1381` |
| future max absolute jump scalar | `-0.1783` | `-0.2020` | `-0.0237` |
| future q90 absolute jump cells | `-0.7672` | `-0.9337` | `-0.1665` |

## Mechanism Read

The observable factors add incremental linear signal for per-cell future levels
and future vol-of-vol, but the absolute holdout R2 remains negative for every
target. That is not enough evidence to justify a new conditioning branch as the
next best path to `11/11`.

This audit does not prove a nonlinear model could never extract signal, but it
does weaken the case that the current bottleneck is simply missing observable
features. A new exogenous-state neural core would be a larger compute program,
not a quick local repair.

## Decision

Do not add an exogenous-state branch as the next in-session experiment. The local
evidence now supports the product conclusion:

- `392a` / `510a` remain the deployable learned frontier at `8/11`;
- `435a` proves the suite is satisfiable only with future-oracle centering;
- reaching deployable `11/11` under this suite likely requires a separately
  reported policy calibration layer, materially more data, or a new scoped
  compute program rather than more local IV-only autoresearch knobs.

