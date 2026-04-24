# 412a Direct Future-Score Path Flow

## Context

The active frontier remains `392a` at `8/11`.

Closed primary routes:

- post-hoc affine/quantile/interval calibration;
- scalar temperature or mixture calibration;
- naive conditional noise scaling;
- simple free-running proper-score fine-tune losses on top of 392a.

The persistent residual failure is not local path mechanics:

- daily-change KS is usually `25/25`;
- conditionality can pass;
- time-series, cross-cell, mean reversion, and pathwise jump suites can pass together.

The residual is long-horizon level/regime allocation:

- level KS stays below `15/25`;
- regime layer2 remains `0/8` or `1/8`;
- coverage failures are sliced: undercovered and overcovered cells differ by horizon/regime.

## Paradigm Shift

Train the full 30-day future path directly in the strongest current coordinate:

```text
history -> conditional flow over future normal-score path
```

This differs from old one-shot logit-path attempts (`312/324`) in the critical data framing:

- old direct path models used logit/standardized-logit coordinates and underperformed;
- the later `340/385/392` line showed empirical normal-score coordinates plus recent quantile framing are much stronger;
- therefore the clean next falsifier is not "old one-shot again", but direct full-path FM in the empirically successful score coordinate.

## Model

Minimal implementation:

- history encoder: same generic GRU/context style as existing direct path flow;
- future object: 30 x 25 normal-score tensor;
- velocity network: efficient axial mixer over horizon, cell, and channel;
- conditioning: history context plus flow time;
- target features: both future score level and implied score transition are exposed inside the velocity network;
- sampler: one-shot full future score path, decoded through the empirical quantile table to IV levels.

No center/residual split, low-rank readout, bounded idiosyncratic path, retrieval, posterior/prior scaffold, or policy calibration.

## Why This Is The Next Clean Test

The current AR transition model learns excellent one-day/local mechanics but relies on recursive rollout to produce the 30-day level law. The failed fine-tunes show that patching the rollout objective after the fact is not enough.

A direct future-score path FM makes the 30-day level occupancy the native likelihood object. If it works, it should improve level KS and coverage/regime geometry without a calibration layer. If it fails, it cleanly falsifies the hypothesis that the remaining issue is one-day rollout factorization rather than data/test difficulty.

## First Experiment

Implement `413a`:

- copy the efficient axial direct path FM structure from the `324c` family;
- replace logit coordinates with empirical normal-score coordinates from the `340/385` family;
- train on the recent-score framing used by the current frontier line;
- evaluate on the official full 11-suite.

Use a feasible small configuration first so the loop can falsify mechanics quickly before scaling.
