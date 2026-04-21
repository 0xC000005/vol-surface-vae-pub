# 257d Ideation Memo

## Context

`257c` proved that the `257` latent-token family was partly **objective-limited**:

- sample identity now matters materially more than in `257a/257b`
- coverage, calibration, change KS, and jump KS improved

But `257c` still stayed at `2/11` because the multi-sample objective mostly
rewards local spread in level/change space. It does not explicitly preserve the
ensemble mean's long-run structure:

- cross-cell rank stays weak
- cointegration regresses
- conditionality stays flat

## Decision

Choose **`257d` next**, not `258a`.

## Why 257d Before 258a

`257c` already showed that the current latent family can respond meaningfully to a
better objective. That means the prior is still not the first bottleneck.

The most principled next test is therefore:

- keep the `257c` architecture
- keep the multi-sample scenario objective
- add **generic structure-preserving anchors** on the ensemble mean path

Only if that fails should the loop escalate to a richer latent prior family (`258a`).

## 257d Hypothesis

The remaining miss is that `257c` lets posterior samples spread locally without
constraining the long-run shared structure of the decoded path.

So `257d` should add losses on the **ensemble mean path** that preserve:

1. cross-cell correlation structure over the 30-step future
2. cross-cell spectrum / effective-rank shape over the 30-step future

Do this for both:

- future changes
- future levels

These are generic panel-time losses, not IV-specific heuristics.

## Proposed Objective

Start from `257c` and add:

### 1. Change covariance-correlation anchor

Compute per-window temporal covariance over cells on the ensemble mean change path
and on the ground-truth future change path.

Penalize mismatch in:

- normalized correlation matrix
- top eigenvalue spectrum

### 2. Level covariance-correlation anchor

Do the same on the ensemble mean level path and ground-truth future level path.

This should preserve long-run shared structure and reduce the tendency for
multi-sample spread to destroy rank / cointegration.

## Why This Is Generalizable

This is not a finance-specific cointegration formula or IV-specific basis prior.

It is a generic panel-time inductive bias:

- preserve shared correlation structure
- preserve low-rank spectrum shape

That should transfer across financial factor panels better than asset-specific
handcrafting.

## 257d-v0 Sketch

- architecture: identical to `257c`
- objective:
  - keep `ms_level_loss`
  - keep `ms_change_loss`
  - add `change_corr_loss`
  - add `change_spec_loss`
  - add `level_corr_loss`
  - add `level_spec_loss`
- apply new structure losses to the **ensemble mean** of posterior samples

## Pre-Registered Kill Criteria

`257d` is meaningful only if it preserves most of `257c`'s stochastic gains while
improving at least one long-run structure metric:

- `rank_ratio`
- `cointegration_ratio`
- `level KS pass`

Hard failure signals:

- coverage gives back most of the `257c` gain
- sample-identity gain collapses back toward `257b`
- long-run structure still does not improve materially

## Recommendation

Next decisive experiment:

- `257d`: `257c` plus ensemble-mean covariance / spectrum anchors

If `257d` fails, escalate to `258a` rather than stacking more objective terms inside
the same latent-token family.
