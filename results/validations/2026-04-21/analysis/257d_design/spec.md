# 257d Design Spec

## Context

`257c` showed that the latent-token VAE family responds meaningfully to a
multi-sample scenario objective:

- sample identity matters more
- coverage and calibration improve
- short-run change-law metrics improve

But the suite score stayed at `2/11` because long-run structure remained weak:

- rank ratio still failed
- cointegration regressed
- level KS remained poor

## Hypothesis

The remaining `257` bottleneck is that `257c` adds local sample spread without
explicitly preserving the ensemble mean's long-run cross-cell structure.

So `257d` should keep the `257c` architecture and multi-sample objective, but add
generic structure-preserving anchors on the **ensemble mean** of posterior samples.

## Architecture

- identical model architecture to `257c`
- same latent-token VAE
- same posterior/prior
- same multi-sample objective

No model-class changes. This is an **objective-only** follow-up.

## New Losses

Apply the following to the ensemble mean of posterior samples:

### 1. Change correlation anchor

Compute per-window temporal covariance over cells for generated mean changes and
ground-truth future changes. Penalize correlation-matrix mismatch.

### 2. Change spectrum anchor

Penalize mismatch in the normalized top eigenvalue spectrum of the generated vs
ground-truth change covariance.

### 3. Level correlation anchor

Do the same for future levels.

### 4. Level spectrum anchor

Do the same for the level covariance spectrum.

These are generic `(T,D)` structure losses and do not depend on IV-specific bases or
cointegration formulas.

## Initial Weights

- `lambda_change_corr = 0.10`
- `lambda_change_spec = 0.05`
- `lambda_level_corr = 0.05`
- `lambda_level_spec = 0.025`
- `structure_topk = 8`

Keep all `257c` losses unchanged otherwise.

## Pre-Registered Kill Criteria

`257d` is meaningful only if it preserves most of `257c`'s stochastic gains while
improving at least one long-run structure metric:

- `rank_ratio`
- `cointegration_ratio`
- `level KS pass`

Hard failure signals:

- coverage falls back materially toward `257b`
- sample-identity gains disappear
- long-run structure still does not improve materially

## Decision Rule

- If `257d` improves long-run structure without giving back most of `257c`'s sample
  gains, stay in the `257` family.
- If `257d` fails, stop objective-stacking in `257` and escalate to `258a`.
