# 254b Spec

Date: 2026-04-21

## Goal

Run the decisive anti-collapse follow-up inside the `254` family after `254a-v0` failed
by collapsing into an over-shared common mode.

## Hypothesis

`254a-v0` did not fail because the dual-timescale backbone was inactive. It failed
because the deterministic common path still flowed through a nearly fixed loading map,
which let the model solve by pushing almost everything into PC1.

`254b` tests whether:

1. modest time-varying loading modulation, plus
2. explicit anti-collapse spectral regularization

can restore cross-sectional flexibility without falling back into deterministic idio
leakage.

## Architecture

Keep:

- non-AR fixed 30-step center path
- dual-timescale slow/fast temporal backbone
- bounded idio path
- bounded EC term

Change:

- add `slow_loading_delta_head`
- add `fast_loading_delta_head`
- construct:

```text
Lambda_t = Lambda_base
         + slow_loading_scale * tanh(Lambda_slow_delta_t)
         + fast_loading_scale * gate_fast_t * tanh(Lambda_fast_delta_t)
```

Then:

```text
common_t = Lambda_t @ latent_t
```

## New Structural Penalty

On the batch/time mean loading matrix:

- penalize top-1 singular-value share above `0.55`
- penalize effective rank below `3.0`

## Warm Start

Warm-start from `254a_v0` on matching tensors only. New loading-delta heads start near
zero so the model begins close to `254a` but has room to escape the collapse.

## Kill Criteria

Mechanism:

1. `loading_top1_share <= 0.60`
2. `loading_eff_rank >= 3.0`
3. `corr_ratio <= 1.6`
4. `rank_ratio >= 0.45`

Outcome:

1. recover at least `4/11`
2. `change KS > 0/25`
3. `max-jump KS < 0.94`
4. avoid large deterministic idio leakage

## Decision Rule

- If `254b` restores `4/11` and improves collapse metrics, stay in `254` once more.
- If `254b` stays below `4/11` or still collapses, treat the `254` family as likely
  capped and switch paradigm.
