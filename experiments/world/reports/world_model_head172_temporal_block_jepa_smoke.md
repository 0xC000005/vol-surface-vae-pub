# World Model HEAD172: Temporal Block JEPA Smoke

Date: 2026-05-11

## Iteration Type

`experiment`

## Objective Family

`context_to_target_jepa_temporal_diagnostic`; smoke-only canonical JEPA
diagnostic. This is not the active Part 1 reference.

## Literature Status

`canonical_jepa_temporal_holdout_diagnostic_not_active_reference`: the
context encoder sees a same-history temporal block hidden by the synthetic
mask channel; the target encoder sees the clean same-window history; the
loss aligns latent rows only. No future-window values, decoder, or raw
reconstruction target is used.

## Hypothesis

A minimal temporal holdout JEPA can reveal whether a canonical
context-to-target latent objective adds useful abstract state beyond raw
surface baselines without becoming the active pretraining route.

## Falsifier

The diagnostic remains smoke-only if latent rows show weak retrieval/rank or
if frozen raw-plus-learned probes fail to improve raw-only guardrails.

## Run

- Train windows: `128`.
- Validation windows: `64`.
- Target time rows: `640` train, `320` validation.
- Target block: last `5` history days.
- EMA decay: `0.990000`.

## Latent Metrics

- Initial validation loss: `0.522053`.
- Final validation loss: `0.256064`.
- Validation alignment MSE: `0.221000`.
- Predicted-target cosine mean: `0.639239`.
- Context-target cosine mean: `0.401764`.
- Predicted retrieval top10: `0.031250`.
- Context retrieval top10: `0.034375`.
- Context effective rank: `3.796004`.
- Target effective rank: `6.820216`.

## Frozen Probe Guardrails

| surface | raw-only MSE | learned-only MSE | raw+learned MSE | raw+learned/raw |
| --- | ---: | ---: | ---: | ---: |
| current IV state | 0.013959 | 0.020722 | 0.012292 | 0.880589 |
| future_mean_delta | 0.006626 | 0.012697 | 0.006628 | 1.000313 |
| future_range | 0.066292 | 0.057285 | 0.056569 | 0.853330 |

## Decision

- Current-state guardrail: `PASS`.
- Promotion decision: `SMOKE_ONLY_DO_NOT_PROMOTE`.
- Part B blocked: `True`.

This diagnostic does not promote Part 1. The active reference remains the
masked-multiview Barlow branch until a learned representation passes the
raw-plus-learned guardrails and downstream probe comparisons.
