# 517a Paradigm Decision After Frontier Closure

## Context

After `516a`, the local evidence is consistent:

- `392a` / `510a` are the deployable learned frontier at `8/11`.
- `435a` proves the suite is logically satisfiable at `11/11`, but only with
  validation-future oracle centering.
- Objective, source, wrapper, calibration, marginal/copula, and compact direct
  path repairs have all failed to produce a deployable learned `11/11`.
- Observable-state linear audit found incremental signal for some per-cell level
  targets, but every audited target still had negative holdout R2.

## External Literature Check

The recent time-series literature does not support another small local knob:

- Lag-Llama frames probabilistic time-series forecasting as a foundation-model
  problem with pretrained decoder-only transformers and lag covariates:
  https://huggingface.co/papers/2310.08278
- Chronos tokenizes time series and trains transformer language-model
  architectures with cross-entropy over quantized values, relying on large
  multi-domain pretraining:
  https://huggingface.co/papers/2403.07815
- Time-MoE scales forecasting with sparse mixture-of-experts pretraining:
  https://huggingface.co/papers/2409.16040
- Moirai 2.0 moves toward a simpler decoder-only time-series foundation model
  with quantile forecasting and multi-token prediction:
  https://huggingface.co/papers/2511.11698

The common direction is data scale, pretraining, and simpler probabilistic
sequence objectives. That is closer to a new scoped compute/data program than
to another local fine-tune around `392a`.

## Decision

The current in-session local autoresearch line is exhausted as a path to a
deployable learned `11/11`.

The next honest paradigm is one of two separately scoped programs:

1. **Deployable risk system**: report base learned model metrics separately from
   a policy calibration layer. This is useful to a risk manager, but it must not
   be claimed as a learned conditional-law improvement.
2. **New learned-law program**: build/pretrain a larger probabilistic sequence
   model on broader factor panels or synthetic/market histories, then fine-tune
   to the IV surface scenario task. This aligns with the current foundation-model
   literature, but is not a quick local in-session iteration.

## In-Session Consequence

Do not run more local `392a` objective/source/wrapper experiments. They are now
brute-force research knobs, not first-principles progress.

The remaining in-session work should leave a clean handoff:

- summarize the frontier and closed branches;
- define the exact acceptance criteria for the next larger learned-law program;
- preserve `392a` / `510a` as deployable base checkpoints and `435a` as oracle
  feasibility evidence.

