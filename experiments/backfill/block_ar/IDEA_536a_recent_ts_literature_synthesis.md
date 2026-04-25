# 536a Recent Time-Series Literature Synthesis

## Context
535a closed the small global-Gaussian panel-law prototype. The current learned frontier remains `392a`/`510a` at `8/11`; the new panel Gaussian model scored `3/11`, so simply adding factor tokens to a weak one-shot density is not enough.

The user asked to explicitly check recent ICLR/AAAI/NeurIPS-style time-series work before continuing, especially:

- GCGNet
- EMAformer
- ShifTS
- APN
- FreqCycle
- DLinear/NLinear
- M2FMoE

## Primary Source Notes
GCGNet (`arXiv:2603.08032`, ICLR 2026) targets forecasting with exogenous variables. Its key transferable claim is that temporal dependence and channel/exogenous dependence should not be modeled as two disconnected stages. It uses a variational generator, graph-structure alignment, and a graph refiner to keep generated forecasts consistent with robust time/channel correlation graphs.

EMAformer (`arXiv:2511.08396`, AAAI 2026) argues that iTransformer-style multivariate forecasting suffers from unstable inter-channel relationships. Its transferable idea is not the exact Transformer wrapper, but the need for stable channel identity, phase sensitivity, and cross-axis specificity before attention/mixing.

ShifTS (OpenReview ICLR 2026) separates temporal shift from concept drift and handles temporal shift before concept drift. This is relevant because many validation failures are not pure one-step unpredictability; they are level-occupancy and regime allocation failures under chronological distribution shift.

APN (`arXiv:2505.11250`, AAAI 2026) shows that simple adaptive patch aggregation can beat more complex irregular-time models. The transferable lesson is efficiency and adaptive summarization, not irregular-time machinery, because the local IV/factor panel is regular daily data.

FreqCycle (`arXiv:2603.09661`, AAAI 2026) argues that mid/high-frequency components matter, not only low-frequency trend/cycle components. This maps to our repeated pathwise jump and daily-change profile failures in non-frontier models.

M2FMoE (`arXiv:2601.08631`, AAAI 2026) targets extreme events by multi-resolution and multi-view frequency mixture-of-experts. The transferable idea is adaptive regular-vs-extreme feature allocation without event labels.

DLinear/NLinear (`arXiv:2205.13504`, AAAI 2023) remains an important control: simple last-value normalization and decomposition can outperform complex Transformers on forecasting. This argues against a default large Transformer reset unless the model also fixes probabilistic dependence.

## What Not To Import
Do not import these papers literally as a stack of modules:

- graph aligner + embedding armor + adaptive patches + frequency MoE + shift module would be an unjustifiable architecture pile;
- most named papers optimize deterministic MSE/MAE forecasting, not deployable conditional scenario generation;
- our acceptance suite cares about calibrated path distributions, marginal level occupancy, regime coverage, and conditionality, not point forecast MSE.

## Transferable Principle
The common useful direction is:

```text
learn conditional dependence with stable channel identities and shift-aware normalization,
while preserving temporal order and local high-frequency path realism.
```

For this repository, the clean next prototype should therefore be autoregressive rather than one-shot:

```text
p(panel_{t+1} | history/prefix) = conditional daily panel density
```

with:

- NLinear-style last-value/score anchoring through current panel state;
- EMAformer-style channel identity embeddings;
- GCGNet-style learned channel interaction inside the daily density, not a fixed global covariance;
- optional frequency/history summaries only as generic input features, not a separate frequency MoE branch;
- exact likelihood or vanilla flow matching as the generative core.

## Next Falsifier
Implement a small autoregressive panel transition density/flow over the aligned 51-variable panel:

- history and generated prefix are encoded causally;
- the target is next-day panel transition in empirical normal-score coordinates;
- the daily innovation law has learned conditional cross-channel dependence;
- sampling rolls forward 30 days and returns the IV subpanel to the unchanged 11-suite.

This is the non-redundant continuation of 535a: it keeps the multi-factor panel-law framing but replaces the failed one-shot global Gaussian residual with learned conditional dependence over daily transitions.

Success criterion: it must at least approach or exceed the old transition-likelihood branch (`348a`, `5/11`) and show a plausible path toward `392a` structure. If it cannot clear that, the local panel-law route is data/capacity limited and should not be patched with more deterministic-forecasting modules.
