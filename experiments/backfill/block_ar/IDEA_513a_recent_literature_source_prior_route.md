# 513a Recent Literature Source-Prior Route

## Context

The active deployable frontier is still the `392a` / `510a` tie at `8/11`.
Both fail the same three suites: coverage, regime coverage, and distributional
fidelity. The midpoint audit in `512a` confirmed there is no simple
parameter-space bridge through this frontier.

Recent local tests already covered the most direct ICLR-style objective and
architecture transfers:

- `504a` DistDF-style joint sliced-Wasserstein fine-tuning scored `7/11`.
- `505a` minimal MixLinear/Minkowski-linear direct path scored `2/11`.
- `509a` / `510a` MMPD-style patch energy returned to `8/11` but did not break
  the level/regime bottleneck.
- `442a`, `495a`, and `496a` showed that learned-law ensembling or stress
  wrappers do not cleanly beat `392a`.

## Literature Read

Recent conference signals remain useful, but they do not justify another
arbitrary wrapper:

- DistDF, ICLR 2026, argues that MSE-style direct forecasting can be biased under
  label autocorrelation and proposes joint-distribution Wasserstein alignment:
  https://openreview.net/forum?id=VrdLwUmzBy
- MMPD, ICLR 2026, replaces plain regression losses with a patch diffusion loss
  to model multi-modal future distributions:
  https://openreview.net/forum?id=NEUgHT8dvH
- MixLinear, ICLR 2026, shows that very small structured linear backbones can be
  competitive for long-horizon forecasting:
  https://openreview.net/forum?id=QUj0KuCumD
- TSFlow, ICLR 2025, is the most relevant remaining signal: fixed white
  diffusion/flow priors are mismatched to temporal data, and GP/data-dependent
  priors can make the transport problem easier:
  https://openreview.net/forum?id=uxVBbSlKQ4

## Mechanism Read

The common frontier failure is not local surface validity or one-step geometry.
It is unconditional level/regime occupancy while preserving conditionality and
cointegration. That points to the stochastic source path, not another readout,
post-hoc calibration map, or scalar interval widening.

The current `340c`-family source prior is limited:

- `path_source_corr = 0`: i.i.d. white transition noise at every future step.
- `path_source_corr > 0`: one constant shared 25-cell noise vector plus local
  white noise.

The older `423a` persistent-source test used the second option and scored below
frontier. That does not test the TSFlow-style middle case where source
increments are temporally correlated but not constant across the whole horizon.

## Decision

Run one deployable source-prior experiment:

- keep the `392a` AR flow core and empirical normal-score coordinate system;
- add a single `path_source_ar` config field for AR(1)-correlated Gaussian
  source increments across future time;
- fine-tune from `392a` using the same recent free-running energy objective;
- evaluate with the official 11-suite.

This is a clean source-prior falsifier, not a calibration layer. If it does not
beat `8/11`, close temporal-source-prior changes unless a later theory explains
why a specific kernel family is necessary.
