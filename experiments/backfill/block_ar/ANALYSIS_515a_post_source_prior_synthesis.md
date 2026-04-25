# 515a Post Source-Prior Synthesis

## Context

`514a` closed the TSFlow-inspired temporal source-prior route. It was a clean
source-prior experiment, but it scored `4/11` on the best checkpoint and `5/11`
on the final checkpoint, far below the `392a` / `510a` deployable frontier.

The active deployable frontier remains:

- `392a`: `8/11`, failed coverage, regime coverage, distributional fidelity.
- `510a`: `8/11`, same failed suites with slightly better aggregate coverage and
  thinner structural margins.
- `435a`: `11/11` only as a validation-future oracle, not deployable.

## What Is Now Closed

The following local routes have been tried enough to classify as capped:

- Local objective swaps around `392a`: energy, marginal CRPS, interval score,
  joint sliced-Wasserstein, patch energy.
- Source-prior changes: persistent constant source noise and AR(1)-correlated
  temporal source increments.
- Learned wrappers: checkpoint interpolation, learned-law ensembling, stress
  ensembles, source transport, residual maps, density-ratio resampling.
- Separable marginal/correlation decompositions: rank-copula conditional
  marginals and path-preserving oracle shifts.
- Compact direct path cores: MixLinear/Minkowski-style minimal direct paths and
  shared-source direct path flows.

These routes move metrics around but do not jointly fix the remaining three
failures while preserving conditionality, cointegration, and path realism.

## Mechanism Read

The recurring failure has one causal shape:

1. The learned generator has enough local geometry to pass surface validity,
   conditionality, serial structure, block-AR smoothness, correlation, mean
   reversion, and often cointegration.
2. It does not allocate future absolute levels/regimes with the same
   unconditional occupancy as realized rolling windows.
3. Attempts to force that occupancy through marginal maps, width scaling,
   source smoothing, or proper-score pressure either over-broaden the path law or
   break local jump/cointegration geometry.

That means the bottleneck is probably not another scalar hyperparameter. It is
either:

- weak identifiability from IV-history-only conditioning; or
- a need to separate learned conditional-law metrics from an auditable risk
  policy calibration layer.

## Next Principled Branch

The clean next branch is an observable-state audit before another model:

- Ask whether the failing future level/regime quantities are predicted by
  broader available market state, not just IV surface history.
- Candidate state already exists in `data/vol_surface_with_ret.npz`: `ret`,
  `price`, `slopes`, `skews`, and `levels`.
- This is first-principles aligned: a conditional scenario generator should
  condition on the available state of the market, not only the IV surface if the
  product objective is multi-factor risk.

This is different from a hand-engineered correction. The model would still learn
the law; the change is the sigma-field being conditioned on.

## Decision

Do not run another 392a-local objective, source, or ensemble knob. Run an
observable-state predictability audit next:

- target the suites that keep failing: level KS, per-regime/per-cell coverage,
  path jump realism, and worst-cell cointegration;
- test whether pre-future observable factors contain signal beyond IV-history
  summaries;
- if no signal exists, stop claiming that an IV-only learned conditional law can
  reach deployable `11/11` without policy calibration or new data.

