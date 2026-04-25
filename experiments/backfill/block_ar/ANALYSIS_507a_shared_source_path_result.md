# 507a Shared Source Path Result

## Context

506a selected one direct-path source-geometry falsifier: keep the empirical
normal-score full-path FM objective and axial direct-path core, but replace
fully independent source noise with a shared-factor/local-noise mixture used in
both training and sampling.

## Result

- Core change: `diffusion/block_ar/empirical_normal_score_path_flow_matching.py`
- Trainer update: `experiments/backfill/block_ar/train_339a_empirical_normal_score_path_flow.py`
- Model: `models/backfill/507a_axial_path_sourcecorr05_s42/best_model.pt`
- Full suite: `results/block_ar/507a_axial_path_sourcecorr05_s42/full11.json`
- Source correlation: `0.5`
- Score: `3/11`
- Passed: `surface`, `block_ar`, `cross_cell_correlation`
- Failed: `coverage`, `conditionality`, `time_series`, `cointegration`,
  `regime_coverage`, `distributional_fidelity`, `mean_reversion`,
  `pathwise_jump_realism`

Key metrics:

- Coverage90: `0.8162`
- Conditionality MAE reduction: `3.57%`
- Daily-change KS: `16/25`
- Level KS: `1/25`
- Median-bias fraction: `16/25`
- Bias magnitude: `22/25`
- Regime layer2: `0/8`
- Cointegration worst-cell ratio: `0.194`
- Cross-cell corr ratio: `0.719`
- Rank ratio: `2.117`
- Mean-reversion ratio: `1.205`
- Path max-jump KS: `0.467`

## Mechanism Read

The shared-source prior did exactly one thing it was supposed to do: it repaired
the gross cross-cell geometry failure from 505a. Corr ratio improved from
`0.376` to `0.719`, rank ratio improved from `3.337` to `2.117`, and path
max-jump KS improved from `0.573` to `0.467`.

But it did not recover a useful conditional future law. Level KS collapsed to
`1/25`, conditionality stayed below gate, cointegration worst cell failed, and
regime layer2 stayed `0/8`. Runtime is also poor: conditionality evaluation took
roughly `85s` per batch, making this path unattractive for a deployable system.

## Decision

Close shared-source direct-path flow as below-frontier. The experiment validates
that source geometry matters, but it also shows that direct full-path flow loses
the level/conditional tradeoff in a different way from 392a.

The next step should not be a `rho` sweep. The clean conclusion is that direct
path new-core work remains below the AR transition frontier unless the model can
inherit 392a's conditional transition structure. Return to the AR frontier for
the next ideation step.
