# 641a Mixed-Coordinate Native Joint Flow

## Hypothesis

641a tests whether the coordinate evidence from 631a and 638a can be unified without post-hoc gluing:

- `iv:*` channels generate empirical level-score deltas to keep IV surfaces support-valid.
- `factor:*` channels generate encoded increments to preserve anchor-factor move realism.
- All 38 channels share one causal memory, one velocity network, one rollout, and one noise path.

This is a preprocessing/generation-coordinate policy, not a separate IV model plus factor deck.

## Result

Artifacts:

- checkpoint: `models/backfill/641a_joint38_mixedcoord_scale_e8_w2048_s641/best_model.pt`
- IV suite: `results/autoresearch/641a_joint38_mixedcoord_scale_e8_w2048_s641/full11.json`
- joint audit: `results/autoresearch/641a_joint38_mixedcoord_scale_e8_w2048_s641/joint_panel.json`

IV 11-suite: `4/11`.

Passed:

- surface validity
- time-series properties
- block-AR smoothness
- cross-cell correlation

Failed:

- coverage
- conditionality
- cointegration
- regime coverage
- distributional fidelity
- mean reversion full-horizon profile
- pathwise max-jump KS

Selected IV metrics:

- explosion rate: `0.0%`
- 90% coverage: `64.7%`
- conditional MAE reduction: `4.5%`
- turbulent/calm width ratio: `0.997`
- daily-change KS pass: `24/25`
- level KS pass: `4/25`
- median-bias pass: `8/25`
- q99 tail pass: `23/25`
- cross-cell corr ratio: `0.949`
- aggregate mean-reversion ratio: `0.919`
- pathwise max-jump KS: `0.610`

Native joint audit:

- finite rate: `1.000`
- factor delta KS mean: `0.098`
- factor delta KS pass `<0.20`: `12/13`
- factor q99 abs-delta pass `[0.5,2.0]`: `13/13`
- factor-factor corr upper-triangle corr: `0.866`
- IV-factor corr matrix corr: `0.852`

## Mechanism Read

The mixed-coordinate policy worked for the anchor-factor side. Compared with 638a, it restores factor move scale and dependence while keeping a single native 38-channel scenario path.

The IV side did not improve beyond the 4/11 frontier for this family. The daily-change law is mostly realistic, but the generated IV levels are misplaced and too narrow over the full horizon:

- coverage is low and worsens with horizon
- level KS is poor despite good daily-change KS
- median placement is biased in many cells
- turbulent/calm width remains almost flat
- pathwise jump scale is acceptable, but max-jump distribution shape is still off

The key bottleneck is therefore not native jointness anymore. It is conditional location/spread calibration inside the unified model.

## Decision

Keep 641a as the clean native joint baseline for one-model IV plus anchor-factor generation. It is more scientifically satisfying than deck composition because per-scenario IV and factor paths share state, memory, and randomness.

Do not add more coordinate branches. The next principled step should target the unified model objective or sampling law so that conditional spread and level placement improve without separating IV and anchor factors again.
