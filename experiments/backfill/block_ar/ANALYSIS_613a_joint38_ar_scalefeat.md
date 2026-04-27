# 613a joint38 AR scale-prefix feature result

## Context

612a showed that naive learned conditional source scale is not a principled risk-width layer: the source-scale head collapsed to the lower clamp and did not learn state-responsive uncertainty. The next clean hypothesis was data-framing rather than another trainable width knob.

613a keeps the same native joint38 AR transition-flow model and fixed source distribution, but changes the memory input from `basic` features to `scale` features:

- basic: `score`, `delta`;
- scale: `score`, `delta`, `abs(delta)`, `delta^2`.

This is a generic financial time-series feature framing. It is not IV-specific, not factor-specific, and does not add a separate risk path.

## Run

Training:

- state scope: `joint38`;
- epochs: `8`;
- recent train windows: `2048`;
- memory/token dim: `128`;
- memory/token layers: `3`;
- flow steps: `16`;
- `prefix_feature_mode=scale`;
- conditional source scale disabled;
- seed: `613`.

Training result:

- best epoch: `2`;
- best validation loss: `0.463413`;
- final validation loss: `0.522760`;
- finite sample rate: `1.0`.

The early best epoch and worse validation loss already indicated that the extra magnitude channels did not make the transition-flow objective easier to learn.

## Full Suite Result

Official IV bridge on 441 windows / 48 samples:

- score: `3/11`;
- passed: surface, block-AR, cross-cell correlation;
- failed: coverage, conditionality, time-series properties, cointegration, regime coverage, distributional fidelity, mean reversion, pathwise jump realism.

Key metrics:

- cov90 overall: `74.9%`;
- h1/h7/h14/h30 cov90: `80.1% / 77.6% / 74.5% / 71.6%`;
- conditional MAE reduction: `8.0%`, but worst-cell MAE reduction `-17.5%`;
- turbulent/calm width ratio: `0.943`;
- daily-change KS cells: `22/25`;
- level KS cells: `4/25`;
- median-bias cells: `17/25`;
- bad coverage windows: `29/441 = 6.6%`;
- persistent severe undercoverage: `1408/11025 = 12.8%`;
- regime layer2: `0/8`;
- ACF correlation: `0.983`;
- kurtosis ratio: `0.747`;
- cointegration gen/GT: `0.673`, but worst-cell ratio `0.224`;
- cross-cell corr/rank: `0.604 / 2.348`;
- mean-reversion ratio: `1.092`, active pass `10/12`, but full-horizon mean-reversion suite failed;
- pathwise max-jump KS: `0.413`, but per-cell q99 jump-scale cells `19/25`.

## Comparison

| run | score | cov90 | cond MAE | turb/calm | level KS | cat. undercov | kurtosis | coin worst | MR ratio | path KS | q99 cells |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 610a basic | 5/11 | 75.7% | 5.0% | 1.009 | 2/25 | 11.0% | 0.703 | 0.328 | 0.892 | 0.446 | 21/25 |
| 612a cond-scale | 6/11 | 77.4% | 11.6% | 0.910 | 13/25 | 9.3% | 0.936 | 0.358 | 0.600 | 0.638 | 22/25 |
| 613a scale-prefix | 3/11 | 74.9% | 8.0% | 0.943 | 4/25 | 12.8% | 0.747 | 0.224 | 1.092 | 0.413 | 19/25 |

## Mechanism Read

Scale-prefix features did not fix the bottleneck. They improved some local movement behavior, including h1 coverage and h1 mean reversion, and made width-vs-vol-of-vol correlations more positive at later horizons. But those gains came with broad distributional damage:

- level occupancy collapsed from `13/25` in 612a to `4/25`;
- persistent severe undercoverage worsened to `12.8%`;
- cointegration worst-cell fell below gate;
- generated cross-cell effective rank moved higher, indicating less coherent panel structure;
- tail scale became too uneven across cells;
- conditionality failed because one cell became materially worse than unconditional.

The clean read is that the model does not merely lack volatility magnitude features. The current flow-matching objective is not reliably producing calibrated conditional probability mass. Local feature additions shift trade-offs rather than resolving the regime/coverage pathology.

## Decision

Close `prefix_feature_mode=scale` as the next deployable direction.

The next principled move should be objective-level, not another feature or width knob. The evidence from 612a and 613a points at a mismatch between flow-matching MSE geometry and calibrated conditional law learning. A cleaner next paradigm is a conditional likelihood-trained transition law over the same generic state panel: fixed preprocessing, one shared memory, one shared transition model, but train by conditional density / NLL rather than velocity MSE.

## Artifacts

- `models/backfill/613a_joint38_ar_scalefeat_e8_w2048_s613/train_summary.json`
- `results/autoresearch/613a_joint38_ar_scalefeat_e8_w2048/full11.json`
- `results/autoresearch/613a_joint38_ar_scalefeat_e8_w2048/full11.md`
