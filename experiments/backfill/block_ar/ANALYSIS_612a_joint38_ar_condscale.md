# 612a joint38 AR conditional-source-scale result

## Context

610a was the cleanest native joint learned law so far: one AR transition-flow model over a 38-variable state panel, with IV and anchor factors generated through the same memory, transition, and source-randomness mechanism. It scored `5/11` on the full IV bridge but failed coverage, conditionality, regime coverage, distributional fidelity, and mean reversion.

611a showed that scalar sampling temperature is not the missing risk layer. It slightly widened intervals but did not fix sparse-window undercoverage or regime-responsive width, and it damaged cointegration or jump-tail realism.

612a therefore tested the next clean hypothesis: let the same generic AR transition model infer a positive per-variable source scale from causal memory state, and apply that scale consistently in both training and sampling.

## Change

Implemented conditional source scale in:

- `diffusion/block_ar/generic_empirical_score_transition_flow_matching.py`
- `experiments/backfill/block_ar/train_609a_unified_ar_transition_flow.py`
- `test_code/test_609a_generic_empirical_score_transition.py`

The mechanism is architecture-generic:

- no separate IV head;
- no separate anchor-factor head;
- no post-hoc deck composition;
- one memory state, one transition velocity, one source-scale head over the selected state panel.

Verification:

```bash
python -m py_compile \
  diffusion/block_ar/generic_empirical_score_transition_flow_matching.py \
  experiments/backfill/block_ar/train_609a_unified_ar_transition_flow.py \
  experiments/backfill/block_ar/evaluate_609a_unified_ar_transition_flow.py

pytest test_code/test_609a_generic_empirical_score_transition.py -q
```

Result: `3 passed`.

## Run

Training:

- state scope: `joint38`;
- epochs: `8`;
- recent train windows: `2048`;
- memory/token dim: `128`;
- memory/token layers: `3`;
- flow steps: `16`;
- conditional source scale enabled;
- source-scale clamp: `[0.25, 4.0]`;
- seed: `612`.

Training result:

- best epoch: `7`;
- best validation loss: `0.109902`;
- final validation loss: `0.115470`;
- finite sample rate: `1.0`.

Important diagnostic: the learned source scale collapsed to the lower clamp after epoch 2:

| epoch | train loss | val loss | source scale mean | source scale std |
|---:|---:|---:|---:|---:|
| 1 | 0.2474 | 0.1732 | 0.3489 | 0.0027 |
| 2 | 0.1008 | 0.1329 | 0.2500 | 0.0000 |
| 7 | 0.0689 | 0.1099 | 0.2500 | 0.0000 |
| 8 | 0.0683 | 0.1155 | 0.2500 | 0.0000 |

## Full Suite Result

Official IV bridge on 441 windows / 48 samples:

- score: `6/11`;
- passed: surface, conditionality, time-series properties, block-AR, cointegration, cross-cell correlation;
- failed: coverage, regime coverage, distributional fidelity, mean reversion, pathwise jump realism.

Key metrics:

- cov90 overall: `77.4%`;
- h1/h7/h14/h30 cov90: `76.7% / 78.0% / 77.1% / 76.0%`;
- conditional MAE reduction: `11.6%`;
- turbulent/calm width ratio: `0.910`;
- daily-change KS cells: `24/25`;
- level KS cells: `13/25`;
- median-bias cells: `18/25`;
- persistent severe undercoverage: `1026/11025 = 9.3%`;
- regime layer2: `0/8`;
- ACF correlation: `0.981`;
- kurtosis ratio: `0.936`;
- cointegration gen/GT: `0.754`, worst-cell ratio `0.358`;
- cross-cell corr/rank: `0.888 / 1.494`;
- mean-reversion ratio: `0.600`, active pass `6/12`;
- pathwise max-jump KS: `0.638`;
- per-cell q99 jump-scale cells: `22/25`.

## Mechanism Read

612a is an improvement in headline score, but not a clean deployable solution.

The source-scale head did not learn state-responsive risk width. It discovered the degenerate lower clamp. Under plain flow matching, a trainable source scale is not a likelihood-calibrated variance parameter; it can reduce the target velocity geometry rather than learn conditional uncertainty. The lower-clamp collapse explains why the validation loss improved sharply while the model still failed regime width, sparse-window coverage, and pathwise max-jump distribution.

The useful positive result is that the native joint AR route remains alive:

- conditionality now passes cleanly;
- time-series ACF and aggregate kurtosis pass;
- daily-change distribution remains strong;
- cointegration and cross-cell structure remain healthy;
- level KS improves from `2/25` in 610a to `13/25` in 612a.

The negative result is equally important:

- learned source scale, as implemented, is not a principled risk-width mechanism;
- regime width is still almost flat across calm and turbulent histories;
- persistent severe undercoverage remains too high;
- h1 mean reversion weakens;
- pathwise max-jump KS breaks despite per-cell q99 jump scale passing.

## Decision

Close naive learned source scale as a deployability layer. Do not keep stacking trainable width knobs inside the flow source distribution.

The next clean move is not another post-hoc temperature or clamp. It should expose generic volatility/magnitude information to the same AR memory without adding a separate IV/factor path or evaluator-specific correction. The existing `prefix_feature_mode=scale` option is the most principled next experiment: it keeps one shared model and one fixed source distribution, but gives the memory encoder signed deltas plus generic magnitude features (`abs(delta)`, `delta^2`) so regime-sensitive transition geometry can be learned rather than imposed.

## Artifacts

- `models/backfill/612a_joint38_ar_condscale_e8_w2048_s612/train_summary.json`
- `models/backfill/612a_joint38_ar_condscale_e8_w2048_s612/training_history.json`
- `results/autoresearch/612a_joint38_ar_condscale_e8_w2048/full11.json`
- `results/autoresearch/612a_joint38_ar_condscale_e8_w2048/full11.md`
