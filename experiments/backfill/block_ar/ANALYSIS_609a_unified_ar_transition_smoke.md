# 609a Unified 38-State AR Transition Smoke

## Context

The previous joint IV-plus-anchor product was risk-manager useful only as a
stress deck: IV paths and factor paths were generated separately and then paired
by severity rank. That preserves broad stress ordering but not per-scenario
shared randomness or a calibrated joint probability law.

609a implements the exact missing hybrid:

- one generic empirical-score AR transition-flow model;
- no IV-specific or factor-specific heads;
- `D=25` for IV-only or `D=38` for IV plus 13 anchor factor states;
- canonical 38-state preprocessing from 576/579, where factor return/diff
  channels are not duplicated future targets;
- the same model class handles IV-only and joint panels by changing only the
  selected state variables and preprocessing.

## Implementation

Added:

- `diffusion/block_ar/generic_empirical_score_transition_flow_matching.py`;
- `experiments/backfill/block_ar/train_609a_unified_ar_transition_flow.py`;
- `experiments/backfill/block_ar/evaluate_609a_unified_ar_transition_flow.py`;
- `test_code/test_609a_generic_empirical_score_transition.py`.

The model is a generic version of the 340/510 empirical-score causal-memory AR
transition flow. It learns daily score increments and rolls them forward
autoregressively. For joint mode, IV and anchor factors share the same transition
memory, same source randomness, and same velocity network.

## Verification

Compile:

```bash
python -m py_compile \
  diffusion/block_ar/generic_empirical_score_transition_flow_matching.py \
  experiments/backfill/block_ar/train_609a_unified_ar_transition_flow.py \
  experiments/backfill/block_ar/evaluate_609a_unified_ar_transition_flow.py
```

Focused tests:

```bash
pytest test_code/test_609a_generic_empirical_score_transition.py -q
```

Result: `2 passed`.

## Smoke Run

Training:

```bash
python experiments/backfill/block_ar/train_609a_unified_ar_transition_flow.py \
  --state_scope joint38 \
  --epochs 2 \
  --max_train_windows 512 \
  --batch_size 32 \
  --memory_dim 96 \
  --memory_layers 2 \
  --memory_heads 4 \
  --memory_ff 192 \
  --token_dim 96 \
  --token_layers 2 \
  --token_heads 4 \
  --token_ff 192 \
  --flow_steps 8 \
  --sample_count 4 \
  --sample_steps 8 \
  --chunk_size 2 \
  --seed 609 \
  --device cuda \
  --output_dir models/backfill/609a_joint38_ar_transition_smoke_s609
```

Result:

- train loss: `0.955 -> 0.761`;
- validation loss: `0.930 -> 0.868`;
- sample finite rate: `1.0`;
- IV sample range: `0.0121` to `0.9233`;
- factor sample range: `0.54` to `20522.76`;
- target state variables: `38`.

Official IV bridge on 128 windows / 24 samples:

- score: `3/11`;
- passed: surface validity, block-AR, cross-cell correlation;
- failed: coverage, conditionality, time_series, cointegration, regime_coverage,
  distributional_fidelity, mean_reversion, pathwise_jump_realism.

Useful diagnostics:

- cov90 overall: `83.2%`;
- h1/h7/h14/h30 coverage: `82.8% / 81.5% / 83.3% / 83.6%`;
- conditional MAE reduction: `3.1%`;
- turbulent/calm width ratio: `0.906`;
- daily-change KS: `21/25`;
- level KS: `6/25`;
- cross-cell corr/rank ratio: `0.630 / 2.965`;
- mean-reversion h30 ratio: `0.913`, but h1 aggregate ratio `0.665`;
- pathwise max-jump KS: `0.891`, despite per-cell q99 jump-scale `24/25`.

## Mechanism Read

609a answers the architectural question: the native shared-state/shared-randomness
joint AR route is mechanically viable. It eliminates the scientific objection to
the rank-matched stress deck because it generates IV and anchor factors in one
model.

The 2-epoch smoke is not quality-competitive yet. It is undertrained and lacks
regime-responsive width; turbulent windows are actually narrower than calm
windows. However, it already has sane surface validity, daily-change shape,
aggregate coverage, and cross-cell structure. The weak parts are long-horizon
conditionality, level occupancy, h1 mean reversion, cointegration weakest cell,
and pathwise max-jump distribution.

## Decision

Keep this branch alive. Do not return to post-hoc IV/factor gluing as the
scientific route.

The next iteration should run a stronger 609-family training run rather than add
new architecture knobs:

- more epochs;
- more recent training windows;
- same generic joint38 AR architecture;
- unchanged official IV bridge.

Acceptance for the next run is not immediate deployability; it must show whether
the exact hybrid can recover the structural frontier when trained beyond smoke
scale.
