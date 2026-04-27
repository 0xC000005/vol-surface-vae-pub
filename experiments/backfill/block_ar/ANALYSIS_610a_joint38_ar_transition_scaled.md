# 610a Scaled Joint38 AR Transition Result

## Context

609a implemented the native shared-state/shared-randomness joint AR route and
proved the mechanics, but the 2-epoch smoke was undertrained. 610a keeps the same
model family and scales only training/capacity:

- same generic empirical-score AR transition-flow model;
- same cleaned canonical 38-state panel;
- no separate IV/factor heads;
- no post-hoc IV/factor deck composition;
- no new risk adapter or calibration layer.

## Run

Training:

```bash
python experiments/backfill/block_ar/train_609a_unified_ar_transition_flow.py \
  --state_scope joint38 \
  --epochs 8 \
  --max_train_windows 2048 \
  --batch_size 32 \
  --memory_dim 128 \
  --memory_layers 3 \
  --memory_heads 4 \
  --memory_ff 256 \
  --token_dim 128 \
  --token_layers 3 \
  --token_heads 4 \
  --token_ff 256 \
  --flow_steps 16 \
  --sample_count 4 \
  --sample_steps 8 \
  --chunk_size 2 \
  --seed 610 \
  --device cuda \
  --output_dir models/backfill/610a_joint38_ar_transition_e8_w2048_s610
```

Training result:

- best epoch: `5`;
- train loss: `0.542 -> 0.254`;
- best validation loss: `0.4357`;
- final validation loss: `0.4484`;
- sample finite rate: `1.0`;
- smoke IV range: `0.0101` to `0.5116`;
- smoke factor range: `0.691` to `18152.93`.

Official bridge:

```bash
python experiments/backfill/block_ar/evaluate_609a_unified_ar_transition_flow.py \
  --checkpoint models/backfill/610a_joint38_ar_transition_e8_w2048_s610/best_model.pt \
  --state_scope joint38 \
  --max_windows 441 \
  --samples 48 \
  --n_steps 30 \
  --batch_size 32 \
  --chunk_size 8 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --seed 6101 \
  --device cuda \
  --output_json results/autoresearch/610a_joint38_ar_transition_e8_w2048/full11.json \
  --output_md results/autoresearch/610a_joint38_ar_transition_e8_w2048/full11.md
```

## Result

Score: `5/11`.

Passed:

- surface;
- block-AR;
- cointegration;
- cross-cell correlation;
- pathwise jump realism.

Failed:

- coverage;
- conditionality;
- time_series;
- regime_coverage;
- distributional_fidelity;
- mean_reversion.

Key metrics:

- cov90 overall: `75.7%`;
- h1/h7/h14/h30 cov90: `80.3% / 77.3% / 76.0% / 72.6%`;
- conditional MAE reduction: borderline `5.0%`, but per-cell conditionality fails;
- turbulent/calm width ratio: `1.009`, far below risk-policy target `1.15`;
- daily-change KS: `24/25`;
- level KS: `2/25`;
- median-bias cells: `11/25`;
- bad coverage windows: `36/441 = 8.2%`;
- persistent severe undercoverage: `1215/11025 = 11.0%`;
- cointegration gen/GT: `0.779`, worst-cell ratio `0.328`;
- cross-cell corr/rank: `0.880 / 1.492`;
- aggregate mean-reversion ratio: `0.892`;
- full-horizon aggregate mean reversion: pass;
- full-horizon active mean pass: `62.8%`, below gate;
- pathwise max-jump KS: `0.446`;
- per-cell q99 jump-scale cells: `21/25`.

## Mechanism Read

Scaling the exact hybrid helped materially:

- score improved from smoke `3/11` to `5/11`;
- cointegration recovered;
- pathwise max-jump realism recovered;
- cross-cell structure stayed healthy;
- daily-change distribution became strong at `24/25`.

The remaining issue is not surface validity, accounting consistency, or shared
state. The model is now a legitimate native joint law prototype. The bottleneck
is allocation of probability mass across future levels and regimes:

- coverage is too low in sparse/high-move windows;
- regime width does not expand enough in turbulent histories;
- level occupancy is poor despite good daily-change shape;
- median direction is biased in many cells;
- mean reversion is structurally present but too uneven across horizons/cells.

This is the same broad-frame bottleneck seen in IV-only learned laws, now
reproduced inside a cleaner joint model.

## Decision

Keep the 609/610 native joint AR family alive. It is not yet risk-manager
deployable, but it is the first scientifically clean route that:

- supports IV-only and IV+anchor factors with the same architecture;
- avoids post-hoc scenario gluing;
- recovers several structural suites after scaling.

The next controlled diagnostic should not add architecture. It should test whether
the remaining failure is a sampling-scale/risk-width issue by evaluating generic
sample temperature on the same checkpoint. If moderate temperature improves
coverage/regime inclusion without breaking daily-change, cointegration,
cross-cell, and pathwise structure, it can become a disclosed risk-policy layer.
If it breaks structure, the next model-side change must learn regime-responsive
source scale inside the AR transition rather than applying a global scale.
