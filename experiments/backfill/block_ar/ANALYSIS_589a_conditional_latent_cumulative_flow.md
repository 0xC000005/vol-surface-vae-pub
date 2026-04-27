# 589a Conditional-Latent Cumulative Unified Flow

## Hypothesis

588a showed that full-dimensional conditional-affine source noise gives too many
independent stochastic degrees of freedom: cross-cell correlation is too low and
effective rank is too high. A learned latent-bottleneck source should recover
joint dependence while keeping the cumulative path objective that fixed 586a's
IV-level explosions.

## Change

Extended `UnifiedIncrementFlow` with `source_mode="conditional_latent"`:

- one shared history encoder;
- one low-dimensional stochastic latent source, default `32`;
- one shared latent-to-path decoder over the full 30x38 future increment tensor;
- cumulative encoded-state flow loss;
- no empirical source bank;
- no IV/factor heads;
- no hard low-rank readout.

This uses the reset-allowed narrow stochastic bottleneck, not a hand-coded
low-rank decoder.

## Run

```bash
python experiments/backfill/block_ar/train_577a_unified_increment_flow.py \
  --clean_nonpositive_log_levels \
  --hidden_dim 384 \
  --depth 5 \
  --source_mode conditional_latent \
  --source_latent_dim 32 \
  --fm_loss_mode cumulative_state \
  --epochs 20 \
  --batch_size 64 \
  --lr 7e-4 \
  --sample_windows 128 \
  --n_samples 8 \
  --sample_steps 16 \
  --output_dir models/backfill/589a_conditional_latent_cumulative_flow_s589 \
  --seed 589 \
  --device cuda
```

Custom audit:

```bash
python experiments/backfill/block_ar/audit_583a_unified_flow_sample_quality.py \
  --checkpoint models/backfill/589a_conditional_latent_cumulative_flow_s589/best_model.pt \
  --sample_windows 441 \
  --n_samples 32 \
  --sample_steps 16 \
  --seed 589 \
  --device cuda \
  --output results/autoresearch/589a_conditional_latent_cumulative_flow/audit.json
```

Official bridge:

```bash
python experiments/backfill/block_ar/evaluate_588a_unified_flow_full11_bridge.py \
  --checkpoint models/backfill/589a_conditional_latent_cumulative_flow_s589/best_model.pt \
  --max_windows 441 \
  --samples 48 \
  --sample_steps 16 \
  --batch_size 32 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --seed 589 \
  --device cuda \
  --output_json results/autoresearch/589a_conditional_latent_cumulative_flow/full11.json \
  --output_md results/autoresearch/589a_conditional_latent_cumulative_flow/full11.md
```

Focused tests:

```bash
pytest test_code/test_577a_unified_increment_flow.py \
  test_code/test_588a_unified_flow_full11_bridge.py -q
```

Result: `10 passed`.

## Result

Custom audit:

- IV max: `1.017`;
- IV 90% coverage: `0.261`;
- factor 90% coverage: `0.225`;
- IV sample std / GT std: `0.883`;
- increment effective rank: `7.17`;
- IV endpoint corr: `0.667`;
- factor endpoint corr: `0.052`.

Official score:

- `3/11`;
- passes: surface validity, block-AR boundary/growing uncertainty, cross-cell
  correlation;
- fails: coverage, conditionality, time-series, cointegration, regime coverage,
  distributional fidelity, mean reversion, pathwise jump realism.

Important official metrics:

- explosion rate: `0.0%`;
- overall 90% coverage: `26.7%`;
- daily-change KS pass cells: `1/25`;
- level KS pass cells: `2/25`;
- cross-cell corr/rank ratios: `0.578 / 1.821`;
- h1 mean-reversion ratio: `0.383`;
- pathwise max-jump KS: `0.999`;
- per-cell q99 jump-scale passes: `1/25`.

## Mechanism Read

The latent bottleneck did exactly what it was supposed to do for dependence:

- official cross-cell correlation passes;
- official effective-rank ratio passes;
- custom increment effective rank is near the GT order of magnitude;
- surface explosion disappears.

But it over-compresses stochastic amplitude:

- coverage collapses;
- daily tail scale is far too small;
- pathwise jump incidence is zero;
- time-series ACF/kurtosis fail because the generated local moves are too smooth.

This is a cleaner pathology than 588a. The architecture is not incoherent; it
needs controlled amplitude/tail recovery while preserving the latent dependence
structure.

## Decision

Keep the latent-bottleneck source alive.

Next step should test whether the latent source has a usable uncertainty
temperature frontier: increase sample amplitude generically at inference/training
time and score the coverage/tail versus cross-cell tradeoff. If a simple generic
temperature can recover coverage/tails while cross-cell still passes, then the
next trained model should learn or calibrate that amplitude. If not, abandon the
unified MLP path-flow branch.
