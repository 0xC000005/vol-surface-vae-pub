# 593a AR Common Latent Wrapper

## Hypothesis

Prior common-source AR attempts were fixed noise-correlation variants and were already falsified. A cleaner remaining AR idea was to keep the 510a/392a transition model frozen, add one scenario-level stochastic latent, and train only a small projection from that latent into the AR memory state through rollout patch-energy loss.

This tests whether a learned persistent scenario factor can improve path-level realism and conditional distributional structure without adding evaluator-specific rules or a separate factor treatment.

## Implementation

- Base model: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`
- Wrapper: `ARCommonLatentWrapper`
- Trainable parameters: one `LayerNorm(latent_dim)` plus one linear projection from latent dimension 16 into the base AR memory dimension.
- Frozen parameters: all base AR transition-model weights.
- Training window: recent validation-adjacent block from index 3569 to 4009.
- Training objective: sampled rollout patch-energy score against realized future scores.
- No fixed `path_source_corr`, no rho sweep, no handcrafted source schedule.

## Commands

```bash
python experiments/backfill/block_ar/train_593a_ar_common_latent_wrapper.py \
  --mode train \
  --checkpoint models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt \
  --latent_dim 16 \
  --epochs 4 \
  --batch_size 8 \
  --train_sample_count 4 \
  --rollout_flow_steps 4 \
  --patch_len 5 \
  --lr 1e-3 \
  --output_dir models/backfill/593a_ar_common_latent_wrapper_s593 \
  --seed 593 \
  --device cuda
```

```bash
python experiments/backfill/block_ar/train_593a_ar_common_latent_wrapper.py \
  --mode eval \
  --wrapper_checkpoint models/backfill/593a_ar_common_latent_wrapper_s593/best_model.pt \
  --max_windows 441 \
  --samples 48 \
  --batch_size 32 \
  --chunk_size 4 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --seed 593 \
  --device cuda \
  --output_json results/autoresearch/593a_ar_common_latent_wrapper/full11.json \
  --output_md results/autoresearch/593a_ar_common_latent_wrapper/full11.md
```

## Result

593a scored `4/11` on the official full suite.

Passed suites:

- `surface`
- `block_ar`
- `cross_cell_correlation`
- `pathwise_jump_realism`

Failed suites:

- `coverage`
- `conditionality`
- `time_series`
- `cointegration`
- `regime_coverage`
- `distributional_fidelity`
- `mean_reversion`

Key metrics:

- Coverage90 overall: `69.7%`
- Coverage90 by horizon h1/h7/h14/h30: `79.4% / 70.2% / 68.9% / 66.8%`
- Conditionality aggregate MAE reduction: `6.55%`
- Daily-change KS pass cells: `25/25`
- Level KS pass cells: `3/25`
- Cross-cell corr ratio / rank ratio: `1.062 / 1.229`
- Kurtosis ratio / skew ratio: `0.623 / -0.133`
- Cointegration aggregate gen/GT ratio: `0.621`
- Cointegration worst-cell ratio: `0.239`
- Regime layer2 pass count: `0/8`
- Mean-reversion active-cell pass rate: `9/12`
- Mean-reversion active-cell correlation: `0.560`
- Pathwise max-jump KS: `0.475`

The wrapper did learn a nonzero projection (`val_latent_proj_abs` about `0.0076` at the best epoch), so this is not a pure no-op. But the learned perturbation is not useful at the suite level.

## Mechanism Read

The learned common latent preserves several local/path properties from the AR base: surface sanity, block-AR smoothness, daily-change KS, cross-cell correlation, and jump realism remain strong. It also improves aggregate conditionality enough to pass the aggregate reduction check.

The failure is that this same latent perturbation disrupts the fine distributional balance that made 510a/392a the frontier. Level KS collapses to `3/25`, tail-shape metrics fail, one cointegration cell drops below the hard floor, and mean-reversion active-cell correlation fails. The model gains a persistent scenario factor, but it does not learn the right conditional law of persistent future paths.

## Decision

593a is a clean falsification of the small learned common-latent wrapper as a direct route past the 8/11 frontier. It is below 510a, not a candidate deployment model.

The next principled step is not to deepen this wrapper blindly. The next iteration should compare 593a against the 510a frontier at the metric-delta level and decide whether there is a single constrained repair that preserves 510a's 8/11 behavior, or whether the learned-wrapper branch should be closed.
