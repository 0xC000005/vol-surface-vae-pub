# 580a Normal-Score Unified Increment Flow

## Context

579a cleaned the unified factor panel and fixed factor-level explosions, but IV
tails remained too large under a path-Gaussian source. The next generic idea was
to control marginal increment tails with empirical normal-score increments.

## Implementation

Extended `train_577a_unified_increment_flow.py` with:

- `--increment_transform standard|normal_score`;
- empirical per-variable increment quantiles;
- normal-score transform for training;
- inverse empirical quantile transform for reconstruction;
- bounded inverse mapping for generated increment scores.

This remains a unified transform over all 38 variables, not an IV-specific cap.

## Run

Command:

```bash
python experiments/backfill/block_ar/train_577a_unified_increment_flow.py \
  --epochs 20 \
  --batch_size 64 \
  --hidden_dim 384 \
  --depth 5 \
  --future_len 30 \
  --source_mode path_gaussian \
  --source_cov_shrinkage 0.10 \
  --source_cov_jitter 0.0003 \
  --increment_transform normal_score \
  --n_increment_quantiles 501 \
  --increment_cdf_eps 0.0005 \
  --clean_nonpositive_log_levels \
  --output_dir models/backfill/580a_clean_normal_score_path_gaussian_flow_s581 \
  --seed 581 \
  --device cuda
```

Result:

- best epoch: `12`;
- best val loss: `1.6911344188`;
- finite sample increment rate: `1.0`;
- finite sample state rate: `1.0`;
- transformed increment std ratio: `1.0396`;
- reconstructed factor max: about `21,063.14`;
- reconstructed IV max: about `7,025.73`.

## Mechanism Read

Normal-score increments improve the transformed-space objective, but they do not
solve path realism.

The failure is now clear:

- per-variable marginal increment bounds are insufficient;
- a Gaussian full-path source can still combine many high positive IV increments
  along the same path;
- cumulative log-IV then explodes after inversion and reconstruction.

So the bottleneck is full-path tail dependence, not marginal increment tails.

## Decision

Do not add an IV-only cap.

The next clean source experiment should be non-Gaussian at the path level:

- sample the flow source from empirical training increment paths, optionally with
  small noise;
- keep the same shared flow architecture;
- evaluate whether full-path empirical source geometry prevents impossible
  cumulative IV paths while preserving the unified model.

This is not a final scenario retrieval product if the flow is still trained and
reported as the conditional law, but it should be treated carefully because it
does introduce a nonparametric source prior.
