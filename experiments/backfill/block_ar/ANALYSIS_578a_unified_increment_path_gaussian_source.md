# 578a Unified Increment Flow with Path-Gaussian Source

## Context

577a showed that the unified architecture can train, but an independent Gaussian
source over daily standardized increments is the wrong path prior. It produces
unrealistic cumulative log paths and explosive reconstructed levels.

578a kept the same shared architecture and changed only the flow source geometry:

- source mode: full-path empirical Gaussian;
- source fitted on standardized `(30, 38)` training increment paths;
- covariance shrinkage: `0.10`;
- covariance jitter: `0.0003`;
- no separate IV/factor heads.

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
  --output_dir models/backfill/578a_unified_increment_path_gaussian_flow_s579 \
  --seed 579 \
  --device cuda
```

Result:

- params: `2,117,716`;
- best epoch: `20`;
- best val loss: `2.5767757211`;
- source Cholesky diag range: `0.40985` to `1.00338`;
- finite sample increment rate: `1.0`;
- finite sample state rate: `1.0`;
- standardized increment std ratio: `1.0882`;
- reconstructed IV max: about `194.42`;
- reconstructed factor max: about `3,764,778.75`.

## Mechanism Read

The path-Gaussian source materially improves IV path geometry:

- 577a short independent source IV max: about `3.5M`;
- 577b longer independent source IV max: about `134k`;
- 578a path-Gaussian source IV max: about `194`.

That is still too high for deployability, but it is a large improvement and
confirms that source geometry matters.

The remaining factor explosion is not primarily an architecture issue. The 576a
audit now has a sharper interpretation:

- the training split has many log-transform floor hits for price-like factor
  levels;
- examples include commodities and metals with zero-filled historical levels;
- log-level increments through an artificial zero floor create huge synthetic
  jumps;
- the path Gaussian then learns and samples from a contaminated target/source
  distribution.

In short: 578a exposed a data-curation problem in the unified factor panel.

## Decision

Do not add factor-specific model heads.

The next principled step is to repair the unified data framing:

- treat non-positive price-like factor levels as missing rather than true zeros;
- fill those level series causally/split-safely before computing canonical
  increments;
- recompute the 576a audit and verify that log-transform floor hits disappear;
- then rerun the same path-Gaussian shared flow.

This is aligned with the user's concern about data processing and with the
time-series literature signal: representation and source geometry must be right
before model complexity is increased.
