# 579a Cleaned Unified Increment Panel

## Context

578a showed that full-path source geometry helps IV, but also exposed a data
curation problem: zero-filled price-like factor levels were creating artificial
log-coordinate jumps in the training target/source.

579a repairs the unified data framing without adding separate IV/factor model
heads.

## Data-Framing Change

Added a data-driven cleaning rule:

- for factor level series with log-return columns and nonnegative support, treat
  non-positive values as missing and fill them before log increments are computed;
- for factor level series that genuinely go negative, use a difference-coordinate
  transform instead of forcing a log transform.

In the current panel:

- cleaned `factor:copper`: `167` values;
- cleaned `factor:wheat`: `135` values;
- cleaned `factor:nikkei`: `1` value;
- cleaned `factor:gold`: `167` values;
- used difference-coordinate fallback for `factor:crude_oil`.

## Audit

Command:

```bash
python experiments/backfill/block_ar/audit_576a_unified_increment_panel.py \
  --future_lens 30 60 90 152 \
  --clean_nonpositive_log_levels \
  --output results/autoresearch/579a_clean_unified_increment_panel/audit.json
```

Result:

- target state variables: `38`;
- no duplicate return-like target channels;
- no test leakage for h30/h60/h90/h152;
- train log-transform floor hits: `0` for all audited horizons;
- validation log-transform floor hits: `0` for all audited horizons;
- reconstruction max absolute error: `0.0`.

## Training Probe

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
  --clean_nonpositive_log_levels \
  --output_dir models/backfill/579a_clean_unified_increment_path_gaussian_flow_s580 \
  --seed 580 \
  --device cuda
```

Result:

- best epoch: `17`;
- best val loss: `2.6060788972`;
- finite sample increment rate: `1.0`;
- finite sample state rate: `1.0`;
- standardized increment std ratio: `1.0627`;
- reconstructed factor max: about `21,309.59`;
- reconstructed IV max: about `261.08`.

## Mechanism Read

The cleaning change fixed the factor explosion:

- 578a factor max: about `3.76M`;
- 579a factor max: about `21.3k`, inside the historical broad factor scale.

The remaining problem is now cleaner and narrower:

- the unified architecture is still shared;
- factor data contamination is no longer the dominant issue;
- source geometry is improved;
- IV log-increment tails remain too permissive under a Gaussian source.

## Decision

The next step should not split IV and factors again.

The next clean step is tail control in the unified increment coordinate, likely by
changing the marginal increment representation or source distribution:

- empirical normal-score increments;
- clipped/quantile-bounded source sampling as an explicit risk policy layer;
- or a non-Gaussian empirical source over full paths.

The first option is the most publication-aligned because it is a generic
data-transform choice rather than an IV-specific correction.
