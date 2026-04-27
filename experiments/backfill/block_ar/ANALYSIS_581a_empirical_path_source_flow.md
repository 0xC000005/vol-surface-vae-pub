# 581a Empirical Full-Path Source Unified Flow

## Context

580a showed that marginal normal-score increment bounds are not enough. The
remaining issue is full-path tail dependence: Gaussian sources can combine too
many high positive IV increments in one reconstructed path.

581a tests a non-Gaussian source at the full-path level while keeping the same
shared unified flow architecture.

## Implementation

Extended `train_577a_unified_increment_flow.py` with:

- `--source_mode empirical_path`;
- a training increment source bank with shape `(4010, 1140)`;
- source samples drawn as full historical 30-day increment paths;
- sample audit loaded from the best-validation checkpoint rather than the final
  epoch.

This does not split IV and factors. The source is one empirical path law over the
cleaned 38-variable increment tensor.

## Run

Best-audited rerun:

```bash
python experiments/backfill/block_ar/train_577a_unified_increment_flow.py \
  --epochs 20 \
  --batch_size 64 \
  --hidden_dim 384 \
  --depth 5 \
  --future_len 30 \
  --source_mode empirical_path \
  --increment_transform standard \
  --clean_nonpositive_log_levels \
  --output_dir models/backfill/581b_clean_empirical_source_unified_flow_s583 \
  --seed 583 \
  --device cuda
```

Result:

- best epoch: `17`;
- best val loss: `2.5859909909`;
- source bank rows: `4010`;
- source bank dim: `1140`;
- finite sample increment rate: `1.0`;
- finite sample state rate: `1.0`;
- standardized increment std ratio: `1.0536`;
- reconstructed factor max: about `20,772.86`;
- reconstructed IV max: about `44.67`.

## Comparison

| Run | Source / transform | IV max | Factor max |
|---|---|---:|---:|
| 579a | cleaned path Gaussian, standard increments | `261.08` | `21,309.59` |
| 580a | cleaned path Gaussian, normal-score increments | `7,025.73` | `21,063.14` |
| 581b | cleaned empirical full-path source, standard increments | `44.67` | `20,772.86` |

## Mechanism Read

Empirical full-path source is the best unified-flow source so far:

- it preserves factor realism after cleaning;
- it materially reduces IV path explosions;
- it confirms the relevant object is full-path source dependence, not marginal
  increment support.

It is still not deployable:

- reconstructed IV max around `44.67` is far outside historical IV surface scale;
- the flow remains weakly conditional, with predicted velocity std around `0.20`
  versus target std around `1.20`;
- an unconditional empirical source can still place a historically extreme
  increment path on a mismatched current history.

## Decision

The unified path remains alive, but the next step must improve conditional source
matching or conditional transport strength.

Clean next options:

- condition the empirical source bank on coarse current-history state using a
  generic nearest-neighbor or kernel source sampler;
- add a stronger conditional objective that penalizes unrealistic reconstructed
  level ranges directly;
- or use source temperature/calibration only as an explicitly reported policy
  layer.

The most principled research step is the first: conditional empirical source
matching over the unified history representation, because it addresses the
failure mechanism without adding IV/factor-specific heads.
