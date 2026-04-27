# 584a Unified Flow Realism Fine-Tune

## Context

583a showed that 582c is dominated by conditional source selection: source-only
and post-flow samples are nearly identical. The next clean experiment was to make
the learned transport stronger without splitting IV and factors.

## Implementation

Added:

- `experiments/backfill/block_ar/train_584a_unified_flow_realism_finetune.py`;
- `test_code/test_584a_unified_flow_realism_finetune.py`.

The fine-tune keeps:

- one shared unified flow;
- one cleaned 38-variable state panel;
- conditional empirical source top-k 4;
- no separate IV/factor heads.

Added generic reconstructed encoded-state losses:

- train-range exceedance penalty;
- sample interval coverage penalty.

## Run

Command:

```bash
python experiments/backfill/block_ar/train_584a_unified_flow_realism_finetune.py \
  --checkpoint models/backfill/582c_clean_conditional_empirical_source_top4_flow_s586/best_model.pt \
  --epochs 6 \
  --batch_size 48 \
  --lambda_realism 0.25 \
  --realism_samples 4 \
  --realism_steps 4 \
  --sample_windows 128 \
  --audit_samples 16 \
  --audit_steps 16 \
  --output_dir models/backfill/584a_unified_flow_realism_finetune_s584 \
  --seed 584 \
  --device cuda
```

Best checkpoint:

- epoch: `4`;
- selection metric: `2.65663746`.

Full 583-style audit:

```bash
python experiments/backfill/block_ar/audit_583a_unified_flow_sample_quality.py \
  --checkpoint models/backfill/584a_unified_flow_realism_finetune_s584/best_model.pt \
  --sample_windows 441 \
  --n_samples 32 \
  --sample_steps 16 \
  --seed 584 \
  --device cuda \
  --output results/autoresearch/584a_unified_flow_realism_finetune/audit.json
```

## Result

582c baseline versus 584a:

| Metric | 582c | 584a |
|---|---:|---:|
| IV max | `10.434` | `10.378` |
| IV 90% coverage | `0.5210` | `0.5220` |
| Factor 90% coverage | `0.5002` | `0.5002` |
| IV sample std / GT std | `2.181` | `2.204` |
| IV endpoint corr | `0.593` | `0.585` |
| Factor endpoint corr | `0.057` | `0.056` |
| Increment effective rank | `69.41` | `69.27` |

## Mechanism Read

584a does not materially improve the system.

The training-time realism losses are numerically tiny relative to the flow loss:

- range loss around `3e-05`;
- coverage loss around `8e-04`;
- flow loss around `1.5`.

As a result, even with `lambda_realism=0.25`, the fine-tune leaves the model
effectively unchanged. The full audit confirms this: IV max and coverage barely
move, while IV global spread remains too high and coverage remains too low.

## Decision

Do not keep tuning this small auxiliary loss. It is not the right lever.

The current unified source-conditioned flow family is hitting a methodological
limit:

- the source does most of the work;
- the learned transport is weak;
- tightening source top-k becomes retrieval-like;
- small reconstructed-state penalties do not change the learned law.

Next step should be research ideation / paradigm review using recent time-series
DNN work, with focus on stronger conditional transport or proper multivariate
likelihood objectives rather than more source-selection knobs.
