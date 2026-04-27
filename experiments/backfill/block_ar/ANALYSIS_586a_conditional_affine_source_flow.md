# 586a Conditional-Affine Source Unified Flow

## Hypothesis

If the 582c/584a ceiling came from empirical-source retrieval dominating the
model, then replacing the source bank with a learned conditional Gaussian prior
should make the flow do more real conditional transport while preserving the clean
unified IV-plus-factor panel.

## Change

Extended `UnifiedIncrementFlow` with:

- `source_mode="conditional_affine"`;
- one source head from the shared GRU history context to a full-path Gaussian
  mean and log-scale;
- optional `source_prior_nll_weight` to train the source as a conditional prior;
- no empirical source bank;
- no IV/factor-specific heads.

## Run

```bash
python experiments/backfill/block_ar/train_577a_unified_increment_flow.py \
  --clean_nonpositive_log_levels \
  --hidden_dim 384 \
  --depth 5 \
  --source_mode conditional_affine \
  --source_prior_nll_weight 0.05 \
  --epochs 20 \
  --batch_size 64 \
  --lr 7e-4 \
  --sample_windows 128 \
  --n_samples 8 \
  --sample_steps 16 \
  --output_dir models/backfill/586a_conditional_affine_source_flow_s586 \
  --seed 586 \
  --device cuda
```

Full audit:

```bash
python experiments/backfill/block_ar/audit_583a_unified_flow_sample_quality.py \
  --checkpoint models/backfill/586a_conditional_affine_source_flow_s586/best_model.pt \
  --sample_windows 441 \
  --n_samples 32 \
  --sample_steps 16 \
  --seed 586 \
  --device cuda \
  --output results/autoresearch/586a_conditional_affine_source_flow/audit.json
```

## Result

Training:

- best epoch: `20`;
- validation loss: `1.8713`, much lower than 582c's `2.5459`;
- parameters: `3.14M`;
- quick audit IV max: `174.0`.

Full source-vs-post-flow audit:

| Metric | Source Only | Post Flow |
|---|---:|---:|
| IV max | `678.56` | `455.48` |
| IV q99.9% | `4.92` | `4.75` |
| IV 90% coverage | `0.788` | `0.820` |
| Factor 90% coverage | `0.493` | `0.542` |
| IV sample std / GT std | `9.14` | `7.56` |
| Increment std | `0.328` | `0.465` |
| Increment effective rank | `906.0` | `256.9` |
| IV endpoint corr | `-0.031` | `0.101` |
| Factor endpoint corr | `0.004` | `0.009` |

Focused tests:

```bash
pytest test_code/test_577a_unified_increment_flow.py \
  test_code/test_583a_unified_flow_sample_quality.py \
  test_code/test_584a_unified_flow_realism_finetune.py -q
```

Result: `11 passed`.

## Mechanism Read

586a is not source-dominated in the same way as 582c:

- post-flow IV max is lower than source-only by `223`;
- IV coverage improves by `3.2` percentage points;
- factor coverage improves by `4.9` percentage points;
- rank drops substantially after flow, so the transport is materially reshaping
  the source.

But it is not deployable:

- IV levels explode because small transformed-increment errors compound through
  cumulative log-level reconstruction;
- low validation increment-space loss is misleading for final path realism;
- endpoint conditionality is weak for both IV and factors;
- the conditional prior is too broad in final IV-level space even though its
  transformed increment std is not large.

The bottleneck is therefore more precise than before: the learned prior/flow is
now doing work, but the objective is still velocity/increment-oriented while the
risk object is the reconstructed future state path.

## Decision

Keep the learned-conditional-prior idea alive only as a component. Do not tune
source log-scale caps or add IV-specific clamps.

The next principled experiment is a final-series-oriented loss: penalize
cumulative velocity error in encoded state space, not only pointwise increment
velocity error. This is the local analogue of the recent final-series-oriented
flow-loss signal and directly targets the compounding pathology without
introducing factor-specific rules.
