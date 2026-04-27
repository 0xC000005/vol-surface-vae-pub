# 587a Conditional-Affine Source With Cumulative Flow Loss

## Hypothesis

586a showed that a learned conditional source makes the flow do real work, but
pointwise increment/velocity loss is misaligned with reconstructed future state
paths. A final-series-oriented loss should reduce log-level compounding without
adding IV-specific clamps.

## Change

Extended `UnifiedIncrementFlow.training_loss` with:

- `fm_loss_mode="velocity"`: existing pointwise velocity MSE;
- `fm_loss_mode="cumulative_state"`: MSE of cumulative velocity residuals across
  future time, which corresponds to encoded-state path error.

The model architecture is unchanged from 586a:

- one shared 38-variable IV-plus-anchor-factor panel;
- one learned conditional-affine Gaussian source over the full future increment
  tensor;
- one shared flow;
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
  --fm_loss_mode cumulative_state \
  --epochs 20 \
  --batch_size 64 \
  --lr 7e-4 \
  --sample_windows 128 \
  --n_samples 8 \
  --sample_steps 16 \
  --output_dir models/backfill/587a_conditional_affine_cumulative_flow_s587 \
  --seed 587 \
  --device cuda
```

Full audit:

```bash
python experiments/backfill/block_ar/audit_583a_unified_flow_sample_quality.py \
  --checkpoint models/backfill/587a_conditional_affine_cumulative_flow_s587/best_model.pt \
  --sample_windows 441 \
  --n_samples 32 \
  --sample_steps 16 \
  --seed 587 \
  --device cuda \
  --output results/autoresearch/587a_conditional_affine_cumulative_flow/audit.json
```

Focused tests:

```bash
pytest test_code/test_577a_unified_increment_flow.py \
  test_code/test_583a_unified_flow_sample_quality.py \
  test_code/test_584a_unified_flow_realism_finetune.py -q
```

Result: `12 passed`.

## Result

Training:

- best epoch: `13`;
- best cumulative validation loss: `6.3875`;
- quick audit IV max: `11.16`, down from 586a's quick-audit `174.0`.

Full source-vs-post-flow audit:

| Metric | 586a Post Flow | 587a Source Only | 587a Post Flow |
|---|---:|---:|---:|
| IV max | `455.48` | `12.33` | `15.08` |
| IV q99.9% | `4.75` | `0.986` | `0.978` |
| IV 90% coverage | `0.820` | `0.529` | `0.543` |
| Factor 90% coverage | `0.542` | `0.244` | `0.480` |
| IV sample std / GT std | `7.56` | `1.200` | `1.198` |
| IV endpoint corr | `0.101` | `0.559` | `0.605` |
| Factor endpoint corr | `0.009` | `0.027` | `0.263` |
| Increment effective rank | `256.9` | `807.8` | `494.9` |

## Mechanism Read

The cumulative loss directly fixed the 586a compounding failure:

- IV extremes dropped by roughly `30x`;
- global IV spread became near-GT instead of `7.6x`;
- endpoint conditionality recovered for IV and improved materially for factors;
- the flow still changes the prior, especially factor coverage and endpoint
  behavior, so this is not pure prior sampling.

The remaining failure is now undercoverage rather than explosion:

- IV 90% coverage is only `0.543`;
- factor 90% coverage is only `0.480`;
- state coverage is `0.521`.

This is a cleaner and more hopeful pathology than 586a: the model is conditional
and realistic in scale, but it is too narrow.

## Decision

587a should become the current learned-prior unified-flow prototype, but it is not
yet proven against the official 11-suite.

Next step: build an official IV-suite bridge for unified-increment-flow checkpoints
so this branch can be scored against the same 11 gates as 392a/510a. If the bridge
shows mostly coverage/regime failures, the next model change should increase
conditional uncertainty under the same cumulative objective rather than changing
the architecture again.
