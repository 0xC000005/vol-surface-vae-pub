# 622a Encoded-State Joint38 AR Transition Result

## Hypothesis

621a identified a data-framing mismatch: the unified panel builder already defines log-level/diff-level state coordinates, but the recent native joint AR scripts trained on raw states. 622a tested whether training the same generic AR transition-flow in the intended encoded state coordinate improves long-horizon level placement, regime behavior, and mean reversion without adding architecture or post-hoc calibration.

## Change

Added `value_coordinate=raw|encoded` to:

- `experiments/backfill/block_ar/train_609a_unified_ar_transition_flow.py`
- `experiments/backfill/block_ar/evaluate_609a_unified_ar_transition_flow.py`

When `encoded`:

- training/validation histories and futures are transformed with `encode_state(...)`;
- the checkpoint records `value_coordinate`;
- evaluation feeds encoded histories to the model;
- generated encoded futures are decoded with `decode_state(...)`;
- decoded IV cells are passed to the official IV suite.

Focused test:

```bash
pytest test_code/test_622a_encoded_state_coordinate.py -q
```

Result: `3 passed`.

## Run

Training:

```bash
python experiments/backfill/block_ar/train_609a_unified_ar_transition_flow.py \
  --state_scope joint38 \
  --value_coordinate encoded \
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
  --seed 622 \
  --device cuda \
  --output_dir models/backfill/622a_joint38_ar_transition_encoded_e8_w2048_s622
```

Training selected epoch 4:

- best validation loss: `0.446832`;
- final validation loss: `0.452807`;
- decoded smoke finite rate: `1.0`.

Official broad-frame bridge:

```bash
python experiments/backfill/block_ar/evaluate_609a_unified_ar_transition_flow.py \
  --checkpoint models/backfill/622a_joint38_ar_transition_encoded_e8_w2048_s622/best_model.pt \
  --state_scope joint38 \
  --value_coordinate encoded \
  --max_windows 441 \
  --samples 48 \
  --n_steps 30 \
  --batch_size 32 \
  --chunk_size 8 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --seed 6221 \
  --device cuda \
  --output_json results/autoresearch/622a_joint38_ar_transition_encoded_e8_w2048/full11.json \
  --output_md results/autoresearch/622a_joint38_ar_transition_encoded_e8_w2048/full11.md
```

## Result

Score: `4/11`.

Passed:

- surface;
- block-AR;
- cross-cell correlation;
- mean reversion.

Failed:

- coverage;
- conditionality;
- time-series;
- cointegration;
- regime coverage;
- distributional fidelity;
- pathwise jump realism.

Key metrics:

- cov90 overall: `49.0%`;
- h1/h7/h14/h30 cov90: `75.3% / 58.9% / 48.4% / 33.9%`;
- conditionality MAE reduction: `-2.4%`;
- turbulent/calm width ratio: `0.973`;
- ACF correlation: `0.983`;
- kurtosis ratio: `0.633`;
- daily-change KS cells: `11/25`;
- level KS cells: `0/25`;
- median-bias cells: `1/25`;
- persistent severe undercoverage: `3748/11025 = 34.0%`;
- cointegration gen/GT ratio: `0.426`, worst-cell `0.134`;
- cross-cell corr/rank: `0.905 / 1.546`;
- mean-reversion ratio: `0.920`, active pass `10/12`, active corr `0.903`;
- pathwise max-jump KS: `0.403`;
- per-cell q99 jump-scale cells: `14/25`.

## Mechanism Read

The encoded coordinate is not a deployable solution, but it is scientifically useful.

Positive:

- it restores mean reversion very cleanly;
- surface validity survives;
- cross-cell structure survives;
- block-AR mechanics remain intact.

Negative:

- it becomes severely under-dispersed over the 30-day horizon;
- generated medians are persistently above realized futures in most cells;
- lower-side coverage collapses, especially at h30;
- cointegration and local daily-change distribution degrade.

The causal read is that encoded log/diff coordinates make the transition geometry more stationary and mean-reverting, but the model now reverts too aggressively toward the training encoded-state center. It fixes one structural pathology by exposing another: broad-frame validation futures sit below the generated median path in many cells, so interval width alone may help coverage but cannot fully fix path location.

## Decision

Keep the encoded-coordinate implementation because it is a general preprocessing option and directly supports IV-only and joint38 with one model. Do not promote 622a.

The next minimal diagnostic is an encoded-coordinate sample-temperature sweep. This is not a proposed final risk knob; it tests whether 622a's failure is mostly stochastic width or fundamentally path-location bias. If moderate temperature cannot recover coverage/regime inclusion without breaking surface, mean reversion, and pathwise realism, close encoded coordinate as a primary route and move to a stochastic path-law objective rather than more preprocessing.

