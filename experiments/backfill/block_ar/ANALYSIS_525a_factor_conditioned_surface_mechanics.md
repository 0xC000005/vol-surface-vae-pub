# 525a Factor-Conditioned Surface FM Mechanics

## Context
524a selected a clean next prototype: keep the generated target as future IV levels in empirical normal-score coordinates, and add broader market state only as observed history conditioning.

## Implementation
Added optional `factor_history` support to `EmpiricalNormalScoreCausalMemoryTransitionFlowMatching`:

- new config field: `factor_dim`
- factor encoder: GRU over `(batch, history_len, factor_dim)`
- factor context: projected into the existing causal memory stream
- safety: zero-gated context scale, so extending a 392a checkpoint does not immediately perturb the base model
- legacy path: `factor_dim=0` keeps the no-factor model path unchanged

Added 525 experiment mechanics:

- `experiments/backfill/block_ar/_factor_conditioning_525_utils.py`
- `experiments/backfill/block_ar/train_525a_factor_conditioned_surface_fm.py`
- `experiments/backfill/block_ar/evaluate_525a_factor_conditioned_surface_fm.py`
- `test_code/test_factor_conditioned_surface_law.py`

## Verification
Focused tests:

```bash
pytest test_code/test_factor_conditioned_surface_law.py test_code/test_522a_38d_alignment.py -q
```

Result: `4 passed`.

Compile check:

```bash
python -m py_compile \
  experiments/backfill/block_ar/train_525a_factor_conditioned_surface_fm.py \
  experiments/backfill/block_ar/evaluate_525a_factor_conditioned_surface_fm.py \
  experiments/backfill/block_ar/_factor_conditioning_525_utils.py \
  diffusion/block_ar/empirical_normal_score_causal_memory_transition_flow_matching.py
```

Result: passed.

Smoke train:

```bash
python experiments/backfill/block_ar/train_525a_factor_conditioned_surface_fm.py \
  --checkpoint models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt \
  --adaptation_windows 64 \
  --epochs 1 \
  --batch_size 16 \
  --lr 5e-4 \
  --device cuda \
  --output_dir models/backfill/525a_factor_conditioned_surface_fm_smoke
```

Result: produced a factor-extended checkpoint with factor dim `26`; train loss `0.48260`; factor context scale moved to `0.00064`.

Smoke eval:

```bash
python experiments/backfill/block_ar/evaluate_525a_factor_conditioned_surface_fm.py \
  --checkpoint models/backfill/525a_factor_conditioned_surface_fm_smoke/best_model.pt \
  --max_windows 16 \
  --samples 8 \
  --conditionality_samples 8 \
  --conditionality_max_batches 1 \
  --batch_size 8 \
  --chunk_size 4 \
  --device cuda \
  --output_json results/autoresearch/525a_factor_conditioned_surface_fm_smoke/full11_smoke.json \
  --output_md results/autoresearch/525a_factor_conditioned_surface_fm_smoke/full11_smoke.md
```

Result: evaluator completed. The `3/11` smoke score is not meaningful because it used only 16 windows and 8 samples; it verifies integration only.

## Decision
525a is a mechanics pass. Proceed to a full 192-window, 48-sample 525a evaluation from a non-smoke short adaptation run. Acceptance remains: recover the 392a/510a structural passes before adding objectives or knobs.
