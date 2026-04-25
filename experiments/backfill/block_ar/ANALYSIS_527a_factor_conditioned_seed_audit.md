# 527a Factor-Conditioned Seed Audit

## Context
526a scored `7/11`, with the only extra lost suite versus 392a/510a being conditionality at `4.74%` against a `>5%` gate. 527a re-evaluated the same checkpoint with a different sampling seed to test whether that miss was random.

## Run

```bash
python experiments/backfill/block_ar/evaluate_525a_factor_conditioned_surface_fm.py \
  --checkpoint models/backfill/525a_factor_conditioned_surface_fm_e4_s525/best_model.pt \
  --max_windows 192 \
  --samples 48 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --batch_size 32 \
  --chunk_size 4 \
  --seed 42 \
  --device cuda \
  --output_json results/autoresearch/527a_factor_conditioned_surface_fm_e4_s525_seed42/full11.json \
  --output_md results/autoresearch/527a_factor_conditioned_surface_fm_e4_s525_seed42/full11.md
```

## Result
Score: `7/11`, same failure set as 526a:

- `coverage`
- `conditionality`
- `regime_coverage`
- `distributional_fidelity`

Comparison:

| metric | 526a seed 525 | 527a seed 42 |
|---|---:|---:|
| score | `7/11` | `7/11` |
| cov90 overall | `0.865` | `0.863` |
| h30 cov90 | `0.878` | `0.882` |
| conditional MAE reduction | `4.74%` | `4.91%` |
| turb/calm ratio | `1.063` | `1.059` |
| cointegration ratio | `0.726` | `0.681` |
| cointegration worst-cell ratio | `0.298` | `0.263` |
| level KS | `12/25` | `12/25` |
| mean-reversion active pass | `0.833` | `0.833` |
| path max-jump KS | `0.374` | `0.396` |

## Mechanism Read
The `7/11` result is stable across these two sampling seeds. The factor-conditioned side-channel consistently improves level KS versus the `10/25` 392a/510a frontier, while remaining just under the conditionality gate. This means the result should not be treated as a lucky or unlucky single-seed artifact.

## Decision
Do not claim an `8/11` tie. The factor-conditioned surface-level branch is promising but below-frontier as currently trained. Before changing the model, run an evaluator-parity audit: score the original 392a checkpoint through the same fixed-sample evaluation path, so we can separate side-channel effect from evaluator/sampling-protocol effect.
