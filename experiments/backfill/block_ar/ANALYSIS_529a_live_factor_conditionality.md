# 529a Live Factor Conditionality Evaluation

## Context
528a showed that fixed-sample evaluation depresses the historical 392a score from `8/11` to `6/11`, so the factor-conditioned `7/11` should not be judged against historical 392a without a protocol correction. 529a updated the 525 evaluator to use live model sampling during conditionality, with factor histories matched by history key.

## Implementation
`evaluate_525a_factor_conditioned_surface_fm.py` now supports:

- `--conditionality_mode fixed`: reuse the precomputed sample tensor for conditionality;
- `--conditionality_mode live`: call the live model sampler during conditionality;
- no-factor checkpoints (`factor_dim=0`) for parity audits;
- factor checkpoints with matched factor histories for shuffled-history conditionality.

Verification:

```bash
python -m py_compile experiments/backfill/block_ar/evaluate_525a_factor_conditioned_surface_fm.py
pytest test_code/test_factor_conditioned_surface_law.py -q
```

Result: compile passed; `3 passed`.

## Run

```bash
python experiments/backfill/block_ar/evaluate_525a_factor_conditioned_surface_fm.py \
  --checkpoint models/backfill/525a_factor_conditioned_surface_fm_e4_s525/best_model.pt \
  --max_windows 192 \
  --samples 48 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --conditionality_mode live \
  --batch_size 32 \
  --chunk_size 4 \
  --seed 42 \
  --device cuda \
  --output_json results/autoresearch/529a_factor_conditioned_surface_fm_live_cond_seed42/full11.json \
  --output_md results/autoresearch/529a_factor_conditioned_surface_fm_live_cond_seed42/full11.md
```

## Result
Score remains `7/11`.

Failed suites:

- `coverage`
- `conditionality`
- `regime_coverage`
- `distributional_fidelity`

Key metrics:

- cov90 overall `0.863`
- h30 cov90 `0.882`
- conditional MAE reduction `4.93%`
- turb/calm ratio `1.058`
- cointegration ratio `0.681`
- cointegration worst-cell ratio `0.263`
- level KS `12/25`
- mean-reversion active pass `0.833`
- path max-jump KS `0.396`

## Mechanism Read
The live conditionality correction does not recover `8/11`. The side-channel remains a constructive change relative to same-protocol 392a, but it is not yet a frontier tie. The signal is consistent: factor conditioning improves level KS and cointegration margin, while conditionality remains just under the gate.

## Decision
Do not add a new loss or branch. One minimal convergence audit is justified because the factor side-channel was trained only four epochs and the learned context scale is still tiny. Run the same architecture/objective for a longer factor-only adaptation. If it still scores below `8/11`, close this factor-side-channel branch as below-frontier.
