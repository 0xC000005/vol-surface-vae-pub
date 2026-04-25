# 530a Factor-Conditioned Surface FM Convergence Audit

## Context
529a showed that live factor-conditioned sampling improved the same-protocol 392a parity result from `6/11` to `7/11`, but still missed the frontier. The only remaining clean question for this side branch was whether the factor context was simply undertrained. 530a therefore ran the same architecture and same objective longer, with no new loss, calibration policy, decoder structure, or evaluator-specific knob.

## Run
Training:

```bash
python experiments/backfill/block_ar/train_525a_factor_conditioned_surface_fm.py \
  --checkpoint models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt \
  --adaptation_windows 441 \
  --epochs 16 \
  --batch_size 32 \
  --lr 5e-4 \
  --device cuda \
  --output_dir models/backfill/525a_factor_conditioned_surface_fm_e16_s525
```

Evaluation:

```bash
python experiments/backfill/block_ar/evaluate_525a_factor_conditioned_surface_fm.py \
  --checkpoint models/backfill/525a_factor_conditioned_surface_fm_e16_s525/best_model.pt \
  --max_windows 192 \
  --samples 48 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --conditionality_mode live \
  --batch_size 32 \
  --chunk_size 4 \
  --seed 42 \
  --device cuda \
  --output_json results/autoresearch/530a_factor_conditioned_surface_fm_e16_live_seed42/full11.json \
  --output_md results/autoresearch/530a_factor_conditioned_surface_fm_e16_live_seed42/full11.md
```

The learned factor context scale increased from about `0.0054` in the 4-epoch run to `0.0215` by epoch 16, so this was a real convergence audit rather than a no-op.

## Result
Score: `7/11`.

Failed suites:

- `coverage`
- `cointegration`
- `regime_coverage`
- `distributional_fidelity`

Key metrics:

| metric | 529a e4 live | 530a e16 live |
|---|---:|---:|
| score | `7/11` | `7/11` |
| cov90 overall | `0.863` | `0.857` |
| h30 cov90 | `0.882` | `0.876` |
| conditional MAE reduction | `4.93%` | `5.21%` |
| turb/calm width ratio | `1.058` | `1.063` |
| cointegration gen/GT ratio | `0.681` | `0.669` |
| cointegration worst-cell ratio | `0.263` | `0.246` |
| regime layer2 | `0/8` | `0/8` |
| daily KS pass cells | `25/25` | `25/25` |
| level KS pass cells | `12/25` | `10/25` |
| median-bias cells | `19/25` | `19/25` |
| mean-reversion active pass | `0.833` | `0.833` |
| path max-jump KS | `0.396` | `0.410` |

## Mechanism Read
Longer adaptation does make the model use conditioning enough to clear the hard conditionality gate (`5.21%` MAE reduction), but the same training does not improve the structural bottlenecks. Coverage drifts slightly down, level KS falls from `12/25` back to `10/25`, and the cointegration worst-cell ratio slips below the `0.25` gate.

The failure mechanism is clean: broader observed factors are not the missing state variable for the suite's remaining failures. The frontier failures are dominated by future IV-level occupancy and per-cell/regime coverage geometry. A small factor side-channel can modulate the existing model, but it does not repair the learned future level law.

## Decision
Close the 525/530 factor-conditioned side-channel branch as below-frontier. It remains useful evidence because it shows that adding broader market history is mechanically safe and can improve conditionality, but it is not the next path to `11/11`.

The next principled HEAD step is research ideation from the frontier artifacts, not another factor epoch or another conditioning side-channel. The next candidate should directly address the future IV-level law while preserving the clean single-stage learned-generator framing.
