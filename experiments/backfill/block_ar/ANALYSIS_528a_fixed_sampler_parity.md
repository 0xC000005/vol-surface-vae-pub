# 528a Fixed-Sampler Parity Audit

## Context
526a and 527a scored the factor-conditioned checkpoint with a fixed precomputed-sample evaluator because the standard full-suite conditionality loader does not know how to provide factor histories. The result was stable `7/11`, below the historical 392a/510a `8/11` frontier. 528a checks whether that comparison is protocol-consistent by evaluating the original 392a checkpoint through the same fixed-sample evaluator and seed.

## Run

```bash
python experiments/backfill/block_ar/evaluate_525a_factor_conditioned_surface_fm.py \
  --checkpoint models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt \
  --max_windows 192 \
  --samples 48 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --batch_size 32 \
  --chunk_size 4 \
  --seed 42 \
  --device cuda \
  --output_json results/autoresearch/528a_392a_fixed_sampler_parity_seed42/full11.json \
  --output_md results/autoresearch/528a_392a_fixed_sampler_parity_seed42/full11.md
```

## Result
Under the same fixed-sample protocol, original 392a scores `6/11`, not its historical `8/11`.

| metric | 392a original | 392a fixed seed42 | factor fixed seed42 |
|---|---:|---:|---:|
| score | `8/11` | `6/11` | `7/11` |
| failed suites | coverage, regime, fidelity | coverage, conditionality, cointegration, regime, fidelity | coverage, conditionality, regime, fidelity |
| cov90 overall | `0.868` | `0.865` | `0.863` |
| conditional MAE reduction | `5.14%` | `4.79%` | `4.91%` |
| cointegration ratio | `0.700` | `0.673` | `0.681` |
| cointegration worst-cell ratio | `0.278` | `0.246` | `0.263` |
| level KS | `10/25` | `11/25` | `12/25` |
| regime layer2 | `0/8` | `0/8` | `0/8` |
| mean-reversion active pass | `0.833` | `0.833` | `0.833` |
| path max-jump KS | `0.373` | `0.396` | `0.396` |

## Mechanism Read
The apparent drop from historical `8/11` to factor-conditioned `7/11` is not primarily caused by the factor side-channel. Under the same fixed-sample protocol, the factor checkpoint improves over 392a: it recovers cointegration, improves conditionality margin, and improves level KS.

The protocol gap matters: the fixed-sample evaluator precomputes one sample tensor and reuses it in conditionality, while the standard evaluator calls the live sampler during conditionality. That can move thin gates such as conditionality and cointegration around the threshold.

## Decision
Do not close the factor-conditioned branch. The next step is evaluator correction, not model modification: add a live conditionality sampler that can provide the matched factor history for each history tensor, while keeping the precomputed sample tensor for the other suites. Then rerun the factor checkpoint under this closer-to-standard protocol.
