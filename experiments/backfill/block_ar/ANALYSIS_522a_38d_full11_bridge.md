# 522a 38-d Official Full 11 Bridge

## Hypothesis
If the current bottleneck is mainly that IV-only models lack broader market state, then an existing 38-d joint IV+factor conditional baseline should score closer to the 392a/510a frontier once evaluated under the official full 11-suite.

## Execution
Added an official-aligned bridge evaluator:

- `test_code/test_522a_38d_alignment.py`
- `experiments/backfill/block_ar/evaluate_522a_38d_full11_bridge.py`

The bridge shifts 38-d daily-change conditioning one change earlier so reconstructed future IV surfaces align with the official full-suite target exactly. The focused regression test passes, and the run used the strongest existing 38-d deep baseline:

```bash
python experiments/backfill/block_ar/evaluate_522a_38d_full11_bridge.py \
  --baseline csdi \
  --samples 48 \
  --max_windows 192 \
  --conditionality_samples 32 \
  --conditionality_max_batches 8 \
  --batch_size 32 \
  --device cuda \
  --output_json results/autoresearch/522a_38d_full11_bridge/csdi_full11.json \
  --output_md results/autoresearch/522a_38d_full11_bridge/csdi_full11.md
```

## Result
Score: `4/11`.

Passed:

- `surface`
- `time_series`
- `block_ar`
- `cross_cell_correlation`

Failed:

- `coverage`
- `conditionality`
- `cointegration`
- `regime_coverage`
- `distributional_fidelity`
- `mean_reversion`
- `pathwise_jump_realism`

Key metrics:

- alignment max error: history `0.000e+00`, future reconstruction `1.341e-07`
- cov90 overall `0.707`; h30 cov90 `0.507`
- conditional MAE reduction `0.77%`
- turb/calm width ratio `1.009`
- cointegration gen/GT ratio `0.205`
- regime layer2 `1/8`
- daily-change KS `24/25`, but level KS `0/25`
- mean-reversion active pass `0.042`
- pathwise max-jump KS `0.678`

## Mechanism Read
The baseline learns plausible daily-change marginal shape and cross-cell dependence, but not a usable conditional IV-level law. It undercovers at long horizons, has almost no history-sensitive improvement versus shuffled histories, drifts below realized levels, and does not preserve the IV/EWMA long-run relation. This is the same core gap expressed in multiple tests: local daily-change realism is not enough for official level-path risk scenarios.

## Decision
Do not tune the existing 38-d CSDI bridge as the next main route. It is below the 392a/510a frontier and fails the suite dimensions that matter most for a deployable risk scenario generator. Keep the bridge as a reproducible baseline and falsifier. The next iteration should either design a new learned core that models future IV levels directly with a likelihood/proper-score objective, or run a targeted postmortem comparing 392a/510a versus 522a to isolate which inductive bias preserves the four extra passes.
