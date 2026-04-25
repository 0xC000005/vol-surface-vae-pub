# 493a Path-Level Shift Oracle

## Question

After 492a failed, test the narrowest path-law-preserving alternative: keep the
392a generated path shape and apply only one constant logit-level shift per
validation window/cell across the whole 30-day future.

This is an oracle, not a deployable model. It uses the realized future to choose
the shift, so the purpose is only to test whether 392a's remaining failures are
compatible with a simple level-location correction.

## Artifacts

- Script: `experiments/backfill/block_ar/analyze_493a_392a_path_level_shift_oracle.py`
- Results: `results/block_ar/493a_392a_path_level_shift_oracle/oracle.json`
- Markdown: `results/block_ar/493a_392a_path_level_shift_oracle/oracle.md`

## Result

The audit omits conditionality because it evaluates already-generated sample
tensors. It compares a same-seed 392a resample against the oracle shifted samples.

| Variant | Score Ex-Conditionality | Failed Suites | Coverage90 | Level KS | Daily KS | Regime L2 | MR Active | Path KS |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 392a resample | 7/10 | coverage, regime_coverage, distributional_fidelity | 0.8661 | 12/25 | 25/25 | 0/8 | 83.3% | 0.394 |
| oracle constant logit shift | 7/10 | coverage, regime_coverage, mean_reversion | 0.9209 | 22/25 | 25/25 | 0/8 | 45.8% | 0.474 |

## Mechanism Read

The oracle shift is partially effective but not sufficient:

- It confirms that a large part of the distributional-fidelity failure is level
  location: level KS improves from `12/25` to `22/25`, median-bias from `20/25`
  to `25/25`, while daily-change KS stays `25/25`.
- It does not solve the risk-manager coverage problem. Overall 90% coverage rises
  to `92.1%`, but h1 coverage drops to `70.3%` and per-regime/per-cell layer2
  remains `0/8`.
- It creates a mean-reversion failure: active pass rate falls from `83.3%` to
  `45.8%`, so oracle level centering is not neutral to deployable dynamics.

## Decision

Do not implement a learned version of this constant shift as the next model. It
would be a narrow calibration branch that is already falsified by its own oracle:
even with future information, it cannot satisfy coverage/regime layer2 and it
breaks mean-reversion.

The remaining pathology is not just center/location. It is horizon/cell/regime
interval allocation under a valid path law. The next move should be a clean
paradigm decision around whether to optimize the path law itself with coverage
and level occupancy as proper training scores, rather than adding a posthoc
location map.
