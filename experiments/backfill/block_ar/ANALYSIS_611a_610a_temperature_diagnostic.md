# 611a 610a Sample-Temperature Diagnostic

## Context

610a recovered several structural suites but remained underinclusive in sparse and
turbulent validation windows. 611a tests the smallest possible risk-width
diagnostic before adding model architecture: generic sample temperature at
evaluation time.

This is not a claim of learned conditional law. It is a diagnostic for whether
the remaining gap is a simple scalar spread problem.

## Change

Added `--sample_temperature` to:

- `experiments/backfill/block_ar/evaluate_609a_unified_ar_transition_flow.py`.

The evaluator now passes the scalar through the generic model's
`sample_batched(..., temperature=...)` path.

Verification:

```bash
python -m py_compile experiments/backfill/block_ar/evaluate_609a_unified_ar_transition_flow.py
pytest test_code/test_609a_generic_empirical_score_transition.py -q
```

Focused tests: `2 passed`.

## Runs

Both diagnostics use the same checkpoint:

- `models/backfill/610a_joint38_ar_transition_e8_w2048_s610/best_model.pt`.

Temperature `1.05`:

- artifact: `results/autoresearch/611a_610a_temperature_diagnostic/temp105_full11.json`;
- score: `4/11`.

Temperature `1.15`:

- artifact: `results/autoresearch/611a_610a_temperature_diagnostic/temp115_full11.json`;
- score: `4/11`.

Baseline 610a temperature `1.00`:

- artifact: `results/autoresearch/610a_joint38_ar_transition_e8_w2048/full11.json`;
- score: `5/11`.

## Comparison

| metric | temp 1.00 | temp 1.05 | temp 1.15 |
|---|---:|---:|---:|
| score | `5/11` | `4/11` | `4/11` |
| cov90 overall | `75.7%` | `76.9%` | `78.6%` |
| conditional MAE reduction | `5.0%` borderline | `6.0%` | `4.1%` |
| turb/calm width ratio | `1.009` | `1.023` | `1.019` |
| daily-change KS | `24/25` | `23/25` | `22/25` |
| level KS | `2/25` | `2/25` | `1/25` |
| median-bias cells | `11/25` | `9/25` | `7/25` |
| cointegration worst-cell | `0.328` pass | `0.200` fail | `0.377` pass |
| cross-cell corr/rank | `0.880 / 1.492` | `0.803 / 1.656` | `0.647 / 2.093` |
| pathwise max-jump KS | `0.446` pass | `0.379` pass | `0.226` pass |
| per-cell q99 jump-scale | `21/25` pass | `20/25` pass | `12/25` fail |
| persistent severe undercoverage | `11.0%` | `10.9%` | `10.7%` |

## Mechanism Read

Global temperature is not the missing risk layer.

It increases aggregate coverage only slightly:

- `75.7% -> 76.9% -> 78.6%`.

But the hard sparse-window/regime problem barely moves:

- persistent severe undercoverage stays around `11%`;
- regime layer2 remains `0/8`;
- turbulent/calm width stays close to `1.0`, not the desired risk-sensitive widening.

At the same time, temperature damages structure:

- `1.05` loses the cointegration worst-cell gate;
- `1.15` breaks per-cell q99 jump-scale and drops pathwise suite;
- median bias and level KS get worse;
- cross-cell rank drifts upward.

The failure is therefore not "all samples are too narrow." It is state-dependent:
the model needs more width and different path placement in specific regimes/cells,
while preserving local tails and dependence elsewhere.

## Decision

Close scalar temperature as a deployability layer for 610a.

The next model-side move should learn conditional source scale inside the AR
transition, using the same generic panel architecture. It should be a single
generic mechanism:

- infer a positive per-variable source scale from the causal memory state;
- apply it to transition source noise during training and sampling;
- keep IV-only and joint38 paths identical except for `D`;
- evaluate whether turbulent/sparse histories widen without globally damaging
  cointegration, cross-cell structure, and per-cell jump tails.
