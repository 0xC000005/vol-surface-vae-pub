# 605a Hard-Window Support Audit

## Question

After 602a-604a closed postforecast interval adapters, the remaining question is whether
the hard validation stress windows are simply outside the observable conditioning support,
or whether their realized future moves are extreme but in-support. 605a audits the hard
windows repeatedly identified by the broad-frame suite, especially the persistent
undercoverage around cell `(3,3)`.

## Method

Added `experiments/backfill/block_ar/audit_605a_hard_window_support.py` and focused tests
in `test_code/test_605a_hard_window_support_audit.py`.

The audit uses no model sampling. It compares hard validation windows against the training
set using standardized nearest-neighbor distances in:

- IV history summary features;
- factor history summary features from `ret`, `price`, `slopes`, `skews`, `levels`;
- joint IV+factor history features.

It also compares the realized h30 delta for hard cell `(3,3)` and a broad max-path
severity score against the training future distribution.

Focused tests passed:

```text
pytest test_code/test_605a_hard_window_support_audit.py -q
4 passed in 1.34s
```

## Result

Hard validation windows audited: `420`, `421`, and `423`.

| Window | IV NN pct val/train | Factor NN pct val/train | Joint NN pct val/train | h30 `(3,3)` delta | h30 delta train pct |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 420 | `0.719 / 0.961` | `0.757 / 0.820` | `0.719 / 0.948` | `-0.01404` | `0.219` |
| 421 | `0.717 / 0.956` | `0.705 / 0.812` | `0.714 / 0.942` | `-0.01146` | `0.271` |
| 423 | `0.798 / 0.992` | `0.927 / 0.879` | `0.800 / 0.988` | `-0.00580` | `0.412` |

Training hard-cell delta distribution p05/p50/p95:

```text
[-0.03173, -0.00231, 0.03921]
```

Artifacts:

- `results/autoresearch/605a_hard_window_support/audit.json`
- `results/autoresearch/605a_hard_window_support/audit.md`

## Mechanism Read

The hard windows are near the upper tail of training-history distance, especially in IV
and joint IV+factor space: roughly `94%-99%` versus train nearest-neighbor support. This
means the model is extrapolating from relatively weak local support.

But the realized h30 `(3,3)` moves are not extreme relative to training futures: they sit
around the `22%-41%` training percentile. So the failure is not an impossible unseen
amplitude. It is a conditional path-location problem under weak history support: the
generator is putting the center/tail mass in the wrong place for an OOD-ish history state.

This explains why 602a-604a failed. Widening the residual envelope cannot solve a
conditional location/allocation error when the relevant history state is near the edge of
training support.

## Decision

Do not run another postforecast width adapter. The next research branch, if any, must
alter conditional path-location allocation under low-support states. The honest alternative
is to document a data requirement: add state variables that make these edge regimes
locally supported or predictable before claiming deployable conditional risk scenarios.
