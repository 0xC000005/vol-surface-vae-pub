# 576a Unified Increment Panel Audit

## Context

575a decided that the publication path should be a native joint IV-plus-anchor
factor law, not a composed IV stress deck plus factor overlay. Before training a
new model, the data contract needs to prove that all factors can be represented as
one reversible future-increment tensor.

## Implementation

Added:

- `experiments/backfill/block_ar/audit_576a_unified_increment_panel.py`;
- `test_code/test_576a_unified_increment_panel.py`.

The audit builds one state-variable list:

- 25 IV cells;
- 13 anchor factor levels;
- total target state variables: 38.

The original source panel has 51 channels because it also contains 13 factor
return/diff columns. Those return/diff columns are no longer future targets. They
are kept only as a diagnostic reference against level-implied canonical
increments.

## Artifact

Command:

```bash
python experiments/backfill/block_ar/audit_576a_unified_increment_panel.py \
  --future_lens 30 60 90 152 \
  --output results/autoresearch/576a_unified_increment_panel/audit.json
```

Output:

- `results/autoresearch/576a_unified_increment_panel/audit.json`.

## Results

Source panel:

- observations: `5822`;
- source channels: `51`;
- target state variables: `38`;
- date range: reported in the JSON artifact.

Split and shape checks:

| Horizon | Train windows | Val windows | Future increment shape | No test leakage |
|---:|---:|---:|---|---|
| 30 | 4010 | 441 | `[441, 30, 38]` on validation | true |
| 60 | 3980 | 441 | `[441, 60, 38]` on validation | true |
| 90 | 3950 | 441 | `[441, 90, 38]` on validation | true |
| 152 | 3888 | 441 | `[441, 152, 38]` on validation | true |

Core framing checks:

- no duplicate target names: true;
- no return-like target channels: true;
- finite increment rate: `1.0`;
- finite reconstruction rate: `1.0`;
- reconstruction max absolute error: `0.0` for all audited horizons.

Diagnostic reference check:

- existing factor return/diff columns are not perfectly identical to
  level-implied canonical increments;
- validation max absolute reference mismatch is about `0.08000016`;
- largest mismatches are in `us2y`, `us10y`, `crude_oil`, `usdjpy`, and `usdcad`.

This is not a blocker for 576a because the canonical target is deliberately
level-implied and reversible. The old return/diff columns should not be treated as
authoritative duplicated targets.

## Verification

Focused tests:

- `pytest test_code/test_576a_unified_increment_panel.py -q` -> `3 passed`.

## Decision

The unified data framing is viable.

The next trainable experiment should be a small V0 shared generative model over
the `(horizon, 38)` canonical future-increment tensor. Do not reintroduce separate
IV and factor model heads at this stage.
