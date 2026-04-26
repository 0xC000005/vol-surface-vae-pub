# 569a Factor Panel Readiness

## Context

The user asked whether the current IV-only risk prototype can extend to a broader factor stack and longer horizons. 569a tests only the first part: whether the local data can be stacked into no-lookahead factor panels suitable for a later joint learned law.

This is a data-framing probe, not a training run.

## Implementation

Added:

- `experiments/backfill/block_ar/audit_569a_factor_panel_readiness.py`
- `test_code/test_569a_factor_panel_readiness.py`

The audit builds two panels:

- local stack: `25` IV channels plus `ret`, `price`, `slopes`, `skews`, `levels`;
- broad stack: `25` IV channels plus `13` factor levels and `13` factor returns/diffs.

The broad factor list is:

- levels: `spx`, `usdcad`, `usdjpy`, `dxy`, `copper`, `wheat`, `crude_oil`, `us2y`, `us10y`, `aaa_oas`, `bbb_oas`, `nikkei`, `gold`;
- returns/diffs: `spx_logret`, `usdcad_logret`, `usdjpy_logret`, `dxy_logret`, `copper_logret`, `wheat_logret`, `crude_oil_logret`, `us2y_diff`, `us10y_diff`, `aaa_oas_diff`, `bbb_oas_diff`, `nikkei_logret`, `gold_logret`.

## Verification

Focused tests:

- Red check: failed because the module did not exist.
- Final command: `pytest test_code/test_569a_factor_panel_readiness.py -q`
- Result: `4 passed`.

Audit artifacts:

- `results/autoresearch/569a_factor_panel_readiness/readiness.json`
- `results/autoresearch/569a_factor_panel_readiness/readiness.md`

## Result

Panel shapes:

- local panel: `(5822, 30)`, finite rate after preprocessing `1.0`;
- broad panel: `(5822, 51)`, finite rate after preprocessing `1.0`.

Raw broad-factor missingness:

- factor levels finite rate before fill: `0.9909`;
- factor returns/diffs finite rate before fill: `0.9878`;
- missing values are concentrated in later-starting commodities/gold and several FX/rate series.

The audit uses a simple per-column backfill-then-forward-fill for readiness only. A final trained joint model should either keep this deterministic preprocessing or replace it with a formal missingness treatment.

No-lookahead window readiness:

| future days | train windows | val windows | last val future end | no test leakage |
| ---: | ---: | ---: | ---: | --- |
| `30` | `4010` | `441` | `4511` | `True` |
| `60` | `3980` | `441` | `4511` | `True` |
| `90` | `3950` | `441` | `4511` | `True` |
| `252` | `3788` | `441` | `4511` | `True` |

## Decision

Factor stacking is mechanically feasible for both the local 30-channel panel and the broad 51-channel panel. The next factor-model step should be a true joint panel generator over the 51-channel panel, not another IV-only wrapper.

This audit does not prove that such a model will learn the joint law. It only removes the data-framing blocker and exposes the missing-value preprocessing requirement.
