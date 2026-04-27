# 651a Joint Wide-Panel Data Framing Audit

## Purpose

Before implementing a shared-stochastic multi-head IV + anchor-factor model, we
need to confirm that the training data is a true wide daily market panel. The
model should co-generate same-day IV surface levels and anchor-factor levels or
movements, not separately aligned fragments.

## Audit Command

```bash
python experiments/backfill/block_ar/audit_576a_unified_increment_panel.py \
  --history_len 30 \
  --future_lens 30 60 90 152 \
  --test_start 4511 \
  --val_size 441 \
  --iv_count 25 \
  --clean_nonpositive_log_levels \
  --output results/autoresearch/651a_joint_wide_panel_audit/unified_increment_audit.json
```

## Findings

- Source panel shape: `5822 x 51`.
- Modeled state panel shape: `38` channels.
- The modeled state panel contains `25` IV surface channels and `13`
  anchor-factor state channels.
- The extra `13` source columns are reference return/diff columns; they are not
  duplicated as generated state targets.
- For 30/60/90/152-day horizons, validation windows all end exactly at
  `test_start=4511`; no test leakage was detected.
- Finite rates for history, future state, and future increment were all `1.0`.
- Reconstructing future state from modeled encoded daily increments had max
  absolute error `0.0` for every audited horizon.
- Cleaning was required for stale/nonpositive level-like factor fields:
  copper `167`, wheat `135`, Nikkei `1`, gold `167`. Crude oil was assigned to
  diff-level fallback because it has negative values.

## Reference-Increment Note

The audit reports a reference-increment max discrepancy around `0.08`, mainly
for `us2y` and `us10y`, and smaller discrepancies for `usdjpy`/`usdcad`.
This is not a modeled-target reconstruction failure. The model target is built
from the wide level panel by applying the selected transform and differencing
adjacent daily encoded states. The reference return/diff columns are only
diagnostics. The exact reconstruction result confirms that the model can learn
one-day market movement from the same wide state panel.

## Decision

The data framing is acceptable for the next model:

- IV-only can be obtained by selecting the first 25 channels.
- Joint IV + anchor-factor modeling can use the full 38-channel state panel.
- The next model should keep one shared sampled latent/source path while using
  separate output heads/readouts for IV-surface channels and generic scalar
  anchor-factor channels.
