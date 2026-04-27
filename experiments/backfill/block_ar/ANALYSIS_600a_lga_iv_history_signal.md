# 600a LGA IV-History Signal Audit

## Hypothesis

Recent local-geometry attention ideas suggest that an encoder may benefit from comparing
the current state against locally similar history states rather than only using global
sequence summaries. Before adding a learned LGA-style module to the generator, 600a
tests the cheaper falsifier: can deterministic local-geometry features from IV history
predict the 510a broad-frame failure map better than simple IV-history summaries?

## Method

The audit regenerates broad-frame 510a validation scenarios on 441 windows with 48
samples per window, then builds three feature sets from the conditioning IV history:

- `iv_summary`: the existing 599a summary baseline using last, mean, standard deviation,
  last-minus-mean, and last-delta features.
- `lga`: a deterministic local-geometry proxy where the last IV state queries prior
  history states under a per-window diagonal scale metric.
- `iv_summary_lga`: concatenation of both feature sets.

Each feature set is scored with the same out-of-sample ridge probe against failure
targets: bad-window undercoverage, window coverage, persistent undercoverage count,
median-bias fraction, and level absolute error.

## Result

Focused implementation tests passed:

```text
pytest test_code/test_600a_lga_iv_history_signal.py -q
3 passed in 1.30s
```

The LGA proxy did not improve the important failure targets. Lift versus the IV-summary
baseline was negative for broad undercoverage, coverage, persistence, and level error:

- `bad_window_lt50`: LGA R2 lift `-0.9025`, AUC lift `-0.1235`; summary+LGA R2 lift `-0.5681`.
- `window_coverage`: LGA R2 lift `-1.0685`; summary+LGA R2 lift `-0.9688`.
- `persistent_under_count`: LGA R2 lift `-1.1746`; summary+LGA R2 lift `-0.6780`.
- `level_abs_error`: LGA R2 lift `-3.7238`; summary+LGA R2 lift `-4.1285`.
- `median_bias_frac`: summary+LGA improved R2 by `0.3725`, but median bias is not the
  main hard risk failure and the combined features still had negative absolute R2.

Artifact paths:

- `experiments/backfill/block_ar/audit_600a_lga_iv_history_signal.py`
- `test_code/test_600a_lga_iv_history_signal.py`
- `results/autoresearch/600a_lga_iv_history_signal/audit.md`
- `results/autoresearch/600a_lga_iv_history_signal/audit.json`

## Mechanism Read

The failure mechanism remains signal-limited, not encoder-shape-limited. A local
nearest-history geometry around the last IV state is not enough to recover the windows
where 510a undercovers or misses the level path. It appears to introduce noisy local
matching features that reduce out-of-sample stability relative to simple IV history
statistics.

## Decision

Do not add a learned local-geometry attention block to the generator as the next
architecture change. 600a closes this as an unhelpful feature source for the current
bottleneck. The next principled step should either separate learned-law quality from a
disclosed risk-policy overlay, or run research ideation for new signal/objective framing
rather than adding another attention variant to the same capped family.
