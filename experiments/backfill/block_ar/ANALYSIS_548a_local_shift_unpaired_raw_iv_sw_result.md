# 548a Local-Shift Unpaired Raw-IV SW Result

## Hypothesis

546a showed that paired raw-IV path energy improves level/reversion metrics by contracting
each conditional sample cloud toward its single realized future. 548a tested the cleaner
alternative: keep the 544a local-shift AR generator, keep local-score FM anchoring, and
align generated and realized raw-IV paths as unpaired batch-level joint laws.

## Implementation

Added:

- `experiments/backfill/block_ar/train_548a_local_shift_unpaired_raw_iv_sw_finetune.py`
- `test_code/test_548a_unpaired_raw_iv_sw.py`

The objective aligns the empirical batch law of `(history, future)` raw-IV paths using
sliced Wasserstein projections. This is not paired per-condition path matching.

Full run:

- source: `models/backfill/544a_local_shift_normalized_ar_flow_s544/best_model.pt`
- checkpoint: `models/backfill/548a_local_shift_unpaired_raw_iv_sw_s548/best_model.pt`
- eval: `results/autoresearch/548a_local_shift_unpaired_raw_iv_sw_s548/full11.json`

## Result

Full 192-window 11-suite score: `5/11`.

Passed:

- `surface`
- `block_ar`
- `cointegration`
- `cross_cell_correlation`
- `pathwise_jump_realism`

Failed:

- `coverage`
- `conditionality`
- `time_series`
- `regime_coverage`
- `distributional_fidelity`
- `mean_reversion`

Key metrics:

- overall 90% coverage: `0.820`
- conditional MAE reduction: `3.3%`
- turbulent/calm width ratio: `1.185`
- daily-change KS cells: `25/25`
- level KS cells: `10/25`
- cross-cell correlation ratio: `0.953`
- rank ratio: `1.405`
- mean-reversion aggregate ratio: `0.414`
- pathwise max-jump KS: `0.335`
- per-cell jump scale cells: `22/25`

## Mechanism Read

The unpaired objective behaved as intended relative to 546a: it preserved support and
pathwise jump realism instead of collapsing the sample cloud. However, the improvement in
raw IV level occupancy was too small: level KS moved from 544a's `4/25` to `10/25`, still
below the `15/25` gate and no better than the 392a/510a frontier.

The deeper failure remains:

- local-shift modeling gives good local move realism,
- unpaired raw-IV alignment can nudge aggregate level occupancy,
- but neither creates enough conditional long-horizon reversion strength,
- and the conditionality score remains below the gate.

## Decision

Close the local-shift objective family. Across 544a, 546a, and 548a:

- pure local-shift modeling preserves support but misses level/reversion,
- paired raw-IV path energy improves level/reversion but collapses support,
- unpaired raw-IV SW avoids collapse but is not strong enough to recover frontier.

The next HEAD step should be a postmortem/paradigm decision. Do not continue with SW
weights, projection counts, recent-window sizes, or checkpoint sweeps.
