# 546a Local-Shift Raw-IV Energy Fine-Tune Result

## Hypothesis

545a proposed that 544a failed because it trained mostly in a local normalized coordinate
while the deployment target is the raw 30-day IV-level path law. 546a therefore kept the
same single-stage 544a generator and added a differentiable raw-IV path energy objective.

The intended effect was:

- keep local-score flow matching for daily move realism,
- add raw-IV path-level alignment for level occupancy and mean-reversion strength,
- avoid a separate center/residual architecture or post-hoc calibration wrapper.

## Implementation

Added `experiments/backfill/block_ar/train_546a_local_shift_raw_iv_energy_finetune.py`
and `test_code/test_546a_raw_iv_energy_finetune.py`.

The fine-tune starts from:

- `models/backfill/544a_local_shift_normalized_ar_flow_s544/best_model.pt`

The raw-IV energy score is computed on differentiable 30-day rollouts after inversion from
local normalized coordinates back to raw IV levels.

Full run:

- checkpoint: `models/backfill/546a_local_shift_raw_iv_energy_s546/best_model.pt`
- eval: `results/autoresearch/546a_local_shift_raw_iv_energy_s546/full11.json`

## Result

Full 192-window 11-suite score: `4/11`.

Passed:

- `surface`
- `block_ar`
- `cointegration`
- `cross_cell_correlation`

Failed:

- `coverage`
- `conditionality`
- `time_series`
- `regime_coverage`
- `distributional_fidelity`
- `mean_reversion`
- `pathwise_jump_realism`

Key metrics:

- overall 90% coverage: `0.721`
- conditional MAE reduction: `4.2%`
- turbulent/calm width ratio: `1.194`
- daily-change KS cells: `25/25`
- level KS cells: `14/25`
- cross-cell correlation ratio: `1.071`
- rank ratio: `1.218`
- mean-reversion aggregate ratio: `0.453`
- pathwise max-jump KS: `0.772`
- per-cell jump scale cells: `24/25`

## Mechanism Read

The dual-coordinate objective moved the intended quantities in the right direction but
damaged deployability:

- level KS improved from `4/25` to `14/25`, still just below the `15/25` gate,
- mean-reversion ratio improved from `0.360` to `0.453`, still far below the `[0.70, 1.30]` gate,
- turbulent/calm width ratio improved from `1.107` to `1.194`,
- aggregate coverage collapsed from `0.869` to `0.721`,
- pathwise max-jump KS worsened from `0.359` to `0.772`.

This is not a training failure; it is a tradeoff. Raw-IV path alignment pulls samples toward
realized level occupancy and reversion, but it contracts the conditional support and removes
too much pathwise jump diversity.

## Decision

Close the local-shift plus raw-IV-energy branch unless a new paradigm emerges. The branch
is clean and informative, but below the `392a`/`510a` `8/11` frontier and now shows a
coverage/jump-diversity tradeoff rather than a path to `11/11`.

The next HEAD step should be a paradigm decision, not another energy-weight or epoch sweep.
The unresolved core issue is how to learn long-horizon conditional level allocation and
mean reversion while preserving sufficient conditional support.
