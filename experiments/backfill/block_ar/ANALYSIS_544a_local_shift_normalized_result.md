# 544a Local Shift-Normalized AR Flow Result

## Hypothesis

The remaining local-data idea from the ShifTS/DLinear/NLinear direction was to change only
the coordinate frame:

1. compute a causal center from the last observed history level,
2. compute a causal local scale from trailing realized history moves,
3. model the future path in this local normalized frame with the same empirical-score AR
   flow family,
4. invert generated paths back to IV levels using the history-only center and scale.

This was intended to attack future level occupancy and regime-width behavior without adding
graph, frequency, expert, retrieval, or calibration modules.

## Implementation

Added `LocalShiftNormalizedEmpiricalScoreTransitionFlowMatching` and trained it from
scratch as `544a`:

- model: `diffusion/block_ar/local_shift_normalized_empirical_score_transition_flow_matching.py`
- train script: `experiments/backfill/block_ar/train_544a_local_shift_normalized_ar_flow.py`
- checkpoint: `models/backfill/544a_local_shift_normalized_ar_flow_s544/best_model.pt`
- official eval: `results/autoresearch/544a_local_shift_normalized_ar_flow_s544/full11.json`

The transform is causal: center and scale are computed only from the 30-day history window.
The model also receives the removed center and log-scale as repeated conditioning features,
because otherwise the normalization would erase absolute level and local volatility state.

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

- overall 90% coverage: `0.869`, but per-cell overcoverage above the 95% cap remains
- conditional MAE reduction: `4.3%`, just below the `>5%` gate
- turbulent/calm width ratio: `1.107`
- ACF correlation: `0.956`
- kurtosis ratio: `1.686`
- daily-change KS cells: `25/25`
- level KS cells: `4/25`
- cross-cell correlation ratio: `1.004`
- rank ratio: `1.356`
- mean-reversion aggregate ratio: `0.360`
- pathwise max-jump KS: `0.359`
- per-cell extreme-jump scale cells: `21/25`

## Mechanism Read

The causal local frame is useful but not sufficient. It strongly improves daily-change
shape, cross-cell dependence, path jump realism, and aggregate coverage, but it does not
recover the unconditional future IV-level occupancy demanded by level KS. It also weakens
mean reversion: generated paths move in the correct broad direction but with only about
one third of the empirical reversion strength.

The common failure pattern is now cleaner:

- the model is too good at locally scaled motion realism,
- but too weak at conditional long-horizon level allocation and reversion strength,
- and broad local scaling creates cell-specific overcoverage even when aggregate coverage
  looks calibrated.

## Decision

Do not sweep local scale floors, temperatures, or feature modes. `544a` is below the
`392a`/`510a` `8/11` learned frontier and confirms that a pure causal shift-normalized
coordinate frame is not enough.

The next principled HEAD step should not stack another paper module. It should perform
post-experiment ideation around the specific residual gap: how to learn long-horizon
conditional level allocation/reversion without returning to hand-engineered center/residual
or policy calibration.
