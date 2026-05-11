# World Model HEAD178: Factor-Panel Probe Scaffold

Date: 2026-05-11

## Iteration Type

`experiment`

## Objective Family

`downstream_probe_scaffold`; this adds factor-panel future targets for frozen
evaluation only. It does not change Part 1 pretraining and does not start Part
B decoder work.

## Hypothesis

The missing factor-panel future-probe coverage identified in HEAD177 can be
addressed first with a small reusable data/target scaffold before any model
objective change.

## Falsifier

The scaffold fails if it cannot build split-safe factor-panel history/future
windows from `multi_factor_data.npz`, or if it turns future targets into a
pretraining loss.

## Implementation

- Added `experiments/world/evaluation/factor_panel_data.py`.
- Added `test_code/test_world_model_factor_panel_probes.py`.
- New data surface:
  `build_factor_panel_world_windows(...) -> FactorPanelWindowBatch`.
- New downstream target surface:
  `make_factor_panel_future_targets(...)`.

## Target Contract

Regression targets:

- `factor_future_mean_delta`
- `factor_future_range`
- `factor_future_terminal_delta`
- `factor_future_max_abs_step`

Metadata marks the scope as
`factor_panel_future_downstream_probe_only`.

## Verification

- Red test first: `ModuleNotFoundError` for
  `experiments.world.evaluation.factor_panel_data`.
- Focused test: `python -m pytest test_code/test_world_model_factor_panel_probes.py -q`
  passed.
- Compile check: `python -m py_compile experiments/world/evaluation/factor_panel_data.py`
  passed.
- Real-data smoke:
  `past=(4, 30, 28)`, `future=(4, 30, 28)`, `columns=28`, target keys
  `factor_future_max_abs_step`, `factor_future_mean_delta`,
  `factor_future_range`, `factor_future_terminal_delta`.

## Decision

The factor-panel future-probe scaffold is ready for a follow-up frozen probe
bakeoff. It is not Part 1 promotion evidence by itself.

Part 1 remains `DO_NOT_PROMOTE`; Part B remains blocked.
