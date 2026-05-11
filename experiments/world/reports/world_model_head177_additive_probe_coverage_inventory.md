# World Model HEAD177: Additive Probe Coverage Inventory

Date: 2026-05-11

## Iteration Type

`post_experiment_analysis`

## Objective Family

`part1_probe_coverage_inventory`; no model change and no Part B decoder work.

## Hypothesis

The current post-temporal bottleneck is not lack of another context-to-target
knob. It is incomplete evidence coverage: additive learned-state value has been
tested mostly on IV-surface future probes, present-state probes, and regime
diagnostics, while factor-panel future probes are still missing from the world
Part 1 gate.

## Falsifier

This inventory would fail if the world Part 1 probe code already produced
factor-panel future targets and compared learned/raw/raw-plus-learned features
against them.

## Local Evidence

- `experiments/world/part1_jepa_latent/masked_multiview_downstream_probe_audit.py`
  defines `make_extended_future_targets(past_surface, future_surface, ...)`.
  Its regression targets are `future_mean_delta`, `future_range`,
  `future_terminal_delta`, `future_max_abs_step`, and `future_drawdown`, all
  derived from IV-surface arrays.
- `experiments/world/evaluation/world_data.py` defines `WorldWindowBatch` and
  `build_iv_world_windows`, loading only `data/vol_surface_with_ret.npz`
  `surface` into 25-cell IV windows.
- Existing package docs already warn that downstream probes target IV-surface
  futures only and that factor-panel future target performance has not been
  claimed.
- Backfill code contains aligned IV+factor panel utilities and joint-panel
  evaluation work, but those are not yet wrapped as a frozen world Part 1 probe
  contract.

## Coverage Gap

Current learned-state evidence is strongest for:

- same-state masked-view retrieval and representation health;
- present-state factor-return/non-surface probes;
- IV path-shape/risk-width probes;
- balanced regime diagnostics.

Current learned-state evidence is incomplete for:

- factor-panel future levels/returns;
- cross-factor future spread or correlation summaries;
- joint IV-plus-factor future state probes;
- held-out horizon/window sensitivity for those factor-panel probes.

## Decision

The next evidence-expanding step should be a probe-contract design or scaffold
for factor-panel future targets, not a new Part 1 pretraining loss and not a
temporal context-to-target knob. Any implementation should keep future targets
as frozen downstream probes only.

Part 1 remains `DO_NOT_PROMOTE`; Part B remains blocked.
