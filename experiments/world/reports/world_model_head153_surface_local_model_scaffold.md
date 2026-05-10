# World Model HEAD153: Surface-Local Token JEPA Model Scaffold

Date: 2026-05-10

## Iteration Type

`experiment`

## Objective Family

`token_geometry_level_context_to_target_jepa` model/loss scaffold.

## Hypothesis

The surface-local data contract can support a minimal token/geometry-level JEPA
loss that selects clean target-encoder outputs at explicit `(window,
relative_time, token)` target positions, without future targets, raw-value
reconstruction, decoder loss, or another row-level shortcut.

## Execution

Used TDD:

- red: `pytest test_code/test_world_model_surface_local_jepa_model.py -q`
  failed on missing `surface_local_jepa_model`;
- green: added
  `experiments/world/part1_jepa_latent/surface_local_jepa_model.py` and the
  focused test.

The scaffold includes:

- `SurfaceLocalTokenJepaConfig`;
- `SurfaceLocalTokenEncoder`, which combines value, observed flag, visible flag,
  relative time, and token descriptors before a per-token temporal GRU;
- `SurfaceLocalTokenJepaModel`, with trainable context encoder, frozen clean
  target encoder initialized from the context encoder, and a small predictor;
- `select_target_token_rows`, which selects only explicit target positions;
- `surface_local_context_target_loss`, which applies alignment plus Barlow-style
  redundancy control to selected latent target tokens.

## Verification

`pytest test_code/test_world_model_surface_local_jepa_model.py -q`

Result: `2 passed`.

## Guardrails

- Target rows are selected by explicit `(window, relative_time, token)` indices.
- The target encoder receives the clean observed window, not target-only values.
- No future target is consumed.
- No raw-value reconstruction loss is introduced.
- No decoder or Part 2 objective is introduced.
- This is not a trained candidate and does not change the active scaled Barlow
  reference.

## Decision

Model/loss scaffold only. The next safe step is one small smoke training run
plus representation/target-token health diagnostics. Do not promote Part 1,
start Part B, or tune architecture knobs from this scaffold alone.
