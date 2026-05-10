# World Model HEAD138: Context-Target Data Scaffold

Date: 2026-05-10

## Iteration Type

`experiment`

## Objective Family

`context_to_target_jepa` data-surface scaffold.

## Hypothesis

A context-to-target diagnostic should first expose the correct same-window data
contract before any model training: context values with masked current/history
target blocks, target values only at those blocks, and no future targets.

## Execution

Added `experiments/world/evaluation/context_target_jepa_data.py` with
`ContextTargetJepaBatch` and `build_context_target_jepa_batch`.

The builder reuses the existing geometry panel loader and typed mask sampler.
It returns:

- `clean_values`;
- `context_values`, where target blocks are hidden;
- `target_values`, nonzero only on observed target blocks;
- `observed_mask`, `context_mask`, and `target_mask`;
- existing geometry token metadata;
- target mask family labels;
- same-window absolute and relative indices.

## Guardrails

- `metadata["objective_family"] == "context_to_target_jepa"`.
- `metadata["uses_future_targets"] is False`.
- `future_len` is used only to preserve the existing split/window contract.
- Target families are the existing typed mask families, not new mask knobs.

## Verification

`pytest test_code/test_world_model_context_target_jepa_data.py -q`

Result: `1 passed`.

## Decision

The minimal same-window data surface is ready for a context-to-target model
smoke. It is still only a scaffold: no model branch is promoted, no future
prediction objective is introduced, and the scaled Barlow reference remains the
comparison point.
