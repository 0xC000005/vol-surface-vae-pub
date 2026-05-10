# World Model HEAD139: Context-Target Model Scaffold

Date: 2026-05-10

## Iteration Type

`experiment`

## Objective Family

`context_to_target_jepa` model/loss scaffold.

## Hypothesis

The same-window context-target data surface can support a minimal latent JEPA
model without future targets or value reconstruction.

## Execution

Added `experiments/world/part1_jepa_latent/context_target_jepa_smoke.py` with:

- `ContextTargetJEPAConfig`;
- `ContextTargetSequenceEncoder`;
- `ContextTargetJEPAModel`;
- `make_context_target_features`;
- `context_target_jepa_loss`.

The model uses a trainable context encoder, frozen target encoder initialized
from the context encoder, and a small predictor. The loss is computed only on
time rows where the current/history target mask hides at least one token.

## Verification

`pytest test_code/test_world_model_context_target_jepa_smoke.py -q`

Result: `1 passed`.

## Guardrails

- No future targets are consumed.
- No value reconstruction loss is introduced.
- The loss aligns latent target rows, not raw values.
- This branch is separate from the scaled Barlow reference.

## Decision

The minimal context-to-target model/loss surface is ready for a smoke training
script. It is not a promoted Part 1 model and has not yet been compared against
scaled Barlow on representation health or exact-state probes.
