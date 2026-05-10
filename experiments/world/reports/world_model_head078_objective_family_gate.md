# World Model HEAD078: Objective-Family Gate

## Objective Family

`masked_multiview_invariance`.

The current Part 1 branch is two structured corrupted views of the same market
window and same relative index, scored on the encoder embeddings themselves. It
is not a future-forecasting objective and not a default EMA/predictor
context-to-target objective.

## Literature Status

`supported_adjacent`.

The route is supported by two-view redundancy-reduction and
variance/covariance SSL families such as Barlow Twins and VICReg. Classic
context-to-target JEPA with EMA/stop-gradient target encoders remains relevant
for a different objective family, but it is not the default route for this
same-state two-corruption branch.

## Context

HEAD066 showed the failure mode of putting a JEPA-style EMA/predictor between
the two-view masked objective and the representation surface we actually score:
predictor-target cosine and MSE looked healthy, while retrieval and rank
collapsed. HEAD068 and HEAD070 showed the cleaner direct two-view Barlow route
is a better Part 1 reference candidate because the invariance and redundancy
terms act on the evaluated encoder embeddings.

## Change

Updated the workflow guardrails so future iterations must classify the Part 1
proposal before code changes:

- `masked_multiview_invariance`: default current branch; shared encoder,
  direct encoder-output comparison, Barlow/VICReg-style health controls.
- `context_to_target_jepa`: separate branch; EMA/stop-gradient targets and
  predictors are allowed only after this family is explicitly selected.
- `downstream_probe`: future range, future mean, state labels, retrieval, and
  generation are evaluation tasks after pretraining.

The protocol now also requires a representation-surface rule: collapse
prevention must attach to the embeddings being evaluated, and high cosine/low
MSE alone cannot certify Part 1.

## Files Updated

- `.agents/skills/world-model-autoresearch/SKILL.md`
- `docs/research_protocols/world_model_autoresearch_plan.md`
- `experiments/world/README.md`
- `experiments/world/part1_jepa_latent/README.md`
- `experiments/world/reports/README.md`
- `autoresearch-session/world_model_goal.json` (ignored local state)

## Decision

Keep HEAD070 as the current Part 1 reference candidate. The next autoresearch
step remains the compact Part 1 scorecard reader/report before adding more
model changes.
