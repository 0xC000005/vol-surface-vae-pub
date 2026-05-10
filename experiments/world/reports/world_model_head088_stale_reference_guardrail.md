# World Model HEAD088: Stale Reference Guardrail

## Iteration Type

`post_experiment_analysis`

## Objective Family

Workflow guardrail for `masked_multiview_invariance`.

## Hypothesis

After HEAD087, workflow-critical docs should route future work to the HEAD070
masked-multiview reference candidate and should not imply that fixed delta-PCA,
future prediction, or predictor heads are the active default.

## Falsifier

The guardrail fails if workflow-critical docs still describe fixed delta-PCA
prediction as the active reference, or if they imply that a predictor head is
the default path for same-state masked views.

## Execution

Searched these workflow-critical surfaces:

- `docs/research_protocols/world_model_autoresearch_plan.md`
- `.agents/skills/world-model-autoresearch/SKILL.md`
- `experiments/world/part1_jepa_latent/README.md`
- `experiments/world/part1_jepa_latent/package_summary.md`
- `experiments/world/part1_jepa_latent/restart_checklist.md`
- `experiments/world/part1_jepa_latent/reference_manifest.json`

## Findings

- No active workflow-critical file still promotes fixed delta-PCA prediction as
  the current reference.
- Remaining fixed-delta-PCA mentions are negative/legacy guardrails.
- The plan had one ambiguous Part 2 handoff phrase, `encoder/predictor`, that
  could invite default predictor use. It now says frozen Part 1
  encoder/reference representation, with predictor use only for explicitly
  selected `context_to_target_jepa`.
- The Part 1 README listed optional variance/covariance controls too
  prominently. It now states the current default HEAD070 loss surface and says
  extra health-control terms need a documented failure.

## Decision

The active restart path now points to HEAD070 masked-multiview invariance with
explicit caveats. Continue only with bounded guardrail/provenance work unless
the user authorizes decoder work or a new Part 1 failure is documented.

## Verification

- `rg` stale-reference scan over workflow-critical files.
- `git diff --check`.
