# World Model HEAD147: Part 1 Candidate Decision Matrix

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`part1_evidence_consolidation`; no model change.

## Hypothesis

After demoting the minimal context-to-target route, the workflow should
have a compact candidate matrix that identifies the active Part 1
candidate and the remaining blocker.

## Candidate Matrix

| candidate | family | current-IV MSE | rank | status | decision |
| --- | --- | ---: | ---: | --- | --- |
| raw_surface_last | raw_baseline_not_part1_model | 0.005630 | 3.241518 | baseline_only | use_as_exact_state_floor_not_representation_candidate |
| scale_barlow_head127 | masked_multiview_invariance | 0.013756 | 18.761132 | best_learned_candidate_do_not_promote | keep_as_active_candidate; representation and corruption pass but baseline superiority, regime, and exact-state gates remain blocked |
| context_target_head140 | minimal_context_to_target_jepa | 0.015289 | 11.592254 | demoted | target latent has mask-family lift 0.467955 and predicted latent is low rank |
| context_target_head144 | minimal_context_to_target_jepa_clean_target | 0.015907 | 8.059559 | demoted | clean-target correction is worse than target-only and scaled Barlow |

## Scale Gate Layers

| layer | status |
| --- | --- |
| representation_health | PASS |
| corruption_robustness | PASS |
| state_content | PARTIAL |
| baseline_superiority | FAIL |
| market_state_regime_probe | FAIL |
| scale_and_stability | PARTIAL |

## Decision

- Active learned candidate: `scale_barlow_head127`.
- Part 1 ready for Part B: `False`.
- Minimal context-to-target demoted: `True`.
- Primary blocker: `exact_state_retention_and_baseline_superiority`.

Allowed next work:
- `bounded evidence consolidation around exact-state gap`
- `new token_geometry_level_jepa_design_gate`
- `quality-gate reconciliation for scaled Barlow`

Blocked next work:
- `Part 2 decoder training`
- `minimal context-to-target knob sweep`
- `future prediction as pretraining objective`
