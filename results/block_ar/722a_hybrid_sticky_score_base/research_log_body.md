### Context

722a tested the next coordinate repair after 721a showed that no-change atom gates alone are insufficient. The hypothesis was that the sticky channels need a better continuous nonzero-shape coordinate, but global score-coordinate was already rejected by 716a. The implementation therefore added `hybrid_sticky_score`: only channels selected by the train no-change-rate rule use empirical-score flow coordinates; all other channels remain in normalized-innovation coordinates.

### Implementation

- Added `innovation_coordinate="hybrid_sticky_score"` to `GenericStateAwareNormalizedInnovationFlowMatching`.
- Added an `innovation_score_mask` buffer and a setter so score transforms apply only to selected channels.
- Added train-time sticky selection in `train_662a_state_aware_normalized_innovation_flow.py`.
- Added a regression test proving masked channels round-trip through score coordinates while unmasked channels remain unchanged.

### Result

The selector behaved as intended:

- IV-only selected no channels.
- Anchor-only selected `factor:aaa_oas` and `factor:bbb_oas`.
- Joint38 selected `factor:aaa_oas` and `factor:bbb_oas`.

But the coordinate was negative at the base audit level:

- Anchor base: factor KS `11/13`, mean `0.110`; AAA KS `0.202`, BBB KS `0.186`; AAA q99 ratio `4.96`; SPX also failed KS at `0.215`.
- Joint base: factor KS `11/13`, mean `0.142`; AAA KS `0.180`, BBB KS `0.320`; AAA q99 ratio `6.64`; BBB q99 ratio `3.57`; SPX also failed KS at `0.234`.
- Joint IV-factor corr shape remained good at `0.904`, but factor tails and marginal realism were too wide.

Artifacts:

- `diffusion/block_ar/generic_state_aware_normalized_innovation_flow_matching.py`
- `experiments/backfill/block_ar/train_662a_state_aware_normalized_innovation_flow.py`
- `experiments/backfill/block_ar/audit_627a_joint_panel_scenario_quality.py`
- `test_code/test_662a_state_aware_normalized_innovation_flow.py`
- `results/block_ar/722a_hybrid_sticky_score_base/anchor_val_panel_s64.json`
- `results/block_ar/722a_hybrid_sticky_score_base/joint_val_panel_s64.json`

Verification:

- `pytest test_code/test_662a_state_aware_normalized_innovation_flow.py -q` passed: `13 passed`.

### Mechanism Read

Hybrid score is too aggressive for sticky spread channels. It fixes neither atom separation nor tail realism; instead it maps a discrete/ticked distribution with a large atom into a continuous score coordinate and decodes too many extreme spread moves. This explains why global score helped some marginal KS readings before but damaged IV/path realism.

### Decision

Reject hybrid sticky-score as the next deployable coordinate. Do not fine-tune it, because the base coordinate already creates pathological OAS tails. The next HEAD step should be research ideation/postmortem over the sticky-channel data object rather than another threshold or score-coordinate tweak.
