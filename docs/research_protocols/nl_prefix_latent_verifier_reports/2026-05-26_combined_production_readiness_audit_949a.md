# Independent Verifier: Combined Production-Readiness Audit 949a

Date: 2026-05-26

## Verification Result

Verdict: `PARTIAL`

The 949a audit correctly combines the current 947b held-out/five-start
calibration evidence with the 948d live fixed-start story-deck evidence. It
supports the claim that the narrative-conditioned ensemble calibration is a
paper/demo candidate with all hard evidence gates passing. It does not support
marking the full goal complete or promoting the method as a silent production
default.

## What I Checked

- Audit code:
  `experiments/backfill/block_ar/nl_production_readiness_audit.py`.
- Audit tests:
  `test_code/test_949a_nl_production_readiness_audit.py`.
- Held-out/five-start calibration artifact:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_947b_full906b_66w_5start_broad_support_deck_matched_seed/narrative_ensemble_calibration_report.json`.
- Live fixed-start story-deck artifact:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_gradio_live_story_deck_948d_fixed_start22_calibrated_snapshots/fixed_start22_calibrated_story_deck_conditionality_summary.json`.
- Generated combined audit:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_production_readiness_audit_949a/production_readiness_audit.json`.
- Markdown summary:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_production_readiness_audit_949a/production_readiness_audit.md`.

## Findings

### Confirmed Correct

- The audit reads 947b and 948d directly rather than relying on research-log
  prose.
- Held-out quality passes:
  - calibrated CRPS delta vs identity: `-0.001810282469`;
  - calibrated energy delta vs identity: `-0.002892835072`;
  - calibrated coverage delta vs identity: `+0.001010101010`;
  - CRPS improvement vs persistence: `0.216852726562`;
  - energy improvement vs persistence: `0.306246897599`;
  - 80% coverage: `0.822636622637`.
- The 947b promotion gates all pass.
- Start-normalized narrative plus interaction is `0.5239292248283334`, above
  the baseline and above the audit threshold.
- Fixed-start distribution gate passes:
  - factor KS `0.3283035714285714`;
  - portfolio KS `0.4110416666666667`;
  - support Jaccard `0.02972110117190209`.
- Live demo support conditionality passes:
  - 948d status `ok`;
  - `6/6` cases passed;
  - calibration applied in `6/6`;
  - minimum calibration support gate `1.0`;
  - maximum pairwise support Jaccard `0.0`.
- Qualitative evidence gate now checks that artifact paths and live snapshots
  actually exist, not just that strings are present.
- Tests passed:
  `uv run pytest test_code/test_949a_nl_production_readiness_audit.py -q`
  reported `3 passed`.

### Issues Found

- `WARNING`: The audit result is `paper_demo_candidate`, not
  `production_ready`. The `production_default` gate is intentionally warning
  level because the method is still a bounded post-rollout support-gated
  calibration layer over the frozen SNI rollout.
- `WARNING`: The audit combines two complementary surfaces: 947b for held-out
  quality and five-start conditionality, 948d for live demo-path validation.
  This is stronger than either surface alone, but it is not a single all-in-one
  production load test.
- `NOTE`: The audit is appropriate as a current promotion/control dashboard. It
  should be rerun whenever paper/demo figures, live UI defaults, or calibration
  parameters change.

## Alternative Explanations

- The passing live support Jaccard gate demonstrates support-selection
  conditionality, but factor-fan visual separation still depends on the bounded
  calibration and the frozen SNI rollout. The paper/demo should continue to use
  portfolio and relevant-factor readouts rather than implying every single
  marginal fan will be visually disjoint.
- Strong held-out CRPS/energy performance comes from the support-grounded SNI
  ensemble plus calibration; it is not evidence for a standalone direct
  text-to-scenario generator.

## My Independent Assessment

949a is the right current status artifact. It upgrades the project from a
collection of separate positive results to a single gate-level evidence packet:
held-out quality, promotion gates, start-normalized response, fixed-start
factor/portfolio separation, live support conditionality, and qualitative
artifact existence all pass. The only warning is the intended one: do not call
the system a silent production default yet.

## Recommended Action

Use 949a as the current production-readiness checkpoint. The next work should
either synchronize paper/demo figures and UI copy with the 949a status or run a
broader multi-start live UX sweep if the goal is to move from paper/demo
candidate toward production default.
