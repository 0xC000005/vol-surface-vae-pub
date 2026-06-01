# Independent Verification: Full-Window Matched Broad-Support Calibration 946d

Verification Result

Verdict: `PARTIAL`

The scoped claim is supported: `946d` is stronger paper/demo candidate evidence
than `946a` because it uses the full906b bridge-evaluation manifest and scores
all `66/66` held-out anchors before applying the same support-gated calibration
and matched broad-support fixed-start decks. It is acceptable as the current
paper/demo candidate. It should not be described as a silent production default
or as direct text-to-scenario generation.

## What I Checked

- Code:
  - `experiments/backfill/block_ar/nl_prefix_latent_component_backtest.py`
  - `experiments/backfill/block_ar/nl_narrative_ensemble_calibration.py`
  - `test_code/test_945a_nl_narrative_ensemble_calibration.py`
- Artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_component_backtest_946c_broad_full906b_66w_s32_d200/component_backtest_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_946d_full906b_66w_matched_broad_support_deck/narrative_ensemble_calibration_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_946d_full906b_66w_matched_broad_support_deck/support_gated_qualitative_review.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_946a_start18_broad_replay_s64_d400/fixed_start_rollout_policy_comparison.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_946a_start22_broad_replay_s64_d400/fixed_start_rollout_policy_comparison.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_946a_start40_broad_replay_s64_d400/fixed_start_rollout_policy_comparison.json`
- Paper surface:
  - `paper/narrative_grounded_scenarios/generated_tables/table_support_gated_calibration.tex`
  - `paper/narrative_grounded_scenarios/main.tex`
  - `paper/narrative_grounded_scenarios/figures/support_gated_narrative_relevant_raw_panels_946d.png`
  - `paper/narrative_grounded_scenarios/figures/support_gated_start_only_null_contrasts_946d.png`
- Research-log context around `945q-r`, `946a`, and the full906b bridge/report
  mismatch.

## Confirmed Correct

- The full906b component backtest report has status `ok` and scores `66/66`
  requested held-out windows.
- The component rollout quality remains strong versus persistence:
  ensemble CRPS improvement `0.2161`, energy score improvement `0.3045`, and
  80% coverage `0.8063`.
- `946d` selected beta `0.25` with `support_gate_mode=direction_status` and
  `selection_objective=quality_constrained_response`.
- Held-out full906b calibrated-minus-identity deltas are favorable:
  CRPS `-0.001810`, energy `-0.002893`, coverage `+0.001010`.
- Calibration split sizes are `33` calibration rows and `33` evaluation rows.
- Matched broad fixed-start attribution remains materially nonzero:
  factor KS `0.309`, portfolio KS `0.426`, support Jaccard `0.028`.
- Per-start floors pass:
  - start18: factor KS `0.299`, portfolio KS `0.402`, support Jaccard `0.032`;
  - start22: factor KS `0.297`, portfolio KS `0.402`, support Jaccard `0.025`;
  - start40: factor KS `0.331`, portfolio KS `0.473`, support Jaccard `0.028`.
- Start-only null controls remain flat in the fixed-start comparison summaries:
  factor KS `0.0`, portfolio KS `0.0`, support Jaccard `1.0`.
- The paper table and figure references now point to the 946d full-window
  evidence surface.

## Issues Found

- `WARNING`: The fixed-start qualitative decks still reuse the three selected
  starts from 946a. That is appropriate for matched product visual evidence,
  but it is not a fresh six- or ten-start visual replication.
- `WARNING`: This is a bounded post-rollout support-gated calibration layer
  over frozen support-grounded SNI rollouts. It is not a direct text-to-latent
  prior, not a retrained generator, and not an LLM forecast.
- `NOTE`: The 66-window component backtest was run component-only, not with the
  averaged-prefix mode. That is acceptable for this calibration gate because
  the calibration script consumes component-prefix mixture rows, but the paper
  should not use this artifact to make averaged-versus-component claims.

## Alternative Explanations

- Part of the stronger visual separation comes from the bounded directional
  calibration applied after support-gated rollouts. This is not a flaw under
  the current method story, but the paper should keep describing the method as
  support-gated ensemble calibration rather than pure support retrieval.
- The absolute held-out score deltas remain small. The result matters because
  the candidate improves or preserves the distributional floor while producing
  stronger fixed-start narrative response and maintaining support provenance.

## Independent Assessment

`946d` is the best current evidence point for the narrative-conditioned system.
It addresses the prior concern that the broad-support fit/evaluation and the
paper qualitative deck came from different support contexts, and it removes the
29-window caveat by using a true `66/66` full906b held-out component backtest.
The result is strong enough for the paper/demo framing: support-grounded
narrative conditioning with bounded calibration, start-only null controls, and
auditable support provenance.

It is still not enough to say the system is a final production default. The
remaining gap is product governance and breadth of visual acceptance, not the
old conditionality bug.

## Recommended Action

- Promote `946d` as the current paper/demo candidate.
- Keep production-default language scoped: support-gated, bounded,
  support-grounded, frozen-SNI scenario generation.
- Next production-readiness step: either run a broader multi-start visual
  acceptance deck, or update the demo default to 946d and run a user-facing
  smoke pass.
