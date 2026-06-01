# Independent Verification: Five-Start Matched Broad-Support Calibration 947b

Verification Result

Verdict: `PARTIAL`

The scoped claim is supported: `947b` is stronger paper/demo evidence than
`946d` for fixed-start acceptance because it keeps the same full906b `66/66`
held-out broad component backtest and extends the matched broad-support
fixed-start calibration deck from three starts to five starts (`0`, `18`, `22`,
`40`, and `77`) under one seed/settings deck. It should supersede `946d` for
paper/demo conditionality tables and plots.

The claim remains scoped. This is still a bounded post-rollout support-gated
calibration layer over the frozen support-grounded SNI ensemble. It is not a
direct text-to-scenario model and not a silent production default.

What I Checked

- Source code:
  - `experiments/backfill/block_ar/nl_narrative_ensemble_calibration.py`
  - `test_code/test_945a_nl_narrative_ensemble_calibration.py`
- Backtest artifact:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_component_backtest_946c_broad_full906b_66w_s32_d200/component_backtest_report.json`
- Five fixed-start decks:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_947a_start0_broad_replay_s64_d400/fixed_start_rollout_policy_comparison.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_947a_start18_broad_replay_s64_d400/fixed_start_rollout_policy_comparison.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_947a_start22_broad_replay_s64_d400/fixed_start_rollout_policy_comparison.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_947a_start40_broad_replay_s64_d400/fixed_start_rollout_policy_comparison.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_947a_start77_broad_replay_s64_d400/fixed_start_rollout_policy_comparison.json`
- Calibration artifact:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_947b_full906b_66w_5start_broad_support_deck_matched_seed/narrative_ensemble_calibration_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_947b_full906b_66w_5start_broad_support_deck_matched_seed/support_gated_qualitative_review.json`
- Paper surface:
  - `paper/narrative_grounded_scenarios/main.tex`
  - `paper/narrative_grounded_scenarios/generated_tables/table_support_gated_calibration.tex`
  - `paper/narrative_grounded_scenarios/figures/support_gated_narrative_relevant_raw_panels_947b.png`
  - `paper/narrative_grounded_scenarios/figures/support_gated_start_only_null_contrasts_947b.png`

Findings

Confirmed Correct

- The full906b broad component backtest artifact has status `ok` and scores all
  `66/66` requested held-out anchors.
- `947b` uses the full906b broad component backtest and selected beta `0.25`
  through the quality-constrained response rule.
- Held-out calibrated-minus-identity deltas remain favorable: CRPS
  `-0.001810282469`, energy `-0.002892835072`, and coverage `+0.001010101010`.
- The five-start calibrated attribution improves fixed-start narrative response:
  raw-level start share `77.1%`, narrative plus interaction `22.9%`, factor KS
  `0.328`, portfolio KS `0.411`, and support Jaccard `0.030`.
- In start-normalized space, calibrated narrative plus interaction rises to
  `52.4%`, with start share `47.6%`.
- Every per-start gate passes. The weakest calibrated per-start floor is factor
  KS `0.311`, portfolio KS `0.373`, and support Jaccard `0.032`.
- The start-only null remains a valid control: it selects identical support
  across narratives and produces zero cross-narrative factor/portfolio
  separation in each fixed-start deck.
- The paper table and figure references now point to the 947b artifacts rather
  than the older 946d three-start figures.

Issues Found

- `WARNING`: Starts `0`, `18`, `22`, and `77` show warning statuses in several
  individual story-smoke reports even though their aggregate fixed-start
  conditionality gates pass. This does not invalidate the fixed-start
  conditionality claim, but product UI should still expose those run warnings
  instead of silently treating every start as equally clean.
- `WARNING`: The 947b qualitative raw-panel plots still use one qualitative
  start label for the detailed factor panels, while the broader five-start
  evidence appears in the attribution and per-start gate table. This is
  acceptable for the short paper if the text says the five-start evidence is
  tabular/diagnostic and the raw panels are illustrative.
- `NOTE`: The 947b method is a bounded calibration/readout layer. It strengthens
  narrative-visible response but does not prove that the frozen SNI decoder is
  directly language-conditioned.

Alternative Explanations

- Part of the improved factor and portfolio separation comes from the bounded
  support-gated calibration layer, not only from support selection. That is the
  intended mechanism, but the paper should describe it as calibration over an
  auditable support-grounded ensemble, not as pure retrieval or pure text
  generation.
- Start level remains a major driver of raw-level path geometry. The five-start
  result reduces the concern that the conditionality is a single favorable
  start artifact, but it does not eliminate start dominance in raw levels.

My Independent Assessment

`947b` is the best current paper/demo evidence surface. It fixes the earlier
mixed-evidence and three-start limitations while preserving the 66-window
full-manifest quality check. It supports the claim that professional narratives
change auditable support mixtures and generated future risk distributions under
fixed starts. It does not support a stronger claim that the system is already a
silent production default or a direct prompt-to-scenario generator.

Recommended Action

- Use `947b` as the current paper/demo candidate.
- Keep the production default claim scoped to support-grounded narrative
  scenario generation with exposed warnings and provenance.
- If moving from paper/demo candidate to production default, next run a
  user-facing demo smoke using the same default path and record whether warning
  visibility, support provenance, and factor/portfolio readouts are
  understandable without research diagnostics.
