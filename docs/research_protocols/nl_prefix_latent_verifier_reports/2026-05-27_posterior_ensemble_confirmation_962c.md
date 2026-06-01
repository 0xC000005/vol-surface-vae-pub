# Independent Verification: Posterior-Ensemble Confirmation 962c

Date: 2026-05-27

## Verification Result

Verdict: `PARTIAL`

The high-sample confirmation supports the **Nearest similar regimes /
Main-regime view** family, but it does **not** support promoting the earlier
low-sample **top 3 / 90%** view as the single paper/demo default. The current
artifact-backed presentation candidate is **top 2 / 80%**. Top 3 / 90% remains
a calibration-leaning main-regime variant, and all-regime pooling remains the
calibration diagnostic.

## What I Checked

- Workflow truth and plan:
  - `docs/research_protocols/nl_prefix_latent_current_truth.md`
  - `docs/research_protocols/nl_prefix_latent_autoresearch_plan.md`
  - `docs/research_protocols/nl_prefix_latent_goal.json`
  - `.agents/skills/nl-prefix-latent-autoresearch/SKILL.md`
- High-sample fixed-start rollout and posterior artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962a_start22_s384_d400/fixed_start_rollout_policy_comparison.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962a_start22_s384_d400/component_posterior_bakeoff_all_modes.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962a_start22_s384_d400/component_posterior_bakeoff_all_modes.md`
- Higher-sample historical quality guardrail:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962b_guardrail_cohesive_66req_s64_d250/component_backtest_report.json`
- Selection report:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962c_selection_confirmation/posterior_ensemble_selection_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962c_selection_confirmation/posterior_ensemble_selection_report.md`
- Paper assets:
  - `paper/narrative_grounded_scenarios/figures/posterior_ensemble_candidate_fans_962a.png`
  - `paper/narrative_grounded_scenarios/generated_tables/table_posterior_ensemble_confirmation.tex`
  - `paper/narrative_grounded_scenarios/main.tex`

## Confirmed Correct

- The high-sample fixed-start posterior bakeoff was generated with 384
  presentation samples for the pooled fixed-start deck, and the start-only null
  remains flat across all posterior views: support Jaccard `1.000`, factor KS
  `0.000`, portfolio KS `0.000`, path energy `0.000`, and VaR95 range `0.000`.
- The nearest-similar support family creates nonzero fixed-start narrative
  separation:
  - all selected regimes: factor KS `0.110`, portfolio KS `0.115`;
  - top 1: factor KS `0.186`, portfolio KS `0.152`;
  - top 2 / 80%: factor KS `0.187`, portfolio KS `0.177`;
  - top 3 / 90%: factor KS `0.165`, portfolio KS `0.149`.
- The higher-sample historical guardrail for the full component-prefix mixture
  is positive versus persistence: CRPS improvement `+0.2146`, energy
  improvement `+0.3086`, and 80% coverage `0.8258` across 29 scored windows.
- The 962c selection report ranks **Nearest similar regimes / Main-regime
  view: top 2 / 80%** first. Its metrics are CRPS improvement `+0.189`, energy
  improvement `+0.281`, 80% coverage `0.778`, factor KS `0.187`, and portfolio
  KS `0.177`.
- The 962c report ranks **top 3 / 90%** second. It is better calibrated than
  top2/80 on CRPS, energy, and coverage, but weaker on factor and portfolio
  separation.

## Issues Found

- `WARNING`: The earlier 961b top3/90 promotion relied on a low-sample
  conditionality deck. The high-sample deck materially reduces the conditionality
  numbers and changes the selected tradeoff candidate to top2/80.
- `WARNING`: The high-sample conditionality is real and above the start-only
  null, but still moderate. It should be framed as improved narrative
  separation through main-regime posterior presentation, not as fully solved
  risk-manager-visible conditionality.
- `NOTE`: All-regime pooling remains the best pure calibration diagnostic, but
  it smooths the narrative signal. The paper/demo should show this as a tradeoff
  rather than as a failure.

## Alternative Explanations

- The stronger low-sample 961b conditionality was likely amplified by the small
  posterior sample count. The 962 high-sample run is a better estimate of visual
  and distributional separation.
- The same accepted starting level still anchors raw path geometry. The
  posterior view improves how narrative-specific support components are shown,
  but does not remove the SNI generator's calibrated local market dynamics.

## My Independent Assessment

Proceed with the **Nearest similar regimes / Main-regime view** family, but do
not claim that the exact top3/90 view is confirmed as the finished default.
Use **top2/80** as the current high-sample product-facing presentation candidate
if a single view is needed. Keep **top3/90** as a calibration-leaning variant and
**all selected regimes** as the calibration diagnostic. The paper/demo should
make this tradeoff explicit.

## Recommended Action

- Update workflow truth and paper text to name top2/80 as the current
  high-sample presentation candidate.
- Keep the confirmation table in the paper so readers see the calibration
  versus conditionality tradeoff.
- Do not call the result production-ready or conditionality-solved. It is
  paper/demo evidence for a stronger, auditable presentation candidate.
