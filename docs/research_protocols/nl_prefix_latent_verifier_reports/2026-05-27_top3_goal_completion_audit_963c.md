# Completion Audit: Top3/90 Posterior-Ensemble Goal After 963c

Date: 2026-05-27

## Objective Audited

Set up and execute the NL prefix-latent autoresearch objective to promote the
selected posterior ensemble candidate: align current-truth, plan, and goal files
around `Nearest similar regimes / Main-regime view: top 3 / 90%`; regenerate
higher-sample paper/demo fan charts; run a higher-sample confirmation backtest;
update the paper around the conditionality-versus-calibration tradeoff; and run
independent verification before any paper/demo-ready claim.

## Requirement Status

| Requirement | Evidence | Status |
| --- | --- | --- |
| Align current-truth, plan, and goal around top3/90 | `docs/research_protocols/nl_prefix_latent_current_truth.md`, `docs/research_protocols/nl_prefix_latent_autoresearch_plan.md`, `docs/research_protocols/nl_prefix_latent_goal.json`, `autoresearch-session/nl_prefix_latent_goal.json`, and `.agents/skills/nl-prefix-latent-autoresearch/SKILL.md` name top3/90 as the current multistart paper/demo candidate, with top2/80 and all-regime tradeoffs preserved. | Complete |
| Regenerate higher-sample paper/demo fan charts | Start22 high-sample chart: `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962a_start22_s384_d400/component_posterior_factor_fans_all_modes.png`; start18 chart: `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_963a_start18_s384_d400/component_posterior_factor_fans_all_modes.png`; start40 chart: `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_963b_start40_s384_d400/component_posterior_factor_fans_all_modes.png`; paper copy: `paper/narrative_grounded_scenarios/figures/posterior_ensemble_candidate_fans_962a.png`. | Complete |
| Run higher-sample confirmation backtest | `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962b_guardrail_cohesive_66req_s64_d250/component_backtest_report.json` and `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962c_selection_confirmation/posterior_ensemble_selection_report.json`. | Complete |
| Resolve single-start contradiction with broader evidence | `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_963c_multistart_confirmation/posterior_ensemble_multistart_confirmation.json` aggregates starts 18, 22, and 40 and ranks top3/90 first with score `0.800`. | Complete |
| Update paper around conditionality-versus-calibration tradeoff | `paper/narrative_grounded_scenarios/main.tex` and `paper/narrative_grounded_scenarios/generated_tables/table_posterior_ensemble_confirmation.tex` state that top3/90 is the current product-facing presentation candidate, top2/80 is the sharper KS-oriented variant, and all-regime pooling is the calibration diagnostic. | Complete |
| Run independent verifier before paper/demo-ready claim | `docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-27_posterior_ensemble_multistart_confirmation_963c.md` gives verdict `AGREE` for promoting top3/90 as the current paper/demo candidate under explicit tradeoff framing. | Complete |
| Validate current state | JSON validation passed, relevant scripts compile, focused tests passed with `18 passed`, and `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` rebuilt the paper successfully. | Complete |

## Final Audit Conclusion

The objective is complete for the paper/demo candidate claim. The promoted
candidate is:

```text
Nearest similar regimes / Main-regime view: top 3 / 90%
```

The exact claim is bounded: top3/90 is the current paper/demo posterior-ensemble
candidate because it is the best three-start calibration-plus-conditionality
tradeoff. Top2/80 remains the sharper KS-oriented variant, all selected regimes
remain the calibration diagnostic, and this is not a production-ready or
conditionality-solved claim.
