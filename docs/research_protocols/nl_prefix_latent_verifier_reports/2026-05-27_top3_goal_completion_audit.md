# Completion Audit: Original Top3/90 Posterior-Ensemble Goal

Date: 2026-05-27

## Objective Audited

Set up and execute the NL prefix-latent autoresearch objective to promote the
selected posterior ensemble candidate: align current-truth, plan, and goal files
around `Nearest similar regimes / Main-regime view: top 3 / 90%`; regenerate
higher-sample paper/demo fan charts; run a higher-sample confirmation backtest;
update the paper around the conditionality-versus-calibration tradeoff; and run
independent verification before any paper/demo-ready claim.

## Requirement Status

| Requirement | Current Evidence | Status |
| --- | --- | --- |
| Align tracked workflow files around the posterior-ensemble candidate | `docs/research_protocols/nl_prefix_latent_current_truth.md`, `docs/research_protocols/nl_prefix_latent_autoresearch_plan.md`, and `docs/research_protocols/nl_prefix_latent_goal.json` now name the nearest-similar main-regime family and record that top3/90 is not confirmed as the single default. | Completed for the evidence-aligned family; not completed for literal top3/90 promotion. |
| Regenerate higher-sample paper/demo fan charts | High-sample fixed-start deck exists at `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962a_start22_s384_d400`; paper figure copy exists at `paper/narrative_grounded_scenarios/figures/posterior_ensemble_candidate_fans_962a.png`. | Completed. |
| Run higher-sample confirmation backtest | Backtest exists at `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962b_guardrail_cohesive_66req_s64_d250/component_backtest_report.json`; selection confirmation exists at `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962c_selection_confirmation/posterior_ensemble_selection_report.json`. | Completed. |
| Update paper around conditionality versus calibration | Ignored paper source `paper/narrative_grounded_scenarios/main.tex` and generated table `paper/narrative_grounded_scenarios/generated_tables/table_posterior_ensemble_confirmation.tex` frame all-regime pooling as calibration diagnostic and main-regime views as product-facing tradeoffs. | Completed in ignored paper tree. |
| Run independent verifier before paper/demo-ready claim | `docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-27_posterior_ensemble_confirmation_962c.md` gives `PARTIAL`. | Completed; verdict blocks paper/demo-ready top3/90 promotion. |
| Promote top3/90 as selected paper/demo candidate | 962c ranks `top2_80` first and `top3_90` second. The verifier says exact top3/90 promotion is not supported. | Not achieved; contradicted by current evidence. |

## Key Metrics

- `top2_80`: CRPS improvement `+0.189`, energy improvement `+0.281`, 80%
  coverage `0.778`, factor KS `0.187`, portfolio KS `0.177`, selection score
  `0.779`.
- `top3_90`: CRPS improvement `+0.207`, energy improvement `+0.295`, 80%
  coverage `0.807`, factor KS `0.165`, portfolio KS `0.149`, selection score
  `0.765`.
- all selected regimes: CRPS improvement `+0.215`, energy improvement `+0.309`,
  80% coverage `0.826`, factor KS `0.110`, portfolio KS `0.115`.
- start-only null: factor KS `0.000`, portfolio KS `0.000`, path energy
  `0.000`, VaR95 range `0.000`.

## Audit Conclusion

The executable parts of the objective were completed: high-sample fan chart,
high-sample backtest, paper update, and verifier. The final requested promotion
of exact `top3_90` is not currently valid. The evidence supports the
nearest-similar main-regime **family**, with `top2_80` as the current
high-sample product-facing presentation candidate, `top3_90` as the
calibration-leaning variant, and all-regime pooling as the calibration
diagnostic.

Do not mark the active goal complete unless the user accepts this evidence-based
pivot or new evidence reverses the top3/90 ranking.
