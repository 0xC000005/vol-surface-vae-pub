# Independent Verifier: Incumbent Cleanup 943a

Date: 2026-05-22

## Verification Result

Verdict: `PARTIAL`

The cleanup claim is supported: the fixed-start comparison, stress-test script,
casebook plotting script, Gradio demo defaults, current-truth note, and paper
now use the incumbent hard-direction start-aware component-preserving support
mixture as the product-facing default, with the start-only policy as the null
control. Response-preview policies remain available as diagnostics but are not
the default paper/demo path.

The stronger claim that the whole narrative-conditioned scenario product is
production-ready is not verified by this pass. This report verifies cleanup,
artifact consistency, default selection, and current paper-facing evidence.

## What I Checked

- `experiments/backfill/block_ar/nl_fixed_start_rollout_policy_comparison.py`
  - public case ids are clean;
  - legacy `*_start18` names are only condition-report aliases;
  - `DEFAULT_POLICY_NAMES` is `("current_start_checked_gap30", "start_only_topk")`.
- `experiments/backfill/block_ar/nl_conditionality_stress_test.py`
  - default input/output roots point to the clean 943a artifacts;
  - default policies are incumbent plus start-only null;
  - selected policy defaults to `current_start_checked_gap30`.
- `experiments/backfill/block_ar/plot_narrative_casebook_backtest.py`
  - default control root points to the clean 943a comparison;
  - default variant is `current_start_checked_gap30`;
  - output filenames are fixed-start neutral rather than `start18`-specific.
- `experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py`
  - demo run args use `memory_prior_mode="diverse_topk_narrative_start_checked"`;
  - demo rollout uses `component_prefix_mixture`;
  - UI default start is 22 and does not expose response-preview as the main flow.
- Artifacts inspected:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_943a_start22_incumbent_clean_s64_d400/fixed_start_rollout_policy_comparison.json`;
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_943a_start22_incumbent_clean_s64_d400/fixed_start_rollout_policy_comparison.md`;
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_943a_start22_incumbent_clean/conditionality_stress_test.json`;
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_943a_start22_incumbent_clean/conditionality_stress_test.md`;
  - `paper/narrative_grounded_scenarios/figures/narrative_qualitative_casebook_summary.json`;
  - `paper/narrative_grounded_scenarios/figures/narrative_portfolio_impact_summary.json`.
- Paper surfaces inspected:
  - `paper/narrative_grounded_scenarios/main.tex`;
  - `paper/narrative_grounded_scenarios/generated_tables/table_fixed_start_rollout_policy_comparison.tex`;
  - compiled `paper/narrative_grounded_scenarios/main.pdf`.

## Findings

### Confirmed Correct

- Clean public case ids are now used for new rollout artifacts:
  `fragile_risk_on`, `defensive_risk_off`, `commodity_inflation`,
  `dollar_liquidity`, `rates_selloff`, and `safe_haven_gold`.
- Legacy `*_start18` labels still exist only as aliases for older cached
  condition reports and fallback readers. New 943a output paths do not use those
  labels.
- The paper/demo default evidence is incumbent-plus-null, not response-preview.
- The 943a fixed-start comparison is internally consistent:
  - incumbent direction checks: `6/6`;
  - start-only direction checks: `1/6`;
  - start-only null has identical support and zero factor/portfolio separation.
- The 943a stress test supports fixed-start narrative conditionality against the
  start-only null:
  - incumbent: `6/6` gates, support Jaccard `0.033`, relevant terminal KS
    `0.166`, path energy `0.0165`, portfolio KS `0.124`, VaR95 loss range
    `2.290`;
  - start-only null: `1/6` gates, support Jaccard `1.000`, zero factor/path/
    portfolio/tail separation.
- The paper compiles successfully to a 33-page PDF after the figure/table
  refresh.

### Issues Found

- `NOTE`: response-preview policies still exist in the policy registry. That is
  acceptable because they are diagnostic options, but the code and docs must
  continue to prevent them from becoming defaults without a later verifier.
- `NOTE`: old `start18` strings remain in fallback alias maps and historical
  artifact lists. They are not current public case labels, but broad string
  searches will still find them.
- `WARNING`: this pass does not rerun the full historical backtest or prove
  final production readiness. It verifies the current cleanup and paper-facing
  incumbent evidence.

## Alternative Explanations

The fixed-start stress-test improvement could still be partly driven by the
chosen start-22 case and the selected six narratives. The start-only null rules
out pure starting-level dependence for this evidence bundle, but broader
production claims still need the already tracked multi-start and held-out
backtest evidence.

## My Independent Assessment

The requested cleanup is complete and evidence-backed. The correct public
framing is: the incumbent hard-direction start-aware support mixture is the
demo/paper default; response-preview remains diagnostic; the refreshed 943a
figures and tables support narrative-conditioned support selection and
portfolio-impact variation under a fixed start.

Do not call the entire product production-ready solely from this report. It is
reasonable to call this version the current product-facing candidate with
verified cleanup and paper-facing conditionality evidence.

## Recommended Action

Proceed with the cleaned incumbent default in the demo and paper. Keep
response-aware methods on a separate diagnostic/research track until they beat
the incumbent on both fixed-start conditionality and held-out scenario quality,
then run a new verifier before promotion.
