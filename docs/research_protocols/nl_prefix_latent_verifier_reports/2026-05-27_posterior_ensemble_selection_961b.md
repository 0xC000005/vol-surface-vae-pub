# Posterior Ensemble Selection 961b Verification

## Verdict

PARTIAL / PROCEED AS CURRENT PRODUCTION CANDIDATE.

The evidence supports selecting **Nearest similar regimes / Main-regime view:
top 3 / 90%** as the current single recommended ensemble method for the
narrative-conditioned scenario generator. It is the best observed balance of
fixed-start narrative separation and held-out distributional quality among the
tested candidates.

## What I Checked

- Implementation:
  - `experiments/backfill/block_ar/nl_posterior_ensemble_selection.py`
  - `experiments/backfill/block_ar/nl_support_component_posterior_bakeoff.py`
- Historical backtest artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_selection_961b_29w/posterior_ensemble_selection_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_selection_961b_guardrail_cohesive_66req_s32_d200/component_backtest_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_selection_961b_guardrail_kernel_66req_s32_d200/component_backtest_report.json`
  - existing current and cluster 29-window reports.
- Fixed-start conditionality artifact:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/support_cohesion_component_posterior_bakeoff_960a_smoke_s8_start22/component_posterior_bakeoff.json`
- Verification commands:
  - `python -m py_compile experiments/backfill/block_ar/nl_posterior_ensemble_selection.py experiments/backfill/block_ar/nl_support_component_posterior_bakeoff.py`
  - A direct equality check that full-view rescoring matches the original current and cluster 29-window component backtest CRPS/energy improvements.

## Confirmed

- The artifact-only scorer does not call OpenAI and does not rerun the SNI
  generator. It reuses saved generated delta samples and component metadata.
- The scorer reconstructs the realized target path using the same validation
  block and selected-window mapping used by the story-smoke workflow.
- Full-view rescoring exactly matches the original source reports for the
  current and cluster 29-window backtests:
  - current full: CRPS and energy improvement differences are `0.0`;
  - cluster full: CRPS and energy improvement differences are `0.0`.
- On the 29-window comparable setup, the selected candidate ranks first:
  - CRPS improvement versus persistence: `+0.172`;
  - energy improvement versus persistence: `+0.269`;
  - 80% coverage: `0.746`;
  - fixed-start factor KS: `0.524`;
  - fixed-start portfolio KS: `0.572`.
- The same candidate also ranked first on the 24-window apples-to-apples
  candidate screen.

## Issues / Scope Limits

- The selected candidate is not the best pure calibration candidate. All-regime
  views have stronger CRPS/energy and coverage, but weaker visible narrative
  separation.
- The selected candidate uses fewer samples on average because it keeps only
  the dominant support components. This is the intended main-regime view, but
  production runs should increase total generated samples so each retained
  component has enough paths.
- Mean-path and terminal MAE are not the target objective for this product and
  are weaker for the selected candidate. This should be described as a
  distributional scenario generator, not a point forecaster.
- The fixed-start conditionality plot is still a smoke visual. It supports the
  mechanism, but the paper/demo should regenerate a higher-sample presentation
  plot using the selected method.

## Independent Assessment

Select **Nearest similar regimes / Main-regime view: top 3 / 90%** as the
current production candidate because it directly solves the product-facing
problem the user identified: full pooling hides narrative conditionality. The
candidate preserves a coherent set of dominant narrative-matched regimes while
remaining distributionally better than persistence in held-out backtests.

Recommended next step: make this method the default in demo/paper settings,
regenerate the presentation fan charts at a higher sample count, and keep the
all-regime view available as a calibration diagnostic rather than the primary
product view.
