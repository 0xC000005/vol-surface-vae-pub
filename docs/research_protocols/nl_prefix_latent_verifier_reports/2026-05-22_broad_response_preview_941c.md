# Independent Verification: Broad Response-Preview Support Weighting 941c

## Verification Result

Verdict: `PARTIAL`

The 941c broad response-preview candidate is a credible active promotion
candidate for stronger fixed-start narrative conditionality, but it is not yet
a production default. The evidence supports the narrow claim that this method
improves fixed-start narrative response while preserving the historical
backtest floor. It does not yet prove full production readiness across starts,
seeds, or deployment cost budgets.

## What I Checked

- Implementation:
  - `experiments/backfill/block_ar/nl_fixed_start_rollout_policy_comparison.py`
  - `experiments/backfill/block_ar/nl_prefix_latent_component_backtest.py`
  - `experiments/backfill/block_ar/nl_prefix_latent_story_smoke.py`
  - `test_code/test_931a_nl_fixed_start_rollout_policy_comparison.py`
  - `test_code/test_939b_nl_prefix_latent_component_backtest.py`
- Artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_941c_broad_response_preview_s64_d400/fixed_start_rollout_policy_comparison.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_941c_broad_response_preview_s64_d400/conditionality_stress_test.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_component_backtest_941b_broad_response_preview_66w/component_backtest_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_component_backtest_940d_current_66w/component_backtest_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_component_backtest_940d_broad_66w/component_backtest_report.json`
- Verification command:
  - `uv run python -m py_compile experiments/backfill/block_ar/nl_prefix_latent_component_backtest.py experiments/backfill/block_ar/nl_fixed_start_rollout_policy_comparison.py && uv run pytest test_code/test_931a_nl_fixed_start_rollout_policy_comparison.py test_code/test_939b_nl_prefix_latent_component_backtest.py -q`
  - Result: `12 passed`.

## Confirmed Correct

- The new `broad_response_preview_gap30` policy combines the broad support
  prior `broad_replay_response_guard_940a` with
  `response_preview_component_mixture`; it does not replace the frozen SNI
  generator or abandon support provenance.
- The response-preview path scores generated preview states with
  `response_channels_from_grounding(...)` and `_component_response_score(...)`.
  I did not find future-target leakage in this reweighting path: the preview is
  based on generated samples from the frozen generator under the selected start
  and current/recent narrative channels.
- The component backtest harness now accepts
  `response_preview_component_mixture` and passes the preview parameters through
  to story-smoke. Focused tests cover this contract.
- The scaled fixed-start result supports the conditionality claim:
  - current: factor KS `0.1836`, portfolio KS `0.1573`, VaR95 range `1.938`;
  - broad-only: factor KS `0.1655`, portfolio KS `0.1604`, VaR95 range `3.277`;
  - broad-plus-preview: factor KS `0.1969`, portfolio KS `0.1677`, VaR95 range
    `3.777`;
  - start-only: factor KS `0.0`, portfolio KS `0.0`, VaR95 range `0.0`.
- The stress-test gate supports the same conclusion:
  - broad-plus-preview passes `6/6`;
  - relevant-terminal KS `0.1881`;
  - relevant path energy `0.0284`;
  - portfolio KS `0.1677`;
  - start-only fails with only `1/6` gates.
- The 29-window held-out backtest remains competitive:
  - current component CRPS `+20.701%`, energy `+30.364%`;
  - broad component CRPS `+20.892%`, energy `+30.160%`;
  - broad-plus-preview CRPS `+20.801%`, energy `+30.215%`.

## Issues Found

- `WARNING`: This is a one-start fixed-start validation. It directly addresses
  the user's start-dominance concern for start 18, but it does not prove the
  same conditionality strength across multiple user-selected starts.
- `WARNING`: The response-preview method improves product-facing
  conditionality, but the historical backtest is competitive rather than a
  clean all-metric win. Energy remains slightly below the current incumbent,
  although it improves over broad-only.
- `WARNING`: The scaled fixed-start run is one seed and one preview budget
  (`4` preview samples/component, blend `0.35`). This is coherent as a bounded
  candidate, but not enough for a production default without seed/repeat
  controls.
- `WARNING`: The method is operationally more expensive because it runs preview
  rollouts before the final deck. The product contract should expose this as a
  higher-quality mode or justify the extra latency.
- `NOTE`: The worktree is heavily mixed with prior NL-prefix files. Any
  checkpoint commit must stage only the files owned by this HEAD iteration.

## Alternative Explanations

- The fixed-start gain could partially reflect seed/sample noise. The 64-sample
  scaled run lowers this risk compared with the 32-sample TestFlight, but a
  repeat seed is still needed before a production-default claim.
- The portfolio-tail gain may come from preview reweighting broadening or
  concentrating support weights rather than from a deeper semantic language
  representation. That is acceptable for the current support-grounded product
  claim, but the paper should frame it as generator-response-aware support
  weighting, not as a direct text-to-generator latent breakthrough.

## My Independent Assessment

The candidate meaningfully improves the current bottleneck. It is stronger than
the earlier broad-only candidate because it improves fixed-start factor,
portfolio, and tail metrics in the scaled run while preserving competitive
historical CRPS/energy. It is also more methodologically coherent than the
temperature sweep because it targets the actual response surface of the frozen
generator.

I would promote it to **active candidate / paper-facing diagnostic result**,
not to **production default**. The next gate should be multi-start and/or repeat
seed validation, plus a concise current-truth update that clearly states the
claim boundary.

## Recommended Action

Proceed with the broad response-preview path as the active candidate. Do not
claim final production readiness yet. Run one of:

1. a repeat-seed scaled fixed-start check at the same start; or
2. a multi-start compact check, for example starts `18`, `22`, and `40`, with a
   smaller sample budget if needed.

If that check preserves the same pattern, update
`docs/research_protocols/nl_prefix_latent_current_truth.md` and the paper/demo
claims around `broad_response_preview_gap30`.
