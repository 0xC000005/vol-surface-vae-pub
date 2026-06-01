# Independent Verification: Broad Weighted Response Guard 940b

Verification date: 2026-05-22

## Verification Result

Verdict: `PARTIAL`

The 940b broad weighted response guard is a legitimate active candidate. The
artifacts support the claim that it reduces start-only dominance and produces
stronger fixed-start factor/tail separation in some risk channels while staying
competitive on held-out historical backtests. The evidence does **not** yet
support promoting it as the production default because portfolio terminal KS is
lower than the incumbent in the scaled fixed-start rollout, and the backtest is
still a 24-window TestFlight rather than the full promotion suite.

## What I Checked

- Code:
  - `experiments/backfill/block_ar/nl_broad_support_response_utility.py`
  - `experiments/backfill/block_ar/nl_prefix_latent_analogue_mixture_prior.py`
  - `experiments/backfill/block_ar/nl_fixed_start_rollout_policy_comparison.py`
  - `experiments/backfill/block_ar/nl_prefix_latent_component_backtest.py`
- Artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_broad_support_response_utility_940a/broad_support_response_utility_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_940b_s64_d400/fixed_start_rollout_policy_comparison.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_component_backtest_940b_current_24w/component_backtest_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_component_backtest_940b_broad_24w/component_backtest_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_940b_s64_d400/fixed_start_rollout_factor_fans.png`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_940b_s64_d400/fixed_start_rollout_portfolio_tail.png`
- Tests:
  - `uv run pytest test_code/test_797a_nl_prefix_latent_analogue_mixture_prior.py test_code/test_931a_nl_fixed_start_rollout_policy_comparison.py test_code/test_939b_nl_prefix_latent_component_backtest.py -q`

## Findings

### Confirmed Correct

- The broad response-utility context selects the support-prior path, not the
  weaker feature model. The report shows:
  - portfolio prior pairwise accuracy `0.5256`;
  - CRPS prior pairwise accuracy `0.5319`;
  - energy prior pairwise accuracy `0.5249`;
  - feature quality pairwise accuracy `0.4793`.
- The weighted aggregation fix is methodologically important. It prevents the
  broad policy from collapsing to one best three-window candidate mixture and
  preserves a broader weighted support set.
- The scaled fixed-start artifact supports a nonzero narrative-conditioning
  claim under the same start:
  - start-only null has factor KS `0.0`, portfolio KS `0.0`, VaR95 range `0.0`;
  - incumbent has factor KS `0.1932`, portfolio KS `0.1677`, VaR95 range `1.9083`;
  - broad guard has factor KS `0.1961`, portfolio KS `0.1302`, VaR95 range `3.5174`.
- The 24-window backtest is competitive with the incumbent:
  - incumbent component CRPS improvement `+21.447%`, energy `+31.012%`;
  - broad component CRPS improvement `+21.673%`, energy `+30.761%`.
- The all-risk-channel factor fan and portfolio-tail plot artifacts exist.
- Relevant regression tests pass (`34 passed`).

### Issues Found

- `WARNING`: The broad guard does not dominate the incumbent on all
  conditionality metrics. Portfolio terminal KS is lower (`0.1302` versus
  `0.1677`) even though VaR95 range is larger.
- `WARNING`: The historical backtest is still a 24-window TestFlight at
  32 samples / 200 decoder steps. It is useful evidence, but not a full
  promotion suite.
- `WARNING`: The support-prior pairwise accuracies are only mildly above
  chance. The method works as a calibrated support prior, not as a high-accuracy
  supervised response predictor.
- `NOTE`: The feature-level model is diagnostic only and should not be described
  as the driver of the 940b result.

## Alternative Explanations

- The larger VaR95 range may reflect support-weight concentration in tails
  rather than uniformly better narrative response.
- The lower portfolio KS suggests that tail differentiation and whole
  distribution differentiation are not aligned. A risk-manager product may
  reasonably favor tails, but the paper/demo should say that explicitly.

## My Independent Assessment

940b is a credible candidate for the next product path, not a finished
production default. The strongest claim is:

> A broad support-prior response guard with weighted direction-safe support
> aggregation reduces start-only collapse and improves fixed-start tail
> differentiation while remaining competitive with the incumbent historical
> backtest.

The weaker claim to avoid is:

> Conditionality is fully solved across all portfolio and factor distribution
> metrics.

## Recommended Action

Proceed with 940b as the active candidate. Do not promote it to default until
one of the following is true:

- a larger promotion suite confirms the 24-window backtest and resolves the
  portfolio-KS trade-off; or
- the product/paper explicitly frames the improvement as tail-risk and
  risk-channel differentiation, with portfolio KS treated as a secondary
  diagnostic.

If the next goal is production promotion, run one bounded calibration on support
weight concentration or probability temperature, then rerun the same fixed-start
and backtest gates. Do not add a new architecture family before resolving that
single trade-off.
