# Independent Verification: Incumbent After Response-Preview Rejection

Verdict: `PARTIAL`

## Claim Checked

The response-preview family should be downgraded to diagnostic status, and the
current hard-direction start-aware component-preserving support mixture should
be treated as the active production candidate for the narrative-conditioned
scenario generator, subject to clear limitations.

## What I Checked

- `experiments/backfill/block_ar/nl_fixed_start_rollout_policy_comparison.py`
- `experiments/backfill/block_ar/nl_prefix_latent_story_smoke.py`
- `experiments/backfill/block_ar/nl_prefix_latent_component_backtest.py`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_component_backtest_940d_current_66w/component_backtest_report.json`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_multistart_response_preview_941f_summary.json`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_start22_repeat_seed_response_preview_942c_summary.json`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_conditionality_stress_test_942d_start22_broad_response_preview_s64_d400/conditionality_stress_test.json`
- Recent research-log entries for 940d through 942d.

## Confirmed Correct

- The current fixed-start policy is explicitly the hard-direction start-aware
  selector: `current_start_checked_gap30` maps to
  `diverse_topk_narrative_start_checked`, applies start-distance and implication
  alignment, and enforces a 30-window non-overlap gap in the fixed-start rollout
  harness.
- The response-preview policies are not stable enough to promote. The larger
  start-22 64-sample / 400-step gate has current beating broad response-preview
  on all key stress columns:
  - relevant terminal KS: `0.1820` current vs `0.1638` preview;
  - relevant path energy: `0.0241` current vs `0.0236` preview;
  - portfolio KS: `0.1521` current vs `0.1271` preview;
  - VaR95 loss range: `4.0234` current vs `3.7300` preview.
- The current policy has held-out scenario-quality evidence on 29 scored
  windows:
  - CRPS improvement vs persistence: `+20.701%`;
  - energy improvement vs persistence: `+30.364%`;
  - 80% coverage mean: `0.8085`.
- Same-start narrative conditionality is not merely a start-level artifact in
  the fixed-start stress tests. Across starts 18, 22, and 40, the current policy
  passes all three compact gates, while the start-only null fails all three and
  has zero factor, path, portfolio, and VaR separation.

## Issues Found

- `WARNING`: The current policy is an active production candidate, not a fully
  solved production-default proof. The strongest multi-start summary is compact
  32-sample evidence for starts 18, 22, and 40 plus a larger start-22 run. Starts
  18 and 40 do not yet have the same larger-sample repeat gate in this verifier
  bundle.
- `WARNING`: The 29-window backtest is strong versus persistence but smaller
  than the filename implies (`66w` requested, 29 scored). Reports should say
  29 scored held-out windows, not 66.
- `WARNING`: Response-preview and portfolio-aware objective code is now present
  as diagnostic machinery. Paper/demo/default code paths should not silently use
  those policies unless a later verifier promotes them.
- `NOTE`: The fixed-start case labels still contain suffixes such as
  `start18` even when the command uses `--start-window-index 22` or `40`. The
  saved `run_config.start_window_index` is the source of truth, but the labels
  can confuse human readers and should be cleaned before publication/demo use.

## Alternative Explanations

- The current policy may look better partly because it is simpler and less
  sensitive to preview-rollout sampling noise. That is a valid product advantage,
  but it means the evidence supports a robust support-mixture product more than
  a sophisticated response-preview method.
- Some fixed-start conditionality metrics are sensitive to sample count and
  rollout seed. The start-only null control is strong, but fine-grained ranking
  among narrative policies still needs repeated larger-sample runs before any
  stronger "best policy" claim.

## Independent Assessment

I agree with downgrading response-preview to diagnostic status. I partially
agree with promoting the incumbent hard-direction start-aware mixture as the
active production candidate: it has the best current evidence bundle, passes
the null-control story, and has a strong historical backtest. I would not state
that the full product is production-ready or that conditionality is completely
solved. The defensible claim is:

> The incumbent support-grounded policy demonstrates risk-manager-meaningful
> narrative conditionality above start-only controls and preserves held-out
> distributional quality; response-preview variants remain diagnostic.

## Recommended Action

Proceed by updating current-truth and paper/demo language around the incumbent
policy, not the response-preview branch. Before any stronger default or
production-ready claim, run larger-sample repeats for starts 18 and 40, clean
the stale case-label suffixes, and ensure the demo defaults cannot select a
diagnostic response-preview policy by accident.
