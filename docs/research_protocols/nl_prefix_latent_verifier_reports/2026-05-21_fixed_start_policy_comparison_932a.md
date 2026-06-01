# Independent Verification: High-Sample Fixed-Start Policy Comparison 932a

## Verification Result

Verdict: `AGREE`

The narrow paper-facing claim is supported: under the same accepted starting
level, hard direction-checked narrative support selects different support pools
and produces non-zero factor/portfolio distribution separation, while the
start-only null selects identical support and produces zero cross-narrative
separation.

## What I Checked

- Code:
  - `experiments/backfill/block_ar/nl_fixed_start_rollout_policy_comparison.py`
  - `experiments/backfill/block_ar/nl_prefix_latent_story_smoke.py`
  - `test_code/test_930a_nl_fixed_start_support_policy_bakeoff.py`
  - `test_code/test_931a_nl_fixed_start_rollout_policy_comparison.py`
- Artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/fixed_start_rollout_policy_comparison_932a_s384/fixed_start_rollout_policy_comparison.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/fixed_start_rollout_policy_comparison_932a_s384/fixed_start_rollout_policy_comparison.md`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/fixed_start_rollout_policy_comparison_932a_s384/fixed_start_rollout_factor_fans.png`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/fixed_start_rollout_policy_comparison_932a_s384/fixed_start_rollout_portfolio_tail.png`
- Paper surfaces:
  - `paper/narrative_grounded_scenarios/main.tex`
  - `paper/narrative_grounded_scenarios/generated_tables/table_fixed_start_rollout_policy_comparison.tex`
  - `paper/narrative_grounded_scenarios/figures/fixed_start_rollout_policy_s384_factor_fans.png`
  - `paper/narrative_grounded_scenarios/figures/fixed_start_rollout_policy_s384_portfolio_tail.png`
- Research log:
  - `RESEARCH_LOG.md` entry dated 2026-05-21, "High-sample fixed-start narrative conditionality control".

## Confirmed Correct

- The 932a comparison uses the same fixed start for all narratives and policies:
  `max_abs_start_difference` is `0` for all reported policies.
- The high-sample run used `384` generated paths per narrative and `400`
  decoder steps.
- The start-only null is flat across narratives:
  - mean support Jaccard: `1.000`
  - mean factor terminal KS: `0.000`
  - mean portfolio terminal KS: `0.000`
  - VaR95-loss range: `0.000`
- Hard direction-checked narrative policies are not flat:
  - current hard direction/start-aware policy: direction pass `6/6`, mean
    support Jaccard `0.006`, mean factor terminal KS `0.178`, mean portfolio
    terminal KS `0.118`, VaR95-loss range `10.34`.
  - narrative-first hard direction policy: direction pass `6/6`, mean support
    Jaccard `0.006`, mean factor terminal KS `0.166`, mean portfolio terminal
    KS `0.120`, VaR95-loss range `8.89`.
- Removing start-distance ranking pressure does not remove the effect, which
  supports the interpretation that fixed start is a boundary condition rather
  than the source of the cross-narrative response.
- The paper-facing table matches the artifact-level metrics.
- The paper-facing figures were regenerated with public labels rather than
  internal policy IDs.
- Verification commands passed:
  - `uv run pytest test_code/test_930a_nl_fixed_start_support_policy_bakeoff.py test_code/test_931a_nl_fixed_start_rollout_policy_comparison.py -q`
    returned `7 passed`.
  - `latexmk -pdf -interaction=nonstopmode main.tex` from
    `paper/narrative_grounded_scenarios` produced `main.pdf`.

## Issues Found

- `NOTE`: The comparison is a fixed-start conditionality diagnostic, not a
  held-out CRPS/energy backtest. It supports the mechanism claim, not a new
  distributional-quality claim.
- `NOTE`: The generated paper and experiment artifacts live under ignored paths
  (`paper/` and `experiments/backfill/block_ar/nl_scenario_demo_outputs/`), so
  they are local paper artifacts unless explicitly force-added or moved.
- `NOTE`: LaTeX reports a non-blocking underfull hbox warning in an existing
  alignment; no undefined references or compilation failure remain.

## Alternative Explanations

- The evidence does not prove that every marginal factor fan will look visually
  disjoint. It proves that hard direction-checked narrative support changes the
  support set and creates non-zero factor/portfolio distribution separation
  relative to a start-only null.
- The hard-direction policies still use historical support as the financial
  inductive bias. The result does not validate a support-free direct text-to-
  scenario generator.

## My Independent Assessment

The result is suitable as paper-facing support for the claim that narrative
conditionality is present under the current support-grounded workflow. The
claim should remain bounded: this is evidence of fixed-start narrative response
through support selection and rollout distributions, not a final proof of
production-grade risk-channel control.

## Recommended Action

Proceed with the paper update around this evidence. For the next research step,
improve response-aware support weighting or support scoring while keeping this
hard direction-checked fixed-start comparison as the promotion control.

