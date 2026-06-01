# Independent Verification: Matched Broad-Support Calibration 946a

Verification Result

Verdict: `PARTIAL`

The scoped claim is supported: `946a` fixes the earlier `945q-r` evidence
mismatch by using the broad-support backtest report and matched broad-support
fixed-start visual/support decks. It is acceptable as stronger paper/demo
candidate evidence for support-gated narrative response. It should not be
described as a silent production default or as direct text-to-scenario
generation.

## What I Checked

- Code:
  - `experiments/backfill/block_ar/nl_narrative_ensemble_calibration.py`
  - `test_code/test_945a_nl_narrative_ensemble_calibration.py`
- Artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_946a_matched_broad_support_deck/narrative_ensemble_calibration_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_946a_matched_broad_support_deck/support_gated_qualitative_review.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_component_backtest_940d_broad_66w/component_backtest_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_946a_start18_broad_replay_s64_d400/fixed_start_rollout_policy_comparison.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_946a_start22_broad_replay_s64_d400/fixed_start_rollout_policy_comparison.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_fixed_start_policy_comparison_946a_start40_broad_replay_s64_d400/fixed_start_rollout_policy_comparison.json`
- Paper surface:
  - `paper/narrative_grounded_scenarios/generated_tables/table_support_gated_calibration.tex`
  - `paper/narrative_grounded_scenarios/main.tex`
  - `paper/narrative_grounded_scenarios/figures/support_gated_narrative_relevant_raw_panels_946a.png`
  - `paper/narrative_grounded_scenarios/figures/support_gated_start_only_null_contrasts_946a.png`
- Research-log context around `945q`, `945r`, and the paper refresh entries.

## Confirmed Correct

- `946a` selected beta `0.25` with `support_gate_mode=direction_status` and
  `selection_objective=quality_constrained_response`.
- Held-out broad-support deltas versus identity are favorable:
  CRPS `-0.001146`, energy `-0.002094`, coverage `+0.000399`.
- The fixed-start decks now come from broad-support roots for starts `18`,
  `22`, and `40`, not from the older current-support visual deck.
- Matched broad fixed-start attribution is materially nonzero:
  factor KS `0.309`, portfolio KS `0.426`, support Jaccard `0.028`.
- Per-start floors pass:
  - start18: factor KS `0.299`, portfolio KS `0.402`, support Jaccard `0.032`;
  - start22: factor KS `0.297`, portfolio KS `0.402`, support Jaccard `0.025`;
  - start40: factor KS `0.331`, portfolio KS `0.473`, support Jaccard `0.028`.
- Start-only null controls remain flat in the fixed-start comparison summaries:
  factor KS `0.0`, portfolio KS `0.0`, support Jaccard `1.0`.
- Broad-support policy summaries show all six narrative supports pass direction
  checks for each matched start deck.
- The new `--start-root LABEL=PATH` wiring is covered by a regression test and
  the focused test suite passed after the change.

## Issues Found

- `WARNING`: The broad backtest artifact name includes `66w`, but the artifact
  contains only `29` usable `window_scores` after failures/filtering. Public
  text must say `29 usable windows`, not `66-window evaluation`.
- `WARNING`: This is a bounded post-rollout support-gated calibration layer
  over the frozen support-grounded SNI ensemble. It is not a direct
  text-to-latent prior and not a standalone LLM scenario generator.
- `WARNING`: The calibration parameter is still a single bounded directional
  beta. The evidence supports a paper/demo candidate and a cleaner product
  story, but not a final production default without either product-owner
  acceptance or a larger usable-window replication.
- `NOTE`: Start-only null direction statuses include expected rejects because
  the null intentionally removes narrative direction from support conditioning.
  The relevant null result is that its support and generated distributions are
  identical across narratives under the same start.

## Alternative Explanations

- The visible narrative separation could partly reflect the bounded calibration
  moving named factors in the grounded direction, not only the support mixture
  itself. This is acceptable under the current method framing, but the paper
  should describe it as support-gated ensemble calibration rather than pure
  support retrieval.
- The quality gain is small in absolute terms. Its importance is that it
  preserves or improves the incumbent metric floor while improving visible
  fixed-start narrative response.

## Independent Assessment

`946a` is a materially better paper/demo evidence point than `945q-r` because
the held-out broad-support fit/evaluation and the qualitative fixed-start deck
now come from the same broad support-bank setup. The conditionality claim is
now better supported at the level risk managers can inspect: raw-level
narrative-relevant panels, start-only-null contrasts, support provenance, and
portfolio/factor KS metrics all agree directionally.

The result should be promoted as the current paper/demo candidate. It should
not yet be called production-ready by default. The remaining blocker is not the
old mixed-evidence bug; it is scale/governance: only `29` usable broad
backtest windows are present, and the method is a bounded calibration overlay.

## Recommended Action

- Update the research log and current-truth file to make `946a` the current
  paper/demo candidate.
- Keep public claims scoped to support-gated narrative-conditioned scenario
  generation with auditability and fixed-start controls.
- Do not say `66-window evaluation`; say `29 usable broad-support held-out
  windows`.
- Next production-promotion step: build or recover a larger usable broad
  backtest manifest and rerun the same matched-deck verifier.
