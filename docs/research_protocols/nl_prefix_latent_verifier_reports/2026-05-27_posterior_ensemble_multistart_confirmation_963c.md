# Independent Verification: Posterior-Ensemble Multistart Confirmation 963c

Date: 2026-05-27

## Verification Result

Verdict: `AGREE`

The 963c multistart confirmation supports promoting **Nearest similar regimes /
Main-regime view: top 3 / 90%** as the current paper/demo posterior-ensemble
candidate under an explicit tradeoff framing. This is not a claim that
conditionality is solved or that the system is production-ready. It is a
paper/demo candidate claim: top3/90 is the best current weighted tradeoff across
historical backtest quality and three-start fixed-start conditionality.

## What I Checked

- Current workflow files:
  - `docs/research_protocols/nl_prefix_latent_current_truth.md`
  - `docs/research_protocols/nl_prefix_latent_autoresearch_plan.md`
  - `docs/research_protocols/nl_prefix_latent_goal.json`
- Prior single-start verifier:
  - `docs/research_protocols/nl_prefix_latent_verifier_reports/2026-05-27_posterior_ensemble_confirmation_962c.md`
- High-sample fixed-start decks:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_963a_start18_s384_d400/fixed_start_rollout_policy_comparison.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962a_start22_s384_d400/fixed_start_rollout_policy_comparison.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_963b_start40_s384_d400/fixed_start_rollout_policy_comparison.json`
- Posterior-mode analyses:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_963a_start18_s384_d400/component_posterior_bakeoff_all_modes.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962a_start22_s384_d400/component_posterior_bakeoff_all_modes.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_963b_start40_s384_d400/component_posterior_bakeoff_all_modes.json`
- Multistart aggregation:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_963c_multistart_confirmation/posterior_ensemble_multistart_confirmation.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_963c_multistart_confirmation/posterior_ensemble_multistart_confirmation.md`
- Historical backtest source:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/posterior_ensemble_candidate_962c_selection_confirmation/posterior_ensemble_selection_report.json`

## Confirmed Correct

- The high-sample decks for starts `18`, `22`, and `40` each use 384 generated
  paths per narrative deck and `decoder_steps=400`.
- The start-only null is flat in all three decks. In the aggregated rows,
  `start_only_full` has factor KS `0.000`, portfolio KS `0.000`, path energy
  `0.000`, and VaR95 range `0.000` for starts `18`, `22`, and `40`.
- The nearest-similar support policy has nonzero narrative separation under all
  three starts.
- The 963c multistart ranking is internally consistent with the stated weighted
  score:
  - top3/90: CRPS improvement `+0.207`, energy improvement `+0.295`, coverage
    `0.807`, average factor KS `0.163`, average portfolio KS `0.134`, average
    path energy `10.109`, average VaR95 range `3.046`, score `0.800`;
  - top2/80: CRPS improvement `+0.189`, energy improvement `+0.281`, coverage
    `0.778`, average factor KS `0.177`, average portfolio KS `0.147`, average
    path energy `10.566`, average VaR95 range `2.288`, score `0.753`;
  - all selected regimes: strongest pure calibration but weaker narrative
    separation, score `0.587`;
  - top1: strongest factor KS but weaker calibration and coverage, score
    `0.433`.
- The 963c evidence explains the apparent contradiction with 962c: one
  single-start report favored top2/80 on KS separation, while the multistart
  confirmation favors top3/90 once calibration, coverage, and tail spread are
  considered across starts.

## Issues Found

- `WARNING`: The multistart aggregate covers three starts, not the full
  five-start demo set. This is sufficient for a paper/demo candidate, but not
  for a final production-default claim.
- `WARNING`: The historical backtest metrics are inherited from the 962c
  posterior-mode rescoring of the 29-window held-out backtest artifacts. This is
  valid because posterior modes rescore saved component-prefix samples, but it
  should be stated clearly in the paper/report.
- `NOTE`: Top2/80 remains stronger on average factor KS and portfolio KS. The
  top3/90 promotion is therefore a tradeoff decision, not a dominance claim.
- `NOTE`: All-regime pooling remains the calibration diagnostic because it has
  the strongest CRPS, energy, and coverage but visibly smooths the narrative
  response.

## Alternative Explanations

- Top3/90 wins because the weighted score values calibration and coverage along
  with conditionality. If a product requirement prioritizes only visual
  KS-style separation, top2/80 may be preferable.
- The accepted start still anchors raw market paths. The main-regime posterior
  changes scenario-family presentation and tail spread; it does not remove the
  SNI generator's calibrated local market dynamics.

## My Independent Assessment

Promote **Nearest similar regimes / Main-regime view: top 3 / 90%** as the
current paper/demo candidate with explicit caveats:

- top3/90 is the best current multistart calibration-plus-conditionality
  tradeoff;
- top2/80 is the sharper KS-oriented variant;
- all selected regimes remain the calibration diagnostic;
- do not claim production-ready or conditionality-solved from this alone.

## Recommended Action

- Update paper/demo text around the 963c multistart evidence.
- Keep the tradeoff table visible so top3/90 is not presented as dominating
  every metric.
- Mark the original top3/90 promotion goal complete only after the tracked truth,
  paper, research log, and validation commands all reflect this 963c verifier.
