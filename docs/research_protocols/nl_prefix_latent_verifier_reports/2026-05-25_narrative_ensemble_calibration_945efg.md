# Independent Verification: Narrative Ensemble Calibration 945e-g

Date: 2026-05-25

## Verification Result

Verdict: `PARTIAL`

The 945e/f/g artifacts support the narrow claim that a bounded
quality-constrained response selector is a stronger candidate than pure
CRPS-based beta selection. The code and artifacts are internally consistent, the
effective beta reporting bug is fixed, and the focused tests pass. I do not
recommend promoting this as a demo/paper/production default yet because the
method intentionally applies a time-increasing directional shift to future
deltas, so the product contract needs one more review: it must be presented as
bounded ensemble calibration, not as the user narrative prescribing the future.

## What I Checked

- Code:
  - `experiments/backfill/block_ar/nl_narrative_ensemble_calibration.py`
  - `test_code/test_945a_nl_narrative_ensemble_calibration.py`
  - `experiments/backfill/block_ar/nl_start_narrative_attribution.py`
- Artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945e_qcr_even_odd/narrative_ensemble_calibration_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945f_qcr_odd_even/narrative_ensemble_calibration_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945g_qcr_chronological/narrative_ensemble_calibration_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945g_qcr_chronological/fixed_start_calibrated_factor_fans.png`
- Documentation/provenance:
  - `RESEARCH_LOG.md`
  - `docs/research_protocols/nl_prefix_latent_current_truth.md`

## Confirmed Correct

- `apply_directional_delta_calibration` applies a bounded shift in delta space
  with shape checks and beta clipping. The approved start remains external to
  the delta calibration and is prepended in raw-level fan plots.
- `_select_beta_candidate(..., selection_objective="quality_constrained_response")`
  selects the largest effective beta that satisfies CRPS, energy, and coverage
  floors relative to identity on the calibration split.
- `fit_directional_beta` now clips and deduplicates beta grid values before
  scoring. The old misleading `0.30` selected-beta report is fixed; the saved
  945e/f/g artifacts report candidate betas `[0.0, ..., 0.25]` and selected beta
  `0.25`.
- The three 945e/f/g reports all use `selection_objective:
  quality_constrained_response`, select beta `0.25`, and pass the current
  candidate quality gates.
- Held-out quality comparisons versus identity are favorable or negligible:
  - even/odd: CRPS `-0.000480`, energy `-0.000642`, coverage `-0.000733`;
  - odd/even: CRPS `-0.000561`, energy `-0.001140`, coverage `-0.000684`;
  - chronological: CRPS `-0.001162`, energy `-0.002080`, coverage `-0.000399`.
- Fixed-start conditionality metrics are materially stronger than the 944a
  incumbent baseline: start-normalized narrative plus interaction share is
  `66.4%`, factor KS is `0.326`, and portfolio KS is `0.402`.
- Focused verification commands passed:
  - `uv run pytest test_code/test_945a_nl_narrative_ensemble_calibration.py test_code/test_944a_nl_start_narrative_attribution.py -q`
  - `python -m py_compile experiments/backfill/block_ar/nl_narrative_ensemble_calibration.py`

## Issues Found

- `WARNING`: The fixed-start conditionality grid is not split-specific. The
  same starts/cases are used to evaluate the selected beta for all three split
  modes. This is acceptable for a candidate mechanism check, but not enough for
  final production promotion.
- `WARNING`: The method improves conditionality by shifting future deltas along
  extracted current/recent market directions. That is useful as bounded
  calibration, but it creates a product-framing risk: if described carelessly,
  it can look like the narrative is prescribing future direction. A promotion
  report must explain the distinction and show guardrails.
- `WARNING`: The selected beta hits the configured bound `0.25` in all
  quality-constrained splits. This does not invalidate the result, but it means
  the bound is now an active modeling assumption. A promotion run should include
  a small bound-sensitivity or rationale.
- `NOTE`: The candidate does not change support selection or the frozen SNI
  rollout. That is consistent with the stated research lane, but the method is
  an ensemble calibration layer, not a new text-to-support bridge.

## Alternative Explanations

- The stronger fixed-start conditionality may be mostly the deterministic
  direction shift rather than a richer learned narrative understanding. This is
  still a plausible product mechanism if kept bounded and audited, but it is
  not evidence that the text embedding bridge itself improved.
- The favorable held-out quality may reflect that the extracted directions are
  correlated with realized moves in the 29-window backtest. This should be
  checked on broader time blocks before any production default claim.

## My Independent Assessment

The candidate is real and worth continuing. It fixes the previous selection
objective mismatch and creates a much clearer narrative response without
breaking the current held-out quality metrics. The result is not yet a final
product default. The next promotion-quality step should be a product-contract
and robustness audit: same method, broader blocks or bootstrap/repeat controls,
bound sensitivity, and a clear explanation that the calibration is bounded and
auditable rather than free-form future prediction.

## Recommended Action

Proceed as a `promotion_candidate` only after one more bounded verification run:

1. confirm the start-only null remains flat under the calibration path;
2. run a same-narrative repeat / shuffled-narrative control if available;
3. add beta-bound sensitivity around `0.15`, `0.25`, and possibly `0.35`;
4. inspect qualitative raw-level fans and narrative-relevant factor panels;
5. then update demo/paper defaults only if those checks pass.
