# Independent Verification: Support-Gated Narrative Ensemble Calibration 945o

Verification Result

Verdict: `PARTIAL`

## What I Checked

- Code:
  - `experiments/backfill/block_ar/nl_narrative_ensemble_calibration.py`
  - `test_code/test_945a_nl_narrative_ensemble_calibration.py`
- Artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945o_qualitative_review/narrative_ensemble_calibration_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945o_qualitative_review/support_gated_qualitative_review.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945o_qualitative_review/support_gated_narrative_relevant_raw_panels.png`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945o_qualitative_review/support_gated_start_only_null_contrasts.png`
  - Earlier control context in `RESEARCH_LOG.md` entries for `945e-g` and `945i-n`
- Commands:
  - `uv run pytest test_code/test_945a_nl_narrative_ensemble_calibration.py -q`
  - `python -m py_compile experiments/backfill/block_ar/nl_narrative_ensemble_calibration.py`
  - chronological support-gated run that produced the `945o` artifacts

## Findings

### Confirmed Correct

- The candidate uses the incumbent support-grounded frozen SNI rollout as the
  base deck and applies a bounded directional calibration after rollout. It does
  not replace the support store or the frozen SNI generator.
- The support evidence gate prevents the calibration from creating narrative
  separation for start-only support. This addresses the main failure found after
  the first verifier pass: a start-only null cannot be treated as
  narrative-conditioned merely because the text contains directional language.
- The chronological held-out report is internally consistent:
  - selected beta: `0.25`;
  - calibration rows: `14`;
  - evaluation rows: `15`;
  - calibrated-minus-identity CRPS: `-0.001162`;
  - calibrated-minus-identity energy: `-0.002080`;
  - calibrated-minus-identity 80% coverage: `-0.000399`;
  - all recorded promotion gates are `true`.
- Fixed-start attribution improves materially relative to the uncalibrated
  incumbent on the start-normalized grid:
  - narrative share plus interaction rises from `0.424` to `0.664`;
  - same-start factor KS rises from `0.176` to `0.326`;
  - same-start portfolio KS rises from `0.149` to `0.402`;
  - same-start support Jaccard remains low at `0.032`, so the support pools are
    still distinct across narratives.
- The qualitative raw-level panels are stronger than the previous generic fan
  evidence. At the same approved start, the candidate shows narrative-relevant
  terminal median differences versus start-only null:
  - fragile risk-on: SPX `+18.74`, VIX `-1.73`, BBB OAS `-0.09`;
  - defensive risk-off: SPX `-8.67`, VIX `+1.20`, BBB OAS `+0.16`;
  - commodity inflation: crude `+1.25`, US10Y `+0.11`;
  - dollar liquidity: DXY `+0.92`, BBB OAS `+0.14`, VIX `+1.23`;
  - rates selloff: US10Y `+0.10`, DXY `+0.89`;
  - safe-haven gold: gold `+9.27`, US10Y `-0.06`, VIX `+1.39`.

### Issues Found

- `WARNING`: The calibration is a bounded post-rollout adjustment, not a
  learned text-to-latent bridge improvement. It is valid as an ensemble
  calibration layer, but should not be described as solving the full
  text-embedding or prefix-latent learning problem.
- `WARNING`: The adjustment follows extracted current/recent market direction
  vectors. This is acceptable only with the existing support gate and language
  that frames it as calibrated scenario response over accepted support, not as
  the user prescribing a desired future path.
- `WARNING`: The current verifier pass inspected the chronological split and
  prior split-robustness artifacts, but it did not rerun a larger full-corpus
  backtest. Promotion to production default still needs either broader held-out
  replication or explicit product-owner acceptance of this candidate as a
  bounded calibration overlay.
- `NOTE`: Qualitative panels are now product-readable, but they should remain
  narrative-relevant panels plus contrasts. Generic single-factor fans can still
  visually understate conditionality.

## Alternative Explanations

- The stronger conditionality could come mainly from the directional calibration
  layer rather than improved support retrieval. That is not necessarily bad,
  but it means the method should be positioned as support-gated ensemble
  calibration rather than better semantic retrieval.
- Because the accepted start remains fixed, raw path geometry is still anchored
  by the starting level and the frozen SNI generator's learned dynamics.
  Narrative response is most visible in narrative-relevant factors and
  portfolio/tail readouts, not every marginal fan.

## My Independent Assessment

The `945o` candidate is a legitimate improvement over the previous weak
conditionality evidence. It passes the support-gated null-control requirement,
preserves provenance, improves held-out CRPS/energy without meaningful coverage
loss, and produces clearer raw-level narrative-relevant differences under the
same start.

It is not yet a full production-default proof. The strongest defensible claim is
that support-gated narrative ensemble calibration is now a paper/demo candidate
for demonstrating risk-manager-visible conditionality. The claim should be
limited to bounded calibration over accepted historical support, not direct LLM
scenario generation or a fully solved text-to-latent bridge.

## Recommended Action

- Proceed with documenting `945o` as the current support-gated calibration
  candidate and use its qualitative panels for internal review.
- Do not silently replace production/demo defaults until one additional
  broader validation or explicit default-promotion pass confirms the candidate
  under the same support-gated contract.
- Keep the start-only null and support-gate controls mandatory for all future
  calibration variants.
