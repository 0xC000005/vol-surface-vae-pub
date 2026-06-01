# Independent Verification Addendum: Support-Gated Calibration 945q-r

Verification Result

Verdict: `PARTIAL`

## What I Checked

- Code:
  - `experiments/backfill/block_ar/nl_narrative_ensemble_calibration.py`
  - `test_code/test_945a_nl_narrative_ensemble_calibration.py`
- Artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945q_per_start_promotion_gates/narrative_ensemble_calibration_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945r_broad_support_backtest/narrative_ensemble_calibration_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_945q_per_start_promotion_gates/support_gated_narrative_relevant_raw_panels.png`
- Commands:
  - `uv run pytest test_code/test_945a_nl_narrative_ensemble_calibration.py test_code/test_944a_nl_start_narrative_attribution.py -q`
  - `python -m py_compile experiments/backfill/block_ar/nl_narrative_ensemble_calibration.py`
  - `git diff --check` on the touched calibration, test, current-truth,
    verifier, and research-log files

## Findings

### Confirmed Correct

- 945q adds a useful broad fixed-start gate: the candidate must show
  cross-narrative separation at every accepted start in the three-start grid.
  It passes:
  - start18: factor KS `0.320`, portfolio KS `0.398`, support Jaccard `0.033`;
  - start22: factor KS `0.316`, portfolio KS `0.393`, support Jaccard `0.033`;
  - start40: factor KS `0.342`, portfolio KS `0.416`, support Jaccard `0.029`.
- 945q keeps the chronological held-out quality result from 945o:
  CRPS `-0.001162`, energy `-0.002080`, coverage `-0.000399` versus identity.
- 945r reruns the calibration fit/evaluation on the broad-support 940d
  backtest artifact. It again selects beta `0.25` and improves held-out quality:
  CRPS `-0.001146`, energy `-0.002094`, coverage `+0.000399` versus identity.
- The focused tests and py-compile checks pass.

### Issues Found

- `WARNING`: The so-called `66w` component backtest artifacts contain `29`
  scored windows in the JSON report. The name reflects the surrounding earlier
  run family, not the number of usable `window_scores` available to this
  calibration script. Do not claim a 66-window calibration evaluation from
  these artifacts.
- `WARNING`: 945r validates the calibration fit/evaluation against a broad
  support backtest report, but the qualitative fixed-start panels still come
  from the same saved current fixed-start decks. This is acceptable as
  replication of the calibration quality side, not as new broad-support visual
  evidence.
- `WARNING`: The mechanism remains a bounded post-rollout calibration overlay.
  It is stronger than the prior candidate, but should still be positioned as a
  support-gated calibration method rather than as direct prompt-to-scenario
  generation.

## My Independent Assessment

The 945q-r addendum strengthens the candidate enough for paper/demo use under
careful framing. The per-start gate rules out the concern that the average
conditionality result was carried by one favorable start. The broad-support
backtest replication rules out the narrowest calibration-fit artifact concern.

This is still not a fully promoted production default. The next public-facing
step should be to update the paper/demo as candidate evidence, or to ask for an
explicit product-owner default-promotion decision with the limitations above.

## Recommended Action

- Use 945q-r as the current paper/demo candidate evidence.
- Keep `support_gate_mode=direction_status`, start-only null, and per-start
  gates mandatory.
- Do not describe the result as a 66-window calibration run, and do not claim
  a solved direct text-to-latent bridge.
