# Independent Verifier: Live Demo Support-Gated Calibration Wiring

Date: 2026-05-26

## Verification Result

Verdict: `PARTIAL`

The live Gradio story path now applies the current support-gated narrative
ensemble calibration after the support-grounded frozen SNI rollout when the
direction-support gate passes. The implementation is visible in JSON and
markdown artifacts, and focused tests cover both the apply path and the
start-only support block. This supports using the calibrated live path as a
demo/paper candidate. It does not by itself promote the whole system as a
silent production default or prove broad live-run stability.

## What I Checked

- `experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py`
  - status display reports whether narrative ensemble calibration was applied
    (`prefix_latent_product_status_markdown`, lines 2093-2102).
  - markdown artifacts receive a `Live Demo Narrative Calibration` section
    (`_write_live_calibration_markdown`, lines 2521-2555).
  - live reports are calibrated by
    `apply_live_support_gated_ensemble_calibration` after rollout arrays are
    present and the direction gate passes (lines 2558-2688).
  - `run_prefix_latent_for_app` applies the live calibration only when rollout
    was not skipped (lines 3020-3022).
- `test_code/test_785a_nl_risk_manager_story_gradio_app.py`
  - imports the live calibration helper (line 15).
  - verifies an apply case shifts SPX up and VIX down, updates path quantiles,
    and writes the markdown section (lines 736-759).
  - verifies start-only support is blocked with `support_gate_blocked`
    (lines 762-788).
- Fresh live OpenAI smoke artifact:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/risk_manager_story_gradio_demo/prefix_latent_live_smoke/condition_only_run/prefix_latent_story_smoke_report.json`.
- Cached-condition smoke markdown:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/risk_manager_story_gradio_demo/prefix_latent_live_smoke/cached_casebook_run/condition_only_report/prefix_latent_story_smoke_report.md`.
- Current research candidate:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_narrative_ensemble_calibration_947b_full906b_66w_5start_broad_support_deck_matched_seed/narrative_ensemble_calibration_report.json`.

## Findings

### Confirmed Correct

- The live smoke used fresh OpenAI grounding and embedding metadata. The report
  records `live_app_openai_conditioning.status = fresh_condition_report` and
  `total_tokens = 1835`.
- The live smoke selected direction-checked, temporally diverse support:
  `3/8` support components, minimum gap `30`, no direction mismatches.
- Forward-looking language was handled as warning-only in the grounding
  sidecar: `FORWARD_RISK_IGNORED` appears in the live grounding warning list.
- The live smoke has calibration metadata:
  `mode = support_gated_directional_delta_calibration`,
  `applied = true`, `effective_beta = 0.25`, `support_gate = 1.0`,
  and `active_direction_count = 3`.
- The cached-condition markdown contains the `Live Demo Narrative Calibration`
  section with effective beta, support gate, and active directional claim count.
- The focused regression suite passed after the wiring:
  `60 passed in 2.85s` for
  `test_code/test_785a_nl_risk_manager_story_gradio_app.py`,
  `test_code/test_945a_nl_narrative_ensemble_calibration.py`, and
  `test_code/test_944a_nl_start_narrative_attribution.py`.

### Issues Found

- `WARNING`: This verification includes one fresh OpenAI live smoke and one
  cached-condition smoke, not a broad multi-narrative live regression.
- `WARNING`: The live calibration is a bounded post-rollout directional delta
  calibration over the frozen support-grounded SNI ensemble. It is not direct
  text-to-scenario generation and should not be described that way.
- `NOTE`: The verifier supports making the calibration visible in the demo and
  paper evidence path. It does not support hiding it as a production default
  without broader live validation and product acceptance checks.

## Alternative Explanations

- The fresh live smoke may be an easy narrative/start pair because the selected
  support passed all current gates. Broader narratives could still skip
  calibration if support direction evidence is weak or if the narrative has no
  active current/recent directional claims.
- The observed demo improvement is from a bounded directional calibration layer
  on top of the incumbent support-grounded rollout, not from a newly trained
  direct text decoder.

## My Independent Assessment

The code, tests, and smoke artifacts support the narrow claim: the live demo now
uses the current support-gated calibrated candidate in a transparent and bounded
way. The implementation also preserves the key safety behavior by refusing to
apply the calibration when the support evidence gate blocks it.

This is meaningful progress toward the product-facing demo because the paper
candidate is no longer disconnected from the UI path. The broader production
claim remains open.

## Recommended Action

Proceed with this as the demo-facing calibrated candidate and keep it visible in
reports. Do not mark the full narrative-conditioned generator production-ready
from this evidence alone. The next promotion step should be a multi-narrative
live or cached-live sweep showing that the calibrated demo path preserves
support provenance, warning behavior, and fixed-start conditionality across the
public narrative deck.
