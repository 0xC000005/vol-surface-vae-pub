# Independent Verifier: Fixed-Start Full Path-Distribution Audit

Date: 2026-05-12

## Verification Result

Verdict: `PARTIAL`

The implementation and saved artifacts support the narrower claim that the
component-preserving narrative support mixture produces measurable full-path
distribution differences under an exactly fixed start, and that these
differences exceed same-narrative repeat and start-only controls on several
path metrics. They do not support a fully promoted production claim yet,
because bootstrap path noise remains close to the observed narrative signal,
the case set is still small and selected, and terminal shape-only
conditionality remains a negative result.

## What I Checked

- Code:
  - `experiments/backfill/block_ar/nl_prefix_latent_fixed_start_shape_audit.py`
  - `experiments/backfill/block_ar/nl_prefix_latent_component_fixed_start_controls.py`
  - `test_code/test_901a_nl_prefix_latent_component_gates.py`
- Artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start_controls_904f_s192_uncalibrated/component_fixed_start_controls.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_path_audit_904f_s192_uncalibrated/fixed_start_shape_audit.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_path_audit_904g_s192_calibrated_from_uncalibrated/fixed_start_shape_audit.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_path_audit_904f_s192_uncalibrated/fixed_start_narrative_path_metric_summary.png`
- Documentation:
  - `docs/research_protocols/nl_prefix_latent_current_truth.md`
  - `RESEARCH_LOG.md` tail entry `HEAD nl-prefix 154 fixed-start full path-distribution audit`
- Commands:
  - `python -m py_compile experiments/backfill/block_ar/nl_prefix_latent_component_fixed_start_controls.py experiments/backfill/block_ar/nl_prefix_latent_fixed_start_shape_audit.py test_code/test_901a_nl_prefix_latent_component_gates.py`
  - `uv run pytest test_code/test_901a_nl_prefix_latent_component_gates.py test_code/test_784a_nl_risk_manager_story_smoke.py test_code/test_785a_nl_risk_manager_story_gradio_app.py -q`

## Confirmed Correct

- The audit now separates:
  - `path_distribution_status` as the product-facing gate;
  - `terminal_shape_only_status` as a stricter diagnostic.
- The saved `904f` uncalibrated audit has:
  - status `warning`;
  - no path-distribution failures;
  - warning `bootstrap_path_noise_close_to_observed`;
  - repeat-to-observed path energy ratio `0.238`;
  - repeat-to-observed path variance ratio `0.338`;
  - repeat-to-observed path Wasserstein ratio `0.621`;
  - start-only path energy, variance, and Wasserstein ratios all `0.0`.
- The saved `904g` calibrated audit has:
  - status `warning`;
  - no path-distribution failures;
  - warnings `bootstrap_path_noise_close_to_observed` and
    `bootstrap_energy_noise_close_to_observed`;
  - repeat-to-observed path energy ratio `0.405`;
  - repeat-to-observed path variance ratio `0.338`;
  - repeat-to-observed path Wasserstein ratio `0.600`;
  - start-only path energy, variance, and Wasserstein ratios all `0.0`.
- The fixed-start control report exists and passed with:
  - repeat-to-observed median ratio `0.423`;
  - bootstrap-to-observed median ratio `0.619`;
  - start-only-to-observed median ratio `0.0`.
- The plotted path-metric summary is non-empty and visually matches the report:
  observed gaps exceed repeat and start-only controls, while bootstrap remains
  close for path Wasserstein.
- The focused test suite passed: `58 passed in 2.62s`.

## Issues Found

- `WARNING`: This is still a selected fixed-start casebook-style audit, not a
  broad historical manifest result. It should not be generalized to arbitrary
  narratives or arbitrary starts without larger manifest hardening.
- `WARNING`: Bootstrap path noise is close enough to observed narrative gaps to
  keep the product status at warning. The calibrated audit is especially mixed:
  bootstrap path energy and path Wasserstein exceed observed narrative ratios.
- `WARNING`: Terminal shape-only conditionality still fails. After removing
  each terminal distribution's own location and scale, same-narrative repeat
  controls are still too close to observed narrative differences. The valid
  claim is full-path distribution response, not robust scale-free terminal
  shape response.
- `NOTE`: The fixed-start control report's primary ratio is a terminal
  standardized mean-gap summary. The full-path audit recomputes stronger path
  metrics, so the control report is useful supporting evidence but not by
  itself sufficient for the path-distribution claim.
- `NOTE`: In `path_event_metrics`, the variable named `terminal_scale` is
  computed from variance over the whole path matrix, not just terminal values.
  This is acceptable as a path-event scale if intended, but the name is
  misleading and event-threshold sensitivity should be checked before
  promotion.
- `NOTE`: Applying the global fan-width scale changes some path audit ratios.
  This calibration layer should continue to be described as uncertainty/fan
  calibration, not as a narrative-conditioning mechanism.

## Alternative Explanations

- Some observed narrative separation may come from support-set selection and
  ensemble-width differences rather than a deeper learned text-to-latent
  representation. That is acceptable for the current support-grounded method,
  but the paper/demo should say this clearly.
- Bootstrap closeness suggests sample-budget and stochastic rollout effects
  are still material. A larger sample budget or repeated-seed manifest could
  reduce uncertainty before promotion.

## My Independent Assessment

The new audit is a real improvement over the earlier terminal-only diagnostic.
It matches the user's product concern better: under the same starting level,
different narratives should change the whole future path distribution. The
evidence is directionally positive, but not decisive. I would keep the
component-preserving support mixture as the candidate path and keep the
full-path audit as the correct product gate, but I would not call the
conditionality problem solved yet.

## Recommended Action

Proceed with larger-manifest hardening and a repeated-seed/stochastic-noise
stability check before promoting the claim. Keep terminal shape-only metrics as
a diagnostic limitation. Do not add another text-to-latent architecture knob
until this gate is stable on broader fixed-start cases.
