# Independent Verification: Global Fan Calibration Wiring

Verification date: 2026-05-12

## Verification Result

Verdict: `PARTIAL`

The implementation correctly wires a one-parameter, mean-preserving fan-width
calibration into the prefix-latent story-smoke runner and the Gradio demo
argument path. The held-out artifacts support the claim that `alpha=3.5`
materially improves raw-level coverage, CRPS, and energy on the current
29-window benchmark. The remaining limitation is scope: this is still a
candidate calibration layer validated on the available held-out narrative
benchmark, not a fully production-promoted uncertainty calibration.

## What I Checked

- Code:
  - `experiments/backfill/block_ar/nl_prefix_latent_story_smoke.py`
  - `experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py`
  - `experiments/backfill/block_ar/nl_prefix_latent_component_global_calibration.py`
  - `test_code/test_901a_nl_prefix_latent_component_gates.py`
  - `test_code/test_785a_nl_risk_manager_story_gradio_app.py`
- Artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_global_calibration_902a_29w_s96/component_global_calibration_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_global_calibration_902b_reverse_29w_s96/component_global_calibration_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_global_calibration_902c_evenodd_29w_s96/component_global_calibration_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_calibrated_app_smoke_902d_s4/prefix_latent_story_smoke_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_calibrated_app_smoke_902d_s4/prefix_latent_story_smoke_arrays.npz`
- Commands:
  - `python -m py_compile experiments/backfill/block_ar/nl_prefix_latent_component_global_calibration.py experiments/backfill/block_ar/nl_prefix_latent_story_smoke.py experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py test_code/test_901a_nl_prefix_latent_component_gates.py test_code/test_785a_nl_risk_manager_story_gradio_app.py`
  - `pytest test_code/test_901a_nl_prefix_latent_component_gates.py test_code/test_784a_nl_risk_manager_story_smoke.py test_code/test_785a_nl_risk_manager_story_gradio_app.py -q`
  - cached story-smoke run with `--rollout-fan-scale 3.5`

## Findings

### Confirmed Correct

- `scale_delta_samples_around_mean` scales generated delta samples across the
  sample axis per variant. This preserves each variant's ensemble mean and
  changes only dispersion.
- The story-smoke runner applies `--rollout-fan-scale` after either
  `component_prefix_mixture` or `averaged_prefix` rollout and recomputes
  generated raw states from the calibrated deltas.
- The runner records `rollout_fan_scale` and a `rollout_fan_calibration` block
  in `generation`, and saves `uncalibrated_samples` in the arrays artifact for
  audit.
- The Gradio app passes `rollout_fan_scale=3.5` through
  `build_prefix_latent_run_args` and displays calibrated-path status text.
- The cached app smoke produced:
  - `rollout_fan_scale = 3.5`;
  - `rollout_fan_calibration.applied = true`;
  - generated shape `[2, 4, 30, 39]`;
  - calibrated and uncalibrated sample means matching to max difference
    `1.9073486328125e-06`.
- Unit/regression tests passed: `55 passed in 3.74s`.

### Issues Found

- `WARNING`: The calibration was selected on the current 29-window benchmark.
  It should be treated as a candidate global calibration layer until it is
  checked on a larger/newer manifest and ideally by factor family.
- `WARNING`: The Gradio default now uses the calibrated fan scale, while the CLI
  default remains `1.0`. This is acceptable for demo safety if documented, but
  paper/report language should distinguish calibrated-demo output from
  uncalibrated research diagnostics.
- `NOTE`: This calibration does not improve mean-path accuracy. It is an
  uncertainty/fan-width correction only, which is consistent with the code.

## Alternative Explanations

- The improved coverage could reflect broad under-dispersion of the frozen
  generator rather than better narrative conditioning. The component-preserving
  mixture still supplies the conditional mean/support mechanism; the fan scale
  should not be described as a new text-to-latent method.
- A single global alpha may be masking factor-family differences. SPX, selected
  IV cells, and credit-spread channels were the known weak spots, so a future
  factor-family calibration may be more faithful.

## My Independent Assessment

Proceed with the wired calibration as a demo-candidate and paper-facing
diagnostic. Do not yet call it fully production-promoted. The evidence supports
that the first-order failure was under-dispersion, and the implementation is
narrow enough that it does not change the narrative-conditioning mechanism or
hide the support mixture.

## Recommended Action

Keep `alpha=3.5` in the Gradio demo path for now, document it as calibrated fan
width, and run the next autoresearch iteration on a larger manifest or a
factor-family calibration split before promoting it as a production default.
