# Independent Verification: Component-Preserving Narrative Rollout Backtest

Verification Result

Verdict: `PARTIAL`

The core mechanism claim is supported: preserving support components through
rollout fixes the visible fixed-start narrative-conditionality failure, and the
29-window held-out cached-narrative backtest is internally consistent. The result
is strong enough to keep `component_prefix_mixture` as the candidate production
path, but not enough to call it fully promoted or production-ready.

## What I Checked

- Implementation:
  - `experiments/backfill/block_ar/nl_prefix_latent_story_smoke.py`
  - `experiments/backfill/block_ar/nl_prefix_latent_component_backtest.py`
  - `experiments/backfill/block_ar/plot_narrative_casebook_backtest.py`
- Artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_backtest_heldout_29w_s32/component_backtest_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_mixture_fixed_start_900a_s96/*/fixed_start_18/decoder_component_topk_narrative_start_checked_gen_temp_0p50/prefix_latent_story_smoke_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_mixture_fixed_start_900a_s96/factor_conditionality_raw_level_component.png`
- Checks:
  - recomputed summary metrics from `window_scores`;
  - checked that component and averaged reports use matching sample shapes;
  - checked component sample allocation sums;
  - checked all fixed-start casebook reports use the same `factor:spx` start;
  - reran py_compile and focused pytest suites.

## Confirmed Correct

- The new component path is not just another averaged-memory rollout. It calls
  `_component_memory_rows_for_prior`, decodes each support memory separately via
  `_decode_prefix_for_component_memory`, rolls each component under the same
  selected raw start, and concatenates weighted samples.
- Held-out report summary is reproducible from saved `window_scores`:
  `summary_recompute_ok True`.
- Held-out run scored `29` windows with `0` failures.
- Component artifacts have matching rollout shapes to averaged-prefix artifacts:
  `(1, 32, 30, 39)` for both `samples` and `generated_states` in inspected runs.
- Component sample allocation sums to the requested `32` samples for all
  `29` held-out runs.
- Fixed-start casebook uses the same raw SPX starting level for all six
  narratives: `2049.580078125`, with zero spread across narratives.
- Fixed-start diagnostic supports the visual conditionality claim:
  median standardized terminal sample correlation is `0.0445`, median normalized
  quantile-shape L2 is `0.0978`, and median terminal KS is `0.1823`.
- Held-out distributional metrics are slightly better than averaged-prefix:
  component CRPS improvement versus persistence is `+12.64%` versus averaged
  `+12.56%`; component energy improvement is `+15.34%` versus averaged `+15.32%`.

## Issues Found

- `WARNING`: This is a cached held-out narrative backtest, not a live OpenAI
  production evaluation. It tests the downstream prefix-latent rollout path with
  cached bridge queries.
- `WARNING`: The sample budget is modest: `32` samples per held-out run and
  `250` decoder steps. This is adequate for a candidate gate, not for final
  paper-grade confidence intervals.
- `WARNING`: 80% coverage is low (`0.2841` for component, `0.2768` for averaged).
  The method improves CRPS/energy, but it is not calibrated as an 80% interval
  generator yet.
- `WARNING`: Point-path metrics remain weaker than persistence:
  component mean-path MAE improvement is `-2.93%` and terminal MAE improvement is
  `-2.30%`. Claims must remain distributional, not point-forecasting.
- `NOTE`: In fixed-start casebook reports with both diagnostic and operational
  variants, the old `rollout_component_count` field counted all variants. The
  code now also records operational-only component count and sample count.
- `NOTE`: The fixed-start plot is a six-narrative, one-start diagnostic. It is
  persuasive for the failure mode but should not replace the broader fixed-start
  grid and null/repeat controls.

## Alternative Explanations

- The stronger visible conditionality may partly reflect different support pools
  rather than a learned latent refinement. That is acceptable for the current
  support-grounded contract, but it should be described honestly as
  component-preserving support-mixture conditioning.
- The component path is slightly better than averaged-prefix on this held-out
  backtest, but the margin is tiny. The main win is product-facing stochastic
  family separation, not a large backtest-score improvement.

## My Independent Assessment

Keep `component_prefix_mixture` as the candidate production path for
narrative-visible conditionality. Do not yet promote it as final. The evidence
supports the mechanism and removes the earlier visual failure, but final
promotion still needs a larger or repeated held-out evaluation, calibration
diagnostics, and the standard null/repeat fixed-start controls.

## Recommended Action

Proceed to candidate hardening:

1. rerun the component-vs-averaged backtest with a larger sample budget;
2. add calibration/coverage diagnostics to the paper-facing evaluation;
3. keep `averaged_prefix` as a diagnostic baseline;
4. avoid claiming point-forecast quality;
5. update paper/demo language to say the method improves distributional,
   support-grounded conditionality, not deterministic forecast accuracy.

## Addendum: Larger Sample Hardening Run

After the verifier pass, the same `29` held-out windows were rerun with `96`
samples per method and the same `250` decoder steps.

Report:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_backtest_heldout_29w_s96/component_backtest_report.json`

Results:

- averaged-prefix CRPS improvement versus persistence: `+12.83%`;
- component-prefix-mixture CRPS improvement versus persistence: `+13.15%`;
- averaged-prefix energy improvement versus persistence: `+15.60%`;
- component-prefix-mixture energy improvement versus persistence: `+15.80%`;
- component minus averaged mean CRPS: `-0.00248`;
- component minus averaged mean energy: `-0.00242`;
- component 80% coverage: `0.2957`;
- averaged 80% coverage: `0.2896`;
- component sample allocation sums to `96` for all `29` held-out windows.

This strengthens the candidate claim: the component-preserving mixture is not
only visually more conditional, it remains slightly better than averaged-prefix
on held-out distributional scores at a larger sample budget. The previous
warnings still apply: coverage is low and point-path metrics are weaker than
persistence, so the claim remains distributional.

## Addendum: Calibration And Fixed-Start Control Gates

Calibration report:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_calibration_29w_s96/component_backtest_calibration_report.json`

Component fixed-start control report:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_fixed_start_controls_901a_s96/component_fixed_start_controls.json`

Calibration findings:

- component overall raw-level 80% coverage: `0.2957`;
- averaged overall raw-level 80% coverage: `0.2896`;
- component terminal raw-level 80% coverage: `0.2485`;
- averaged terminal raw-level 80% coverage: `0.2502`;
- component minus averaged overall coverage: `+0.0061`;
- component minus averaged PIT uniform-error: `-0.0014`;
- worst component-coverage channels include `factor:spx`, `iv:07`,
  `iv:20`, `iv:19`, `iv:21`, and `factor:bbb_oas`.

Fixed-start control findings:

- observed narrative median gap: `1.3540`;
- within-run bootstrap median gap: `0.7346`, ratio `0.543`;
- same-narrative repeat median gap: `0.3512`, ratio `0.259`;
- start-only null median gap: `0.0000`, ratio `0.000`;
- status: `pass`, no warnings or failures.

Interpretation: component-preserving mixture now passes the immediate
conditionality control gate: narrative gaps are larger than sampling noise,
repeat-seed noise, and start-only null effects. The remaining blocker is
calibration. The generated 10-90% fan is too narrow or miscentered for realized
historical paths, especially in SPX and selected IV cells.

## Addendum: Global Fan Calibration Candidate

Calibration script:

`experiments/backfill/block_ar/nl_prefix_latent_component_global_calibration.py`

Chronological split report:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_global_calibration_902a_29w_s96/component_global_calibration_report.json`

Reverse split report:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_global_calibration_902b_reverse_29w_s96/component_global_calibration_report.json`

Even/odd split report:

`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_component_global_calibration_902c_evenodd_29w_s96/component_global_calibration_report.json`

Mechanism: scale generated delta paths around their ensemble mean by a single
global `alpha`, so the mean path is unchanged and only fan width changes.

Split-stability result:

- chronological split selected `alpha=3.5`;
- reverse split selected `alpha=3.5`;
- even/odd split selected `alpha=3.5`.

Evaluation-row improvements versus uncalibrated component:

- chronological: coverage `+0.4771`, CRPS `-0.0750`, energy `-0.1910`;
- reverse: coverage `+0.4915`, CRPS `-0.0789`, energy `-0.1058`;
- even/odd: coverage `+0.4936`, CRPS `-0.0787`, energy `-0.1471`.

Full 29-window summary at `alpha=3.5`:

- calibrated coverage: `0.7802` versus uncalibrated component `0.2957`;
- calibrated CRPS improvement versus persistence: `+23.06%` versus
  uncalibrated component `+13.15%`;
- calibrated energy improvement versus persistence: `+28.25%` versus
  uncalibrated component `+15.80%`;
- mean-path and terminal-MAE metrics are unchanged up to floating-point noise
  because the calibration preserves the ensemble mean.

Interpretation: the main calibration blocker is likely under-dispersion, not
wrong narrative support selection. A single global fan-width scale materially
improves coverage, CRPS, and energy across several train/eval splits. This is a
candidate post-hoc calibration layer, not a change to the narrative-conditioning
mechanism.
