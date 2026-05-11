# Verification Result

Verdict: `AGREE`

The narrow claim is supported: in the full 192-path fixed-start control run, five of six selected starts pass the fixed-start narrative controls, while `fixed_start_178` remains repeat-unstable and should receive a high-instability warning rather than a normal pass.

## What I Checked

- Code:
  - `experiments/backfill/block_ar/nl_prefix_latent_fixed_start_control_suite.py`
  - `experiments/backfill/block_ar/nl_prefix_latent_start_reliability_gate.py`
- Artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_control_suite_865a_full_narrative_s192/start_conditioned_bakeoff.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_control_suite_865d_full_s192_symmetric/fixed_start_control_suite.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_start_reliability_gate_865d_full_s192_symmetric/start_reliability_gate.json`
- Checks:
  - Confirmed the control suite now propagates per-start failures to top-level `fail`.
  - Confirmed the reliability gate maps per-start failures to `warn_high_instability`.
  - Confirmed the reported pass/warn counts and ratios from JSON artifacts, not prose.

## Findings

### Confirmed Correct

- The 192-path narrative bakeoff covers 36 runs and reports `status: pass`.
- The full 192-path control suite reports `status: fail` only because of `per_start_control_failure`.
- Per-start controls:
  - `fixed_start_0`: pass; bootstrap ratio `0.480`, repeat ratio `0.253`.
  - `fixed_start_18`: pass; bootstrap ratio `0.339`, repeat ratio `0.375`.
  - `fixed_start_22`: pass; bootstrap ratio `0.398`, repeat ratio `0.393`.
  - `fixed_start_40`: pass; bootstrap ratio `0.381`, repeat ratio `0.309`.
  - `fixed_start_77`: pass; bootstrap ratio `0.420`, repeat ratio `0.473`.
  - `fixed_start_178`: fail; bootstrap ratio `0.683`, repeat ratio `0.764`, failure `repeat_noise_close`.
- The gate manifest reports `status_counts: {"pass": 5, "warn_high_instability": 1}` and maps `fixed_start_178` to `warn_high_instability`.

### Issues Found

- `WARNING`: Repeat stability is based on two repeat seeds. This is enough for a promotion guardrail, not enough for a final production reliability certificate.
- `NOTE`: The 36-run bakeoff still has 6 validation warnings in `operational_status_counts`, even though direction checks and scenario metrics pass. Product messaging should separate scenario validation from start-reliability status.

## Alternative Explanations

- The remaining `fixed_start_178` failure is more consistent with start-specific repeat instability than generic sample-count noise: bootstrap ratio passes at 192 (`0.683`), but repeat ratio remains just above the `0.75` threshold (`0.764`).
- The five passing starts may still be sensitive to a wider seed set, but the current evidence rules out the earlier simpler explanation that all warnings were caused by low sample count.

## My Independent Assessment

Proceed with a restricted start-reliability manifest: five starts can be shown as supported under the current evidence, and `fixed_start_178` should be allowed only with a prominent high-instability warning. Do not claim universal production readiness for arbitrary starts yet.

## Recommended Action

- Use `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_start_reliability_gate_865d_full_s192_symmetric/start_reliability_gate.json` as the current demo reliability manifest.
- Before a paper-facing or boss-facing production claim, add a wider repeat-seed check or state clearly that the current repeat gate uses two seeds.
- Keep `fixed_start_178` as a hard-case example in the casebook, not as a recommended default start.
