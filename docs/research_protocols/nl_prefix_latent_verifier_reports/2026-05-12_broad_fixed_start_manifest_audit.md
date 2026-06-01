# Independent Verification: Broad Fixed-Start Manifest Audit

Verification date: 2026-05-12

## Verification Result

Verdict: `PARTIAL`

The new manifest audit correctly reads cached fixed-start artifacts, groups
cross-narrative comparisons within the same start, and reports that observed
narrative differences exceed same-narrative repeat, bootstrap, and start-only
controls. The narrow claim "the cached broad soft-topk fixed-start suite shows
path-distribution narrative sensitivity across six starts" is supported.

The broader claim "the current component-preserving support-mixture production
candidate has now passed broad fixed-start promotion" is not yet supported by
this artifact. The observed suite is
`decoder_soft_topk_narrative_start_checked_gen_temp_0p50`, while the newer
component-preserving candidate uses the component-mixture path. This audit is
therefore a diagnostic broadening result, not a production promotion.

## What I Checked

- Audit implementation:
  `experiments/backfill/block_ar/nl_prefix_latent_fixed_start_manifest_audit.py`
- Test coverage:
  `test_code/test_904p_nl_prefix_latent_fixed_start_manifest_audit.py`
- Broad audit artifact:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_manifest_audit_904q_s192_with_start_only/fixed_start_manifest_audit.json`
- Observed narrative source:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_control_suite_865a_full_narrative_s192/start_conditioned_bakeoff.json`
- Start-only source:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_control_suite_862d_full_start_only_s96/start_conditioned_bakeoff.json`
- Repeat source:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_fixed_start_control_suite_865b_full_s192_repeat`
- Research-log entry:
  `RESEARCH_LOG.md` lines `125438`-`125506`

## Confirmed Correct

- The report covers `36` observed cases, `72` repeat cases, and `36`
  start-only cases.
- Cross-narrative pairs are formed only within each fixed start, for `90`
  observed pairs.
- Same-narrative repeat controls produce `36` pairs.
- Start-only controls produce `90` pairs.
- Max per-start start difference is `0.0`.
- The reported pass metrics match the artifact:
  - repeat-to-observed path energy: `0.381`;
  - repeat-to-observed path variance: `0.533`;
  - repeat-to-observed path Wasserstein: `0.641`;
  - bootstrap-to-observed path energy: `0.294`;
  - bootstrap-to-observed path variance: `0.417`;
  - bootstrap-to-observed path Wasserstein: `0.671`;
  - start-only path energy, variance, and Wasserstein ratios: `0.0`.
- Local validation passed:
  - `python -m py_compile experiments/backfill/block_ar/nl_prefix_latent_fixed_start_manifest_audit.py test_code/test_904p_nl_prefix_latent_fixed_start_manifest_audit.py`
  - `uv run pytest test_code/test_904p_nl_prefix_latent_fixed_start_manifest_audit.py -q`
  - `uv run pytest test_code/test_901a_nl_prefix_latent_component_gates.py test_code/test_904m_nl_prefix_latent_calibration_conditionality.py test_code/test_904p_nl_prefix_latent_fixed_start_manifest_audit.py -q`

## Issues Found

- `WARNING`: The observed broad suite is not the current
  component-preserving rollout candidate. It uses
  `decoder_soft_topk_narrative_start_checked_gen_temp_0p50`. Do not use this
  report alone to promote the component-preserving production path.
- `WARNING`: The start-only null suite has `96` samples while the observed and
  repeat suites use `192` samples. The zero start-only result is expected
  because start-only removes narrative ranking; still, the sample-budget
  mismatch should be mentioned when citing the artifact.
- `NOTE`: The audit is a conditionality diagnostic, not a realized-future
  backtest. It does not establish CRPS, energy, coverage, or factor-family
  calibration quality for the broad component-preserving path.

## Alternative Explanations

- The broad pass may reflect that the older soft-topk narrative path already
  had narrative sensitivity across starts, while the component-preserving path
  still needs its own broad manifest check.
- The zero start-only ratios are consistent with a correct no-narrative null,
  but they do not replace a same-sample-budget regenerated start-only suite for
  final promotion.

## My Independent Assessment

The audit is useful and should be kept. It closes the immediate missing-null
gap for the older cached broad fixed-start suite and supports the statement that
fixed-start narrative sensitivity is not only a one-case artifact.

It should not be framed as final production evidence for the latest
component-preserving support-mixture method. The next promotion-quality step is
to either generate a broad component-preserving fixed-start suite with matching
observed, repeat, bootstrap, and start-only controls, or explicitly mark this
as historical soft-topk broadening evidence while keeping component-preserving
promotion tied to the narrower 900/904 artifacts.

## Recommended Action

Proceed with the diagnostic result, but modify current-truth language to avoid
overclaiming. The next HEAD iteration should build or run the same broad audit
on the current component-preserving rollout path before promotion.
