# Conditionality-Aware Readout Calibration Intake

Date: 2026-05-12

## Problem

The current component-preserving narrative workflow selects distinct support
components for different narratives, but warning starts show weak generated path
separation relative to bootstrap/readout noise. The bottleneck is downstream of
text and support selection:

- support-overlap medians are `0.0` for starts `18`, `22`, `40`, and `77`;
- decoded-prefix distances for warning starts are comparable to or larger than
  the clean start `18`;
- generated path energy is much lower for warning starts;
- simple support sharpening and lower generator temperature help only
  directionally, not enough to clear the gate.

The latest fixed-decoder refresh sharpens this. Start `18` passes the
`384`-sample path gate cleanly, while start `22` remains a warning even with a
fixed/cached decoder seed. For start `22`, repeat path controls are below gate,
but bootstrap path energy and Wasserstein are too close to observed
cross-narrative differences. The refreshed start stratification is one `pass`,
three `warning`, and one `fail`, so the core problem is start-dependent
rollout/readout signal-to-noise, not uniform text-conditioning failure.

## Method Story

The next candidate should be a conditionality-aware readout/calibration layer,
not another text embedding or support-ranker feature. The goal is to calibrate
spread and preserve realistic path dependence without washing out narrative
differences.

The weather and hydrology ensemble-postprocessing literature is directly
relevant. Ensemble Copula Coupling (ECC) is used when univariate calibration
improves marginal reliability but can lose multivariate, spatial, or temporal
dependence. ECC reconstructs scenario dependence using the raw ensemble rank
structure. Related work also warns that shuffling/calibration can fail when the
template dependence is mismatched, so the dependence template must be audited.

For this project, the analogue is:

```text
component support rollout paths
-> marginal/readout calibration
-> preserve narrative-conditioned rank/path template
-> evaluate CRPS/energy/coverage and fixed-start conditionality
```

## Related Work Basis

- Ensemble Copula Coupling for calibrated ensemble forecasts: DTU record for
  dynamic ECC, https://orbit.dtu.dk/en/publications/generation-of-scenarios-from-calibrated-ensemble-forecasts-with-a-2
- ECC and member-by-member postprocessing preserve correlation/dependence
  structures after calibration:
  https://www.researchgate.net/publication/316558895_Ensemble_calibration_with_preserved_correlations_Unifying_and_comparing_ensemble_copula_coupling_and_member-by-member_postprocessing
- Schaake shuffle and empirical copula methods are useful but can be problematic
  when the dependence template is wrong:
  https://www.sciencedirect.com/science/article/pii/S0022169420304510
- Multivariate quantile mapping literature frames the same issue: univariate
  postprocessing needs an empirical copula/dependence template:
  https://www.sciencedirect.com/science/article/pii/S2212094721000086

## Candidate Design

Start with a guarded readout-only candidate:

1. Keep the narrative support selection and component-preserving rollout fixed.
2. Estimate marginal calibration transforms from held-out historical backtests.
3. Apply calibration in a rank-preserving way so each narrative's pathwise
   component/template ordering remains intact.
4. Evaluate two views:
   - base uncalibrated component rollout for conditionality;
   - calibrated ECC-style readout for coverage/distributional quality.
5. Require the calibrated view to retain a minimum fraction of uncalibrated
   conditionality signal.
6. Select any readout through a joint gate: held-out quality must improve and
   fixed-start path audits must pass without bootstrap/repeat warnings.

## Novelty Claim

The contribution would not be "ECC for finance" by itself. The local novelty is
using a support-grounded narrative-conditioned latent scenario generator, then
postprocessing the frozen generator output with a calibration layer that is
explicitly constrained to preserve narrative-conditioned path dependence.

## Falsifier

Kill or downgrade this candidate if:

- calibrated CRPS/energy/coverage improves but fixed-start conditionality
  collapses below repeat/bootstrap controls;
- the rank/dependence template is not stable across historical backtest splits;
- the method requires many hand-tuned factor-specific rules;
- it improves display metrics but makes the audit less understandable for risk
  managers.

## Minimal Next Experiment

Completed first-stage diagnostics:

- current uncalibrated component rollout;
- existing global mean-preserving fan scale;
- rank-preserving marginal calibration/ECC-style readout.

Findings:

- uncalibrated component rollout is still the safest conditionality-preserving
  default;
- marginal rank-preserving readout improves CRPS/energy materially but fails
  fixed-start conditionality;
- a tiny global fan scale `alpha=1.05` passes the start-18 path gate and gives
  a small held-out quality gain;
- larger global scales improve historical backtest quality more but trigger
  bootstrap warnings;
- start `22` remains warning after fixed-decoder correction, so the next
  candidate must handle start-dependent signal-to-noise, not only marginal fan
  width.

Next experiment:

Build a start-aware readout/gating diagnostic that reports when a selected
start is `clean`, `warning`, or `incompatible`, and tests whether any
calibration candidate improves historical backtests while passing the gate on
more than only start `18`. Do not add factor-specific hand rules or another
text model before this gate is satisfied.

## Start-Aware Readout Gate Result

The first start-aware readout gate is implemented in
`experiments/backfill/block_ar/nl_prefix_latent_start_aware_readout_gate.py`.
It combines the existing readout selector with fixed-start conditionality
audits, so the selected readout must improve held-out quality and pass on
multiple starts before being promoted broadly.

Artifact:
`experiments/backfill/block_ar/nl_scenario_demo_outputs/prefix_latent_start_aware_readout_gate_919b/start_aware_readout_gate.json`.

Result:

- selected readout: `alpha1p05`;
- status: `warning`;
- recommendation: `keep_selected_readout_as_local_candidate_only`;
- start counts: one `pass` and three `warning`;
- clean start: `18`;
- warning starts: `22`, `40`, `77`;
- findings: `too_few_clean_starts_for_broad_promotion` and
  `warning_starts_remain_bootstrap_limited`.

Interpretation: the tiny global fan scale remains useful as a local candidate
because it improves held-out quality and preserves the clean start-18
conditionality gate. It should not be promoted as the broad product readout:
starts `22`, `40`, and `77` still show narrative response versus repeat and
start-only controls, but the within-run bootstrap/readout noise remains too
large relative to the cross-narrative path separation.

Updated next experiment: build either a conditionality-aware readout that adapts
to start/readout signal-to-noise without factor-specific hand rules, or a
portfolio-risk-aware support response label that measures whether narrative
differences matter in portfolio risk space even when individual factor fans are
bootstrap-limited.
