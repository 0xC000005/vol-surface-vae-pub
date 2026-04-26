# 567a Risk-Manager Deployability Package for 564a

## Executive Decision

`564a` is risk-manager presentable as an IV-only conditional stress scenario generator, with a constrained deployment contract.

It should not be presented as a calibrated probability law or as a full multi-factor enterprise risk engine. The correct framing is:

- learned base law: `510a` final checkpoint;
- policy overlay: deterministic severity-stratified scenario selection;
- intended use: conditional stress challenge set for IV-surface risk review;
- non-intended use: calibrated probability forecasting, capital model approval, or joint cross-asset scenario generation.

## System Contract

Base learned law:

- checkpoint: `models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt`
- model type: `340c`
- input: 30-day IV-surface history
- output support: 30-day future IV-surface paths

Risk policy:

- generate `192` authentic candidate paths per history from the `510a` learned law;
- score each path by average future IV level;
- select `48` paths using calm, central, and stress severity bands;
- treat the selected paths as a stress deck, not as probability-weighted Monte Carlo draws.

Primary artifact:

- `results/autoresearch/564a_510a_stress_selected_policy/full11.json`

## Evidence Summary

| metric | 510a raw learned law | 564a stress deck | risk-manager read |
| --- | ---: | ---: | --- |
| suite score | `8/11` | `7/11` | raw score is not the product criterion |
| coverage90 | `0.873` | `0.897` | 564a is more conservative |
| worst-cell coverage h1/h7/h14/h30 | `0.714/0.745/0.745/0.729` | `0.745/0.740/0.740/0.760` | lower stress inclusion is acceptable |
| conditionality MAE reduction | `5.12%` | `6.85%` | 564a is more conditionally responsive |
| daily-change KS | `25/25` | `25/25` | local day-to-day moves remain realistic |
| level KS | `10/25` | `1/25` | accepted as a frequency warning, not a stress blocker |
| median-bias cells | `20/25` | `20/25` | median location remains controlled |
| cointegration worst-cell ratio | `0.257` | `0.222` | main near-miss caveat |
| regime layer2 | `0/8` | `1/8` | still incomplete, but not pure regime blindness |
| persistent severe undercoverage | `0.875%` | `0.583%` | severe stress omission is low |
| cross-cell corr ratio | `0.968` | `0.878` | dependence remains acceptable |
| effective-rank ratio | `1.462` | `1.604` | selected deck is more diverse |
| mean-reversion ratio | `0.986` | `1.003` | mean reversion is preserved |
| path max-jump KS | `0.361` | `0.488` | close to the relaxed `0.5` gate, still pass |

## Why 564a Is the Deployable Risk Prototype

The raw `510a` learned law is the correct object to report for scientific model quality. It remains the `8/11` learned-law frontier and has better formal level-frequency and cointegration metrics.

The `564a` system is the better risk-manager stress product because it converts the same learned law into a scenario deck that is more useful for risk review:

- it increases conditionality from `5.12%` to `6.85%`;
- it keeps lower stress inclusion acceptable at all audited horizons;
- it keeps surface validity, time-series behavior, daily-change realism, cross-cell dependence, mean reversion, and pathwise jump realism passing;
- it exposes calm, central, and high-IV future paths for the same conditioning history instead of pretending the selected scenario frequencies are calibrated probabilities.

This is an auditable policy calibration on top of a learned law, not a hidden model hack.

## Non-Negotiable Caveats

These caveats must be disclosed with any risk-manager presentation:

- Scenario frequencies are not probabilities. The selected set is intentionally severity-balanced.
- The system is IV-only. It does not yet generate returns, rates, credit, macro, or other portfolio factors.
- Regime layer2 remains weak: `1/8` cells pass, so regime-frequency allocation is not solved.
- Cointegration worst-cell ratio is a near-miss: `0.222` versus the `0.25` gate.
- Level KS is poor under selected stress sampling: `1/25`, which means historical level occupancy frequencies are deliberately distorted.
- Path max-jump realism is close to the relaxed gate: `0.488` versus `0.5`, so it should be monitored on any new split.

## Deployment Boundary

Acceptable use:

- IV-surface stress exploration;
- risk committee challenge scenarios;
- sensitivity analysis around calm/central/stress future paths;
- model-risk discussion of conditional scenario plausibility.

Not acceptable use:

- calibrated VaR or expected shortfall without separate probability calibration;
- capital model approval;
- multi-factor portfolio stress without an explicit factor extension;
- claims that the 48 selected paths are unbiased samples from the conditional law.

## Operational Recommendation

Ship `564a` as a prototype stress-deck generator if the deployment label is explicit:

`510a learned conditional IV law + 564a conservative stress-selection policy`.

For the research program, do not continue candidate-count tuning. The next genuine improvement path is either:

- extend the learned law to a true joint multi-factor scenario generator, or
- add a cointegration-aware selection diagnostic only if it can be justified as a transparent risk-policy constraint rather than a hidden evaluator-specific knob.
