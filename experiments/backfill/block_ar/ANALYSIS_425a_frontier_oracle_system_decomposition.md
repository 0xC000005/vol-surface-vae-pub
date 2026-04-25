# 425a: Frontier / Oracle-System Decomposition

## Context

The active learned frontier remains `392a` at `8/11`.

It is still the cleanest model in the current line: an empirical normal-score AR
transition-flow generator, weakly fine-tuned with a free-running path-energy score. It
passes the suites that require local conditional mechanics and stochastic geometry:

- surface validity;
- conditionality;
- time-series statistics;
- block-AR mechanics;
- cointegration;
- cross-cell correlation;
- mean reversion;
- pathwise jump realism.

The persistent failures are:

- coverage;
- regime coverage;
- distributional fidelity, mostly level KS rather than daily-change KS.

## Evidence From Closed Repairs

Recent attempts separate the failure mechanism cleanly:

| Branch | Score | Key Read |
| --- | ---: | --- |
| `392a` | `8/11` | Best learned frontier; strong local/structural mechanics, weak long-horizon level/regime occupancy. |
| `405a` interval scaling | `6/11` | More coverage width did not solve worst-cell coverage and damaged conditionality/time-series. |
| `407a` deadband scaling | `7/11` | Smaller intervention preserved more structure but still failed coverage/regime/distributional suites. |
| `419a` student-forced transition FM | `6/11` | Off-policy AR training acted mainly as a width actuator and worsened level occupancy. |
| `421a` joint transition-path FM | `4/11` | Joint path law collapsed cross-cell stochastic geometry and mean-reversion structure. |
| `423a` persistent source noise | `6/11` | Path-persistent noise preserved correlation but overbroadened levels and broke cointegration/path jumps. |

The shared pattern is not "model too small" or "missing one scalar knob." Width,
student-forcing, joint path mixing, and persistent source noise all move the same
frontier tradeoff: they can alter dispersion, but they do not learn the validation
level-occupancy law while preserving the structural suites that `392a` already passes.

## Mechanism Read

`392a` is a strong learned conditional transition law, but the remaining tests ask for
the aggregate 30-day scenario universe to match historical marginal level occupancy and
regime-cell reliability. Those requirements are valid for a risk scenario product, but
the evidence so far says they are only weakly identified by the available conditional
signal in the learned generator.

That makes the next research question different:

- not "which architecture variant is next";
- but "is `11/11` feasible as a reported risk system if the base learned law is kept
  separate from a controlled calibration or oracle layer?"

## Decision

Stop rotating learned-model variants inside the same AR/path-flow family until an oracle
feasibility bound is measured.

The next iteration should implement a diagnostic, not a claimed final model:

1. Freeze the `392a` base generator.
2. Apply the least invasive validation-oracle monotone correction needed to target the
   remaining failed suites.
3. Evaluate the unchanged full 11-suite.
4. Report base learned-law metrics and oracle/calibrated-system metrics separately.

If even an oracle monotone system cannot exceed `8/11` without breaking structural suites,
then the current test/product framing is internally too hard for this data/model setup.
If the oracle system reaches or approaches `11/11`, the next problem becomes replacing
the oracle with a defensible pre-validation policy calibration while keeping the paper
honest about what is learned versus calibrated.

