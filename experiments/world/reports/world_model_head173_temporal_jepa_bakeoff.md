# World Model HEAD173: Temporal JEPA Frozen Bakeoff

Date: 2026-05-11

## Iteration Type

`experiment`

## Objective Family

`downstream_probe_frozen_bakeoff`; this evaluates frozen feature
surfaces and does not turn future targets into Part 1 pretraining.

## Literature Status

`frozen_evaluation_protocol_from_ijepa_vjepa_tsjepa`: the comparison
follows the JEPA habit of judging representation quality through frozen
downstream probes and baselines, not pretraining loss alone.

## Hypothesis

If HEAD172's temporal JEPA signal is useful, the same train/validation
probe contract should show incremental raw+learned value versus raw
features, random temporal features, and the scaled Barlow candidate.

## Falsifier

The temporal branch remains non-promotable if its raw+learned surface does
not beat raw and scaled-Barlow surfaces on the same frozen probes, or if
rank/health remains weak.

## Feature Surfaces

- `raw_surface_last`
- `raw_surface_flat`
- `temporal_jepa_last`
- `raw_surface_last_plus_temporal_jepa_last`
- `random_temporal_last`
- `raw_surface_last_plus_random_temporal_last`
- `scale_barlow_last`
- `raw_surface_last_plus_scale_barlow_last`

## Current IV Guardrail

| feature | MSE | ratio to raw-last | ratio to best raw |
| --- | ---: | ---: | ---: |
| temporal_jepa_last | 0.020525 | 1.470337 | 1.470337 |
| raw_surface_last_plus_temporal_jepa_last | 0.011695 | 0.837799 | 0.837799 |
| random_temporal_last | 0.020782 | 1.488773 | 1.488773 |
| raw_surface_last_plus_random_temporal_last | 0.011413 | 0.817571 | 0.817571 |
| scale_barlow_last | 0.019438 | 1.392458 | 1.392458 |
| raw_surface_last_plus_scale_barlow_last | 0.012416 | 0.889423 | 0.889423 |

## Future Probe Summary

| target | temporal+raw/raw | random+raw/raw | scale+raw/raw | temporal learned/best raw | scale learned/best raw |
| --- | ---: | ---: | ---: | ---: | ---: |
| future_mean_delta | 1.062470 | 1.051578 | 1.077521 | 2.171417 | 1.925007 |
| future_range | 0.964248 | 0.870070 | 0.893755 | 1.117205 | 1.022918 |
| future_terminal_delta | 1.231518 | 1.136915 | 1.133662 | 1.786249 | 1.459249 |
| future_max_abs_step | 1.110273 | 0.987311 | 0.978259 | 1.244513 | 1.087707 |
| future_drawdown | 1.035173 | 0.908197 | 0.962685 | 1.190138 | 1.092590 |

## Counts

- Temporal raw+learned future improvements: `1/5`.
- Temporal learned standalone best-raw wins: `0/5`.
- Random raw+feature future improvements: `3/5`.
- Scale raw+learned future improvements: `3/5`.
- Temporal raw+learned beats scale raw+learned: `1/5`.
- Current-IV temporal raw+learned status: `PASS`.

## Random-Control Check

The current-IV raw+temporal improvement is not sufficient evidence
of useful learned temporal state because raw+random temporal features
perform at least as well on that guardrail in this smoke bakeoff.

- Raw+temporal/current-IV ratio:
  `0.837799`.
- Raw+random/current-IV ratio:
  `0.817571`.
- Warning:
  `raw_plus_random_beats_or_matches_raw_plus_temporal_on_current_iv`.

## Decision

Promotion decision: `DO_NOT_PROMOTE`.

Temporal route status: `do_not_promote`.

Future utility status: `temporal_raw_plus_underperforms_controls`.

This is an evaluation bakeoff, not Part B authorization. A temporal JEPA
follow-up is only justified if it improves the same frozen guardrails
without relying on small knob tuning.
