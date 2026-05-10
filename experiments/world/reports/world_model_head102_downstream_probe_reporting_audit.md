# World Model HEAD102: Downstream Probe Reporting Audit

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

Downstream-probe evidence audit.

## Hypothesis

The downstream-probe caveat should account for all feature rows present in the
HEAD085 JSON, not only the selected rows shown in the HEAD085 markdown table.

## Falsifier

The audit fails if omitted JSON rows change the downstream interpretation and
the package caveat does not mention that.

## Evidence

HEAD085 markdown reports:

- `barlow_clean_last`;
- `raw_surface_last`;
- `raw_surface_last_plus_barlow_clean_last`.

The JSON also contains:

- `barlow_clean_mean`;
- `raw_surface_flat`.

Relevant additional `raw_surface_flat` metrics:

| target | raw surface flat MSE/R2 | Barlow last MSE/R2 |
| --- | ---: | ---: |
| future_mean_delta | `0.009746 / 0.298396` | `0.011635 / 0.162420` |
| future_range | `0.050972 / -2.704215` | `0.047185 / -2.428997` |
| future_terminal_delta | `0.027846 / 0.143420` | `0.031830 / 0.020872` |
| future_max_abs_step | `0.035980 / -2.232086` | `0.039526 / -2.550683` |
| future_drawdown | `0.044180 / -2.253920` | `0.042269 / -2.113154` |

## Interpretation

The prior statement that Barlow improves over `raw_surface_last` on
range/max-step/drawdown is still true. The stronger statement "Barlow beats all
raw baselines on risk-width/path-shape probes" is false because
`raw_surface_flat` beats Barlow on max-absolute-step MSE and is stronger on
mean/terminal deltas.

## Package Update

- Updated `reference_manifest.json` downstream caveat with
  `raw_surface_flat_better_than_barlow_on`.
- Updated `package_summary.md` to avoid overclaiming downstream utility.

## Decision

Keep the downstream utility claim narrow: HEAD070 shows partial utility on some
path-width probes versus raw last-surface features, not universal dominance
over raw baselines.
