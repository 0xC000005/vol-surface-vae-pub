# 725a Nonzero-Update Scale Attribution

## Split Diagnostics

| name | train zero | val zero | train q99 | val q99 | val/train q99 | train mean | val mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `factor:spx` | 0.000 | 0.002 | 54.140 | 43.640 | 0.81 | 0.2343 | 1.5208 |
| `factor:us2y` | 0.230 | 0.211 | 0.200 | 0.110 | 0.55 | -0.0011 | 0.0021 |
| `factor:us10y` | 0.067 | 0.125 | 0.190 | 0.110 | 0.58 | -0.0011 | 0.0012 |
| `factor:aaa_oas` | 0.435 | 0.474 | 0.330 | 0.040 | 0.12 | -0.0002 | -0.0008 |
| `factor:bbb_oas` | 0.265 | 0.383 | 0.140 | 0.050 | 0.36 | 0.0002 | -0.0034 |

## Generated OAS Attribution

| source | scope | variant | name | KS | zero gen/val | q99 gen/val/train | gen/val q99 | gen/train q99 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `721` | `anchor` | `history_bin_atom` | `factor:aaa_oas` | 0.225 | 0.454/0.474 | 0.039/0.040/0.330 | 0.97 | 0.12 |
| `721` | `anchor` | `history_bin_atom` | `factor:bbb_oas` | 0.275 | 0.305/0.383 | 0.057/0.050/0.140 | 1.14 | 0.41 |
| `721` | `joint` | `history_bin_atom` | `factor:aaa_oas` | 0.210 | 0.454/0.474 | 0.036/0.040/0.330 | 0.90 | 0.11 |
| `721` | `joint` | `history_bin_atom` | `factor:bbb_oas` | 0.272 | 0.304/0.383 | 0.055/0.050/0.140 | 1.10 | 0.39 |
| `724` | `anchor` | `history_bin_atom` | `factor:aaa_oas` | 0.231 | 0.454/0.474 | 0.152/0.040/0.330 | 3.79 | 0.46 |
| `724` | `anchor` | `history_bin_atom` | `factor:bbb_oas` | 0.335 | 0.305/0.383 | 0.130/0.050/0.140 | 2.60 | 0.93 |
| `724` | `joint` | `history_bin_atom` | `factor:aaa_oas` | 0.154 | 0.454/0.474 | 0.062/0.040/0.330 | 1.55 | 0.19 |
| `724` | `joint` | `history_bin_atom` | `factor:bbb_oas` | 0.255 | 0.304/0.383 | 0.090/0.050/0.140 | 1.79 | 0.64 |

## Mechanism Read

Validation OAS is a calmer regime than the 2048-window training tail: AAA nonzero q99 is 0.040 versus train 0.330, and BBB nonzero q99 is 0.050 versus train 0.140. The 724 masked-zero loss removes the damping effect of exact-zero targets, so the continuous anchor-only law learns too much of the broader train-tail scale. 721 stayed closer to validation tails because the continuous loss still mixed zeros and nonzeros, but that is an accidental dampener rather than a clean observation model.

## Decision

Do not add another atom-probability knob. The next principled repair should target nonzero-update scale allocation under regime shift, preferably through a shared, history-conditioned scale or frequency-balanced objective that is deterministic from data statistics and applies to IV-only, anchor-only, and joint without scope-specific recipes.
