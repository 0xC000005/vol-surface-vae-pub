# 726a Normalized-Scale Adequacy

## Anchor Scope

| name | raw q99 train/val | raw val/train | norm q99 train/val | norm val/train | scale q90 train/val | scale val/train |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `factor:spx` | 54.140/43.640 | 0.81 | 4.00/4.11 | 1.03 | 0.0194/0.0106 | 0.55 |
| `factor:us2y` | 0.200/0.110 | 0.55 | 3.66/3.43 | 0.94 | 0.0844/0.0401 | 0.47 |
| `factor:us10y` | 0.190/0.110 | 0.58 | 3.15/2.95 | 0.94 | 0.0928/0.0522 | 0.56 |
| `factor:aaa_oas` | 0.330/0.040 | 0.12 | 11.60/4.30 | 0.37 | 0.0486/0.0143 | 0.29 |
| `factor:bbb_oas` | 0.140/0.050 | 0.36 | 6.26/3.66 | 0.58 | 0.0455/0.0272 | 0.60 |

## Mechanism Read

Current history-RMS normalization removes most raw scale shift for ordinary channels, for example SPX normalized q99 is 4.00 train versus 4.11 validation. It does not fully stationarize sticky OAS: AAA normalized q99 remains 11.60 train versus 4.30 validation, and BBB remains 6.26 versus 3.66. The history scale itself is lower in validation, but the normalized target tail is still regime-shifted, so raw scale alone is not enough.

## Decision

The next repair should target the coordinate/objective for mixed-frequency channels, not the atom gate. A principled candidate is a deterministic frequency-aware tail coordinate or loss balance that makes nonzero updates comparable across regimes while preserving one shared AR flow framework.
