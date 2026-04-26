# 556a Regime Under-Inclusion Geometry

## Question

Is the remaining risk-readiness blocker global width, persistent collapse, or localized regime/horizon/cell occupancy?

## Candidate Summary

| Candidate | Suite | Layer2 | Under | Over | Layer3 | Worst |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `510a_patch_energy` | `8/11` | `0/8` | `6` | `8` | `True` | turb h30 [2, 3] = 0.538 |
| `555a_local_factor_core` | `8/11` | `0/8` | `5` | `7` | `True` | turb h30 [2, 3] = 0.385 |

## Stable Undercovered Cells

- `turb` h`30` cell `[2, 3]`: seen in `2` candidates, min worst `0.385`
- `turb` h`7` cell `[4, 3]`: seen in `2` candidates, min worst `0.538`
- `calm` h`1` cell `[1, 0]`: seen in `2` candidates, min worst `0.615`
- `calm` h`30` cell `[3, 4]`: seen in `1` candidates, min worst `0.564`
- `turb` h`14` cell `[4, 3]`: seen in `1` candidates, min worst `0.564`
- `calm` h`14` cell `[0, 3]`: seen in `1` candidates, min worst `0.641`
- `calm` h`30` cell `[0, 3]`: seen in `1` candidates, min worst `0.667`
- `turb` h`1` cell `[1, 3]`: seen in `1` candidates, min worst `0.667`

## Worst Rows

### 510a_patch_energy
- `turb` h`30` worst `0.538` cell `[2, 3]`, best `0.974` cell `[2, 0]` (under, over)
- `calm` h`30` worst `0.564` cell `[3, 4]`, best `0.974` cell `[0, 2]` (under, over)
- `turb` h`7` worst `0.564` cell `[4, 3]`, best `0.974` cell `[0, 3]` (under, over)
- `turb` h`14` worst `0.564` cell `[4, 3]`, best `1.000` cell `[1, 1]` (under, over)
- `calm` h`1` worst `0.615` cell `[1, 0]`, best `0.974` cell `[3, 1]` (under, over)
- `calm` h`14` worst `0.641` cell `[0, 3]`, best `1.000` cell `[2, 2]` (under, over)
- `turb` h`1` worst `0.718` cell `[1, 3]`, best `0.974` cell `[4, 0]` (over)
- `calm` h`7` worst `0.744` cell `[1, 0]`, best `1.000` cell `[2, 1]` (over)

### 555a_local_factor_core
- `turb` h`30` worst `0.385` cell `[2, 3]`, best `1.000` cell `[4, 0]` (under, over)
- `turb` h`7` worst `0.538` cell `[4, 3]`, best `0.974` cell `[0, 3]` (under, over)
- `calm` h`1` worst `0.615` cell `[1, 0]`, best `0.974` cell `[3, 3]` (under, over)
- `calm` h`30` worst `0.667` cell `[0, 3]`, best `1.000` cell `[1, 1]` (under, over)
- `turb` h`1` worst `0.667` cell `[1, 3]`, best `0.923` cell `[4, 0]` (under)
- `turb` h`14` worst `0.744` cell `[4, 3]`, best `1.000` cell `[3, 1]` (over)
- `calm` h`7` worst `0.769` cell `[0, 4]`, best `1.000` cell `[4, 0]` (over)
- `calm` h`14` worst `0.795` cell `[0, 3]`, best `1.000` cell `[0, 2]` (over)

## Decision

The blocker is localized regime/horizon/cell occupancy, not global width or persistent scenario collapse. Broadening every path would be a blunt fix and would likely damage authenticity. The next clean move should target hard conditional stress states through a learned objective or sampling law that allocates mass to sparse regime cells without evaluator-time cell tables.
