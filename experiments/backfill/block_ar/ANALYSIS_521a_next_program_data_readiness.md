# 521a Next-Program Data Readiness Audit

## Context

After removing the hard cap in `520a`, the loop resumed on the new learned-law
program path from `PLAN_518a_next_learned_law_program.md`. The first question is
whether the repo already has enough broader panel data to justify a local
foundation-style pretraining prototype.

## Checks Run

Command:

```bash
python experiments/backfill/baselines/data_loader_38d.py
```

Result:

- aligned IV/factor loader passes;
- SPX return alignment correlation is `1.000000`;
- IV reconstruction max error is `1.11e-16`;
- joint changes shape is `(5821, 38)`;
- test windows from surface index `4540+`: `1222` windows.

Raw data availability:

- `data/vol_surface_with_ret.npz`: `5822` IV rows plus `ret`, `price`, `slopes`,
  `skews`, `levels`, no NaNs.
- `data/multi_factor_data.npz`: `5825` dates, `13` factor levels, `13` factor
  returns; raw factor arrays include NaNs, handled by the existing loader.
- Baseline split used by the 38-d harness: `4039` train changes, `500`
  validation changes, test starts at surface index `4540`.

Existing 38-d baseline results:

| baseline | IV suites | CRPS overall | Energy score | factor KS |
|---|---:|---:|---:|---:|
| CSDI | `3/7` | `0.01119` | `2.8968` | `0/13` |
| TimeGrad | `2/7` | `0.01155` | `2.8865` | `0/13` |
| DeepVAR | `2/7` | `0.01171` | `2.9348` | `0/13` |
| FilteredHS | `2/7` | `0.01141` | `2.9263` | `0/13` |
| HistoricalSim | `2/7` | `0.01157` | `2.9519` | `0/13` |

## Mechanism Read

The data infrastructure is usable, but the local data scale is not sufficient to
make the new program truly foundation-model-like. The current 38-d deep
baselines already show the issue:

- CSDI is strongest but only reaches `3/7` on the reduced IV suite;
- all 38-d baselines fail coverage, regime coverage, and distributional
  fidelity analogs;
- factor KS is `0/13` for all baselines.

This means the next learned-law program cannot honestly be sold as "foundation
model scale" if it only uses the existing local 5822-day panel. A local prototype
can still test architecture mechanics, but it should not be expected to jump
from `8/11` to deployable `11/11` by scale alone.

## Decision

Proceed only with a small local prototype if the goal is to test mechanics. Do
not interpret a failed local prototype as falsifying the broader
foundation-model/data-scale paradigm.

The next most principled executable step is to create a minimal new-core
prototype with the acceptance gate from `518a`: it must reach or approach the
`392a` structural passes before any coverage/regime/distributional optimization.

