# 615a 614a Gaussian likelihood temperature diagnostic

## Context

614a showed that likelihood training changes the failure mode: aggregate coverage and persistent severe undercoverage improve, but the plain Gaussian transition is too diffuse and too smooth. 615a tests whether this is mainly a sampling-scale issue before adding a richer likelihood family.

## Runs

Checkpoint:

- `models/backfill/614a_joint38_ar_gaussian_nll_e8_w2048_s614/best_model.pt`

Temperature diagnostics:

- temp `0.50`: `results/autoresearch/615a_614a_gaussian_temperature_diagnostic/temp050_full11.json`
- temp `0.75`: `results/autoresearch/615a_614a_gaussian_temperature_diagnostic/temp075_full11.json`

Baseline temp `1.00` is the 614a full-suite result.

## Results

| run | score | surface | cov90 | cond MAE | cat. undercov | daily KS | level KS | bias cells | bad windows | kurtosis | coin worst | MR ratio | path KS | q99 cells |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 614a temp 1.00 | 4/11 | fail | 90.9% | 6.7% | 4.0% | 12/25 | 0/25 | 8/25 | 0.0% | 0.305 | 0.299 | 0.304 | 0.646 | 6/25 |
| 615a temp 0.50 | 3/11 | pass | 77.7% | 4.7% | 8.7% | 13/25 | 3/25 | 9/25 | 2.7% | 0.459 | 0.216 | 0.218 | 0.846 | 12/25 |
| 615a temp 0.75 | 4/11 | pass | 87.2% | 4.7% | 4.8% | 15/25 | 2/25 | 9/25 | 0.0% | 0.352 | 0.254 | 0.248 | 0.801 | 10/25 |

## Mechanism Read

Down-temperature is not enough.

Temperature `0.50` overcorrects:

- aggregate coverage drops to `77.7%`;
- persistent severe undercoverage worsens to `8.7%`;
- conditionality fails;
- cointegration worst-cell fails;
- pathwise KS worsens to `0.846`.

Temperature `0.75` is the better diagnostic point:

- surface validity recovers;
- persistent severe undercoverage remains just under the gate at `4.8%`;
- daily-change KS reaches the `15/25` gate;
- cointegration remains barely passable.

But it still fails the core deployability concerns:

- conditionality is just below gate at `4.7%`;
- level KS remains `2/25`;
- median-bias cells remain `9/25`;
- mean-reversion ratio remains very weak at `0.248`;
- pathwise max-jump KS remains poor at `0.801`;
- per-cell q99 jump scale remains badly uneven at `10/25`.

The issue is therefore not a single global sampling scale. The Gaussian likelihood family has the wrong shape: it produces broad coverage but poor level occupancy, weak conditional pull, low aggregate kurtosis, and uneven per-cell tails.

## Decision

Close simple temperature calibration for 614a.

Keep the likelihood-trained AR transition paradigm, but replace the Gaussian innovation family. The next clean experiment should be a fixed-degree-of-freedom multivariate Student-t transition over the same empirical-score increments. This preserves the exact conditional likelihood objective and one shared state-panel model while giving the density family a heavier-tailed shape without post-hoc stress calibration or separate IV/factor treatment.
