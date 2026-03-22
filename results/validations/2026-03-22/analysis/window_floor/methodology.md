# Window Floor Methodology Audit -- 144b

## What is a "window"?

Each window is a stride-1 sliding (history=30d, future=30d) pair from the test set.
The test set starts at index 4540 in the full surfaces array (corresponding to the
test split boundary). With max_batches=20 and batch_size=64, N=1223 total windows
are evaluated. These are overlapping 60-day periods, each shifted by 1 trading day.

## What is "coverage" per window?

For each window, 50 ensemble samples are generated. The 5th and 95th percentiles
across samples define a 90% confidence interval at each (timestep, row, col) point.

Coverage = fraction of the 750 points (30 timesteps x 5 rows x 5 cols) where
the ground truth falls inside the 90% CI.

## What counts as a "bad window"?

A window is "bad" if its coverage < 50%. The gate requires that fewer than 5% of
all windows are bad.

## 144b Results

- **Total windows**: 1223
- **Bad windows**: 98 (8.0%)
- **Gate**: < 5% required, so this FAILS
- **Worst window**: index 502, coverage = 16.4%
- **P10 coverage**: 52.2%
- **Coverage distribution**: [16.4%, P25=64.4%, P50=77.5%, P75=85.6%, 98.5%]

## Temporal Clustering

Bad windows are heavily clustered in time:
- 65 of 97 inter-bad-window gaps are exactly 1 (consecutive windows)
- Median gap between consecutive bad windows = 1
- 21 distinct clusters (gap > 5 days between clusters)
- Largest cluster: 25 bad windows at local indices [480, 504] (global future [5050, 5074])

This is expected: stride-1 windows share 29 of 30 future days, so if the model
fails on a particular market episode, all overlapping windows will also fail.

**Major clusters** (>= 5 bad windows):
| Cluster | Local range | N bad | Global future range | Duration |
|---------|-------------|-------|---------------------|----------|
| 2       | 145-151     | 7     | 4715-4721           | 7 days   |
| 6       | 281-293     | 11    | 4851-4863           | 13 days  |
| 8       | 480-504     | 25    | 5050-5074           | 25 days  |
| 10      | 565-570     | 6     | 5135-5140           | 6 days   |
| 11      | 618-627     | 8     | 5188-5197           | 10 days  |
| 18      | 1120-1127   | 6     | 5690-5697           | 8 days   |
| 19      | 1135-1142   | 5     | 5705-5712           | 8 days   |
| 20      | 1170-1180   | 5     | 5740-5750           | 11 days  |
| 21      | 1212-1219   | 7     | 5782-5789           | 8 days   |

## Regime Analysis -- COUNTERINTUITIVE FINDING

Bad windows concentrate in CALM periods, not turbulent ones:

| Regime | N windows | N bad | Bad rate |
|--------|-----------|-------|----------|
| Calm   | 306       | 42    | 13.7%    |
| Mid    | 611       | 47    | 7.7%     |
| Turb   | 306       | 9     | 2.9%     |

This is the opposite of what might be expected. The model's ensemble spread
is well-calibrated for turbulent periods but too narrow for calm periods.

IV level (high vs low) shows no significant effect (8.2% vs 7.8%).

## Root Cause Analysis

### Spread mismatch
- Bad window mean spread: 0.0268
- Good window mean spread: 0.0302
- Ratio: 0.888

Bad windows have LESS ensemble spread (11% narrower), not more. The model
underestimates uncertainty in these periods.

### GT moves faster than expected
- Bad window GT daily change magnitude: 0.0245
- Good window GT daily change magnitude: 0.0158
- Ratio: 1.56x

- Bad window GT 30-day displacement: 0.069
- Good window GT 30-day displacement: 0.037
- Ratio: 1.84x

Bad windows have ground truth that moves 56% faster daily and displaces 84%
more over 30 days. The model's confidence intervals are too narrow for these
larger-than-expected moves.

### Cell-level pattern
The worst coverage in bad windows is concentrated in the middle cells (ATM,
near-ATM strikes, medium tenors). Edge cells (deep OTM/ITM, short/long tenors)
retain better coverage. This is consistent with the CRPS rank-1 attractor:
all cells move together, but ATM cells in reality have more independent dynamics.

## Methodology Assessment

The window floor methodology is sound:
1. It correctly identifies windows where the model's uncertainty bands are too narrow
2. The 90% CI / 50% floor threshold is reasonable (a well-calibrated model should have
   ~90% coverage per window, so <50% indicates a serious failure)
3. The 5% tolerance accounts for inevitable edge cases

The main issue is not methodological but structural: the model's CRPS training
produces rank-1 correlation structure, which means all cells' CIs expand/contract
together. When the market moves in a way that doesn't align with the dominant PC1,
many cells simultaneously lose coverage, creating "bad windows."
