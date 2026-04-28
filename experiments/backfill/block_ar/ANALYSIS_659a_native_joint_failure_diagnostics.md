# 659a Native-Joint Failure Diagnostics

## Executive Read

The native joint models are not mainly failing because IV and anchors cannot share one stochastic source. They are failing because the shared objective sees two very different statistical jobs at once: IV level-score moves are small, mean-reverting, and level-occupancy sensitive, while anchor-factor increment scores are near unit scale and dominate the mixed-coordinate flow target. The models learn local daily realism and broad correlation shape, but underlearn conditional IV level placement and attenuate per-scenario IV-factor shock amplitude.

## IV Suite Comparison

| model | score | cov90 | cond MAE red | daily KS | level KS | median bias | kurt ratio | corr ratio | path KS | failed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 641a_ar_singlehead | 4/11 | 0.647 | 4.51% | 24/25 | 4/25 | 8/25 | 0.946 | 0.949 | 0.610 | coverage, conditionality, cointegration, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism |
| 647a_oneshot_singlehead | 4/11 | 0.663 | -0.68% | 14/25 | 10/25 | 14/25 | 0.337 | 0.718 | 0.618 | coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism |
| 652a_oneshot_multihead_joint | 4/11 | 0.698 | 2.64% | 16/25 | 10/25 | 22/25 | 0.320 | 0.662 | 0.619 | coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism |
| 652a_oneshot_multihead_ivonly | 5/11 | 0.708 | 6.56% | 19/25 | 12/25 | 21/25 | 0.333 | 0.804 | 0.591 | coverage, time_series, regime_coverage, distributional_fidelity, mean_reversion, pathwise_jump_realism |
| 658a_ar_multihead | 5/11 | 0.687 | -0.45% | 24/25 | 7/25 | 12/25 | 0.742 | 0.750 | 0.480 | coverage, conditionality, time_series, regime_coverage, distributional_fidelity, mean_reversion |

## Anchor-Factor Audit

| model | factor KS mean | factor KS pass | q99 pass | factor corr shape | factor corr abs gen/GT | IV-factor shape | IV-factor abs gen/GT | worst factor KS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 641a_ar_singlehead | 0.098 | 12/13 | 13/13 | 0.866 | 0.610 | 0.852 | 0.682 | 0.233 |
| 647a_oneshot_singlehead | 0.087 | 13/13 | 11/13 | 0.829 | 0.733 | 0.845 | 0.378 | 0.138 |
| 652a_oneshot_multihead_joint | 0.085 | 13/13 | 13/13 | 0.842 | 0.786 | 0.846 | 0.373 | 0.129 |
| 652a_oneshot_multihead_ivonly | missing | missing | missing | missing | missing | missing | missing | missing |
| 658a_ar_multihead | 0.109 | 11/13 | 13/13 | 0.892 | 0.610 | 0.868 | 0.592 | 0.246 |

## Target-Scale Interference

- AR mixed-coordinate train target: IV std `0.436`, anchor std `1.005`, dimension-weighted factor share `0.734`.
- One-shot path-coordinate train target: IV std `0.677`, anchor std `1.005`, dimension-weighted factor share `0.535`.
- Interpretation: in the clean joint objective, factor increment channels carry most of the target energy. IV channels are numerous, but their generated coordinate is much smaller. This makes IV conditional level placement easy to underfit while still achieving good global flow loss and strong factor realism.

## Data Framing

- IV train-vs-val future level KS mean `0.534`, delta KS mean `0.113`.
- Anchor train-vs-val future level KS mean `0.650`, delta KS mean `0.090`.
- Validation realized IV turbulent/calm future absolute-move ratio `1.000` with rank-corr proxy `0.007`.
- Interpretation: daily increments are much more stable than levels, so models can learn realistic local movement while failing level occupancy. The validation split does not provide a strong realized future-width signal from history volatility, so regime-width conditionality is weakly identifiable.

## Failure Mechanism

1. IV surface: generated daily changes, path jumps, cross-cell structure, and aggregate mean reversion are mostly alive; the failure is conditional level placement and coverage geometry, especially per-cell/regime coverage and level KS.
2. Anchor list: factor daily increments and factor-factor correlation shape are realistic, but generated factor levels can leave plausible validation ranges for some sparse/stale factors, and absolute co-movement amplitude is systematically attenuated.
3. Joint scenario: IV-factor correlation shape is learned, but the per-scenario shock strength is too small. This is why scenarios can look directionally coherent on average but still feel weak as a joint stress story.
4. Interference: the shared flow objective is clean, but not group-balanced. It rewards learning high-energy anchor increments and local daily movement more than the low-amplitude IV level-score decisions that drive coverage, conditionality, and level occupancy.

## 658a Failure Anatomy

- Worst IV level-KS cells: iv:23=0.437, iv:02=0.425, iv:24=0.410, iv:18=0.403, iv:13=0.364.
- Lowest IV h30 coverage cells: iv:13=0.401, iv:23=0.401, iv:02=0.422, iv:18=0.422, iv:07=0.433.
- Worst IV conditional MAE reductions: iv:10=-15.294, iv:04=-12.356, iv:09=-11.558, iv:02=-11.348, iv:06=-11.095.
- Worst anchor delta-KS factors: factor:aaa_oas=0.246, factor:spx=0.244, factor:crude_oil=0.139, factor:us2y=0.113, factor:nikkei=0.112.
- Largest anchor level-range excursions: factor:aaa_oas=2.735, factor:gold=0.755, factor:wheat=0.748, factor:spx=0.654, factor:bbb_oas=0.622.
- Largest anchor tail-ratio errors: factor:spx=1.585, factor:usdjpy=0.665, factor:aaa_oas=0.750, factor:crude_oil=1.245, factor:us10y=1.186.
- Interpretation: the latest coherent joint AR model is not exploding and is not ignoring anchors. Its realism breaks at conditional placement: certain IV surface cells are persistently undercovered at h30, and some macro/market anchors drift outside the validation level range even when their one-day deltas look statistically acceptable.

## Decision

Do not add another architecture component yet. The next principled move is a group-balanced training falsifier: same architecture, same shared source, same data, but make IV level-score placement and anchor increments comparably visible to the loss. If that improves IV coverage/level occupancy without damaging anchor KS/correlation, the current paradigm remains viable. If not, the bottleneck is not decoder expressiveness but missing conditional signal/data framing.
