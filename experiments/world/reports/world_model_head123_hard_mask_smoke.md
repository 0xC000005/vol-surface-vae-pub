# World Model HEAD123: Hard Mask Smoke Analysis

Date: 2026-05-10

## Iteration Type

`experiment`

## Objective Family

`masked_multiview_invariance` hard-mask training smoke.

## Literature Status

`supported_adjacent_direct_barlow_twins_for_same_state_masked_multiview`.

## Hypothesis

If the Part 1 gap against simple market-state baselines is mainly caused
by under-aggressive masks, then the HEAD122 hard-mask preset should
improve downstream baseline superiority while preserving healthy
same-state representation metrics.

## Falsifier

The hypothesis is falsified if the hard-mask checkpoint learns alignment
but loses downstream probe quality or still fails against raw/simple
market-state features.

## Representation Metrics

| run | hidden A | hidden B | top10 | raw top10 | mrr | raw mrr | median rank | eff rank | variance min | offdiag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HEAD070 default | 6.63% | 7.76% | 0.841927 | 0.373177 | 0.478177 | 0.149323 | 3.000000 | 14.501471 | 0.020293 | 0.224284 |
| HEAD123 hard | 22.65% | 23.60% | 0.604688 | 0.343750 | 0.349711 | 0.130177 | 6.000000 | 11.756882 | 0.016057 | 0.251097 |

## Downstream Baseline Superiority

| target | default Barlow MSE | default best raw MSE | default win | hard Barlow MSE | hard best raw MSE | hard win | hard raw-last+Barlow improves raw-last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| future_mean_delta | 0.011635 | 0.006484 | False | 0.014174 | 0.006484 | False | True |
| future_range | 0.047185 | 0.050972 | True | 0.056081 | 0.050972 | False | True |
| future_terminal_delta | 0.031830 | 0.019208 | False | 0.033722 | 0.019208 | False | False |
| future_max_abs_step | 0.039526 | 0.035980 | False | 0.042961 | 0.035980 | False | True |
| future_drawdown | 0.042269 | 0.044180 | True | 0.049510 | 0.044180 | False | True |

## Regime Probe

| run | Barlow accuracy | raw last accuracy | raw+Barlow accuracy | majority |
| --- | ---: | ---: | ---: | ---: |
| HEAD085 default | 0.109375 | 0.554688 | 0.332031 | 0.597656 |
| HEAD123 hard | 0.394531 | 0.554688 | 0.398438 | 0.597656 |

## Interpretation

Harder masks are useful as a falsifier, but this run says mask
aggression alone is not the missing ingredient. The hard checkpoint
still beats the raw masked-view baseline on retrieval, so it is not
dead, but it loses top10 retrieval, effective rank, and standalone
future-probe quality versus the default checkpoint. Baseline
superiority gets worse: default Barlow beats the best raw surface
baseline on `2/5` IV future targets, while hard-mask Barlow beats it
on `0/5`.

The regime probe improves under the hard mask, but it remains below
both raw last-surface features and the majority baseline. That is not
a promotion signal.

## Decision

- Mask aggression alone fixed the baseline gap: `False`.
- Promote hard-mask checkpoint: `False`.
- Next: audit present-state and factor-panel information in the frozen
  embedding before changing architecture, loss, or adding more mask
  knobs.
