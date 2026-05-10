# World Model HEAD128: Scale Downstream Quality

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`downstream_probe` audit for frozen Part 1 candidates.

## Hypothesis

If HEAD127 is a better Part 1 candidate, it should improve frozen
downstream probes and incremental value over raw baselines without using
future targets during pretraining.

## Falsifier

The scaled checkpoint is not Part-B-ready if it still fails broad baseline
superiority or market-state regime probes.

## Baseline Superiority

| target | default Barlow MSE | default best raw MSE | default win | scale Barlow MSE | scale best raw MSE | scale win | scale raw-last+Barlow improves raw-last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| future_mean_delta | 0.011635 | 0.006484 | False | 0.010780 | 0.006484 | False | True |
| future_range | 0.047185 | 0.050972 | True | 0.044396 | 0.050972 | True | True |
| future_terminal_delta | 0.031830 | 0.019208 | False | 0.028396 | 0.019208 | False | False |
| future_max_abs_step | 0.039526 | 0.035980 | False | 0.036666 | 0.035980 | False | True |
| future_drawdown | 0.042269 | 0.044180 | True | 0.039983 | 0.044180 | True | True |

## Summary Counts

- Default Barlow standalone wins: `2/5`.
- Scale Barlow standalone wins: `2/5`.
- Default raw-last+Barlow improvements: `3/5`.
- Scale raw-last+Barlow improvements: `4/5`.

## Regime Probe

| run | Barlow accuracy | raw last accuracy | raw+Barlow accuracy | majority |
| --- | ---: | ---: | ---: | ---: |
| default | 0.109375 | 0.554688 | 0.332031 | 0.597656 |
| scale | 0.515625 | 0.554688 | 0.453125 | 0.597656 |

## Representation Health In Probe Space

| run | effective rank | variance min | offdiag abs mean |
| --- | ---: | ---: | ---: |
| default | 12.795929 | 0.012702 | 0.235656 |
| scale | 18.761132 | 0.010173 | 0.191622 |

## Decision

- Scale improves downstream quality: `True`.
- Scale clears baseline superiority: `False`.
- Scale ready for Part B: `False`.
- Reason: Scaled Barlow improves most downstream diagnostics but still wins only 2/5 standalone IV future targets and remains below majority on regime accuracy.
