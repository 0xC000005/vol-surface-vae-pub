# World Model HEAD085: Downstream Probe Coverage

## Objective Family

`downstream_probe` audit for frozen Part 1 representations.

## Regression Probes

| feature | target | MSE | R2 |
| --- | --- | ---: | ---: |
| barlow_clean_last | future_mean_delta | 0.011635 | 0.162420 |
| barlow_clean_last | future_range | 0.047185 | -2.428997 |
| barlow_clean_last | future_terminal_delta | 0.031830 | 0.020872 |
| barlow_clean_last | future_max_abs_step | 0.039526 | -2.550683 |
| barlow_clean_last | future_drawdown | 0.042269 | -2.113154 |
| raw_surface_last | future_mean_delta | 0.006484 | 0.533258 |
| raw_surface_last | future_range | 0.054625 | -2.969679 |
| raw_surface_last | future_terminal_delta | 0.019208 | 0.409125 |
| raw_surface_last | future_max_abs_step | 0.041928 | -2.766468 |
| raw_surface_last | future_drawdown | 0.050058 | -2.686818 |
| raw_surface_last_plus_barlow_clean_last | future_mean_delta | 0.006584 | 0.526010 |
| raw_surface_last_plus_barlow_clean_last | future_range | 0.047705 | -2.466780 |
| raw_surface_last_plus_barlow_clean_last | future_terminal_delta | 0.021879 | 0.326972 |
| raw_surface_last_plus_barlow_clean_last | future_max_abs_step | 0.040023 | -2.595357 |
| raw_surface_last_plus_barlow_clean_last | future_drawdown | 0.042762 | -2.149423 |

## Regime Classification Probe

| feature | target | accuracy | majority | lift | macro recall |
| --- | --- | ---: | ---: | ---: | ---: |
| barlow_clean_last | regime_label | 0.109375 | 0.597656 | -0.488281 | 0.386260 |
| raw_surface_last | regime_label | 0.554688 | 0.597656 | -0.042969 | 0.352704 |
| raw_surface_last_plus_barlow_clean_last | regime_label | 0.332031 | 0.597656 | -0.265625 | 0.331762 |

## Decision

This is downstream evaluation only. It should not become a Part 1
pretraining objective.
