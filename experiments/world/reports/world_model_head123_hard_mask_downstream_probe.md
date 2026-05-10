# World Model HEAD123: Hard Mask Downstream Probe

## Objective Family

`downstream_probe` audit for frozen Part 1 representations.

## Regression Probes

| feature | target | MSE | R2 |
| --- | --- | ---: | ---: |
| barlow_clean_last | future_mean_delta | 0.014174 | -0.020379 |
| barlow_clean_last | future_range | 0.056081 | -3.075501 |
| barlow_clean_last | future_terminal_delta | 0.033722 | -0.037324 |
| barlow_clean_last | future_max_abs_step | 0.042961 | -2.859247 |
| barlow_clean_last | future_drawdown | 0.049510 | -2.646420 |
| raw_surface_last | future_mean_delta | 0.006484 | 0.533258 |
| raw_surface_last | future_range | 0.054625 | -2.969679 |
| raw_surface_last | future_terminal_delta | 0.019208 | 0.409125 |
| raw_surface_last | future_max_abs_step | 0.041928 | -2.766468 |
| raw_surface_last | future_drawdown | 0.050058 | -2.686818 |
| raw_surface_last_plus_barlow_clean_last | future_mean_delta | 0.006338 | 0.543752 |
| raw_surface_last_plus_barlow_clean_last | future_range | 0.054415 | -2.954451 |
| raw_surface_last_plus_barlow_clean_last | future_terminal_delta | 0.020345 | 0.374155 |
| raw_surface_last_plus_barlow_clean_last | future_max_abs_step | 0.041784 | -2.753496 |
| raw_surface_last_plus_barlow_clean_last | future_drawdown | 0.048010 | -2.535973 |

## Regime Classification Probe

| feature | target | accuracy | majority | lift | macro recall |
| --- | --- | ---: | ---: | ---: | ---: |
| barlow_clean_last | regime_label | 0.394531 | 0.597656 | -0.203125 | 0.577082 |
| raw_surface_last | regime_label | 0.554688 | 0.597656 | -0.042969 | 0.352704 |
| raw_surface_last_plus_barlow_clean_last | regime_label | 0.398438 | 0.597656 | -0.199219 | 0.319006 |

## Decision

This is downstream evaluation only. It should not become a Part 1
pretraining objective.
