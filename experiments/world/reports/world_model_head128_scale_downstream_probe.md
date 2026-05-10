# World Model HEAD128: Scale Downstream Probe

## Objective Family

`downstream_probe` audit for frozen Part 1 representations.

## Regression Probes

| feature | target | MSE | R2 |
| --- | --- | ---: | ---: |
| barlow_clean_last | future_mean_delta | 0.010780 | 0.223943 |
| barlow_clean_last | future_range | 0.044396 | -2.226375 |
| barlow_clean_last | future_terminal_delta | 0.028396 | 0.126502 |
| barlow_clean_last | future_max_abs_step | 0.036666 | -2.293756 |
| barlow_clean_last | future_drawdown | 0.039983 | -1.944756 |
| raw_surface_last | future_mean_delta | 0.006484 | 0.533258 |
| raw_surface_last | future_range | 0.054625 | -2.969679 |
| raw_surface_last | future_terminal_delta | 0.019208 | 0.409125 |
| raw_surface_last | future_max_abs_step | 0.041928 | -2.766468 |
| raw_surface_last | future_drawdown | 0.050058 | -2.686818 |
| raw_surface_last_plus_barlow_clean_last | future_mean_delta | 0.006266 | 0.548906 |
| raw_surface_last_plus_barlow_clean_last | future_range | 0.045222 | -2.286356 |
| raw_surface_last_plus_barlow_clean_last | future_terminal_delta | 0.020934 | 0.356035 |
| raw_surface_last_plus_barlow_clean_last | future_max_abs_step | 0.037278 | -2.348693 |
| raw_surface_last_plus_barlow_clean_last | future_drawdown | 0.040234 | -1.963242 |

## Regime Classification Probe

| feature | target | accuracy | majority | lift | macro recall |
| --- | --- | ---: | ---: | ---: | ---: |
| barlow_clean_last | regime_label | 0.515625 | 0.597656 | -0.082031 | 0.521244 |
| raw_surface_last | regime_label | 0.554688 | 0.597656 | -0.042969 | 0.352704 |
| raw_surface_last_plus_barlow_clean_last | regime_label | 0.453125 | 0.597656 | -0.144531 | 0.346618 |

## Decision

This is downstream evaluation only. It should not become a Part 1
pretraining objective.
