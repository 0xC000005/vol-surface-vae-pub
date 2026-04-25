# 424a: Persistent Source-Noise Cap Analysis

## Context

`423a` tested the cleanest path-level stochastic-state idea that avoided a posterior/prior latent scaffold: persistent source noise inside the existing 392a AR transition flow.

It scored `6/11`, below the 392a `8/11` frontier.

## Comparison

| run | score | cov90 | conditionality | level KS | median-bias cells | coint worst | regime L2 | corr ratio | MR active | path KS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 392a | 8/11 | 0.868 | 5.14% | 10/25 | 20/25 | 0.278 | 0/8 | 0.963 | 83.3% | 0.373 |
| 423a rho=0.35 | 6/11 | 0.967 | 6.40% | 0/25 | 11/25 | 0.228 | 1/8 | 0.923 | 79.2% | 0.748 |

## Mechanism Read

Persistent source noise is not useless: it preserved conditionality, time-series realism, cross-cell correlation, and mean reversion together. That is a meaningful distinction from the failed joint transition-path model, which collapsed cross-cell geometry.

But the direction is wrong for the frontier bottleneck:

- coverage moved from mixed under/overcoverage to severe overcoverage;
- level KS collapsed from `10/25` to `0/25`;
- median bias became strongly upward;
- worst-cell cointegration fell below the gate;
- pathwise max-jump KS failed badly.

A weaker rho would mostly interpolate back toward 392a. It could reduce the overcoverage damage, but there is no evidence it would improve level KS or regime layer2 beyond the frontier. The only missing-suite movement was regime layer2 `0/8 -> 1/8`, while the decisive distributional suite moved sharply backward.

## Decision

Close fixed persistent source-noise FM as a primary route. Do not run a rho sweep.

The active frontier remains 392a. The remaining bottleneck is not a missing scalar stochastic persistence knob. The next step must be a broader synthesis of what the 8/11 frontier actually cannot learn from the available conditional signal, or a controlled oracle/system decomposition that separates learned conditional law from risk-policy calibration.
