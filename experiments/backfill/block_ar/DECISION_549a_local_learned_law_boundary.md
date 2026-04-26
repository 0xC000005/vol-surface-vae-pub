# 549a Local Learned-Law Boundary Decision

## Evidence Boundary

The local learned-law search has now tested several clean paradigms after the 392a/510a
frontier:

- coherent Gaussian score-path density,
- small multi-factor panel law,
- panel daily transition density,
- factor-conditioned patch/energy synthesis,
- causal local-shift normalized AR flow,
- paired raw-IV path energy,
- unpaired raw-IV joint-law alignment.

The best deployable learned models remain `392a` and `510a` at `8/11`. The newer
544a/546a/548a branch did not recover frontier:

| model | score | useful movement | blocking failure |
| --- | ---: | --- | --- |
| 544a | `5/11` | daily KS, cross-cell, support, path jumps | level KS `4/25`, weak MR |
| 546a | `4/11` | level KS `14/25`, MR ratio `0.453` | support collapse, path KS fail |
| 548a | `5/11` | level KS `10/25`, support preserved | conditionality `3.3%`, weak MR |

This is not a missing architecture block. It is a boundary of what the local single-SPX
learned conditional law has been able to identify without either more data or an explicit
risk calibration policy.

## Mechanism

The remaining suites want mutually hard behavior:

- conditional support must be wide enough for coverage and regime layer2,
- but not so wide that overcoverage exceeds the 95% cap,
- raw level occupancy must match the validation historical marginal law,
- mean-reversion strength must be close to empirical slopes,
- path jumps must preserve high-frequency realism,
- conditional medians must improve over unconditional baselines.

Local learned objectives can move one side of this set, but so far they pay for it by
breaking another side.

## Decision

Close further local learned-law architecture/objective mutation as the default route.

If external option-surface data remains unavailable, the next deployable route must be
framed honestly as a **risk system**:

1. report the base learned model separately (`392a`/`510a`, `8/11` frontier),
2. add an auditable training-only calibration/policy layer if the deployment objective is
   risk-manager usability rather than pure learned-law purity,
3. evaluate final system metrics separately from base learned-law metrics.

This is not Bitter-Lesson-pure, but it is methodologically honest: the learned core remains
the conditional generator, while the policy layer expresses a conservative risk prior and
must be reported as such.

## Next Step

If continuing without external data, run one clean calibrated-risk-system iteration using
training-only calibration artifacts, not validation oracle information. The target should be
to repair coverage/regime support while preserving the learned core's conditionality,
cross-cell structure, cointegration, and path-jump realism.

Do not run more local learned-law sweeps.
