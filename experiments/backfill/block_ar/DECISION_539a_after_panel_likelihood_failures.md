# 539a Paradigm Decision After Panel Likelihood Failures

## Evidence
The 536a-538a panel-law branch tested the cleanest likelihood-based continuation of the larger financial-panel idea:

- 535a one-shot panel Gaussian path law: `3/11`, level KS `1/25`, cross-cell correlation failed.
- 537a AR panel daily Gaussian transition: `4/11`, cross-cell correlation passed, path max-jump passed, but daily KS `4/25`, level KS `1/25`, conditionality `3.83%`.
- 538a AR panel daily mixture transition: `3/11`, daily KS improved to `20/25` and kurtosis passed, but level KS stayed `1/25`, conditionality fell to `2.0%`, cointegration worst-cell failed, and active mean reversion stayed weak.

This reproduces the older 522a lesson: a model can match daily-change distributions while free rollouts occupy the wrong future IV-level region. The current suite is primarily asking for a conditional law over future IV-level paths, not just a realistic one-step innovation law.

The deployable learned frontier remains:

- `392a`: empirical-score AR transition flow with rollout energy, `8/11`;
- `510a`: patch-energy final checkpoint, `8/11`;
- both fail coverage, regime coverage, and distributional fidelity, with level KS around `10/25`.

The only constructive post-frontier signal that moved the hard bottleneck without collapse was the factor side-channel:

- 526a/529a improved level KS to `12/25` and strengthened cointegration under matched protocol;
- 530a showed longer factor-only adaptation regressed, so factor conditioning alone is not enough.

## Decision
Close one-step panel likelihood variants as the active path. They are elegant but below-frontier, and their failure mechanism is now clear.

Do not keep adding:

- more mixture components;
- Student-t heads;
- frequency/event modules;
- graph/channel modules;
- hand calibration layers.

The next principled executable experiment should return to the strongest deployable core and combine only two previously justified generic ingredients:

```text
392a empirical-score AR transition flow
+ aligned financial factor history conditioning
+ horizon-aware patch/path proper scoring
```

This is not a reset to brute-force local tuning. It is a targeted synthesis:

- the 392a/510a core is the only learned family that preserves structural passes;
- factor conditioning is the only observed learned signal that nudged level occupancy upward;
- patch/rollout energy is the only observed objective family that keeps free-rollout path geometry in scope.

## Next Falsifier
Implement a single factor-conditioned patch-energy fine-tune from the 392a/525 mechanics line.

Acceptance gate:

- must recover at least the 392a/510a structural passes;
- must score at least `8/11` before any further work;
- meaningful success requires improving level KS or regime layer2 without losing conditionality and cointegration;
- if it falls below frontier, close this synthesis and stop returning to local 392a variants.

