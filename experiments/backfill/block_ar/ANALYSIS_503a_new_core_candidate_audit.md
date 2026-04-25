# 503a New-Core Candidate Audit

## Context

After 502a closed deployable calibration/wrapper research around `392a`, the only way
to continue toward `11/11` without changing product framing would be a genuinely new
learned core.

## Prior New-Core Evidence

The repo already contains several new-core attempts or baselines:

- CSDI and TimeGrad deep time-series baselines were run as P0 baselines and scored only
  `2/7` in the older suite.
- DiT-style denoisers were explored early; they were not a breakthrough and were noted
  as requiring much more data/compute while leaving the fundamental bottleneck
  unchanged.
- Direct full-path empirical-score FM/diffusion (`339`, `345`, `354`, `413`, `417`) and
  exact full-path likelihood (`346`) scored far below the `392a` frontier.
- Conditional future-token density and latent full-path bottleneck branches were also
  falsified below frontier.

## Read

The clean new-core option is not "try CSDI/TimeGrad/DiT/full-path flow again." Those
families have already answered the key question in this dataset: they do not preserve
the local structural geometry as well as the `392a` AR empirical-score transition law.

The remaining failure is not simply model capacity. It is weakly identified future
level/regime allocation under one realized future per history. Larger generic sequence
models may fit training likelihood better, but prior results show they tend to lose
conditionality, mean reversion, or cross-cell/path geometry.

## Decision

No immediate new-core experiment is clean enough to recommend from the current repo
state. A serious new-core program would need to be explicitly scoped as new research,
with new compute budget and acceptance criteria, rather than another in-session local
autoresearch iteration.

Current defensible position remains:

```text
deployable learned frontier: 392a, 8/11
oracle feasibility:         435a, 11/11 but nondeployable
current 11/11 status:       not achieved, not deployably supported
```

Continuing to run local experiments without new data, new compute, or changed product
framing would be brute-force knob search rather than first-principles research.
