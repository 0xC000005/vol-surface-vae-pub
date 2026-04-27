# 640a Coordinate-Policy Audit

## Question

After 638a, should we implement a mixed generated-coordinate model, or is that just an unprincipled IV/factor branch?

## Evidence

The 631a and 638a comparison is unusually clean because the conditioning structure is similar but the generated coordinate differs.

### IV Surface

| metric | 631a encoded-increment | 638a level-score | read |
| --- | ---: | ---: | --- |
| IV score | 4/11 | 4/11 | tied count |
| surface explosion | 51.7% | 0.0% | level-score fixes support |
| time-series suite | fail | pass | level-score fixes tail/kurtosis profile |
| kurtosis ratio | high/fail | 0.940 | level-score fixes global kurtosis |
| q99 tail-scale pass | 19/25 | 22/25 | level-score improves cell tails |
| cross-cell corr ratio | 0.939 | 0.998 | level-score preserves IV geometry |
| pathwise q90/q99 ratios | 2.519 / 6.007 | 0.733 / 0.896 | level-score fixes jump scale |
| coverage | 78.1% | 69.7% | level-score undercovers more |
| conditionality MAE reduction | 8.9% | 4.3% | level-score loses gate |

For IV, level-score generation is clearly the better support/tail coordinate, but it is still too narrow and insufficiently conditional.

### Anchor Factors / Joint Panel

| metric | 631a encoded-increment | 638a level-score | read |
| --- | ---: | ---: | --- |
| factor delta KS mean | 0.115 | 0.156 | increment better |
| factor KS pass | 12/13 | 9/13 | increment better |
| factor q99 median ratio | 1.235 | 2.135 | increment much better |
| factor q99 pass | 13/13 | 6/13 | increment much better |
| factor-factor corr shape | 0.865 | 0.514 | increment better |
| factor-factor generated mean abs corr | 0.133 | 0.052 | increment much better |
| IV-factor corr shape | 0.889 | 0.663 | increment better |
| IV-factor generated mean abs corr | 0.119 | 0.073 | increment better |

For anchor factors, encoded-increment generation is clearly the better coordinate.

## Mechanism

This is not a random metric tradeoff. It matches the economics of the generated variables:

- IV surface cells are bounded/support-sensitive and mean-reverting; direct level-score generation prevents drift and unrealistic surface explosions.
- Anchor price/rate/spread variables are better represented through changes/log-returns/diffs over a 30-day stress horizon; level-score generation makes their changes too wide and attenuates co-movement.

The key distinction is architectural versus preprocessing:

- Bad version: two models, one for IV and one for anchors, later glued together.
- Acceptable version: one model, one memory, one velocity network, one rollout, but a `generated_coordinate` field in each variable spec determines whether that channel's stochastic output is a level-score change or an encoded increment.

## Decision

A mixed generated-coordinate model is justified as a preprocessing/specification rule, not as a model branch, if the implementation keeps one shared transition model and one joint stochastic rollout.

The first defensible map is:

- `iv:*`: generate next empirical level-score changes;
- `factor:*`: generate encoded increments/log-returns/diffs.

This is explicit, but it is supported by the 631a/638a falsifier pair. The paper framing should call it a coordinate choice for heterogeneous financial state variables, not a separate treatment path.

## Next

Implement 641a as a one-model mixed-coordinate transition:

- same state-conditioned memory as 631a/638a;
- same `joint38` panel and `iv_only` support;
- output vector contains level-score deltas for level-score channels and encoded-increment values for increment channels;
- during rollout, decode/update each channel according to its coordinate, then feed back both level scores and increment scores;
- evaluate IV full 11-suite and joint-panel audit.

Acceptance: preserve 638a's IV support/tail improvements while recovering 631a-like anchor-factor q99 and joint dependence.
