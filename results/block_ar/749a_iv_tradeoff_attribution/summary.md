# 749a IV Trade-Off Attribution

## Scorecards

| run | pass | cov90 | calerr | cond MAE% | risk-state | level KS | bias | coint | regime L2 | TS | MR | path KS |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---|---|---:|
| 734a_val_incumbent | 6/11 | 0.836 | 0.046 | 9.24 | True | 17/25 | 16/25 | 0.723 | 0/8 | True | True | 0.442 |
| 742a_val_interval_score | 4/11 | 0.866 | 0.018 | 9.07 | True | 9/25 | 12/25 | 0.830 | 1/8 | False | False | 0.543 |
| 745a_val_scale_local | 6/11 | 0.799 | 0.072 | 10.58 | True | 12/25 | 14/25 | 0.807 | 2/8 | True | True | 0.324 |
| 748a_val_state_tail_sampler | 6/11 | 0.748 | 0.117 | 8.71 | True | 10/25 | 12/25 | 0.692 | 1/8 | True | True | 0.313 |
| 746a_train_tail_incumbent | 6/11 | 0.861 | 0.014 | -1.63 | False | 20/25 | 25/25 | 0.492 | 2/8 | False | True | 0.529 |
| 748a_train_tail_state_tail_sampler | 5/11 | 0.803 | 0.074 | -1.96 | False | 20/25 | 24/25 | 0.449 | 1/8 | False | False | 0.417 |

## Hard-Cell Candidate Versus Incumbent

- incumbent median low-tertile coverage90: `0.316327`
- candidate median low-tertile coverage90: `0.095238`
- incumbent median hard lower-miss rate: `0.394558`
- candidate median hard lower-miss rate: `0.571429`

| horizon | cell | inc cov | cand cov | delta cov | inc low cov | cand low cov | delta low cov | cand lower miss |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 14 | (2,3) | 0.667 | 0.501 | -0.166 | 0.347 | 0.129 | -0.218 | 0.492 |
| 30 | (0,2) | 0.628 | 0.440 | -0.188 | 0.551 | 0.279 | -0.272 | 0.549 |
| 30 | (2,3) | 0.519 | 0.351 | -0.168 | 0.109 | 0.041 | -0.068 | 0.649 |
| 30 | (3,3) | 0.576 | 0.406 | -0.170 | 0.286 | 0.061 | -0.224 | 0.594 |

## Attribution

### objective_weighting
- classification: `rejected_as_primary_bottleneck`
- 742a interval-score objective improved aggregate cov90 to 0.866494 but fell to 4/11 and damaged level/path realism.
- 748a state-tail sampler lowered validation cov90 from 0.836384 to 0.748073 and train-tail score from 6/11 to 5/11.

### simple_state_geometry
- classification: `insufficient_as_prefix_feature`
- 745a scale-local prefix stayed 6/11 and reduced validation level-KS cells versus the incumbent.
- The validation risk-state allocation diagnostic already passes for 734a/748a, so the model responds to broad regimes; the miss is per-cell and late-horizon.

### validation_shift
- classification: `real_contributor_not_complete_explanation`
- 743a showed large train-tail/validation level shifts for the stable hard cells.
- 746a train-tail incumbent passes coverage/distribution but still fails conditionality, time-series, cointegration, regime coverage, and pathwise jumps at 6/11.

### transition_readout_capacity
- classification: `most_likely_next_target`
- Daily increments are largely realistic, but integrated level distributions and late-horizon hard cells fail; this points to multi-step transition/readout allocation, not one-day marginal support.
- 748a hard-cell low-tertile median coverage is 0.095238 with lower-miss median 0.571429, worse than the incumbent hard-cell audit.
- The model can allocate broad risk-state width, but does not carry state-local directional tail geometry through the 30-day path.

## Decision

Do not continue tuning sampler weights or scalar losses. The next experiment should change one architectural axis: a minimal shared transition/readout capacity increase that preserves the normalized-innovation law, one stochastic source, and the same tri-scope framework recipe.
