# World Model HEAD130: Scale Part 1 Quality Gate

Date: 2026-05-10

## Iteration Type

`post_experiment_analysis`

## Objective Family

`masked_multiview_invariance` scaled-candidate quality gate.

## Verdict

- Quality gate passed: `False`.
- Promotion decision: `DO_NOT_PROMOTE`.
- Part 1 ready for Part B: `False`.

## Layer Results

| layer | status | evidence |
| --- | --- | --- |
| representation_health | PASS | top10=0.849740 vs raw=0.343229; mrr=0.497244 vs raw=0.132390; effective_rank=22.324; variance_min=0.021119; offdiag_abs_mean=0.168158 |
| corruption_robustness | PASS | mask_artifact=no_large_mask_family_leakage; stratified=no_large_stratified_failure; min_stratified_top10=0.804348; max_stratified_offdiag=0.170515 |
| state_content | PARTIAL | state probes improve versus HEAD070, but exact IV retention still loses to raw surface; scale_beats_raw_surface_on_iv=False |
| baseline_superiority | FAIL | standalone Barlow wins 2/5 IV future targets versus best raw surface baselines |
| market_state_regime_probe | FAIL | regime_accuracy=0.515625; raw_last=0.554688; majority=0.597656 |
| scale_and_stability | PARTIAL | scale improved one-seed smoke to 1024 train windows and 256 validation windows, but multi-seed and full-data stability are still not run |

## Interpretation

HEAD127 is the best Part 1 candidate so far and validates scale as useful, but it is not ready for Part B because baseline superiority and market-state regime probes still fail.

## Next Required Evidence

- multi-seed scale stability
- exact-state retention improvement versus raw surface features
- stronger market-state probes that beat majority and raw baselines
- broad baseline superiority beyond 2/5 standalone IV future targets
