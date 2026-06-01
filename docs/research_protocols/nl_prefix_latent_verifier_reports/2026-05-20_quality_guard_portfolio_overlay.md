# Independent Verification: Quality-Guard Portfolio Overlay

## Verdict

`AGREE`, with scope limited to **candidate overlay**, not production default.

The independent verifier inspected the implementation and the CRN `925`/`926`
artifacts and found no blocker invalidating the claim that the quality-guarded
portfolio-response support policy improves portfolio response without weakening
broad scenario quality on those paired held-out comparisons.

## Checked

- `experiments/backfill/block_ar/nl_portfolio_response_quality_guard_policy.py`
- `test_code/test_924a_nl_portfolio_response_quality_guard_policy.py`
- `experiments/backfill/block_ar/nl_learned_mixture_policy_testflight.py`
- `experiments/backfill/block_ar/nl_portfolio_response_support_policy_testflight.py`
- `experiments/backfill/block_ar/nl_scenario_level_evaluation.py`
- CRN `925` and `926` equal/candidate scenario reports and comparison reports.
- `RESEARCH_LOG.md` entries around the preceding support-reliability work.

## Confirmed

- Raw CRPS, energy, and portfolio path scores are lower-is-better, while
  `build_mixture_policy_training_table` stores `-value`, so the learned priors
  are higher-is-better utilities.
- The quality guard uses train-derived support priors only. Held-out realized
  futures are used only in evaluation reports, not in bridge construction.
- The paired comparisons use the same `66` held-out windows, sample count,
  checkpoint/eval setup, and per-query common random seeds.
- CRN `925` and `926` both improve CRPS, energy, coverage, and reliable
  portfolio path score versus the equal support floor.

## Warnings

- Two paired CRN seeds were enough for candidate-overlay promotion, but not for
  production-default promotion.
- The first bridge artifact did not record source paths. This was fixed by
  adding `mixture_policy.source_artifacts` to the bridge output.
- Unit tests initially covered score arithmetic only. This was fixed by adding
  an integration-style bridge construction test.

## Post-Verifier Update

A third paired CRN seed, `927`, was run after the verifier response. It still
improved reliable portfolio path score, CRPS, and coverage, but had a small
energy regression. This keeps the method in the candidate-overlay lane rather
than promoting it as the default narrative generator support policy.

## Recommended Status

Proceed with explicit wording:

> quality-guarded portfolio-response overlay; portfolio useful and promising;
> not yet production default because broad energy stability is not clean across
> all tested paired seeds.
