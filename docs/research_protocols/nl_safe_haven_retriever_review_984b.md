# Safe-haven Gold Retriever Review 984b

This review packet compares two non-targeted retrieval methods for the same
Safe-haven Gold casebook narrative:

1. `text_embedding_grounded_preference_984a`: OpenAI text embedding retrieval,
   generic grounding, frozen-SNI historical replay preference reranking, and
   unchanged top3/90 support assembly.
2. `projected_memory_rich_narrative`: rich Codex-authored narrative embedding
   projected into SNI final-hidden memory, generic grounding, and unchanged
   top3/90 support assembly.

The packet is human-review only. It does not introduce a Safe-haven-specific
response target, does not regenerate narratives, and does not change the
paper/demo default.

## Main Review Artifact

- Markdown:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/safe_haven_gold_retriever_comparison_review_984b/safe_haven_gold_retriever_comparison_review.md`
- JSON:
  `experiments/backfill/block_ar/nl_scenario_demo_outputs/safe_haven_gold_retriever_comparison_review_984b/safe_haven_gold_retriever_comparison_review.json`
- Generator:
  `experiments/backfill/block_ar/nl_safe_haven_retriever_comparison_review.py`

## Exact Safe-haven Support Sets

`text_embedding_grounded_preference_984a` selected:

- `joint39_train_1177`: equity-vol risk-off with gold-duration bid and credit
  relief;
- `joint39_train_0546`: broad risk-off with partial safe-haven gold
  confirmation;
- `joint39_train_3517`: gold-led haven bid with weak volatility confirmation.

`projected_memory_rich_narrative` selected:

- `joint39_train_2798`: commodity pressure with residual defensive hedging;
- `joint39_train_2644`: Financial-accident risk-off with partial safe-haven
  confirmation;
- `joint39_train_2853`: rates-led risk-off with partial gold confirmation.

Top-3 overlap is `0`.

## Metric Context

On the 66-window heldout frozen-SNI scenario evaluation:

- `984a`: CRPS improvement `+0.219440`, Energy improvement `+0.251990`,
  coverage80 `0.649767`;
- projected memory: CRPS improvement `+0.202889`, Energy improvement
  `+0.241549`, coverage80 `0.662251`.

On the start-only conditionality lift audit:

- `984a`: terminal factor KS `0.274038`, path energy `0.181869`, CRPS delta
  versus start-only `+0.015291`, Energy delta `+0.026206`;
- projected memory: terminal factor KS `0.281299`, path energy `0.182276`,
  CRPS delta versus start-only `+0.026352`, Energy delta `+0.035789`.

Interpretation: `984a` is stronger on broad historical CRPS/Energy and slightly
closer to start-only on absolute fidelity, while projected memory is slightly
stronger on coverage and marginally stronger on the aggregate lift metrics.
For the exact Safe-haven Gold support narratives, projected memory has more
high-confidence selected supports, but both methods produce direction-passing,
financial-accident support sets with explicit ambiguity.

## Review Question

The open human-review question is not whether Gold must continue upward. The
narrative describes the current/recent prefix. The review question is whether
the selected supports read like plausible Safe-haven Gold conditioning prefixes
without requiring a hand-coded Safe-haven response target.
