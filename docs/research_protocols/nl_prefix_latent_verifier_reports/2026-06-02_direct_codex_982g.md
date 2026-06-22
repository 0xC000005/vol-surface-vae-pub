# Independent Verifier: Direct-Codex 982g Episode Retrieval Branch

Verification Result

Verdict: `PARTIAL`

What I Checked

- `experiments/backfill/block_ar/nl_episode_card_v3_codex_testflight.py`
- `experiments/backfill/block_ar/nl_episode_narrative_retrieval.py`
- `experiments/backfill/block_ar/nl_episode_narrative_embedding_bridge_report.py`
- `experiments/backfill/block_ar/nl_episode_narrative_conditionality_lift.py`
- `experiments/backfill/block_ar/nl_scenario_level_evaluation.py`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_multiformat_982g_sharded/final/final_codex_corpus_report.json`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_982g_embedding_bridge_hybrid_66q/hybrid_embedding_start_bridge_report.json`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_982g_scenario_embedding_hybrid_66q_s16/scenario_level_eval_report.json`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_full_codex_982g_lift_embedding_hybrid_vs_start/conditionality_lift_report.json`
- `RESEARCH_LOG.md` tail entry appended on 2026-06-02.

Findings

Confirmed Correct

- The final 982g corpus report claims a complete corpus: `4010 / 4010` cards,
  no missing windows, no validation errors, no invalid cards, and retrieval guard
  passed.
- The final corpus metadata reports `authoring =
  direct_codex_multiformat_gpt_5_5_xhigh` and `local_template_prose_used =
  false`.
- Sample final cards expose `narrative_authoring =
  direct_codex_multiformat` and `valid_for_training_retrieval = true`; generated
  text is stored under `views` and `codex_multiformat_fields`.
- Retrieval-side validation rejects locally deterministic/template EpisodeCardV3
  records before retrieval.
- The OpenAI embedding report used `text-embedding-3-large`, backend `openai`,
  `31618` embeddings, and embedding dimension `3072`.
- The embedding hybrid scenario report is a 66-window frozen-SNI scenario eval
  and reports CRPS `0.525272`, Energy `0.682662`, and 80% coverage `0.653212`
  for `narrative_generator_topk`.
- The embedding hybrid conditionality audit reports
  `conditionality_lift_detected`, support Jaccard distance `0.936364`,
  terminal factor KS `0.189483`, and path energy distance `0.089874`.
- Focused tests passed: `31 passed`.

Issues Found

- `WARNING`: The evidence supports conditionality lift, but not a default-method
  promotion. Start-only remains better on historical CRPS/Energy than all tested
  narrative-aware variants.
- `WARNING`: The direct-Codex authoring claim is supported by saved process
  metadata, shard reports, retrieval guards, and card fields; I did not
  independently replay all Codex calls.
- `NOTE`: Pure episode-text retrieval gives stronger conditionality than the
  hybrids, but its historical CRPS/Energy regression is materially larger.
- `NOTE`: OpenAI large-embedding hybrid is slightly better than local hybrid on
  CRPS/Energy in this run, but slightly weaker on conditionality lift.

Alternative Explanations

- The observed scenario-quality gap may reflect the frozen SNI generator's
  strong dependence on accepted starting level rather than a failure of the
  new narrative corpus.
- The narrative-aware methods may need a learned support posterior or fusion
  rule, not only better text matching, to preserve conditionality while matching
  start-only historical fidelity.

My Independent Assessment

The claim "982g generated a complete direct-Codex corpus and shows measurable
fixed-start conditionality beyond start-only" is supported. The stronger claim
"982g should replace the current paper/demo default" is not supported.

Recommended Action

Keep the current paper/demo default unchanged. Use the 982g corpus as the valid
new evidence base for the next research step: improve the narrative-aware
support posterior or fusion mechanism so the conditionality lift survives while
CRPS/Energy approach the start-only/incumbent guardrail.
