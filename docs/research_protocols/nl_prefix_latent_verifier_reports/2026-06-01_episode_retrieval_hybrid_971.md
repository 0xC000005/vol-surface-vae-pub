# Independent Verification: Episode-Retrieval Hybrid 971

Verification date: 2026-06-01

## Verification Result

Verdict: `PARTIAL`

The narrow claim is supported: the deterministic start-aware hybrid
(`text_weight=0.25`, `start_weight=0.75`) passes the defined
narrative-vs-start-only conditionality-lift gate at 16 samples per support
component, while keeping CRPS/Energy close to the start-only baseline.

The broader claim is not yet supported: this is not enough to promote the
method to the paper/demo default or call the product problem solved. Start-only
still has better CRPS, Energy, and coverage, and the audit is quantitative but
not yet a full risk-manager qualitative review of support matches.

## What I Checked

- Code:
  - `experiments/backfill/block_ar/nl_episode_narrative_bridge_report.py`
  - `experiments/backfill/block_ar/nl_episode_narrative_conditionality_lift.py`
  - `test_code/test_nl_episode_narrative_retrieval.py`
- Artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_hybrid_start_text_confirmation_971q/hybrid_start_text_confirmation_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_hybrid_start_text_lift_971p_t25_s75_s16_vs_start_only/conditionality_lift_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_hybrid_start_text_eval_971o_t25_s75_top3_full66_s16/scenario_level_eval_report.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_narrative_start_only_scenario_eval_971n_top3_full66_s16/scenario_level_eval_report.json`
- Verification command:
  - `uv run pytest test_code/test_nl_episode_narrative_retrieval.py -q`

## Findings

### Confirmed Correct

- The start-only comparator is genuinely narrative-free: it ranks candidates by
  terminal-state distance to the accepted start and records
  `query_text_source = none_start_only_terminal_state`.
- The hybrid selector is transparent: it recalls candidates with local
  narrative-to-narrative retrieval, computes terminal start match, and reranks
  by a normalized weighted sum of text score and start score.
- The conditionality-lift audit compares generated arrays for the same
  evaluated windows and reports support Jaccard, terminal KS, path energy
  distance, and CRPS/Energy guardrail deltas.
- The focused regression suite passes: `12 passed in 0.17s`.
- Artifact numbers match the reported summary:
  - start-only CRPS `0.506351084872`, Energy `0.66034357134`, coverage `0.672390572391`;
  - hybrid CRPS `0.518818034367`, Energy `0.675310894297`, coverage `0.634641284641`;
  - hybrid lift verdict `conditionality_lift_detected`;
  - mean support Jaccard `0.042424242424`;
  - mean terminal factor KS `0.2125501813`;
  - path energy distance `0.075309890689`.

### Issues Found

- `WARNING`: Start-only remains the stronger historical-quality baseline. The
  hybrid is close, but it is not better on CRPS, Energy, or coverage.
- `WARNING`: The current lift threshold is hand-defined. It is reasonable as an
  internal gate, but a paper/demo claim should explain why these thresholds are
  product-relevant or provide qualitative support examples.
- `WARNING`: This verifier did not inspect every selected support narrative for
  risk-manager semantic correctness. The support Jaccard and KS results prove
  distributional movement, not that every retrieved episode is the best
  human-judged analogue.
- `NOTE`: The deterministic hybrid reduces the need for immediate Codex/agentic
  reranking, but it does not rule out Codex reranking as a useful later
  semantic-quality layer.

## Alternative Explanations

- The lift may come from changing support families through the start/text
  tradeoff rather than from a deep semantic understanding of the narrative.
- The factor KS and path energy lift are real but smaller at higher sample count
  than in the low-sample screen, so the visible product effect still needs
  qualitative fan/support review.
- Better start-only calibration may be partly due to the held-out windows being
  close in state space to train supports, which is expected for a market-state
  generator and should not be treated as a failure by itself.

## My Independent Assessment

The current evidence justifies keeping `hybrid_start_text` with
`text_weight=0.25`, `start_weight=0.75` as the next research candidate. It also
justifies the revised framing: conditionality is the main differentiator, while
CRPS/Energy are guardrails.

It does not justify changing the public demo/paper default yet. The next
candidate still needs qualitative support-match review and, if semantic
mismatches appear, Codex/agentic shortlist reranking.

## Recommended Action

Proceed with the hybrid candidate as an isolated research branch. Do not promote
it to the public default. Run a qualitative support-match spot check on the
selected 25/75 supports; use Codex/agentic adjudication only if that check finds
semantic mismatches that deterministic text/start scoring cannot resolve.
