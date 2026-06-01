# Independent Verification: Full-Corpus Fixed-Start Caption Audit 957c

## Verification Result

Verdict: AGREE, with bounded scope.

The artifact-backed claim is supported: under a shared fixed start, the
professional caption plus fact-token condition produces larger final pooled
scenario separation than both simple fact-token text and a true repeated
start-only null.

This does not prove broad production readiness across all starts or sample
budgets. It supports a paper/demo claim that the full professional caption
corpus changes final support-conditioned scenario distributions in this audit.

## What I Checked

- Code:
  - `experiments/backfill/block_ar/nl_full_corpus_fixed_start_conditionality_audit.py`
  - `test_code/test_957a_nl_full_corpus_fixed_start_conditionality_audit.py`
- Artifacts:
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/full_corpus_fixed_start_conditionality_audit_957c_start22_s8_shared_startonly/full_corpus_fixed_start_conditionality_audit.json`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/full_corpus_fixed_start_conditionality_audit_957c_start22_s8_shared_startonly/full_corpus_fixed_start_conditionality_audit.md`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/full_corpus_fixed_start_conditionality_audit_957c_start22_s8_shared_startonly/full_corpus_fixed_start_raw_level_fan_panel.png`
  - `experiments/backfill/block_ar/nl_scenario_demo_outputs/full_corpus_fixed_start_conditionality_audit_957c_start22_s8_shared_startonly/full_corpus_fixed_start_terminal_range.png`
- Paper surface:
  - `paper/narrative_grounded_scenarios/main.tex`
  - `paper/narrative_grounded_scenarios/generated_tables/table_full_corpus_fixed_start_caption_audit.tex`
- Research log:
  - `RESEARCH_LOG.md` tail entry dated 2026-05-26.

## Confirmed Correct

- The start-only null was corrected to one shared fixed-start reference
  distribution repeated across cases. This is the right control for the paper
  claim because it prevents query-window or random-seed variation from being
  counted as narrative conditionality.
- The decision artifact reports `status: pass`.
- The decision checks are all true:
  - above-start-only factor separation;
  - above-start-only portfolio separation;
  - above-start-only path-energy separation;
  - non-collapsed professional support.
- The paper table matches the JSON artifact:
  - professional: support Jaccard 0.030, factor KS 0.261, portfolio KS 0.375,
    path energy 7.213, SPX range 23.92, VIX range 1.07;
  - simple facts: support Jaccard 0.580, factor KS 0.182, portfolio KS 0.167,
    path energy 2.787, SPX range 4.50, VIX range 0.16;
  - start-only null: support Jaccard 1.000 and zero movement on all final
    scenario separation metrics.
- The result is correctly framed in the paper as fixed-start conditionality
  evidence, not as free-form LLM forecasting.

## Issues Found

- WARNING: The audit uses six cases and eight samples per condition. The result
  is enough for a paper/demo evidence refresh, but not enough for a broad
  production-readiness claim across all starts.
- WARNING: The current paper still retains older balanced-80 pilot figures for
  historical context. They are now labeled as earlier pilot evidence, while the
  current full-corpus scenario result is carried by the new table and figures.
- NOTE: The LaTeX build has an underfull hbox warning in the new table. This is
  cosmetic and does not affect the PDF build.

## Alternative Explanations Checked

- Starting-level artifact: ruled out for this audit by the repeated shared
  start-only null, which has zero cross-case scenario movement.
- Support-table-only artifact: not sufficient, because the audit measures final
  pooled scenario states, portfolio terminal KS, and full-path energy.
- Simple fact-token sufficiency: weakened by the artifact because simple facts
  produce lower final pooled scenario separation and much higher support
  overlap than professional captions.

## Recommended Action

Proceed with the bounded paper/demo claim:

> The full professional caption corpus changes final fixed-start scenario
> distributions more than simple fact-token text and more than a true
> start-only null.

Do not claim complete production readiness until the same audit is repeated at
larger sample budgets and across additional accepted starts.
