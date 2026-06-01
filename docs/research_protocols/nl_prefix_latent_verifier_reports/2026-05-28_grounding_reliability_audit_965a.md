# Independent Verification: Grounding Reliability Audit 965a

Verdict: PARTIAL / PROCEED WITH CAUTIOUS PAPER CLAIM

## What I Checked

- `experiments/backfill/block_ar/nl_grounding_reliability_audit.py`
- `test_code/test_965a_nl_grounding_reliability_audit.py`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_grounding_reliability_audit_965a_66case/grounding_reliability_audit.json`
- `experiments/backfill/block_ar/nl_scenario_demo_outputs/nl_grounding_reliability_audit_965a_66case/grounding_reliability_audit.md`
- `paper/narrative_grounded_scenarios/generated_tables/table_grounding_reliability.tex`
- `paper/narrative_grounded_scenarios/main.tex`

## Confirmed Correct

- The focused test suite passes and covers the intended API, unsupported evidence flagging, future-language leakage flagging, and prefix-report loading path.
- The 66-case historical plus 6-case live audit artifact reports the numbers used in the paper table: 72 cases, 651 claims, 100.0% claim faithfulness, 100.0% future-language detection, 0.0% leakage, 100.0% historical direction agreement, 94.0% historical direction coverage, and 100.0% support-direction pass rate.
- The paper now includes a table and a subsection that correctly separates grounding reliability from generated scenario-quality metrics.
- The script writes reproducible JSON, Markdown, and LaTeX table artifacts.

## Issues Found

- WARNING: Claim faithfulness is based on evidence snippets returned by the grounding layer and checked against the story text. This verifies that extracted evidence is text-supported, but it is not a human semantic-label benchmark.
- WARNING: Historical direction agreement is easiest for generated historical captions that explicitly encode market motions. It is still useful as a regression/sanity check, but should not be presented as proof that the LLM understands every subtle risk narrative.
- NOTE: The audit currently measures visible grounding and support consistency, not scenario distribution quality. CRPS, energy, coverage, and fixed-start conditionality remain the scenario-quality evidence surface.

## Independent Assessment

The implementation and reported numbers are internally consistent. The evidence is strong enough for a paper-facing reliability table if the claim is framed as an auditable structured grounding sanity check. It is not strong enough to claim general human-level semantic grounding without a separately labeled human or expert benchmark.

## Recommended Action

Proceed with the current paper wording, including the explicit caveat that this is a structured reliability sanity check rather than a human-labeled semantic benchmark. Keep the 965a artifacts as the source of truth for the grounding reliability claim.
