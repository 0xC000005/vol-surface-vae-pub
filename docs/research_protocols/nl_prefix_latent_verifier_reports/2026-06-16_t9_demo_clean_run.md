# T9.1 — Boss Demo Clean End-to-End Run Record

**Date:** 2026-06-16 · **Status:** PASS (first end-to-end run record since the 2026-06-15 hold)

## Config (current, post-T9 fixes)
- Entry: `run_live_openai_prefix_for_app(samples=12, fan_market="SPX", analogue_scope="ALL",
  story=<safe-haven/risk-off narrative>, explicit_start_window_index=18)`
- Live grounding: gpt-5.4-mini · embedding: **text-embedding-3-small (1536-d)** (post-T9 alignment)
- Query bridge: legacy oracle adapter (1536→128, clean) · support bank: clean 939a top3/90
- Generator: frozen 734a · baseline: start-only included

## Result
- Generator completed (2 yields, final UI tuple len 17), **no error**.
- `generation.path_quantiles`: 15 market rows (conditioned fan).
- `generation.start_only_baseline.path_quantiles`: 15 rows (baseline present).
- Fan figure: 12 traces incl. conditioned band/median/mean/P90 + 6 sample paths AND
  **"Start-only P10-P90 (baseline)" + "Start-only median (baseline)"** → baseline IS overlaid.
- No embedding-dim mismatch (text-embedding-3-small ↔ 1536-d adapter coherent).

## Conclusion
The clean backend (legacy-oracle query bridge + clean 939a top3/90 + frozen 734a) is fully
operational end-to-end with the post-T9 fixes. Confirms the demo serves uncontaminated,
narrative-conditioned scenarios with the baseline overlay. No contaminated artifact in the path.
