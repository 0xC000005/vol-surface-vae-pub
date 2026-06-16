# Verifier Report: Contamination Recovery and Clean 14x14 Rebuild

**Date:** 2026-06-15
**Thread:** NL prefix-latent narrative (iter ~231 state loop / experiment labels reaching 996b)
**Author:** Claude (sonnet-4-6) — THIS IS A CLAUDE-AUTHORED RECOVERY RECORD, NOT AN INDEPENDENT
VERIFIER VERDICT. The binding promotion verdict still requires the independent Codex verifier
after clean-corpus retrain. This report documents factual recovery status and serves as the
pre-retrain evidence record for the post-train Codex T5 check.

---

## 1. The Bug and the Fix

**Root cause:** `nl_episode_narrative_support_cards.py` (MARKETS section) mapped factor
names to hard-coded column indices instead of deriving them from the canonical column
order in `data/multi_factor_data.npz`. The result: AAA_OAS was read from col 36 (NIKKEI)
and USDJPY was read from col 29 (COPPER). Every narrative authored against these indices
described copper price moves as USDJPY and nikkei moves as AAA_OAS.

**Fix committed:** 2c0aa31d (2026-06-11 ~14:11 EDT, owner Codex session)

**Canonical map:** `experiments/backfill/block_ar/nl_joint39_anchor_map.py` derives the
column mapping at runtime directly from `data/multi_factor_data.npz` level_columns+25,
producing a single-source-of-truth factor→column lookup. No hard-coded indices remain.

**Grounding gate:** `test_nl_joint39_anchor_map_grounding.py` — PASSING. This regression
test verifies the derived map against known factor names and catches any future column-order
drift.

---

## 2. Scope of Contamination (Pre-Fix)

The following TEXT artifacts were authored under the contaminated mapping and are superseded:

| Artifact | Description | Status |
|---|---|---|
| 970c/970f/972b support banks | R1 support card corpora | Superseded by clean regen |
| Daily 982g corpus (4010 cards) | Full direct-Codex multi-format corpus | Superseded by clean stride-5 regen |
| 988b stride-5 14+14 bank | 802-target paired positive+negative bank | CONTAMINATED — clean rebuild in progress |
| 990a training manifest | text_hash_digest `c1586f2f...` links to contaminated 988b cards | CONTAMINATED |
| 995a/995c val-frame corpus | Val-frame narrative cards | 995c regenerated clean (89 cards) |
| Demo saved packets | Casebook displayed analogue narratives | Superseded |
| 984a/episode_text/embedding eval numbers | Text-method eval numbers (990g, 991a/992a, 996a/b) | Measurements of contaminated inputs; geometric/structural conclusions stand; exact numbers need restamping |

**CLEAN (unaffected):** Production grounding/query lane (name-based factor lookups); all
numeric machinery and results (734a/739a, 939a, 994a/b harness+oracle, 995b chassis, 995d
labels, 992b ceiling, start-only baselines). The contamination affected only TEXT authoring
routes; the SNI generator, support-bank numeric selection, and quantitative evaluation
pipeline were never touched.

---

## 3. Clean Regenerations (Completed 2026-06-12)

Commit f018dc38 certifies R1 completion.

### R1: Support card corpora (970c / 970f / 972b)

Three support banks regenerated from scratch under the corrected canonical map.
Grounding gate passes on all three. These are the source of truth for support
card taxonomy and the input to any downstream corpus build.

### R3: Val-frame corpus (995c)

89 val-frame cards regenerated clean. Used for held-out retrieval evaluation
on the validation split.

### Clean stride-5 rich narratives (982g)

802-card stride-5 rich multi-format corpus regenerated clean.
Artifact: `experiments/backfill/block_ar/nl_scenario_demo_outputs/episode_card_v3_codex_multiformat_982g_clean_stride5_20260612/`

---

## 4. Clean 14x14 Hard-Negative Bank Rebuild (In Progress)

### Context

The contaminated 988b/990a 14+14 paired corpus is not usable for retriever training.
The 990a training manifest's `text_hash_digest` (`c1586f2f5a4c1787ff4b874b21e3e742f6b011964f13d59ee9413dd52ea002ac`)
and `bank_report_path` point to the contaminated 988b bank.

### Pilot validation (2-window sanity check)

**Artifact:** `experiments/backfill/block_ar/nl_scenario_demo_outputs/stride5_fourteen_view_bank_PILOT_clean_2tgt/`

- Windows: `joint39_train_0000` (window 0000), `joint39_train_0005` (window 0005)
- Status: `pass` (both windows); 2/2 pass, 0 failures, 0 validation errors
- `local_prose_generated: false`
- 14 positive / 14 negative pairs per window

**USDJPY direction check (before/after evidence):**

Under the contaminated mapping, `joint39_train_0000` USDJPY narrative would have described
copper's direction (col 29 = COPPER in the old hard-coded mapping). Under the corrected
canonical map, the pilot's `joint39_train_0000` report shows:

```
USDJPY higher large (+6.95, +1.99579 sigma)
```

Sampled positive pair texts confirm this:
- `sparse_user_query`: "Dollar squeeze with oil surging, USDJPY higher, and SPX under pressure."
- `factor_list_baseline`: "DXY higher large; USDJPY higher large; crude higher large."
- `technical_factor_evidence`: "USDJPY is up near 2 sigma"

The contaminated version would have rendered copper's direction as "USDJPY"; the clean version
correctly renders USDJPY actual +6.95 (higher large). This is a direct sanity-check confirmation
that the canonical-map fix propagates through to narrative authoring.

### Full rebuild (running as of 2026-06-15)

**Artifact (in progress):**
`experiments/backfill/block_ar/nl_scenario_demo_outputs/stride5_fourteen_view_bank_clean_20260615/`

Build log confirms first batches are processing (joint39_train_0000 through 0020 visible in
log as of report time). The full build covers ~802 stride-5 targets.

---

## 5. Owner Decision: "Invest Smart" (2026-06-15)

After the 992b information-ceiling finding (exact-window retrieval objective is
information-limited in 128-dim SNI memory space; oracle adjacent-window rank 208/4010,
recall@10 0.112), the owner decided the re-run should use an IMPROVED retriever objective,
not a straight re-run of the old contaminated protocol.

**Decision (verbatim scope):**
- Rebuild clean 14x14 bank ONCE from clean 972b support cards
- Train an IMPROVED retriever: NV-Retriever false-negative filtering + locality-soft/
  posterior P5 target (this sidesteps the 992b exact-window information ceiling)
- Restamp the old text_space/projected_memory results as an honest baseline on clean data
- Gate via validation framework v1 + independent Codex verifier after retrain

The 992b finding falsifies the **exact-window training objective**, not retrieval-conditioning
in general. The "invest smart" direction is a pivot to a better-aligned objective within
the same retrieval-conditioning paradigm.

**No retrieval method is promoted.** Start-only and the incumbent top3/90 workflow remain
the paper/demo defaults.

---

## 6. Verifier Gap Closure Plan

This report serves as the pre-retrain evidence record. The binding promotion verdict
requires a post-retrain T5 check by the independent Codex verifier. The required gates are:

1. Clean 14x14 bank rebuild COMPLETE (stride5_fourteen_view_bank_clean_20260615/ fully built
   and validated — 0 validation errors, local_prose_generated=false, all targets pass)
2. Improved retriever trained on clean corpus (NV-Retriever + locality-soft/P5 target)
3. Baseline restamped on clean data (old text_space/projected_memory re-evaluated against
   clean 982g and clean 14x14 bank)
4. Downstream backtest gate passes (CRPS/Energy competitive with start-only guardrails;
   conditionality lift detected vs start-only on held-out windows)
5. Independent Codex verifier reviews T5 output and issues a PASS/PARTIAL/FAIL verdict

Until step 5 is complete, no paper/demo method claim may state that retrieval conditioning
is trained on a clean hard-negative corpus.

---

## 7. Summary

| Item | Status |
|---|---|
| Contamination fix (canonical map) | COMPLETE (commit 2c0aa31d, grounding gate PASS) |
| R1 clean (970c/970f/972b) | COMPLETE (commit f018dc38) |
| R3 clean (995c, 89 cards) | COMPLETE |
| Clean stride-5 982g (802 cards) | COMPLETE |
| Pilot 2-window USDJPY sanity check | PASS |
| Full clean 14x14 bank rebuild | IN PROGRESS (2026-06-15) |
| Improved retriever training | PENDING (awaiting rebuild completion) |
| Independent Codex verifier (post-retrain) | PENDING |
| Any retrieval method promoted | NO |
