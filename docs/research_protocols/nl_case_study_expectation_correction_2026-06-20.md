# Case Study — Grounded Conditional History Corrects an Unconditional Prior

Date: 2026-06-20 · Thread: NL narrative-conditioned scenario generator · Status: diagnostic
illustration (reproducible; NOT a promotion claim)

## One-line claim
The narrative-grounded generator faithfully retrieves the *historical condition a risk manager
describes*, and the scenario it produces shows **what historically followed that condition** — which
can **contradict the manager's unconditional macro prior**. This makes the tool an
**expectation-checking instrument**, not merely a scenario generator. The worked example: a
"dollar-squeeze, oil+gold being sold" narrative, where the intuitive call is *gold down*, but the
grounded conditional history says *gold tends to rebound*.

## The scenario
- **Narrative (verbatim):** "Dollar funding is tight: DXY is surging, USDJPY is breaking lower,
  equities and BBB credit are under pressure, and oil plus gold are being sold."
- **Day-0 start:** 2008-10-24 (joint39 window 2185), a post-Lehman crash bottom (gold ≈ $729, SPX
  ≈ 877, VIX ≈ 79).
- **The naive (unconditional) prior:** a firmer dollar + risk-off ⇒ *gold should fall*.
- **What the tool shows:** gold ≈ **+13%** (conditioned median up; 85% of terminal paths up).
  Surprising at first glance — the engine seems to ignore a "sell gold" narrative.

## Why this is not a bug — three verified facts

### 1. The narrative is the *conditioning setup*, not a forecast
"oil plus gold are being sold" describes the **recent/current state** (the 30-day history leading
into day-0), per the product's own contract ("examples describe current/recent market conditions").
It is *not* a prediction that gold will keep falling. The grounding parsed it correctly:
`CRUDE_OIL: down`, `GOLD: down`.

### 2. The retrieval is correct — it finds the described condition (12/12)
Every retrieved support window has **both crude and gold falling over its own 30-day history** —
i.e. the retrieval matched genuine "oil+gold being sold" setups, spread across two decades:

| support window (history end) | crude over history | gold over history |
|---|---|---|
| 2008-10-17 | 106→72 DOWN | 798→785 DOWN |
| 2000-09-29 | 32→31 DOWN | 274→274 DOWN |
| 2015-07-27 | 60→47 DOWN | 1185→1096 DOWN |
| 2007-08-28 | 75→72 DOWN | 673→664 DOWN |
| 2007-08-30 | 76→73 DOWN | 684→665 DOWN |
| 2012-05-25 | 103→91 DOWN | 1649→1569 DOWN |
| 2014-10-29 | 93→82 DOWN | 1226→1224 DOWN |
| 2008-12-01 | 74→49 DOWN | 788→775 DOWN |
| 2007-12-14 | 96→91 DOWN | 806→793 DOWN |
| 2005-05-12 | 57→49 DOWN | 426→422 DOWN |
| 2008-09-09 | 122→103 DOWN | 916→787 DOWN |
| 2011-12-16 | 94→94 DOWN | 1755→1596 DOWN |

**SETUP (history): crude DOWN 12/12, gold DOWN 12/12.** The retrieval is exonerated — it correctly
identifies the condition the manager described.

### 3. The *forward* after those setups is a rebound — that is the conditional history
For the same 12 windows, the realized **30-day forward** (what actually happened after the sell-off):

**FORWARD (after the setup): gold UP 9/12 (75%), crude UP 6/12 (50%).**

So the historical record itself says: *after oil+gold have just been liquidated, gold rebounds about
three times in four over the next month.* The generator's "gold up" is a faithful reflection of this
conditional history — **not** a failure to honor the narrative.

## The mechanism, stated cleanly
- **Unconditional prior:** "firmer dollar ⇒ gold down" (a long-horizon, all-else-equal association).
- **Conditional reality:** conditioned on *"gold has just been sold off,"* gold tends to **rebound**
  over the next 30 days (short-horizon mean-reversion after a liquidation).
- The tool retrieves the *condition* (12/12 correct) and surfaces the *conditional forward* (rebound),
  which **corrects the unconditional prior**. The on-screen "Similar to baseline" / "How direction is
  set" labels are what flag this honestly.

## A secondary, corroborating fact — the start state governs direction
Same narrative, only the start differs:
- Fixed 2008-10-24 (crash bottom) → gold **+13% up** (mean-reversion bounce dominates).
- Narrative-led start (the narrative's own best-fit window, 2015-07-15, a rates-up/firm-dollar era)
  → gold **−0.8% down** (the narrative's regime expresses).

So *which day-0 state you condition on* is the primary directional lever; the narrative grounds
*which* conditional history is used, within the (settled) direction ceiling.

## Honest scope (what this is and is NOT)
- **It IS:** a reproducible demonstration that the tool faithfully grounds a described condition and
  shows its conditional forward — turning a counter-intuitive output into an expectation-check.
- **It is NOT:** a discovery that gold mean-reverts (a known practitioner fact), nor a statistical
  market claim — this is **n = 12, one narrative, one start-region**: illustrative, not inferential.
- **It does NOT change promotion status:** start-only still wins absolute CRPS; the direction ceiling
  stands. This *strengthens* the deliverable's actual value thesis — useful = grounded what-if +
  honest labels + provenance — it is not a forecast-accuracy claim.

## Reproducibility
- Validation script: `experiments/backfill/block_ar/case_study_setup_vs_forward_validation.py`
  (re-runs the live pipeline + prints the SETUP 12/12 and FORWARD 9/12 tables).
- Ground truth: `prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz`
  (`history_raw` / `future_raw`, anchor cols CRUDE=31, GOLD=37).
- Run artifacts: `nl_scenario_demo_outputs/_probe_setup_check/` (support windows + grounding).

## Literature framing (verified; audit DOIs/pages before submission)
The rebound is **consistent with documented short-horizon behavior — not a discovery**:
- **Baur & Lucey (2010), *Financial Review* 45(2):217–229** — gold's safe-haven bid in stress is
  **transient (~15 trading days)** before reverting: the closest real match to "gold rebounds within
  30 days of a sell-off."
- **Jegadeesh (1990), *J. Finance* 45(3):881–898** — canonical short-horizon (one-month) return
  reversal.
- **Conditional vs. unconditional:** Rebonato (2010), *Coherent Stress Testing* (Wiley); Barone-Adesi,
  Giannopoulos & Vosper (1999), *J. Futures Markets* 19(5):583–602 (filtered historical simulation).
- **Referee caveat (honest):** there is **no clean peer-reviewed "30-day gold reversal" paper**;
  commodity-reversal results are 12–30 *month* horizon (do NOT lean on them). With n=12, frame as
  "directionally consistent with documented short-horizon reversal," not a calibrated probability.

## Follow-ups — status (2026-06-20)
1. **Paper — DONE (unified, owner decision).** The TWO gold examples (old "safe-haven gold" caution
   + the new one) were **merged into ONE** unified case study §4.8 "Setup versus forward: faithful
   retrieval, mean-reverting outcomes" (C-on-A framing). Structure: **Q1 retrieval fidelity** — both
   directions 12/12 (squeeze: crude+gold down; safe-haven: gold up + yields down) via Table 12 +
   a symmetry sentence; **Q2 forward = mean-reversion** — population diagnostic Table 13 (recast of
   the deleted `tab:safe_haven_gold_mechanism`): gold-sold n=856 **61.2% up, +2.0% mean, corr −0.13**;
   dollar-squeeze n=255 **57.6% up, +2.5%, corr −0.44**; gold-up mirror n=2219 **54.4% up, corr −0.29**;
   full-pop corr **−0.17** (n=4010). Forward % are scale-free (gold's 2000–2015 trend makes both means
   positive, so the **negative correlation + up-fraction differential** carry the mean-reversion, not
   the mean). Old safe-haven framing removed and **all cross-refs re-pointed** (§grounding-reliability,
   §4.5, §4.6, §limitations, qualitative-guide row); **no dangling refs**; compiles **0 errors, 44pp**,
   citations (Jegadeesh 1990, Baur & Lucey 2010; Rebonato 2010 + Barone-Adesi 1999 if FHS cited) resolve.
   Population numbers reproducible via `case_study_gold_meanreversion_population.py`. **TODO before
   submission: independent citation audit (DOIs/pages).**
2. **Demo UI — PROPOSED (not yet applied).** Two framing-only touches recommended:
   P0 rewrite `_HOW_IT_WORKS_NOTE` and P1 augment the "How direction is set" line, to state
   "your narrative describes conditions *now*; the fan shows what historically *followed* them —
   which can run opposite to your forecast intuition." Awaiting owner go-ahead (boss-ready demo).
3. **Literature framing — DONE** (see above; audit pending).
