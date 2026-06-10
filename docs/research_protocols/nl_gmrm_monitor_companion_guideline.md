# NL Scenario Generator — GMRM Monitor Companion Guideline

Date: 2026-06-10
Source style reference: "Global Macro Risk Monitor" (Portfolio Strategy, TFM, May 2026 —
`202605 GMRM.pdf`, content reviewed in-session; PDF not stored in repo).
Status: product guideline (no method/promotion changes; top3/90 default + all claim bans unchanged).

## 1. What the GMRM is, structurally

1. **Summary risk update** — prose: base case + risk skew (e.g., WTI $90 base case, downside
   growth / upside inflation skew).
2. **Risk register** — 9 named risks in 3 themes (Economy / Financial Market / Geopolitics),
   each with Probability (Low/Med/High), Status (New/Unchanged), Fund Impact
   (Favourable/Neutral/Unfavourable).
3. **Per-risk exhibit** — title; macro-impact icons (Growth/Inflation/Rates); three structured
   paragraphs: **Background** (present-tense conditions), **Key Implications** (forward
   hypothetical transmission), **Direction of Travel** (trend of the risk); **KPIs to Monitor**
   (observable indicators); likelihood + fund-impact gauges.
4. **Macro & portfolio implications matrix** — rows = Growth/Inflation/Rates + asset classes
   (Equities, Fixed Income, Inflation-Sensitive, Credit, Real Assets, USD vs CAD); columns =
   the 9 risks; cells = favourable/neutral/unfavourable; bottom row = total-fund impact.

## 2. Mapping to the NL pipeline

| GMRM element | Pipeline equivalent | Status |
|---|---|---|
| Background (present tense) | Condition-only narrative (conditioning contract) | exact match |
| Key Implications (forward) | Forward language -> warnings, never a target | contract handles |
| Direction of Travel | trigger/sequencing/ambiguity sidecar | episode card v3 |
| KPIs to Monitor | evidence/grounding sidecar fields | present |
| Probability | none — real-world event probabilities are out of scope | stays human |
| Fund Impact / implications matrix | baseline-relative terminal tilt per factor (baseline + conditioned rollout from the SAME start) | computed by the demo today, not yet rendered as an exhibit |
| Monthly "as of" cadence | none (starts are ad-hoc window indices) | gap |

Casebook narratives already covering GMRM rows: `commodity_inflation_pressure` ~ Risk 3
(oil/stagflation); `dollar_liquidity_squeeze` ~ Risk 6 (USD confidence); `defensive_risk_off_shock`
~ Risk 5 (AI bubble burst); `rates_selloff_tightening_fear` ~ Risk 4 (Fed independence).
joint39 -> matrix-column proxies: Equities=SPX/Nikkei; Fixed Income=US2Y/US10Y; Credit=AAA/BBB OAS;
Inflation-Sensitive=gold/copper/wheat/crude; USD=DXY/USDCAD; vol overlay=VIX/IV. Weakest: Real Assets.

## 3. The structural tension and its resolution (governance-safe)

GMRM risks are forward hypotheticals; the conditioning contract allows current/recent conditions
only. Resolution — **onset-state narratives** (the casebook pattern): author each risk as a
present-tense description of markets as the risk BEGINS expressing (Codex/GPT-authored; local
template prose remains banned). The defensible quantitative claim per risk:
"conditional on the onset pattern of this risk, the 30-day distributions tilt X vs the
same-start baseline." Probability stays human; impact becomes reproducible, baseline-relative,
and auditable. Never present a terminal forecast; safe-haven-gold caveat discipline applies.

## 4. Tiered plan

### Tier 1 — demo -> exhibit gap (days; zero research risk; promoted boundary only)
1. **Plot the start-only baseline fan** as an overlay (grey band + dashed median) on the
   conditioned fan. Data already computed and stored at
   `generation.start_only_baseline.path_quantiles` (app ~line 2922); plotting never reads it —
   fix `fan_chart_figure` (~2097-2251) + `_path_quantile_row` (~2005-2042).
2. Calendar-date the start ("as of <date>"; window<->history-end-date mapping exists in the
   support table); unify the runbook/app default start (18 vs 22 drift).
3. Terminal quantile table per factor: day-30 P10/P50/P90 baseline AND conditioned (today it is
   qualitative direction labels only).
4. Fixed small-multiples factor grid (SPX, VIX, US10Y, BBB OAS, Gold, ATM IV) on one exportable
   page (PNG/PDF) with a one-line provenance footnote (734a checkpoint, top3/90, grounding
   model, run-record id, analogue windows + weights).
5. Hygiene: wire or remove the never-called `apply_live_support_gated_ensemble_calibration`;
   expose per-analogue fan comparison.

### Tier 2 — the GMRM companion (about a week; plumbing + authoring lane)
1. **Per-risk exhibit template**: GMRM-style header (title, human Probability/Status) +
   onset-state narrative card with KPIs-as-evidence + dual fan (baseline vs conditioned) for the
   risk's mapped factors + terminal tilt table + auto-emitted caveats (wire the casebook
   acceptance audit pass/warn/fail tags) + provenance footnote.
2. **Auto-filled implications matrix**: risks x asset-class proxies; cell = sign/strength of the
   baseline-relative terminal tilt (median shift in units of baseline band width; thresholded to
   favourable/neutral/unfavourable), each cell linked to its fan exhibit. Total-fund row needs
   portfolio weights (portfolio-casebook P&L panel is the precedent).
3. **Monthly cadence**: month-end start convention; data-refresh path for a current joint39
   start (currently unsolved — needs `scripts/download_multi_factor_data.py` refresh + start
   validation); month-over-month tilt-tracking exhibit (same risk, successive starts).
4. **Authoring lane**: one Codex-authored onset-state narrative per GMRM risk (9 for 202605);
   validators in the loop; four can be adapted from the existing casebook.
5. Governance hard-coded into the template: top3/90 + start-only baseline only; no 984a/990
   research outputs; no explicit-hard-negative-training claims; probability column = human.

### Tier 3 — retrieval research (next phase; brainstorm + literature)
The GMRM use case sharpens the target: retrieval must produce **mechanism-matched analogue
pools** (e.g., historical oil-supply-shock onsets) that stay regime-local enough to roll out
well — NOT exact-window identification (independently falsified by 991a/992a/992b: the
exact-window gate exceeds the memory-space information ceiling; oracle memory[w+5] rank 208,
recall@10 0.112). Direction candidates for ideation: locality-soft contrastive targets;
supervision in observable factor-feature space rather than frozen SNI memory space;
text -> support-POSTERIOR distillation (imitate accepted top3/90 selections);
Stage-1 retrieval + 984a-style replay-preference Stage-2. Keep 992a's contrastive-dominant
geometry for any future bridge work.

## 5. Out of scope / explicitly not claimed
- Real-world event probabilities (human judgment).
- Terminal forecasts ("the future follows the prompt" reading is banned; claims are
  baseline-relative distribution tilts).
- Anything beyond the promoted top3/90 workflow until a candidate passes backtest gate +
  independent verifier AGREE.
