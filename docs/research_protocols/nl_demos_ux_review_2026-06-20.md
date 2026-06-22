# NL Demos — UX/UI Review & Punch-List (2026-06-20)

Method: live Chrome capture of both demos (Demo 1 incl. a live dollar-squeeze run; Demo 2 incl. a
live 2008 narrative-packet generation) + a 7-dimension fan-out review (51 agents, 43 findings →
30 confirmed by adversarial verification against the source). All fixes are UX/copy/framing —
**none touch the frozen 734a generator and none remove honest signals; they make them legible.**

Files:
- Demo 1 (narrative→scenario): `experiments/backfill/block_ar/nl_risk_manager_story_gradio_app.py`
- Demo 2 (scenario→narrative): `experiments/backfill/block_ar/nl_scenario_to_narrative_workbench_app.py`

## Executive summary
- **Demo 1** is feature-complete with a sound grounding/honesty story but **not yet boss-ready**:
  the hero fan chart renders broken at the top (title + wrapped legend + corner annotations
  collide), and the "Narrative View" arrows can contradict the typed narrative with no
  reconciliation at point-of-use. **Top fix:** repair the fan-chart top-band layout; co-critical:
  the Narrative-View framing line.
- **Demo 2** (now visually verified) is functionally complete with **no P0s**, content is strong,
  but it leaks internals (Sidecar/snake_case view keys/window ids/filesystem path), has a
  **Confidence-column bug** (shows `raw_history_path` in the historical path), float-noise numbers,
  and two unguarded dead-ends (malformed CSV, Codex timeout/missing-binary). It DOES have a numeric
  progress timer. **Top fix:** the Confidence-column bug + catch upload/normalize errors into the
  status panel.

---

## Demo 1 — narrative→scenario

### P0
1. **Fan chart renders broken at the top** (VISUALLY CONFIRMED). The 13-trace horizontal legend
   (`y=1.02`) grows into the 60px top margin where the centered title sits; two `y=1.14`
   annotations (provenance + maturity-K) land on top, leaving title + honesty/provenance label
   illegible on the hero chart. *Fix:* move legend below plot (`orientation='h', y=-0.18`, bottom
   margin ~120), top margin ~90, fold the two annotations into a subtitle; collapse the six
   sample-path legend entries. **:2812** (annotations :2820-2843).
2. **"Narrative View" arrows can contradict the typed narrative** (VISUALLY CONFIRMED: risk-off
   narrative → SPX "↑ Up 64%", GOLD up, CRUDE up, VIX down). Direction comes from the
   retrieved-analogue rollout (narrative-independent per the settled direction-ceiling); captions
   reconcile the band/gap but never the View arrows at point-of-use. *Fix (framing ONLY — never
   make text steer direction):* add an element-level caption line: "The direction under each View
   comes from the retrieved historical analogues and the day-0 state; your narrative selects which
   analogues are used but does not set the direction, so a View can differ from what your narrative
   literally describes." Optional: rename the column to an analogue/state-attributed name (3-site
   change :731 / :1479 / :1507). **:4654-4663**.

### P1
3. **Lead with the fan chart — it's buried** below Story Grounding + 8-col Historical Support +
   three stacked strips. *Fix:* render the Scenario Distribution block right after `prefix_status`;
   move grounding/support into a closed "Provenance & grounding" accordion; one thin badge chip-row
   under the chart. **:4612** (grounding :4579-4598, support :4599-4611).
4. **Day-0 "Starting market state" input is opaque, unbounded, fails loudly** — bare
   `gr.Number(value=22)`, help text leaks "historical window index"; out-of-range spends the paid
   OpenAI call then throws raw `IndexError`. *Fix:* relabel; resolve index→date on change; set
   `minimum/maximum` (N from the loaded bank); validate before the grounding call with a plain
   message. **:4542** (guard :4436-4439).
5. **Fan-chart hierarchy & legend declutter.** "Realized future" hindsight line is the
   thickest/blackest (width 3.5 > median 3) — invites "the model forecast this"; Mean vs Start-only
   median are two confusable grey dashes; 6 rainbow paths + standalone P90 = ~12-row wrapping
   legend. *Fix:* demote realized to thin dotted "Actual outcome (hindsight)"; differentiate Mean
   by color+dash; collapse the 6 paths to one legend row; `showlegend=False` on P90 (keep line).
   **:2762-2769 / :2742-2753**.
6. **Replace "top3/90 ensemble" jargon** on the headline subtitle + progress copy (VISUALLY
   CONFIRMED subtitle "Nearest similar regimes: top3/90 ensemble"). *Fix:* rename constant to plain
   language; de-jargon progress strings :3028-3043; explainer :4534. **:143**.
7. **Stop leaking raw internal ids in primary tables/strips** (VISUALLY CONFIRMED:
   `joint39_train_3875`, `joint39_val_0040`). Also Warnings "Code" column
   (`PROXY_MAPPING`, `non_conditioning_forward_language`). *Fix:* map window_id→date+rank; drop
   "(window …)" suffix when date present; primary Warnings table = `[Severity, Message]` only, Code
   to Audit. **:2088** (day-0 strip :1804-1805; warning Code :697/:1136/:1149/:4593).
8. **Gloss the quant labels** "Story Match" 0.855 / "Start Gap" 2.323 / "Start Distance" 14.398
   (VISUALLY CONFIRMED unexplained). *Fix:* "Story/Support Match" → "Narrative match (cosine)";
   "Start Gap/Start Distance" → "Distance from start (σ)"; one-line legend under each table.
   **:757-771**.
9. **Fix the dangling "Details in Audit" pointer + badge jargon** (MY EDIT; CONFIRMED dangling —
   κ-ladder/`leaves_hull_at_kappa`/Mahalanobis are computed locally, never persisted, so Audit has
   no such section). *Fix:* drop the "Details in Audit." clause (the inline sentence is
   self-contained) OR serialize the κ-ladder into the report; plain-language the green badge's
   "pool Mahalanobis"/"κ≤2"; lead the ESS strip with "Effective analogues:" not the "ESS" acronym.
   **:1685** (badge :1625/:1693; Audit :4668).
10. **Per-factor units/decimals in Terminal Day-30 levels** (VISUALLY CONFIRMED: SPX 1852.0594,
    IV 0.1576, yields as decimals, no separators). *Fix:* per-market formatter — index levels
    thousands-sep 0-1dp, FX/VIX/crude 1-2dp, rates/OAS append "%" (levels are %), IV ×100+"%".
    Do NOT reuse the move-formatter. **:1750-1755**.
11. **Guard empty narrative** — `str(story or DEFAULT_STORY)` means an empty box silently produces a
    confident conditioned fan from a hidden canned default. *Fix:* short-circuit blank story at both
    live handlers → empty-output panel; never substitute `DEFAULT_STORY`. **:4436** (:4170-4171/:4188).
12. **Map OpenAI failures to plain guidance + add a timeout.** Live `responses.parse` has no
    timeout/max_retries; failure leaks raw exceptions to the primary panel; a hung call holds the
    full-page spinner. *Fix:* explicit `timeout`/`max_retries`; except on known OpenAI types → plain
    messages, raw detail to Audit. **:4246** + `nl_prefix_latent_temporal_grounding_testflight.py:428-434`.

### P2
13. **Reframe "Required Claims: pass 0/4 mismatches"** (double negative) and stop tagging the
    start-only baseline "warning" (it ignores the narrative by design). *Fix:* "Narrative directions
    matched", show `match/checked` green/amber, drop "mismatches", neutral baseline styling.
    **:2080-2082** (header :772, baseline :2104).
14. **"Path Count" columns are mislabeled** — value is a share ("60% up"), not a count. *Fix:*
    "Path Share" (+ dict keys :1475/:1482); caption → "path shares show what fraction of terminal
    paths point that way." **:729,732** (caption :4657-4658).
15. **Reorder "Read first" to lead with usefulness** — currently opens with "self-consistency …
    not real-world conditional fidelity" before the plain reframe. *Fix (ordering only):* lead with
    "grounded what-if from the closest real historical episodes", trail the caveat. **:4494-4500**.

---

## Demo 2 — scenario→narrative workbench

(Now VISUALLY VERIFIED — earlier source-inferred items confirmed/corrected below.)

### P1
D1. **"Confidence" column shows `raw_history_path`** (MY FINDING — VISUALLY CONFIRMED; NOT in the
   source-inferred pass). In the Historical case every row's Confidence cell is the provenance
   string, not high/medium/low (uploaded case shows medium/high correctly). *Fix:* map/relabel at
   the render layer, or fix the upstream historical sidecar that sets `confidence="raw_history_path"`.
   **:77** (render) + upstream sidecar builder.
D2. **Float noise + missing units in the factor table** (VISUALLY CONFIRMED: SPX start
   1207.0899658203125, delta -238.3399658203125, VIX 33.849998474121094; uploaded case
   -3.0999999999999943). Mixed scales (SPX ~1200 / IV ~0.18 / OAS ~0.01) with no units; each plot
   panel prints its factor name twice. *Fix:* format Start/End/Delta into per-factor unit strings
   before the DataFrame (points / % / bp); one label+unit per plot panel. **:68-81** (plot :189/:181).
D3. **No deliverable statement + "Sidecar" jargon in primary output** (VISUALLY CONFIRMED:
   "Sidecar: ScenarioSidecarV1", "Type: historical_joint39"). *Fix:* add a one-line subtitle stating
   the deliverable in plain RM terms; de-jargon `status_cards_markdown` ("Scenario: loaded"), raw
   JSON to Technical details. **:678** (status card :132/:135).
D4. **"Hard negative" unexplained** (VISUALLY CONFIRMED: bare "Positive"/"Hard negative" headings,
   no framing) — an RM could read it as a valid alternative/more-bearish reading. *Fix (copy, keep
   the term):* one-line explainer in the dynamic `packet_review_markdown` return: "…a
   deliberately-contradictory description used to confirm your scenario can be told apart — NOT a
   valid reading and NOT a more-bearish variant." **:107** (empty-state :105).
D5. **Catch upload/normalize validation errors into the status panel** — parser `ValueError`
   ("missing required column(s): end") raises uncaught, leaving status stuck at "waiting" with only
   an ephemeral toast. *Fix:* wrap normalize handlers to catch `ValueError`/`ValidationError` → render
   the message into the persistent status panel; set `show_error=True`. **:786-796** (chain :247-250).
D6. **Codex generation: add an upfront latency notice + handle timeout/missing-binary**
   (CORRECTED from source-inferred: a numeric progress timer DOES exist — "processing | X/135.9s",
   ~135s observed — so latency *feedback* is present; the gap is (a) no "this can take several
   minutes" expectation-set and (b) `subprocess.TimeoutExpired`/`FileNotFoundError` propagate
   uncaught). *Fix:* prepend "Generating … can take several minutes."; catch TimeoutExpired +
   FileNotFoundError → in-panel failure text. **:805-810** (handler :578-616; root
   `nl_14_view_variant_pilot.py:967-975`).

### P2
D7. **Snake_case view keys shown as headings** (MY FINDING — VISUALLY CONFIRMED: "1.
   sparse_user_query", "2. weekly_risk_monitor", "full_professional", "sparse_variant_tape_read").
   *Fix:* map view keys to human-readable titles. (packet rendering)
D8. **Leaks `joint39_*` window ids in negative-window provenance** (VISUALLY CONFIRMED: "Negative
   window: joint39_train_1925") — inconsistent with the app's date convention elsewhere. *Fix:* show
   the calendar period alongside (or instead of) the id; update assertion at
   `test_code/test_nl_scenario_to_narrative_workbench_app.py:101`. **:119** (rows :84-95).
D9. **Packet JSON filesystem path exposed** (MY FINDING — VISUALLY CONFIRMED: full
   `/home/max/.../scenario_narrative_packet.json` in the primary Generation Status). *Fix:* a
   download button or move to Technical details. (Generation Status render)
D10. **Two-step flow + stale data on mode-switch** (MY FINDING — VISUALLY CONFIRMED). Must
   "Visualize" before "Generate"; switching Input mode leaves stale prior data on the right until
   re-visualize. *Fix:* signpost the two steps (or auto-visualize on input change); clear/refresh the
   right panel on mode switch. Consider defaulting to the friendlier Historical-Case mode.
D11. **Tall stacked chart at many factors** (MY FINDING — VISUALLY CONFIRMED: 11 stacked panels for
   joint39 → long scroll, pushes Generate + output far down). *Fix:* cap panel height / add a compact
   multi-column small-multiples layout, or collapse to a normalized single-panel option.

### Strength to preserve (do not undersell)
Demo 2's packet content is genuinely good — the 14 positives describe the 2008 dollar-funding stress
accurately across distinct registers (desk note, committee memo, mechanism-first, factor evidence),
and the hard negatives are real contrastive near-misses (reflation/relief). Keep this.

---

## RESOLUTION (2026-06-20) — ALL ITEMS APPLIED, BOTH DEMOS BOSS-READY

Two parallel implementation tracks (one agent per file, no same-file swarm) applied every item;
verified by an adversarial workflow (51+9 agents) AND an independent Codex (read-only) pass.

- **Demo 1 (`nl_risk_manager_story_gradio_app.py`):** all 15 items done (P0 fan-chart layout +
  narrative-view framing caption; lead-with-chart reorder; day-0 input guard+date hint; legend
  declutter; subtitle/explainer de-jargon; id→date mapping incl. Selected-starting-level;
  quant-label glosses; dangling "Details in Audit" removed; terminal per-factor units;
  empty-narrative guard; OpenAI timeout + friendly errors; "matched/checked" relabel; "Path Share";
  Read-first reorder). Tests: **84 passed**. Live Chrome QA: fan chart clean (top+bottom),
  badges compact, formatting + framing verified.
- **Demo 2 (`nl_scenario_to_narrative_workbench_app.py`):** all 11 items done (Confidence→"Observed";
  per-factor number formatting pts/%/bp; deliverable subtitle + de-jargoned status; hard-negative
  explainer; upload/normalize error handling; Codex latency notice + timeout/missing-binary +
  **FileNotFoundError** guard on the technical-path; human view titles; window-id→calendar dates;
  packet-JSON path → download button (filename only); two-step signposting + default Historical
  Case + mode-switch panel clearing incl. the two Code blocks; compact 2-col small-multiples).
  Tests: **78 passed**. Live Chrome QA verified end-to-end incl. a real 2008 narrative packet.

**Verdicts:** Claude adversarial workflow = boss-ready, zero blocking (Demo 1 YES-with-nits,
Demo 2 YES). Codex independent = both YES-with-nits, constraint PASS. Both confirm: **frozen 734a /
sampling / direction logic untouched in the demo diffs; honest signals preserved (made legible);
the narrative-View contradiction handled by a point-of-use caption only.**

**Optional remaining polish (non-blocking, documented; NOT applied):**
- Demo 1: add a one-line legend under "Selected starting level"; green/amber color cue on the
  "matched/checked" cell; dedicated unit tests for the new Track-D surfaces; align the "Start
  Distance" label in the diagnostic/accordion tables with the primary "Distance from start (σ)".
- (No remaining Demo 2 polish of note.)

**COMMIT HYGIENE (important):** `diffusion/block_ar/generic_state_aware_normalized_innovation_flow_matching.py`
is modified in the working tree from pre-existing, gated/inert Track-B work — NOT part of these demo
fixes and it does not change runtime sampling for either demo. Do NOT bundle it into a demo-fix commit.
