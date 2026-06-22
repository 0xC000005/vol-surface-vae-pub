# NL Paper Reframe Proposal — 2026-06-18

**Status: PROPOSAL / DRAFT ONLY. NOT APPLIED. NOT FINALIZED.**
Paper claims may not change without **owner + independent-Codex sign-off** (the
NL-thread promotion bar). This document assesses what the NL paper *must* change
to reflect this session's verified findings and drafts the reframe; it does
**not** edit `paper/narrative_grounded_scenarios/main.tex`.

- Target paper: `paper/narrative_grounded_scenarios/main.tex` (Paper 2, the
  NL-conditioned generator; `\method` = "Natural Language-Conditioned
  Heterogeneous Multivariate Scenario Generation"). 1893 lines, 27 `\cite`s.
- Target venue (per `project_paper_submission_strategy` memo): **NeurIPS
  GenAI-in-Finance workshop** (non-archival, WIP explicitly welcomed, ~4pp). This
  reframe follows **Option A (right-size honestly)**, *not* Option B (the
  conditionable-generator retrain research program). At a WIP workshop, a
  rigorously-established mechanistic negative is a **feature**, not a weakness.
- Companion Paper 1 (SNI backbone) = `paper/main.tex` → ICAIF. Out of scope here.

---

## 0. One distinction that organizes the entire assessment

Every claim in the paper sorts into exactly one of two buckets. The reframe is
about classifying each claim correctly and keeping the abstract/intro consistent
with the body+appendix (which are already mostly honest).

| Bucket | Status | Evidence |
|---|---|---|
| **Separation / distinguishability** — different narratives select different supports → distinguishable scenario *distributions* | **TRUE, supported** | fixed-start KS 0.261 (factor), 0.375 (portfolio), path energy 7.213; start-only null flat by construction |
| **Directional faithfulness** — the narrative's *stated direction* shows up in the generator's *forward* scenario | **FALSIFIED, 5× independent confirmations** | T7 history-match 0.849 → forward-match 0.428 (≈chance); B1 real−shuffled +0.0015 (fidelity hurt); B2 real−shuffled −0.0026 (both arms) |
| **Width / uncertainty transmission** — the generator modulates *spread* with the numerical condition | **TRUE (generator property)**; narrative→width is the *ungated* "crack" | turb/calm 1.229; width-vs-vov ρ 0.33–0.49 (p<1e-12); risk-state width-spearman 0.92–0.94 |

The paper's actual claims are **mostly bucket-1 (separation)**. The problem is
**not** that the paper claims directional steering outright — it (mostly)
doesn't. The problem is twofold: (a) several bucket-1 phrasings are framed to
**invite a directional reading**, and (b) the paper **under-sells its actual
usefulness** — a narrative-driven, historically-grounded what-if scenario tool
with auditable provenance and honesty labels. The reframe therefore **leads with
usefulness** (the headline product), **propagates the appendix's existing honesty
up to the abstract and contributions**, and **adds the transmission dichotomy
(width transmits, direction washes out) as a supporting characterization** that
explains *why* the scenarios stay historically grounded. The ceiling is honest
nuance, **not** the thesis. That framing is both more accurate and far less
invasive than "the paper is wrong," and it does not over-correct into an accuracy
overclaim (start-only still wins absolute CRPS 0.506; the value is grounded
what-if exploration, not a better point forecast).

---

## 1. Claims that CONFLICT with the verified findings (quoted)

### 1a. Abstract — invites a directional reading (SOFTEN, do not delete)
> "Fixed-start case studies show that different professional narratives select
> different support regimes **and change factor and portfolio-risk readouts**
> even when the initial market level is held fixed." (L93–96)

**Classification:** bucket-1 (separation) phrasing that reads as bucket-2
(directional). "Change … readouts" is true as *distributional separation*; it is
false if read as "the narrative moves the factor in its stated direction."
**Action:** reword to "produce **distinguishable** scenario distributions and
risk readouts" + one clause stating the separation is in distribution/dispersion,
not stated direction (the ceiling).

### 1b. Contribution 3 — the memo's flagged target (REFRAME)
> "We demonstrate fixed-start narrative conditionality through support
> provenance, factor fans, and portfolio-risk readouts." (L150–151)

**Classification:** "narrative conditionality" is ambiguous between separation
(true) and directional control (false). The `project_paper_submission_strategy`
memo explicitly flags this: *"MUST reframe contribution 3 to 'separation +
auditability' … NEVER directional steering. Shipping steering = retraction-grade."*
**Action:** reframe to **distinguishability + auditable provenance** explicitly,
and **add a 4th contribution** = the directional-transmission ceiling as a
mechanistic negative result (see §2).

### 1c. Contribution 2 — NOW conflicts with Track C (REFRAME — newly surfaced)
> "We show that this interface improves held-out distributional backtests versus
> persistence, replay-style alternatives, and **direct text-to-memory
> projection**." (L147–149)

**Classification:** this implies the *trained* support-grounded bridge is the
win. **Track C (clean leakage-free matched-eval, 2026-06-17/18) settled the
opposite:** training-free raw-OpenAI embedding retrieval **beats both trained
retrievers** (CRPS 0.5879 < 0.594 text-space < 0.630 projected-memory), and
**start-only still wins absolute CRPS (0.506)**. So:
- the contribution is **not** "our trained retriever";
- the method is **not** a fidelity win over start-only.

**Action:** reframe Contribution 2 to "the *support-grounding principle*
(language → historically-supported recent regimes → frozen generator) is what
helps, and a **training-free** embedding retriever realizes it as well as or
better than trained bridges." Disclose start-only's absolute-fidelity lead.

### 1d. "Displayed factors should match the story" — closest to crossing the line (SOFTEN)
> "This is also why the displayed factors should match the story. … commodity-
> inflation, dollar-liquidity, safe-haven, and rates narratives should be
> inspected through the channels they invoke." (L907–910)

**Classification:** borderline bucket-2. Telling the reader to read the story's
named channel implies the channel moves with the story — which the Safe-haven
gold mechanism (paper's own appendix) and the ceiling refute. **Action:** keep
the channel-relevance point but state explicitly that the channel shows a
*baseline-relative distributional tilt of bounded/uncertain sign*, not a
story-following move.

### 1e. Conclusion — same separation/direction conflation (SOFTEN)
> "Fixed-start case studies demonstrate that different professional narratives
> **change support regimes, path families, and portfolio-impact readouts** …"
> (L1060–1063)

**Action:** same fix as the abstract — "**distinguishable** … distributions,"
plus a one-sentence statement of the ceiling.

### 1f. Table 4 footnote — invariance assertion needs re-verification (FLAG)
> "The backtest numbers reported in `tab:main_backtest` … were produced from the
> pre-correction 906b/939a corpus and **are unaffected by the regeneration**."
> (L683–688)

**Classification:** this *asserts invariance* to the joint39 factor-map
correction. The memo's Option-A directive is *"re-run 29/66-window backtests on
CORRECTED corpus + report deltas (don't assert invariance)."* The trusted
yardstick is now the **clean leakage-free matched-eval (`*_clean_20260617/18`)**,
not the pre-correction corpus. **Action:** do not take the footnote at face
value; re-run on the corrected/clean corpus and **report the deltas**. If the
numbers move, update Table 4; if they hold, replace the assertion with the
measured delta.

### 1g. NOT a conflict (confirm, keep): the paper's existing honesty
The body and appendix are **already correct** and should be cited as the anchor
for the reframe (propagate these *up*):
- "It is treated as a condition, **not as an instruction to force** a particular
  future outcome." (L271)
- "the figure … **does not claim that the narrative prescribes the future**." (L974–975)
- "This is a model-behavior attribution test, **not a causal claim**." (L888–889)
- Safe-haven gold mechanism: 564 windows, next-30d Gold only +5.62 mean / 53.9%
  up, prefix↔future corr −0.37 — "**grounding validates the conditioning prefix,
  not the terminal sign**." (L722–729)
- Appendix: "**this does not condition the generator**" (L1509); demo tables are
  "**baseline-relative … tilts, not terminal sign targets**" (L1526).

There is **no** "first NL financial scenario generation" novelty claim in the
file (good — it would be refuted by Soleimani 2512.07867 / RealGen / Dalmasso).
Keep it that way; add the prior-art citations (§4).

---

## 2. Reframed contribution statement + abstract sketch (DRAFT)

> **EMPHASIS NOTE (2026-06-18 owner revision).** The headline is the **useful
> product**: a narrative-driven, historically-grounded *what-if* scenario tool.
> The directional-transmission ceiling is **honest nuance / characterization**,
> demoted from headline to a supporting result + Limitations. The factual content
> below is unchanged from the prior draft; only the **ordering and emphasis**
> changed. Guardrail held: we do **not** claim the narrative conditioning beats the
> unconditional (start-only) baseline on point-forecast accuracy — start-only still
> wins absolute CRPS (0.506). The honest value proposition is **what-if
> exploration + historical grounding + auditable provenance + honesty labels**,
> not a more-accurate-than-baseline forecast.

### Honest one-line contribution (usefulness-first)
> A **narrative-driven, historically-grounded conditional scenario generator**
> that lets a risk manager describe a market narrative and *see its impact*: a
> differentiated, grounded factor-scenario *distribution* over the 14-anchor panel
> (which factors move, by how much, with what dispersion), anchored in retrieved
> historical analogues, with **honest in/out-of-historical-precedent labeling**.
> It is a useful *what-if exploration + provenance* tool — different narratives
> retrieve different analogues and produce different, grounded scenarios
> (retrieval-conditioned generation beats persistence by **+10.7% CRPS / +14.3%
> Energy** on a clean leakage-free eval). As an honest characterization of *how*
> that impact flows, we establish that the narrative acts **through grounded
> retrieval** and does not additionally **override** the frozen generator's
> forward direction beyond the retrieved analogues — a clean, 5×-confirmed,
> data-signal-localized mechanistic result that explains *why* the scenarios stay
> historically grounded rather than free-form. This is narrative-**grounded**
> generation with honest support labeling — **not** narrative-**steered**
> generation, and not a more-accurate-than-baseline forecaster.

### Proposed contributions (DRAFT — 4 items, usefulness-first ordering)
1. **A useful narrative-driven, grounded what-if scenario generator (HEADLINE).**
   A risk manager describes a current-market narrative + accepted start; the
   system returns a differentiated, *historically-grounded* factor-scenario
   distribution over the 14-anchor panel — which factors move, magnitude, and
   dispersion — anchored in retrieved historical analogues. The narrative **has
   impact**: different narratives retrieve different analogues → different,
   grounded scenarios. Retrieval-conditioned generation beats persistence by
   **+10.7% CRPS / +14.3% Energy** (clean leakage-free eval). *(NEW headline —
   was implicit; now the lead)*
2. **Auditable provenance + honest support labeling.** Every scenario exposes its
   top-k historical analogues with weights, and an **honesty layer** (convex-hull
   support gate + coherence sign-gate + ESS) labels each scenario as inside or
   outside historical precedent. This makes the tool a *trustworthy* what-if
   surface — the user sees both the impact and whether it has historical support.
   *(reframed from old C3/honesty — now a product-trust contribution)*
3. **The grounding *principle* is what helps — and a training-free retriever
   realizes it.** On a clean leakage-free matched-eval, a **training-free**
   embedding retriever matches or beats trained bridges, and the support-grounded
   interface improves distributional backtests vs persistence/replay/
   direct-projection. We disclose, plainly, that the start-only baseline retains
   the absolute point-forecast-fidelity lead: the value is grounded what-if
   exploration, not beating persistence on accuracy. *(reframed — §1c)*
4. **Honest characterization: narrative impact flows through retrieval, not
   override (SUPPORTING result, not the thesis).** Across five independent probes
   (incl. limited backbone adaptation), the narrative does **not** additionally
   override the frozen generator's forward *direction* beyond the retrieved
   analogues: the generator transmits **width/uncertainty** but **washes out
   stated direction**, a limit localized to the **data signal** (not merely the
   architecture). We present this as (a) **why** the scenarios are historically
   grounded rather than free-form, and (b) a clean mechanistic result — a
   supporting finding placed in characterization/Limitations, not the headline.
   *(demoted from headline to supporting nuance — §2 emphasis note)*

### Abstract sketch (DRAFT — replaces L82–98; usefulness-first)
> Risk managers describe market conditions in language, but scenario engines
> require numerical states, and free-form language models cannot be trusted to
> invent calibrated future prices. We present a **narrative-driven,
> historically-grounded what-if scenario generator** that closes this gap: a risk
> manager supplies a professional current-market narrative and an accepted
> starting level, and the system returns a differentiated, grounded factor-scenario
> *distribution* over a 14-anchor market panel — which factors move, by how much,
> and with what dispersion — anchored in **retrieved historical analogues** and
> rolled through a **frozen** state-normalized–innovation generator. The narrative
> has real impact: different narratives retrieve different analogues and produce
> different, historically-grounded scenarios, and retrieval-conditioned generation
> improves distributional scenario quality over persistence (**+10.7% CRPS,
> +14.3% Energy** on a clean leakage-free backtest). Every run exposes its top-k
> analogues with weights and an **honesty layer** — convex-hull support labeling
> and a coherence sign-gate — that flags each scenario as inside or outside
> historical precedent, making the tool an auditable what-if surface rather than a
> black box. We are explicit about the value proposition and its limits: the
> system is for grounded what-if exploration and provenance, **not** a
> more-accurate-than-baseline forecaster (a persistence/start-only baseline retains
> the absolute point-forecast lead), and the language model interprets and audits
> the condition rather than inventing prices. As an honest characterization of
> *how* the narrative's impact flows, we establish across five probes that it acts
> **through grounded retrieval** and does not additionally **override** the frozen
> generator's forward direction beyond the retrieved analogues — the generator
> transmits width/uncertainty but washes out stated direction, a limit we localize
> to the data signal. The result is a practical, auditable language interface that
> grounds risk-manager narratives in numerical scenario distributions, with its
> mechanism and limits made explicit.

---

## 3. Per-section change list (DRAFT — every item gated on owner + Codex sign-off)

**Ordering principle (owner revision):** lead every section with usefulness (the
grounded what-if product + provenance + honesty labels); place the
directional-transmission ceiling as supporting characterization + Limitations,
never as the headline.

| Section / line | Current | Proposed action | Type |
|---|---|---|---|
| Abstract (L82–98) | "change factor and portfolio-risk readouts" | Replace with the usefulness-first DRAFT in §2 (lead with the grounded what-if tool + provenance + honesty labels + the +10.7%/+14.3%-vs-persistence headline; ceiling as one supporting clause; explicit no-accuracy-overclaim) | REWRITE |
| Contributions (L142–152) | 3 items, C2 = "trained interface beats projection", C3 = "narrative conditionality" | Replace with the 4-item usefulness-first DRAFT in §2 (C1 = headline grounded what-if product; C2 = provenance + honesty labels; C3 = grounding principle + training-free retriever + start-only disclosure; **C4 = ceiling DEMOTED to supporting characterization**) | REFRAME/ADD |
| Intro (L101–152) | division-of-labor framing | Keep (already strong). Lead the contribution paragraph with the useful product; add one sentence: scope is *grounded* what-if generation, not *steered*, and not a more-accurate-than-baseline forecaster | ADD |
| Related Work (L154–251) | no Soleimani/RealGen/Dalmasso | Add prior-art (§4) so the paper positions against the closest NL-scenario-gen work; **no "first" claim** | ADD |
| Results / Eval protocol (L622–693) | four claims incl. "fixed-start narratives change the scenario deck" | Lead with the usefulness/beats-persistence + grounded-distinguishability results; reframe claim wording to *distinguishable distributions*; add the ceiling as a later **supporting** characterization block, not the lead | REFRAME/ADD |
| Backtest table footnote (L683–688) | asserts numbers "unaffected by regeneration" | Re-run on clean/corrected corpus; **report deltas**, do not assert invariance; cite the clean matched-eval as the yardstick | RE-VERIFY |
| Backtest (L731–767) | "support-grounded mixture outperforms … supporting historical market memory" | Keep the support-memory point; **add Track C finding** (training-free ≥ trained; start-only wins absolute CRPS). Report absolute CRPS/Energy alongside %-improvements | ADD/SOFTEN |
| Caption A/B (L769–812) | "professional captions improve CRPS/energy" | Keep — it is a *support-selection* gain, already correctly framed; cross-check numbers against clean corpus | RE-VERIFY |
| "Does the narrative change the scenario?" (L814–913) | "changes the final scenario law"; "displayed factors should match the story" (L907) | Keep separation metrics (KS/path-energy — these are TRUE). Reframe L907 to *baseline-relative tilt of uncertain sign*. Add explicit "separation ≠ direction" sentence | SOFTEN/ADD |
| Casebook (L914–982) | already hedged ("not a sign-following forecast") | Keep; ensure each factor read is labeled baseline-relative distributional tilt | KEEP/MINOR |
| Portfolio casebook (L984–1017) | "different stories can change the downside-tail profile" | Keep (true as separation); ensure no implied directional P&L claim | KEEP/MINOR |
| **NEW §: Directional-transmission ceiling** | absent | **ADD a SUPPORTING subsection** (not the headline) presenting T7 + risk_context oracle + B1 + B2(×2 arms) as a 5×-confirmed mechanistic result; framed as *why the scenarios stay historically grounded* (impact flows through retrieval, not override); the width/direction dichotomy; localization to the data signal (B2 gave the narrative-consuming readout real capacity, fit improved CRPS −0.016, yet added ZERO direction sensitivity, real−shuffled −0.003) | ADD (supporting) |
| **NEW §: Honesty / support-validation layer** | grounding reliability exists (L695–730) but no hull/coherence gates | **ADD** the framework-v1 convex-hull support gate (§I), the coherence sign-gate (§B), and ESS as the product's in/out-of-support labeling. Note Codex verdict = SOUND-WITH-FIXES (σ-normalized gate math; Mahalanobis density; LP-failure labeled distinct) | ADD |
| Discussion / Future work (L1019–1049) | "response-aware support prior" + "more ambitious text-to-scenario models" | Add: **narrative→width modulation (B-width) is the natural positive extension** (the verified "crack"), explicitly **ungated** — do not present as established. Breaking the directional ceiling = conditionable-generator retrain (Option B research program) | ADD |
| **NEW §: Limitations** | **absent** | **ADD** an explicit Limitations section consolidating: directional ceiling; start-only wins absolute fidelity; trained retriever not promoted; single-asset-class corpus; no human study; no external LLM-forecaster baseline | ADD |
| Conclusion (L1051–1069) | "change support regimes, path families, and portfolio-impact readouts" | Reword to "distinguishable"; add one ceiling sentence | SOFTEN/ADD |

---

## 4. New evidence / figures the reframe needs — with readiness status

The owner catches narratives and prefers Codex verification, so each evidence
item is tagged **PAPER-READY** (durable artifact exists) vs
**NEEDS-PERSISTENCE** (must be made durable before it can be cited).

### PAPER-READY (durable artifacts exist on disk)
- **B1 directional gate (JSON).** `nl_track_b_directional_gate.py` on
  `generator_conditioning_probe_b1/`: 115 held-out narrative windows, 50 samples,
  CRN-by-query, real-vs-shuffled-vs-zero. real 0.5954 / shuffled 0.5938 / zero
  0.6015 → real−shuffled +0.0015; fidelity hurt. → ceiling figure/table.
- **B2 directional gate (JSON, both arms).** `generator_conditioning_probe_b2/`
  (velocity_readout) + `..._encoder_control/`: arm-1 loss 3.078→1.620, CRPS
  −0.0157, real−shuffled −0.0026; arm-2 control behaves as wiring predicts. This
  is the **decisive new fact**: downstream capacity that improved fit added ZERO
  direction sensitivity → ceiling localized to the data signal. → ceiling table.
- **Track C clean matched-eval (JSON).** `stride5_14x14_matched_eval_summary_all_methods_clean_20260618/`:
  training-free 0.5879 < text-space 0.594 < projected-memory 0.630; start-only
  0.506 absolute. → Contribution-2 reframe + a baseline-comparison table.
- **Hull gate + coherence sign-gate (code + tests).** `nl_coherence_sign_gate.py`
  (+15 tests), framework-v1 hull gate (σ-normalized, Mahalanobis diagnostic,
  LP-failure labeled). Codex verdict SOUND-WITH-FIXES. → honesty-layer section +
  a hull-discrimination figure on **real** scenarios (joint vs marginal
  discrimination already strengthened per Codex fix).
- **Width-transmission evidence.** turb/calm 1.229; width-vs-vov ρ 0.33–0.49;
  risk-state width-spearman 0.92–0.94. → the "width transmits" half of the
  dichotomy figure. *(Confirm a JSON artifact backs each number before citing.)*

### NEEDS-PERSISTENCE before it can be a paper claim
- **T7 directional hit-rate (0.849 → 0.428).** RESEARCH_LOG explicitly marks this
  decisive number as **markdown-only, NO JSON artifact**
  (`2026-06-16_t7_reweighter_directional.md` is markdown; the directional-hitrate
  is not JSON-persisted). This supporting-characterization number is currently not
  in a machine-checkable artifact. **Action before sign-off:** persist a
  JSON directional-hitrate artifact from the T7 harness so owner/Codex can verify
  the 0.849/0.428 figure independently. This persistence gap is itself a finding —
  it is exactly what the Codex sign-off will hinge on.
- **Clean-corpus re-run of the 29/66-window backtests** (Table 4). The reported
  numbers are pre-correction (906b/939a). Re-run on the corrected/clean corpus and
  report deltas before keeping or updating the table.
- **Caption A/B numbers on the clean corpus** — re-verify (same reason).

### Figures to add
0. **Headline usefulness figure (lead):** the grounded what-if product —
   different narratives → different retrieved analogues → differentiated, grounded
   factor-scenario fans over the 14-anchor panel, with the +10.7%/+14.3%-vs-
   persistence result and the in/out-of-support honesty label visible.
1. **Width/direction dichotomy** (supporting characterization): one panel showing
   width transmission (turb/calm, width-vs-vov) vs one panel showing flat
   real-vs-shuffled-vs-zero direction match across T7/B1/B2.
2. **Real-vs-shuffled ablation bar chart** (B1 + B2 both arms): real ≈ shuffled ≈
   zero, with the +0.05 pre-registered break margin drawn — visually decisive.
3. **Hull discrimination on real scenarios**: in-support vs out-of-support
   scenarios separated by the σ-normalized hull/Mahalanobis label (joint, not
   marginal).

---

## 5. Net assessment

- **The headline is the useful product** (owner revision): a narrative-driven,
  historically-grounded what-if scenario generator with auditable provenance and
  honest in/out-of-precedent labeling. The narrative has real impact
  (retrieval-conditioned generation beats persistence **+10.7% CRPS / +14.3%
  Energy**, clean leakage-free eval). This is what leads the abstract,
  contributions, and results.
- The paper is **~70% already honest**: its body and appendix correctly state the
  generator is not steered and that grounding validates the *prefix*, not the
  *terminal sign*. The reframe is mostly **propagating that honesty up to the
  abstract and contributions** while **leading with usefulness** — not rewriting
  the science.
- **Three real conflicts** to fix, in priority order:
  1. **Contribution 2** (NEW this session): the *trained retriever* is not the
     win — training-free beats it (Track C). Reframe to the grounding *principle*
     + training-free retriever, and disclose plainly that start-only wins absolute
     point-forecast fidelity. **No accuracy overclaim** (value = grounded what-if
     exploration, not a better forecast).
  2. **Contribution 3 + abstract + conclusion**: separation phrasing that invites
     a directional reading — reframe to distinguishability + provenance.
  3. **Table 4 footnote**: drop the invariance assertion; re-run on clean corpus
     and report deltas.
- **The ceiling is honest nuance, demoted from headline to supporting
  characterization + Limitations.** The directional-transmission ceiling
  (5×-confirmed, localized to the data signal by B2) is a clean, publishable
  mechanistic result; framed as *why the scenarios stay historically grounded*
  (impact flows through retrieval, not override) it strengthens the usefulness
  story rather than undercutting it. The convex-hull / coherence-sign honesty
  layer is a genuine product-trust contribution. At a WIP workshop these are
  features — but they support, not headline.
- **Do NOT over-correct into a width overclaim**: narrative→width (B-width) is the
  verified *crack* but is **ungated** — present it as the natural extension, not as
  established. The clean characterization is the *dichotomy* (width transmits,
  direction washes out), a property of the frozen generator.
- **Sign-off dependency**: the supporting ceiling number (T7 0.849/0.428) is
  markdown-only. Persist it as JSON before owner/Codex review; the headline
  usefulness number (+10.7%/+14.3% vs persistence) and B1/B2/Track C/hull
  artifacts are already durable.

**This proposal modifies no paper file. All changes above require owner +
independent-Codex sign-off per the NL-thread promotion bar.**
