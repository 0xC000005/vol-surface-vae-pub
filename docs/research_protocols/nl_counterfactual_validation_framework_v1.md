# Counterfactual Scenario Validation Framework (Acceptance Spec v1 — PENDING OWNER RATIFICATION)

Date: 2026-06-11
Origin: owner objective correction (CRPS = reconstruction diagnostic + fidelity floor;
"beat start-only" retired as promotion target) -> 6-agent ideation workflow (5 literature
areas: stress-test plausibility (Breuer/Csiszar, Studer, BJRS), SBC/generative eval (Talts,
Modrak, TARP, C2ST), institutional practice (NGFS, SR 11-7/15-18, CCAR/EBA), metamorphic/causal
refutation, matched-episode conditional verification (Lerch, CWZ, Christoffersen, GW)) +
adversarial panel. Replaces CRPS-vs-start-only as the promotion basis for the NL scenario line.

Summary of tiers:
- Tier 1 (always-on, per scenario/issue): honesty block (hull-feasibility support label, ESS,
  episode-calibrated KL, innovation-q, severity on separate axis); metamorphic core (null
  narrative, placebo refuter, direction checks, jointly reported); authenticity floor;
  conditional-completion coherence; ex-post envelope + scenario-set distinguishability loop.
- Tier 2 (promotion gates, verifier-required): matched-episode conditional backtest (ex-ante
  registry, leave-episode-out retrieval, MC exact nulls, published MDE); closed-loop SBC+TARP
  with Modrak data-dependent quantities AND information-gain co-gate; memorization triple gate +
  C2ST with effect sizes; full metamorphic suite; LLM-judge pre-gate if applicable.
- Tier 3 (research/sign-off): L-C2ST per-query trust flag, posterior-SBC, Bayes-consistency
  diagnostic (demoted, never a gate), SWIM severity frontier, expert protocols + regulatory
  equal-plausibility exhibits.

Flags: most load-bearing = matched-episode backtest; most gameable = vanilla SBC (passed
perfectly by a conditioning-dead pipeline — Modrak) and chance-AUC C2ST (passable by
underpowering); most important new build = convex-hull/support honesty gate; LIVE BUG TO AUDIT
FIRST = potential pool/eval-window overlap leakage in 994a/matched-eval scripts.

Full adversarial panel verdict follows verbatim.

---

# Adversarial Panel Verdict: Counterfactual Validation Framework (no realized futures)

Panel: Skeptical Statistician (SS), Systems Engineer (SE), Risk-Manager Domain Expert (RM). Data realities held constant in every attack: ~5,825 days / 39-dim panel, 30-day windows, analogue pool ~90 → top-3, K=48 paths, episodes per mechanism archetype realistically 8–20, overlapping windows everywhere.

---

## Candidate-by-candidate adjudication

### A. Matched-episode conditional backtest — KEEP (heavily revised). **MOST LOAD-BEARING.**

- **SS attack:** As pre-registered, A is a propriety trap. "Episodes where the hypothesized mechanism actually occurred" selects on *realized in-window outcomes*; Lerch et al. prove this makes every proper score improper and systematically rewards over-tilted ("extremist") generators — your candidate would *beat the true law* on this scorecard by exaggerating. Second kill-shot: with ~10–15 episodes per archetype, DM-type tests are near-powerless; "no rejection" will be sold as "calibrated" and that's a wrong story waiting to happen. Third: if the analogue pool can contain windows overlapping the scored episode, you have self-retrieval leakage — the exact CWZ "didn't impose the null" size distortion. **Audit this first in `nl_994a_val_frame_start_only_eval.py` / `nl_14x14_matched_eval_summary.py` before trusting any existing matched-eval number.**
- **SE:** Moderate cost. Episode registry = one frozen JSON; leave-episode-out (LEO) retrieval = pool filter, cheap; MC exact nulls = ~500 start-only rollouts per episode date on the frozen generator (GPU is free per owner). All scoring plumbing exists in the 994a harness.
- **RM:** This is the only validator that touches conditional reality. It is literally the institutional license (Clark–McCracken: validate the machinery where the condition realized, then trust it counterfactually). A risk manager who accepts nothing else will accept this.
- **Concrete spec:**
  - *Inputs:* Episode registry per archetype (oil spike, USD stress, credit blowout): trigger = deterministic function of t0-information only (e.g., 10d crude return > expanding-window pre-t0 q95), first-trigger-per-cluster, ≥30d separation, frozen in a dated `docs/research_protocols/` intake doc *before* scoring, independently re-derived by Codex from the rule (disagreement → adjudicate → re-freeze). Narratives authored as-of t0 (no realized-future text — existing guardrail).
  - *Statistic:* Per episode, LEO-retrieval conditioned ensemble vs start-only baseline: paired ΔCRPS, Δenergy, band coverage; primary = across-episode sign test + episode-clustered paired block bootstrap (`nl_994a_paired_block_bootstrap.py`, block ≥30d); per-episode CWZ conformal flag (S_1 on standardized residuals, cyclic-block permutation p); GW conditional-predictive-ability with *ex-ante* instruments only. For outcome-defined queries ("oil spikes in-window"): do NOT subset — twCRPS with smooth weight on the oil tail over ALL val windows, or condition on realized driver path and score the other 38 factors.
  - *Pass philosophy:* Pre-registered one-sided directional hypotheses; exact p-values from the MC null (Dufour pattern), and the **minimum detectable effect published next to every result**. "No rejection" reported as "insufficient evidence," never "calibrated."
  - *Assets:* 994a harness + `nl_14x14_matched_eval_summary.py` (extend), MC null = new thin wrapper over existing rollout code.

### B. Internal coherence — KEEP (revised into two named checks).

- **SS attack:** "Consistency with the generator's law" is circular — the generator grading its own homework; strike that clause. "Magnitude bounded by analogue-pool dispersion" is gameable from both sides: retrieval chooses the pool, so a tight pool inflates apparent tilts and a loose pool passes anything. Conditional correlations from few stressed windows are noise; you need closed-form targets and bootstrap bands, not eyeballed co-movement.
- **SE:** Nearly free. Historical 30-day-move covariance (train frame, block-aware) gives the BJRS conditional-expectation completion μ_{-S|S} in closed form; compare to existing tilt tables.
- **RM:** Maps verbatim to Fed internal-consistency sign-off (Okun/Phillips analogue) — "every non-targeted factor tracks its conditional relation, deviations are documented scenario-specific assumptions." Strong language for the monitor.
- **Concrete spec:** (1) *Conditional-completion check:* narrative pins factor set S; compute μ_{-S|S} from historical 30-day moves; hard gate = sign agreement on all factors with |conditional beta| above a 994a-bootstrap noise floor; magnitude disagreement = flag-and-investigate, NOT auto-fail (could be learned tail dependence; Glasserman–Kang κ=(ν−1)/ν tie-breaker). Red flag = material tilt on near-zero-beta factors. (2) *Stressed-regime stylized facts* (Sparviero–Viola): conditioned ensembles must reproduce empirical stressed-window signatures (negative skew, vol expansion, stressed-regime correlation signs), paired block bootstrap for significance. Validation-side covariance only — Bitter-Lesson compliant because it never enters the conditioning path. *Assets:* tilt tables + 994a; new ~100-line completion script.

### C. Statistical authenticity floor — KEEP as floor (revised); merge memorization gate in.

- **SS attack:** "Classifier AUC at chance" is the single most underpowered-by-design pass criterion in the candidate list: with a few hundred non-overlapping real windows and block splits, failing to reject 0.5 is cheap and means little — you can *engineer* a pass by impoverishing features or sample size. Also unconditional: a conditionally insane ensemble of individually-plausible paths sails through. And k-NN/manifold anything in raw 39×30 space is meaningless (high-dim k-NN pathologies).
- **SE:** Fidelity suite exists (`test_block_ar_requirements_v2.py` philosophy); C2ST on summary features is a day's work. The genuinely new build is the memorization triple gate — and it polices the architecture's #1 structural risk (top-3 replay).
- **RM:** A floor, not a selling point; but the memorization number ("the scenario is not a replayed 2008") is directly presentable and operationalizes the existing "no hidden single-prefix generator" rule.
- **Concrete spec:** (a) C2ST on per-window summary vectors (moments, ACF, realized vol, cross-corr Cholesky entries), block-respecting CV; **report AUC as effect size with CI, conformal-C2ST p-value for the formal test; never headline "passed at chance"**; run pooled AND per stressed stratum. (b) *Memorization triple gate:* Junike ratio Π^ρ (ρ=0.25; H0 limit ρ/(ρ+M/N), H0 calibrated by simulation on own panel — i.i.d. is violated); Meehan three-sample Z_U (T=support corpus, P=non-overlapping val windows, Q=generated); Alaa per-path authenticity audit with flagged paths excluded from monitor output. All in low-dim summary embedding space, embedding reported. *Assets:* fidelity suite, `nl_995d_train_pool_replay_labels.py` lineage, 994a nulls.

### D. Metamorphic/invariance axioms — KEEP (tier-1 core). Cheapest real falsification you own.

- **SS attack:** Without a null band every "violation" is seed noise and every pass is unverifiable — mandatory DefaultVariation protocol (≥30 unmodified runs or the 994a bootstrap) before any verdict. LLM paraphrases silently change meaning — validate claim-set equality before counting INV failures. Each axiom is individually gameable (identity pipeline aces composition+reversibility; start-ignoring pipeline aces effectiveness): **joint reporting is non-negotiable.** Placebo p floor = 1/(N+1) — don't report p<0.011 from a 90-pool.
- **SE:** Nearly free given assets: reverse workbench (`nl_scenario_to_narrative_workbench.py`) authors null- and counter-narratives; exposed weights make weight-stability L1 one-liners; grounding checker (`nl_caption_grounding_audit.py` / `nl_grounding_reliability_audit.py`) is the effectiveness pseudo-oracle. New builds: ~50-text placebo corpus (scrambled / style-matched / off-domain), intensity ladders, pre-registered negative-control factor sets.
- **RM:** These are exactly what a skeptical risk manager pokes at live in a demo ("say it harder — does the fan move more? negate it — does it flip?"). Failing DIR/negation in front of the boss is product death; passing is table stakes, not proof of realism.
- **Concrete spec (battery, all vs 994a null band):** (1) null-narrative composition: workbench-authored neutral narrative must reproduce start-only (energy distance + full tilt table inside null band); (2) placebo refuter: real narrative's tilt magnitude ranked against ≥50 placebos, p≤0.05, AND placebo tilts inside start-only band — style-matched scrambles producing tilt = bridge keys on style → hard blocker; (3) effectiveness: mechanical direction checks (existing) jointly reported with (1); (4) paraphrase INV: K≥10 fidelity-validated paraphrases, failure budget pre-registered (~10%), register-change sub-cell as style detector; (5) DIR ladders: sign correct at every rung, monotone magnitude, negation flips, composition ("oil AND dollar") coherent; (6) negative controls: relative null — no excess tilt beyond what the unconditioned support mixture already implies, factor sets pre-registered per narrative. *Assets:* workbench, grounding checker, 994a, `nl_conditionality_stress_test.py`.

### E. Bayes-consistency of the tilt — REVISE → demote to tier-3 diagnostic. Not a gate.

- **SS attack:** Fatal circularity risk: if the grounding checker informs the retrieval reweighting anywhere in the pipeline, you're comparing the pipeline to itself. Even if independent, the checker is an uncalibrated likelihood proxy, so agreement/disagreement has no null; and importance-weighting an unconditional cloud by hypothesis-expression has catastrophic ESS for exactly the stress narratives you care about. You'd be comparing two noisy wrong things and narrating the difference.
- **SE:** Needs unconditional mega-ensembles (K=500–1000, cheap) + checker scoring per path. Doable, but pass criterion undefinable without a placebo-calibrated null.
- **RM:** Unexplainable to the audience; maps to no institutional concept. Zero sign-off value.
- **Keep only as:** a research diagnostic for *mechanism localization* — when retrieval-tilt and likelihood-tilt disagree on narrative-named factors, that localizes a bridge bug (style keying, pool asymmetry). Require checker ⊥ bridge, report ESS, never gate on it.

### F. Plausibility budget — KEEP (revised from "bound" to "calibrated statistic block + frontier").

- **SS attack:** k has no intrinsic scale — the literature's own admitted weak point; an un-calibrated KL bound is rhetoric, and a *tunable* bound is the most owner-distrusted thing imaginable. Worse, F alone actively rewards condition-ignoring: zero tilt = zero KL = maximal "plausibility." And KL is structurally blind to the one failure that matters most for retrieval (requests outside pool support) — absolute continuity guarantees it. F without I (below) is a false-comfort machine.
- **SE:** Weight-space KL(w‖b) over exposed pool weights is a one-liner. Innovation-space modesty q (Antolin-Diaz: z = KL of implied SNI innovations vs N(0,I), q=(1+√(1−e^{−2z/nh}))/2) is elegant and Bitter-Lesson-clean (reference law = model's own N(0,I)); moderate plumbing to extract innovations from rollouts.
- **RM:** As a *reported pair of numbers per scenario* this is exactly ECB/Fed practice and highly convincing: sigma-units/q next to every fan, severity on a separate axis, Leeper-Zha modesty label for regime-change narratives ("USD confidence breaks" → "outside historical support / model validity," not a wider fan).
- **Concrete spec:** Per scenario report (never reject on): (a) ESS = 1/Σw² over the FULL 90-pool pre-truncation (floor ~ a pre-registered effective-analogue count; below → label downgraded from "historically grounded" to "expert judgment"); (b) D_KL(w‖b) vs k* = high quantile of the same quantity computed for workbench-authored narratives of realized regime shifts (2008/2020/2022) — "as plausible as realized regime shifts" is the only honest calibration; (c) innovation-space q, benchmarked on the same named episodes; (d) SWIM-style severity-plausibility frontier over the existing path cloud (exponential tilt of monitored metric within {KL≤k}, monitor ESS collapse) — each scenario plotted against its frontier (BJRS equal-plausibility comparison). *Assets:* exposed weights (free), SNI innovations (generator internals), 994a for calibration quantiles.

### G. Cycle-consistency / SBC — KEEP (revised). **MOST GAMEABLE as pre-registered.**

- **SS attack:** This is the trap candidate. Modrák's result: vanilla SBC with parameter-style test quantities is passed *perfectly* by a pipeline that ignores the narrative and returns start-only — i.e., the validator family advertised as testing the conditioning is blind to conditioning collapse, the product's central failure mode. Also passed by conservative inflation. Second: a sycophantic reverse workbench that parrots query phrasing inflates round-trip scores — authoring must see the path ONLY. Third: it certifies self-consistency under (generator law × workbench authoring), not reality; if workbench narratives differ distributionally from human queries, you've calibrated the wrong channel. Fourth: K=48 → 49 rank bins is too coarse for tails; overlapping starts break band assumptions.
- **SE:** M≥300 full pipeline runs, inference-only — days of background compute, fine. Säilynoja bands and TARP are small implementations; workbench and 994a exist.
- **RM:** "The model agrees with itself" persuades nobody alone; but the derived per-query trust flag (L-C2ST / posterior-SBC at this month's actual query) is a genuinely shippable product feature.
- **Concrete spec:** Closed loop (path → workbench narrative authored from path alone → pipeline → rank of source path), M≥300, K≥100 validation-only; test-quantity battery MUST include Modrák data-dependent quantities (grounded direction-score of path vs authored narrative; retrieval log-score) alongside per-factor terminals, vols, drawdown, 2–3 cross-factor co-move products; Säilynoja simultaneous discrete-uniformity bands, block-adjusted. **Mandatory co-gate:** information gain — median energy-distance(conditioned, start-only from same start) > 0 with paired-bootstrap CI excluding 0, plus direction hit-rate strictly above start-only. TARP joint-coverage curve on path summaries (two reference distributions), pass = within bootstrap band; deployment preference = on-or-above diagonal (conservative, per Hermans). Plus workbench-vs-human-narrative C2ST indistinguishability check before trusting the loop. *Assets:* workbench, 994a, generator; new: rank harness + TARP (~300 lines).

### H. Expert protocols — REVISE → tier-3 / sign-off layer. Never a gate.

- **SS attack:** n raters tiny, style bias dominates (0.76–0.92 for LLM judges; humans not much better on fluent prose), rubric scores unanchored. Energy-scenario literature: user-perceived plausibility diverges from formal consistency in both directions. As evidence, this is the weakest family on the list.
- **SE:** Expensive in scarce human time; LLM-assisted versions require the full judge pre-gate (repetition stability, position/rubric permutation, style probe, ≥90% agreement with mechanical checker on anchored subset) before any verdict counts.
- **RM:** Paradox: least reliable as evidence, most persuasive to the audience. Structured right, it's the sign-off package, which institutions *do* require (SR 15-18 / effective challenge).
- **Concrete spec:** (a) Blinded discrimination: risk manager shown matched decks (generated counterfactual vs workbench-narrated historical episode), pre-registered design, report discrimination AUC with exact CI; (b) regulatory benchmark exhibit: for matching hypotheses, express our scenario AND the CCAR/EBA/BoE scenario in the same sigma/q units, compare at equal plausibility (BJRS illusion-of-safety check); (c) decision-usefulness question ("does this change your hedge?" — Emperor's-New-Scenarios test); (d) full sign-off doc per scenario: grounded channel narrative, severity-vs-episodes, plausibility block, sensitivity (DIR ladder), independent challenge (Codex verifier pattern), explicit role label "exploratory what-if — not a forecast." *Assets:* casebook (`nl_casebook_acceptance_audit.py`), monitor companion guideline doc.

### I. (NEW) Pool-coverage / convex-hull honesty gate — KEEP. Highest-priority new build.

- **SS:** The only check that catches the failure every divergence/EL/cycle method is *structurally blind to*: a narrative whose plausible completion lies outside the analogue pool's representable set. Retrieval-within-history inherits exactly the historical-bias blind spot the systematic-stress-testing program was built to fix; nothing currently in the stack detects it — the system silently returns the nearest representable scenario instead.
- **SE:** Cheap: EL-feasibility LP (does any simplex w satisfy Σwᵢzᵢ = conditional-completion target?), pool max-Maha, angle between requested direction and pool principal subspace. scipy.linprog, per-query milliseconds.
- **RM:** "Outside historical analogue support" is the single most trust-building sentence the monitor can print. Pairs with the Leeper-Zha modesty label.
- **Spec:** Per query: hull-feasibility of the BJRS completion target; if infeasible → mandatory label + ESS/q numbers, never silent nearest-representable output. Pass philosophy: it's a labeling gate (honesty), not a rejection gate. *Assets:* `nl_14x14_support_audit.py` is the natural home.

### J. (NEW) Ex-post envelope monitoring + scenario-set distinguishability — KEEP (tier-1 standing loop).

- **SS:** Weak per-issue power, but it's the only validator that accumulates evidence over the product's life, and it's immune to gaming (reality scores it). Distinguishability: NGFS-style homogeneity (fans indistinguishable from baseline) is a documented decision-usefulness failure that fidelity suites cannot see.
- **SE:** Trivial: standing log; for partially-realized triggers record realized quantile in published fan; energy distance between published fans per issue.
- **RM:** SR 11-7 outcomes analysis — supervisors expect exactly this; retiring stale archetypes (RCP8.5 lesson) demonstrates governance maturity.
- **Spec:** Monthly: (a) realized-quantile log for prior issues' scenarios whose triggers partially fired; (b) pairwise energy distance + terminal-quantile overlap across the issue's scenario set and vs baseline (pre-registered minimum separation, calibrated from seed noise); (c) retirement/relabel rule for archetypes whose near-term segment diverged from observations.

---

## Final ranked acceptance framework

### Tier 1 — cheap, always-on (every query / every monthly issue; failures block publication of that scenario)
1. **Honesty block per scenario [I + F-statistics]:** hull-feasibility LP → support label; ESS (full pool); D_KL(w‖b) vs episode-calibrated k*; innovation-q; severity column (per-factor percentile vs named episodes) on a SEPARATE axis. Graded labels, not silent rejections.
2. **Metamorphic core [D]:** null-narrative composition + placebo refuter + mechanical effectiveness/direction checks, jointly reported, vs 994a null band. Placebo failure = hard blocker.
3. **Admission vetting [C-floor]:** untilted pool baseline passes fidelity suite on val frame; tilted-path fidelity spot-checks (arbitrage, jumps); per-path authenticity flags excluded from output.
4. **Coherence quick-check [B]:** conditional-completion sign gate + negative-control excess-tilt check.
5. **Standing loop [J]:** ex-post envelope log + scenario-set distinguishability, monthly.

### Tier 2 — per-release / promotion gates (verifier-report required, pre-registered in protocol docs)
1. **Matched-episode conditional backtest [A revised]** — ex-ante registry, LEO retrieval, paired sign test + clustered bootstrap + MC exact nulls + published MDE; GW conditional test; twCRPS for outcome-defined queries. *Prerequisite audit: pool/eval-window overlap leakage in current matched-eval scripts.*
2. **Closed-loop SBC + TARP with Modrák quantities + information-gain co-gate [G revised].**
3. **Memorization triple gate + C2ST battery with effect sizes [C revised].**
4. **Full metamorphic suite [D extended]:** paraphrase INV, DIR ladders, composition, in-space/in-time placebos, robustness-value (claim-vs-nuisance partial-R²) text-bridge diagnostic.
5. **Judge pre-gate** (only if any LLM verdict feeds a gate anywhere).

### Tier 3 — research-grade / sign-off layer
1. L-C2ST amortized per-query trust flag (build once at ~2k closed-loop sims → becomes a tier-1 product feature).
2. Posterior-SBC localized to specific monthly queries.
3. Bayes-consistency diagnostic [E demoted] with independence guard; SWIM severity frontier studies.
4. Expert protocols [H revised]: blinded discrimination, regulatory equal-plausibility exhibit, decision-usefulness, sign-off documentation package.
5. Stressed-regime stylized-fact deep study (Sparviero–Viola full battery).

---

## Flags

- **Most load-bearing: A (matched-episode conditional backtest, revised form).** It is the only validator connecting the conditioning machinery to conditional reality, and the institutional license for every counterfactual claim downstream. Everything else certifies internal consistency of a system that could be consistently wrong. It is also the easiest to corrupt (outcome-conditioned episode selection, self-retrieval leakage, "no rejection = calibrated" spin) — which is why its spec carries the most armor.
- **Most gameable: G (cycle-consistency/SBC) as pre-registered.** A pipeline that ignores the narrative entirely passes vanilla SBC rank-uniformity *perfectly* (Modrák posterior=prior), and a parroting reverse workbench inflates round-trip scores — so the candidate marketed as the zero-futures centerpiece would certify a dead conditioning channel. Runner-up: C's "AUC at chance" criterion, which is passable by underpowering. Neither survives without the information-gain co-gate and data-dependent test quantities; with them, G is worth keeping.
- **Single most important new build:** the convex-hull/support honesty gate [I] — the one failure mode (narrative outside analogue support) that every pre-registered candidate and every divergence-based score is structurally blind to.
- **Live bug to audit before anything else:** whether the current 994a/matched-eval analogue pools can contain windows overlapping the evaluated window (leave-episode-out leakage) — files: `experiments/backfill/block_ar/nl_994a_val_frame_start_only_eval.py`, `nl_14x14_matched_eval_summary.py`, `nl_episode_narrative_retrieval.py`.
- **Standing epistemic disclaimer for all reports:** tier-2 closed-loop passes certify self-consistency under (frozen generator law × workbench authoring), not real-world conditional fidelity; that burden rests on tier-2 item 1 plus the val-frame checks where futures legitimately exist. State it before a verifier does.
