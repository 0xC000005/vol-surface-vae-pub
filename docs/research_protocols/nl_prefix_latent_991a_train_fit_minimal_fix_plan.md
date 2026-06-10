# NL 991-Series Plan: train_fit Minimal Fix for Stride-5 14+14 Contrastive Retrieval Training

Date: 2026-06-10
Status: stored plan (not yet executed)
Predecessors: 990a manifest, 990e pilot training, 990f support audit, 990g matched eval ("Do not promote")
Falsification classification for 990g: PRIMARY CLASS = train_fit (see RESEARCH_LOG 2026-06-10 entry)

## Diagnosis summary (990g loss)

Verified: `_target_cosine_and_rank` normalizes both prediction and memory rows before ranking (lines 518-519), so the in-sample rank-801 result is a true cosine-ranking failure, not a metric-implementation artifact. `reconstruction` is ruled out. I now have everything needed to adjudicate.

---

# FALSIFICATION ADJUDICATION — 990g (trained stride-5 contrastive retrieval lost to start-only downstream)

Workflow applied: `docs/research_protocols/autoresearch_falsification_workflow.md` (11 classes, exactly one primary; Minimal-Fix Rule; one-axis-per-iteration; Switch Bar).

## Verified load-bearing evidence (re-checked against artifacts this session)

1. **In-sample fit failure is real and not a metric artifact.** `990e/projected_memory/projected_memory_training_report.json`: true_memory_rank_median = 801/4010, recall@10 = 0.019, target_cosine_mean = 0.824 — computed on TRAINING texts, and `_target_cosine_and_rank` (nl_14x14_manifest_retrieval_training.py:511-534) row-normalizes both sides, so this is genuine cosine-rank failure on the model's own training targets. Text-space adapted same-label recall@1 = 0.0927 (raw 0.046).
2. **Downstream CRPS rank is PERFECTLY monotone (Spearman = 1.0, 7/7 methods) in mean support temporal distance** (recomputed from `990f/matched_top3_90_bridge_report.json`): start_only 126 → 984a 818 → 982g 1568 → 981t 1761 → text_space_14x14 1863 → raw_openai_14x14 2029 → projected_memory_14x14 2105. This exactly matches the 990g CRPS ordering 1→7.

## Candidate adjudication

**(a) `train_fit` / undertraining.**
FOR: The bridge fails *in-sample* — median rank 801/4010 and recall@10 1.9% on texts it was MSE-aligned to; 80 steps ≈ 1.8 epochs; single seed; loss traces still descending/noisy at step 80 (text-space 4.67→4.78 bounce; projected MSE 0.338); no validation-coupled stopping, which paper Alg 1 *mandates* ("until validation support quality and held-out scenario scores stop improving") and the trainer omits (fixed-step loop); no checkpoints saved so fit can never even be measured out-of-sample. Training moved things the right way where it acted: recall@1 doubled (0.046→0.093) and trained text-space beat its own raw baseline downstream (0.5641 vs 0.5751 CRPS) — the fit axis is live and far from its ceiling. current_truth already flags 990e as "pilot-scale."
AGAINST (adversarial): raw_openai_14x14 — zero training — also lost (rank 6), so undertraining cannot explain the whole 990g ranking. Answer: the gated question is why the *trained* methods lost; the raw baseline's failure is the control showing raw embedding geometry lacks regime locality, which is precisely what training is supposed to add and hasn't yet. Also, "loss still decreasing" is inferred from 6 trace points; possible the curve was near-flat. But rank 801 in-sample at any loss level means the objective's fixed point was not reached.

**(b) data/coverage mismatch (trained on 802 stride-5 targets, ranked over 4010-row bank).**
FOR: bank is ~1.8x larger than the 2,279 unique training labels; untrained rows act as distractors.
AGAINST (decisive): labels span 0..4009 — training targets are spread across the full bank, and the rank failure occurs on the model's OWN training labels in-sample. A fit bridge ranks its own MSE target near 1 regardless of distractor count; 801 is not a distractor effect. Index alignment was explicitly handled (eval required `--eval_split train_tail --max_windows 4010`), ruling out the `reconstruction`/audit-alignment reading. Not a named workflow class anyway. Rejected as primary; at most secondary aggravator.

**(c) `test_mismatch` (13-window matched eval too narrow/structurally biased).**
FOR: 13 windows, one contiguous train-tail stretch; queries are train-tail of the all-train 734a checkpoint and 939a train_all bank, structurally favoring temporally-local supports; adapters were trained on the query texts themselves; far smaller than the 66-window/16-sample frame.
AGAINST (decisive under governance): the workflow requires an oracle/split audit demonstrating the metric is "unstable, contradictory, or not aligned with the risk objective" BEFORE assigning test_mismatch — no such audit exists. And the 990g ordering *replicates* the independently verified 66-window 982g scoreboard (start-only > all narrative-aware; verifier 2026-06-02 PARTIAL says exactly this), i.e., the test is consistent with the established gate, not contradicting it. Low power is a breadth caveat to log, not a failure class. Rejected.

**(d) objective/method mismatch (margins satisfied locally, no global alignment; MSE conditional-mean regression).**
FOR: the geometry is suggestive — MSE weight 1.0 dominates 0.2/0.2/0.5/0.25; hinge margins only enforce order vs ONE matched negative (75.2%/69.2% positive) and singleton-label mates make some hinges trivial; cosine 0.824 with rank 801 is the signature of regression toward a central memory vector; ~10 texts/label with heterogeneous view families (sparse fact tokens) could make the conditional mean the true optimum even at convergence.
AGAINST (decisive under the workflow's own definitions): the closest class is `backend` ("...train-fit ... adequate, but the sampler/objective still cannot allocate mass") — train-fit adequacy is a stated PRECONDITION of that class, and it is not met. At 1.8 epochs we cannot distinguish "MSE regresses to the mean at convergence" from "MSE hasn't converged." (d) is unfalsifiable until (a) is cleared. Retained as the pre-registered reclassification target.

**(e) baseline-strength reality (start-only genuinely strong; value only via Stage-2 reranker).**
FOR: strongest steady-state evidence in the file — start-only wins absolute CRPS/Energy on BOTH the 66-window and 13-window frames; 984a (replay-supervised Stage-2) is the best narrative-aware method on both; the perfect temporal-distance monotonicity shows this slice rewards regime locality, which start-only encodes by construction and text similarity does not; raw retrieval (no training) loses too.
AGAINST (decisive): not a workflow failure class — it is a steady-state *conclusion*, and it requires the trained method to be at its fit ceiling to be adjudicable. It is not: the trained heads cannot yet retrieve their own training targets. Also 990g confounds it structurally: the 14x14 methods ran Stage-1-only against 984a-with-Stage-2. (e) is the live hypothesis the minimal fix is designed to test, not the assignable class today.

## PRIMARY CLASS: `train_fit` (exactly one)

Secondary (logged, not primary): objective-geometry concern (d)→`backend`-analogue as pre-registered reclassification target; eval-breadth caveat under (c) as a reporting note only.

**Mechanistic 3-level causal story (knob → propagation → metric):**
- **Knob:** training budget/selection — 80 optimizer steps (~1.8 epochs over 22,456 texts), single unrecorded seed/lr, fixed-step loop with NO held-out split, NO validation-coupled stopping (violating paper Alg 1's committed stopping rule), and no checkpoint saved (weights discarded, so fit cannot be extended or measured on new queries).
- **Propagation:** the heads never reach their objectives' fixed points — the bridge lands in the central region of memory space (cosine 0.824 to target yet ~800 memories rank above it, in-sample, under verified cosine ranking; recall@10 1.9%) and the text-space adapter resolves only coarse semantic clusters, not window/regime identity (recall@1 9.3% over 2,279 labels). A retriever that cannot resolve window identity cannot find the query's near-duplicate regime neighbors (stride-5 windows ±60-100 share most of their 30-day prefix and near-identical narratives); it instead selects cluster-level matches a median ~1,800-2,100 windows away with 0.0 support Jaccard vs every baseline.
- **Metric:** temporally/economically distant supports fed through the frozen 734a SNI + top3/90 produce worse-conditioned mixtures; downstream CRPS is perfectly monotone in support temporal distance across all 7 methods (verified), placing the trained methods at ranks 5 and 7 vs start-only's 0.5193 CRPS.

This chain is checkable at each level: activate the knob (steps↑ + holdout + stopping) → level-2 observables must move (held-out true-memory rank, recall@k, AND support temporal distance for in-regime queries dropping) → level-3 gap to start-only must shrink on the unchanged matched gate.

## Prescribed next move (Minimal-Fix Rule, one axis: training budget/model selection only)

1. **Code prerequisites to measure fit (reporting contract, not new research axes):** add a window-level held-out text split and checkpoint saving (`torch.save`) to `nl_14x14_manifest_retrieval_training.py`; both are already mandated by paper Alg 1's stopping rule and the 14x14 plan's diagnostics. Record lr/seed in the report (990e did not).
2. **Retrain the SAME two methods, same losses/weights/architecture untouched:** ~2,000-5,000 steps (cost: minutes on the 3070 Ti; the 276 MB embedding matrix fits) with early stopping on held-out support quality (true-memory rank / same-label recall), 3 seeds. Reuse the 990e embedding cache by symlinking `shared_embedding_cache` and splitting at index level with the text list unchanged (so the blake2b digest still hits; $0 OpenAI).
3. **Include the plan-required missing diagnostics** in the same run: sparse-query recall, view-family robustness, direction pass rate/start fit, fixed-start conditionality — these also discriminate whether sparse views drag recall@1 (relevant to the (d) reclassification).
4. **Pre-registered kill condition:** if at converged loss across seeds the HELD-OUT true_memory_rank_median does not reach ~top-decile of the bank (≤~400) with materially improved recall@10, `train_fit` is falsified as primary → reclassify to the objective-level class (`backend` analogue for the retrieval objective: MSE conditional-mean regression + local-only margins) and only then change ONE objective-axis knob (e.g., margin/contrastive vs MSE reweighting, or replay-preference supervision = the 984a Stage-2 composition the log already names).
5. **Only after fit passes:** rerun the UNCHANGED 990f/990g matched gate (same 13 windows, CRN seed 9907; optionally add the 66-window frame as additive reporting for power — same test, not a test change). If trained retrieval still loses *with demonstrated retrieval fidelity and reduced support temporal distance*, hypothesis (e) becomes adjudicable and contrastive-Stage-1 + 984a-Stage-2 is the sanctioned one-axis follow-up.

**Explicitly NOT licensed now:** changing the eval (no test_mismatch audit; the 13-window ordering replicates the verified 66-window scoreboard), switching retrieval family/backend/architecture (Switch Bar: zero minimal fixes attempted; `backend` precondition unmet; architecture switch requires SUSTAINED train_fit), any promotion/default change (nearest-similar top3/90 stays), and any "paper method" framing of the text-space lane (the paper trains only B_phi; text-space is research-only).

Key files: `experiments/backfill/block_ar/nl_14x14_manifest_retrieval_training.py` (add split + checkpointing + stopping), `experiments/backfill/block_ar/nl_scenario_demo_outputs/stride5_14x14_retrieval_training_openai_full_990e/` (cache to symlink), `experiments/backfill/block_ar/nl_scenario_demo_outputs/stride5_14x14_support_audit_990f/matched_top3_90_bridge_report.json` (temporal-distance verification source), `docs/research_protocols/autoresearch_falsification_workflow.md` (classes/rules applied).

---

# EXPERIMENT DESIGN — 991-series: train_fit minimal fix for stride-5 14+14 contrastive retrieval

Primary class: `train_fit`. One research axis moved: **training budget / model selection**. Losses, weights, margins, temperatures, architecture, optimizer family, batch size, embedding model: ALL frozen at 990e values. ID check done: nothing above 990g exists in `nl_scenario_demo_outputs/` — 991a–991e are free.

---

## 0. Code prerequisites (reporting contract, pre-registered as NOT research axes; mandated by paper Alg 1 + plan diagnostics)

Edit `experiments/backfill/block_ar/nl_14x14_manifest_retrieval_training.py` (+ extend `test_code/test_nl_14x14_manifest_retrieval_training.py` first, TDD):

1. **Window-level holdout (Tier A)** — new flags `--val-window-ranges`, `--purge-gap 30`. Split on `label_window_index`, NOT on text. Fixed deterministic blocks (identical across seeds): `[610,730) [1490,1610) [2370,2490) [3250,3370) [3915,4010)` = 575 val windows (14.3% of 4,010; all inside global train[0:4040]; last block contains the 13 matched query windows 3945–4005, so the gate's query texts are never trained on). Purge ±30 windows (one prefix length → zero prefix-day overlap between train and val) excluded from BOTH sides. Training sampler excludes any example whose label ∈ val∪purge; pair rows excluded if EITHER endpoint window ∈ val∪purge; val pairs reserved for held-out margin metrics. Emit `holdout_split.json` (block ranges, purge ranges, train/val example+pair counts, manifest digest).
2. **View-level holdout (Tier B)** — `--holdout-view-families <sparse_fact_family>,<one_professional_family>`: exclude 2 of the 14 view families (and their paired hard-negatives) from the sampler on TRAIN windows; they become same-label recall queries against trained views. This is evaluation-only and directly implements the plan's sparse-query-recall and view-family-robustness requirements.
3. **Checkpoints** — `torch.save({"state_dict", "config", "seed", "lr", "argv", "git_sha", "best_step"})` → `text_space_adapter_{best,final}.pt`, `projected_memory_bridge_{best,final}.pt`. (Unblocks projecting NEW query texts; required for 991d.)
4. **Validation-coupled stopping** — `--max-steps 5000 --eval-every 250 --patience 8`. Best-checkpoint selection: bridge = held-out true_memory_rank_median (Tier A, exact: the val window's own memory IS in the frozen 4,010 bank); text-space = Tier-B held-out same-label recall@10. Knob Ledger entries: patience + val blocks, class=train_fit, removal criterion = converged well before max-steps across seeds. Downstream scenario-score coupling is done at checkpoint level (best-val checkpoint → single 991c run), logged as partial Alg-1 compliance.
5. **Report fixes** — record lr/seed/argv (990e didn't); loss trace every 50 steps; pair-margin eval on rng-sampled + val pairs instead of FIRST-5000 prefix; expose `--embedding-cache-dir` (kwarg exists at lines 339/543, just not in argparse).

LR stays constant 1e-3 AdamW wd=1e-4 (no schedule — selection via best-val checkpoint; adding decay would be a second knob).

## 1. Iteration 1 — 991a retraining (the minimal fix)

Per seed S ∈ {0,1,2} (3 seeds):

```bash
OUT=experiments/backfill/block_ar/nl_scenario_demo_outputs/stride5_14x14_retrieval_training_openai_holdout_991a_seed${S}
mkdir -p $OUT && ln -s "$(pwd)/experiments/backfill/block_ar/nl_scenario_demo_outputs/stride5_14x14_retrieval_training_openai_full_990e/shared_embedding_cache" $OUT/shared_embedding_cache

PYTHONPATH=. uv run python experiments/backfill/block_ar/nl_14x14_manifest_retrieval_training.py \
  --method both --embedding-backend openai --embedding-model text-embedding-3-large \
  --steps 5000 --batch-size 512 --lr 1e-3 --pair-margin 0.15 \
  --val-window-ranges 610:730,1490:1610,2370:2490,3250:3370,3915:4010 --purge-gap 30 \
  --holdout-view-families sparse_facts,professional_v2 \
  --eval-every 250 --patience 8 --seed ${S} --device cuda \
  --output-dir $OUT
```

(`uv run` mandatory — bare `python` resolves to conda without pyarrow. Use exact view-family names from the 990a manifest's view tags.)

- **Embedding cost: $0.** Same `--examples-jsonl`, no `--max-targets`, text list byte-identical → blake2b digest hits the symlinked 990e cache (split is index-level only). Verify `cache_hit: true` in each report before proceeding.
- **Runtime:** ~10–15 min/seed both methods on the 3070 Ti (276 MB embedding matrix on-device, ~0.83M-param MLPs; 5,000 steps ≈ 114 epochs at B=512); < 1 h total for 3 seeds.
- **Unchanged (one axis!):** SupCon temp 0.07 + 0.5·hinge(0.15) for text-space; 1.0·MSE + 0.2·cos + 0.2·InfoNCE + 0.5·source-margin + 0.25·reciprocal-margin for the bridge; batch 512/pair 256; architecture; 990a manifest; 939a memory bank.

**Required outputs per seed:** `training_run_report.json/.md`, `holdout_split.json`, `text_space/text_space_training_report.json` + arrays NPZ + both checkpoints, `projected_memory/projected_memory_training_report.json` + arrays NPZ + both checkpoints. **Seed aggregate:** `stride5_14x14_retrieval_training_openai_holdout_991a_aggregate/seed_aggregate_report.json` (median/min/max of every held-out metric across seeds; ~30-line summarizer added to the trainer or standalone).

## 2. Required diagnostics (in the 991a reports — plan contract)

Per method, per seed, **held-out**:
- Bridge: `heldout_true_memory_rank_median`, `heldout_recall_at_10_true_memory`, `heldout_target_cosine_mean/median` (Tier-A val windows vs full 4,010 bank); in-sample equivalents kept for the gap.
- Text-space: Tier-B `heldout_same_label_recall_at_{1,10}` per held-out view family (sparse family = **sparse-query recall**; professional family = view-robustness); Tier-A `median_topk_temporal_distance` (median |retrieved window − query window| over top-10 train-pool retrievals, raw vs adapted) — the regime-locality observable from the level-2 causal chain.
- Both: held-out pair-margin mean + positive rate (source AND reciprocal for the bridge) on val pairs only; per-view-family recall table over all 14 families (in-sample, Tier-B held-out where applicable); loss traces at 50-step cadence.

## 3. Downstream gate — 991b/991c/991d/991e (only after the fit gate passes)

**Fit gate (kill condition, pre-registered):** at converged val metric (early stop triggered or 5,000 steps), in ≥2/3 seeds:
- Bridge: held-out true_memory_rank_median ≤ 400 (top decile of 4,010) AND held-out recall@10 ≥ 0.10 (vs 0.019 in-sample now).
- Text-space: Tier-B held-out same-label recall@10 ≥ 2× the raw-embedding control (computed in the same run, deterministic) AND Tier-A median top-10 temporal distance reduced ≥ 30% vs raw.
**If not met → `train_fit` is FALSIFIED as primary; STOP — do not run 991c.** Reclassify to the objective-geometry class (`backend` analogue) per the pre-registration; see Iteration 2A.

**991b — support audit** (extend `nl_14x14_support_audit.py` to emit the plan-required `direction_pass_rate`, `accepted_start_fit`, and `mean_abs_support_temporal_distance` per method): point its trainer-arrays inputs at the best seed's 991a dir; same 13 query windows 3945..4005, `--query-view full_professional --top-k 3 --support-pool-size 8 --temporal-gap 30`; output `stride5_14x14_support_audit_991b/{support_level_audit_report.json,.md}` + bridge reports. **Mechanism gate:** trained methods' mean |support−query| must drop from 1,863/2,105 to ≤ 1,000 ("knob propagated"); ≤ 800 (984a level) to be promotion-eligible.

**991c — matched 13-window frozen-SNI eval, byte-identical to 990g** (no test change without a test_mismatch audit): 734a checkpoint, CRN base_seed 9907, `--support-sampling-mode field_weight --top-k 3 --samples 4 --n-steps 30 --eval_split train_tail --max_windows 4010`, seed 776, same 7-method roster. Output `stride5_14x14_matched_top3_90_scenario_eval_991c/scenario_level_eval_report.json` + `matched_scenario_eval_by_method.json/.md` via `nl_14x14_matched_eval_summary.py`, plus `paired_window_deltas.json` (per-window CRN-paired CRPS/Energy deltas vs start_only + sign test).

**991d — eval breadth (additive reporting, NOT a test change):** 13 windows is sufficient for the unchanged gate but NOT for a promotion claim (single contiguous train-tail stretch, low power, structurally locality-favoring). Promotion-grade comparison vs start-only/984a runs on the **66-window / 16-sample frame** that produced the verified 982g/984a scoreboard. Requires the saved checkpoints: embed the 66 query texts (<$0.01) and project through `*_best.pt` (small utility or `--checkpoint` flag on the audit script). Output `stride5_14x14_matched_66w_scenario_eval_991d/`. Runtime ~1–3 h.

**991e — fixed-start conditionality lift** (plan-required, missing from the 990 series): run the existing fixed-start audit harness over the trained-method supports → `stride5_14x14_fixed_start_conditionality_991e/`.

**Promotion condition (all required):** fit gate passed + mechanism gate ≤ 800 + trained method beats start-only on absolute CRPS AND Energy and ≥ matches 984a on the 66-window frame (cov80 within ±0.03), 13-window frame non-regressing + independent verifier AGREE (dated report in `docs/research_protocols/nl_prefix_latent_verifier_reports/` + spec in `nl_prefix_latent_promoted_specs/`). Anything less: logged negative, "Do not promote" stands, nearest-similar top3/90 default unchanged.

## 4. Iteration 2+ contingencies (keyed to outcomes, one axis each)

- **A. Fit gate fails at convergence (3 seeds)** → reclassify to objective-geometry (`backend` analogue: MSE conditional-mean regression + local-only margins). 992a changes ONE objective knob with a Knob Ledger entry: rebalance `mse_weight` 1.0→0.2 with `contrastive_weight` 0.2→1.0 (global ranking term dominant). Nothing else.
- **B. Fit passes, mechanism moves, but 991c/991d still lose to start-only** → hypothesis (e) is now adjudicable; sanctioned one-axis follow-up = **contrastive-Stage-1 + 984a-Stage-2 composition**: retrain the replay-preference reranker (`nl_episode_grounded_text_support_preference_reranker.py`) on the trained-retrieval candidate pool (991b already emits the bridge-report shape it consumes; reranker features shift under the new pool, so retraining it is required, still one axis). 992b.
- **C. Fit passes but support temporal distance does NOT drop** (high retrieval fidelity, still-distant supports) → this is the only outcome licensing a **test_mismatch AUDIT** (not a test change): out-of-regime / oracle split audit per the workflow, before touching the eval.
- **D. Fit + downstream both pass** → verifier request, then promotion procedure.
- **NOT available as a contingency:** bank-coverage densification via the 32,080-row daily corpus — permanently de-scoped 2026-06-10 (`nl_prefix_latent_hard_negative_corpus_plan.md:121-131`); stride-5 is the accepted corpus criterion. Do not reopen.

## 5. Claim bans (in force until the named gate passes)

- **Text-space lane: NO paper-alignment claim, ever** — the paper trains only B_phi; text-space is research-only (plan says it does not change the paper/demo default). Permanent framing constraint, not gate-dependent.
- **Bridge "paper Alg 1 method alignment"**: claimable only after 991a lands validation-coupled selection + held-out metrics, and only with the embedding-model deviation flagged (trainer uses text-embedding-3-large; the paper's experiments name -3-small). Number-equivalence to the 380-window-corpus figures (cosine 0.8476–0.8597, recall@3 17.5%, etc.): never claimable.
- **No "trained on the full 4,010-window corpus" claim** — stride-5 only (plan's forbidden-claims list).
- **No promotion / demo / current_truth / paper-default change** until the full promotion condition in §3 including verifier AGREE. Until the fit gate passes, the only claimable result remains: 990g negative + `train_fit` primary classification (no intrinsic claims about contrastive retrieval either way).
- **HEDA**: 991a results documented in RESEARCH_LOG.md (blocking) before 991b launches; `qmd update --collection research && qmd embed` after.

Key files: `experiments/backfill/block_ar/nl_14x14_manifest_retrieval_training.py` (edits §0), `test_code/test_nl_14x14_manifest_retrieval_training.py` (TDD first), `experiments/backfill/block_ar/nl_14x14_support_audit.py` (991b extensions), `experiments/backfill/block_ar/nl_14x14_matched_eval_summary.py` (991c summarizer), 990e cache at `experiments/backfill/block_ar/nl_scenario_demo_outputs/stride5_14x14_retrieval_training_openai_full_990e/shared_embedding_cache`.
