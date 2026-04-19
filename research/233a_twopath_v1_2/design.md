# 233a-v1.2 Design Spec — Targeted Bug Fixes on v1

**Date:** 2026-04-18
**Parent experiment:** 233a-v1 (see `research/233a_twopath_v1/design.md`; results at commits b09dd28..fa857df)
**Diagnostic basis:** Four-agent mechanistic investigation, commits `b732ee0..37b635f` (2026-04-18)
**Objective:** Test whether the 6 bugs diagnosed on v1 are fixable within the AR paradigm, via 6-variant ablation grid. Single seed (42). Architecture/loss exploration phase, not statistical-validation phase.

---

## 1. Architecture Overview

Inherits 233a-v1's SlowPath, FiLM, ScaleJumpHead, CoarseFeatures, TwoPathFactorAR. v1.2 modifies ONLY the **wiring and emission** paths. Everything else (curriculum, anchor, BK batch expansion, factor head, idio head, Λ/D modulation, GRU cells, jump mixture mechanism) is frozen from v1.

### Changes vs v1 (targeting 4 bug categories)

**C1. FiLM input pipe (fixes Bugs 2 + 5).**
FiLM no longer consumes `(log1p(s_hybrid), log1p(lam_hybrid))`. Instead reads `h_slow: (B, 8)` directly. The `linear_s`, `linear_lam`, `s_hybrid`, `lam_hybrid` computations are removed from the FiLM-input path (the slow-path internal state machine still maintains them for the aux diagnostic heads, but they do not feed FiLM).

**Rationale:** v1 diagnostic (commit `54192dd`) showed `linear_s` / `linear_lam` learned large negative biases that ReLU-kill `lam_hybrid` in 2/3 seeds and suppress `s_hybrid` turb/calm ratio from 3.1× (pure EWMA) to 1.04-1.08× (hybrid). `h_slow` is directly discriminative: PC1 AUC 0.79 on val regime labels, Cohen's d 1.17. Piping `h_slow` directly eliminates the broken scalar bottleneck.

**C2. Jump supervision wiring (fixes Bug 1).**
Add first-class BCE loss term:
```python
L_film_jump_bce = (1/(N-1)) · Σ_{t=0}^{N-2} BCE_with_logits(
    q_seq_film[t],                                                       # FiLM's p_jump_logit output
    (||future[:, t+1] - future[:, t]||_2 > q90_train).float()             # q90 exceedance target
)
```
where `q_seq_film` is a new per-step list of FiLM's `p_jump_logit` outputs. The v1 slow-path BCE on `q_seq_slow` (against `slow_path.jump_prob_head`) remains, but it no longer has to carry the burden of teaching the mask.

**Rationale:** v1 diagnostic (commit `1170b21`) showed `film.logit.weight` received **zero direct gradient** from L_jump (the BCE went to `slow_path.jump_prob_head`, not FiLM). Only L_ES via straight-through Bernoulli reached FiLM, ~8× weaker than what L_jump would have given. By ep 8 the logit had drifted negative enough that `σ(logit) << 0.5` → `mask ≈ 0` → straight-through gradient ≡ 0 → dead zone. Direct BCE supervision breaks this gradient starvation.

**C3. State consistency regularizer (fixes Bug 2).**
```python
L_state = (1/W) · Σ_{t=0}^{W-1} MSE(h_slow_free[t], h_slow_teacher[t].detach())
```
with W = 5 steps. Uses the existing `_run_teacher_branch` method (already implemented in v1 for the teacher-forced slow-state trajectory). Pulls self-fed h_slow toward the teacher-forced trajectory over the first 5 steps of each rollout.

**Rationale:** v1 diagnostic (commit `12b1820`) showed `s_t` diversity collapses 50-70% under self-fed rollout (ratio 0.29-0.92 across seeds) while teacher-forced maintains diversity (ratio 1.08+). The reg restores regime discriminability late in the rollout.

**C4. Tail-aware emission (fixes Bug 6) — two orthogonal mechanisms, toggleable:**

**C4a (twCRPS aux loss):** threshold-weighted CRPS on the pathwise-max-|Δx| functional:
```python
gen_max = (samples[:, :, 1:] - samples[:, :, :-1]).abs().max(dim=2).values   # (B, K, D)
gt_max  = (future[:, 1:] - future[:, :-1]).abs().max(dim=1).values            # (B, D)
indicator = (gt_max > q90_train).float()                                      # (B, D) — tail weight
term1 = (gen_max - gt_max.unsqueeze(1)).abs().mean(dim=1)                     # (B, D)
term2 = 0.5 · (gen_max.unsqueeze(1) - gen_max.unsqueeze(2)).abs().mean((1,2)) # (B, D)
L_twcrps = ((term1 - term2) · indicator).mean()
```

**C4b (learned emission link):** replaces `delta = sinh(v) * local_scale` with a per-cell learned convex combination:
```python
alpha = σ(link_gate(cond))          # (BK, D), per-cell
g(v)  = alpha · tanh(v) + (1 - alpha) · sinh(v)
delta = g(v) · local_scale
```
`link_gate: Linear(hidden_dim, D)` with weights and bias zero-initialized, so `alpha = σ(0) = 0.5` at init (halfway between bounded tanh and unbounded sinh). Data determines per-condition whether bounded (calm) or unbounded (turb) behavior is preferred.

**Rationale for both:** v1 diagnostic (commit `d23a82d`, `75bab59`) showed max|δ| attenuation is a shared 227a-family emission issue: generated tails are systematically suppressed by `sinh · local_scale`, producing max_jump_ks 0.7-0.95 vs the 0.20 gate. Historical non-AR H=1 architectures (183c) achieved max_jump_ks 0.12, proving the gate is achievable with the right emission. C4a provides a training signal toward heavier tails; C4b provides architectural capacity for tail flexibility.

### Invariants (unchanged from v1)

- `use_scale_anchor = True, scale_anchor_alpha = 0.50` throughout (proven production setting)
- BK batch expansion for fast path
- AR(1) factor noise with `rho = 0.8`
- Jump mixture (straight-through Bernoulli) in emission
- Curriculum `0:5,10:15,25:30`, feedback-decay end at ep 30
- Loss core `L_ES + 0.05·L_VS`
- Factor rank 6, slow_hidden 8, hidden_dim 128

---

## 2. Variant Grid

Six variants, each seed 42, each 60 epochs. Isolates the effect of each fix through carefully-designed knob toggling.

| # | Variant | FiLM pipe (C1) | BCE→film.logit (C2) | twCRPS aux (C4a) | Learned link (C4b) | State reg (C3) |
|---|---|:-:|:-:|:-:|:-:|:-:|
| 1 | `v1.2-control` | v1 bottleneck | v1 wiring | off | off | off |
| 2 | `v1.2-minimal` | h_slow direct | on | off | off | on |
| 3 | `v1.2-aux` | h_slow direct | on | on | off | on |
| 4 | `v1.2-link` | h_slow direct | on | off | on | on |
| 5 | `v1.2-both` | h_slow direct | on | on | on | on |
| 6 | `v1.2-noreg` | h_slow direct | on | on | on | off |

### Attribution logic

- `minimal vs control`: effect of C1 + C2 (FiLM pipe + BCE wiring) alone
- `aux vs minimal`: effect of C4a (twCRPS aux) in isolation
- `link vs minimal`: effect of C4b (learned link) in isolation
- `both vs minimal`: combined emission effect; is it additive, redundant, or superadditive?
- `both vs noreg`: is state reg (C3) load-bearing?
- `best v1.2 variant vs 229a`: did we beat the incumbent?
- `best v1.2 variant vs v1 (commits `fa857df`)`: did we close gates that v1 failed?

---

## 3. Data Flow

### Shape contracts

```
INPUTS
  history : (B, 30, 5, 5) in [0,1]           (grid) or (B, 30, 25) flat
  future  : (B, 30, 25) in [0,1]             flat is canonical; reshape to (B, 30, 5, 5) for encode_history only

SLOW PATH (unchanged from v1)
  init_slow_state(history) produces:
    h_slow     : (B, 8)        GRUSlow warm-up
    s_ewma     : (B,)          analytic EWMA (still computed, not fed to FiLM in v1.2)
    lam_hawkes : (B,)          analytic Hawkes (not fed to FiLM in v1.2)
    buffer     : list[(B, 25)] × 30
    delta_t_last_jump : (B,)

FAST PATH (227a BK convention, BK = B*K)
  cond     : (BK, 128)   from encode_history -> expand
  local_scale : (BK, 25)
  prev     : (BK, 25)
  scale_anchor_bk : (BK, 25)
  z_f (AR(1)) : (BK, 6)
```

### Per-step dataflow (training, with all v1.2 knobs ON; see Section 3 diagram detail)

1. Update slow state: `sp_out = slow_path.step(coarse_t, h_slow, s_ewma, lam_hawkes, delta_t_last, mean_sq_dx, j_t)`
2. **[C1] FiLM = FiLMFromHSlow(h_slow)** → `γ_Λ, β_Λ, γ_D, β_D, drift_bias, p_jump_logit` each shaped (B, …). Expand to BK.
3. Fast emission: `v = einsum("bdr,br->bd", Lambda_mod, f_scores) + D_mod·i_resid + drift_bk`
4. Jump mixture: `v = v + mask · scale_jump_head(lam_t) · ε_extra` (mask from straight-through Bernoulli on p_jump_logit)
5. **[C4b] Emission link:** `g_v = α·tanh(v) + (1−α)·sinh(v)` (if `use_learned_link`) else `g_v = sinh(v)`
6. `delta = g_v · local_scale`, `next_iv = (prev + delta).clamp(1e-4, 1−1e-4)`
7. Sample stored, feedback chosen (PF or self-fed), fast state updated
8. **[C2] Collect `q_seq_film[t] = p_jump_logit`** for BCE loss
9. **[C3] Collect `h_slow` into `h_slow_free_seq`** for state-reg loss
10. (If `return_teacher_h`) run `_run_teacher_branch(history, future, W)` for detached teacher h_slow trajectory

### External API (preserved from v1)

- `sample_batched(history, n_samples, n_steps, ...) → (B, K, N, 5, 5)` — unchanged
- `forward(history, n_members, n_steps, ...) → (B, K, N, 5, 5)` — unchanged (variant dispatch)
- `load_model(ckpt, device) → (model, payload)` — same signature

Allows evaluator (`evaluate_220b_multihorizon_path_suite.py`) to treat v1.2 model identically to v1.

---

## 4. Loss Decomposition

```
L_total = L_ES
         + λ_VS     · L_VS             (v1-inherited)
         + λ_BCE    · L_film_jump_bce  (C2; new)
         + λ_twcrps · L_twcrps         (C4a; new)
         + λ_state  · L_state          (C3; new)
```

Per-variant λ:

| variant | λ_VS | λ_BCE | λ_twcrps | λ_state |
|---|---|---|---|---|
| `v1.2-control` | 0.05 | 0.00 | 0.00 | 0.00 |
| `v1.2-minimal` | 0.05 | 0.05 | 0.00 | 0.10 |
| `v1.2-aux` | 0.05 | 0.05 | 0.05 | 0.10 |
| `v1.2-link` | 0.05 | 0.05 | 0.00 | 0.10 |
| `v1.2-both` | 0.05 | 0.05 | 0.05 | 0.10 |
| `v1.2-noreg` | 0.05 | 0.05 | 0.05 | 0.00 |

Notes:
- `v1.2-minimal` and `v1.2-link` have identical λ — they differ only in the `use_learned_link` architectural flag.
- `v1.2-both` and `v1.2-noreg` differ only in `λ_state`.
- `v1.2-control` and `v1.2-minimal` differ in `use_v1_film_pipe` (C1) plus all 3 new λ.
- All λ values are chosen at the same order of magnitude as v1's `λ_VS = 0.05`. If any loss term dominates during training (> 3× `L_ES`), lower it 2× and restart — this is a Stage-A (smoke) gate (see Section 8).
- L_ES, L_VS unchanged from v1 (average over 30 steps).

Bitter-Lesson check: all new loss terms weight LEARNED model outputs; no per-cell constants, no domain heuristics. `q90_train` is a single scalar threshold computed once from training-split data (`coarse_pca_233a.npz`) — same usage as v1.

---

## 5. Training Recipe

### Hyperparameters (identical to v1 — keeps fix-attribution clean)

- **Epochs:** 60
- **Batch size:** 32
- **n_members:** 8
- **n_steps (curriculum max):** 30
- **Curriculum schedule:** `0:5,10:15,25:30`
- **Feedback decay end:** 30
- **Optimizer:** AdamW
- **lr:** 4e-3
- **weight_decay:** 1e-4
- **grad_clip:** 1.0
- **rho (AR factor noise):** 0.8
- **ewma_alpha:** 0.20
- **scale_floor:** 1e-4
- **use_scale_anchor:** True
- **scale_anchor_alpha:** 0.50
- **factor_rank:** 6
- **hidden_dim:** 128
- **slow_hidden:** 8
- **coarse_window:** 10
- **state_reg_window W:** 5

### CLI flag schema (all fixes are OPT-IN to prevent silent enablement)

```
--use_v1_film_pipe               action='store_true', default=False  (C1 opt-out)
--lambda_film_bce                float, default=0.0                   (C2 weight; 0.05 enables)
--use_learned_link               action='store_true', default=False   (C4b enable)
--lambda_twcrps                  float, default=0.0                   (C4a weight; 0.05 enables)
--lambda_state                   float, default=0.0                   (C3 weight; 0.10 enables)
```

Plus inherited v1 flags (seed, output_dir, data_path, pca_artifact, lambda_vs, curriculum_schedule, etc.).

### Variant launch commands

```bash
COMMON="--seed 42 --epochs 60 --batch_size 32 --n_members 8 \
        --curriculum_schedule 0:5,10:15,25:30 --feedback_decay_end 30 \
        --lambda_vs 0.05 --state_reg_window 5 \
        --pca_artifact models/backfill/coarse_pca_233a.npz \
        --data_path data/vol_surface_with_ret.npz \
        --device cuda"

# v1.2-control: disable all fixes (re-baseline v1)
python train_233a_v1_2_twopath_factor_ar.py $COMMON --variant full \
    --output_dir models/backfill/233a_v1_2_control_25d_s42 \
    --use_v1_film_pipe                        # Opt back into broken pipe
    # (all lambdas default to 0; no learned link)

# v1.2-minimal: fix C1 + C2 + C3 only
python train_233a_v1_2_twopath_factor_ar.py $COMMON --variant full \
    --output_dir models/backfill/233a_v1_2_minimal_25d_s42 \
    --lambda_film_bce 0.05 --lambda_state 0.10

# v1.2-aux: fix C1 + C2 + C3 + C4a
python train_233a_v1_2_twopath_factor_ar.py $COMMON --variant full \
    --output_dir models/backfill/233a_v1_2_aux_25d_s42 \
    --lambda_film_bce 0.05 --lambda_state 0.10 --lambda_twcrps 0.05

# v1.2-link: fix C1 + C2 + C3 + C4b
python train_233a_v1_2_twopath_factor_ar.py $COMMON --variant full \
    --output_dir models/backfill/233a_v1_2_link_25d_s42 \
    --lambda_film_bce 0.05 --lambda_state 0.10 --use_learned_link

# v1.2-both: fix C1 + C2 + C3 + C4a + C4b
python train_233a_v1_2_twopath_factor_ar.py $COMMON --variant full \
    --output_dir models/backfill/233a_v1_2_both_25d_s42 \
    --lambda_film_bce 0.05 --lambda_state 0.10 --lambda_twcrps 0.05 --use_learned_link

# v1.2-noreg: fix C1 + C2 + C4a + C4b (no C3)
python train_233a_v1_2_twopath_factor_ar.py $COMMON --variant full \
    --output_dir models/backfill/233a_v1_2_noreg_25d_s42 \
    --lambda_film_bce 0.05 --lambda_twcrps 0.05 --use_learned_link
```

### Wall-time estimate

Per-variant ~15-20 min at H=30 under 3-way GPU parallel. Six variants sequential: ~100 min. With 3-way parallelism: ~35 min. Compute estimate assumes no training issues (smoke test gates this).

---

## 6. Files to Create / Modify

### New files

```
research/233a_twopath_v1_2/
  ├── design.md                             # This spec
  └── plan.md                                # writing-plans output (next stage)

experiments/backfill/block_ar/
  ├── train_233a_v1_2_twopath_factor_ar.py   # ~400 LoC: subclass + new modules + CLI
  ├── eval_233a_v1_2_ladder.sh               # orchestrator for 6 variants
  └── compare_233a_v1_2_variants.py          # extends compare_233a_variants.py

models/backfill/                             # created per-variant, empty dir skeletons
  ├── 233a_v1_2_control_25d_s42/
  ├── 233a_v1_2_minimal_25d_s42/
  ├── 233a_v1_2_aux_25d_s42/
  ├── 233a_v1_2_link_25d_s42/
  ├── 233a_v1_2_both_25d_s42/
  └── 233a_v1_2_noreg_25d_s42/
```

### Modified files

```
experiments/backfill/block_ar/
  ├── _rollout_220_utils.py          # +5 LoC: add dispatch for "233a_v1_2"
  └── evaluate_220b_multihorizon_path_suite.py  # +3 LoC: extend native_families + startswith
```

### Subclass structure for `train_233a_v1_2_twopath_factor_ar.py`

- `FiLMFromHSlow(nn.Module)` — reads `h_slow: (B, 8)` directly; same 6 outputs as v1 FiLM; same γ-identity init convention
- `LearnedLink(nn.Module)` — `g(v) = σ(Linear(cond))·tanh(v) + (1−σ(·))·sinh(v)`; zero-init gate
- `twcrps_pathwise_max(samples, future, threshold) → scalar` — threshold-weighted pathwise-max CRPS
- `class TwoPathFactorARv1_2(TwoPathFactorAR)` — overrides `__init__` (to swap FiLM when not `use_v1_film_pipe`, add `emission_link` when `use_learned_link`), `forward_full` (to apply link + collect `q_seq_film`)
- `compute_loss_v1_2` — adds L_film_jump_bce, L_twcrps, L_state to the loss dict
- `main()` — argparse + data loading + training loop; mirrors v1's `main()` with the 5 new flags
- `load_model(ckpt, device)` — reconstructs `TwoPathFactorARv1_2` from checkpoint payload

---

## 7. Evaluation + Attribution Logic

### Per-variant evaluation

Run `evaluate_220b_multihorizon_path_suite.py` on each variant's `best_model.pt` with identical settings to v1's 9-run ladder:

```bash
python evaluate_220b_multihorizon_path_suite.py \
    --model_type 233a_v1_2 \
    --checkpoint models/backfill/233a_v1_2_${variant}_25d_s42/best_model.pt \
    --force_native_anchor \
    --max_windows 192 --samples 48 \
    --output_json results/block_ar/233a_v1_2/${variant}_s42/suite.json \
    --output_md   results/block_ar/233a_v1_2/${variant}_s42/suite.md \
    --device cuda
```

### Metrics extracted (per suite.json)

| metric | JSON path | gate |
|---|---|---|
| n_pass | `summary.n_pass` | — |
| turb_calm_ratio | `conditionality.turb_calm_ratio` | > 1.15 |
| worstC_h30 | `coverage.worst_cell_per_horizon.30` | > 0.70 |
| max_jump_ks | `pathwise_jump_realism.pathwise_max_jump.ks_stat` | < 0.20 |
| ks_test_n_pass | `distributional_fidelity.ks_test.n_pass` | ≥ 15/25 |
| MR_ratio | `mean_reversion.gt_ratio` | ∈ [0.70, 1.30] |

### Attribution calls (in `compare_233a_v1_2_variants.py`)

```python
deltas = {
    "filmpipe_wiring_effect":     v1_2_minimal   - v1_2_control,
    "twcrps_aux_isolated_effect": v1_2_aux       - v1_2_minimal,
    "learned_link_isolated":      v1_2_link      - v1_2_minimal,
    "combined_emission_effect":   v1_2_both      - v1_2_minimal,
    "state_reg_effect":           v1_2_both      - v1_2_noreg,
    "vs_incumbent":               best_v1_2      - baseline_229a_newproxy,
    "vs_v1":                      best_v1_2      - v1_full_s42,
}
```

### Success targets (seed 42, native+anchor, new RV proxy)

| metric | v1 baseline | 229a baseline | v1.2 success target |
|---|---|---|---|
| n_pass | 2 | 3 | ≥ 4 |
| turb_calm | 1.010 | 1.025 | > 1.15 (pass gate) |
| worstC_h30 | 0.385 | 0.270 | > 0.50 (directional) |
| max_jump_ks | 0.735 | 0.940 | < 0.50 (directional) |
| ks_test/25 | 13 | 19 | ≥ 15 (pass gate) |
| MR_ratio | 0.794 | 1.318 | ∈ [0.70, 1.30] (pass gate) |

**Headline success:** v1.2-best passes ≥ 3 new suites (turb_calm + change_ks + MR) for a 5/7 score without regressing surface/cross_cell. Would be the first multi-day model to reach 5/7 since v3 harness.

**Stretch:** max_jump_ks < 0.20 (pass Bug-6 gate) — would dethrone historical best in multi-day AR.

### Post-training mechanism diagnostics

Re-run the 4 diagnostic scripts from the v1 investigation on v1.2 best variant:
- `diagnose_233a_film_collapse.py`: verify `p_jump_logit` std > 0.01 (no dead zone)
- `diagnose_233a_slow_state.py`: verify h_slow still discriminative AND post-rollout collapse reduced
- `diagnose_233a_ar_compounding.py`: verify lag-1 autocorr no longer ≤ −0.30 (oscillation resolved)
- `diagnose_233a_regime_breakdown.py`: verify regime sign inversion resolved (calm_wr < 1.0 OR turb_wr > 1.0)

These give MECHANISM confirmation, not just metric confirmation. Crucial for learning from either success or failure.

---

## 8. Stop / Falsification Criteria

Three stages with explicit kill conditions.

### Stage A — Pre-training smoke (1 epoch on `v1.2-both`, batch=8, n_members=4, H=5)

**Pass:**
- Completes without exception
- `film.logit.weight.grad.abs().sum() > 0` after epoch 0 backward (C2 routes gradient to FiLM)
- `emission_link.gate.weight.grad.abs().sum() > 0` (C4b connected to loss)
- No NaN in any parameter
- Val loss finite

**Kill:**
- Any exception / NaN → bug; fix before Stage B
- `film.logit.weight.grad == 0` → C2 fix didn't take; cross-check that `q_seq_film` is stacked and passed to `compute_loss_v1_2`
- `emission_link` grad == 0 → C4b disconnected from loss graph

### Stage B — Mid-training single-variant check (epoch 15 of `v1.2-both`)

**Pass:**
- `film.p_jump_logit` std > 0.01 on 20 val windows (primary C2 success signal)
- Val loss trending down from ep 10
- h_slow PC1 AUC on val regime > 0.65 (state encoding preserved)
- No NaN in per-step loss components

**Kill:**
- FiLM std < 0.01 → dead zone returned despite BCE wiring; need variance regularizer or different architectural fix
- Val divergent → loss term scale mismatch
- h_slow AUC collapsed → architectural change broke something upstream

### Stage C — Full 6-variant decision tree (all variants completed)

**Branch 1 (clean success):** `v1.2-both` ≥ 5/7 AND max_jump_ks < 0.50 → v1.2 wins. Declare AR viable with correct wiring. Plan v1.3: multi-seed + multi-factor validation.

**Branch 2 (partial):** `v1.2-both` ≥ 4/7 BUT max_jump_ks ≥ 0.50 → FiLM fixed, emission structural cap remains. Isolate via aux/link deltas. Publish 5/7; Bug 6 is next bottleneck.

**Branch 3 (failure):** all variants ≤ 3/7 → architectural pivot justified. Launch H3 (external scaffold) + joint-path flow matching design. Paradigm pivot has clean evidence basis.

**Branch 4 (control anomaly):** `v1.2-control` ≠ 2/7 → drift between v1 and v1.2-control. Stop, diagnose, ensure attribution logic is valid.

**Branch 5 (YAGNI minimum):** `v1.2-minimal` ≈ `v1.2-both` → emission changes redundant. Deploy `minimal`.

**Branch 6 (one fix dominates):** `v1.2-aux` or `v1.2-link` alone matches `v1.2-both` → prefer the simpler single-fix variant.

### Rollback triggers (global)

- `L_total` diverges (> 10× ep-0 value) at any epoch
- Final val loss > 10× ep-0 val loss
- Checkpoint corruption / unloadable
- Eval harness exception on any variant

---

## 9. Research Log + Artifact Protocol

### Artifacts to commit

**Training (force-add; models/ gitignored):**
- `models/backfill/233a_v1_2_{control,minimal,aux,link,both,noreg}_25d_s42/{best_model.pt, final_model.pt, training_log.json, training.log}`

**Evaluation (force-add; results/ gitignored):**
- `results/block_ar/233a_v1_2/{control,minimal,aux,link,both,noreg}_s42/{suite.json, suite.md, eval.log}`
- `results/block_ar/233a_v1_2/decision.md`
- `results/block_ar/233a_v1_2/_diagnostic_{film_collapse,slow_state,ar_compounding,regime_breakdown}.{json,md}`

**Code (normal git add):**
- `research/233a_twopath_v1_2/design.md` (this file)
- `research/233a_twopath_v1_2/plan.md` (writing-plans output)
- `experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py`
- `experiments/backfill/block_ar/eval_233a_v1_2_ladder.sh`
- `experiments/backfill/block_ar/compare_233a_v1_2_variants.py`
- Modifications to `_rollout_220_utils.py` and `evaluate_220b_multihorizon_path_suite.py`

### RESEARCH_LOG.md entry outline

Append via research-log skill idiom (`cat >> RESEARCH_LOG.md << 'EOF'`), dated via `$(date +%Y-%m-%d)`:

```
## YYYY-MM-DD: 233a-v1.2 — Targeted Bug-Fix Experiment
### Context
### Variant Grid Results (table)
### Attribution Deltas (6 rows from Section 7)
### Mechanism Confirmation (post-train diagnostic reruns)
### Decision (Branch X from Section 8)
### Artifacts (paths + commit range)
### Next Direction (v1.3 OR paradigm pivot OR production adoption)
---
```

Re-index after: `qmd update --collection research && qmd embed`

### MEMORY.md update

Rewrite CURRENT STATE section per Branch outcome. Add `rc23_233a_v1_2_outcome.md` topic file linked from "Active" section.

### Commit hygiene

- Per-task commits during implementation (20+ small commits per writing-plans idiom)
- Force-add for `models/` and `results/` artifacts
- Final commit tag: `chore(233a-v1.2): plan complete — Branch X decision, next Y`

---

## Appendix: Cross-References

- v1 design: `research/233a_twopath_v1/design.md`
- v1 plan: `research/233a_twopath_v1/plan.md`
- v1 model code: `experiments/backfill/block_ar/train_233a_twopath_factor_ar.py`
- Diagnostic scripts: `experiments/backfill/block_ar/diagnose_233a_{film_collapse, slow_state, ar_compounding, regime_breakdown}.py`
- Diagnostic results: `results/block_ar/233a/_diagnostic_*.{json, md}`
- RESEARCH_LOG entries: 2026-04-18 (v1 ladder) + 2026-04-18 (diagnostic synthesis) + 2026-04-18 (Bug 5 amendment) + 2026-04-18 (Bug 6 peak-and-recover refinement) + 2026-04-18 (Research Compass)
- Commits: `b09dd28` (v1 complete), `b732ee0..37b635f` (diagnostics), `6f4b568` (compass)
