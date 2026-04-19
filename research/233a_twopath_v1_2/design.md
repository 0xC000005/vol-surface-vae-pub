# 233a-v1.2 Design Spec — Targeted Bug Fixes on v1

**Date:** 2026-04-18
**Parent experiment:** 233a-v1 (see `research/233a_twopath_v1/design.md`; results at commits b09dd28..fa857df)
**Diagnostic basis:** Four-agent mechanistic investigation, commits `b732ee0..37b635f` (2026-04-18)
**Objective:** Test whether the 6 bugs diagnosed on v1 are fixable within the AR paradigm, via 7-variant ablation grid (amended from 6 after review). Single seed (42). Architecture/loss exploration phase, not statistical-validation phase.

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

**C4a (twCRPS aux loss):** threshold-weighted CRPS on the pathwise-max-|Δx| functional. **Formula corrected per review** — the pairwise `term2` must exclude the K×K diagonal to give an unbiased variance estimate:
```python
gen_max = (samples[:, :, 1:] - samples[:, :, :-1]).abs().max(dim=2).values   # (B, K, D)
gt_max  = (future[:, 1:] - future[:, :-1]).abs().max(dim=1).values            # (B, D)
indicator = (gt_max > q90_train).float()                                      # (B, D) — tail weight

# term1: mean |gen - gt| across K samples, per cell
term1 = (gen_max - gt_max.unsqueeze(1)).abs().mean(dim=1)                     # (B, D)

# term2: pairwise |gen_i - gen_j| across K×K samples, EXCLUDING diagonal (i==j contributes 0 but inflates denominator)
pairwise = (gen_max.unsqueeze(1) - gen_max.unsqueeze(2)).abs()                # (B, K, K, D)
# Sum and divide by K*(K-1) pairs (not K*K), matching energy_score convention in train_212b
K = gen_max.shape[1]
term2 = pairwise.sum(dim=(1, 2)) / (2 * K * (K - 1))                          # (B, D)

# Threshold-weighted: only tail-exceeding cells contribute
L_twcrps = ((term1 - term2) * indicator).mean()                               # scalar
```
Note: this matches the off-diagonal energy-score convention already used in `train_212b_h1_minimal_direct_stochastic_delta.py:energy_score`.

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

Seven variants, each seed 42, each 60 epochs. Isolates the effect of each fix through carefully-designed knob toggling. **Amendment (post-review):** added `v1.2-minreg` to cleanly separate C3 (state reg) from C1+C2 in the attribution logic — without it, the `minimal - control` delta conflates all three.

| # | Variant | FiLM pipe (C1) | BCE→film.logit (C2) | twCRPS aux (C4a) | Learned link (C4b) | State reg (C3) |
|---|---|:-:|:-:|:-:|:-:|:-:|
| 1 | `v1.2-control` | v1 bottleneck | v1 wiring | off | off | off |
| 2 | `v1.2-minreg` | h_slow direct | on | off | off | **off** |
| 3 | `v1.2-minimal` | h_slow direct | on | off | off | on |
| 4 | `v1.2-aux` | h_slow direct | on | on | off | on |
| 5 | `v1.2-link` | h_slow direct | on | off | on | on |
| 6 | `v1.2-both` | h_slow direct | on | on | on | on |
| 7 | `v1.2-noreg` | h_slow direct | on | on | on | off |

### Attribution logic (revised)

- `minreg vs control`: effect of **C1 + C2 alone** (FiLM pipe fix + BCE wiring, NO state reg)
- `minimal vs minreg`: effect of **C3 (state reg) alone**, given C1+C2 already applied
- `aux vs minimal`: effect of **C4a (twCRPS aux)** in isolation
- `link vs minimal`: effect of **C4b (learned link)** in isolation
- `both vs minimal`: combined emission effect; is it additive, redundant, or superadditive?
- `both vs noreg`: is state reg (C3) load-bearing when emission fixes are also on?
- `best v1.2 variant vs 229a`: did we beat the incumbent?
- `best v1.2 variant vs v1-full_s42 (commit `fa857df`)`: did we close gates that v1 failed?

**One extra variant (~15 min compute) in exchange for a clean C3-vs-C1+C2 decomposition.**

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
         + λ_VS         · L_VS                     (v1-inherited)
         + λ_rv         · L_rv_mse                 (v1-inherited: slow-path RV aux)
         + λ_jump_slow  · L_slow_jump_bce          (v1-inherited: slow-path BCE on q_seq_slow)
         + λ_BCE        · L_film_jump_bce          (C2; new: BCE on q_seq_film)
         + λ_twcrps     · L_twcrps                 (C4a; new)
         + λ_state      · L_state                  (C3; new)
```

Per-variant λ (**amended post-review to include v1-inherited `λ_rv` and `λ_jump_slow`** which v1's `compute_loss` uses and any re-baseline must preserve):

| variant | λ_VS | λ_rv | λ_jump_slow | λ_BCE | λ_twcrps | λ_state |
|---|---|---|---|---|---|---|
| `v1.2-control` | 0.05 | 0.10 | 0.05 | 0.00 | 0.00 | 0.00 |
| `v1.2-minreg` | 0.05 | 0.10 | 0.05 | 0.05 | 0.00 | **0.00** |
| `v1.2-minimal` | 0.05 | 0.10 | 0.05 | 0.05 | 0.00 | 0.10 |
| `v1.2-aux` | 0.05 | 0.10 | 0.05 | 0.05 | 0.05 | 0.10 |
| `v1.2-link` | 0.05 | 0.10 | 0.05 | 0.05 | 0.00 | 0.10 |
| `v1.2-both` | 0.05 | 0.10 | 0.05 | 0.05 | 0.05 | 0.10 |
| `v1.2-noreg` | 0.05 | 0.10 | 0.05 | 0.05 | 0.05 | 0.00 |

Notes:
- **`λ_rv` and `λ_jump_slow`** are v1's existing loss weights (on slow-path's `rv_head` MSE against log-RV target and slow-path's own `jump_prob_head` BCE against jump target). They stay at their v1 values for ALL variants, including control, so that `v1.2-control` recovers v1's loss function exactly.
- `λ_BCE` is the NEW C2 term that wires BCE to FiLM's `p_jump_logit` (separately from slow-path's `q_seq_slow`).
- `v1.2-minimal` and `v1.2-link` have identical λ — they differ only in the `use_learned_link` architectural flag.
- `v1.2-both` and `v1.2-noreg` differ only in `λ_state`.
- `v1.2-control` and `v1.2-minreg` differ in `use_v1_film_pipe` (C1) AND in `λ_BCE` (0 vs 0.05). This isolates C1+C2 effect jointly.
- `v1.2-minreg` and `v1.2-minimal` differ only in `λ_state` (0 vs 0.10). This isolates C3 effect.
- L_ES, L_VS, L_rv_mse, L_slow_jump_bce unchanged from v1.

**Loss-scale kill condition (formalized):** at epochs 5 AND 15 of `v1.2-both` (Stage-B variant), compute `ratio_i = mean(λ_i · L_i) / mean(L_ES)` for each non-ES term on the val set. If `ratio_i > 3.0` for any term, halve `λ_i` and restart training. This prevents any single aux loss from dominating ES and creating optimization pathologies.

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

### CLI flag schema (all NEW fixes opt-in; all v1 INHERITED defaults preserved)

```
# NEW v1.2 flags (opt-in):
--use_v1_film_pipe               action='store_true', default=False   (C1 opt-out: True = v1 broken bottleneck)
--lambda_film_bce                float, default=0.0                    (C2 weight; 0.05 enables)
--use_learned_link               action='store_true', default=False    (C4b enable)
--lambda_twcrps                  float, default=0.0                    (C4a weight; 0.05 enables)
--lambda_state                   float, default=0.0                    (C3 weight; 0.10 enables)

# v1-INHERITED flags (defaults match v1's compute_loss defaults — DO NOT CHANGE per variant):
--lambda_vs                      float, default=0.05                   (v1 variogram weight)
--lambda_rv                      float, default=0.10                   (v1 slow-path RV MSE weight)
--lambda_jump                    float, default=0.05                   (v1 slow-path BCE weight, on q_seq_slow)
```

Plus inherited v1 flags (seed, output_dir, data_path, pca_artifact, curriculum_schedule, etc.).

**Gating:** `return_teacher_h` (used to run the `_run_teacher_branch` for state reg) is gated by `args.lambda_state > 0` in the training loop. Variants with `λ_state=0` skip teacher-branch compute (save ~1s/epoch).

### Variant launch commands

```bash
COMMON="--seed 42 --epochs 60 --batch_size 32 --n_members 8 \
        --curriculum_schedule 0:5,10:15,25:30 --feedback_decay_end 30 \
        --lambda_vs 0.05 --state_reg_window 5 \
        --pca_artifact models/backfill/coarse_pca_233a.npz \
        --data_path data/vol_surface_with_ret.npz \
        --device cuda"

# v1.2-control: all NEW fixes disabled (re-baseline v1)
python train_233a_v1_2_twopath_factor_ar.py $COMMON --variant full \
    --output_dir models/backfill/233a_v1_2_control_25d_s42 \
    --use_v1_film_pipe                        # Opt back into broken pipe
    # v1-inherited λ defaults apply; no learned link

# v1.2-minreg: C1 + C2 only (no C3, no C4) — isolates C1+C2 effect
python train_233a_v1_2_twopath_factor_ar.py $COMMON --variant full \
    --output_dir models/backfill/233a_v1_2_minreg_25d_s42 \
    --lambda_film_bce 0.05
    # no --lambda_state: C3 OFF

# v1.2-minimal: C1 + C2 + C3 only (no emission fixes)
python train_233a_v1_2_twopath_factor_ar.py $COMMON --variant full \
    --output_dir models/backfill/233a_v1_2_minimal_25d_s42 \
    --lambda_film_bce 0.05 --lambda_state 0.10

# v1.2-aux: C1 + C2 + C3 + C4a
python train_233a_v1_2_twopath_factor_ar.py $COMMON --variant full \
    --output_dir models/backfill/233a_v1_2_aux_25d_s42 \
    --lambda_film_bce 0.05 --lambda_state 0.10 --lambda_twcrps 0.05

# v1.2-link: C1 + C2 + C3 + C4b
python train_233a_v1_2_twopath_factor_ar.py $COMMON --variant full \
    --output_dir models/backfill/233a_v1_2_link_25d_s42 \
    --lambda_film_bce 0.05 --lambda_state 0.10 --use_learned_link

# v1.2-both: all fixes
python train_233a_v1_2_twopath_factor_ar.py $COMMON --variant full \
    --output_dir models/backfill/233a_v1_2_both_25d_s42 \
    --lambda_film_bce 0.05 --lambda_state 0.10 --lambda_twcrps 0.05 --use_learned_link

# v1.2-noreg: all fixes EXCEPT C3
python train_233a_v1_2_twopath_factor_ar.py $COMMON --variant full \
    --output_dir models/backfill/233a_v1_2_noreg_25d_s42 \
    --lambda_film_bce 0.05 --lambda_twcrps 0.05 --use_learned_link
```

### Wall-time estimate

Per-variant ~15-20 min at H=30 under 3-way GPU parallel. **Seven variants** sequential: ~110 min. With 3-way parallelism: ~40 min. Compute estimate assumes no training issues (smoke test gates this).

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

### Subclass structure for `train_233a_v1_2_twopath_factor_ar.py` (**revised per review**)

- `FiLMFromHSlow(nn.Module)`:
  - reads `h_slow: (B, 8)` directly; same 6 outputs as v1 FiLM; same γ-identity init convention
  - **Attribute names preserved** (`self.mlp`, `self.g_lambda`, `self.b_lambda`, `self.g_d`, `self.b_d`, `self.drift`, `self.logit`) so that existing `diagnose_233a_film_collapse.py` runs unchanged on v1.2 checkpoints without modification.
- `LearnedLink(nn.Module)`:
  - `g(v) = σ(gate(cond))·tanh(v) + (1−σ(·))·sinh(v)`; `self.gate = Linear(cond_dim, D)` zero-init (weight + bias)
  - **`cond` timing** (per review): uses PER-STEP `cond` tensor at the moment of emission (i.e., `cond` AFTER the current step's gru_cell update but BEFORE next step's advance). NOT initial-history cond. This lets α adapt to the evolving regime within a trajectory.
- `twcrps_pathwise_max(samples, future, threshold) → scalar` — threshold-weighted pathwise-max CRPS; **term2 uses K×(K−1) denominator** (off-diagonal pairs only) matching 212b's energy_score convention.
- `class TwoPathFactorARv1_2(TwoPathFactorAR)`:
  - Overrides `__init__`:
    - If `not use_v1_film_pipe`: replace inherited `self.film` (v1 `FiLM(D, k)`) with `FiLMFromHSlow(slow_hidden=8, D, k)`.
    - If `use_learned_link`: add `self.emission_link = LearnedLink(cond_dim=hidden_dim, D=self.D)`.
    - Else: `self.emission_link = None`.
  - Overrides `forward_full`:
    - **Branches on `use_v1_film_pipe`** for FiLM input: if False call `self.film(h_slow)`; if True call `self.film(s_t, lam_t)` (v1 signature). Both branches produce the same output dict.
    - **Branches on `use_learned_link`** for delta computation: if True do `delta = self.emission_link(v, cond) * local_scale`; if False do `delta = torch.sinh(v) * local_scale` (v1 path).
    - Collects `q_seq_film` (list of `film_out["p_jump_logit"]` per step) into the returned dict. The existing `q_seq` from v1 (slow-path's `jump_prob_head` output) is **renamed `q_seq_slow`** in the returned dict to avoid naming collision. Both are returned.
  - `_run_teacher_branch` unchanged from v1 (already produces detached teacher h_slow trajectory).
- `compute_loss_v1_2`:
  - Inherits all v1 loss terms (L_ES, L_VS, L_rv_mse on `rv_pred_seq`, L_slow_jump_bce on `q_seq_slow`) — DO NOT remove or rename.
  - Adds:
    - `L_film_jump_bce` = BCE on `q_seq_film` against the same jump_target_seq used by L_slow_jump_bce (same training signal, different consumer).
    - `L_twcrps` = `twcrps_pathwise_max(samples, future, q90_train)` if `λ_twcrps > 0`.
    - `L_state` = mean per-step MSE between `h_slow_free_seq[:W]` and `h_slow_teacher_seq[:W].detach()` if `λ_state > 0`.
  - Returns dict with all 6 (or 8) loss components for logging.
- `main()` argparse + training loop:
  - **Teacher-branch gating**: `return_teacher_h = (args.lambda_state > 0)`. Variants with `λ_state=0` skip teacher-branch compute (save ~1s/epoch).
  - **FiLM-pipe branching** visible in model-construction log line so operator can confirm.
  - **Model config saved in checkpoint payload** includes all 5 new flags + all v1 flags, so `load_model` reconstructs correctly.
- `load_model(ckpt, device)`: reads payload["args"], reconstructs `TwoPathFactorARv1_2` with all flags. Returns `(model, payload)`.

**Critical subclass constraint:** `linear_s`, `linear_lam` (the broken residual heads inside SlowPath) are NOT removed — they stay in the slow-path for backward compatibility with v1 diagnostic scripts (those scripts read `slow_path.linear_s`). But their OUTPUTS (`s_hybrid`, `lam_hybrid`) are **not consumed by FiLM in v1.2**. They still feed `scale_jump_head(lam_t)` for the jump-mixture term, consistent with v1 behavior (this is the correction to the minor Section 1 text error flagged in review — `linear_s`/`linear_lam` primary consumer is `scale_jump_head`, not "the aux heads").

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

### Attribution calls (in `compare_233a_v1_2_variants.py`, **revised for 7 variants**)

```python
deltas = {
    "C1_C2_alone_effect":         v1_2_minreg    - v1_2_control,    # FiLM pipe + BCE wiring, NO state reg
    "C3_state_reg_isolated":      v1_2_minimal   - v1_2_minreg,     # state reg alone (given C1+C2)
    "twcrps_aux_isolated":        v1_2_aux       - v1_2_minimal,    # twCRPS alone
    "learned_link_isolated":      v1_2_link      - v1_2_minimal,    # learned link alone
    "combined_emission_effect":   v1_2_both      - v1_2_minimal,    # C4a + C4b combined
    "state_reg_with_emission":    v1_2_both      - v1_2_noreg,      # C3 load-bearing test
    "vs_incumbent":               best_v1_2      - baseline_229a_newproxy,
    "vs_v1":                      best_v1_2      - v1_full_s42,
}
```

### Success targets (seed 42, native+anchor, new RV proxy) — **raised per review**

| metric | v1 baseline | 229a baseline | v1.2 success target |
|---|---|---|---|
| n_pass | 2 | 3 | **≥ 5** (raised from 4; match production bar) |
| turb_calm | 1.010 | 1.025 | > 1.15 (pass gate) |
| worstC_h30 | 0.385 | 0.270 | > 0.50 (directional) |
| max_jump_ks | 0.735 | 0.940 | < 0.50 (directional) |
| ks_test/25 | 13 | 19 | ≥ 15 (pass gate) |
| MR_ratio | 0.794 | 1.318 | ∈ [0.70, 1.30] (pass gate) |

**Seed-42 caveat (per adversarial review):** seed 42 was v1's best seed for jumpKS (0.735) vs seeds 1337/2024 at 0.898/0.838 — a 23% swing within the same variant. Any partial improvement on single-seed v1.2 must be treated as an **architectural signal requiring multi-seed replication** before claiming "AR paradigm viable." Branch-1 language in Section 8 reflects this.

**Headline success:** v1.2-best passes ≥ 3 new suites (turb_calm + change_ks + MR) for a 5/7 score without regressing surface/cross_cell. Would be the first multi-day model to reach 5/7 since v3 harness.

**Stretch:** max_jump_ks < 0.20 (pass Bug-6 gate) — would dethrone historical best in multi-day AR.

### Post-training mechanism diagnostics

Re-run the 4 diagnostic scripts from the v1 investigation on v1.2 best variant:
- `diagnose_233a_film_collapse.py`: verify `p_jump_logit` std > 0.01 (no dead zone)
- `diagnose_233a_slow_state.py`: verify h_slow still discriminative AND post-rollout collapse reduced
- `diagnose_233a_ar_compounding.py`: verify lag-1 autocorr no longer ≤ −0.30 (oscillation resolved)
- `diagnose_233a_regime_breakdown.py`: verify regime sign inversion resolved (calm_wr < 1.0 OR turb_wr > 1.0)

**NEW — `diagnose_233a_v1_2_emission_link.py` (for variants with C4b enabled):**
- Compute `α = σ(link_gate(cond))` on 200 val windows, stratified by calm/turb regime
- Verify `α` has non-trivial variance across conditions (std > 0.05)
- Verify **regime separation**: `mean(α | turb) − mean(α | calm)` should be non-zero (either sign OK; zero suggests α collapsed like v1's FiLM γ)
- If `α` collapses uniformly (std < 0.01), record this as the Bug-6-B failure mode and flag for v1.3.

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

### Stage B — Mid-training single-variant check (`v1.2-both`, checked at ep 5, ep 15, ep 20)

**Pass (all epochs):**
- No NaN in per-step loss components
- Val loss trending down (not divergent)

**Pass (ep 5 — loss-scale check):**
- For each aux term `L_i ∈ {L_VS, L_rv_mse, L_slow_jump_bce, L_film_jump_bce, L_twcrps, L_state}`:
  `mean(λ_i · L_i) / mean(L_ES) < 3.0` (no single term dominates ES)
  **Kill:** if any ratio > 3.0 → halve that `λ_i` and restart training.

**Pass (ep 15 — C2 success signal):**
- `film.p_jump_logit` std > 0.01 on 20 val windows (Bug 1 fixed)
- h_slow PC1 AUC on val regime > 0.65 (slow state preserved)
- **Per-regime FiLM logit separation** (NEW): `mean(p_jump_logit | turb) − mean(p_jump_logit | calm)` > 0.3 — confirms FiLM is doing regime-conditional work, not just producing non-zero variance.
  **Kill:** any of these fails → dead zone has returned in a new form; need variance regularizer or architectural backup.

**Pass (ep 20 — C4a overshoot guard, for variants with λ_twcrps > 0):**
- Per-regime `calm_wr ≤ 1.30` on val sample (computed as width(calm regime) / width(uncond)) — twCRPS should not inflate calm dispersion beyond v1's 1.21 failure level.
  **Kill:** `calm_wr > 1.30` → twCRPS is cheating for max_jump_ks at cost of calm overdispersion; halve `λ_twcrps` and restart.

**Pass (ep 20 — C4b α-collapse guard, for variants with `use_learned_link`):**
- `α = σ(link_gate(cond))` distribution on val windows: `std(α) > 0.05` AND `|mean(α | turb) − mean(α | calm)| > 0.02`.
  **Kill:** α collapsed to constant → learned link is not adapting; revert to sinh-only in next iteration.

### Stage C — Full 7-variant decision tree (all variants completed)

**Branch 1 (clean success):** `v1.2-both` ≥ **5/7** AND max_jump_ks < 0.50 → **architectural signal, not paradigm victory.** Multi-seed (3 seeds) replication required before claiming "AR paradigm viable." Plan v1.3: multi-seed + H=252 smoke test + multi-factor validation. Also run `diagnose_233a_v1_2_emission_link.py` to confirm α didn't collapse (v1 FiLM collapse lesson).

**Branch 2 (partial):** `v1.2-both` ≥ 4/7 BUT max_jump_ks ≥ 0.50 → FiLM fixed (C1+C2 signals confirmed), emission structural cap remains. Isolate C4a vs C4b via `aux`/`link` deltas. Publish 4/7; Bug 6 is next bottleneck. Parallel H3 scaffold design accelerates alternative-paradigm evaluation.

**Branch 3 (failure):** all variants ≤ 3/7 → architectural pivot justified with clean evidence. Launch H3 (external scaffold) + joint-path flow matching design. 229a remains production incumbent.

**Branch 4 (control anomaly):** `v1.2-control` does NOT match v1-full_s42's **per-suite pass-set** (v1-full_s42 passed exactly {surface_validity, cross_cell_correlation}). Checking n_pass scalar alone is insufficient — a variant could score 2/7 via passing a different pair of suites and still represent a drift. **Specific gate:** v1.2-control.passed_suites must == v1-full_s42.passed_suites. Any mismatch → stop, diagnose drift in v1.2 codebase before trusting other variant deltas.

**Branch 5 (YAGNI minimum):** `v1.2-minimal` ≈ `v1.2-both` (within 1 n_pass) → emission changes redundant. Deploy `minimal`.

**Branch 6 (one fix dominates):** `v1.2-aux` or `v1.2-link` alone matches `v1.2-both` → prefer the simpler single-fix variant.

**Branch 7 (C3 not load-bearing):** `v1.2-minreg` ≈ `v1.2-minimal` on pass metrics → state reg adds no value given C1+C2 pipe fix. Drop C3 from production recipe.

### Long-horizon readiness (Branch-1 follow-up requirement)

Before declaring v1.3-ready, **any promoted v1.2 variant must pass a 252-day smoke test**: run `sample_batched` for n_steps=252 on 20 val windows, verify (a) no NaN, (b) final-step IV levels stay in [0.01, 1.0], (c) max-delta distribution at h=200-252 doesn't diverge from h=25-30 distribution. Formal H=252 eval (`test_long_horizon.py` style) is v1.3 scope but the smoke check is v1.2's exit gate.

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

---

## Appendix B: Review Amendments (2026-04-18)

Five parallel review agents audited the v1.0 draft of this spec. The inline amendments below incorporate their findings. Where a reviewer flagged a long-term strategic concern (e.g., H=252 integration, multi-factor I/O boundary) that is out of scope for v1.2, the concern is noted but deferred to v1.3.

### Applied to spec

| # | Reviewer | Change | Section |
|---|---|---|---|
| A | Principle | Added `v1.2-minreg` 7th variant (C1+C2, no C3) to cleanly isolate C3 effect | §2, §4, §5, §7 |
| B | Consistency | Added `λ_rv=0.10, λ_jump_slow=0.05` (v1-inherited) to loss-weight table and CLI schema; control must match v1 loss exactly | §4, §5 |
| C | Consistency | Renamed v1's `q_seq` to `q_seq_slow`; new C2 output is `q_seq_film`; both returned from `forward_full` | §3, §6 |
| D | Consistency | `return_teacher_h` gated on `args.lambda_state > 0` in training loop | §5, §6 |
| E | Consistency | `forward_full` explicitly branches on `use_v1_film_pipe` and `use_learned_link` flags; documented in §6 | §6 |
| F | Adversarial | Section 7 attribution labels revised to match new variant grid (minreg vs control isolates C1+C2 alone) | §7 |
| G | Adversarial | Branch 4 now checks per-suite pass-set match (Surface + CrossCell), not n_pass scalar | §8 |
| H | Adversarial | Added `diagnose_233a_v1_2_emission_link.py` for α collapse detection | §7 |
| I | Principle | Loss-scale kill condition formalized: ratio = `mean(λ_i·L_i)/mean(L_ES)` at ep 5 and ep 15, kill + halve if > 3.0 | §4, §8 |
| K | Long-term | Branch 1 n_pass gate raised from 4 to 5 (match production target) | §7, §8 |
| N | Adversarial | Added Stage-B kill gates at ep 20: `calm_wr ≤ 1.30` (twCRPS overshoot), α separation `> 0.02` (C4b collapse), per-regime FiLM logit separation `> 0.3` (C2 effectiveness) | §8 |
| O | Adversarial | `twcrps_pathwise_max` term2 denominator corrected to K×(K−1) off-diagonal (matches 212b energy_score convention) | §1 |
| P | Adversarial | Branch 1 reframed as "architectural signal only, multi-seed required" — single-seed success does NOT claim paradigm viability | §8 |
| R | Consistency | Corrected minor text error: `linear_s`/`linear_lam` primary consumer is `scale_jump_head`, not the aux heads | §6 |
| — | Long-term | H=252 smoke test added to Branch 1 exit criteria (sample_batched n_steps=252 on 20 val windows; formal H=252 eval remains v1.3 scope) | §8 |

### Deferred to v1.3 (acknowledged, not applied)

| # | Reviewer | Concern | Rationale for deferral |
|---|---|---|---|
| L | Long-term | Parallelize H3 (external-scaffold) alongside v1.2 instead of as Branch-3 contingency | H3 is a separate architecture with its own design spec; parallel launch is a scheduling decision, not a v1.2 spec decision. The Research Compass (`6f4b568`) already tracks H3 independently. |
| Q | Generalizability | Multi-factor I/O blockers (unconditional `reshape(B,T,5,5)`, hardwired `(B,K,N,5,5)` output, `[1e-4, 1-1e-4]` clamp) | These are the concrete v1.3 scope items for extending to `multi_factor_data.npz`. Fixing them in v1.2 would confound attribution vs v1. |
| Seed variance | Adversarial | Seed 42 is v1's BEST seed for jumpKS; partial v1.2 improvement may be outlier | Caveat is now flagged in §7 success-targets table. v1.3 multi-seed replication is the cure. |

### Verdict after amendments

All 5 reviewers raised LIKELY-FIXABLE concerns. Amendments A-R close them. Deferred items L, Q are legitimate scope boundaries for v1.3. Spec is cleared for implementation.

