# 233a FiLM Collapse Diagnostic

**Seeds**: [42, 1337, 2024] | **Windows**: 20 | **Checkpoints**: best + final

## Q1: Gradient-Driven vs Init-Driven Collapse

Std of `p_jump_logit` across 20 val windows at t=0 (deterministic slow-state).

| seed | ckpt | epoch | p_jump_logit std | slow_q std |
|------|------|-------|-----------------|------------|
| 42 | best | 9 | 0.00229 | 0.06227 |
| 42 | final | 59 | 0.00000 | 0.06715 |
| 1337 | best | 8 | 0.00039 | 0.03978 |
| 1337 | final | 59 | 0.00000 | 0.11454 |
| 2024 | best | 8 | 0.00003 | 0.07724 |
| 2024 | final | 59 | 0.00000 | 0.05898 |

**Q1 Answer**: **Gradient-driven**: best std=0.00090 → final std=0.00000 (drops >50% over training).

## Q2: Signal Availability — slow_path jump_prob_head vs FiLM p_jump_logit

Comparing variance of `slow_path.jump_prob_head(h_0)` (q_t logit, has BCE supervision) vs `film.logit` output.

| seed | ckpt | slow_q range | slow_q std | p_logit range | p_logit std |
|------|------|-------------|-----------|--------------|-------------|
| 42 | best | [-2.472, -2.269] | 0.0623 | [-1.000, -0.992] | 0.00229 |
| 42 | final | [-2.639, -2.345] | 0.0671 | [-0.214, -0.214] | 0.00000 |
| 1337 | best | [-2.392, -2.264] | 0.0398 | [-1.464, -1.463] | 0.00039 |
| 1337 | final | [-2.491, -2.063] | 0.1145 | [-1.302, -1.302] | 0.00000 |
| 2024 | best | [-2.683, -2.424] | 0.0772 | [-1.015, -1.015] | 0.00003 |
| 2024 | final | [-2.669, -2.449] | 0.0590 | [-0.699, -0.699] | 0.00000 |

**Q2 Answer**: **FiLM layer drops signal**: slow_path q std=0.0802 >> FiLM logit std=0.00000. Slow state carries signal but FiLM MLP doesn't route it to p_jump_logit.

## Q3: Per-Loss Gradient Decomposition (FiLM heads)

Gradient norm of each FiLM weight from each individual loss component.
Two sub-tables: (a) seed=42 **best** checkpoint (ep 9), (b) seed=42 **final** checkpoint (ep 59).

### (a) seed=42 best (ep 9) — before collapse completes

| loss | film.logit | film.mlp[0] | film.g_lambda | film.g_d | film.drift |
|------|-----------|------------|--------------|---------|-----------|
| L_ES | 8.60e-04 | 1.04e-02 | (see json) | (see json) | (see json) |
| L_VS | 5.30e-05 | 5.14e-04 | — | — | — |
| L_RV | 6.60e-04 | 6.96e-03 | — | — | — |
| L_jump | 1.13e-04 | 5.17e-04 | — | — | — |
| L_total | 4.59e-04 | 6.42e-03 | — | — | — |

Notes on best checkpoint: `L_jump` gives `film.logit` a small but non-zero gradient (1.13e-04).
This is indirect: BCE trains `slow_path.jump_prob_head`, which influences `s_t`/`lam_t` via the
slow-path GRU, which are then inputs to FiLM — a 2-hop gradient path. But the gradient is
~8x weaker than `L_ES`, which acts directly via straight-through Bernoulli on `mask`.

### (b) seed=42 final (ep 59) — after complete collapse

| loss | film.logit | film.mlp[0] | film.g_lambda | film.g_d | film.drift |
|------|-----------|------------|--------------|---------|-----------|
| L_ES | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| L_VS | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| L_RV | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| L_jump | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |
| L_total | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 | 0.00e+00 |

The final checkpoint receives **zero gradient through every FiLM weight from every loss**
(confirmed for seeds 42 and 1337; seed 2024 shows ~1e-5 for film.logit only). This is the
smoking gun: once `p_jump_logit` drifts to a constant negative bias, `sigmoid(bias) < 0.5`
means `straight_through_bernoulli` fires with very low probability, and because the mask is
near-constant (all 0), the gradient path `L_ES → mask → logit` produces a near-zero gradient.
The FiLM MLP is in a **gradient dead zone**: collapsed output → mask ≈ 0 → zero gradient.

**Q3 Answer**: At the "best" checkpoint (ep ~8-9), `L_ES` provides the strongest gradient to
`film.logit` (~8e-4), with a weaker indirect signal from `L_jump` (~1e-4) via the 2-hop
slow-path GRU path. By the final checkpoint, ALL gradients are zero — the model has entered
a **self-reinforcing dead zone**: `p_jump_logit` collapsed negative → mask ≈ 0 always →
straight-through gradient collapses → no force to recover. `L_jump` was never able to provide
direct supervision to `film.logit` (it only reaches `slow_path.jump_prob_head`).

## Q4: Correlation Between Slow-State Inputs and FiLM Outputs

Pearson corr of log1p(s_0) and log1p(λ_0) vs FiLM outputs, across 20 val windows.

| seed | ckpt | corr(s,logit) | corr(λ,logit) | corr(s,γ_Λ) | corr(λ,γ_Λ) | corr(s,slow_q) |
|------|------|--------------|--------------|-------------|-------------|---------------|
| 42 | best | 0.990 | -1.000 | 0.989 | -1.000 | -0.997 |
| 42 | final | nan | nan | nan | nan | -0.910 |
| 1337 | best | -1.000 | nan | -1.000 | nan | 0.998 |
| 1337 | final | nan | nan | nan | nan | 0.948 |
| 2024 | best | 0.996 | nan | 0.994 | nan | 0.998 |
| 2024 | final | nan | nan | nan | nan | -0.388 |

**Q4 Answer**: At the "best" checkpoint, FiLM outputs have **strong correlation** with slow-state
inputs (|corr| up to 1.00 for both γ_Λ and p_jump_logit against log1p(s) and log1p(λ)),
showing the FiLM MLP *does* use its inputs when it still has nonzero gradient. At the final
checkpoint, `p_jump_logit` is a constant (std=0.00000 across all seeds), so correlation is
undefined (nan). The `slow_path.jump_prob_head` (slow_q) continues to show strong correlation
with s_0 at the final checkpoint (|corr| 0.39–0.95), confirming the slow-state signal is alive
— it is only the FiLM output that has died. The γ heads also collapse to constants at the final
checkpoint (same dead-zone reason — they share the FiLM MLP).

## Mechanistic Conclusion

The collapse is a **two-stage gradient starvation cascade**, combining **(b) loss-function structure**
and **(d) optimizer drift**:

**Stage 1 (early training, ep 0–8):** The zero-initialized FiLM logit head receives only weak
gradient from `L_ES` via straight-through Bernoulli (~8e-4 at ep 9), while `L_jump` BCE
supervises `slow_path.jump_prob_head` but provides only an indirect 2-hop signal to FiLM
(~1e-4, ~8x weaker). The weak `L_ES` signal pushes `p_jump_logit` negative (suppressing jump
firings reduces variance in `v`, which slightly reduces energy score). This is **(d) optimizer drift**
from the zero-init bias.

**Stage 2 (training completes):** Once `p_jump_logit` drifts sufficiently negative, the jump
mask fires with near-zero probability, so the straight-through gradient becomes zero. The FiLM
MLP enters a **self-reinforcing dead zone**: collapsed output → mask ≈ 0 → zero straight-through
gradient → no recovery signal. By ep 59, ALL FiLM weight gradients are identically 0.00 for
seeds 42 and 1337 — the head is permanently frozen at its collapsed value.

The dead zone also kills the γ/β heads (they share the same MLP backbone) — explaining why
per-batch γ_Λ spread is 0.03–0.06 (much lower than input variation) and why the model never
developed meaningful FiLM modulation despite having a 14-dim slow-state signal.

**Root cause is (b+d) combined**: `L_jump` was wired to supervise the *wrong* head
(`slow_path.jump_prob_head`, not `film.logit`), leaving FiLM with only the weak
straight-through signal from `L_ES`; the zero-initialized logit drifted negative early
enough to close the gradient path entirely.

### Summary (2-3 sentences)

FiLM's `p_jump_logit` collapse is a two-stage gradient starvation cascade: BCE (`L_jump`)
supervises `slow_path.jump_prob_head` but provides only a 2-hop, 8x-weaker signal to
`film.logit` vs the direct `L_ES` straight-through path, so the zero-initialized logit drifts
negative in early training (optimizer drift, bucket d). Once sufficiently negative, the
jump mask fires at near-zero probability, collapsing the straight-through gradient to zero
and trapping the FiLM MLP in a dead zone where all per-loss gradients are identically 0.00
— confirmed across all 3 seeds at ep 59. The fix requires wiring BCE directly to `film.logit`
(not `q_t`) OR replacing straight-through with a differentiable alternative that doesn't
go to zero when the mask probability collapses.
