#!/usr/bin/env python
"""
233a-v1.2: Targeted bug fixes on 233a-v1.

Subclasses TwoPathFactorAR from v1. Design spec: research/233a_twopath_v1_2/design.md

7 variants via CLI flags:
  control: use_v1_film_pipe (all new fixes OFF) — re-baseline v1
  minreg:  C1 + C2 only (no C3, no C4)
  minimal: C1 + C2 + C3 (no emission fixes)
  aux:     C1 + C2 + C3 + C4a (twCRPS)
  link:    C1 + C2 + C3 + C4b (learned link)
  both:    all fixes
  noreg:   all fixes EXCEPT C3
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Inherit v1 infrastructure
from experiments.backfill.block_ar.train_233a_twopath_factor_ar import (
    TwoPathFactorAR,
    CoarseFeatures,
    SlowPath,
    ScaleJumpHead,
    straight_through_bernoulli,
    compute_loss as compute_loss_v1,
    FiLM as FiLMv1,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    energy_score,
)
from experiments.backfill.block_ar.train_212s_h1_minimal_direct_stochastic_delta_es_vs import (
    variogram_score,
)

VALID_VARIANT_NAMES = {"control", "minreg", "minimal", "aux", "link", "both", "noreg"}


# --- stubs filled in by later tasks ---
class FiLMFromHSlow(nn.Module):
    """
    C1 fix: FiLM reads h_slow directly (8-dim, discriminative) instead of
    the broken (s_hybrid, lam_hybrid) 2-scalar bottleneck.

    Preserves attribute names from v1 FiLM (self.mlp, self.g_lambda, etc.)
    so diagnostic scripts can target the same paths — but self.mlp[0]
    input dim is 8 (h_slow) instead of 2 (log1p(s, lam)). Any strict
    state_dict load from v1 → v1.2 will fail; must load via load_model dispatch.

    Same γ-identity init convention as v1 FiLM:
      γ heads output 0 at init; forward adds 1 → γ = 1 at init
      β, drift, logit heads output 0 at init
    """

    def __init__(self, slow_hidden: int, D: int, k: int, hidden: int = 32):
        super().__init__()
        self.D, self.k = D, k
        self.mlp = nn.Sequential(
            nn.Linear(slow_hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Output heads — zero-init weights AND biases; γ = 1 + head in forward()
        self.g_lambda = nn.Linear(hidden, k)
        self.b_lambda = nn.Linear(hidden, k)
        self.g_d = nn.Linear(hidden, D)
        self.b_d = nn.Linear(hidden, D)
        self.drift = nn.Linear(hidden, D)
        self.logit = nn.Linear(hidden, 1)
        for head in (self.g_lambda, self.b_lambda, self.g_d, self.b_d, self.drift, self.logit):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, h_slow: torch.Tensor) -> dict:
        """h_slow: (B, slow_hidden) -> dict with 6 modulation tensors."""
        h = self.mlp(h_slow)
        return dict(
            gamma_lambda=1.0 + self.g_lambda(h),       # (B, k), identity at init
            beta_lambda=self.b_lambda(h),
            gamma_d=1.0 + self.g_d(h),                  # (B, D), identity at init
            beta_d=self.b_d(h),
            drift_bias=self.drift(h),
            p_jump_logit=self.logit(h).squeeze(-1),    # (B,)
        )


class LearnedLink(nn.Module):
    """
    C4b fix: learned convex combination of tanh and sinh emission links.
       g(v) = α · tanh(v) + (1 - α) · sinh(v)
       α = σ(gate(cond))

    Zero-init gate (weight + bias) → σ(0) = 0.5 at init, so
    g(v) = 0.5·tanh(v) + 0.5·sinh(v) at init (halfway between bounded + unbounded).

    Data chooses per-condition whether bounded (α→1, calm) or unbounded (α→0, turb).

    Note: `cond` must be the SAME cond tensor that factor_head consumes at emission
    time (NOT post-gru_cell cond). See Section 6 of design.md.
    """
    def __init__(self, cond_dim: int, D: int):
        super().__init__()
        self.gate = nn.Linear(cond_dim, D)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, v: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        v:    (BK, D) — pre-link innovation
        cond: (BK, cond_dim) — fast-path condition AT EMISSION TIME
        Returns: (BK, D) — modulated innovation
        """
        alpha = torch.sigmoid(self.gate(cond))   # (BK, D)
        return alpha * torch.tanh(v) + (1 - alpha) * torch.sinh(v)


def twcrps_pathwise_max(
    samples: torch.Tensor,    # (B, K, N, D)
    future: torch.Tensor,     # (B, N, D)
    threshold: torch.Tensor,  # scalar q90_train
) -> torch.Tensor:
    """
    C4a fix: threshold-weighted CRPS on the per-cell max |Δx| functional.
    Pairwise denominator is 2*K*(K-1) for off-diagonal normalization
    (matches energy_score convention in train_212b).
    """
    gen_delta = (samples[:, :, 1:] - samples[:, :, :-1]).abs()    # (B, K, N-1, D)
    gt_delta = (future[:, 1:] - future[:, :-1]).abs()              # (B, N-1, D)
    gen_max = gen_delta.max(dim=2).values                          # (B, K, D)
    gt_max = gt_delta.max(dim=1).values                             # (B, D)

    # term1: mean |gen - gt| across K samples
    term1 = (gen_max - gt_max.unsqueeze(1)).abs().mean(dim=1)      # (B, D)

    # term2: pairwise off-diagonal |gen_i - gen_j|
    K = gen_max.shape[1]
    if K < 2:
        term2 = torch.zeros_like(term1)
    else:
        pairwise = (gen_max.unsqueeze(1) - gen_max.unsqueeze(2)).abs()   # (B, K, K, D)
        # Sum over K*K pairs (including zero diagonal) and divide by 2*K*(K-1)
        term2 = pairwise.sum(dim=(1, 2)) / (2 * K * (K - 1))             # (B, D)

    # Threshold weighting: only tail-exceeding cells contribute
    indicator = (gt_max > threshold).float()                       # (B, D)
    return ((term1 - term2) * indicator).mean()                    # scalar


class TwoPathFactorARv1_2(TwoPathFactorAR):
    """
    233a-v1.2 model. Subclasses v1's TwoPathFactorAR.

    Variant behavior controlled by two architectural flags:
      use_v1_film_pipe: False (default) → FiLM reads h_slow directly (Bugs 2+5 fix)
                         True            → v1's broken (s_hybrid, lam_hybrid) pipe
      use_learned_link: False (default) → sinh(v) · local_scale (v1 emission)
                         True            → α·tanh + (1-α)·sinh (Bug 6-B fix)

    Loss-level flags (applied in compute_loss_v1_2):
      lambda_film_bce > 0 → C2 BCE on film.logit
      lambda_twcrps > 0   → C4a threshold-weighted aux
      lambda_state > 0    → C3 state consistency reg
    """

    def __init__(
        self,
        variant: str = "full",
        use_v1_film_pipe: bool = False,
        use_learned_link: bool = False,
        pca_artifact_path: str = "models/backfill/coarse_pca_233a.npz",
        **base_kwargs,
    ):
        super().__init__(variant=variant, pca_artifact_path=pca_artifact_path, **base_kwargs)
        self.use_v1_film_pipe = use_v1_film_pipe
        self.use_learned_link = use_learned_link

        # C1 fix: swap FiLM module if not keeping v1 pipe, for variant="full"
        if variant == "full" and not use_v1_film_pipe:
            # Replace the v1 FiLM(D, k) (reads log1p(s), log1p(lam))
            # with FiLMFromHSlow(slow_hidden, D, k) (reads h_slow)
            self.film = FiLMFromHSlow(
                slow_hidden=self.slow_path.slow_hidden,
                D=self.D,
                k=self.k,
                hidden=32,
            )

        # C4b fix: add learned emission link if enabled
        if use_learned_link:
            self.emission_link = LearnedLink(cond_dim=self.hidden_dim, D=self.D)
        else:
            self.emission_link = None

    def forward_full(
        self,
        history: torch.Tensor,
        future: Optional[torch.Tensor] = None,
        n_members: int = 8,
        n_steps: int = 30,
        p_gt_feedback: float = 0.0,
        return_teacher_h: bool = False,
        state_reg_window: int = 5,
        **kwargs,
    ) -> dict:
        """v1.2 forward_full. Inherits v1's structure with 3 branches for C1/C4b/C2."""
        # Reshape guard (same as v1)
        if history.dim() == 4:
            B, T, H5, W5 = history.shape
            D = H5 * W5
            hist_flat = history.reshape(B, T, D)
        else:
            B, T, D = history.shape
            hist_flat = history
        if future is not None and future.dim() == 4:
            future = future.reshape(future.shape[0], future.shape[1], -1)
        device = history.device
        K = n_members
        BK = B * K

        # Init slow state (B-shape)
        state = self.init_slow_state(hist_flat)
        buffer_B = list(state["buffer"])
        h_slow = state["h_slow"]
        s_ewma, lam_hawkes = state["s_ewma"], state["lam_hawkes"]
        s_t, lam_t = state["s"], state["lam"]
        delta_t_last = state["delta_t_last_jump"]

        # Init fast state (BK-shape, 227a convention)
        hist_4d = hist_flat.reshape(B, T, 5, 5)
        cond_B_0, local_scale_B = self.encode_history(hist_4d)
        scale_anchor_B = local_scale_B.clone()

        cond = cond_B_0.unsqueeze(1).expand(B, K, -1).reshape(BK, -1)
        local_scale = local_scale_B.unsqueeze(1).expand(B, K, -1).reshape(BK, -1)
        prev = hist_flat[:, -1].unsqueeze(1).expand(B, K, -1).reshape(BK, D)
        scale_anchor_bk = scale_anchor_B.unsqueeze(1).expand(B, K, -1).reshape(BK, -1)
        x_prev_B = hist_flat[:, -1]

        # AR(1) factor noise
        z_f = torch.randn(BK, self.factor_rank, device=device)
        rho_sq_comp = math.sqrt(1.0 - self.rho ** 2)

        samples = torch.empty(B, K, n_steps, D, device=device)
        h_slow_free_seq, h_slow_teacher_seq = [], []
        s_seq, lam_seq = [], []
        q_seq_slow, q_seq_film = [], []   # CHANGED: slow vs film separation
        rv_pred_seq, mean_sq_dx_seq = [], []

        for t in range(n_steps):
            # === C1 fix: FiLM input pipe branch ===
            if self.use_v1_film_pipe:
                # v1 path: FiLM reads (s_t, lam_t) scalars
                film_out = self.film(s_t, lam_t)
            else:
                # v1.2 path: FiLM reads h_slow directly
                film_out = self.film(h_slow)

            # Broadcast FiLM outputs B → BK
            gamma_L_B = film_out["gamma_lambda"]
            beta_L_B = film_out["beta_lambda"]
            gamma_D_B = film_out["gamma_d"]
            beta_D_B = film_out["beta_d"]
            drift_B = film_out["drift_bias"]
            logit_B = film_out["p_jump_logit"]

            gamma_L_bk = gamma_L_B.unsqueeze(1).expand(B, K, self.k).reshape(BK, self.k)
            beta_L_bk = beta_L_B.unsqueeze(1).expand(B, K, self.k).reshape(BK, self.k)
            gamma_D_bk = gamma_D_B.unsqueeze(1).expand(B, K, D).reshape(BK, D)
            beta_D_bk = beta_D_B.unsqueeze(1).expand(B, K, D).reshape(BK, D)
            drift_bk = drift_B.unsqueeze(1).expand(B, K, D).reshape(BK, D)
            logit_bk = logit_B.unsqueeze(1).expand(B, K).reshape(BK)

            # === C2-ready: collect film.logit output for BCE loss ===
            q_seq_film.append(logit_B)   # (B,) per step

            # Fast emission (227a pattern, cond at emission time)
            if t > 0:
                z_f = self.rho * z_f + rho_sq_comp * torch.randn_like(z_f)
            z_i = torch.randn(BK, D, device=device)
            pos = self.pos_embed(t, BK, device)

            factor_in = torch.cat([prev, cond, z_f, pos], dim=-1)
            f_scores = self.factor_head(factor_in)
            idio_in = torch.cat([prev, cond, z_i, pos], dim=-1)
            i_resid = self.idio_head(idio_in)

            Lambda_base = self.get_lambda(cond)
            Lambda_mod = Lambda_base * gamma_L_bk.unsqueeze(-2) + beta_L_bk.unsqueeze(-2)
            D_base = self.get_d(cond)
            D_mod = D_base * gamma_D_bk + beta_D_bk

            v = torch.einsum("bdr,br->bd", Lambda_mod, f_scores) + D_mod * i_resid + drift_bk

            # Jump mixture (unchanged)
            lam_t_bk = lam_t.unsqueeze(1).expand(B, K).reshape(BK)
            mask = straight_through_bernoulli(logit_bk, shape=(BK, 1))
            eps_extra = torch.randn(BK, D, device=device)
            s_jump_bk = self.scale_jump_head(lam_t_bk)
            v = v + mask * s_jump_bk.unsqueeze(-1) * eps_extra

            # === C4b fix: emission link branch ===
            if self.use_learned_link:
                g_v = self.emission_link(v, cond)   # per-cell α·tanh + (1-α)·sinh; cond at emission time
            else:
                g_v = torch.sinh(v)                  # v1 unchanged

            delta = g_v * local_scale
            next_iv = (prev + delta).clamp(1e-4, 1.0 - 1e-4)
            samples[:, :, t] = next_iv.view(B, K, D)

            # Feedback selection (unchanged from v1)
            if self.training and future is not None and torch.rand(1).item() < p_gt_feedback:
                x_feedback_B = future[:, t]
                x_feedback_bk = x_feedback_B.unsqueeze(1).expand(B, K, -1).reshape(BK, D)
            else:
                x_feedback_bk = next_iv
                x_feedback_B = next_iv.view(B, K, D).mean(dim=1)

            # Fast-path state update (BK)
            feat, local_scale = self._step_features(prev, x_feedback_bk, local_scale)
            cond = self.gru_cell(feat, cond)
            prev = x_feedback_bk

            if self.use_scale_anchor:
                log_s = ((1.0 - self.scale_anchor_alpha)
                         * torch.log(local_scale.clamp_min(self.scale_floor))
                         + self.scale_anchor_alpha
                         * torch.log(scale_anchor_bk.clamp_min(self.scale_floor)))
                local_scale = torch.exp(log_s)

            # Slow-path state update (B) — unchanged from v1
            dx_B = x_feedback_B - x_prev_B
            mean_sq_dx = (dx_B ** 2).mean(dim=-1)
            j_t_B = (dx_B.norm(dim=-1) > self.q90_train).float()
            buffer_B.append(x_feedback_B); buffer_B = buffer_B[-30:]
            coarse_t = self.coarse(x_feedback_B, buffer_B)

            sp_out = self.slow_path.step(
                coarse_t, h_slow, s_ewma, lam_hawkes, delta_t_last,
                mean_sq_dx, j_t_B,
            )
            h_slow = sp_out["h_t"]
            s_ewma = sp_out["s_ewma_t"]
            lam_hawkes = sp_out["lam_hawkes_t"]
            s_t, lam_t = sp_out["s_t"], sp_out["lam_t"]

            rv_pred_seq.append(sp_out["rv_pred"])
            q_seq_slow.append(sp_out["q_t"])    # RENAMED from v1's q_seq
            s_seq.append(s_t); lam_seq.append(lam_t)
            mean_sq_dx_seq.append(mean_sq_dx)
            h_slow_free_seq.append(h_slow)

            delta_t_last = torch.where(j_t_B.bool(), torch.zeros_like(delta_t_last), delta_t_last + 1.0)
            x_prev_B = x_feedback_B

        if return_teacher_h:
            h_slow_teacher_seq = self._run_teacher_branch(hist_flat, future, state_reg_window)

        return dict(
            samples=samples,
            h_slow_free_seq=h_slow_free_seq,
            h_slow_teacher_seq=h_slow_teacher_seq,
            s_seq=s_seq, lam_seq=lam_seq,
            q_seq_slow=q_seq_slow,             # v1's q_seq, renamed
            q_seq_film=q_seq_film,             # NEW: FiLM's p_jump_logit per step
            rv_pred_seq=rv_pred_seq,
            mean_sq_dx_seq=mean_sq_dx_seq,
        )


def compute_loss_v1_2(*args, **kwargs):
    """Extends v1's compute_loss with L_film_jump_bce, L_twcrps, L_state."""
    raise NotImplementedError


def load_model(checkpoint_path, device):
    """Reconstruct TwoPathFactorARv1_2 from checkpoint."""
    raise NotImplementedError


def main():
    raise NotImplementedError


def _sanity_check_film_from_hslow():
    film = FiLMFromHSlow(slow_hidden=8, D=25, k=6, hidden=32)
    h_slow = torch.randn(3, 8)
    out = film(h_slow)
    # At init, γ heads = 1.0 exactly (because zero weights + zero bias + add 1.0)
    assert torch.allclose(out["gamma_lambda"], torch.ones(3, 6)), "γ_Λ not 1.0 at init"
    assert torch.allclose(out["gamma_d"], torch.ones(3, 25)), "γ_D not 1.0 at init"
    assert torch.allclose(out["beta_lambda"], torch.zeros(3, 6))
    assert torch.allclose(out["drift_bias"], torch.zeros(3, 25))
    assert torch.allclose(out["p_jump_logit"], torch.zeros(3))
    print("FiLMFromHSlow sanity check PASS")


def _sanity_check_learned_link():
    link = LearnedLink(cond_dim=128, D=25)
    v = torch.randn(8, 25)
    cond = torch.randn(8, 128)
    # At init, α = σ(0) = 0.5 exactly
    alpha = torch.sigmoid(link.gate(cond))
    assert torch.allclose(alpha, torch.full_like(alpha, 0.5)), "α != 0.5 at init"
    g_v = link(v, cond)
    expected = 0.5 * torch.tanh(v) + 0.5 * torch.sinh(v)
    assert torch.allclose(g_v, expected, atol=1e-6), "link output mismatch"
    # Gradient check
    loss = g_v.sum()
    loss.backward()
    assert link.gate.weight.grad.abs().sum() > 0, "gate.weight has no gradient"
    print("LearnedLink sanity check PASS")


def _sanity_check_forward_full_v1_2():
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        np.savez(tmp.name,
            pca_mean_components=np.random.randn(4, 25).astype(np.float32),
            pca_mean_mean=np.random.randn(25).astype(np.float32),
            pca_disp_components=np.random.randn(4, 25).astype(np.float32),
            pca_disp_mean=np.random.randn(25).astype(np.float32),
            pca_rsc_components=np.random.randn(4, 25).astype(np.float32),
            pca_rsc_mean=np.random.randn(25).astype(np.float32),
            q90_train=np.array(0.1, dtype=np.float32),
            coarse_window=np.array(10, dtype=np.int32))
        path = tmp.name

    kwargs = dict(hidden_dim=128, factor_rank=6, ewma_alpha=0.20, scale_floor=1e-4, n_cells=25)
    history = torch.rand(2, 30, 25) * 0.3 + 0.1
    future = torch.rand(2, 10, 25) * 0.3 + 0.1

    for v1_pipe in (False, True):
        for link in (False, True):
            m = TwoPathFactorARv1_2(
                variant="full",
                use_v1_film_pipe=v1_pipe,
                use_learned_link=link,
                pca_artifact_path=path,
                **kwargs,
            ).eval()
            with torch.no_grad():
                out = m.forward_full(history, future, n_members=4, n_steps=5,
                                     p_gt_feedback=0.5, return_teacher_h=False)
            assert out["samples"].shape == (2, 4, 5, 25)
            assert len(out["q_seq_film"]) == 5
            assert out["q_seq_film"][0].shape == (2,)    # (B,) per step
            assert len(out["q_seq_slow"]) == 5
            assert out["h_slow_teacher_seq"] == []       # not requested

    os.unlink(path)
    print("v1_2 forward_full PASS (all 4 flag combos)")


def _sanity_check_v1_2_constructor():
    import tempfile, os
    # Create minimal fake PCA artifact
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        np.savez(tmp.name,
            pca_mean_components=np.random.randn(4, 25).astype(np.float32),
            pca_mean_mean=np.random.randn(25).astype(np.float32),
            pca_disp_components=np.random.randn(4, 25).astype(np.float32),
            pca_disp_mean=np.random.randn(25).astype(np.float32),
            pca_rsc_components=np.random.randn(4, 25).astype(np.float32),
            pca_rsc_mean=np.random.randn(25).astype(np.float32),
            q90_train=np.array(0.1, dtype=np.float32),
            coarse_window=np.array(10, dtype=np.int32))
        path = tmp.name

    kwargs = dict(hidden_dim=128, factor_rank=6, ewma_alpha=0.20, scale_floor=1e-4, n_cells=25)

    # 7 variant configs: (use_v1_film_pipe, use_learned_link)
    configs = {
        "control": (True,  False),   # v1 pipe, no link
        "minreg":  (False, False),   # h_slow pipe, no link
        "minimal": (False, False),
        "aux":     (False, False),
        "link":    (False, True),
        "both":    (False, True),
        "noreg":   (False, True),
    }
    for name, (v1_pipe, link) in configs.items():
        m = TwoPathFactorARv1_2(
            variant="full",
            use_v1_film_pipe=v1_pipe,
            use_learned_link=link,
            pca_artifact_path=path,
            **kwargs,
        )
        has_hslow_film = isinstance(m.film, FiLMFromHSlow)
        has_link = m.emission_link is not None
        print(f"{name}: FiLMFromHSlow={has_hslow_film}, LearnedLink={has_link}, "
              f"params={sum(p.numel() for p in m.parameters())/1e3:.1f}k")
        # Invariants
        if v1_pipe:
            assert not has_hslow_film, f"{name}: should keep v1 FiLM"
        else:
            assert has_hslow_film, f"{name}: should use FiLMFromHSlow"
        if link:
            assert has_link, f"{name}: should have emission_link"
        else:
            assert not has_link, f"{name}: should not have emission_link"
    os.unlink(path)
    print("v1_2 constructor sanity check PASS")


def _sanity_check_twcrps():
    # Shape check
    B, K, N, D = 4, 8, 30, 25
    samples = torch.randn(B, K, N, D)
    future = torch.randn(B, N, D)
    threshold = torch.tensor(0.1)
    result = twcrps_pathwise_max(samples, future, threshold)
    assert result.dim() == 0, f"expected scalar, got shape {result.shape}"
    assert torch.isfinite(result).item(), "L_twcrps not finite"

    # Numerical check: if samples == future (perfect match), term1 = 0 → L_twcrps = -term2 ≤ 0
    samples2 = future.unsqueeze(1).expand(B, K, N, D).contiguous()
    r2 = twcrps_pathwise_max(samples2, future, threshold)
    # With perfect match, all gen_max equal gt_max → term1=0, term2=0, L_twcrps=0
    assert r2.abs().item() < 1e-6, f"perfect match should give ~0, got {r2.item()}"

    # Gradient check
    samples.requires_grad_(True)
    loss = twcrps_pathwise_max(samples, future, threshold)
    loss.backward()
    assert samples.grad is not None and samples.grad.abs().sum() > 0
    print("twcrps_pathwise_max sanity check PASS")


if __name__ == "__main__":
    main()
