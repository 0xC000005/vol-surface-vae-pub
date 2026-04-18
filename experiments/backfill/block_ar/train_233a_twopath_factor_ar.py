#!/usr/bin/env python
"""
233a: Two-Path Factor AR — v1 + two ablation variants (v1-B, v1-C).

Spec: research/233a_twopath_v1/design.md

Variants (--variant flag):
  full  : two-path with learned slow state + Hawkes + FiLM + jump mixture
  B     : no slow path; aux-supervised cond_fast with parallel residual MLP
  C     : HAR-feature control (no learned state, 5 deterministic HAR features)
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import namedtuple
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-use existing 227a/226a infrastructure unchanged
from experiments.backfill.block_ar.train_227a_factor_ar import (
    FactorARModel as FactorARModel227a,
)

FastFeatures = namedtuple("FastFeatures", ["feature_vec", "local_scale"])


class CoarseFeatures(nn.Module):
    """
    Data-agnostic coarse features — all 5 features from Section 2.3 of design.md.
    PCA matrices loaded from coarse_pca_233a.npz (frozen after precompute).

    Input:
      x_t: (B, D) current day
      buffer: list of up to coarse_window (B, D) tensors (oldest first)

    Output:
      features: (B, F) where F ≈ 13 (4+4+4+1-ish = 13)
    """

    def __init__(self, pca_artifact_path: str, coarse_window: int = 10):
        super().__init__()
        artifact = np.load(pca_artifact_path)
        self.coarse_window = int(artifact["coarse_window"])
        assert self.coarse_window == coarse_window, f"window mismatch: {coarse_window} vs artifact"
        self.register_buffer("pca_mean_comp", torch.tensor(artifact["pca_mean_components"]))
        self.register_buffer("pca_mean_mean", torch.tensor(artifact["pca_mean_mean"]))
        self.register_buffer("pca_disp_comp", torch.tensor(artifact["pca_disp_components"]))
        self.register_buffer("pca_disp_mean", torch.tensor(artifact["pca_disp_mean"]))
        self.register_buffer("pca_rsc_comp",  torch.tensor(artifact["pca_rsc_components"]))
        self.register_buffer("pca_rsc_mean",  torch.tensor(artifact["pca_rsc_mean"]))
        self.register_buffer("q90_train",     torch.tensor(float(artifact["q90_train"])))
        self.output_dim = 4 + 4 + 4 + 1 + 1   # 14

    def _pad_window(self, buffer: list) -> torch.Tensor:
        """Left-pad buffer to coarse_window size using oldest frame."""
        window = buffer[-self.coarse_window:]
        if len(window) < self.coarse_window:
            window = [window[0]] * (self.coarse_window - len(window)) + window
        return torch.stack(window, dim=1)   # (B, W, D)

    def forward(self, x_t: torch.Tensor, buffer: list) -> torch.Tensor:
        W = self._pad_window(buffer)                     # (B, 10, D)
        # 1. Mean level -> PCA 4d
        mean_level = W.mean(dim=1)                       # (B, D)
        mean_proj = (mean_level - self.pca_mean_mean) @ self.pca_mean_comp.T   # (B, 4)
        # 2. Dispersion -> PCA 4d
        disp = W.std(dim=1)
        disp_proj = (disp - self.pca_disp_mean) @ self.pca_disp_comp.T
        # 3. Realized squared change -> PCA 4d
        d_w = W[:, 1:] - W[:, :-1]
        rsc = (d_w ** 2).mean(dim=1)
        rsc_proj = (rsc - self.pca_rsc_mean) @ self.pca_rsc_comp.T
        # 4. Cross-series coactivation
        coact = d_w.abs().mean(dim=-1).mean(dim=-1, keepdim=True)   # (B, 1)
        # 5. Quantile-threshold jump indicator (most recent only — 1-d)
        dx_latest = (x_t - buffer[-1]) if len(buffer) > 0 else torch.zeros_like(x_t)
        j_indicator = (dx_latest.norm(dim=-1, keepdim=True) > self.q90_train).float()
        return torch.cat([mean_proj, disp_proj, rsc_proj, coact, j_indicator], dim=-1)   # (B, 14)


# --- stubs filled in by later tasks ---
class SlowPath(nn.Module):
    """
    Slow path: GRUSlow hidden state + hybrid analytic/learned EWMA + Hawkes.

    Inputs per step: coarse_features (B, F_coarse), x_t, x_{t-1}, j_t, Δt_last_jump
    Outputs:        h_t (B, slow_hidden), s_t (B,), λ_t (B,)
                    + q_t (B,) jump_prob logit, rv_pred (B,) for aux supervision
    """

    def __init__(
        self,
        coarse_feature_dim: int = 14,
        slow_hidden: int = 8,
        alpha_init: float = -1.4,       # sigmoid(-1.4) ≈ 0.2
        lambda_base_init: float = 0.1,
        alpha_H_init: float = 0.5,
        beta_H_init: float = 0.5,
    ):
        super().__init__()
        self.slow_hidden = slow_hidden
        self.gru_slow = nn.GRUCell(coarse_feature_dim, slow_hidden)

        # Scalar learned parameters
        self.theta_alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
        self.lambda_base = nn.Parameter(torch.tensor(lambda_base_init, dtype=torch.float32))
        self.alpha_H = nn.Parameter(torch.tensor(alpha_H_init, dtype=torch.float32))
        self.beta_H = nn.Parameter(torch.tensor(beta_H_init, dtype=torch.float32))

        # Learned residual correction heads
        self.linear_s = nn.Linear(slow_hidden, 1)
        self.linear_lam = nn.Linear(slow_hidden, 1)

        # Auxiliary heads
        self.rv_head = nn.Linear(slow_hidden, 1)
        self.jump_prob_head = nn.Linear(slow_hidden, 1)

    @property
    def alpha_learn(self) -> torch.Tensor:
        return torch.sigmoid(self.theta_alpha)

    def step(
        self,
        coarse_t: torch.Tensor,
        h_prev: torch.Tensor,
        s_ewma_prev: torch.Tensor,
        lam_hawkes_prev: torch.Tensor,
        delta_t_last_jump: torch.Tensor,   # (B,) days since last jump
        mean_sq_dx: torch.Tensor,           # (B,) mean((Δx)^2) this step
        j_t: torch.Tensor,                   # (B,) binary jump indicator
    ) -> dict:
        """Single-step slow-path update. Returns dict with all derived quantities."""
        a = self.alpha_learn
        h_t = self.gru_slow(coarse_t, h_prev)

        # Analytic backbone
        s_ewma_t = (1.0 - a) * s_ewma_prev + a * mean_sq_dx
        decay = torch.exp(-F.softplus(self.beta_H) * delta_t_last_jump)
        lam_hawkes_t = self.lambda_base + F.softplus(self.alpha_H) * decay * j_t

        # Hybrid (analytic + learned residual)
        s_t = F.softplus(s_ewma_t + self.linear_s(h_t).squeeze(-1))      # (B,)
        lam_t = F.relu(lam_hawkes_t + self.linear_lam(h_t).squeeze(-1))  # (B,)

        # Aux heads (used for training losses only)
        rv_pred = self.rv_head(h_t).squeeze(-1)
        q_t = self.jump_prob_head(h_t).squeeze(-1)

        return dict(
            h_t=h_t, s_t=s_t, lam_t=lam_t,
            s_ewma_t=s_ewma_t, lam_hawkes_t=lam_hawkes_t,
            rv_pred=rv_pred, q_t=q_t,
        )


class FiLM(nn.Module):
    """
    Restricted FiLM coupling: (s_t, λ_t) -> modulation parameters.

    Input pre-scaling with log1p spreads the small-magnitude inputs so the
    MLP's output is not bias-dominated early in training.
    γ heads start at identity (γ = 1) so the fast path is initially
    equivalent to 226a (no modulation). drift_bias, β heads, logit start at 0.
    """

    def __init__(self, D: int, k: int, hidden: int = 32):
        super().__init__()
        self.D = D
        self.k = k
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Output heads — zero-init all weights AND biases; γ is handled with +1.0 in forward()
        self.g_lambda = nn.Linear(hidden, k)        # γ_Λ (identity-at-init)
        self.b_lambda = nn.Linear(hidden, k)        # β_Λ
        self.g_d = nn.Linear(hidden, D)              # γ_D
        self.b_d = nn.Linear(hidden, D)
        self.drift = nn.Linear(hidden, D)
        self.logit = nn.Linear(hidden, 1)

        for head in (self.g_lambda, self.b_lambda, self.g_d, self.b_d, self.drift, self.logit):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, s_t: torch.Tensor, lam_t: torch.Tensor) -> dict:
        """
        s_t, lam_t: (B,) scalars
        Returns dict with keys: gamma_lambda, beta_lambda, gamma_d, beta_d, drift_bias, p_jump_logit
        """
        u = torch.stack([torch.log1p(s_t), torch.log1p(lam_t)], dim=-1)   # (B, 2)
        h = self.mlp(u)                                                     # (B, hidden)
        return dict(
            gamma_lambda=1.0 + self.g_lambda(h),   # (B, k) — identity at init
            beta_lambda=self.b_lambda(h),           # (B, k) — 0 at init
            gamma_d=1.0 + self.g_d(h),              # (B, D) — identity at init
            beta_d=self.b_d(h),
            drift_bias=self.drift(h),
            p_jump_logit=self.logit(h).squeeze(-1), # (B,) — 0 at init -> σ(0) = 0.5
        )


def straight_through_bernoulli(logit: torch.Tensor, shape: tuple) -> torch.Tensor:
    """
    Straight-through Bernoulli sampling.
    Forward: hard 0/1 sample. Backward: σ(logit) sigmoid gradient.

    logit: (B,) or broadcastable to shape
    shape: desired output shape (B, K, 1) or similar
    """
    p = torch.sigmoid(logit)
    p_broadcast = p.view(*p.shape, *([1] * (len(shape) - len(p.shape))))
    U = torch.rand(shape, device=logit.device)
    mask_hard = (U < p_broadcast).float()
    # Straight-through: forward=hard, backward=soft
    return mask_hard.detach() + p_broadcast - p_broadcast.detach()


class ScaleJumpHead(nn.Module):
    """Single learned scalar magnitude for the jump term; softplus(linear(λ_t → 1))."""
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(1, 1)

    def forward(self, lam_t: torch.Tensor) -> torch.Tensor:
        """lam_t: (B,) -> scale: (B,)"""
        return F.softplus(self.lin(lam_t.unsqueeze(-1)).squeeze(-1))


class TwoPathFactorAR(FactorARModel227a):
    """
    v1-full: 227a fast-path inherited; slow-path + FiLM + jump mixture bolted on.
    Variant switching via __init__ arg `variant`:
      "full" = slow path + FiLM + jump
      "B"    = no slow path; aux-supervised cond_fast with parallel residual MLP
      "C"    = HAR-feature control, no learned slow state
    """

    def __init__(
        self,
        variant: str = "full",
        pca_artifact_path: str = "models/backfill/coarse_pca_233a.npz",
        slow_hidden: int = 8,
        coarse_window: int = 10,
        har_windows: tuple = (1, 5, 22),
        alpha_init: float = -1.4,
        hawkes_init: tuple = (0.1, 0.5, 0.5),
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)
        self.variant = variant
        self.D = self.n_cells                       # inherit from 227a
        self.k = self.factor_rank                   # inherit from 227a (default 6 at D=25)
        self.coarse_window = coarse_window
        self.har_windows = har_windows

        # Anchor is always ON for 233a
        self.use_scale_anchor = True
        self.scale_anchor_alpha = 0.50

        # -- variant-specific wiring --
        if variant == "full":
            self.coarse = CoarseFeatures(pca_artifact_path, coarse_window=coarse_window)
            self.slow_path = SlowPath(
                coarse_feature_dim=self.coarse.output_dim,
                slow_hidden=slow_hidden,
                alpha_init=alpha_init,
                lambda_base_init=hawkes_init[0],
                alpha_H_init=hawkes_init[1],
                beta_H_init=hawkes_init[2],
            )
            self.film = FiLM(D=self.D, k=self.k, hidden=32)
            self.scale_jump_head = ScaleJumpHead()
            # q90_train is loaded into coarse.q90_train buffer
            self.register_buffer("q90_train", self.coarse.q90_train.clone())

        elif variant == "B":
            # v1-B: no slow path, aux-supervised cond_fast via bottleneck MLP
            self.aux_bottleneck = nn.Sequential(
                nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
            )
            self.rv_head_B = nn.Linear(32, 1)
            self.jump_head_B = nn.Linear(32, 1)
            self.cond_extra = nn.Sequential(
                nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU(),
            )
            self.cond_residual_proj = nn.Linear(32, 128)
            nn.init.zeros_(self.cond_residual_proj.weight)   # start at identity
            nn.init.zeros_(self.cond_residual_proj.bias)

        elif variant == "C":
            # v1-C: HAR-feature control (5 features); no learned state, aux, FiLM, jump
            self.har_concat_mlp = nn.Sequential(
                nn.Linear(128 + 5, 160), nn.ReLU(),
                nn.Linear(160, 128), nn.ReLU(),
            )
            # Need q90 for HAR jump-frequency feature
            artifact = np.load(pca_artifact_path)
            self.register_buffer("q90_train", torch.tensor(float(artifact["q90_train"])))

        else:
            raise ValueError(f"Unknown variant: {variant}")


def _sanity_check_model_constructors():
    # Minimal kwargs to match 227a's signature; adjust once we know what 227a needs
    kwargs = dict(hidden_dim=128, factor_rank=6, ewma_alpha=0.20, scale_floor=1e-4, n_cells=25)
    import tempfile, os
    # Create minimal fake PCA artifact so CoarseFeatures / variant-C can load
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
    try:
        for variant in ["full", "B", "C"]:
            m = TwoPathFactorAR(variant=variant, pca_artifact_path=path, **kwargs)
            n_params = sum(p.numel() for p in m.parameters())
            print(f"{variant}: {n_params/1e3:.1f}k params")
        print("constructor sanity check PASS")
    finally:
        os.unlink(path)


def _sanity_check_coarse_features():
    """Smoke test: coarse features produce expected shape."""
    import tempfile, os
    # Create minimal fake PCA artifact for testing
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
    cf = CoarseFeatures(path, coarse_window=10)
    x = torch.randn(2, 25)
    buffer = [torch.randn(2, 25) for _ in range(5)]   # only 5 days, tests padding
    out = cf(x, buffer)
    assert out.shape == (2, 14), f"expected (2, 14), got {out.shape}"
    print("CoarseFeatures sanity check PASS")
    os.unlink(path)


def _sanity_check_slow_path():
    """Smoke test: SlowPath.step produces expected shapes and gradients."""
    sp = SlowPath(coarse_feature_dim=14, slow_hidden=8)
    B = 3
    coarse = torch.randn(B, 14)
    h_prev = torch.zeros(B, 8)
    s_ewma = torch.full((B,), 0.01, dtype=torch.float32)
    lam_hawkes = torch.full((B,), 0.1, dtype=torch.float32)
    delta_t = torch.full((B,), 5.0, dtype=torch.float32)
    mean_sq = torch.full((B,), 0.02, dtype=torch.float32)
    j_t = torch.tensor([0.0, 1.0, 0.0])
    out = sp.step(coarse, h_prev, s_ewma, lam_hawkes, delta_t, mean_sq, j_t)
    assert out["h_t"].shape == (B, 8), f"h_t shape wrong: {out['h_t'].shape}"
    assert out["s_t"].shape == (B,)
    assert out["lam_t"].shape == (B,)
    assert (out["s_t"] >= 0).all(), "s_t should be non-negative (softplus)"
    assert (out["lam_t"] >= 0).all(), "lam_t should be non-negative (ReLU)"
    # Gradient check
    loss = out["s_t"].sum() + out["lam_t"].sum() + out["rv_pred"].sum() + out["q_t"].sum()
    loss.backward()
    assert sp.theta_alpha.grad is not None, "theta_alpha should receive gradient"
    print("SlowPath sanity check PASS")


def _sanity_check_film_identity_init():
    """FiLM γ outputs should be exactly 1.0 at init; β/drift should be 0."""
    film = FiLM(D=25, k=6, hidden=32)
    s = torch.tensor([0.1, 0.5])
    lam = torch.tensor([0.2, 0.3])
    out = film(s, lam)
    assert torch.allclose(out["gamma_lambda"], torch.ones_like(out["gamma_lambda"])), "γ_Λ not identity at init"
    assert torch.allclose(out["gamma_d"], torch.ones_like(out["gamma_d"])), "γ_D not identity at init"
    assert torch.allclose(out["beta_lambda"], torch.zeros_like(out["beta_lambda"])), "β_Λ not 0 at init"
    assert torch.allclose(out["beta_d"], torch.zeros_like(out["beta_d"])), "β_D not 0 at init"
    assert torch.allclose(out["drift_bias"], torch.zeros_like(out["drift_bias"])), "drift_bias not 0 at init"
    assert torch.allclose(out["p_jump_logit"], torch.zeros_like(out["p_jump_logit"])), "p_jump_logit not 0 at init"
    print("FiLM identity-at-init check PASS")


def _sanity_check_straight_through():
    logit = torch.tensor([0.0, 0.0], requires_grad=True)
    mask = straight_through_bernoulli(logit, shape=(2, 4, 1))
    assert mask.shape == (2, 4, 1)
    assert ((mask == 0.0) | (mask == 1.0)).all(), "forward must be hard 0/1"
    loss = mask.sum()
    loss.backward()
    # sigmoid(0)*(1-sigmoid(0)) = 0.25 per element; total 2*4*1 = 8 elements
    assert logit.grad is not None and logit.grad.abs().sum() > 0, "gradient must flow"
    print(f"straight-through Bernoulli grad check PASS (grad={logit.grad})")


if __name__ == "__main__":
    main()   # defined in later task
