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
    """Restricted FiLM coupling: (s, λ) -> (γ_Λ, β_Λ, γ_D, β_D, drift_bias, p_jump_logit)."""
    pass


class TwoPathFactorAR(nn.Module):
    """v1-full model. Subclasses FactorARModel227a and adds slow path + FiLM + jump mixture."""
    pass


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


if __name__ == "__main__":
    main()   # defined in later task
