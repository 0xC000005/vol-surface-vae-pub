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
    """v1.2 = v1 + selective overrides for C1, C2, C3, C4a, C4b fixes."""
    pass


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
