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
    """Slow path: GRUSlow + hybrid (EWMA + Hawkes) analytic backbone + learned residuals."""
    pass


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


if __name__ == "__main__":
    main()   # defined in later task
