# 233a-v1.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 233a-v1.2 (7-variant ablation testing 6 diagnosed bugs from v1 investigation) + full evaluation pipeline, applying the decision tree in the design spec.

**Architecture:** Subclass `TwoPathFactorAR` from v1. Override `__init__`, `forward_full` with branches on `use_v1_film_pipe` and `use_learned_link`. Add 2 new modules (`FiLMFromHSlow`, `LearnedLink`) + 1 new loss fn (`twcrps_pathwise_max`). All details in `research/233a_twopath_v1_2/design.md` (683 lines, commit `39a7153`).

**Tech Stack:** PyTorch, existing 227a/233a-v1 infrastructure, repo's v3 evaluation harness. Code in `experiments/backfill/block_ar/`. Data in `data/vol_surface_with_ret.npz`.

---

## Phase 0 — Setup + prerequisites

### Task 0.1: Create the v1.2 research workspace

**Files:**
- Create: `models/backfill/233a_v1_2_{control,minreg,minimal,aux,link,both,noreg}_25d_s42/` (7 empty dirs + .gitkeep)
- Create: `results/block_ar/233a_v1_2/` (empty dir + .gitkeep)

- [ ] **Step 1: Create output directories + .gitkeep sentinels**

```bash
cd /home/max/Documents/vol-surface-vae-pub
for V in control minreg minimal aux link both noreg; do
  mkdir -p "models/backfill/233a_v1_2_${V}_25d_s42"
  touch "models/backfill/233a_v1_2_${V}_25d_s42/.gitkeep"
done
mkdir -p results/block_ar/233a_v1_2
touch results/block_ar/233a_v1_2/.gitkeep
ls -la models/backfill/ | grep 233a_v1_2
```

Expected: 7 directories listed.

- [ ] **Step 2: Verify v1 PCA artifact + v1 checkpoints still exist**

```bash
# PCA artifact (from v1 Task 0.2)
ls -la models/backfill/coarse_pca_233a.npz

# v1 checkpoints (for comparison baselines)
ls -la models/backfill/233a_v1_full_25d_s42/best_model.pt
ls -la models/backfill/factor_ar_229a_wide_decoder/checkpoint_ep30.pt

# v1 eval results
ls -la results/block_ar/233a/full_s42/suite.json
ls -la results/block_ar/233a/_baseline_229a_newproxy/suite.json
```

Expected: all files exist.

- [ ] **Step 3: Commit workspace skeleton**

```bash
git add -f models/backfill/233a_v1_2_control_25d_s42/.gitkeep \
           models/backfill/233a_v1_2_minreg_25d_s42/.gitkeep \
           models/backfill/233a_v1_2_minimal_25d_s42/.gitkeep \
           models/backfill/233a_v1_2_aux_25d_s42/.gitkeep \
           models/backfill/233a_v1_2_link_25d_s42/.gitkeep \
           models/backfill/233a_v1_2_both_25d_s42/.gitkeep \
           models/backfill/233a_v1_2_noreg_25d_s42/.gitkeep \
           results/block_ar/233a_v1_2/.gitkeep
git commit -m "chore(233a-v1.2): add workspace skeleton directories"
```

---

## Phase 1 — Core new modules

### Task 1.1: Create skeleton `train_233a_v1_2_twopath_factor_ar.py`

**Files:**
- Create: `experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py`

- [ ] **Step 1: Create skeleton**

```python
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
    """C1 fix: FiLM reads h_slow directly, bypassing broken (s_hybrid, lam_hybrid) bottleneck."""
    pass


class LearnedLink(nn.Module):
    """C4b fix: learned emission link g(v) = α·tanh(v) + (1−α)·sinh(v)."""
    pass


def twcrps_pathwise_max(samples, future, threshold):
    """C4a fix: threshold-weighted CRPS on per-cell max|Δx| functional."""
    raise NotImplementedError


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


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntactic validity + imports resolve**

```bash
PYTHONPATH=. python -c "
import ast
with open('experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py') as f:
    ast.parse(f.read())
print('syntax OK')
"
PYTHONPATH=. python -c "
from experiments.backfill.block_ar.train_233a_twopath_factor_ar import (
    TwoPathFactorAR, CoarseFeatures, SlowPath, ScaleJumpHead,
    straight_through_bernoulli, FiLM,
)
print('v1 imports OK')
"
```

Expected: both print OK.

- [ ] **Step 3: Commit**

```bash
git add experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py
git commit -m "feat(233a-v1.2): add skeleton train_233a_v1_2_twopath_factor_ar.py"
```

### Task 1.2: Implement `FiLMFromHSlow`

**Files:**
- Modify: `experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py`

- [ ] **Step 1: Replace `FiLMFromHSlow` stub with implementation**

```python
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
```

- [ ] **Step 2: Add sanity check — identity-at-init**

Append before `if __name__`:

```python
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
```

- [ ] **Step 3: Run sanity check**

```bash
PYTHONPATH=. python -c "
from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import _sanity_check_film_from_hslow
_sanity_check_film_from_hslow()
"
```

Expected: `FiLMFromHSlow sanity check PASS`.

- [ ] **Step 4: Commit**

```bash
git add experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py
git commit -m "feat(233a-v1.2): FiLMFromHSlow — reads h_slow directly, γ-identity init preserved"
```

### Task 1.3: Implement `LearnedLink`

**Files:**
- Modify: `experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py`

- [ ] **Step 1: Replace `LearnedLink` stub with implementation**

```python
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
```

- [ ] **Step 2: Add sanity check — zero-init gives α=0.5, gradient flows**

```python
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
```

- [ ] **Step 3: Run**

```bash
PYTHONPATH=. python -c "
from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import _sanity_check_learned_link
_sanity_check_learned_link()
"
```

Expected: `LearnedLink sanity check PASS`.

- [ ] **Step 4: Commit**

```bash
git add experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py
git commit -m "feat(233a-v1.2): LearnedLink — zero-init gate (α=0.5 at init); gradient verified"
```

### Task 1.4: Implement `twcrps_pathwise_max`

**Files:**
- Modify: `experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py`

- [ ] **Step 1: Replace stub**

```python
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
```

- [ ] **Step 2: Add sanity check**

```python
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
```

- [ ] **Step 3: Run**

```bash
PYTHONPATH=. python -c "
from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import _sanity_check_twcrps
_sanity_check_twcrps()
"
```

- [ ] **Step 4: Commit**

```bash
git add experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py
git commit -m "feat(233a-v1.2): twcrps_pathwise_max — threshold-weighted CRPS; K*(K-1) denominator; gradient verified"
```

---

## Phase 2 — TwoPathFactorARv1_2 subclass

### Task 2.1: Subclass constructor with variant branches

**Files:**
- Modify: `experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py`

- [ ] **Step 1: Replace `TwoPathFactorARv1_2` stub**

```python
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
```

- [ ] **Step 2: Smoke-test construction for all 7 flag combos**

```python
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
```

- [ ] **Step 3: Run**

```bash
PYTHONPATH=. python -c "
from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import _sanity_check_v1_2_constructor
_sanity_check_v1_2_constructor()
"
```

Expected: 7 lines printed, all variants correctly configured, `v1_2 constructor sanity check PASS`.

- [ ] **Step 4: Commit**

```bash
git add experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py
git commit -m "feat(233a-v1.2): TwoPathFactorARv1_2.__init__ with use_v1_film_pipe + use_learned_link branches"
```

### Task 2.2: Override `forward_full` with branches + `q_seq_film`

**Files:**
- Modify: `experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py`

This is the largest task. The override copies v1's `forward_full` body but adds 3 specific changes:
1. **FiLM input branch:** `self.film(h_slow)` if `not use_v1_film_pipe`, else `self.film(s_t, lam_t)`
2. **Emission link branch:** `self.emission_link(v, cond)` if `use_learned_link`, else `torch.sinh(v)`
3. **Collect `q_seq_film`:** list of per-step `film_out["p_jump_logit"]` for the new BCE loss
4. **Rename `q_seq` → `q_seq_slow` in the returned dict**

- [ ] **Step 1: Add forward_full override to TwoPathFactorARv1_2**

Insert as a method of `TwoPathFactorARv1_2` (use the SAME structure as v1's forward_full, with marked differences):

```python
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
```

- [ ] **Step 2: Sanity check shape contracts for both `use_v1_film_pipe` values**

```python
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
```

- [ ] **Step 3: Run**

```bash
PYTHONPATH=. python -c "
from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import _sanity_check_forward_full_v1_2
_sanity_check_forward_full_v1_2()
"
```

- [ ] **Step 4: Commit**

```bash
git add experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py
git commit -m "feat(233a-v1.2): forward_full override with C1, C2, C4b branches; q_seq_slow/q_seq_film separation"
```

### Task 2.3: Override `forward_B`/`forward_C` for learned link (variants v1.2-B/C not used, but keep API)

**Files:**
- Modify: `experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py`

v1.2 only uses variant="full". But for cleanliness and diagnostic reuse, override forward_B/C to also apply the learned link if enabled (otherwise they fall back to v1 behavior).

- [ ] **Step 1: Minimal overrides**

Add to `TwoPathFactorARv1_2`:

```python
    def forward_B(self, history, future=None, n_members=8, n_steps=30, p_gt_feedback=0.0, **kwargs):
        """v1.2 variant=B: just delegate to v1's forward_B (C4b doesn't apply; no FiLM in B variant)."""
        # v1.2 ablation variants are variant="full" only. forward_B is kept for API compatibility.
        return super().forward_B(history, future, n_members, n_steps, p_gt_feedback, **kwargs)

    def forward_C(self, history, future=None, n_members=8, n_steps=30, p_gt_feedback=0.0, **kwargs):
        """v1.2 variant=C: just delegate to v1's forward_C."""
        return super().forward_C(history, future, n_members, n_steps, p_gt_feedback, **kwargs)
```

- [ ] **Step 2: Verify inheritance works**

```bash
PYTHONPATH=. python -c "
from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import TwoPathFactorARv1_2
# Make sure subclass has these methods and they're inherited (same id or delegated)
m_cls = TwoPathFactorARv1_2
assert hasattr(m_cls, 'forward_full')
assert hasattr(m_cls, 'forward_B')
assert hasattr(m_cls, 'forward_C')
assert hasattr(m_cls, 'forward')
print('v1_2 method inheritance OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py
git commit -m "feat(233a-v1.2): forward_B / forward_C delegate to v1 (API compat, not used by ablation)"
```

---

## Phase 3 — Loss, CLI, smoke trains

### Task 3.1: Implement `compute_loss_v1_2`

**Files:**
- Modify: `experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py`

- [ ] **Step 1: Replace stub with full loss**

```python
def compute_loss_v1_2(
    model_output: dict,
    future: torch.Tensor,               # (B, N, D) flat or (B, N, 5, 5)
    q90_train: torch.Tensor,
    # v1-inherited weights:
    lambda_vs: float = 0.05,
    lambda_rv: float = 0.10,
    lambda_jump: float = 0.05,          # slow-path BCE weight (on q_seq_slow)
    # v1.2 new weights:
    lambda_film_bce: float = 0.0,
    lambda_twcrps: float = 0.0,
    lambda_state: float = 0.0,
    state_reg_window: int = 5,
) -> dict:
    """
    7-component loss per design spec §4:
      L_ES + λ_VS·L_VS + λ_rv·L_rv_mse + λ_jump·L_slow_jump_bce
           + λ_BCE·L_film_jump_bce + λ_twcrps·L_twcrps + λ_state·L_state
    """
    samples = model_output["samples"]
    B, K, N, D = samples.shape
    device = samples.device

    # Flatten future if grid-shaped
    if future.dim() == 4:
        future = future.reshape(future.shape[0], future.shape[1], -1)

    # Per-step ES + VS (v1 inherited)
    L_ES = torch.zeros((), device=device)
    L_VS = torch.zeros((), device=device)
    for t in range(N):
        L_ES = L_ES + energy_score(samples[:, :, t], future[:, t])
        L_VS = L_VS + variogram_score(samples[:, :, t], future[:, t])
    L_ES = L_ES / N
    L_VS = L_VS / N

    # v1-inherited slow-path aux: log-RV MSE + jump BCE on q_seq_slow
    L_rv = torch.zeros((), device=device)
    L_slow_jump = torch.zeros((), device=device)
    # v1.2 new: jump BCE on q_seq_film
    L_film_jump = torch.zeros((), device=device)

    rv_seq = model_output.get("rv_pred_seq", [])
    q_seq_slow = model_output.get("q_seq_slow", [])
    q_seq_film = model_output.get("q_seq_film", [])
    valid = min(N - 1, len(rv_seq), len(q_seq_slow), len(q_seq_film))
    if valid > 0:
        for t in range(valid):
            dy = future[:, t + 1] - future[:, t]
            mean_sq_dy = (dy ** 2).mean(dim=-1).clamp_min(1e-10)
            target_log_rv = torch.log(mean_sq_dy)
            target_jump = (dy.norm(dim=-1) > q90_train).float()
            L_rv = L_rv + F.mse_loss(rv_seq[t], target_log_rv)
            L_slow_jump = L_slow_jump + F.binary_cross_entropy_with_logits(q_seq_slow[t], target_jump)
            L_film_jump = L_film_jump + F.binary_cross_entropy_with_logits(q_seq_film[t], target_jump)
        L_rv = L_rv / valid
        L_slow_jump = L_slow_jump / valid
        L_film_jump = L_film_jump / valid

    # C4a: twCRPS aux loss
    L_twcrps = torch.zeros((), device=device)
    if lambda_twcrps > 0:
        future_nd = future.reshape(B, N, D) if future.dim() == 3 else future.reshape(B, N, D)
        L_twcrps = twcrps_pathwise_max(samples, future_nd[:, :N], q90_train)

    # C3: state consistency reg
    L_state = torch.zeros((), device=device)
    if lambda_state > 0 and model_output.get("h_slow_teacher_seq"):
        free_seq = model_output["h_slow_free_seq"]
        teacher_seq = model_output["h_slow_teacher_seq"]
        W = min(state_reg_window, len(teacher_seq), len(free_seq))
        if W > 0:
            for t in range(W):
                L_state = L_state + F.mse_loss(free_seq[t], teacher_seq[t])
            L_state = L_state / W

    L_total = (
        L_ES
        + lambda_vs * L_VS
        + lambda_rv * L_rv
        + lambda_jump * L_slow_jump
        + lambda_film_bce * L_film_jump
        + lambda_twcrps * L_twcrps
        + lambda_state * L_state
    )

    return dict(
        L_total=L_total,
        L_ES=L_ES, L_VS=L_VS,
        L_rv=L_rv, L_slow_jump=L_slow_jump,
        L_film_jump=L_film_jump, L_twcrps=L_twcrps,
        L_state=L_state,
    )
```

- [ ] **Step 2: Sanity check — all 7 variants produce finite loss with correct-zero contributions**

```python
def _sanity_check_compute_loss_v1_2():
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

    # v1.2-both config: all fixes on
    m = TwoPathFactorARv1_2(
        variant="full", use_v1_film_pipe=False, use_learned_link=True,
        pca_artifact_path=path, **kwargs,
    ).train()
    out = m.forward_full(history, future, n_members=4, n_steps=5,
                         p_gt_feedback=0.5, return_teacher_h=True, state_reg_window=3)
    losses = compute_loss_v1_2(
        out, future[:, :5], q90_train=m.q90_train,
        lambda_vs=0.05, lambda_rv=0.10, lambda_jump=0.05,
        lambda_film_bce=0.05, lambda_twcrps=0.05, lambda_state=0.10,
    )
    for k in ["L_total", "L_ES", "L_VS", "L_rv", "L_slow_jump", "L_film_jump", "L_twcrps", "L_state"]:
        assert k in losses
        assert torch.isfinite(losses[k]).item(), f"{k} not finite"
    losses["L_total"].backward()    # must not raise

    # v1.2-control config: all new fixes off
    m_c = TwoPathFactorARv1_2(
        variant="full", use_v1_film_pipe=True, use_learned_link=False,
        pca_artifact_path=path, **kwargs,
    ).train()
    out_c = m_c.forward_full(history, future, n_members=4, n_steps=5,
                              p_gt_feedback=0.5, return_teacher_h=False)
    losses_c = compute_loss_v1_2(
        out_c, future[:, :5], q90_train=m_c.q90_train,
        lambda_film_bce=0.0, lambda_twcrps=0.0, lambda_state=0.0,
    )
    # All "new" losses weighted to 0 → don't contribute to L_total
    assert losses_c["L_state"].item() == 0.0
    # But their values still computed for logging:
    assert losses_c["L_film_jump"].item() > 0   # computed but zero-weighted

    os.unlink(path)
    print("compute_loss_v1_2 sanity check PASS")
```

- [ ] **Step 3: Run**

```bash
PYTHONPATH=. python -c "
from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import _sanity_check_compute_loss_v1_2
_sanity_check_compute_loss_v1_2()
"
```

- [ ] **Step 4: Commit**

```bash
git add experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py
git commit -m "feat(233a-v1.2): compute_loss_v1_2 with 7 loss terms; v1-inherited + C2/C4a/C3 adds"
```

### Task 3.2: Training loop + CLI with variant guard

**Files:**
- Modify: `experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py`

- [ ] **Step 1: Add `main()` and helpers**

Insert at the bottom (before `if __name__`):

```python
def parse_curriculum(spec: str) -> list:
    return [(int(a.split(":")[0]), int(a.split(":")[1])) for a in spec.split(",")]


def get_horizon(curriculum: list, epoch: int) -> int:
    H = curriculum[0][1]
    for e, h in curriculum:
        if epoch >= e:
            H = h
    return H


# Mapping from --variant_name to (use_v1_film_pipe, use_learned_link, λ_film_bce, λ_twcrps, λ_state)
VARIANT_CONFIGS = {
    "control":  (True,  False, 0.00, 0.00, 0.00),
    "minreg":   (False, False, 0.05, 0.00, 0.00),
    "minimal":  (False, False, 0.05, 0.00, 0.10),
    "aux":      (False, False, 0.05, 0.05, 0.10),
    "link":     (False, True,  0.05, 0.00, 0.10),
    "both":     (False, True,  0.05, 0.05, 0.10),
    "noreg":    (False, True,  0.05, 0.05, 0.00),
}


def apply_variant_config(args):
    """Given --variant_name, set all variant-specific flags. Overrides any manually-set flags
    (variant_name is the source of truth). Fails if variant_name is not in VARIANT_CONFIGS."""
    if args.variant_name not in VARIANT_CONFIGS:
        raise SystemExit(
            f"Unknown --variant_name: {args.variant_name}. "
            f"Must be one of {sorted(VARIANT_CONFIGS.keys())}"
        )
    v1_pipe, link, lfb, ltwc, lstate = VARIANT_CONFIGS[args.variant_name]
    args.use_v1_film_pipe = v1_pipe
    args.use_learned_link = link
    args.lambda_film_bce = lfb
    args.lambda_twcrps = ltwc
    args.lambda_state = lstate
    return args


def parse_args():
    p = argparse.ArgumentParser()
    # Variant guard: explicit variant selection; overrides all architectural flags
    p.add_argument("--variant_name", choices=sorted(VARIANT_CONFIGS.keys()), required=True,
                   help="Named variant; fills in all architectural + loss-weight flags.")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--pca_artifact", default="models/backfill/coarse_pca_233a.npz")
    p.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    p.add_argument("--history_len", type=int, default=30)
    p.add_argument("--n_steps", type=int, default=30)
    p.add_argument("--test_start", type=int, default=4511)
    p.add_argument("--val_size", type=int, default=441)
    p.add_argument("--max_train_windows", type=int, default=4010)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--n_members", type=int, default=8)
    p.add_argument("--lr", type=float, default=4e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # v1-inherited loss weights (defaults match v1)
    p.add_argument("--lambda_vs", type=float, default=0.05)
    p.add_argument("--lambda_rv", type=float, default=0.10)
    p.add_argument("--lambda_jump", type=float, default=0.05)
    # v1.2 new flags (set by --variant_name; can be overridden manually but discouraged)
    p.add_argument("--use_v1_film_pipe", action="store_true")
    p.add_argument("--use_learned_link", action="store_true")
    p.add_argument("--lambda_film_bce", type=float, default=0.0)
    p.add_argument("--lambda_twcrps", type=float, default=0.0)
    p.add_argument("--lambda_state", type=float, default=0.0)
    p.add_argument("--state_reg_window", type=int, default=5)
    # Curriculum + feedback
    p.add_argument("--curriculum_schedule", default="0:5,10:15,25:30")
    p.add_argument("--feedback_decay_end", type=int, default=30)
    # Model
    p.add_argument("--hidden_dim", type=int, default=128)
    p.add_argument("--factor_rank", type=int, default=6)
    p.add_argument("--slow_hidden", type=int, default=8)
    p.add_argument("--coarse_window", type=int, default=10)
    p.add_argument("--har_windows", default="1,5,22")
    p.add_argument("--alpha_init", type=float, default=-1.4)
    p.add_argument("--hawkes_init", default="0.1,0.5,0.5")
    p.add_argument("--rho", type=float, default=0.8)
    p.add_argument("--ewma_alpha", type=float, default=0.20)
    p.add_argument("--scale_floor", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    return apply_variant_config(args)


def _training_step(model, hist, fut, H, p_gt, args):
    fut_H = fut[:, :H]
    return model.forward_full(
        hist, fut_H,
        n_members=args.n_members, n_steps=H,
        p_gt_feedback=p_gt,
        return_teacher_h=(args.lambda_state > 0),   # gate teacher branch
        state_reg_window=args.state_reg_window,
    ), fut_H


def main():
    args = parse_args()
    print(f"\n233a-v1.2 variant={args.variant_name} seed={args.seed}")
    print(f"  Flags: use_v1_film_pipe={args.use_v1_film_pipe}, use_learned_link={args.use_learned_link}")
    print(f"  λ: vs={args.lambda_vs}, rv={args.lambda_rv}, jump_slow={args.lambda_jump}, "
          f"film_bce={args.lambda_film_bce}, twcrps={args.lambda_twcrps}, state={args.lambda_state}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    curriculum = parse_curriculum(args.curriculum_schedule)

    # Data (same as v1)
    from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows
    from torch.utils.data import DataLoader, TensorDataset

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf = torch.from_numpy(surfaces).to(device)
    max_train_idx = args.test_start - args.history_len - args.n_steps
    train_idx = np.arange(0, max_train_idx - args.val_size)[:args.max_train_windows]
    val_idx = np.arange(max_train_idx - args.val_size, max_train_idx)

    train_hist, train_future = build_multistep_windows(train_idx, surf, args.history_len, args.n_steps)
    train_future = train_future.view(train_hist.shape[0], args.n_steps, 5, 5)
    val_hist, val_future = build_multistep_windows(val_idx, surf, args.history_len, args.n_steps)
    val_future = val_future.view(val_hist.shape[0], args.n_steps, 5, 5)

    train_loader = DataLoader(TensorDataset(train_hist, train_future),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False)

    # Model
    hawkes_init = tuple(float(x) for x in args.hawkes_init.split(","))
    model = TwoPathFactorARv1_2(
        variant="full",
        use_v1_film_pipe=args.use_v1_film_pipe,
        use_learned_link=args.use_learned_link,
        pca_artifact_path=args.pca_artifact,
        slow_hidden=args.slow_hidden,
        coarse_window=args.coarse_window,
        har_windows=tuple(int(x) for x in args.har_windows.split(",")),
        alpha_init=args.alpha_init,
        hawkes_init=hawkes_init,
        hidden_dim=args.hidden_dim,
        factor_rank=args.factor_rank,
        rho=args.rho,
        ewma_alpha=args.ewma_alpha,
        scale_floor=args.scale_floor,
        n_cells=25,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_log = []
    best_val = float("inf")

    for epoch in range(args.epochs):
        t0 = time.time()
        H = get_horizon(curriculum, epoch)
        p_gt = max(0.0, 1.0 - epoch / max(args.feedback_decay_end, 1))

        model.train()
        train_totals = {k: 0.0 for k in ["L_total", "L_ES", "L_VS", "L_rv", "L_slow_jump", "L_film_jump", "L_twcrps", "L_state"]}
        train_batches = 0
        for hist, fut in train_loader:
            hist, fut = hist.to(device), fut.to(device)
            out, fut_H = _training_step(model, hist, fut, H, p_gt, args)
            losses = compute_loss_v1_2(
                out, fut_H, q90_train=model.q90_train,
                lambda_vs=args.lambda_vs, lambda_rv=args.lambda_rv, lambda_jump=args.lambda_jump,
                lambda_film_bce=args.lambda_film_bce,
                lambda_twcrps=args.lambda_twcrps,
                lambda_state=args.lambda_state,
                state_reg_window=args.state_reg_window,
            )
            opt.zero_grad(set_to_none=True)
            losses["L_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            for k in train_totals:
                train_totals[k] += float(losses[k].detach().item())
            train_batches += 1
        train_means = {k: train_totals[k] / max(train_batches, 1) for k in train_totals}

        model.eval()
        val_losses = []
        with torch.no_grad():
            for hist, fut in val_loader:
                hist, fut = hist.to(device), fut.to(device)
                out, fut_H = _training_step(model, hist, fut, H, 0.0, args)
                losses = compute_loss_v1_2(
                    out, fut_H, q90_train=model.q90_train,
                    lambda_vs=args.lambda_vs, lambda_rv=args.lambda_rv, lambda_jump=args.lambda_jump,
                    lambda_film_bce=args.lambda_film_bce, lambda_twcrps=args.lambda_twcrps,
                    lambda_state=args.lambda_state,
                )
                val_losses.append(float(losses["L_total"].item()))

        mean_val = float(np.mean(val_losses)) if val_losses else float("inf")
        dt = time.time() - t0
        row = dict(epoch=epoch, H=H, p_gt=round(p_gt, 3), val_loss=mean_val, dt_sec=round(dt, 2),
                   **{f"train_{k}": train_means[k] for k in train_means})
        history_log.append(row)
        print(f"ep {epoch:3d}  H={H:2d}  p_gt={p_gt:.2f}  train_total={train_means['L_total']:.4f}  "
              f"val={mean_val:.4f}  dt={dt:.1f}s", flush=True)

        ckpt = {"model_state_dict": model.state_dict(), "epoch": epoch,
                "args": vars(args), "variant": "full", "variant_name": args.variant_name}
        torch.save(ckpt, output_dir / "final_model.pt")
        if mean_val < best_val:
            best_val = mean_val
            torch.save(ckpt, output_dir / "best_model.pt")

    with open(output_dir / "training_log.json", "w") as f:
        json.dump(history_log, f, indent=2)
    print(f"\nTraining complete. Best val loss = {best_val:.4f}")
    print(f"Saved: {output_dir / 'best_model.pt'}, {output_dir / 'final_model.pt'}")


def load_model(checkpoint_path, device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = payload["args"]
    model = TwoPathFactorARv1_2(
        variant="full",
        use_v1_film_pipe=args.get("use_v1_film_pipe", False),
        use_learned_link=args.get("use_learned_link", False),
        pca_artifact_path=args["pca_artifact"],
        slow_hidden=args["slow_hidden"],
        coarse_window=args["coarse_window"],
        har_windows=tuple(int(x) for x in args["har_windows"].split(",")),
        alpha_init=args["alpha_init"],
        hawkes_init=tuple(float(x) for x in args["hawkes_init"].split(",")),
        hidden_dim=args["hidden_dim"],
        factor_rank=args["factor_rank"],
        rho=args.get("rho", 0.8),
        ewma_alpha=args.get("ewma_alpha", 0.20),
        scale_floor=args.get("scale_floor", 1e-4),
        n_cells=25,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload
```

- [ ] **Step 2: Verify import + variant guard rejects unknown names**

```bash
PYTHONPATH=. python -c "
from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import (
    main, parse_args, VARIANT_CONFIGS, load_model, apply_variant_config
)
assert set(VARIANT_CONFIGS.keys()) == {'control','minreg','minimal','aux','link','both','noreg'}
# Test apply_variant_config rejects invalid
import argparse
args = argparse.Namespace(variant_name='bogus')
try:
    apply_variant_config(args); assert False, 'should have raised'
except SystemExit as e:
    print('variant guard rejects invalid: OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py
git commit -m "feat(233a-v1.2): main() with --variant_name guard (7 named variants); lambda_state gates teacher branch"
```

### Task 3.3: Stage-A smoke — `v1.2-both` (1 epoch)

**Files:** no changes.

- [ ] **Step 1: Run 1-epoch smoke**

```bash
PYTHONPATH=. python -u experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py \
    --variant_name both --seed 42 --output_dir /tmp/233a_v1_2_smoke_both \
    --epochs 1 --batch_size 8 --n_members 4 \
    --curriculum_schedule 0:5 --feedback_decay_end 1 \
    --device cuda 2>&1 | tail -40
```

Expected: "Training complete", finite val_loss (likely 0.4-0.7), ~20-30s wall time.

- [ ] **Step 2: Verify Stage-A gradient check**

```bash
PYTHONPATH=. python -c "
import torch
from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import load_model, compute_loss_v1_2
m, payload = load_model('/tmp/233a_v1_2_smoke_both/best_model.pt', torch.device('cuda'))
# FiLM logit weights should have received non-zero gradient during training
# (can't check grad directly after training, but verify params are not all-zero after 1 epoch)
logit_w = m.film.logit.weight
if logit_w.abs().sum().item() > 0:
    print(f'film.logit.weight has non-zero values: abs sum = {logit_w.abs().sum():.6f}')
else:
    print('WARNING: film.logit.weight is all zeros — C2 may not be wiring')
if m.emission_link is not None:
    gate_w = m.emission_link.gate.weight
    print(f'emission_link.gate.weight abs sum = {gate_w.abs().sum():.6f}')
"
```

Expected: both weights should have non-zero magnitudes (they started zero and received gradient during 1 epoch of training).

- [ ] **Step 3: Clean up**

```bash
rm -rf /tmp/233a_v1_2_smoke_both
```

- [ ] **Step 4: Commit (empty tag)**

```bash
git commit --allow-empty -m "test(233a-v1.2): Stage-A smoke on v1.2-both — 1 epoch, verifies C2 + C4b gradient flow"
```

### Task 3.4: Stage-A smoke — `v1.2-link` (1 epoch)

**Files:** no changes.

- [ ] **Step 1: Run 1-epoch smoke**

```bash
PYTHONPATH=. python -u experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py \
    --variant_name link --seed 42 --output_dir /tmp/233a_v1_2_smoke_link \
    --epochs 1 --batch_size 8 --n_members 4 \
    --curriculum_schedule 0:5 --feedback_decay_end 1 \
    --device cuda 2>&1 | tail -40
```

Expected: "Training complete", finite val_loss, ~20-30s wall.

- [ ] **Step 2: Verify emission_link weights moved**

```bash
PYTHONPATH=. python -c "
import torch
from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import load_model
m, payload = load_model('/tmp/233a_v1_2_smoke_link/best_model.pt', torch.device('cuda'))
assert m.emission_link is not None, 'emission_link missing for link variant'
gate_w = m.emission_link.gate.weight
print(f'emission_link.gate.weight abs sum = {gate_w.abs().sum():.6f}')
assert gate_w.abs().sum().item() > 0, 'learned link got no gradient'
print('link variant gradient check PASS')
"
```

- [ ] **Step 3: Clean up + commit**

```bash
rm -rf /tmp/233a_v1_2_smoke_link
git commit --allow-empty -m "test(233a-v1.2): Stage-A smoke on v1.2-link — learned link gradient flow verified"
```

---

## Phase 4 — Integration with eval harness + diagnostics

### Task 4.1: Extend loader dispatch in `_rollout_220_utils.py`

**Files:**
- Modify: `experiments/backfill/block_ar/_rollout_220_utils.py`

- [ ] **Step 1: Add dispatch clause**

Find the existing 233a dispatch (from v1) and add a v1.2 clause immediately after:

Find: `if model_type in {"233a", "233a_full", "233a_B", "233a_C"}:` and add AFTER its block:

```python
    if model_type in {"233a_v1_2", "233a_v1_2_full"}:
        from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import (
            load_model as load_233a_v1_2_model,
        )
        return load_233a_v1_2_model(checkpoint_path, device)
```

- [ ] **Step 2: Verify dispatch**

```bash
# Use a recent smoke checkpoint (re-generate if cleaned up)
PYTHONPATH=. python -u experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py \
    --variant_name minreg --seed 42 --output_dir /tmp/233a_v1_2_dispatch_test \
    --epochs 1 --batch_size 8 --n_members 4 \
    --curriculum_schedule 0:5 --feedback_decay_end 1 \
    --device cuda 2>&1 | tail -5

PYTHONPATH=. python -c "
import torch
from experiments.backfill.block_ar._rollout_220_utils import load_one_day_kernel
m, payload = load_one_day_kernel('233a_v1_2', '/tmp/233a_v1_2_dispatch_test/best_model.pt', torch.device('cuda'))
print(f'loaded: {type(m).__name__}, variant_name={payload[\"variant_name\"]}')
"
rm -rf /tmp/233a_v1_2_dispatch_test
```

Expected: `loaded: TwoPathFactorARv1_2, variant_name=minreg`.

- [ ] **Step 3: Commit**

```bash
git add experiments/backfill/block_ar/_rollout_220_utils.py
git commit -m "feat(233a-v1.2): extend loader dispatch for 233a_v1_2 model_type"
```

### Task 4.2: Wire 233a_v1_2 into `evaluate_220b_multihorizon_path_suite.py`

**Files:**
- Modify: `experiments/backfill/block_ar/evaluate_220b_multihorizon_path_suite.py`

- [ ] **Step 1: Add 233a_v1_2 to anchor-override + use_native gates**

Find the existing 233a override block (around line 90) and add after it:

```python
    # 233a_v1_2 variants: anchor always on, same as v1
    if args.model_type.startswith("233a_v1_2"):
        model.use_scale_anchor = True
        model.scale_anchor_alpha = 0.50
        print(f"[eval override] 233a_v1_2 {args.model_type}: use_scale_anchor=True, alpha=0.50")
```

Find the `use_native` condition and add to the set:

```python
    use_native = hasattr(model, 'sample_batched') and (
        hasattr(model, 'temporal_adapter')
        or args.model_type == '183c'
        or args.model_type in {'231a', '231b', '231c', '232a', '232b', '232c', '232d'}
        or args.model_type.startswith('233a')
        or args.model_type.startswith('233a_v1_2')
        or args.force_native_anchor
    )
```

- [ ] **Step 2: Verify syntax + wiring**

```bash
PYTHONPATH=. python -c "
import ast
with open('experiments/backfill/block_ar/evaluate_220b_multihorizon_path_suite.py') as f:
    ast.parse(f.read())
print('syntax OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add experiments/backfill/block_ar/evaluate_220b_multihorizon_path_suite.py
git commit -m "feat(233a-v1.2): wire 233a_v1_2 into evaluate_220b native+anchor path"
```

### Task 4.3: Create α-collapse diagnostic

**Files:**
- Create: `experiments/backfill/block_ar/diagnose_233a_v1_2_emission_link.py`

- [ ] **Step 1: Write diagnostic script**

```python
#!/usr/bin/env python
"""
Diagnose whether v1.2-link / v1.2-both / v1.2-noreg learned-link α
differentiates regimes or collapsed to a constant.

Outputs:
  - α distribution stats (std, min, max) across val windows
  - Per-regime α mean (calm vs turb)
  - Regime separation: |mean(α|turb) - mean(α|calm)|
  - Pass/fail vs Stage-B thresholds (std > 0.05, separation > 0.02)
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    ap.add_argument("--test_start", type=int, default=4511)
    ap.add_argument("--val_size", type=int, default=441)
    ap.add_argument("--n_windows", type=int, default=200)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import load_model
    from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows

    device = torch.device(args.device)
    m, payload = load_model(args.checkpoint, device)

    if m.emission_link is None:
        result = {"skipped": "variant has no learned link; α collapse N/A"}
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(result, f, indent=2)
        print("No learned link in this variant; skipping")
        return

    # Load val windows
    raw = np.load(args.data_path)
    surf = torch.from_numpy(raw["surface"].astype(np.float32)).to(device)
    max_train_idx = args.test_start - 30 - 30
    val_idx = np.arange(max_train_idx - args.val_size, max_train_idx)[:args.n_windows]
    hist, _ = build_multistep_windows(val_idx, surf, 30, 30)

    # Regime labels from realized-variance proxy
    hist_np = hist.cpu().numpy()
    dhist = np.diff(hist_np.reshape(hist_np.shape[0], 30, 25), axis=1)
    rv = (dhist ** 2).mean(axis=(1, 2))
    q20, q80 = np.quantile(rv, [0.20, 0.80])
    calm_mask = rv <= q20
    turb_mask = rv >= q80

    # Collect α at step 0 from each val window
    m.eval()
    with torch.no_grad():
        # Use encode_history to get initial cond for each window
        hist_in = hist.to(device)
        cond_B, _ = m.encode_history(hist_in)   # (B, hidden)
        alpha_init = torch.sigmoid(m.emission_link.gate(cond_B))   # (B, D)

    alpha_np = alpha_init.cpu().numpy()
    alpha_flat = alpha_np.reshape(-1)

    result = {
        "checkpoint": args.checkpoint,
        "variant_name": payload.get("variant_name", "?"),
        "n_windows": int(hist.shape[0]),
        "n_calm": int(calm_mask.sum()),
        "n_turb": int(turb_mask.sum()),
        "alpha_overall_stats": {
            "mean": float(alpha_flat.mean()),
            "std": float(alpha_flat.std()),
            "min": float(alpha_flat.min()),
            "max": float(alpha_flat.max()),
        },
        "alpha_per_regime": {
            "calm_mean": float(alpha_np[calm_mask].mean()) if calm_mask.any() else None,
            "turb_mean": float(alpha_np[turb_mask].mean()) if turb_mask.any() else None,
        },
        "regime_separation": float(
            alpha_np[turb_mask].mean() - alpha_np[calm_mask].mean()
        ) if calm_mask.any() and turb_mask.any() else None,
    }

    # Pass/fail gates (design spec Stage B thresholds)
    result["gates"] = {
        "alpha_std_gt_0p05": result["alpha_overall_stats"]["std"] > 0.05,
        "abs_separation_gt_0p02": (
            abs(result["regime_separation"]) > 0.02 if result["regime_separation"] else False
        ),
    }
    result["alpha_collapsed"] = not (
        result["gates"]["alpha_std_gt_0p05"] or result["gates"]["abs_separation_gt_0p02"]
    )

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke-run on Stage-A `v1.2-link` checkpoint (regenerate it)**

```bash
PYTHONPATH=. python -u experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py \
    --variant_name link --seed 42 --output_dir /tmp/233a_v1_2_link_test \
    --epochs 1 --batch_size 8 --n_members 4 \
    --curriculum_schedule 0:5 --feedback_decay_end 1 \
    --device cuda 2>&1 | tail -3

PYTHONPATH=. python experiments/backfill/block_ar/diagnose_233a_v1_2_emission_link.py \
    --checkpoint /tmp/233a_v1_2_link_test/best_model.pt \
    --output_json /tmp/alpha_test.json --n_windows 40 --device cuda | head -30

rm -rf /tmp/233a_v1_2_link_test /tmp/alpha_test.json
```

Expected: JSON with `alpha_overall_stats` showing mean near 0.5 (zero-init gate, 1-epoch training won't move it much), `alpha_collapsed` likely True (1 epoch is too short), which is correct behavior for a smoke check.

- [ ] **Step 3: Commit**

```bash
git add experiments/backfill/block_ar/diagnose_233a_v1_2_emission_link.py
git commit -m "feat(233a-v1.2): diagnose_233a_v1_2_emission_link — α-collapse detection diagnostic"
```

### Task 4.4: Create comparator with roll-up table

**Files:**
- Create: `experiments/backfill/block_ar/compare_233a_v1_2_variants.py`

- [ ] **Step 1: Write comparator**

```python
#!/usr/bin/env python
"""
Compare 233a-v1.2 variants vs 229a and v1 baselines.

Produces:
  1. Per-variant metrics table (n_pass, turb_calm, worstC_h30, max_jump_ks, ks_test, MR)
  2. Attribution deltas (from design spec §7)
  3. Roll-up table: 5 diagnostics × 7 variants
  4. Decision tree outcome (Branch 1 / 2a / 2b / 3 / etc.)
"""

import json
from pathlib import Path
import numpy as np

RESULT_DIR = Path("results/block_ar/233a_v1_2")
V1_RESULT_DIR = Path("results/block_ar/233a")
SEEDS = [42]
VARIANTS = ["control", "minreg", "minimal", "aux", "link", "both", "noreg"]

INCUMBENT_229a = {
    "n_pass": 3, "turb_calm": 1.025, "worst_cell_cov": 0.270,
    "max_jump_ks": 0.940, "change_ks_h30": 19, "MR_ratio": 1.318,
}

METRIC_PATHS = {
    "n_pass":         "summary.n_pass",
    "turb_calm":      "conditionality.turb_calm_ratio",
    "worst_cell_cov": "coverage.worst_cell_per_horizon.30",
    "max_jump_ks":    "pathwise_jump_realism.pathwise_max_jump.ks_stat",
    "change_ks_h30":  "distributional_fidelity.ks_test.n_pass",
    "MR_ratio":       "mean_reversion.gt_ratio",
}

DIAGNOSTIC_FIELDS = [
    ("film_logit_std", "_diagnostic_film_collapse.json", "logit_std_final_seed_42"),
    ("alpha_std", "_diagnostic_emission_link.json", "alpha_overall_stats.std"),
    ("lag1_autocorr", "_diagnostic_ar_compounding.json", "model_v1_2.lag1_autocorr_h29"),
    ("h_slow_auc", "_diagnostic_slow_state.json", "h_slow_pc1_auc_seed_42"),
    ("regime_inversion_flag", "_diagnostic_regime_breakdown.json", "regime_inversion_detected"),
]


def _get(tree, dotted):
    cur = tree
    for k in dotted.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def load_suite(variant, seed=42):
    path = RESULT_DIR / f"{variant}_s{seed}" / "suite.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def summarise_variant(variant):
    d = load_suite(variant)
    if d is None:
        return None
    return {name: _get(d, path) for name, path in METRIC_PATHS.items()}


def diagnostic_rollup(variant):
    row = {"variant": variant}
    for field_name, file_basename, dotted in DIAGNOSTIC_FIELDS:
        path = RESULT_DIR / f"{variant}_s42" / file_basename
        if path.exists():
            with open(path) as f:
                d = json.load(f)
            row[field_name] = _get(d, dotted)
        else:
            row[field_name] = None
    return row


def main():
    # Per-variant metrics
    results = {v: summarise_variant(v) for v in VARIANTS}
    # Baselines
    v1_path = V1_RESULT_DIR / "full_s42" / "suite.json"
    b229_path = V1_RESULT_DIR / "_baseline_229a_newproxy" / "suite.json"
    v1_metrics = {name: _get(json.load(open(v1_path)), path) for name, path in METRIC_PATHS.items()} if v1_path.exists() else None
    b229_metrics = {name: _get(json.load(open(b229_path)), path) for name, path in METRIC_PATHS.items()} if b229_path.exists() else None

    # Print metrics table
    print("=" * 110)
    print("233a-v1.2 — 7-variant comparison (seed 42)")
    print("=" * 110)
    header = f"{'config':<24}"
    for m in ["n_pass", "turb_calm", "worstC_h30", "max_jump_ks", "change_ks_h30", "MR_ratio"]:
        header += f" {m:>12}"
    print(header)
    print("-" * 110)

    def _fmt(row, m):
        v = row.get(m) if row else None
        if v is None: return "     —"
        if m == "n_pass": return f"{int(v):>12}"
        if m == "change_ks_h30": return f"{int(v):>12}"
        return f"{float(v):>12.3f}"

    if b229_metrics:
        print(f"{'229a @ep30 (incumbent)':<24}" + "".join(_fmt(b229_metrics, m) for m in METRIC_PATHS))
    if v1_metrics:
        print(f"{'v1-full_s42 (prev)':<24}" + "".join(_fmt(v1_metrics, m) for m in METRIC_PATHS))
    for v in VARIANTS:
        row = results[v]
        label = f"v1.2-{v}_s42"
        print(f"{label:<24}" + "".join(_fmt(row, m) for m in METRIC_PATHS))

    # Attribution deltas
    print("\n" + "=" * 110)
    print("Attribution deltas")
    print("=" * 110)
    def _d(a, b, key):
        va = a.get(key) if a else None
        vb = b.get(key) if b else None
        if va is None or vb is None: return None
        return va - vb

    if results["control"] and results["minreg"]:
        print(f"C1+C2 alone (minreg - control): Δn_pass={_d(results['minreg'], results['control'], 'n_pass')}, "
              f"Δturb_calm={_d(results['minreg'], results['control'], 'turb_calm'):.3f}")
    if results["minimal"] and results["minreg"]:
        print(f"C3 state-reg alone (minimal - minreg): Δn_pass={_d(results['minimal'], results['minreg'], 'n_pass')}")
    if results["aux"] and results["minimal"]:
        print(f"C4a twCRPS alone (aux - minimal): Δmax_jump_ks={_d(results['aux'], results['minimal'], 'max_jump_ks'):.3f}")
    if results["link"] and results["minimal"]:
        print(f"C4b learned-link alone (link - minimal): Δmax_jump_ks={_d(results['link'], results['minimal'], 'max_jump_ks'):.3f}")
    if results["both"] and results["minimal"]:
        print(f"C4a+C4b combined (both - minimal): Δmax_jump_ks={_d(results['both'], results['minimal'], 'max_jump_ks'):.3f}")

    # Diagnostic roll-up table
    print("\n" + "=" * 110)
    print("Diagnostic roll-up (5 diagnostics × 7 variants)")
    print("=" * 110)
    diag_header = f"{'variant':<12}"
    for f, _, _ in DIAGNOSTIC_FIELDS:
        diag_header += f" {f:>22}"
    print(diag_header)
    print("-" * 110)
    for v in VARIANTS:
        row = diagnostic_rollup(v)
        cells = [f"{row['variant']:<12}"]
        for f, _, _ in DIAGNOSTIC_FIELDS:
            val = row.get(f)
            cells.append(f"{'—' if val is None else (str(val) if isinstance(val, bool) else f'{val:.3f}'):>22}")
        print("".join(cells))

    # Decision tree
    print("\n" + "=" * 110)
    print("Decision tree (per Section 8 of design spec)")
    print("=" * 110)
    best = None
    best_n = -1
    for v in VARIANTS:
        r = results[v]
        if r and r["n_pass"] is not None and r["n_pass"] > best_n:
            best = v; best_n = r["n_pass"]
    if best is None:
        print("INSUFFICIENT DATA — missing suite.json for all variants")
        return
    bm = results[best]
    n = bm["n_pass"]; jk = bm.get("max_jump_ks", 1.0) or 1.0
    if n >= 5 and jk < 0.50:
        print(f"BRANCH 1 (clean success): best={best}, n/7={n}, max_jump_ks={jk:.3f}")
        print("  → architectural signal only; multi-seed replication required in v1.3")
        print("  → H=252 smoke test next; if fails, DEMOTE to Branch 2a")
    elif n >= 4 and jk < 0.50:
        print(f"BRANCH 2b (partial success + emission cracked): best={best}, n/7={n}, max_jump_ks={jk:.3f}")
        print("  → most informative partial; design targeted v1.2.x fix for remaining failing suite")
    elif n >= 4:
        print(f"BRANCH 2a (partial — FiLM fixed, emission cap): best={best}, n/7={n}, max_jump_ks={jk:.3f}")
        print("  → publish 4/7; Bug 6 is next bottleneck; parallel H3 scaffold design")
    else:
        print(f"BRANCH 3 (failure): best={best}, n/7={n}")
        print("  → paradigm pivot justified; launch H3 + joint-path flow matching")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run (all variants missing — should print INSUFFICIENT DATA)**

```bash
PYTHONPATH=. python experiments/backfill/block_ar/compare_233a_v1_2_variants.py 2>&1 | head -30
```

Expected: output with headers, all cells `—`, decision tree prints "INSUFFICIENT DATA" OR shows Branch 3 since all missing variants score 0.

- [ ] **Step 3: Commit**

```bash
git add experiments/backfill/block_ar/compare_233a_v1_2_variants.py
git commit -m "feat(233a-v1.2): compare_233a_v1_2_variants.py with attribution deltas + 5-diagnostic roll-up + decision tree"
```

---

## Phase 5 — Orchestration + full training

### Task 5.1: Write orchestration script

**Files:**
- Create: `experiments/backfill/block_ar/eval_233a_v1_2_ladder.sh`

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
set -euo pipefail

# 233a-v1.2 ladder: 7 variants × 1 seed = 7 training runs + 7 evals + 5 diagnostics on best.
# Expected wall time ~50 min with 3-way GPU parallelism.

SEED=42
COMMON="--seed ${SEED} --epochs 60 --batch_size 32 --n_members 8 \
        --curriculum_schedule 0:5,10:15,25:30 --feedback_decay_end 30 \
        --lambda_vs 0.05 --lambda_rv 0.10 --lambda_jump 0.05 \
        --state_reg_window 5 --device cuda"

for VARIANT in control minreg minimal aux link both noreg; do
  OUT_DIR=models/backfill/233a_v1_2_${VARIANT}_25d_s${SEED}
  mkdir -p ${OUT_DIR}
  echo "=== Training v1.2-${VARIANT} seed=${SEED} ==="
  PYTHONPATH=. python -u experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py \
      --variant_name ${VARIANT} --output_dir ${OUT_DIR} \
      ${COMMON} 2>&1 | tee ${OUT_DIR}/training.log
done

echo ""
echo "=== Evaluating 7 v1.2 variants ==="
for VARIANT in control minreg minimal aux link both noreg; do
  OUT_DIR=models/backfill/233a_v1_2_${VARIANT}_25d_s${SEED}
  RESULT_DIR=results/block_ar/233a_v1_2/${VARIANT}_s${SEED}
  mkdir -p ${RESULT_DIR}
  echo "=== Eval v1.2-${VARIANT} ==="
  PYTHONPATH=. python experiments/backfill/block_ar/evaluate_220b_multihorizon_path_suite.py \
      --model_type 233a_v1_2 \
      --checkpoint ${OUT_DIR}/best_model.pt \
      --force_native_anchor \
      --max_windows 192 --samples 48 \
      --output_json ${RESULT_DIR}/suite.json \
      --output_md   ${RESULT_DIR}/suite.md \
      --device cuda 2>&1 | tee ${RESULT_DIR}/eval.log
done

echo "=== Ladder complete. Run compare_233a_v1_2_variants.py for decision. ==="
```

- [ ] **Step 2: Make executable + syntax check + commit**

```bash
chmod +x experiments/backfill/block_ar/eval_233a_v1_2_ladder.sh
bash -n experiments/backfill/block_ar/eval_233a_v1_2_ladder.sh
echo "syntax OK"
git add experiments/backfill/block_ar/eval_233a_v1_2_ladder.sh
git commit -m "feat(233a-v1.2): orchestration script for 7-variant ladder + eval"
```

### Task 5.2: Launch full 7-variant training

**Files:** no changes. This is the compute.

- [ ] **Step 1: Launch 3 parallel jobs (control + minreg + minimal first)**

```bash
# Control
PYTHONPATH=. python -u experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py \
    --variant_name control --seed 42 \
    --output_dir models/backfill/233a_v1_2_control_25d_s42 \
    --epochs 60 --batch_size 32 --n_members 8 \
    --curriculum_schedule 0:5,10:15,25:30 --feedback_decay_end 30 \
    --device cuda 2>&1 | tee models/backfill/233a_v1_2_control_25d_s42/training.log &

# Minreg
PYTHONPATH=. python -u experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py \
    --variant_name minreg --seed 42 \
    --output_dir models/backfill/233a_v1_2_minreg_25d_s42 \
    --epochs 60 --batch_size 32 --n_members 8 \
    --curriculum_schedule 0:5,10:15,25:30 --feedback_decay_end 30 \
    --device cuda 2>&1 | tee models/backfill/233a_v1_2_minreg_25d_s42/training.log &

# Minimal
PYTHONPATH=. python -u experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py \
    --variant_name minimal --seed 42 \
    --output_dir models/backfill/233a_v1_2_minimal_25d_s42 \
    --epochs 60 --batch_size 32 --n_members 8 \
    --curriculum_schedule 0:5,10:15,25:30 --feedback_decay_end 30 \
    --device cuda 2>&1 | tee models/backfill/233a_v1_2_minimal_25d_s42/training.log &

wait
```

- [ ] **Step 2: Launch next batch (aux + link + both + noreg)**

```bash
# After first batch finishes, run remaining 4 (3-way parallel + 1 sequential)
for VARIANT in aux link both; do
  PYTHONPATH=. python -u experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py \
      --variant_name ${VARIANT} --seed 42 \
      --output_dir models/backfill/233a_v1_2_${VARIANT}_25d_s42 \
      --epochs 60 --batch_size 32 --n_members 8 \
      --curriculum_schedule 0:5,10:15,25:30 --feedback_decay_end 30 \
      --device cuda 2>&1 | tee models/backfill/233a_v1_2_${VARIANT}_25d_s42/training.log &
done
wait

# Noreg last (sequential-ish)
PYTHONPATH=. python -u experiments/backfill/block_ar/train_233a_v1_2_twopath_factor_ar.py \
    --variant_name noreg --seed 42 \
    --output_dir models/backfill/233a_v1_2_noreg_25d_s42 \
    --epochs 60 --batch_size 32 --n_members 8 \
    --curriculum_schedule 0:5,10:15,25:30 --feedback_decay_end 30 \
    --device cuda 2>&1 | tee models/backfill/233a_v1_2_noreg_25d_s42/training.log
```

- [ ] **Step 3: Verify all 7 completed + commit training logs**

```bash
for V in control minreg minimal aux link both noreg; do
  ls models/backfill/233a_v1_2_${V}_25d_s42/training_log.json || echo "MISSING: ${V}"
done
git add -f models/backfill/233a_v1_2_*_25d_s42/training_log.json
git commit -m "run(233a-v1.2): 7-variant training complete (seed 42)"
```

---

## Phase 6 — Evaluation + decision

### Task 6.1: Run evaluate_220b on all 7 checkpoints

**Files:** no changes.

- [ ] **Step 1: Evaluate**

```bash
for V in control minreg minimal aux link both noreg; do
  mkdir -p results/block_ar/233a_v1_2/${V}_s42
  PYTHONPATH=. python experiments/backfill/block_ar/evaluate_220b_multihorizon_path_suite.py \
      --model_type 233a_v1_2 \
      --checkpoint models/backfill/233a_v1_2_${V}_25d_s42/best_model.pt \
      --force_native_anchor \
      --max_windows 192 --samples 48 \
      --output_json results/block_ar/233a_v1_2/${V}_s42/suite.json \
      --output_md   results/block_ar/233a_v1_2/${V}_s42/suite.md \
      --device cuda 2>&1 | tee results/block_ar/233a_v1_2/${V}_s42/eval.log
done
```

- [ ] **Step 2: Commit eval artifacts**

```bash
git add -f results/block_ar/233a_v1_2/*_s42/suite.json results/block_ar/233a_v1_2/*_s42/suite.md results/block_ar/233a_v1_2/*_s42/eval.log
git commit -m "eval(233a-v1.2): 7-variant v3 suite evaluation (seed 42)"
```

### Task 6.2: Run 5 diagnostics on all 7 variants

**Files:** no changes.

- [ ] **Step 1: Run all 5 diagnostic scripts on each variant**

```bash
for V in control minreg minimal aux link both noreg; do
  CKPT=models/backfill/233a_v1_2_${V}_25d_s42/best_model.pt
  DDIR=results/block_ar/233a_v1_2/${V}_s42

  # FiLM collapse (reuse v1 script with new model-type dispatch note)
  PYTHONPATH=. python experiments/backfill/block_ar/diagnose_233a_film_collapse.py \
      --checkpoints ${CKPT} --checkpoint_tags ${V}_v1_2 --model_type 233a_v1_2 \
      --output_dir ${DDIR}/_diagnostic_film_collapse 2>&1 | tail -5 || echo "film_collapse diagnostic may need v1.2 dispatch"

  # Slow state
  PYTHONPATH=. python experiments/backfill/block_ar/diagnose_233a_slow_state.py \
      --checkpoints ${CKPT} --checkpoint_tags ${V}_v1_2 --model_type 233a_v1_2 \
      --output_dir ${DDIR}/_diagnostic_slow_state 2>&1 | tail -5 || true

  # AR compounding
  PYTHONPATH=. python experiments/backfill/block_ar/diagnose_233a_ar_compounding.py \
      --checkpoints ${CKPT} --checkpoint_tags ${V}_v1_2 --model_type 233a_v1_2 \
      --output_dir ${DDIR}/_diagnostic_ar_compounding 2>&1 | tail -5 || true

  # Regime breakdown (from suite.json; no per-variant rerun needed; already done)

  # α collapse (NEW) — only applicable to variants with learned link
  if [[ "${V}" == "link" || "${V}" == "both" || "${V}" == "noreg" ]]; then
    PYTHONPATH=. python experiments/backfill/block_ar/diagnose_233a_v1_2_emission_link.py \
        --checkpoint ${CKPT} --output_json ${DDIR}/_diagnostic_emission_link.json \
        --n_windows 200 --device cuda | tail -30
  fi
done
```

Expected: diagnostic JSONs written for each variant.

- [ ] **Step 2: Commit diagnostics**

```bash
git add -f results/block_ar/233a_v1_2/*_s42/_diagnostic_*.{json,md} 2>/dev/null
git commit -m "diag(233a-v1.2): 5-diagnostic rerun on 7 variants; α-collapse for link/both/noreg"
```

### Task 6.3: Run comparator + decision

**Files:** no changes; outputs `results/block_ar/233a_v1_2/decision.md`.

- [ ] **Step 1: Run comparator**

```bash
PYTHONPATH=. python experiments/backfill/block_ar/compare_233a_v1_2_variants.py \
    | tee results/block_ar/233a_v1_2/decision.md
```

Expected: per-variant metrics table, attribution deltas, diagnostic roll-up, and the Branch decision (1 / 2a / 2b / 3) printed.

- [ ] **Step 2: Commit decision**

```bash
git add -f results/block_ar/233a_v1_2/decision.md
git commit -m "eval(233a-v1.2): comparator + decision tree applied"
```

### Task 6.4: H=252 smoke test on best variant

**Files:** no changes; run the smoke test programmatically.

- [ ] **Step 1: Determine best variant from decision output**

```bash
BEST_VARIANT=$(grep -oE 'best=[a-z]+' results/block_ar/233a_v1_2/decision.md | head -1 | cut -d'=' -f2)
echo "Best variant: ${BEST_VARIANT}"
```

- [ ] **Step 2: Run 252-day smoke**

```bash
PYTHONPATH=. python -c "
import torch, numpy as np, json
from experiments.backfill.block_ar.train_233a_v1_2_twopath_factor_ar import load_model
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv

best = '${BEST_VARIANT}'
if not best: import sys; sys.exit('no best variant parsed')
ckpt = f'models/backfill/233a_v1_2_{best}_25d_s42/best_model.pt'
m, payload = load_model(ckpt, torch.device('cuda'))

# Need enough future days to evaluate at h=200-252
raw = np.load('data/vol_surface_with_ret.npz')
surf = torch.from_numpy(raw['surface'].astype(np.float32)).cuda()
max_train_idx = 4511 - 30 - 252
val_idx = np.arange(max_train_idx - 20, max_train_idx)   # 20 windows (H=252)
hist, fut = build_multistep_windows(val_idx, surf, 30, 252)

with torch.no_grad():
    hist_norm = normalize_iv(hist)
    samples = m.sample_batched(hist_norm, n_samples=24, n_steps=252, chunk_size=8)   # (20, 24, 252, 5, 5)

# Gates
s_flat = samples.reshape(20, 24, 252, 25)
has_nan = bool(torch.isnan(s_flat).any())
iv_in_range = bool((s_flat >= 0.01).all() and (s_flat <= 1.0).all())

# Max-delta distribution h=25-30 vs h=200-252
gen_delta = (s_flat[:, :, 1:] - s_flat[:, :, :-1]).abs()
max_delta_early = gen_delta[:, :, 24:30].max(dim=2).values   # (20, 24, 25)
max_delta_late = gen_delta[:, :, 199:252].max(dim=2).values  # (20, 24, 25)
early_mean = float(max_delta_early.mean())
late_mean = float(max_delta_late.mean())
ratio = late_mean / max(early_mean, 1e-8)
delta_ratio_ok = bool(0.5 < ratio < 2.0)

# MR_ratio at h=200-252 (on GT-conditioned future)
fut_flat = fut.reshape(20, 252, 25).cuda()
dfut = (fut_flat[:, 200:] - fut_flat[:, 199:-1]).abs()
dgen = gen_delta[:, :, 199:].mean(dim=1)    # (20, 52, 25)
mr_ratio_long = float(dgen.mean() / max(dfut.mean(), 1e-8))
mr_long_ok = bool(0.60 < mr_ratio_long < 1.40)

result = dict(
    best_variant=best,
    has_nan=has_nan, iv_in_range=iv_in_range,
    max_delta_early=early_mean, max_delta_late=late_mean, delta_ratio=ratio,
    delta_ratio_ok=delta_ratio_ok,
    mr_ratio_long=mr_ratio_long, mr_long_ok=mr_long_ok,
    all_gates_pass=not has_nan and iv_in_range and delta_ratio_ok and mr_long_ok,
)
with open('results/block_ar/233a_v1_2/h252_smoke.json', 'w') as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
"
```

- [ ] **Step 2b: Commit smoke result**

```bash
git add -f results/block_ar/233a_v1_2/h252_smoke.json
git commit -m "eval(233a-v1.2): H=252 smoke test on best variant (Branch 1 follow-up gate)"
```

### Task 6.5: Write RESEARCH_LOG entry

**Files:**
- Modify: `RESEARCH_LOG.md`

- [ ] **Step 1: Append entry using research-log skill idiom**

```bash
cat >> RESEARCH_LOG.md << 'LOG_EOF'

## $(date +%Y-%m-%d): 233a-v1.2 — Targeted Bug-Fix Experiment

### Context
v1.2 tests whether the 6 bugs diagnosed on v1 (commits b732ee0..37b635f) are fixable
within the AR paradigm. Seven-variant ablation grid (control, minreg, minimal, aux, link,
both, noreg) × 1 seed × 60 epochs. Architecture/loss exploration phase.

### Variant Grid Results (seed 42)
[Table from `compare_233a_v1_2_variants.py` output; PASTE HERE from results/block_ar/233a_v1_2/decision.md]

### Attribution Deltas
[6 delta lines from the comparator output; PASTE HERE]

### Mechanism Confirmation (post-train diagnostics)
[From diagnostic roll-up; note which variants had FiLM std > 0.01, α separation OK, lag1_autocorr improved, regime inversion resolved]

### Decision (Branch from Section 8)
[State which Branch outcome: 1 / 2a / 2b / 3 / 4 / 5 / 6 / 7]
[If Branch 1: did H=252 smoke pass?]

### Next Direction
[Per Branch: either v1.3 multi-seed + multi-factor, targeted v1.2.x fix, or paradigm pivot]

### Artifacts
- Training: models/backfill/233a_v1_2_{variant}_25d_s42/
- Eval: results/block_ar/233a_v1_2/{variant}_s42/suite.json
- Decision: results/block_ar/233a_v1_2/decision.md
- Diagnostics: results/block_ar/233a_v1_2/*/diagnostic_*.{json,md}
- H=252 smoke: results/block_ar/233a_v1_2/h252_smoke.json

---
LOG_EOF
```

- [ ] **Step 2: Fill in the table placeholders manually**

After reviewing the comparator output, edit the RESEARCH_LOG entry's `[PASTE HERE]` placeholders with the actual numeric table + decision. Use a text editor or Edit tool.

- [ ] **Step 3: Re-index + commit**

```bash
qmd update --collection research && qmd embed
git add RESEARCH_LOG.md
git commit -m "docs(233a-v1.2): research log entry with variant grid + decision"
```

### Task 6.6: Update MEMORY.md

**Files:**
- Modify: `/home/max/.claude/projects/-home-max-Documents-vol-surface-vae-pub/memory/MEMORY.md`

- [ ] **Step 1: Update CURRENT STATE section with Branch-specific outcome text**

Use Edit tool to replace `MEMORY.md` CURRENT STATE section with:
- Branch 1 outcome: "Phase: v1.3 multi-seed validation. 233a-v1.2 achieved ≥5/7 on seed 42..."
- Branch 2a outcome: "Phase: Bug 6 emission redesign..."
- Branch 3 outcome: "Phase: paradigm pivot to joint-path flow matching..."

Pick whichever matches the actual Branch outcome.

- [ ] **Step 2: Add `rc24_233a_v1_2_outcome.md` topic file**

Create a new topic file in memory with:
- Title of outcome
- Key numbers from decision
- Specific fixes that worked / didn't

- [ ] **Step 3: Link from MEMORY.md "Active" section**

```markdown
### Active
- [RC24 233a-v1.2 outcome](rc24_233a_v1_2_outcome.md) — 2026-04-XX: [Branch X outcome summary]
```

---

## Phase 7 — Close-out

### Task 7.1: Mark plan complete

- [ ] **Step 1: Final empty commit**

```bash
git commit --allow-empty -m "chore(233a-v1.2): implementation plan complete — Branch X decision applied"
git log --oneline | head -5
```

---

## Self-Review

### Spec coverage

- §1 Architecture (C1-C4) → Tasks 1.2, 1.3, 1.4, 2.1, 2.2 ✓
- §2 Variant Grid (7 variants) → Task 3.2 (`VARIANT_CONFIGS`) ✓
- §3 Data Flow (BK expansion + slow-state branching) → Task 2.2 ✓
- §4 Loss (7 terms) → Task 3.1 ✓
- §5 Training Recipe (CLI, defaults, variant guard) → Task 3.2 ✓
- §6 Files to Create/Modify → Phase 1-4 ✓
- §7 Evaluation + Attribution → Tasks 4.4, 6.1, 6.3 ✓
- §8 Stop Criteria (Stage A/B/C) → Tasks 3.3, 3.4, 6.1 ✓ (Stage B monitoring embedded in training loop, not task-enforced — acknowledged)
- §9 RESEARCH_LOG + Artifact Protocol → Tasks 6.5, 6.6 ✓
- H=252 smoke test (Branch 1 gate) → Task 6.4 ✓
- α-collapse diagnostic → Task 4.3 ✓

### Placeholder scan

No TBD / TODO / "implement later". Task 6.5 has explicit `[PASTE HERE]` directives for the human-filled sections (decision outcomes are data-dependent; can't be pre-written).

### Type consistency

- `TwoPathFactorARv1_2` used consistently
- `compute_loss_v1_2` returns dict with 8 keys (`L_total, L_ES, L_VS, L_rv, L_slow_jump, L_film_jump, L_twcrps, L_state`) — matches design §4's 7 loss components + L_total
- `q_seq_slow` / `q_seq_film` rename consistent across forward_full, compute_loss_v1_2, and diagnostic scripts
- `VARIANT_CONFIGS` has all 7 entries; variant guard in main() matches

---
