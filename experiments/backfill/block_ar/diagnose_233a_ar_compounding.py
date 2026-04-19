#!/usr/bin/env python
"""
AR Compounding Diagnostic for 233a / 229a

Mechanistic investigation of WHERE and HOW quality degrades in 30-step AR rollout.

Four per-step statistics collected for both models, both modes (self-fed / teacher-forced):
  1. Innovation energy: mean squared daily delta per step
  2. Pathwise extremes: max |delta| per step
  3. Cross-member diversity: std of deltas across K members per step
  4. KS distance: 2-sample KS(generated deltas, GT deltas) at each step t

Additionally: lag-1 autocorrelation of daily deltas (self-fed only).

Teacher-forcing is handled via an explicit manual AR loop to bypass the
`if self.training` gate in forward_full / forward_B (would silently fall
through to self-fed under model.eval()).

Usage (from repo root, PYTHONPATH=.):
    python experiments/backfill/block_ar/diagnose_233a_ar_compounding.py \\
        --model_233a models/backfill/233a_v1_full_25d_s42/best_model.pt \\
        --model_229a models/backfill/factor_ar_229a_wide_decoder/checkpoint_ep30.pt \\
        --output_dir results/block_ar/233a \\
        --n_members 24 --max_windows 100 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, ".")


# ─── loaders ──────────────────────────────────────────────────────────────────
from experiments.backfill.block_ar._rollout_220_utils import (
    build_rollout_windows,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.train_233a_twopath_factor_ar import (
    TwoPathFactorAR,
    load_model as load_233a,
)
from experiments.backfill.block_ar.train_227a_factor_ar import (
    FactorARModel,
    load_model as load_227a,
)


# ─── KS helper ────────────────────────────────────────────────────────────────
def ks_2samp_np(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample KS statistic (max absolute CDF difference)."""
    a = np.sort(a.ravel())
    b = np.sort(b.ravel())
    all_vals = np.concatenate([a, b])
    all_vals = np.unique(all_vals)
    cdf_a = np.searchsorted(a, all_vals, side="right") / len(a)
    cdf_b = np.searchsorted(b, all_vals, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


# ─── 229a (FactorARModel / 227a architecture) manual loop ────────────────────
@torch.no_grad()
def rollout_227a_manual(
    model: FactorARModel,
    history_01: torch.Tensor,    # (B, 30, 5, 5)
    future_01: torch.Tensor,     # (B, 30, 5, 5)
    n_members: int,
    n_steps: int,
    teacher_forced: bool = False,
) -> tuple[torch.Tensor, list[dict]]:
    """
    Manual AR loop for 227a/229a architecture.
    Returns: trajectory (B, K, N, 25), step_stats list[dict].
    """
    B = history_01.shape[0]
    device = history_01.device
    D = model.n_cells  # 25

    cond, local_scale = model.encode_history(history_01)
    prev = history_01[:, -1].reshape(B, D)

    if model.use_scale_anchor:
        scale_anchor = local_scale.clone()  # (B, D)

    BK = B * n_members
    cond = cond.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
    local_scale = local_scale.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
    prev = prev.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)

    if model.use_scale_anchor:
        scale_anchor_bk = scale_anchor.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)

    # GT feedback shape
    if future_01 is not None:
        gt_flat = future_01.reshape(B, n_steps, D)
        gt_bk = gt_flat.unsqueeze(1).expand(B, n_members, n_steps, D).reshape(BK, n_steps, D)

    z_f = torch.randn(BK, model.factor_rank, device=device)
    rho_sq_comp = math.sqrt(1.0 - model.rho ** 2)

    frames = []  # each: (BK, D)
    step_stats = []

    for t in range(n_steps):
        if t > 0:
            z_f = model.rho * z_f + rho_sq_comp * torch.randn_like(z_f)
        z_i = torch.randn(BK, D, device=device)
        pos = model.pos_embed(t, BK, device)

        factor_in = torch.cat([prev, cond, z_f, pos], dim=-1)
        f_scores = model.factor_head(factor_in)

        idio_in = torch.cat([prev, cond, z_i, pos], dim=-1)
        i_resid = model.idio_head(idio_in)

        Lambda = model.get_lambda(cond)      # (BK, D, r)
        D_scale = model.get_d(cond)          # (BK, D)
        factor_contribution = torch.einsum("bdr,br->bd", Lambda, f_scores)
        idio_contribution = D_scale * i_resid
        v = factor_contribution + idio_contribution

        if model.noise_skip:
            v = v + torch.tanh(model.noise_skip_proj(z_i))
        if model.cell_spread_enabled:
            cs = F.softplus(model.cell_spread_proj(cond))
            v = v * cs

        delta = torch.sinh(v) * local_scale     # (BK, D)
        next_iv = (prev + delta).clamp(0.001, 1.0)
        frames.append(next_iv)

        # Collect per-step raw stats (keep on CPU as numpy for later processing)
        step_stats.append({
            "delta_cpu": delta.cpu().float().numpy(),   # (BK, D)
            "next_iv_cpu": next_iv.cpu().float().numpy(),
            "prev_cpu": prev.cpu().float().numpy(),
            "local_scale_mean": local_scale.mean().item(),
            "local_scale_std": local_scale.std().item(),
            "cond_norm": cond.norm(dim=-1).mean().item(),
            "factor_norm": factor_contribution.abs().mean().item(),
            "idio_norm": idio_contribution.abs().mean().item(),
        })

        # State update
        if teacher_forced and future_01 is not None:
            # Feed GT back — bypass self.training gate
            x_feedback_bk = gt_bk[:, t]  # (BK, D)
        else:
            x_feedback_bk = next_iv

        feat, local_scale = model._step_features(prev, x_feedback_bk, local_scale)
        cond = model.gru_cell(feat, cond)
        prev = x_feedback_bk

        if model.use_scale_anchor:
            log_s = (
                (1.0 - model.scale_anchor_alpha) * torch.log(local_scale.clamp_min(model.scale_floor))
                + model.scale_anchor_alpha * torch.log(scale_anchor_bk.clamp_min(model.scale_floor))
            )
            local_scale = torch.exp(log_s)

    traj = torch.stack(frames, dim=1)   # (BK, N, D)
    traj = traj.reshape(B, n_members, n_steps, D)
    return traj, step_stats


# ─── 233a-full manual loop ────────────────────────────────────────────────────
@torch.no_grad()
def rollout_233a_manual(
    model: TwoPathFactorAR,
    history_01: torch.Tensor,    # (B, T, 5, 5)
    future_01: torch.Tensor,     # (B, N, 5, 5)
    n_members: int,
    n_steps: int,
    teacher_forced: bool = False,
) -> tuple[torch.Tensor, list[dict]]:
    """
    Manual AR loop for 233a v1-full variant.
    Bypasses self.training gate by explicitly choosing feedback source.
    Returns: trajectory (B, K, N, 25), step_stats list[dict].
    """
    assert model.variant == "full", f"Expected full, got {model.variant}"
    B = history_01.shape[0]
    device = history_01.device
    D = model.n_cells   # 25
    K = n_members
    BK = B * K

    # Flatten grid to (B, T, D)
    if history_01.dim() == 4:
        hist_flat = history_01.reshape(B, history_01.shape[1], D)
    else:
        hist_flat = history_01

    if future_01 is not None:
        if future_01.dim() == 4:
            fut_flat = future_01.reshape(B, n_steps, D)
        else:
            fut_flat = future_01
    else:
        fut_flat = None

    # --- Init slow state (B) ---
    state = model.init_slow_state(hist_flat)
    buffer_B = list(state["buffer"])
    h_slow = state["h_slow"]
    s_ewma, lam_hawkes = state["s_ewma"], state["lam_hawkes"]
    s_t, lam_t = state["s"], state["lam"]
    delta_t_last = state["delta_t_last_jump"]
    x_prev_B = hist_flat[:, -1]   # (B, D)

    # --- Init fast state (BK) ---
    cond_B, local_scale_B = model.encode_history(history_01)
    scale_anchor_B = local_scale_B.clone()

    cond = cond_B.unsqueeze(1).expand(B, K, -1).reshape(BK, -1)
    local_scale = local_scale_B.unsqueeze(1).expand(B, K, -1).reshape(BK, -1)
    prev = hist_flat[:, -1].unsqueeze(1).expand(B, K, -1).reshape(BK, D)
    scale_anchor_bk = scale_anchor_B.unsqueeze(1).expand(B, K, -1).reshape(BK, -1)

    z_f = torch.randn(BK, model.factor_rank, device=device)
    rho_sq_comp = math.sqrt(1.0 - model.rho ** 2)

    frames = []
    step_stats = []

    for t in range(n_steps):
        # FiLM from slow state
        film_out = model.film(s_t, lam_t)
        gamma_L_B = film_out["gamma_lambda"]   # (B, k)
        beta_L_B  = film_out["beta_lambda"]
        gamma_D_B = film_out["gamma_d"]         # (B, D)
        beta_D_B  = film_out["beta_d"]
        drift_B   = film_out["drift_bias"]
        logit_B   = film_out["p_jump_logit"]    # (B,)

        gamma_L_bk = gamma_L_B.unsqueeze(1).expand(B, K, model.factor_rank).reshape(BK, model.factor_rank)
        beta_L_bk  = beta_L_B.unsqueeze(1).expand(B, K, model.factor_rank).reshape(BK, model.factor_rank)
        gamma_D_bk = gamma_D_B.unsqueeze(1).expand(B, K, D).reshape(BK, D)
        beta_D_bk  = beta_D_B.unsqueeze(1).expand(B, K, D).reshape(BK, D)
        drift_bk   = drift_B.unsqueeze(1).expand(B, K, D).reshape(BK, D)
        logit_bk   = logit_B.unsqueeze(1).expand(B, K).reshape(BK)

        # Fast-path step
        if t > 0:
            z_f = model.rho * z_f + rho_sq_comp * torch.randn_like(z_f)
        z_i = torch.randn(BK, D, device=device)
        pos = model.pos_embed(t, BK, device)

        factor_in = torch.cat([prev, cond, z_f, pos], dim=-1)
        f_scores = model.factor_head(factor_in)

        idio_in = torch.cat([prev, cond, z_i, pos], dim=-1)
        i_resid = model.idio_head(idio_in)

        Lambda_base = model.get_lambda(cond)
        Lambda_mod  = Lambda_base * gamma_L_bk.unsqueeze(-2) + beta_L_bk.unsqueeze(-2)
        D_base      = model.get_d(cond)
        D_mod       = D_base * gamma_D_bk + beta_D_bk

        factor_contribution = torch.einsum("bdr,br->bd", Lambda_mod, f_scores)
        idio_contribution   = D_mod * i_resid
        v = factor_contribution + idio_contribution + drift_bk

        # Jump mixture
        lam_t_bk = lam_t.unsqueeze(1).expand(B, K).reshape(BK)
        p_j = torch.sigmoid(logit_bk)
        mask = (torch.rand(BK, 1, device=device) < p_j.unsqueeze(-1)).float()
        eps_extra = torch.randn(BK, D, device=device)
        s_jump = model.scale_jump_head(lam_t_bk)   # (BK,)
        v = v + mask * s_jump.unsqueeze(-1) * eps_extra

        delta = torch.sinh(v) * local_scale
        next_iv = (prev + delta).clamp(1e-4, 1.0 - 1e-4)
        frames.append(next_iv)

        step_stats.append({
            "delta_cpu": delta.cpu().float().numpy(),
            "next_iv_cpu": next_iv.cpu().float().numpy(),
            "prev_cpu": prev.cpu().float().numpy(),
            "local_scale_mean": local_scale.mean().item(),
            "local_scale_std": local_scale.std().item(),
            "cond_norm": cond.norm(dim=-1).mean().item(),
            "factor_norm": factor_contribution.abs().mean().item(),
            "idio_norm": idio_contribution.abs().mean().item(),
            "s_t_mean": s_t.mean().item(),
            "lam_t_mean": lam_t.mean().item(),
        })

        # Feedback selection (explicit, no self.training gate)
        if teacher_forced and fut_flat is not None:
            x_feedback_B = fut_flat[:, t]              # (B, D) — GT
            x_feedback_bk = x_feedback_B.unsqueeze(1).expand(B, K, -1).reshape(BK, D)
        else:
            x_feedback_bk = next_iv                    # (BK, D) — self-fed
            x_feedback_B = next_iv.view(B, K, D).mean(dim=1)  # (B, D)

        # Fast-path state update
        feat, local_scale = model._step_features(prev, x_feedback_bk, local_scale)
        cond = model.gru_cell(feat, cond)
        prev = x_feedback_bk

        if model.use_scale_anchor:
            log_s = (
                (1.0 - model.scale_anchor_alpha) * torch.log(local_scale.clamp_min(model.scale_floor))
                + model.scale_anchor_alpha * torch.log(scale_anchor_bk.clamp_min(model.scale_floor))
            )
            local_scale = torch.exp(log_s)

        # Slow-path state update (B, not BK)
        dx_B = x_feedback_B - x_prev_B
        mean_sq_dx = (dx_B ** 2).mean(dim=-1)
        j_t_B = (dx_B.norm(dim=-1) > model.q90_train).float()

        buffer_B.append(x_feedback_B)
        buffer_B = buffer_B[-30:]
        coarse_t = model.coarse(x_feedback_B, buffer_B)

        sp_out = model.slow_path.step(
            coarse_t, h_slow, s_ewma, lam_hawkes, delta_t_last, mean_sq_dx, j_t_B,
        )
        h_slow   = sp_out["h_t"]
        s_ewma   = sp_out["s_ewma_t"]
        lam_hawkes = sp_out["lam_hawkes_t"]
        s_t, lam_t = sp_out["s_t"], sp_out["lam_t"]
        delta_t_last = torch.where(j_t_B.bool(), torch.zeros_like(delta_t_last), delta_t_last + 1.0)
        x_prev_B = x_feedback_B

    traj = torch.stack(frames, dim=1)    # (BK, N, D)
    traj = traj.reshape(B, K, n_steps, D)
    return traj, step_stats


# ─── batched rollout wrapper ──────────────────────────────────────────────────
def batched_rollout(
    rollout_fn,
    model,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    n_members: int,
    n_steps: int,
    batch_size: int,
    teacher_forced: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """
    Run rollout_fn in mini-batches. Returns:
      trajectories: (W, K, N, D) numpy
      step_stats:   list[dict] averaged across batches (one entry per step)
    """
    all_traj = []
    agg_stats: list[dict] | None = None
    n_total = 0

    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        b = end - start
        h_b = history_01[start:end]
        f_b = future_01[start:end]

        traj_b, stats_b = rollout_fn(
            model, h_b, f_b, n_members=n_members,
            n_steps=n_steps, teacher_forced=teacher_forced,
        )
        all_traj.append(traj_b.cpu().numpy())

        # Accumulate stats: concatenate raw numpy arrays, weight-sum scalars
        _CPU_KEYS = {"delta_cpu", "next_iv_cpu", "prev_cpu"}
        if agg_stats is None:
            agg_stats = []
            for s in stats_b:
                entry = {}
                for k, v in s.items():
                    if k in _CPU_KEYS:
                        entry[k] = v.copy()
                    else:
                        entry[k] = v * b
                agg_stats.append(entry)
        else:
            for i, s in enumerate(stats_b):
                for k, v in s.items():
                    if k in _CPU_KEYS:
                        agg_stats[i][k] = np.concatenate([agg_stats[i][k], v], axis=0)
                    else:
                        agg_stats[i][k] += v * b
        n_total += b

    # Average scalar fields
    for entry in agg_stats:
        for k in list(entry.keys()):
            if not k.endswith("_cpu"):
                entry[k] /= n_total

    return np.concatenate(all_traj, axis=0), agg_stats


# ─── per-step aggregation ─────────────────────────────────────────────────────
def compute_per_step_metrics(
    traj_bknd: np.ndarray,       # (W, K, N, D) generated
    gt_wnd: np.ndarray,          # (W, N, D) ground truth
    step_stats: list[dict],      # raw per-step info from rollout
    n_steps: int,
) -> list[dict]:
    """
    Compute scalar per-step metrics:
      - mean_sq_delta: mean((x_{t+1} - x_t)^2)
      - max_abs_delta: max|x_{t+1} - x_t|
      - cross_member_std: std of delta across K members
      - ks_stat: 2-sample KS vs GT deltas at step t
      - lag1_autocorr: corr(delta_t, delta_{t-1}) for t >= 1
    """
    W, K, N, D = traj_bknd.shape
    # Build full trajectory (including t=0 = last history frame) for delta computation
    # We embed the initial state implicitly via step_stats[0]["prev_cpu"]

    rows = []
    for t in range(n_steps):
        raw = step_stats[t]
        delta_bkd = raw["delta_cpu"]    # (W*K, D)
        delta_3d = delta_bkd.reshape(W, K, D)

        # 1. Mean squared delta
        msd = float(np.mean(delta_3d ** 2))

        # 2. Max abs delta (pathwise max over batch × members × cells)
        max_abs = float(np.max(np.abs(delta_3d)))

        # 3. Cross-member std (std over K dimension, mean over W × D)
        cm_std = float(np.mean(np.std(delta_3d, axis=1)))

        # 4. KS vs GT
        # GT delta at step t: gt_wnd[:, t] - (gt_wnd[:, t-1] if t>0 else prev from history)
        if t == 0:
            # prev is the last history frame, stored in step_stats
            prev_cpu = step_stats[0]["prev_cpu"].reshape(W, K, D)[:, 0, :]  # (W, D)
        else:
            prev_cpu = gt_wnd[:, t - 1, :]   # (W, D)
        gt_delta = gt_wnd[:, t, :] - prev_cpu   # (W, D)

        gen_delta_flat = delta_3d.reshape(-1)   # W*K*D
        gt_delta_flat  = gt_delta.reshape(-1)   # W*D
        ks = ks_2samp_np(gen_delta_flat, gt_delta_flat)

        # 5. Lag-1 autocorrelation
        if t == 0:
            lag1_corr = float("nan")
        else:
            prev_delta_3d = step_stats[t - 1]["delta_cpu"].reshape(W, K, D)
            # corr over (W*K*D) population
            x = prev_delta_3d.reshape(-1)
            y = delta_3d.reshape(-1)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() > 10:
                lag1_corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
            else:
                lag1_corr = float("nan")

        rows.append({
            "step": t,
            "mean_sq_delta": msd,
            "max_abs_delta": max_abs,
            "cross_member_std": cm_std,
            "ks_vs_gt": ks,
            "lag1_autocorr": lag1_corr,
            "local_scale_mean": raw["local_scale_mean"],
            "local_scale_std":  raw["local_scale_std"],
            "cond_norm":        raw["cond_norm"],
            "factor_norm":      raw["factor_norm"],
            "idio_norm":        raw["idio_norm"],
        })
    return rows


# ─── markdown writer ──────────────────────────────────────────────────────────
def rows_to_md_table(rows: list[dict], cols: list[str], headers: list[str]) -> list[str]:
    header = "| " + " | ".join(headers) + " |"
    sep    = "| " + " | ".join(["---"] * len(headers)) + " |"
    lines  = [header, sep]
    for r in rows:
        vals = []
        for c in cols:
            v = r.get(c, float("nan"))
            if isinstance(v, float):
                if math.isnan(v):
                    vals.append("nan")
                else:
                    vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return lines


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="AR Compounding Diagnostic")
    parser.add_argument("--model_233a", type=str,
                        default="models/backfill/233a_v1_full_25d_s42/best_model.pt")
    parser.add_argument("--model_229a", type=str,
                        default="models/backfill/factor_ar_229a_wide_decoder/checkpoint_ep30.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--output_dir", type=str, default="results/block_ar/233a")
    parser.add_argument("--n_members", type=int, default=24)
    parser.add_argument("--max_windows", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("Loading validation windows...")
    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.n_steps,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_windows,
        device=device,
        split="val",
    )
    W = batch.history_01.shape[0]
    print(f"  Windows: {W}  Steps: {args.n_steps}  Members: {args.n_members}")

    gt_np = batch.future_01.cpu().numpy().reshape(W, args.n_steps, 25)   # (W, N, 25)

    # ── Load models ───────────────────────────────────────────────────────────
    print("\nLoading 233a v1-full model...")
    model_233a, _ = load_233a(args.model_233a, device)
    model_233a.eval()

    print("Loading 229a model...")
    model_229a, payload_229a = load_227a(args.model_229a, device)
    model_229a.eval()
    print(f"  229a epoch={payload_229a.get('epoch', '?')}")

    results: dict[str, Any] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Block A: 233a — Self-fed
    # ──────────────────────────────────────────────────────────────────────────
    print("\n[1/4] 233a self-fed (inference mode)...")
    traj_233a_sf, stats_233a_sf = batched_rollout(
        rollout_fn=rollout_233a_manual,
        model=model_233a,
        history_01=batch.history_01,
        future_01=batch.future_01,
        n_members=args.n_members,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        teacher_forced=False,
    )
    metrics_233a_sf = compute_per_step_metrics(traj_233a_sf, gt_np, stats_233a_sf, args.n_steps)

    # ──────────────────────────────────────────────────────────────────────────
    # Block B: 233a — Teacher-forced
    # ──────────────────────────────────────────────────────────────────────────
    print("[2/4] 233a teacher-forced...")
    traj_233a_tf, stats_233a_tf = batched_rollout(
        rollout_fn=rollout_233a_manual,
        model=model_233a,
        history_01=batch.history_01,
        future_01=batch.future_01,
        n_members=args.n_members,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        teacher_forced=True,
    )
    metrics_233a_tf = compute_per_step_metrics(traj_233a_tf, gt_np, stats_233a_tf, args.n_steps)

    # ──────────────────────────────────────────────────────────────────────────
    # Block C: 229a — Self-fed
    # ──────────────────────────────────────────────────────────────────────────
    print("[3/4] 229a self-fed (inference mode)...")
    traj_229a_sf, stats_229a_sf = batched_rollout(
        rollout_fn=rollout_227a_manual,
        model=model_229a,
        history_01=batch.history_01,
        future_01=batch.future_01,
        n_members=args.n_members,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        teacher_forced=False,
    )
    metrics_229a_sf = compute_per_step_metrics(traj_229a_sf, gt_np, stats_229a_sf, args.n_steps)

    # ──────────────────────────────────────────────────────────────────────────
    # Block D: 229a — Teacher-forced
    # ──────────────────────────────────────────────────────────────────────────
    print("[4/4] 229a teacher-forced...")
    traj_229a_tf, stats_229a_tf = batched_rollout(
        rollout_fn=rollout_227a_manual,
        model=model_229a,
        history_01=batch.history_01,
        future_01=batch.future_01,
        n_members=args.n_members,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        teacher_forced=True,
    )
    metrics_229a_tf = compute_per_step_metrics(traj_229a_tf, gt_np, stats_229a_tf, args.n_steps)

    # ──────────────────────────────────────────────────────────────────────────
    # Summary statistics
    # ──────────────────────────────────────────────────────────────────────────
    def summarize(rows: list[dict]) -> dict:
        steps = [r["step"] for r in rows]
        ks    = [r["ks_vs_gt"] for r in rows]
        msd   = [r["mean_sq_delta"] for r in rows]
        cmstd = [r["cross_member_std"] for r in rows]
        lag1  = [r["lag1_autocorr"] for r in rows if not math.isnan(r["lag1_autocorr"])]
        max_d = [r["max_abs_delta"] for r in rows]
        return {
            "ks_h1": ks[0] if ks else float("nan"),
            "ks_h5": ks[4] if len(ks) > 4 else float("nan"),
            "ks_h10": ks[9] if len(ks) > 9 else float("nan"),
            "ks_h20": ks[19] if len(ks) > 19 else float("nan"),
            "ks_h30": ks[29] if len(ks) > 29 else float("nan"),
            "ks_mean": float(np.mean(ks)),
            "ks_late_mean": float(np.mean(ks[20:])) if len(ks) > 20 else float("nan"),
            "ks_early_mean": float(np.mean(ks[:5])) if len(ks) >= 5 else float("nan"),
            "ks_monotone_check": bool(ks[-1] > ks[0]) if ks else False,
            "mean_sq_delta_h1": msd[0] if msd else float("nan"),
            "mean_sq_delta_h30": msd[-1] if msd else float("nan"),
            "mean_sq_delta_ratio": msd[-1] / msd[0] if (msd and msd[0] > 1e-12) else float("nan"),
            "cross_member_std_h1": cmstd[0] if cmstd else float("nan"),
            "cross_member_std_h30": cmstd[-1] if cmstd else float("nan"),
            "lag1_autocorr_mean": float(np.mean(lag1)) if lag1 else float("nan"),
            "max_abs_delta_h1": max_d[0] if max_d else float("nan"),
            "max_abs_delta_h30": max_d[-1] if max_d else float("nan"),
        }

    summary = {
        "233a_self_fed": summarize(metrics_233a_sf),
        "233a_teacher_forced": summarize(metrics_233a_tf),
        "229a_self_fed": summarize(metrics_229a_sf),
        "229a_teacher_forced": summarize(metrics_229a_tf),
    }

    results = {
        "summary": summary,
        "per_step": {
            "233a_self_fed": metrics_233a_sf,
            "233a_teacher_forced": metrics_233a_tf,
            "229a_self_fed": metrics_229a_sf,
            "229a_teacher_forced": metrics_229a_tf,
        },
        "meta": {
            "windows": W,
            "n_members": args.n_members,
            "n_steps": args.n_steps,
            "model_233a": args.model_233a,
            "model_229a": args.model_229a,
        },
    }

    # Save JSON
    json_path = out_dir / "_diagnostic_ar_compounding.json"
    json_path.write_text(json.dumps(make_serializable(results), indent=2))
    print(f"\nSaved JSON: {json_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Determine conclusion
    # ──────────────────────────────────────────────────────────────────────────
    s_233a_sf  = summary["233a_self_fed"]
    s_233a_tf  = summary["233a_teacher_forced"]
    s_229a_sf  = summary["229a_self_fed"]
    s_229a_tf  = summary["229a_teacher_forced"]

    # Key diagnostic questions:
    # 1. Does teacher-forcing materially improve KS? (compounding culprit)
    #    Threshold: >0.15 gap between TF and SF at h30 = compounding dominant
    ks_gap_233a = s_233a_sf["ks_h30"] - s_233a_tf["ks_h30"]
    ks_gap_229a = s_229a_sf["ks_h30"] - s_229a_tf["ks_h30"]
    THRESHOLD_COMPOUNDING = 0.15

    # 2. Does TF still show poor KS at h1? (single-step law weak)
    #    Threshold: KS > 0.30 at h1 under TF = single-step law insufficient
    ks_tf_h1_233a = s_233a_tf["ks_h1"]
    ks_tf_h1_229a = s_229a_tf["ks_h1"]
    THRESHOLD_SINGLE_STEP = 0.30

    # 3. Is diversity collapsing? (cross-member std ratio)
    diversity_collapse_233a = s_233a_sf["cross_member_std_h1"] > 0 and \
        (s_233a_sf["cross_member_std_h30"] / s_233a_sf["cross_member_std_h1"]) < 0.5
    diversity_collapse_229a = s_229a_sf["cross_member_std_h1"] > 0 and \
        (s_229a_sf["cross_member_std_h30"] / s_229a_sf["cross_member_std_h1"]) < 0.5

    # 4. Is KS monotonically increasing? (structural degradation)
    ks_growing_233a = s_233a_sf["ks_monotone_check"]
    ks_growing_229a = s_229a_sf["ks_monotone_check"]

    def classify_model(name, ks_gap, ks_tf_h1, diversity_collapse, ks_growing):
        lines = [f"### {name}"]
        if ks_gap > THRESHOLD_COMPOUNDING and ks_tf_h1 < THRESHOLD_SINGLE_STEP:
            verdict = "COMPOUNDING IS PRIMARY CULPRIT"
            explanation = (
                f"Teacher-forced KS at h30 = {(ks_gap_229a if '229a' in name else ks_gap_233a + s_233a_tf['ks_h30']):.3f} "
                f"vs self-fed = {s_233a_sf['ks_h30'] if '233a' in name else s_229a_sf['ks_h30']:.3f} "
                f"(gap {ks_gap:+.3f}). Single-step law is adequate (TF h1 KS={ks_tf_h1:.3f}). "
                "Error accumulates via state drift under self-feeding."
            )
        elif ks_tf_h1 >= THRESHOLD_SINGLE_STEP:
            verdict = "SINGLE-STEP LAW IS WEAK"
            explanation = (
                f"Even under teacher-forcing, h1 KS = {ks_tf_h1:.3f} >= {THRESHOLD_SINGLE_STEP}. "
                "The per-step generation law itself is insufficient regardless of compounding."
            )
        elif ks_gap > THRESHOLD_COMPOUNDING and ks_tf_h1 >= THRESHOLD_SINGLE_STEP:
            verdict = "BOTH: COMPOUNDING + SINGLE-STEP LAW WEAK"
            explanation = (
                f"Teacher-forcing improves KS (gap {ks_gap:+.3f}) but TF itself has poor "
                f"h1 KS = {ks_tf_h1:.3f}. Both factors contribute."
            )
        else:
            # State drift classification
            verdict = "STATE DRIFT (moderate compounding)"
            explanation = (
                f"KS gap TF-SF={ks_gap:.3f} (below {THRESHOLD_COMPOUNDING} threshold). "
                "Compounding effect is moderate; state conditioning may be drifting "
                "to a wrong region under self-feeding."
            )
        lines.append(f"**Verdict: {verdict}**")
        lines.append(explanation)
        if diversity_collapse:
            lines.append("- WARNING: Cross-member diversity collapses by >50% from h1→h30 (rank collapse).")
        if ks_growing:
            lines.append(f"- KS monotonically increases (h1→h30): structural distributional drift confirmed.")
        return "\n".join(lines)

    conclusion_233a = classify_model(
        "233a v1-full", ks_gap_233a, ks_tf_h1_233a,
        diversity_collapse_233a, ks_growing_233a
    )
    conclusion_229a = classify_model(
        "229a", ks_gap_229a, ks_tf_h1_229a,
        diversity_collapse_229a, ks_growing_229a
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Write Markdown report
    # ──────────────────────────────────────────────────────────────────────────
    kscols = ["step", "ks_vs_gt", "mean_sq_delta", "max_abs_delta", "cross_member_std", "lag1_autocorr"]
    kshdrs = ["Step", "KS vs GT", "MSD (δ²)", "Max|δ|", "CM-Std", "Lag1-AutoCorr"]

    md_lines = [
        f"- model_233a: `{args.model_233a}`",
        f"- model_229a: `{args.model_229a}`",
        f"- windows: {W}, members: {args.n_members}, steps: {args.n_steps}",
        "",
        "## Conclusions",
        "",
        conclusion_233a,
        "",
        conclusion_229a,
        "",
        "### Key Numbers at h1/h5/h10/h20/h30",
        "",
        "| Model / Mode | KS@h1 | KS@h5 | KS@h10 | KS@h20 | KS@h30 | CM-Std h1 | CM-Std h30 | Lag1-AutoCorr |",
        "|---|---|---|---|---|---|---|---|---|",
        f"| 233a SF | {s_233a_sf['ks_h1']:.3f} | {s_233a_sf['ks_h5']:.3f} | {s_233a_sf['ks_h10']:.3f} | {s_233a_sf['ks_h20']:.3f} | {s_233a_sf['ks_h30']:.3f} | {s_233a_sf['cross_member_std_h1']:.4f} | {s_233a_sf['cross_member_std_h30']:.4f} | {s_233a_sf['lag1_autocorr_mean']:.4f} |",
        f"| 233a TF | {s_233a_tf['ks_h1']:.3f} | {s_233a_tf['ks_h5']:.3f} | {s_233a_tf['ks_h10']:.3f} | {s_233a_tf['ks_h20']:.3f} | {s_233a_tf['ks_h30']:.3f} | {s_233a_tf['cross_member_std_h1']:.4f} | {s_233a_tf['cross_member_std_h30']:.4f} | {s_233a_tf['lag1_autocorr_mean']:.4f} |",
        f"| 229a SF | {s_229a_sf['ks_h1']:.3f} | {s_229a_sf['ks_h5']:.3f} | {s_229a_sf['ks_h10']:.3f} | {s_229a_sf['ks_h20']:.3f} | {s_229a_sf['ks_h30']:.3f} | {s_229a_sf['cross_member_std_h1']:.4f} | {s_229a_sf['cross_member_std_h30']:.4f} | {s_229a_sf['lag1_autocorr_mean']:.4f} |",
        f"| 229a TF | {s_229a_tf['ks_h1']:.3f} | {s_229a_tf['ks_h5']:.3f} | {s_229a_tf['ks_h10']:.3f} | {s_229a_tf['ks_h20']:.3f} | {s_229a_tf['ks_h30']:.3f} | {s_229a_tf['cross_member_std_h1']:.4f} | {s_229a_tf['cross_member_std_h30']:.4f} | {s_229a_tf['lag1_autocorr_mean']:.4f} |",
        "",
        "### TF vs SF KS Gap at h30 (compounding diagnostic)",
        f"- 233a TF–SF gap: {ks_gap_233a:+.3f}  (threshold={THRESHOLD_COMPOUNDING})",
        f"- 229a TF–SF gap: {ks_gap_229a:+.3f}  (threshold={THRESHOLD_COMPOUNDING})",
        "",
        "## 233a — Self-Fed Per-Step Table",
        "",
    ] + rows_to_md_table(metrics_233a_sf, kscols, kshdrs) + [
        "",
        "## 233a — Teacher-Forced Per-Step Table",
        "",
    ] + rows_to_md_table(metrics_233a_tf, kscols, kshdrs) + [
        "",
        "## 229a — Self-Fed Per-Step Table",
        "",
    ] + rows_to_md_table(metrics_229a_sf, kscols, kshdrs) + [
        "",
        "## 229a — Teacher-Forced Per-Step Table",
        "",
    ] + rows_to_md_table(metrics_229a_tf, kscols, kshdrs) + [
        "",
        "## Interpretation Guide",
        "",
        "- **KS vs GT**: 2-sample KS stat between generated deltas (W×K×25 pool) and GT deltas (W×25 pool) at step t.",
        "  KS=0 is perfect match; KS>0.5 is severe mismatch. Monotone increase → structural error accumulation.",
        "- **MSD (δ²)**: mean squared daily delta. Rising → volatility amplification. Falling → variance collapse.",
        "- **Max|δ|**: pathwise extremes. Declining → innovation attenuation (regime smoothing away tails).",
        "- **CM-Std**: cross-member diversity. Collapsed by h30 → ensemble degeneracy.",
        "- **Lag1-AutoCorr**: positive = AR smoothing dominates; near zero = faithful uncorrelated noise.",
        "",
        "**Compounding vs single-step decision rule:**",
        f"- If TF−SF KS gap > {THRESHOLD_COMPOUNDING} AND TF h1 KS < {THRESHOLD_SINGLE_STEP}: compounding is primary.",
        f"- If TF h1 KS > {THRESHOLD_SINGLE_STEP} regardless of gap: single-step law is the bottleneck.",
        "- If gap is modest (< threshold): state drift is likely — cond/local_scale drift to wrong region.",
    ]

    md_path = out_dir / "_diagnostic_ar_compounding.md"
    write_markdown_summary(str(md_path), "AR Compounding Diagnostic — 233a / 229a", md_lines)
    print(f"Saved MD:   {md_path}")

    # Print summary to console
    print("\n=== AR COMPOUNDING DIAGNOSTIC SUMMARY ===")
    print(f"\n233a v1-full:")
    print(f"  Self-fed   KS: h1={s_233a_sf['ks_h1']:.3f} h10={s_233a_sf['ks_h10']:.3f} h30={s_233a_sf['ks_h30']:.3f}")
    print(f"  Teacher-f  KS: h1={s_233a_tf['ks_h1']:.3f} h10={s_233a_tf['ks_h10']:.3f} h30={s_233a_tf['ks_h30']:.3f}")
    print(f"  TF-SF gap at h30: {ks_gap_233a:+.3f}")
    print(f"  CM-Std h1/h30: {s_233a_sf['cross_member_std_h1']:.4f}/{s_233a_sf['cross_member_std_h30']:.4f}")
    print(f"  MSD ratio h30/h1: {s_233a_sf['mean_sq_delta_ratio']:.2f}")
    print(f"  Verdict: {conclusion_233a.split(chr(10))[1]}")

    print(f"\n229a:")
    print(f"  Self-fed   KS: h1={s_229a_sf['ks_h1']:.3f} h10={s_229a_sf['ks_h10']:.3f} h30={s_229a_sf['ks_h30']:.3f}")
    print(f"  Teacher-f  KS: h1={s_229a_tf['ks_h1']:.3f} h10={s_229a_tf['ks_h10']:.3f} h30={s_229a_tf['ks_h30']:.3f}")
    print(f"  TF-SF gap at h30: {ks_gap_229a:+.3f}")
    print(f"  CM-Std h1/h30: {s_229a_sf['cross_member_std_h1']:.4f}/{s_229a_sf['cross_member_std_h30']:.4f}")
    print(f"  MSD ratio h30/h1: {s_229a_sf['mean_sq_delta_ratio']:.2f}")
    print(f"  Verdict: {conclusion_229a.split(chr(10))[1]}")

    print(f"\nArtifacts:")
    print(f"  {json_path}")
    print(f"  {md_path}")


if __name__ == "__main__":
    main()
