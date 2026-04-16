#!/usr/bin/env python
"""
230a Causal Decomposition: Scale Collapse vs State Drift.

Runs a controlled intervention matrix on existing checkpoints (no training)
to answer: how much of the self-fed rollout failure is caused by EWMA scale
collapse, how much by recurrent-state drift, and which should be fixed next?

Intervention knobs:
  state_mode:
    - teacher_forced: GT drives state updates (upper bound)
    - native_self_fed: true autonomous rollout
    - wrapper_reencode: re-encode sliding history each step (fresh cond + scale)

  scale_mode:
    - self_fed_scale: normal EWMA update
    - anchor_blend: blend EWMA with initial history-derived anchor in log-space
    - oracle_teacher_forced_scale: inject pre-computed TF scale trajectory

  factor_noise_mode:
    - ar1: AR(1) process z_f = rho*z_f + sqrt(1-rho^2)*eps (native training behavior)
    - iid: fresh z_f each step (matches wrapper evaluation behavior)

Scale interventions are applied BEFORE delta generation (after encode_history
or EWMA update, before sinh(v)*local_scale), so they causally affect output.

All comparable modes use the same pre-generated noise tensors to ensure
differences come from the intervention, not Monte Carlo variance.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import sys
sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    write_markdown_summary,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_ci_coverage_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
)


# ── TF scale pre-computation ───────────────────────────────────────────────

@torch.no_grad()
def precompute_tf_scale_trajectory(
    model, history_01: torch.Tensor, future_01: torch.Tensor, n_steps: int
) -> torch.Tensor:
    """Compute the teacher-forced EWMA scale at each step.

    Returns (W, n_steps, 25). This depends only on GT deltas and the initial
    history-derived scale — no GRU state, no noise, no generated output.
    The EWMA update in _step_features is the complete scale-update law:
      new_scale = alpha * |gt_delta|.clamp_min(floor) + (1-alpha) * scale
    which has no dependence on cond, noise, or model predictions.
    """
    W = history_01.shape[0]
    _, local_scale = model.encode_history(history_01)
    prev = history_01[:, -1].reshape(W, model.n_cells)
    gt_flat = future_01.reshape(W, n_steps, model.n_cells)
    scales = []
    for t in range(n_steps):
        gt_step = gt_flat[:, t]
        _, local_scale = model._step_features(prev, gt_step, local_scale)
        scales.append(local_scale.clone())
        prev = gt_step
    return torch.stack(scales, dim=1)  # (W, n_steps, 25)


# ── Noise pre-generation ───────────────────────────────────────────────────

def pregenerate_noise(
    n_windows: int,
    n_members: int,
    n_steps: int,
    factor_rank: int,
    n_cells: int,
    rho: float,
    device: torch.device,
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Pre-generate all noise tensors for paired comparisons.

    Returns dict with:
      z_f_ar1: (W, K, n_steps, factor_rank) — AR(1) factor noise
      z_f_iid: (W, K, n_steps, factor_rank) — iid factor noise
      z_i:     (W, K, n_steps, n_cells)     — iid idiosyncratic noise
      eps_f:   (W, K, n_steps, factor_rank) — raw factor innovations (for AR(1))
    """
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    # Raw innovations
    eps_f = torch.randn(
        n_windows, n_members, n_steps, factor_rank, device=device, generator=gen
    )
    z_i = torch.randn(
        n_windows, n_members, n_steps, n_cells, device=device, generator=gen
    )

    # iid factor noise = eps_f directly
    z_f_iid = eps_f.clone()

    # AR(1) factor noise
    rho_sq_comp = math.sqrt(1.0 - rho ** 2)
    z_f_ar1 = torch.zeros_like(eps_f)
    z_f_ar1[:, :, 0] = eps_f[:, :, 0]
    for t in range(1, n_steps):
        z_f_ar1[:, :, t] = rho * z_f_ar1[:, :, t - 1] + rho_sq_comp * eps_f[:, :, t]

    return {
        "z_f_ar1": z_f_ar1,
        "z_f_iid": z_f_iid,
        "z_i": z_i,
    }


# ── Core intervention forward ──────────────────────────────────────────────

@torch.no_grad()
def forward_intervention(
    model,
    history_01: torch.Tensor,      # (B, 30, 5, 5)
    future_01: torch.Tensor,       # (B, 30, 5, 5) — always needed for TF modes
    n_members: int,
    n_steps: int,
    z_f_batch: torch.Tensor,       # (B, K, n_steps, factor_rank) — pre-generated
    z_i_batch: torch.Tensor,       # (B, K, n_steps, n_cells) — pre-generated
    state_mode: str,               # teacher_forced | native_self_fed | wrapper_reencode
    scale_mode: str,               # self_fed_scale | anchor_blend | oracle_teacher_forced_scale
    anchor_alpha: float = 0.5,
    tf_scale_traj: torch.Tensor | None = None,  # (B, n_steps, 25) for oracle mode
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Run AR loop with state and scale interventions.

    Scale interventions are applied BEFORE delta generation.
    Returns trajectory (B, K, N, 5, 5) and per-step diagnostics.
    """
    B = history_01.shape[0]
    device = history_01.device
    BK = B * n_members

    # Initial encoding
    cond, local_scale = model.encode_history(history_01)
    prev = history_01[:, -1].reshape(B, model.n_cells)
    s_anchor = local_scale.clone()  # (B, 25) — anchor from real history

    # Expand to BK
    cond = cond.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
    local_scale = local_scale.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
    prev = prev.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
    s_anchor_bk = s_anchor.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)

    # Prepare GT if needed
    if future_01 is not None:
        gt_flat = future_01.reshape(B, n_steps, model.n_cells)
        gt_expanded = (
            gt_flat.unsqueeze(1)
            .expand(B, n_members, n_steps, model.n_cells)
            .reshape(BK, n_steps, model.n_cells)
        )

    # Prepare TF scale trajectory if oracle mode
    if tf_scale_traj is not None:
        tf_scale_expanded = (
            tf_scale_traj.unsqueeze(1)
            .expand(B, n_members, n_steps, model.n_cells)
            .reshape(BK, n_steps, model.n_cells)
        )

    # Reshape noise: (B, K, T, D) → (BK, T, D)
    z_f_flat = z_f_batch.reshape(BK, n_steps, -1)
    z_i_flat = z_i_batch.reshape(BK, n_steps, -1)

    # For wrapper mode: maintain sliding history
    if state_mode == "wrapper_reencode":
        sliding_hist = (
            history_01.unsqueeze(1)
            .expand(B, n_members, 30, 5, 5)
            .reshape(BK, 30, 5, 5)
            .clone()
        )

    initial_scale_mean = local_scale.mean().item()
    frames = []
    step_diagnostics = []

    for t in range(n_steps):
        # ── State source ──
        if state_mode == "wrapper_reencode":
            # Re-encode from sliding history (fresh cond + local_scale)
            cond, local_scale = model.encode_history(sliding_hist)
            prev = sliding_hist[:, -1].reshape(BK, model.n_cells)

        # ── Scale intervention (BEFORE delta generation) ──
        if scale_mode == "anchor_blend":
            log_s = (
                (1.0 - anchor_alpha) * torch.log(local_scale.clamp_min(model.scale_floor))
                + anchor_alpha * torch.log(s_anchor_bk.clamp_min(model.scale_floor))
            )
            local_scale = torch.exp(log_s)
        elif scale_mode == "oracle_teacher_forced_scale":
            local_scale = tf_scale_expanded[:, t]

        # ── Noise ──
        z_f = z_f_flat[:, t]
        z_i = z_i_flat[:, t]

        # ── Positional embedding ──
        pos = model.pos_embed(t, BK, device)

        # ── Factor and idiosyncratic heads ──
        factor_in = torch.cat([prev, cond, z_f, pos], dim=-1)
        f_scores = model.factor_head(factor_in)

        idio_in = torch.cat([prev, cond, z_i, pos], dim=-1)
        i_resid = model.idio_head(idio_in)

        Lambda = model.get_lambda(cond)
        D = model.get_d(cond)
        factor_contribution = torch.einsum("br,bcr->bc", f_scores, Lambda)
        idio_contribution = D * i_resid
        v = factor_contribution + idio_contribution
        if model.noise_skip:
            v = v + torch.tanh(model.noise_skip_proj(z_i))

        if model.cell_spread_enabled:
            cs = torch.nn.functional.softplus(model.cell_spread_proj(cond))
            v = v * cs

        delta = torch.sinh(v) * local_scale
        next_iv = (prev + delta).clamp(0.001, 1.0)
        frames.append(next_iv)

        # ── Collect diagnostics ──
        factor_norm = factor_contribution.abs().mean().item()
        idio_norm = idio_contribution.abs().mean().item()
        cur_scale_mean = local_scale.mean().item()
        diag = {
            "step": t,
            "local_scale_mean": cur_scale_mean,
            "local_scale_std": local_scale.std().item(),
            "scale_collapse_ratio": cur_scale_mean / max(initial_scale_mean, 1e-10),
            "cond_norm": cond.norm(dim=-1).mean().item(),
            "factor_norm": factor_norm,
            "idio_norm": idio_norm,
            "factor_share": factor_norm / max(factor_norm + idio_norm, 1e-8),
            "delta_abs_mean": delta.abs().mean().item(),
            "v_abs_mean": v.abs().mean().item(),
        }
        step_diagnostics.append(diag)

        # ── State update for next step ──
        if state_mode == "teacher_forced":
            gt_step = gt_expanded[:, t]
            feat, local_scale = model._step_features(prev, gt_step, local_scale)
            cond = model.gru_cell(feat, cond)
            prev = gt_step
        elif state_mode == "native_self_fed":
            feat, local_scale = model._step_features(prev, next_iv, local_scale)
            cond = model.gru_cell(feat, cond)
            prev = next_iv
        elif state_mode == "wrapper_reencode":
            # Shift sliding history: drop oldest, append generated frame
            sliding_hist = torch.cat(
                [sliding_hist[:, 1:], next_iv.view(BK, 1, 5, 5)], dim=1
            )
            # local_scale and cond will be re-encoded at next step's start

    trajectory = torch.stack(frames, dim=0).permute(1, 0, 2)
    trajectory = trajectory.view(B, n_members, n_steps, 5, 5)
    return trajectory, step_diagnostics


# ── Batched generation ─────────────────────────────────────────────────────

def generate_intervention_samples(
    model,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    noise: dict[str, torch.Tensor],
    n_members: int,
    n_steps: int,
    batch_size: int,
    state_mode: str,
    scale_mode: str,
    factor_noise_mode: str = "ar1",
    anchor_alpha: float = 0.5,
    tf_scale_traj: torch.Tensor | None = None,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Generate samples in batches with intervention."""
    W = history_01.shape[0]
    z_f_key = "z_f_ar1" if factor_noise_mode == "ar1" else "z_f_iid"
    z_f_all = noise[z_f_key]  # (W, K, T, rank)
    z_i_all = noise["z_i"]    # (W, K, T, 25)

    outputs = []
    all_diagnostics = None

    for start in range(0, W, batch_size):
        end = min(start + batch_size, W)
        bs = end - start

        hist_batch = history_01[start:end]
        fut_batch = future_01[start:end]
        z_f_batch = z_f_all[start:end, :n_members]
        z_i_batch = z_i_all[start:end, :n_members]
        tf_batch = tf_scale_traj[start:end] if tf_scale_traj is not None else None

        traj, diags = forward_intervention(
            model, hist_batch, fut_batch,
            n_members=n_members, n_steps=n_steps,
            z_f_batch=z_f_batch, z_i_batch=z_i_batch,
            state_mode=state_mode, scale_mode=scale_mode,
            anchor_alpha=anchor_alpha, tf_scale_traj=tf_batch,
        )
        outputs.append(traj.cpu().numpy())

        if all_diagnostics is None:
            all_diagnostics = [
                {k: v * bs for k, v in d.items() if k != "step"} for d in diags
            ]
            for d, orig in zip(all_diagnostics, diags):
                d["step"] = orig["step"]
        else:
            for i, d in enumerate(diags):
                for k, v in d.items():
                    if k != "step":
                        all_diagnostics[i][k] += v * bs

    for d in all_diagnostics:
        for k in d:
            if k != "step":
                d[k] /= W

    return np.concatenate(outputs, axis=0), all_diagnostics


# ── Metrics ────────────────────────────────────────────────────────────────

def compute_horizon_metrics(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    """Compute suite metrics at a specific horizon."""
    samp_h = cond_samples[:, :, :horizon]
    gt_h = ground_truth[:, :horizon]
    results = {}

    try:
        cov = run_ci_coverage_tests(samp_h, gt_h)
        last_h = max(cov["per_horizon"].keys()) if cov["per_horizon"] else horizon
        results["coverage_90"] = cov["per_horizon"].get(last_h, {}).get(0.9, float("nan"))
        results["worst_cell_coverage"] = cov["worst_cell_per_horizon"].get(last_h, float("nan"))
    except Exception:
        results["coverage_90"] = float("nan")
        results["worst_cell_coverage"] = float("nan")

    try:
        dist = run_distributional_fidelity_tests(samp_h, gt_h, history)
        results["change_ks_pass"] = dist["ks_test"]["n_pass"]
        results["level_ks_pass"] = dist["ks_level_test"]["n_pass"]
        results["bad_windows_pct"] = dist["window_floor"]["pct_bad"]
    except Exception:
        results["change_ks_pass"] = -1
        results["level_ks_pass"] = -1
        results["bad_windows_pct"] = float("nan")

    try:
        xcorr = run_cross_cell_correlation_tests(samp_h, gt_h)
        results["corr_ratio"] = xcorr["corr_ratio"]
        results["rank_ratio"] = xcorr["rank_ratio"]
    except Exception:
        results["corr_ratio"] = float("nan")
        results["rank_ratio"] = float("nan")

    try:
        mr = run_mean_reversion_tests(samp_h, gt_h, history)
        results["mr_ratio"] = mr.get("mr_gt_ratio", float("nan"))
    except Exception:
        results["mr_ratio"] = float("nan")

    try:
        jump = run_pathwise_jump_realism_tests(samp_h, gt_h)
        results["jump_ks"] = jump["pathwise_max_jump"]["ks_stat"]
    except Exception:
        results["jump_ks"] = float("nan")

    return results


def compute_turb_calm_ratio(
    samples: np.ndarray, history_01: np.ndarray
) -> float:
    """Turb/calm CI width ratio from pre-generated samples.

    Splits windows by history realized vol (top/bottom quartile).
    Returns ratio of 90-10 percentile widths.
    """
    W = history_01.shape[0]
    hist_flat = history_01.reshape(W, 30, 25)
    hist_changes = np.diff(hist_flat, axis=1)
    realized_vol = np.mean(np.abs(hist_changes), axis=(1, 2))  # (W,)

    q25, q75 = np.percentile(realized_vol, [25, 75])
    calm_mask = realized_vol <= q25
    turb_mask = realized_vol >= q75

    if calm_mask.sum() < 5 or turb_mask.sum() < 5:
        return float("nan")

    p95 = np.percentile(samples, 95, axis=1)  # (W, T, 5, 5)
    p05 = np.percentile(samples, 5, axis=1)
    width = p95 - p05

    calm_width = width[calm_mask].mean()
    turb_width = width[turb_mask].mean()
    return float(turb_width / max(calm_width, 1e-8))


def compute_catastrophic_pct(
    samples: np.ndarray, ground_truth: np.ndarray
) -> float:
    """Percentage of windows with < 50% coverage."""
    W, K, T = samples.shape[:3]
    gt_expanded = ground_truth[:, np.newaxis]  # (W, 1, T, 5, 5)
    lo = np.percentile(samples, 5, axis=1, keepdims=True)
    hi = np.percentile(samples, 95, axis=1, keepdims=True)
    in_ci = (gt_expanded >= lo) & (gt_expanded <= hi)
    per_window_cov = in_ci.mean(axis=(1, 2, 3, 4))  # (W,)
    return float((per_window_cov < 0.5).mean() * 100)


# ── Mode definitions ───────────────────────────────────────────────────────

MODES = [
    {"tag": "m1_tf_self",         "state_mode": "teacher_forced",    "scale_mode": "self_fed_scale",               "factor_noise_mode": "ar1", "anchor_alpha": None, "label": "Teacher-forced (upper bound)"},
    {"tag": "m2_native_self",     "state_mode": "native_self_fed",   "scale_mode": "self_fed_scale",               "factor_noise_mode": "ar1", "anchor_alpha": None, "label": "Native self-fed (baseline)"},
    {"tag": "m3_native_anc025",   "state_mode": "native_self_fed",   "scale_mode": "anchor_blend",                 "factor_noise_mode": "ar1", "anchor_alpha": 0.25, "label": "Native + anchor(0.25)"},
    {"tag": "m4_native_anc050",   "state_mode": "native_self_fed",   "scale_mode": "anchor_blend",                 "factor_noise_mode": "ar1", "anchor_alpha": 0.50, "label": "Native + anchor(0.50)"},
    {"tag": "m5_native_anc075",   "state_mode": "native_self_fed",   "scale_mode": "anchor_blend",                 "factor_noise_mode": "ar1", "anchor_alpha": 0.75, "label": "Native + anchor(0.75)"},
    {"tag": "m6_native_oracle",   "state_mode": "native_self_fed",   "scale_mode": "oracle_teacher_forced_scale",  "factor_noise_mode": "ar1", "anchor_alpha": None, "label": "Native + oracle TF scale"},
    {"tag": "m7_wrapper_self",    "state_mode": "wrapper_reencode",  "scale_mode": "self_fed_scale",               "factor_noise_mode": "iid", "anchor_alpha": None, "label": "Wrapper-path (re-encode+iid)"},
    {"tag": "m8_wrapper_anc050",  "state_mode": "wrapper_reencode",  "scale_mode": "anchor_blend",                 "factor_noise_mode": "iid", "anchor_alpha": 0.50, "label": "Wrapper-path + anchor(0.50)"},
    {"tag": "m9_wrapper_oracle",  "state_mode": "wrapper_reencode",  "scale_mode": "oracle_teacher_forced_scale",  "factor_noise_mode": "iid", "anchor_alpha": None, "label": "Wrapper-path + oracle TF scale"},
]


# ── Plotting ───────────────────────────────────────────────────────────────

def plot_metric_vs_horizon(
    all_results: dict[str, dict],
    metric_key: str,
    horizons: list[int],
    title: str,
    ylabel: str,
    save_path: str,
):
    """Line plot of a metric across horizons for all modes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODES)))

    for mode, color in zip(MODES, colors):
        tag = mode["tag"]
        if tag not in all_results:
            continue
        vals = []
        for h in horizons:
            v = all_results[tag].get(f"h{h}", {}).get(metric_key, float("nan"))
            vals.append(v if v != -1 else float("nan"))
        ax.plot(horizons, vals, "o-", label=mode["label"], color=color, markersize=5)

    ax.set_xlabel("Horizon")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_step_trajectory(
    all_diagnostics: dict[str, list[dict]],
    metric_key: str,
    title: str,
    ylabel: str,
    save_path: str,
):
    """Line plot of a per-step diagnostic across all modes."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODES)))

    for mode, color in zip(MODES, colors):
        tag = mode["tag"]
        if tag not in all_diagnostics:
            continue
        steps = [d["step"] for d in all_diagnostics[tag]]
        vals = [d.get(metric_key, float("nan")) for d in all_diagnostics[tag]]
        ax.plot(steps, vals, "-", label=mode["label"], color=color, linewidth=1.5)

    ax.set_xlabel("AR Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


# ── Decision logic ─────────────────────────────────────────────────────────

def compute_decision(
    all_results: dict[str, dict], horizons: list[int]
) -> dict[str, Any]:
    """Apply the hard decision rules from the experiment spec."""

    def get_chg(tag: str, h: int) -> int:
        return all_results.get(tag, {}).get(f"h{h}", {}).get("change_ks_pass", -1)

    def get_cov(tag: str, h: int) -> float:
        return all_results.get(tag, {}).get(f"h{h}", {}).get("coverage_90", float("nan"))

    def get_corr(tag: str, h: int) -> float:
        return all_results.get(tag, {}).get(f"h{h}", {}).get("corr_ratio", float("nan"))

    def get_rank(tag: str, h: int) -> float:
        return all_results.get(tag, {}).get(f"h{h}", {}).get("rank_ratio", float("nan"))

    def get_jump(tag: str, h: int) -> float:
        return all_results.get(tag, {}).get(f"h{h}", {}).get("jump_ks", float("nan"))

    memo = {}

    # Q1: Does oracle scale recover most of the gap?
    eval_horizons = [h for h in [10, 20, 30] if h <= max(horizons)]
    q1_recoveries = []
    for h in eval_horizons:
        tf = get_chg("m1_tf_self", h)
        native = get_chg("m2_native_self", h)
        oracle = get_chg("m6_native_oracle", h)
        gap = tf - native
        if gap > 0:
            recovery = (oracle - native) / gap
        else:
            recovery = float("nan")
        q1_recoveries.append({"horizon": h, "tf": tf, "native": native,
                               "oracle": oracle, "gap": gap, "recovery": recovery})
    memo["Q1_oracle_scale_recovery"] = q1_recoveries

    # Q2: Wrapper vs anchor
    q2_comparisons = []
    for h in eval_horizons:
        native = get_chg("m2_native_self", h)
        wrapper = get_chg("m7_wrapper_self", h)
        anchor = get_chg("m4_native_anc050", h)
        q2_comparisons.append({
            "horizon": h, "native": native, "wrapper": wrapper, "anchor050": anchor,
            "wrapper_gain": wrapper - native, "anchor_gain": anchor - native,
        })
    memo["Q2_wrapper_vs_anchor"] = q2_comparisons

    # Q3: Combined saturation
    q3_comparisons = []
    for h in eval_horizons:
        tf = get_chg("m1_tf_self", h)
        combined = get_chg("m8_wrapper_anc050", h)
        oracle = get_chg("m6_native_oracle", h)
        q3_comparisons.append({
            "horizon": h, "tf": tf, "combined": combined, "oracle": oracle,
            "remaining_gap": tf - combined,
        })
    memo["Q3_combined_saturation"] = q3_comparisons

    # Decision with broadened criteria (not just h30 change-KS)
    # Require improvement across h=10,20,30 AND no regression on other metrics
    h30 = 30 if 30 in horizons else max(horizons)

    tf_chg = get_chg("m1_tf_self", h30)
    native_chg = get_chg("m2_native_self", h30)
    total_gap = tf_chg - native_chg

    anchor_chg = get_chg("m4_native_anc050", h30)
    wrapper_chg = get_chg("m7_wrapper_self", h30)
    oracle_chg = get_chg("m6_native_oracle", h30)
    combined_chg = get_chg("m8_wrapper_anc050", h30)

    anchor_gain = anchor_chg - native_chg
    wrapper_gain = wrapper_chg - native_chg
    oracle_gain = oracle_chg - native_chg

    # Check regressions for anchor and wrapper
    def check_no_regression(tag: str) -> dict:
        issues = []
        for h in eval_horizons:
            native_cov = get_cov("m2_native_self", h)
            mode_cov = get_cov(tag, h)
            native_corr = get_corr("m2_native_self", h)
            mode_corr = get_corr(tag, h)
            native_rank = get_rank("m2_native_self", h)
            mode_rank = get_rank(tag, h)
            native_jump = get_jump("m2_native_self", h)
            mode_jump = get_jump(tag, h)

            if not math.isnan(mode_cov) and not math.isnan(native_cov):
                if mode_cov < native_cov - 0.05:
                    issues.append(f"h{h}: coverage regressed {native_cov:.3f}→{mode_cov:.3f}")
            if not math.isnan(mode_corr) and not math.isnan(native_corr):
                if abs(mode_corr - 1.0) > abs(native_corr - 1.0) + 0.15:
                    issues.append(f"h{h}: corr regressed {native_corr:.3f}→{mode_corr:.3f}")
            if not math.isnan(mode_jump) and not math.isnan(native_jump):
                if mode_jump > native_jump + 0.1:
                    issues.append(f"h{h}: jump regressed {native_jump:.3f}→{mode_jump:.3f}")
        return {"clean": len(issues) == 0, "issues": issues}

    anchor_reg = check_no_regression("m4_native_anc050")
    wrapper_reg = check_no_regression("m7_wrapper_self")

    # Multi-horizon consistency for anchor
    anchor_consistent = all(
        get_chg("m4_native_anc050", h) > get_chg("m2_native_self", h)
        for h in eval_horizons
    )

    # Apply decision rules
    if total_gap <= 0:
        decision = "D"
        reason = "No TF-to-native gap to recover."
    elif oracle_gain <= 2:
        decision = "D"
        reason = f"Oracle scale recovery only {oracle_gain} cells — scale is not the bottleneck."
    elif (anchor_gain >= 0.6 * total_gap and anchor_consistent and anchor_reg["clean"]
          and wrapper_gain < anchor_gain + 2):
        decision = "A"
        reason = (f"Anchor recovers {anchor_gain}/{total_gap} cells ({anchor_gain/total_gap:.0%}), "
                  f"consistent across horizons, no regressions. Wrapper adds only {wrapper_gain - anchor_gain} extra.")
    elif (wrapper_gain >= anchor_gain + 2 and wrapper_reg["clean"]):
        decision = "B"
        reason = (f"Wrapper gains {wrapper_gain} vs anchor {anchor_gain} — "
                  f"state/re-encoding is the larger lever.")
    elif combined_chg >= tf_chg - 3:
        decision = "C"
        reason = (f"Neither alone sufficient (anchor={anchor_gain}, wrapper={wrapper_gain}), "
                  f"but combined={combined_chg} nearly matches TF={tf_chg}.")
    elif oracle_gain >= 0.6 * total_gap and not anchor_reg["clean"]:
        decision = "A*"
        reason = (f"Scale is dominant (oracle recovers {oracle_gain}/{total_gap}), "
                  f"but anchor has regressions: {anchor_reg['issues']}. "
                  f"Need smarter anchor (learnable, not fixed).")
    else:
        decision = "C"
        reason = (f"Mixed: anchor={anchor_gain}, wrapper={wrapper_gain}, "
                  f"combined={combined_chg}, TF={tf_chg}. Both needed.")

    memo["decision"] = decision
    memo["reason"] = reason
    memo["h30_summary"] = {
        "tf": tf_chg, "native": native_chg, "gap": total_gap,
        "anchor050": anchor_chg, "wrapper": wrapper_chg,
        "oracle": oracle_chg, "combined": combined_chg,
    }
    memo["anchor_regression_check"] = anchor_reg
    memo["wrapper_regression_check"] = wrapper_reg
    memo["anchor_multi_horizon_consistent"] = anchor_consistent

    return memo


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="230a Causal Decomposition: Scale Collapse vs State Drift"
    )
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True)
    parser.add_argument("--checkpoint_tags", type=str, nargs="+", required=True)
    parser.add_argument("--model_type", type=str, default="227a")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=200)
    parser.add_argument("--n_members", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--horizons", type=str, default="1,5,10,20,30")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert len(args.checkpoints) == len(args.checkpoint_tags), \
        "Must provide same number of checkpoints and tags"

    horizons = [int(h) for h in args.horizons.split(",")]
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_windows,
        device=device,
        split="val",
    )
    W = batch.history_01.shape[0]
    n_steps = batch.future_01.shape[1]
    ground_truth = batch.future_01.detach().cpu().numpy()
    history_np = batch.history_01.detach().cpu().numpy()
    print(f"Data: {W} windows, {n_steps} steps")

    # Consolidated results
    consolidated = {}

    for ckpt_path, ckpt_tag in zip(args.checkpoints, args.checkpoint_tags):
        print(f"\n{'='*60}")
        print(f"CHECKPOINT: {ckpt_tag}")
        print(f"  path: {ckpt_path}")
        print(f"{'='*60}")

        model, payload = load_one_day_kernel(args.model_type, ckpt_path, device)
        model.eval()
        print(f"  epoch={payload.get('epoch', '?')}")

        # Pre-generate noise (paired across all modes for this checkpoint)
        noise = pregenerate_noise(
            n_windows=W,
            n_members=args.n_members,
            n_steps=n_steps,
            factor_rank=model.factor_rank,
            n_cells=model.n_cells,
            rho=model.rho,
            device=device,
            seed=args.seed,
        )
        print(f"  Pre-generated noise: z_f_ar1={noise['z_f_ar1'].shape}, z_i={noise['z_i'].shape}")

        # Pre-compute TF scale trajectory
        print("  Computing TF scale trajectory...")
        tf_scale = precompute_tf_scale_trajectory(
            model, batch.history_01, batch.future_01, n_steps
        )
        print(f"  TF scale shape: {tf_scale.shape}, "
              f"mean(s0)={tf_scale[:, 0].mean():.5f}, mean(s29)={tf_scale[:, -1].mean():.5f}")

        # Sanity: verify TF scale is purely from _step_features
        # (no GRU/noise dependence) by checking first and last are different
        # but magnitude is preserved (no collapse)
        s0_mean = tf_scale[:, 0].mean().item()
        s29_mean = tf_scale[:, -1].mean().item()
        tf_drift = abs(s29_mean / s0_mean - 1.0)
        if tf_drift > 0.1:
            print(f"  WARNING: TF scale drifted {tf_drift:.1%} ({s0_mean:.5f} -> {s29_mean:.5f}). "
                  f"Expected <10%. Check _step_features for hidden dependencies.")

        ckpt_results = {}
        ckpt_diagnostics = {}

        for mode in MODES:
            tag = mode["tag"]
            print(f"\n  [{tag}] {mode['label']}...")

            samples, step_diags = generate_intervention_samples(
                model=model,
                history_01=batch.history_01,
                future_01=batch.future_01,
                noise=noise,
                n_members=args.n_members,
                n_steps=n_steps,
                batch_size=args.batch_size,
                state_mode=mode["state_mode"],
                scale_mode=mode["scale_mode"],
                factor_noise_mode=mode["factor_noise_mode"],
                anchor_alpha=mode["anchor_alpha"] or 0.5,
                tf_scale_traj=tf_scale if mode["scale_mode"] == "oracle_teacher_forced_scale" else None,
            )

            # Compute horizon metrics
            mode_results = {}
            for h in horizons:
                if h > n_steps:
                    continue
                metrics = compute_horizon_metrics(samples, ground_truth, history_np, h)
                # Add turb/calm and catastrophic
                samp_h = samples[:, :, :h]
                gt_h = ground_truth[:, :h]
                metrics["turb_calm_ratio"] = compute_turb_calm_ratio(samp_h, history_np)
                metrics["catastrophic_pct"] = compute_catastrophic_pct(samp_h, gt_h)
                mode_results[f"h{h}"] = metrics

            ckpt_results[tag] = mode_results
            ckpt_diagnostics[tag] = step_diags

            # Print summary
            for h in horizons:
                if h > n_steps:
                    continue
                m = mode_results[f"h{h}"]
                chg = m.get("change_ks_pass", -1)
                lvl = m.get("level_ks_pass", -1)
                cov = m.get("coverage_90", float("nan"))
                corr = m.get("corr_ratio", float("nan"))
                jump = m.get("jump_ks", float("nan"))
                tc = m.get("turb_calm_ratio", float("nan"))
                print(f"    h={h:2d}: ChgKS={chg:3d}/25 LvlKS={lvl:3d}/25 "
                      f"cov={cov:.3f} corr={corr:.3f} jump={jump:.3f} tc={tc:.3f}")

        # Decision memo
        decision = compute_decision(ckpt_results, horizons)
        print(f"\n  DECISION: {decision['decision']} — {decision['reason']}")

        # Save per-checkpoint
        ckpt_output = {
            "checkpoint": ckpt_path,
            "tag": ckpt_tag,
            "n_windows": W,
            "n_members": args.n_members,
            "horizons": horizons,
            "results": ckpt_results,
            "per_step_diagnostics": ckpt_diagnostics,
            "decision_memo": decision,
        }
        ckpt_json = out_dir / f"{ckpt_tag}.json"
        ckpt_json.write_text(json.dumps(make_serializable(ckpt_output), indent=2))
        print(f"  Saved: {ckpt_json}")

        consolidated[ckpt_tag] = ckpt_output

        # Plots
        for metric, ylabel in [
            ("change_ks_pass", "Change KS pass (/25)"),
            ("coverage_90", "90% Coverage"),
            ("corr_ratio", "Correlation Ratio"),
            ("jump_ks", "Jump KS"),
        ]:
            plot_metric_vs_horizon(
                ckpt_results, metric, horizons,
                f"{ckpt_tag}: {ylabel} vs Horizon",
                ylabel,
                str(out_dir / f"{metric}_vs_horizon_{ckpt_tag}.png"),
            )

        for metric, ylabel in [
            ("local_scale_mean", "Mean local_scale"),
            ("scale_collapse_ratio", "Scale collapse ratio (s_t / s_0)"),
            ("cond_norm", "||cond||"),
            ("factor_share", "Factor share"),
            ("delta_abs_mean", "Mean |delta|"),
        ]:
            plot_step_trajectory(
                ckpt_diagnostics, metric,
                f"{ckpt_tag}: {ylabel} over AR steps",
                ylabel,
                str(out_dir / f"{metric}_trajectory_{ckpt_tag}.png"),
            )

    # ── Consolidated markdown summary ──────────────────────────────────────

    lines = []
    for ckpt_tag, data in consolidated.items():
        lines += [f"## {ckpt_tag}", ""]

        # Main comparison table at h=30
        h_key = f"h{max(horizons)}"
        lines += [
            f"### Comparison at h={max(horizons)}",
            "",
            "| Mode | ChgKS | LvlKS | Cov90 | Corr | Rank | MR | Jump | T/C |",
            "|------|-------|-------|-------|------|------|----|------|-----|",
        ]
        for mode in MODES:
            tag = mode["tag"]
            m = data["results"].get(tag, {}).get(h_key, {})
            lines.append(
                f"| {mode['label'][:35]:35s} | "
                f"{m.get('change_ks_pass', -1):2d}/25 | "
                f"{m.get('level_ks_pass', -1):2d}/25 | "
                f"{m.get('coverage_90', float('nan')):.3f} | "
                f"{m.get('corr_ratio', float('nan')):.3f} | "
                f"{m.get('rank_ratio', float('nan')):.3f} | "
                f"{m.get('mr_ratio', float('nan')):.3f} | "
                f"{m.get('jump_ks', float('nan')):.3f} | "
                f"{m.get('turb_calm_ratio', float('nan')):.3f} |"
            )

        # Multi-horizon change KS
        lines += [
            "",
            "### Change KS across horizons",
            "",
            "| Mode | h=1 | h=5 | h=10 | h=20 | h=30 |",
            "|------|-----|-----|------|------|------|",
        ]
        for mode in MODES:
            tag = mode["tag"]
            vals = []
            for h in horizons:
                chg = data["results"].get(tag, {}).get(f"h{h}", {}).get("change_ks_pass", -1)
                vals.append(f"{chg:2d}" if chg >= 0 else " ?")
            lines.append(f"| {mode['label'][:35]:35s} | " + " | ".join(vals) + " |")

        # Scale trajectory summary
        lines += [
            "",
            "### Scale collapse ratio (s_t / s_0) at selected steps",
            "",
            "| Mode | t=0 | t=5 | t=10 | t=20 | t=29 |",
            "|------|-----|-----|------|------|------|",
        ]
        for mode in MODES:
            tag = mode["tag"]
            diags = data["per_step_diagnostics"].get(tag, [])
            if not diags:
                continue
            vals = []
            for step in [0, 5, 10, 20, 29]:
                if step < len(diags):
                    vals.append(f"{diags[step].get('scale_collapse_ratio', float('nan')):.3f}")
                else:
                    vals.append("?")
            lines.append(f"| {mode['label'][:35]:35s} | " + " | ".join(vals) + " |")

        # Decision memo
        memo = data["decision_memo"]
        lines += [
            "",
            "### Decision Memo",
            "",
            f"**Decision: {memo['decision']}**",
            f"**Reason:** {memo['reason']}",
            "",
        ]
        h30s = memo.get("h30_summary", {})
        lines += [
            f"h30 summary: TF={h30s.get('tf')}, native={h30s.get('native')}, "
            f"gap={h30s.get('gap')}, anchor0.50={h30s.get('anchor050')}, "
            f"wrapper={h30s.get('wrapper')}, oracle={h30s.get('oracle')}, "
            f"combined={h30s.get('combined')}",
        ]

        # Q1-Q3 details
        for qi, label in [
            ("Q1_oracle_scale_recovery", "Q1: Oracle scale recovery"),
            ("Q2_wrapper_vs_anchor", "Q2: Wrapper vs Anchor"),
            ("Q3_combined_saturation", "Q3: Combined saturation"),
        ]:
            lines += ["", f"**{label}:**"]
            for entry in memo.get(qi, []):
                lines.append(f"  h={entry['horizon']}: {entry}")

        lines += ["", "---", ""]

    write_markdown_summary(
        str(out_dir / "230a_summary.md"),
        "230a Causal Decomposition: Scale Collapse vs State Drift",
        lines,
    )

    # Save consolidated JSON
    (out_dir / "230a_consolidated.json").write_text(
        json.dumps(make_serializable(consolidated), indent=2)
    )

    print(f"\nAll outputs saved to {out_dir}/")
    print("Files:")
    for f in sorted(out_dir.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
