#!/usr/bin/env python
"""
229a Compounding Diagnostic: Teacher-Forced vs Self-Fed vs Wrapper evaluation.

Answers: does the per-step innovation law look wrong at h=1 (capacity problem)
or does it look reasonable early and drift under self-feeding (compounding)?

Three evaluation modes:
  - teacher_forced: real GT drives state updates, model generates predictions
  - self_fed_native: model's own AR loop (forward() with maintained GRUCell state)
  - self_fed_wrapper: OneDayKernelRolloutWrapper (re-encodes sliding window each step)

Per-step intermediate statistics:
  - factor/idio contribution norms
  - local_scale drift
  - GRU hidden state norms
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

from experiments.backfill.block_ar._rollout_220_utils import (
    OneDayKernelRolloutWrapper,
    build_rollout_windows,
    load_one_day_kernel,
    make_serializable,
    rollout_samples_in_batches,
    write_markdown_summary,
)
from experiments.backfill.block_ar.test_block_ar_requirements_v2 import (
    run_ci_coverage_tests,
    run_cross_cell_correlation_tests,
    run_distributional_fidelity_tests,
    run_mean_reversion_tests,
    run_pathwise_jump_realism_tests,
)


@torch.no_grad()
def forward_with_diagnostics(
    model,
    history_01: torch.Tensor,
    future_01: torch.Tensor | None,
    n_members: int,
    n_steps: int,
    teacher_forced: bool = False,
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    """Run AR loop with per-step diagnostic collection.

    Args:
        model: FactorARModel instance
        history_01: (B, 30, 5, 5) in [0,1]
        future_01: (B, n_steps, 5, 5) ground truth (required if teacher_forced)
        n_members: K ensemble members
        n_steps: rollout steps
        teacher_forced: if True, use GT for state updates instead of generated output

    Returns:
        trajectory: (B, K, N, 5, 5)
        step_diagnostics: list of dicts, one per step
    """
    B = history_01.shape[0]
    device = history_01.device

    cond, local_scale = model.encode_history(history_01)
    prev = history_01[:, -1].reshape(B, model.n_cells)

    BK = B * n_members
    cond = cond.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
    local_scale = local_scale.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)
    prev = prev.unsqueeze(1).expand(B, n_members, -1).reshape(BK, -1)

    if teacher_forced and future_01 is not None:
        gt_flat = future_01.reshape(B, n_steps, model.n_cells)
        gt_expanded = (
            gt_flat.unsqueeze(1)
            .expand(B, n_members, n_steps, model.n_cells)
            .reshape(BK, n_steps, model.n_cells)
        )

    z_f = torch.randn(BK, model.factor_rank, device=device)
    rho_sq_comp = math.sqrt(1.0 - model.rho ** 2)

    frames = []
    step_diagnostics = []

    for t in range(n_steps):
        if t > 0:
            z_f = model.rho * z_f + rho_sq_comp * torch.randn_like(z_f)
        z_i = torch.randn(BK, model.n_cells, device=device)

        pos = model.pos_embed(t, BK, device)

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
            cs = F.softplus(model.cell_spread_proj(cond))
            v = v * cs

        delta = torch.sinh(v) * local_scale
        next_iv = (prev + delta).clamp(0.001, 1.0)
        frames.append(next_iv)

        # Collect per-step diagnostics
        factor_norm = factor_contribution.abs().mean().item()
        idio_norm = idio_contribution.abs().mean().item()
        diag = {
            "step": t,
            "factor_norm": factor_norm,
            "idio_norm": idio_norm,
            "factor_ratio": factor_norm / max(factor_norm + idio_norm, 1e-8),
            "local_scale_mean": local_scale.mean().item(),
            "local_scale_std": local_scale.std().item(),
            "cond_norm": cond.norm(dim=-1).mean().item(),
            "delta_abs_mean": delta.abs().mean().item(),
            "v_abs_mean": v.abs().mean().item(),
            "f_scores_std": f_scores.std().item(),
            "i_resid_std": i_resid.std().item(),
            "D_mean": D.mean().item(),
        }
        step_diagnostics.append(diag)

        # State update: teacher-forced uses GT, self-fed uses own output
        if teacher_forced and future_01 is not None:
            gt_step = gt_expanded[:, t]
            feat, local_scale = model._step_features(prev, gt_step, local_scale)
            cond = model.gru_cell(feat, cond)
            prev = gt_step
        else:
            feat, local_scale = model._step_features(prev, next_iv, local_scale)
            cond = model.gru_cell(feat, cond)
            prev = next_iv

    trajectory = torch.stack(frames, dim=0).permute(1, 0, 2)
    trajectory = trajectory.view(B, n_members, n_steps, 5, 5)
    return trajectory, step_diagnostics


def generate_samples(
    model,
    history_01: torch.Tensor,
    future_01: torch.Tensor | None,
    n_members: int,
    n_steps: int,
    batch_size: int,
    teacher_forced: bool = False,
) -> tuple[np.ndarray, list[dict[str, float]]]:
    """Generate samples in batches, collecting diagnostics."""
    outputs = []
    all_diagnostics = None

    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        hist_batch = history_01[start:end]
        fut_batch = future_01[start:end] if future_01 is not None else None

        traj, diags = forward_with_diagnostics(
            model, hist_batch, fut_batch,
            n_members=n_members, n_steps=n_steps,
            teacher_forced=teacher_forced,
        )
        outputs.append(traj.cpu().numpy())

        if all_diagnostics is None:
            all_diagnostics = [{k: v * (end - start) for k, v in d.items() if k != "step"}
                               for d in diags]
            for d, orig in zip(all_diagnostics, diags):
                d["step"] = orig["step"]
        else:
            for i, d in enumerate(diags):
                for k, v in d.items():
                    if k != "step":
                        all_diagnostics[i][k] += v * (end - start)

    # Average diagnostics
    n_total = history_01.shape[0]
    for d in all_diagnostics:
        for k in d:
            if k != "step":
                d[k] /= n_total

    return np.concatenate(outputs, axis=0), all_diagnostics


def generate_wrapper_samples(
    model,
    history_norm: torch.Tensor,
    n_samples: int,
    n_steps: int,
    batch_size: int,
    chunk_size: int = 8,
) -> np.ndarray:
    """Generate samples via OneDayKernelRolloutWrapper (re-encodes each step)."""
    wrapper = OneDayKernelRolloutWrapper(model).eval()
    return rollout_samples_in_batches(
        wrapper=wrapper,
        history_norm=history_norm,
        n_samples=n_samples,
        n_steps=n_steps,
        batch_size=batch_size,
        chunk_size=chunk_size,
    )


def compute_horizon_metrics(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
    horizon: int,
) -> dict[str, Any]:
    """Compute suite metrics at a specific horizon (using data up to that horizon).

    Some tests may fail at very short horizons (h=1) due to degenerate
    correlation matrices. Those metrics are set to NaN.
    """
    samp_h = cond_samples[:, :, :horizon]  # (W, K, h, 5, 5)
    gt_h = ground_truth[:, :horizon]  # (W, h, 5, 5)

    results = {}

    # Coverage
    try:
        cov = run_ci_coverage_tests(samp_h, gt_h)
        last_h = max(cov["per_horizon"].keys()) if cov["per_horizon"] else horizon
        results["coverage_90"] = cov["per_horizon"].get(last_h, {}).get(0.9, float("nan"))
        results["worst_cell_coverage"] = cov["worst_cell_per_horizon"].get(last_h, float("nan"))
    except Exception:
        results["coverage_90"] = float("nan")
        results["worst_cell_coverage"] = float("nan")

    # Distributional fidelity (change KS, level KS)
    try:
        dist = run_distributional_fidelity_tests(samp_h, gt_h, history)
        results["change_ks_pass"] = dist["ks_test"]["n_pass"]
        results["level_ks_pass"] = dist["ks_level_test"]["n_pass"]
        results["median_bias_pass"] = dist["median_bias"]["n_pass"]
        results["worst_floor"] = dist["explosion"]["worst_cell_floor"]
        results["bad_windows_pct"] = dist["window_floor"]["pct_bad"]
    except Exception:
        results["change_ks_pass"] = -1
        results["level_ks_pass"] = -1
        results["median_bias_pass"] = -1
        results["worst_floor"] = float("nan")
        results["bad_windows_pct"] = float("nan")

    # Cross-cell correlation (can fail at h=1 due to degenerate covariance)
    try:
        xcorr = run_cross_cell_correlation_tests(samp_h, gt_h)
        results["corr_ratio"] = xcorr["corr_ratio"]
        results["rank_ratio"] = xcorr["rank_ratio"]
    except Exception:
        results["corr_ratio"] = float("nan")
        results["rank_ratio"] = float("nan")

    # Mean reversion
    try:
        mr = run_mean_reversion_tests(samp_h, gt_h, history)
        results["mr_ratio"] = mr.get("mr_gt_ratio", float("nan"))
    except Exception:
        results["mr_ratio"] = float("nan")

    # Pathwise jump realism
    try:
        jump = run_pathwise_jump_realism_tests(samp_h, gt_h)
        results["jump_ks"] = jump["pathwise_max_jump"]["ks_stat"]
    except Exception:
        results["jump_ks"] = float("nan")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="229a Compounding Diagnostic: Teacher-Forced vs Self-Fed"
    )
    parser.add_argument("--model_type", type=str, default="227a")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=200)
    parser.add_argument("--n_members", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--horizons", type=str, default="1,5,10,20,30")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    horizons = [int(h) for h in args.horizons.split(",")]
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )

    # Load model
    model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    model.eval()
    print(f"Loaded {args.model_type} from {args.checkpoint}")
    print(f"  epoch={payload.get('epoch', '?')}")

    # Build test windows
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
    n_steps = batch.future_01.shape[1]
    print(f"  windows={batch.history_01.shape[0]}, steps={n_steps}")

    # ---- Mode 1: Teacher-Forced ----
    print("\n[1/3] Teacher-forced generation...")
    tf_samples, tf_diagnostics = generate_samples(
        model,
        batch.history_01,
        batch.future_01,
        n_members=args.n_members,
        n_steps=n_steps,
        batch_size=args.batch_size,
        teacher_forced=True,
    )

    # ---- Mode 2: Self-Fed Native (maintained GRUCell state) ----
    print("[2/3] Self-fed native generation...")
    sf_native_samples, sf_diagnostics = generate_samples(
        model,
        batch.history_01,
        None,
        n_members=args.n_members,
        n_steps=n_steps,
        batch_size=args.batch_size,
        teacher_forced=False,
    )

    # ---- Mode 3: Self-Fed Wrapper (re-encodes sliding window) ----
    print("[3/3] Self-fed wrapper generation...")
    sf_wrapper_samples = generate_wrapper_samples(
        model,
        batch.history_norm,
        n_samples=args.n_members,
        n_steps=n_steps,
        batch_size=args.batch_size,
        chunk_size=args.chunk_size,
    )

    ground_truth = batch.future_01.detach().cpu().numpy()
    history_01 = batch.history_01.detach().cpu().numpy()

    # ---- Compute per-horizon metrics for all three modes ----
    print("\nComputing per-horizon metrics...")
    results = {
        "teacher_forced": {},
        "self_fed_native": {},
        "self_fed_wrapper": {},
        "per_step_diagnostics": {
            "teacher_forced": tf_diagnostics,
            "self_fed_native": sf_diagnostics,
        },
    }

    for h in horizons:
        if h > n_steps:
            continue
        print(f"  h={h}...")
        results["teacher_forced"][f"h{h}"] = compute_horizon_metrics(
            tf_samples, ground_truth, history_01, h
        )
        results["self_fed_native"][f"h{h}"] = compute_horizon_metrics(
            sf_native_samples, ground_truth, history_01, h
        )
        results["self_fed_wrapper"][f"h{h}"] = compute_horizon_metrics(
            sf_wrapper_samples, ground_truth, history_01, h
        )

    # ---- Wrapper/Native Gap (Decision Gate) ----
    wrapper_native_gap = {}
    for h in horizons:
        if h > n_steps:
            continue
        key = f"h{h}"
        nat = results["self_fed_native"].get(key, {})
        wrp = results["self_fed_wrapper"].get(key, {})
        wrapper_native_gap[key] = {
            "change_ks_diff": nat.get("change_ks_pass", 0) - wrp.get("change_ks_pass", 0),
            "level_ks_diff": nat.get("level_ks_pass", 0) - wrp.get("level_ks_pass", 0),
            "corr_diff": abs(nat.get("corr_ratio", 0) - wrp.get("corr_ratio", 0)),
            "coverage_diff": nat.get("coverage_90", 0) - wrp.get("coverage_90", 0),
        }
    results["wrapper_native_gap"] = wrapper_native_gap

    # Check decision gate
    max_ks_diff = max(abs(g["change_ks_diff"]) for g in wrapper_native_gap.values())
    max_corr_diff = max(g["corr_diff"] for g in wrapper_native_gap.values())
    gate_triggered = max_ks_diff >= 3 or max_corr_diff >= 0.15
    results["wrapper_native_gate"] = {
        "triggered": gate_triggered,
        "max_change_ks_diff": max_ks_diff,
        "max_corr_diff": max_corr_diff,
        "recommendation": (
            "HARNESS FIX REQUIRED: wrapper/native mismatch is large. "
            "Fix evaluation harness before making compounding claims."
            if gate_triggered else
            "Gate passed: wrapper/native within noise. Proceed with wrapper for comparison."
        ),
    }

    # ---- Degradation Summary ----
    degradation = {}
    for h in horizons:
        if h > n_steps:
            continue
        key = f"h{h}"
        tf = results["teacher_forced"].get(key, {})
        sf = results["self_fed_native"].get(key, {})
        degradation[key] = {
            "change_ks_tf": tf.get("change_ks_pass", 0),
            "change_ks_sf": sf.get("change_ks_pass", 0),
            "change_ks_gap": tf.get("change_ks_pass", 0) - sf.get("change_ks_pass", 0),
            "level_ks_tf": tf.get("level_ks_pass", 0),
            "level_ks_sf": sf.get("level_ks_pass", 0),
            "corr_tf": tf.get("corr_ratio", 0),
            "corr_sf": sf.get("corr_ratio", 0),
            "coverage_tf": tf.get("coverage_90", 0),
            "coverage_sf": sf.get("coverage_90", 0),
        }
    results["degradation_summary"] = degradation

    # ---- Save JSON ----
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(results), indent=2))

    # ---- Save Markdown ----
    lines = [
        f"- model: `{args.model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- windows: {batch.history_01.shape[0]}",
        f"- members: {args.n_members}",
        "",
        "## Wrapper/Native Decision Gate",
        f"- Gate triggered: **{gate_triggered}**",
        f"- Max change KS diff: {max_ks_diff}",
        f"- Max corr diff: {max_corr_diff:.3f}",
        f"- {results['wrapper_native_gate']['recommendation']}",
        "",
        "## Teacher-Forced vs Self-Fed Degradation",
        "",
        "| Horizon | ChgKS (TF) | ChgKS (SF) | Gap | LvlKS (TF) | LvlKS (SF) | Corr (TF) | Corr (SF) | Cov (TF) | Cov (SF) |",
        "|---------|-----------|-----------|-----|-----------|-----------|----------|----------|---------|---------|",
    ]
    for h in horizons:
        if h > n_steps:
            continue
        d = degradation[f"h{h}"]
        lines.append(
            f"| h={h} | {d['change_ks_tf']}/25 | {d['change_ks_sf']}/25 | "
            f"{d['change_ks_gap']:+d} | {d['level_ks_tf']}/25 | {d['level_ks_sf']}/25 | "
            f"{d['corr_tf']:.3f} | {d['corr_sf']:.3f} | "
            f"{d['coverage_tf']:.3f} | {d['coverage_sf']:.3f} |"
        )

    lines += [
        "",
        "## Per-Step Diagnostics (Self-Fed Native)",
        "",
        "| Step | Factor% | |Λ@f| | |D⊙ε| | Scale μ | Scale σ | |cond| | |δ| | |v| |",
        "|------|---------|-------|-------|---------|---------|--------|-----|-----|",
    ]
    for d in sf_diagnostics:
        lines.append(
            f"| {d['step']:2d} | {d['factor_ratio']:.3f} | {d['factor_norm']:.4f} | "
            f"{d['idio_norm']:.4f} | {d['local_scale_mean']:.5f} | "
            f"{d['local_scale_std']:.5f} | {d['cond_norm']:.2f} | "
            f"{d['delta_abs_mean']:.5f} | {d['v_abs_mean']:.4f} |"
        )

    lines += [
        "",
        "## Per-Step Diagnostics (Teacher-Forced)",
        "",
        "| Step | Factor% | |Λ@f| | |D⊙ε| | Scale μ | Scale σ | |cond| | |δ| | |v| |",
        "|------|---------|-------|-------|---------|---------|--------|-----|-----|",
    ]
    for d in tf_diagnostics:
        lines.append(
            f"| {d['step']:2d} | {d['factor_ratio']:.3f} | {d['factor_norm']:.4f} | "
            f"{d['idio_norm']:.4f} | {d['local_scale_mean']:.5f} | "
            f"{d['local_scale_std']:.5f} | {d['cond_norm']:.2f} | "
            f"{d['delta_abs_mean']:.5f} | {d['v_abs_mean']:.4f} |"
        )

    write_markdown_summary(args.output_md, "229a Compounding Diagnostic", lines)
    print(f"\nSaved: {args.output_json}")
    print(f"       {args.output_md}")

    # Print summary
    print("\n=== DEGRADATION SUMMARY ===")
    for h in horizons:
        if h > n_steps:
            continue
        d = degradation[f"h{h}"]
        print(
            f"  h={h:2d}: ChgKS TF={d['change_ks_tf']}/25 SF={d['change_ks_sf']}/25 "
            f"(gap={d['change_ks_gap']:+d})  "
            f"Corr TF={d['corr_tf']:.3f} SF={d['corr_sf']:.3f}"
        )

    print(f"\n=== WRAPPER/NATIVE GATE: {'TRIGGERED' if gate_triggered else 'PASSED'} ===")


if __name__ == "__main__":
    main()
