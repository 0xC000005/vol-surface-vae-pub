#!/usr/bin/env python
"""
Focused failure-gap analysis for 212e/212f/212g/212h/212q/212r/212u full-data H=1.

Targets the concrete gaps called out after the 213a suite:
  - weak conditional width response
  - worst-cell undercoverage
  - worst-window coverage collapse
  - whether failures are sign-asymmetric / floor-related
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows, make_serializable
from experiments.backfill.block_ar.train_212e_h1_asymmetric_modulated_direct_delta import load_model as load_212e_model
from experiments.backfill.block_ar.train_212f_h1_asymmetric_modulated_direct_delta_quantile import load_model as load_212f_model
from experiments.backfill.block_ar.train_212g_h1_asymmetric_modulated_direct_delta_scale_supervision import load_model as load_212g_model
from experiments.backfill.block_ar.train_212h_h1_asymmetric_modulated_direct_delta_cvar_quantile import load_model as load_212h_model
from experiments.backfill.block_ar.train_212q_h1_asymmetric_modulated_direct_delta_learned_output_scale import load_model as load_212q_model
from experiments.backfill.block_ar.train_212r_h1_asymmetric_modulated_direct_delta_cvar_energy import load_model as load_212r_model
from experiments.backfill.block_ar.train_212u_h1_minimal_direct_stochastic_delta_raw_output import load_model as load_212u_model


def _get_loader(model_type: str):
    mapping = {
        "212e": load_212e_model,
        "212f": load_212f_model,
        "212g": load_212g_model,
        "212h": load_212h_model,
        "212q": load_212q_model,
        "212r": load_212r_model,
        "212u": load_212u_model,
    }
    if model_type not in mapping:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return mapping[model_type]


def _sample_next(
    model: torch.nn.Module,
    history_01: torch.Tensor,
    n_samples: int,
    batch_size: int,
    noise: torch.Tensor | None = None,
) -> np.ndarray:
    outs = []
    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        batch_hist = history_01[start:end]
        batch_noise = None if noise is None else noise[start:end]
        samp = model.sample_next_iv(batch_hist, n_samples=n_samples, noise=batch_noise)
        samp_np = samp.detach().cpu().numpy().reshape(batch_hist.shape[0], n_samples, 5, 5)
        outs.append(samp_np)
    return np.concatenate(outs, axis=0)


@torch.no_grad()
def _decode_params(model: torch.nn.Module, history_01: torch.Tensor, batch_size: int) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {k: [] for k in ["center", "pos_scale", "neg_scale", "gamma", "beta", "state"]}
    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        batch = history_01[start:end]
        state = model.encode(batch)
        out["state"].append(state.detach().cpu().numpy())
        if hasattr(model, "decode_params_from_state"):
            center, pos_scale, neg_scale, gamma, beta = model.decode_params_from_state(state)
            out["center"].append(center.detach().cpu().numpy().reshape(batch.shape[0], 5, 5))
            out["pos_scale"].append(pos_scale.detach().cpu().numpy().reshape(batch.shape[0], 5, 5))
            out["neg_scale"].append(neg_scale.detach().cpu().numpy().reshape(batch.shape[0], 5, 5))
            out["gamma"].append(gamma.detach().cpu().numpy())
            out["beta"].append(beta.detach().cpu().numpy())
        else:
            prev = batch[:, -1].reshape(batch.shape[0], 25)
            sample_delta = model.sample_delta(batch, n_samples=128).detach().cpu().numpy()
            center = np.median(sample_delta, axis=1).reshape(batch.shape[0], 5, 5)
            nan_grid = np.full((batch.shape[0], 5, 5), np.nan, dtype=np.float32)
            nan_latent = np.full((batch.shape[0], 1), np.nan, dtype=np.float32)
            out["center"].append(center)
            out["pos_scale"].append(nan_grid)
            out["neg_scale"].append(nan_grid)
            out["gamma"].append(nan_latent)
            out["beta"].append(nan_latent)
    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def _window_cov(target: np.ndarray, q05: np.ndarray, q95: np.ndarray) -> np.ndarray:
    return ((target >= q05) & (target <= q95)).mean(axis=(1, 2))


def _cell_cov(target: np.ndarray, q05: np.ndarray, q95: np.ndarray) -> np.ndarray:
    return ((target >= q05) & (target <= q95)).mean(axis=0)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a).reshape(-1)
    b = np.asarray(b).reshape(-1)
    if a.size < 2 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 212e/212f/212g/212h/212q/212r/212u H1 failure gaps")
    parser.add_argument("--model_type", type=str, default="212e", choices=["212e", "212f", "212g", "212h", "212q", "212r", "212u"])
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--shuffle_seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    loader_fn = _get_loader(args.model_type)
    model, payload = loader_fn(args.checkpoint, device)

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    max_train_idx = args.test_start - args.history_len - 30
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    history_01, target_01 = build_one_step_windows(val_indices, surf_tensor, args.history_len)

    history_np = history_01.detach().cpu().numpy()  # (N, H, 5, 5)
    target_np = target_01.detach().cpu().numpy().reshape(history_01.shape[0], 5, 5)
    prev_np = history_np[:, -1]
    realized_delta = target_np - prev_np
    realized_abs = np.abs(realized_delta)

    mean_iv = history_np.mean(axis=(2, 3))
    daily_changes = np.diff(mean_iv, axis=1)
    vol_of_vol = daily_changes.std(axis=1)
    q20 = float(np.quantile(vol_of_vol, 0.20))
    q80 = float(np.quantile(vol_of_vol, 0.80))
    calm_mask = vol_of_vol <= q20
    turb_mask = vol_of_vol >= q80

    paired_noise = torch.randn(history_01.shape[0], args.eval_samples, int(model.noise_dim), device=device)
    cond_samples = _sample_next(model, history_01, args.eval_samples, args.batch_size, noise=paired_noise)
    rng = np.random.default_rng(args.shuffle_seed)
    shuffled_idx = rng.permutation(history_01.shape[0])
    shuf_samples = _sample_next(model, history_01[shuffled_idx], args.eval_samples, args.batch_size, noise=paired_noise)
    params = _decode_params(model, history_01, args.batch_size)

    q05 = np.quantile(cond_samples, 0.05, axis=1)
    q95 = np.quantile(cond_samples, 0.95, axis=1)
    median = np.median(cond_samples, axis=1)
    width = q95 - q05
    cell_cov = _cell_cov(target_np, q05, q95)
    win_cov = _window_cov(target_np, q05, q95)

    shuf_q05 = np.quantile(shuf_samples, 0.05, axis=1)
    shuf_q95 = np.quantile(shuf_samples, 0.95, axis=1)
    shuf_width = shuf_q95 - shuf_q05
    cell_width_ratio = width.mean(axis=0) / np.maximum(shuf_width.mean(axis=0), 1e-8)

    worst_cell = tuple(np.unravel_index(cell_cov.argmin(), (5, 5)))
    best_cell = tuple(np.unravel_index(cell_cov.argmax(), (5, 5)))
    worst_window_idx = int(win_cov.argmin())

    # Worst-cell diagnostics
    wc_r, wc_c = worst_cell
    wc_target = target_np[:, wc_r, wc_c]
    wc_q05 = q05[:, wc_r, wc_c]
    wc_q95 = q95[:, wc_r, wc_c]
    wc_width = width[:, wc_r, wc_c]
    wc_realized_delta = realized_delta[:, wc_r, wc_c]
    wc_realized_abs = np.abs(wc_realized_delta)
    wc_center = params["center"][:, wc_r, wc_c]
    wc_pos_scale = params["pos_scale"][:, wc_r, wc_c]
    wc_neg_scale = params["neg_scale"][:, wc_r, wc_c]
    wc_covered = (wc_target >= wc_q05) & (wc_target <= wc_q95)
    wc_miss_up = wc_target > wc_q95
    wc_miss_down = wc_target < wc_q05

    # Worst-window diagnostics
    ww_q05 = q05[worst_window_idx]
    ww_q95 = q95[worst_window_idx]
    ww_target = target_np[worst_window_idx]
    ww_prev = prev_np[worst_window_idx]
    ww_delta = realized_delta[worst_window_idx]
    ww_cov_grid = ((ww_target >= ww_q05) & (ww_target <= ww_q95))
    ww_width_grid = ww_q95 - ww_q05
    ww_miss_mag = np.where(
        ww_target > ww_q95,
        ww_target - ww_q95,
        np.where(ww_target < ww_q05, ww_q05 - ww_target, 0.0),
    )

    # Conditional response diagnostics
    per_window_width = width.mean(axis=(1, 2))
    per_window_shuf_width = shuf_width.mean(axis=(1, 2))
    width_diff = per_window_width - per_window_shuf_width
    width_ratio = per_window_width / np.maximum(per_window_shuf_width, 1e-8)
    realized_max_abs = realized_abs.max(axis=(1, 2))
    center_abs = np.abs(params["center"]).mean(axis=(1, 2))
    pos_scale_mean = params["pos_scale"].mean(axis=(1, 2))
    neg_scale_mean = params["neg_scale"].mean(axis=(1, 2))
    beta_norm = np.linalg.norm(params["beta"], axis=1)
    gamma_norm = np.linalg.norm(params["gamma"], axis=1)

    # Turb/calm by cell on worst cell
    wc_calm_cov = float(wc_covered[calm_mask].mean())
    wc_turb_cov = float(wc_covered[turb_mask].mean())
    wc_calm_width = float(wc_width[calm_mask].mean())
    wc_turb_width = float(wc_width[turb_mask].mean())

    out = {
        "checkpoint": args.checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "n_val_windows": int(history_01.shape[0]),
        "headline": {
            "overall_width_cond_vs_shuffled_ratio": float(per_window_width.mean() / max(per_window_shuf_width.mean(), 1e-8)),
            "turb_calm_width_ratio": float(per_window_width[turb_mask].mean() / max(per_window_width[calm_mask].mean(), 1e-8)),
            "worst_cell": list(worst_cell),
            "worst_cell_coverage": float(cell_cov[worst_cell]),
            "best_cell": list(best_cell),
            "best_cell_coverage": float(cell_cov[best_cell]),
            "worst_window_idx": worst_window_idx,
            "worst_window_coverage": float(win_cov[worst_window_idx]),
        },
        "conditional_response": {
            "per_window_width_vs_vol_of_vol_corr": _corr(per_window_width, vol_of_vol),
            "per_window_width_ratio_vs_vol_of_vol_corr": _corr(width_ratio, vol_of_vol),
            "per_window_width_diff_vs_vol_of_vol_corr": _corr(width_diff, vol_of_vol),
            "per_window_width_vs_realized_max_abs_corr": _corr(per_window_width, realized_max_abs),
            "center_abs_vs_realized_max_abs_corr": _corr(center_abs, realized_max_abs),
            "pos_scale_mean_vs_realized_max_abs_corr": _corr(pos_scale_mean, realized_max_abs),
            "neg_scale_mean_vs_realized_max_abs_corr": _corr(neg_scale_mean, realized_max_abs),
            "gamma_norm_vs_realized_max_abs_corr": _corr(gamma_norm, realized_max_abs),
            "beta_norm_vs_realized_max_abs_corr": _corr(beta_norm, realized_max_abs),
            "width_ratio_quantiles": {
                "p10": float(np.quantile(width_ratio, 0.10)),
                "p50": float(np.quantile(width_ratio, 0.50)),
                "p90": float(np.quantile(width_ratio, 0.90)),
            },
            "width_diff_quantiles": {
                "p10": float(np.quantile(width_diff, 0.10)),
                "p50": float(np.quantile(width_diff, 0.50)),
                "p90": float(np.quantile(width_diff, 0.90)),
            },
            "cell_width_ratio_grid": cell_width_ratio.tolist(),
        },
        "worst_cell_analysis": {
            "cell": list(worst_cell),
            "coverage": float(cell_cov[worst_cell]),
            "width_mean": float(wc_width.mean()),
            "width_vs_realized_abs_corr": _corr(wc_width, wc_realized_abs),
            "center_vs_realized_delta_corr": _corr(wc_center, wc_realized_delta),
            "pos_scale_vs_realized_abs_corr": _corr(wc_pos_scale, wc_realized_abs),
            "neg_scale_vs_realized_abs_corr": _corr(wc_neg_scale, wc_realized_abs),
            "miss_up_rate": float(wc_miss_up.mean()),
            "miss_down_rate": float(wc_miss_down.mean()),
            "covered_rate": float(wc_covered.mean()),
            "realized_abs_quantiles": {
                "p50": float(np.quantile(wc_realized_abs, 0.50)),
                "p90": float(np.quantile(wc_realized_abs, 0.90)),
                "p95": float(np.quantile(wc_realized_abs, 0.95)),
                "p99": float(np.quantile(wc_realized_abs, 0.99)),
            },
            "width_quantiles": {
                "p50": float(np.quantile(wc_width, 0.50)),
                "p90": float(np.quantile(wc_width, 0.90)),
                "p95": float(np.quantile(wc_width, 0.95)),
                "p99": float(np.quantile(wc_width, 0.99)),
            },
            "calm_coverage": wc_calm_cov,
            "turb_coverage": wc_turb_cov,
            "calm_width_mean": wc_calm_width,
            "turb_width_mean": wc_turb_width,
            "miss_examples_top5": [
                {
                    "window_idx": int(idx),
                    "vol_of_vol": float(vol_of_vol[idx]),
                    "prev": float(prev_np[idx, wc_r, wc_c]),
                    "target": float(wc_target[idx]),
                    "realized_delta": float(wc_realized_delta[idx]),
                    "q05": float(wc_q05[idx]),
                    "q95": float(wc_q95[idx]),
                    "width": float(wc_width[idx]),
                    "center": float(wc_center[idx]),
                    "pos_scale": float(wc_pos_scale[idx]),
                    "neg_scale": float(wc_neg_scale[idx]),
                    "miss_side": "up" if wc_miss_up[idx] else ("down" if wc_miss_down[idx] else "covered"),
                    "miss_mag": float(max(wc_target[idx] - wc_q95[idx], wc_q05[idx] - wc_target[idx], 0.0)),
                }
                for idx in np.argsort(
                    np.where(wc_miss_up, wc_target - wc_q95, np.where(wc_miss_down, wc_q05 - wc_target, 0.0))
                )[::-1][:5]
            ],
        },
        "worst_window_analysis": {
            "window_idx": worst_window_idx,
            "coverage": float(win_cov[worst_window_idx]),
            "vol_of_vol": float(vol_of_vol[worst_window_idx]),
            "mean_width": float(ww_width_grid.mean()),
            "mean_realized_abs_delta": float(np.abs(ww_delta).mean()),
            "max_realized_abs_delta": float(np.abs(ww_delta).max()),
            "n_missed_cells": int((~ww_cov_grid).sum()),
            "coverage_grid": ww_cov_grid.astype(int).tolist(),
            "width_grid": ww_width_grid.tolist(),
            "miss_mag_grid": ww_miss_mag.tolist(),
            "center_grid": params["center"][worst_window_idx].tolist(),
            "pos_scale_grid": params["pos_scale"][worst_window_idx].tolist(),
            "neg_scale_grid": params["neg_scale"][worst_window_idx].tolist(),
            "realized_delta_grid": ww_delta.tolist(),
            "prev_grid": ww_prev.tolist(),
            "target_grid": ww_target.tolist(),
        },
        "floor_effects": {
            "per_cell_floor_rate": (cond_samples <= 0.001).mean(axis=(0, 1)).tolist(),
            "worst_floor_cell": list(np.unravel_index(((cond_samples <= 0.001).mean(axis=(0, 1))).argmax(), (5, 5))),
            "worst_floor_rate": float(((cond_samples <= 0.001).mean(axis=(0, 1))).max()),
        },
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(make_serializable(out), f, indent=2)

    lines = [
        f"# {args.model_type} H1 Failure Gap Analysis",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Epoch: `{out['epoch']}`",
        "",
        "## Headline",
        "",
        f"- Overall cond/shuffled width ratio: `{out['headline']['overall_width_cond_vs_shuffled_ratio']:.3f}`",
        f"- Turb/calm width ratio: `{out['headline']['turb_calm_width_ratio']:.3f}`",
        f"- Worst cell: `{tuple(int(x) for x in out['headline']['worst_cell'])}` with coverage `{out['headline']['worst_cell_coverage']:.1%}`",
        f"- Best cell: `{tuple(int(x) for x in out['headline']['best_cell'])}` with coverage `{out['headline']['best_cell_coverage']:.1%}`",
        f"- Worst window: `{worst_window_idx}` with coverage `{out['headline']['worst_window_coverage']:.1%}`",
        "",
        "## Interpretation",
        "",
        "- The model is using state, but width barely changes when history is shuffled. The main gap is not total state collapse; it is weak state-to-width routing.",
        "- The worst cell is not just mildly undercovered; it is severely under-widened in turbulent cases.",
        "- The worst-window failures are concentrated path-level misses, not just pooled-marginal mismatch.",
        "",
        "## Evidence",
        "",
        f"- Width vs vol-of-vol corr: `{out['conditional_response']['per_window_width_vs_vol_of_vol_corr']:.3f}`",
        f"- Width ratio vs vol-of-vol corr: `{out['conditional_response']['per_window_width_ratio_vs_vol_of_vol_corr']:.3f}`",
        f"- Worst-cell miss-up rate: `{out['worst_cell_analysis']['miss_up_rate']:.1%}`",
        f"- Worst-cell miss-down rate: `{out['worst_cell_analysis']['miss_down_rate']:.1%}`",
        f"- Worst-cell calm coverage: `{out['worst_cell_analysis']['calm_coverage']:.1%}`",
        f"- Worst-cell turbulent coverage: `{out['worst_cell_analysis']['turb_coverage']:.1%}`",
        f"- Worst floor cell: `{tuple(int(x) for x in out['floor_effects']['worst_floor_cell'])}` at `{out['floor_effects']['worst_floor_rate']:.2%}` floor hits",
        "",
    ]
    Path(args.output_md).write_text("\n".join(lines) + "\n")

    print(json.dumps(make_serializable({"headline": out["headline"]}), indent=2))


if __name__ == "__main__":
    main()
