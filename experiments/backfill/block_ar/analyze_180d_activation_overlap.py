#!/usr/bin/env python
"""
Targeted overlap review for 180d.

Goal:
  Confirm whether the now-active sparse jump branch is actually aligned with the
  remaining bad S2/S3/S7 slices, or whether it mostly widens the wrong parts of
  the surface while only weakly affecting pathwise jump shape.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.analyze_170d_mechanisms import (
    build_test_subset,
    make_serializable,
)
from experiments.backfill.block_ar.config_block_ar import get_default_config
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    unconstrained_to_iv,
)
from experiments.backfill.block_ar.train_180d_sparse_centered_jump_residual import (
    SparseCenteredJumpResidualMeanRevertingCovarianceMixtureModel,
)


SELECT_HORIZONS = [1, 7, 14, 30]
LATE_HORIZONS = [14, 30]


def ks_detail(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if len(x) == 0 or len(y) == 0:
        return {"ks_stat": float("nan"), "peak_x": float("nan"), "cdf_gt": float("nan"), "cdf_gen": float("nan")}
    grid = np.unique(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, grid, side="right") / len(x)
    cdf_y = np.searchsorted(y, grid, side="right") / len(y)
    diff = np.abs(cdf_x - cdf_y)
    idx = int(np.argmax(diff))
    return {
        "ks_stat": float(diff[idx]),
        "peak_x": float(grid[idx]),
        "cdf_gt": float(cdf_x[idx]),
        "cdf_gen": float(cdf_y[idx]),
    }


def quantile_profile(x: np.ndarray, qs: tuple[float, ...] = (0.5, 0.9, 0.99)) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return {str(q): float(np.quantile(x, q)) for q in qs}


def ratio_profile(gt: np.ndarray, gen: np.ndarray) -> dict[str, float]:
    q_gt = quantile_profile(gt)
    q_gen = quantile_profile(gen)
    return {k: float(q_gen[k] / max(q_gt[k], 1e-12)) for k in q_gt}


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0 or np.std(x) <= 1e-12 or np.std(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def regime_masks_from_history(history_norm: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    history_01 = denormalize_iv(history_norm)
    mean_iv = history_01.mean(dim=(-1, -2))
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    vov = daily_chg.std(dim=1).cpu().numpy()
    q20 = float(np.quantile(vov, 0.2))
    q80 = float(np.quantile(vov, 0.8))
    calm = vov <= q20
    turb = vov >= q80
    return vov, calm, turb, q20, q80


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    expected = "sparse_centered_jump_residual_mean_reverting_covariance_mixture_structured_joint_student_t_180d"
    if raw_config["type"] != expected:
        raise ValueError(f"Expected {expected}, got {raw_config['type']}")
    enc_cfg = EncoderConfig(**raw_config["encoder"])
    model = SparseCenteredJumpResidualMeanRevertingCovarianceMixtureModel(
        encoder_config=enc_cfg,
        decoder_config=raw_config["decoder"],
        flow_config=raw_config["flow"],
        jump_config=raw_config["jump"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
        cov_jitter=raw_config.get("cov_jitter", 1e-5),
        base_nu=raw_config.get("base_nu", 8.0),
        mix_chunk_size=raw_config.get("mix_chunk_size", 27),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def sample_future_mode(
    model: SparseCenteredJumpResidualMeanRevertingCovarianceMixtureModel,
    history_01: torch.Tensor,
    n_samples: int,
    mode: str,
    seed: int,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    (
        mu,
        time_factor,
        time_diag,
        cell_factor,
        cell_diag,
        scale,
        flow_context,
        base_local_delta,
        block_logits,
    ) = model.forward_from_history(history_01)
    batch, n_frames, n_cells = mu.shape
    cov_t, cov_c = model.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
    chol_t = torch.linalg.cholesky(cov_t)
    chol_c = torch.linalg.cholesky(cov_c)

    logits, probs, scales = model.jump.params(flow_context)
    expected_strength = probs * scales

    torch.manual_seed(seed)
    base = torch.distributions.StudentT(df=model.base_nu)
    z = base.sample((batch * n_samples, n_frames * n_cells)).to(device=mu.device, dtype=mu.dtype)
    ctx = flow_context.unsqueeze(1).expand(batch, n_samples, -1).reshape(batch * n_samples, -1)
    smooth_white_flat, _ = model.flow.inverse(z, ctx)
    smooth_basis_flat = model.flow.to_basis_flat(smooth_white_flat).view(batch, n_samples, -1)

    if mode == "default":
        jump_shift, jump_stats = model.jump.sample_shifts(flow_context, n_samples=n_samples)
    elif mode == "zero":
        jump_shift = smooth_basis_flat.new_zeros(batch, n_samples, smooth_basis_flat.shape[-1])
        jump_stats = {
            "group_probs": probs,
            "group_scales": scales,
            "group_entropy": (
                -(probs * torch.log(probs.clamp_min(1e-8)) + (1.0 - probs) * torch.log((1.0 - probs).clamp_min(1e-8)))
            ).mean(dim=-1),
            "active_rate": torch.zeros(batch, device=probs.device, dtype=probs.dtype),
            "jump_l2": torch.zeros(batch, device=probs.device, dtype=probs.dtype),
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    basis_with_jump = smooth_basis_flat + jump_shift
    base_white_flat = model.flow.from_basis_flat(basis_with_jump.reshape(batch * n_samples, -1))
    base_white = base_white_flat.view(batch * n_samples, model.decoder.n_blocks, model.decoder.block_len, n_cells)

    block_probs = F.softmax(block_logits, dim=-1)
    torch.manual_seed(seed + 1)
    sampled_blocks = []
    for b in range(model.decoder.n_blocks):
        sampled = torch.multinomial(block_probs[:, b], num_samples=n_samples, replacement=True)
        sampled_blocks.append(sampled)
    sampled_assign = torch.stack(sampled_blocks, dim=-1)
    assign_flat = sampled_assign.reshape(batch * n_samples, model.decoder.n_blocks)
    sampled_factors, _logdet_cov, _offdiag_rms = model.decoder.build_template_factors(assign_flat)

    lhs = base_white.permute(0, 1, 3, 2).reshape(batch * n_samples * model.decoder.n_blocks, n_cells, model.decoder.block_len)
    factor_flat = sampled_factors.reshape(batch * n_samples * model.decoder.n_blocks, n_cells, n_cells)
    routed_white = torch.matmul(factor_flat, lhs)
    routed_white = routed_white.reshape(batch * n_samples, model.decoder.n_blocks, n_cells, model.decoder.block_len).permute(0, 1, 3, 2)
    routed_white = routed_white.reshape(batch, n_samples, n_frames, n_cells)

    temp = torch.einsum("bij,bsjk->bsik", chol_t, routed_white)
    noise = torch.einsum("bstj,bcj->bstc", temp, chol_c)
    shared_local_delta = model.decoder.build_shared_local_delta(base_local_delta)
    local_scale = torch.exp(0.5 * shared_local_delta).unsqueeze(1)
    u_samples = mu.unsqueeze(1) + noise * local_scale
    iv_samples = unconstrained_to_iv(u_samples, lo=model.support_lo, hi=model.support_hi).reshape(batch, n_samples, n_frames, 5, 5)

    meta = {
        "group_probs": probs,
        "group_scales": scales,
        "expected_strength": expected_strength,
        "active_rate": jump_stats["active_rate"],
        "jump_l2": jump_stats["jump_l2"],
    }
    return iv_samples, meta


def identify_slice_sets(
    gt: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    turb_mask: np.ndarray,
) -> dict[str, list[dict[str, Any]]]:
    hard_under = []
    for h in LATE_HORIZONS:
        inside = (gt[turb_mask, h - 1] >= lo[turb_mask, h - 1]) & (gt[turb_mask, h - 1] <= hi[turb_mask, h - 1])
        coverage = inside.mean(axis=0)
        for r in range(5):
            for c in range(5):
                if coverage[r, c] < 0.75:
                    hard_under.append(
                        {
                            "horizon": h,
                            "cell": [r, c],
                            "coverage_turb": float(coverage[r, c]),
                        }
                    )
    hard_under.sort(key=lambda x: x["coverage_turb"])

    overcovered = []
    for h in SELECT_HORIZONS:
        inside = (gt[:, h - 1] >= lo[:, h - 1]) & (gt[:, h - 1] <= hi[:, h - 1])
        coverage = inside.mean(axis=0)
        for r in range(5):
            for c in range(5):
                if coverage[r, c] > 0.95:
                    overcovered.append(
                        {
                            "horizon": h,
                            "cell": [r, c],
                            "coverage_all": float(coverage[r, c]),
                        }
                    )
    overcovered.sort(key=lambda x: -x["coverage_all"])
    return {"hard_under": hard_under, "overcovered": overcovered}


def slice_indicator_counts(gt: np.ndarray, lo: np.ndarray, hi: np.ndarray, slices: list[dict[str, Any]]) -> np.ndarray:
    counts = np.zeros(gt.shape[0], dtype=np.int64)
    for spec in slices:
        h = spec["horizon"] - 1
        r, c = spec["cell"]
        inside = (gt[:, h, r, c] >= lo[:, h, r, c]) & (gt[:, h, r, c] <= hi[:, h, r, c])
        counts += (~inside).astype(np.int64)
    return counts


def aggregate_group_strength(strength: np.ndarray, metadata: list[dict[str, Any]], mask: np.ndarray) -> dict[str, float]:
    if mask.sum() == 0:
        return {}
    out: dict[str, float] = {}
    for gi, meta in enumerate(metadata):
        key = f"block{meta['block']+1}_{meta['band']}"
        out[key] = float(strength[mask, gi].mean())
    return out


def top_group_deltas(
    strength: np.ndarray,
    metadata: list[dict[str, Any]],
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    top_k: int = 6,
) -> list[dict[str, Any]]:
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return []
    mean_a = strength[mask_a].mean(axis=0)
    mean_b = strength[mask_b].mean(axis=0)
    delta = mean_a - mean_b
    order = np.argsort(-delta)
    rows = []
    for idx in order[:top_k]:
        meta = metadata[int(idx)]
        rows.append(
            {
                "group_index": int(idx),
                "block": int(meta["block"] + 1),
                "band": str(meta["band"]),
                "bad_mean": float(mean_a[idx]),
                "clean_mean": float(mean_b[idx]),
                "delta": float(delta[idx]),
            }
        )
    return rows


def slice_set_stats(
    gt: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    width: np.ndarray,
    slices: list[dict[str, Any]],
    window_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    if not slices:
        return {}
    if window_mask is None:
        window_mask = np.ones(gt.shape[0], dtype=bool)
    widths = []
    covered = []
    per_slice = []
    for spec in slices:
        h = spec["horizon"] - 1
        r, c = spec["cell"]
        w = width[window_mask, h, r, c]
        inside = (gt[window_mask, h, r, c] >= lo[window_mask, h, r, c]) & (gt[window_mask, h, r, c] <= hi[window_mask, h, r, c])
        widths.append(w)
        covered.append(inside.astype(np.float32))
        per_slice.append(
            {
                "horizon": spec["horizon"],
                "cell": spec["cell"],
                "mean_width": float(w.mean()),
                "coverage": float(inside.mean()),
            }
        )
    return {
        "mean_width": float(np.concatenate(widths).mean()),
        "coverage": float(np.concatenate(covered).mean()),
        "per_slice": per_slice,
    }


def max_jump_distribution(history_01: np.ndarray, samples_01: np.ndarray) -> np.ndarray:
    hist_last = history_01[:, -1:]
    path = np.concatenate([np.repeat(hist_last[:, None], samples_01.shape[1], axis=1), samples_01], axis=2)
    diff = np.diff(path, axis=2)
    return np.abs(diff).max(axis=(2, 3, 4)).reshape(-1)


def gt_max_jump_distribution(history_01: np.ndarray, future_01: np.ndarray) -> np.ndarray:
    path = np.concatenate([history_01[:, -1:], future_01], axis=1)
    diff = np.diff(path, axis=1)
    return np.abs(diff).max(axis=(1, 2, 3)).reshape(-1)


@torch.no_grad()
def analyze_overlap(
    model: SparseCenteredJumpResidualMeanRevertingCovarianceMixtureModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: str,
    batch_size: int,
    n_samples: int,
    seed: int,
) -> dict[str, Any]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm)
    history_np = history_01.cpu().numpy()
    future_np = future_01.cpu().numpy()

    vov, calm_mask, turb_mask, q20, q80 = regime_masks_from_history(history_norm.cpu())

    lo_default, hi_default, width_default = [], [], []
    lo_zero, hi_zero, width_zero = [], [], []
    default_samples_all, zero_samples_all = [], []
    expected_strength_all, active_rate_all, jump_l2_all = [], [], []

    n_windows = history_norm.shape[0]
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist_01_b = history_01[start:end]
        samples_default, meta_default = sample_future_mode(model, hist_01_b, n_samples=n_samples, mode="default", seed=seed + start)
        samples_zero, _meta_zero = sample_future_mode(model, hist_01_b, n_samples=n_samples, mode="zero", seed=seed + start)

        s_def = samples_default.cpu().numpy()
        s_zero = samples_zero.cpu().numpy()
        lo_d = np.quantile(s_def, 0.05, axis=1)
        hi_d = np.quantile(s_def, 0.95, axis=1)
        lo_z = np.quantile(s_zero, 0.05, axis=1)
        hi_z = np.quantile(s_zero, 0.95, axis=1)

        lo_default.append(lo_d)
        hi_default.append(hi_d)
        width_default.append(hi_d - lo_d)
        lo_zero.append(lo_z)
        hi_zero.append(hi_z)
        width_zero.append(hi_z - lo_z)
        default_samples_all.append(s_def)
        zero_samples_all.append(s_zero)
        expected_strength_all.append(meta_default["expected_strength"].cpu().numpy())
        active_rate_all.append(meta_default["active_rate"].cpu().numpy())
        jump_l2_all.append(meta_default["jump_l2"].cpu().numpy())

    lo_default = np.concatenate(lo_default, axis=0)
    hi_default = np.concatenate(hi_default, axis=0)
    width_default = np.concatenate(width_default, axis=0)
    lo_zero = np.concatenate(lo_zero, axis=0)
    hi_zero = np.concatenate(hi_zero, axis=0)
    width_zero = np.concatenate(width_zero, axis=0)
    default_samples_all = np.concatenate(default_samples_all, axis=0)
    zero_samples_all = np.concatenate(zero_samples_all, axis=0)
    expected_strength_all = np.concatenate(expected_strength_all, axis=0)
    active_rate_all = np.concatenate(active_rate_all, axis=0)
    jump_l2_all = np.concatenate(jump_l2_all, axis=0)

    slice_sets = identify_slice_sets(future_np, lo_default, hi_default, turb_mask)
    hard_slices = slice_sets["hard_under"]
    over_slices = slice_sets["overcovered"]

    hard_miss_count = slice_indicator_counts(future_np, lo_default, hi_default, hard_slices)
    hard_miss_window = hard_miss_count > 0
    bad_turb_mask = turb_mask & hard_miss_window
    clean_turb_mask = turb_mask & (~hard_miss_window)

    hard_stats_default = slice_set_stats(future_np, lo_default, hi_default, width_default, hard_slices, turb_mask)
    hard_stats_zero = slice_set_stats(future_np, lo_zero, hi_zero, width_zero, hard_slices, turb_mask)
    over_stats_default = slice_set_stats(future_np, lo_default, hi_default, width_default, over_slices)
    over_stats_zero = slice_set_stats(future_np, lo_zero, hi_zero, width_zero, over_slices)

    width_delta = width_default - width_zero
    width_delta_h = {}
    for h in LATE_HORIZONS:
        idx = h - 1
        width_delta_h[f"h{h}_all"] = width_delta[:, idx].mean(axis=0).tolist()
        width_delta_h[f"h{h}_turb"] = width_delta[turb_mask, idx].mean(axis=0).tolist()

    default_jump = max_jump_distribution(history_np, default_samples_all)
    zero_jump = max_jump_distribution(history_np, zero_samples_all)
    gt_jump = gt_max_jump_distribution(history_np, future_np)

    total_strength = expected_strength_all.sum(axis=1)
    metadata = model.jump_group_metadata
    overlap_summary = {
        "n_windows": int(n_windows),
        "calm_windows": int(calm_mask.sum()),
        "turb_windows": int(turb_mask.sum()),
        "hard_miss_windows": int(hard_miss_window.sum()),
        "hard_miss_turb_windows": int(bad_turb_mask.sum()),
        "hard_miss_clean_turb_windows": int(clean_turb_mask.sum()),
        "vov_q20": q20,
        "vov_q80": q80,
        "hard_miss_count_mean_all": float(hard_miss_count.mean()),
        "hard_miss_count_mean_turb": float(hard_miss_count[turb_mask].mean()) if turb_mask.any() else float("nan"),
        "total_expected_strength_mean": {
            "all": float(total_strength.mean()),
            "calm": float(total_strength[calm_mask].mean()) if calm_mask.any() else float("nan"),
            "turb": float(total_strength[turb_mask].mean()) if turb_mask.any() else float("nan"),
            "bad_turb": float(total_strength[bad_turb_mask].mean()) if bad_turb_mask.any() else float("nan"),
            "clean_turb": float(total_strength[clean_turb_mask].mean()) if clean_turb_mask.any() else float("nan"),
        },
        "jump_active_rate_mean": {
            "all": float(active_rate_all.mean()),
            "bad_turb": float(active_rate_all[bad_turb_mask].mean()) if bad_turb_mask.any() else float("nan"),
            "clean_turb": float(active_rate_all[clean_turb_mask].mean()) if clean_turb_mask.any() else float("nan"),
        },
        "jump_l2_mean": {
            "all": float(jump_l2_all.mean()),
            "bad_turb": float(jump_l2_all[bad_turb_mask].mean()) if bad_turb_mask.any() else float("nan"),
            "clean_turb": float(jump_l2_all[clean_turb_mask].mean()) if clean_turb_mask.any() else float("nan"),
        },
        "hard_miss_count_corr_with_total_strength_turb": pearson_corr(
            hard_miss_count[turb_mask],
            total_strength[turb_mask],
        ) if turb_mask.any() else float("nan"),
        "group_strength_mean_all": aggregate_group_strength(expected_strength_all, metadata, np.ones(n_windows, dtype=bool)),
        "group_strength_mean_bad_turb": aggregate_group_strength(expected_strength_all, metadata, bad_turb_mask),
        "group_strength_mean_clean_turb": aggregate_group_strength(expected_strength_all, metadata, clean_turb_mask),
        "top_group_deltas_bad_minus_clean_turb": top_group_deltas(expected_strength_all, metadata, bad_turb_mask, clean_turb_mask),
    }

    result = {
        "checkpoint_path": str(getattr(model, "_checkpoint_path", "")),
        "subset": {
            "max_windows": int(n_windows),
            "n_samples": int(n_samples),
            "batch_size": int(batch_size),
            "seed": int(seed),
        },
        "derived_failure_slices": {
            "hard_undercovered_turbulent_late_horizons": hard_slices,
            "overcovered_slices": over_slices,
        },
        "jump_overlap": overlap_summary,
        "counterfactual_width_effect": {
            "hard_slices_default": hard_stats_default,
            "hard_slices_zero_jump": hard_stats_zero,
            "hard_slices_width_delta_mean": (
                float(hard_stats_default["mean_width"] - hard_stats_zero["mean_width"])
                if hard_stats_default and hard_stats_zero
                else float("nan")
            ),
            "hard_slices_coverage_delta": (
                float(hard_stats_default["coverage"] - hard_stats_zero["coverage"])
                if hard_stats_default and hard_stats_zero
                else float("nan")
            ),
            "overcovered_slices_default": over_stats_default,
            "overcovered_slices_zero_jump": over_stats_zero,
            "overcovered_slices_width_delta_mean": (
                float(over_stats_default["mean_width"] - over_stats_zero["mean_width"])
                if over_stats_default and over_stats_zero
                else float("nan")
            ),
            "per_horizon_width_delta_grid": width_delta_h,
        },
        "pathwise_jump_counterfactual": {
            "default_vs_gt": {
                "ks": ks_detail(gt_jump, default_jump),
                "ratio_quantiles": ratio_profile(gt_jump, default_jump),
            },
            "zero_jump_vs_gt": {
                "ks": ks_detail(gt_jump, zero_jump),
                "ratio_quantiles": ratio_profile(gt_jump, zero_jump),
            },
            "jump_enabled_minus_zero": {
                "mean_abs_iv_shift": float(np.abs(default_samples_all - zero_samples_all).mean()),
                "q90_abs_iv_shift": float(np.quantile(np.abs(default_samples_all - zero_samples_all).reshape(-1), 0.90)),
                "max_abs_iv_shift": float(np.abs(default_samples_all - zero_samples_all).max()),
            },
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted overlap review for 180d sparse jump activations")
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/backfill/sparse_centered_jump_residual_mean_reverting_covariance_mixture_structured_joint_student_t_180d/best_model.pt",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/validations/2026-04-05/analysis/180d_activation_overlap/mechanistic_summary.json",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_windows", type=int, default=512)
    parser.add_argument("--n_samples", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = get_default_config()
    history_norm, future_norm = build_test_subset(
        data_path=cfg.data_path,
        history_len=cfg.history_len,
        future_len=cfg.future_len,
        test_start=cfg.test_start,
        max_windows=args.max_windows,
    )
    model, checkpoint = load_model(args.model_path, args.device)
    model._checkpoint_path = args.model_path

    summary = analyze_overlap(
        model=model,
        history_norm=history_norm,
        future_norm=future_norm,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
        seed=args.seed,
    )
    summary["checkpoint_epoch"] = int(checkpoint.get("epoch", -1))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(make_serializable(summary), indent=2))
    print(f"Saved targeted overlap review to {output_path}")


if __name__ == "__main__":
    main()
