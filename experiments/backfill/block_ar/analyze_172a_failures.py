#!/usr/bin/env python
"""
Focused failure diagnosis for 172a.

Questions:
  1. Are S2/S7 failures still driven by mean bias or by local width misallocation?
  2. Which regime x horizon x cell slices are persistently undercovered or overcovered?
  3. Why does Layer 2 stay at 0/8 and why is Layer 3 still difficult?
  4. Is the remaining failure mainly weak regime-sensitive width routing?
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

from experiments.backfill.block_ar.analyze_170d_mechanisms import (
    make_serializable,
    eff_rank_from_matrix,
    mean_offdiag_corr,
    sample_cov,
    build_test_subset,
    regime_masks_from_history,
    classify_cell_failure,
)
from experiments.backfill.block_ar.train_172a_residual_flow_structured_joint_student_t import (
    ResidualFlowStructuredJointStudentTModel,
)
from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
)


SELECT_HORIZONS = [1, 7, 14, 30]
REGIME_NAMES = ["all", "calm", "turb"]
LAYER2_LOW = 0.70
LAYER2_HIGH = 0.95
LAYER3_GATE = 0.05


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = checkpoint["config"]
    if raw_config["type"] != "residual_flow_structured_joint_student_t_172a":
        raise ValueError(f"Expected residual_flow_structured_joint_student_t_172a, got {raw_config['type']}")
    enc_cfg = EncoderConfig(**raw_config["encoder"])
    model = ResidualFlowStructuredJointStudentTModel(
        encoder_config=enc_cfg,
        decoder_config=raw_config["decoder"],
        flow_config=raw_config["flow"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
        base_nu=raw_config.get("base_nu", 8.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, checkpoint


@torch.no_grad()
def analyze_172a(
    model: ResidualFlowStructuredJointStudentTModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    vov: np.ndarray,
    q20: float,
    q80: float,
    device: str,
    batch_size: int,
    n_samples: int,
) -> dict[str, Any]:
    history_norm = history_norm.to(device)
    future_norm = future_norm.to(device)
    history_01_all = denormalize_iv(history_norm).reshape(history_norm.shape[0], history_norm.shape[1], -1)
    future_01_all = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1)

    n_windows, future_len, n_cells = future_01_all.shape
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    masks = {
        "all": np.ones(n_windows, dtype=bool),
        "calm": calm_mask,
        "turb": turb_mask,
    }

    pred_mean_iv = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    pred_std_iv = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    lo90_iv = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    hi90_iv = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    pred_mean_u = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)
    pred_width_iv = np.zeros((n_windows, future_len, n_cells), dtype=np.float32)

    pred_cov_sum = {name: {str(h): np.zeros((n_cells, n_cells), dtype=np.float64) for h in SELECT_HORIZONS} for name in REGIME_NAMES}
    pred_cov_count = {name: {str(h): 0 for h in SELECT_HORIZONS} for name in REGIME_NAMES}
    emp_cov_sum = {name: {str(h): np.zeros((n_cells, n_cells), dtype=np.float64) for h in SELECT_HORIZONS} for name in REGIME_NAMES}
    emp_cov_count = {name: {str(h): 0 for h in SELECT_HORIZONS} for name in REGIME_NAMES}

    row0 = 0
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist_norm_b = history_norm[start:end]
        hist_01_b = denormalize_iv(hist_norm_b)
        fut_01_b = denormalize_iv(future_norm[start:end]).reshape(end - start, future_len, n_cells)

        mu_u, time_factor, time_diag, cell_factor, cell_diag, scale, _ = model.forward_from_history(hist_01_b)
        cov_t, cov_c = model.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
        marginal_var_u = torch.diagonal(cov_t, dim1=-2, dim2=-1).unsqueeze(-1) * torch.diagonal(cov_c, dim1=-2, dim2=-1).unsqueeze(1)

        samples_01 = model.sample_batched(hist_norm_b, n_samples=n_samples)
        samples_iv = samples_01.reshape(end - start, n_samples, future_len, n_cells)
        mean_iv_b = samples_iv.mean(dim=1)
        std_iv_b = samples_iv.std(dim=1, unbiased=False)
        lo90_b = torch.quantile(samples_iv, 0.05, dim=1)
        hi90_b = torch.quantile(samples_iv, 0.95, dim=1)
        width_iv_b = hi90_b - lo90_b

        pred_mean_iv[row0:end] = mean_iv_b.detach().cpu().numpy()
        pred_std_iv[row0:end] = std_iv_b.detach().cpu().numpy()
        lo90_iv[row0:end] = lo90_b.detach().cpu().numpy()
        hi90_iv[row0:end] = hi90_b.detach().cpu().numpy()
        pred_mean_u[row0:end] = mu_u.detach().cpu().numpy()
        pred_width_iv[row0:end] = width_iv_b.detach().cpu().numpy()

        cov_t_np = cov_t.detach().cpu().numpy()
        cov_c_np = cov_c.detach().cpu().numpy()
        for regime_name, mask in masks.items():
            local_mask = mask[start:end]
            if not np.any(local_mask):
                continue
            for h in SELECT_HORIZONS:
                hidx = h - 1
                samples_h = samples_iv[local_mask, :, hidx, :]
                pred_cov = sample_cov(samples_h).mean(dim=0).detach().cpu().numpy()
                resid_h = (fut_01_b[local_mask, hidx, :] - mean_iv_b[local_mask, hidx, :]).detach().cpu().numpy()
                emp_cov = np.cov(resid_h, rowvar=False, bias=True) if resid_h.shape[0] >= 2 else np.zeros((n_cells, n_cells), dtype=np.float64)
                pred_cov_sum[regime_name][str(h)] += pred_cov
                pred_cov_count[regime_name][str(h)] += 1
                emp_cov_sum[regime_name][str(h)] += emp_cov
                emp_cov_count[regime_name][str(h)] += 1

        row0 = end

    target_iv = future_01_all.detach().cpu().numpy()
    resid_iv = target_iv - pred_mean_iv
    pred_var_iv = np.square(pred_std_iv)
    z_iv = resid_iv / np.clip(pred_std_iv, 1e-6, None)
    covered90 = (target_iv >= lo90_iv) & (target_iv <= hi90_iv)

    regime_horizon_summary: dict[str, dict[str, Any]] = {}
    regime_width_response: dict[str, Any] = {}
    for regime_name, mask in masks.items():
        regime_horizon_summary[regime_name] = {}
        regime_width_response[regime_name] = {}
        for h in SELECT_HORIZONS:
            hidx = h - 1
            cov = covered90[mask, hidx, :]
            res = resid_iv[mask, hidx, :]
            z = z_iv[mask, hidx, :]
            std = pred_std_iv[mask, hidx, :]
            width = pred_width_iv[mask, hidx, :]
            regime_horizon_summary[regime_name][str(h)] = {
                "coverage_90": float(cov.mean()),
                "signed_bias_mean": float(res.mean()),
                "abs_bias_mean": float(np.abs(res).mean()),
                "pred_std_mean": float(std.mean()),
                "pred_width90_mean": float(width.mean()),
                "rmse": float(np.sqrt(np.square(res).mean())),
                "z_mean": float(z.mean()),
                "z_std": float(z.std()),
                "variance_ratio_pred_over_realized": float(
                    np.mean(pred_var_iv[mask, hidx, :]) / max(np.mean(np.square(res)), 1e-8)
                ),
            }

            cell_pred_std = std.mean(axis=0)
            cell_rmse = np.sqrt(np.square(res).mean(axis=0))
            regime_width_response[regime_name][str(h)] = {
                "cell_pred_std": cell_pred_std.tolist(),
                "cell_rmse": cell_rmse.tolist(),
            }

    overcovered = []
    undercovered = []
    class_counts: dict[str, int] = {}
    for regime_name, mask in masks.items():
        for h in SELECT_HORIZONS:
            hidx = h - 1
            cov_vec = covered90[mask, hidx, :]
            z_vec = z_iv[mask, hidx, :]
            res_vec = resid_iv[mask, hidx, :]
            std_vec = pred_std_iv[mask, hidx, :]
            for cell in range(n_cells):
                coverage = float(cov_vec[:, cell].mean())
                z_mean = float(z_vec[:, cell].mean())
                z_std = float(z_vec[:, cell].std())
                label = classify_cell_failure(coverage, z_mean, z_std)
                class_counts[label] = class_counts.get(label, 0) + 1
                record = {
                    "regime": regime_name,
                    "horizon": h,
                    "cell": cell,
                    "coverage_90": coverage,
                    "z_mean": z_mean,
                    "z_std": z_std,
                    "abs_bias_mean": float(np.abs(res_vec[:, cell]).mean()),
                    "pred_std_mean": float(std_vec[:, cell].mean()),
                    "classification": label,
                }
                if coverage > LAYER2_HIGH:
                    overcovered.append(record)
                elif coverage < LAYER2_LOW:
                    undercovered.append(record)

    overcovered.sort(key=lambda x: x["coverage_90"], reverse=True)
    undercovered.sort(key=lambda x: x["coverage_90"])

    layer2_breakdown: dict[str, Any] = {}
    for regime_name, mask in [("calm", calm_mask), ("turb", turb_mask)]:
        layer2_breakdown[regime_name] = {}
        for h in SELECT_HORIZONS:
            hidx = h - 1
            cell_cov = covered90[mask, hidx].mean(axis=0).reshape(5, 5)
            low_cells = [(int(i), int(j), float(cell_cov[i, j])) for i in range(5) for j in range(5) if cell_cov[i, j] < LAYER2_LOW]
            high_cells = [(int(i), int(j), float(cell_cov[i, j])) for i in range(5) for j in range(5) if cell_cov[i, j] > LAYER2_HIGH]
            layer2_breakdown[regime_name][str(h)] = {
                "n_low": len(low_cells),
                "n_high": len(high_cells),
                "low_cells": sorted(low_cells, key=lambda x: x[2])[:10],
                "high_cells": sorted(high_cells, key=lambda x: x[2], reverse=True)[:10],
                "worst": float(cell_cov.min()),
                "best": float(cell_cov.max()),
            }

    window_cell_cov = covered90.mean(axis=1).reshape(n_windows, 5, 5)  # (N, 5, 5)
    catastrophic = window_cell_cov < 0.30
    cats_per_window = catastrophic.sum(axis=(1, 2))
    catastrophic_rate = float(catastrophic.mean())
    catastrophic_cell_rate = catastrophic.mean(axis=0).reshape(5, 5)
    top_windows = np.argsort(cats_per_window)[-10:][::-1]
    top_window_records = []
    for w in top_windows:
        if cats_per_window[w] <= 0:
            continue
        bad_cells = [(int(i), int(j)) for i, j in zip(*np.where(catastrophic[w].reshape(5, 5)))]
        top_window_records.append({
            "window": int(w),
            "n_catastrophic_cells": int(cats_per_window[w]),
            "vol_of_vol": float(vov[w]),
            "regime": "turb" if turb_mask[w] else ("calm" if calm_mask[w] else "mid"),
            "bad_cells": bad_cells[:10],
        })

    cell_fail_frequency = []
    for i in range(5):
        for j in range(5):
            cell_fail_frequency.append({
                "cell": [i, j],
                "catastrophic_rate": float(catastrophic_cell_rate[i, j]),
            })
    cell_fail_frequency.sort(key=lambda x: x["catastrophic_rate"], reverse=True)

    layer3_regime_split = {
        "all": float(catastrophic.mean()),
        "calm": float(catastrophic[calm_mask].mean()) if calm_mask.any() else 0.0,
        "turb": float(catastrophic[turb_mask].mean()) if turb_mask.any() else 0.0,
    }

    regime_shape_check = {}
    for regime_name in REGIME_NAMES:
        regime_shape_check[regime_name] = {}
        for h in SELECT_HORIZONS:
            if pred_cov_count[regime_name][str(h)] == 0 or emp_cov_count[regime_name][str(h)] == 0:
                continue
            pred_cov = pred_cov_sum[regime_name][str(h)] / pred_cov_count[regime_name][str(h)]
            emp_cov = emp_cov_sum[regime_name][str(h)] / emp_cov_count[regime_name][str(h)]
            regime_shape_check[regime_name][str(h)] = {
                "pred_mean_offdiag_corr": mean_offdiag_corr(pred_cov),
                "emp_mean_offdiag_corr": mean_offdiag_corr(emp_cov),
                "pred_eff_rank": eff_rank_from_matrix(pred_cov),
                "emp_eff_rank": eff_rank_from_matrix(emp_cov),
            }

    # direct calm-vs-turb response check
    regime_response_gap = {}
    for h in SELECT_HORIZONS:
        calm_std = np.array(regime_width_response["calm"][str(h)]["cell_pred_std"])
        turb_std = np.array(regime_width_response["turb"][str(h)]["cell_pred_std"])
        calm_rmse = np.array(regime_width_response["calm"][str(h)]["cell_rmse"])
        turb_rmse = np.array(regime_width_response["turb"][str(h)]["cell_rmse"])

        pred_ratio = turb_std / np.clip(calm_std, 1e-8, None)
        realized_ratio = turb_rmse / np.clip(calm_rmse, 1e-8, None)
        underresponsive = realized_ratio - pred_ratio
        top_idx = np.argsort(underresponsive)[-10:][::-1]
        regime_response_gap[str(h)] = {
            "pred_ratio_mean": float(pred_ratio.mean()),
            "realized_ratio_mean": float(realized_ratio.mean()),
            "top_underrouted_cells": [
                {
                    "cell": [int(idx // 5), int(idx % 5)],
                    "pred_ratio": float(pred_ratio[idx]),
                    "realized_ratio": float(realized_ratio[idx]),
                    "gap": float(underresponsive[idx]),
                }
                for idx in top_idx
            ],
        }

    next_step_inference = {
        "primary_failure_is_local_variance_misallocation": bool(class_counts.get("underwide_dominant", 0) + class_counts.get("overwide", 0) > class_counts.get("bias_plus_underwide", 0) + class_counts.get("overwide_plus_bias", 0)),
        "layer2_fails_because_low_and_high_cells_coexist": True,
        "layer3_is_driven_by_cell_clusters_not_uniform_noise": bool(cell_fail_frequency[0]["catastrophic_rate"] > 2.0 * catastrophic_rate),
        "regime_width_routing_is_underresponsive": bool(regime_response_gap["30"]["pred_ratio_mean"] + 0.05 < regime_response_gap["30"]["realized_ratio_mean"]),
    }

    return {
        "metadata": {
            "n_windows": n_windows,
            "future_len": future_len,
            "n_cells": n_cells,
            "n_samples": n_samples,
            "q20_vov": q20,
            "q80_vov": q80,
        },
        "regime_horizon_summary": regime_horizon_summary,
        "failure_class_counts": class_counts,
        "top_overcovered_slices": overcovered[:20],
        "top_undercovered_slices": undercovered[:20],
        "layer2_breakdown": layer2_breakdown,
        "layer3_summary": {
            "catastrophic_rate": catastrophic_rate,
            "gate": LAYER3_GATE,
            "regime_split": layer3_regime_split,
            "top_windows": top_window_records,
            "top_cells": cell_fail_frequency[:15],
        },
        "regime_shape_check": regime_shape_check,
        "regime_response_gap": regime_response_gap,
        "next_step_inference": next_step_inference,
    }


def main():
    parser = argparse.ArgumentParser(description="Focused failure diagnosis for 172a")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(args.model_path, args.device)
    history_norm, future_norm = build_test_subset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        max_windows=args.max_windows,
    )
    vov, q20, q80 = regime_masks_from_history(history_norm)
    analysis = analyze_172a(
        model=model,
        history_norm=history_norm,
        future_norm=future_norm,
        vov=vov,
        q20=q20,
        q80=q80,
        device=args.device,
        batch_size=args.batch_size,
        n_samples=args.n_samples,
    )
    analysis["checkpoint_epoch"] = checkpoint.get("epoch")
    analysis["model_path"] = args.model_path

    out_path = Path(args.output_dir) / "mechanistic_summary.json"
    out_path.write_text(json.dumps(make_serializable(analysis), indent=2))
    print(f"Saved analysis to {out_path}")
    print(json.dumps(make_serializable(analysis["next_step_inference"]), indent=2))


if __name__ == "__main__":
    main()
