#!/usr/bin/env python
"""702a: audit whether a learned risk-state branch controls sampled uncertainty.

This is a post-experiment diagnostic for 701a. It does not propose a new
architecture; it asks why the risk-state auxiliary learned an in-sample ranking
without improving validation deployability.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, ".")

from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: E402
    load_model,
)
from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.audit_700a_tri_scope_conditional_signal import (  # noqa: E402
    _load_panel,
    _scope_arrays,
    ordinal_spearman,
)


def future_risk_targets_np(future_norm: np.ndarray) -> dict[str, np.ndarray]:
    future = np.asarray(future_norm, dtype=np.float64)
    abs_x = np.abs(future)
    per_step = abs_x.mean(axis=2)
    return {
        "activity": np.mean(future * future, axis=(1, 2)),
        "mean_abs": abs_x.mean(axis=(1, 2)),
        "max_abs": abs_x.max(axis=(1, 2)),
        "temporal_peak": per_step.max(axis=1) / np.maximum(per_step.mean(axis=1), 1e-12),
    }


def width_by_window(samples: np.ndarray) -> np.ndarray:
    arr = np.asarray(samples, dtype=np.float64)
    if arr.ndim != 4:
        raise ValueError("samples must have shape [windows, samples, horizon, channels]")
    return np.mean(np.quantile(arr, 0.95, axis=1) - np.quantile(arr, 0.05, axis=1), axis=(1, 2))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    keep = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_arr = x_arr[keep]
    y_arr = y_arr[keep]
    if x_arr.size < 3 or np.std(x_arr) < 1e-12 or np.std(y_arr) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


@torch.no_grad()
def risk_prediction_summary(
    model: Any,
    arrays: dict[str, Any],
    *,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    history_level = arrays["history_level"].astype(np.float32)
    history_norm = arrays["history_norm"].astype(np.float32)
    future_norm = arrays["future_norm"].astype(np.float32)
    center = arrays["center"].astype(np.float32)
    scale = arrays["scale"].astype(np.float32)
    drift_feature = arrays["drift_feature"].astype(np.float32)
    preds: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    memory_norms: list[np.ndarray] = []
    velocity_ratios: list[np.ndarray] = []
    for start in range(0, history_level.shape[0], int(batch_size)):
        end = min(start + int(batch_size), history_level.shape[0])
        h_level = torch.from_numpy(history_level[start:end]).to(device)
        h_norm = torch.from_numpy(history_norm[start:end]).to(device)
        ctr = torch.from_numpy(center[start:end]).to(device)
        scl = torch.from_numpy(scale[start:end]).to(device)
        drift = torch.from_numpy(drift_feature[start:end]).to(device)
        h_score = model.level_values_to_scores(h_level)
        h_flow = model._to_flow_coordinate(h_norm)
        pred, context = model._risk_state_from_history(h_score, h_flow, ctr, scl, drift)
        if pred is None or context is None:
            raise RuntimeError("checkpoint does not have an active risk-state branch")
        hidden = model._encode_prefix(h_score, h_flow, ctr, scl, drift)
        memory = hidden[:, -1]
        current_level = h_score[:, -1]
        x = torch.zeros_like(current_level)
        t = torch.full((end - start,), 0.5, device=device, dtype=current_level.dtype)
        v_without = model.velocity(x, current_level, memory, t)
        v_with = model.velocity(x, current_level, memory + context, t)
        ratios = (v_with - v_without).norm(dim=1) / v_without.norm(dim=1).clamp_min(1e-8)
        preds.append(pred.detach().cpu().numpy())
        contexts.append(context.detach().cpu().numpy())
        memory_norms.append(memory.norm(dim=1).detach().cpu().numpy())
        velocity_ratios.append(ratios.detach().cpu().numpy())
    pred_arr = np.concatenate(preds, axis=0)
    context_arr = np.concatenate(contexts, axis=0)
    memory_norm_arr = np.concatenate(memory_norms, axis=0)
    velocity_ratio_arr = np.concatenate(velocity_ratios, axis=0)
    targets = future_risk_targets_np(future_norm)
    history_activity = np.mean(history_norm.astype(np.float64) ** 2, axis=(1, 2))
    risk0 = pred_arr[:, 0]
    return {
        "risk_prediction_std": pred_arr.std(axis=0).tolist(),
        "risk_prediction_mean": pred_arr.mean(axis=0).tolist(),
        "risk0_future_activity_spearman": ordinal_spearman(risk0, targets["activity"]),
        "risk0_future_activity_corr": _corr(risk0, targets["activity"]),
        "risk0_history_activity_spearman": ordinal_spearman(risk0, history_activity),
        "risk0_history_activity_corr": _corr(risk0, history_activity),
        "history_future_activity_spearman": ordinal_spearman(history_activity, targets["activity"]),
        "context_norm_mean": float(np.linalg.norm(context_arr, axis=1).mean()),
        "context_norm_p90": float(np.quantile(np.linalg.norm(context_arr, axis=1), 0.9)),
        "memory_norm_mean": float(memory_norm_arr.mean()),
        "context_to_memory_norm_ratio": float(
            np.mean(np.linalg.norm(context_arr, axis=1) / np.maximum(memory_norm_arr, 1e-12))
        ),
        "velocity_delta_ratio_mean": float(velocity_ratio_arr.mean()),
        "velocity_delta_ratio_p90": float(np.quantile(velocity_ratio_arr, 0.9)),
    }


@torch.no_grad()
def sample_context_ablation(
    model: Any,
    arrays: dict[str, Any],
    *,
    device: torch.device,
    n_windows: int,
    samples: int,
    n_steps: int,
    chunk_size: int,
    seed: int,
) -> dict[str, float]:
    n = min(int(n_windows), int(arrays["history_level"].shape[0]))
    args = (
        torch.from_numpy(arrays["history_level"][:n].astype(np.float32)).to(device),
        torch.from_numpy(arrays["history_norm"][:n].astype(np.float32)).to(device),
        torch.from_numpy(arrays["center"][:n].astype(np.float32)).to(device),
        torch.from_numpy(arrays["scale"][:n].astype(np.float32)).to(device),
    )
    drift = torch.from_numpy(arrays["drift_feature"][:n].astype(np.float32)).to(device)
    torch.manual_seed(int(seed))
    normal = model.sample_batched(
        *args,
        drift_feature=drift,
        n_samples=int(samples),
        n_steps=int(n_steps),
        chunk_size=int(chunk_size),
    ).detach().cpu().numpy()

    zero_model = copy.deepcopy(model).to(device).eval()
    if zero_model.risk_context_proj is not None:
        for param in zero_model.risk_context_proj.parameters():
            param.zero_()
    torch.manual_seed(int(seed))
    zeroed = zero_model.sample_batched(
        *args,
        drift_feature=drift,
        n_samples=int(samples),
        n_steps=int(n_steps),
        chunk_size=int(chunk_size),
    ).detach().cpu().numpy()
    width_normal = width_by_window(normal)
    width_zeroed = width_by_window(zeroed)
    targets = future_risk_targets_np(arrays["future_norm"][:n].astype(np.float32))
    return {
        "ablation_windows": int(n),
        "normal_width_mean": float(width_normal.mean()),
        "zeroed_width_mean": float(width_zeroed.mean()),
        "width_mean_relative_change_pct": float(
            (width_normal.mean() - width_zeroed.mean()) / max(width_zeroed.mean(), 1e-12) * 100.0
        ),
        "sample_mean_abs_diff": float(np.mean(np.abs(normal - zeroed))),
        "normal_width_future_activity_spearman": ordinal_spearman(width_normal, targets["activity"]),
        "zeroed_width_future_activity_spearman": ordinal_spearman(width_zeroed, targets["activity"]),
    }


def write_markdown(path: Path, result: dict[str, Any]) -> None:
    lines = [
        "# 702a Risk-State Conditioning Channel Audit",
        "",
        f"- checkpoint: `{result['config']['checkpoint']}`",
        f"- scope: `{result['config']['state_scope']}`",
        "",
        "| split | risk0/future rho | risk0/history rho | context/memory | velocity delta | width change vs zero-risk | normal width/future rho |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["audits"]:
        r = row["risk_summary"]
        a = row["ablation"]
        lines.append(
            f"| {row['split']} | "
            f"{r['risk0_future_activity_spearman']:.3f} | "
            f"{r['risk0_history_activity_spearman']:.3f} | "
            f"{r['context_to_memory_norm_ratio']:.4f} | "
            f"{r['velocity_delta_ratio_mean']:.4f} | "
            f"{a['width_mean_relative_change_pct']:.2f}% | "
            f"{a['normal_width_future_activity_spearman']:.3f} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state_scope", choices=["iv_only", "anchor_only", "joint38"], default="joint38")
    parser.add_argument("--splits", nargs="+", choices=["val", "train_tail", "train"], default=["val", "train_tail"])
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--max_windows", type=int, default=441)
    parser.add_argument("--sample_windows", type=int, default=128)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--n_steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=7021)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--positive_level_policy", choices=["reference_based", "observed_positive"], default="reference_based")
    parser.add_argument("--iv_transform", choices=["log_level", "bounded_logit"], default="bounded_logit")
    parser.add_argument("--iv_lower_bound", type=float, default=1e-4)
    parser.add_argument("--iv_upper_bound", type=float, default=1.0)
    parser.add_argument("--scale_half_life", type=float, default=0.0)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--center_mode", choices=["zero", "ewma_mean"], default="zero")
    parser.add_argument("--drift_feature_mode", choices=["none", "ewma_mean"], default="none")
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    panel, columns = _load_panel(args)
    audits = []
    for split in args.splits:
        arrays = _scope_arrays(
            panel,
            columns,
            args,
            scope=args.state_scope,
            split=split,
            max_windows=int(args.max_windows),
        )
        audits.append(
            {
                "split": split,
                "n_windows": int(arrays["history_norm"].shape[0]),
                "risk_summary": risk_prediction_summary(
                    model,
                    arrays,
                    device=device,
                    batch_size=int(args.batch_size),
                ),
                "ablation": sample_context_ablation(
                    model,
                    arrays,
                    device=device,
                    n_windows=int(args.sample_windows),
                    samples=int(args.samples),
                    n_steps=int(args.n_steps),
                    chunk_size=int(args.chunk_size),
                    seed=int(args.seed),
                ),
            }
        )
    result = {
        "config": {
            "checkpoint": args.checkpoint,
            "state_scope": args.state_scope,
            "splits": args.splits,
            "max_windows": int(args.max_windows),
            "sample_windows": int(args.sample_windows),
            "samples": int(args.samples),
            "n_steps": int(args.n_steps),
            "device": str(device),
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_best_val": payload.get("best_val"),
        },
        "audits": audits,
    }
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(result), indent=2), encoding="utf-8")
    write_markdown(out_md, result)
    print(json.dumps(make_serializable(result), indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
