#!/usr/bin/env python
"""
Frozen 208b event-spread calibration sweep.

Sweep only event spread parameters on saved 208b checkpoints:
  - event_noise_mult
  - event_noise_floor

Keep gate/family selection fixed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows
from experiments.backfill.block_ar.train_207a_local_conditional_family_student_t import evaluate_family_statistics
from experiments.backfill.block_ar.train_208a_h1_teacher_guided_local_family_student_t import compute_h1_shape_stats
from experiments.backfill.block_ar.train_208b_h1_hard_separated_local_family_student_t import load_model


def evaluate_checkpoint_with_spread(
    checkpoint: str,
    event_noise_mult: float,
    event_noise_floor: float,
    history_01: torch.Tensor,
    target_01: torch.Tensor,
    q95: float,
    q99: float,
    batch_size: int,
    eval_samples: int,
    device: torch.device,
) -> dict[str, float]:
    model, payload = load_model(checkpoint, device)
    model.event_noise_mult = float(event_noise_mult)
    model.event_noise_floor = float(event_noise_floor)

    samples = []
    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        samp = model.sample_next_iv(history_01[start:end], n_samples=eval_samples)
        samples.append(samp.detach().cpu().numpy())
    samples = np.concatenate(samples, axis=0)

    target = target_01.detach().cpu().numpy()
    prev = history_01[:, -1].reshape(history_01.shape[0], -1).detach().cpu().numpy()
    q05 = np.quantile(samples, 0.05, axis=1)
    q95s = np.quantile(samples, 0.95, axis=1)
    mean_pred = samples.mean(axis=1)

    realized_abs = np.abs(target - prev)
    q95_mask = realized_abs >= q95
    q99_mask = realized_abs >= q99

    gt_delta = target - prev
    sample_delta = samples - prev[:, None, :]
    shape = compute_h1_shape_stats(gt_delta, sample_delta)

    family_loader = DataLoader(TensorDataset(history_01, target_01), batch_size=batch_size, shuffle=False)
    family_stats = evaluate_family_statistics(model, family_loader)

    metrics = {
        "epoch": int(payload.get("epoch", -1)),
        "event_noise_mult": float(event_noise_mult),
        "event_noise_floor": float(event_noise_floor),
        "coverage_90": float(((target >= q05) & (target <= q95s)).mean()),
        "mae": float(np.abs(mean_pred - target).mean()),
        "width_90": float(np.mean(q95s - q05)),
        "realized_q95_coverage_90": float(((target[q95_mask] >= q05[q95_mask]) & (target[q95_mask] <= q95s[q95_mask])).mean()) if q95_mask.any() else float("nan"),
        "realized_q99_coverage_90": float(((target[q99_mask] >= q05[q99_mask]) & (target[q99_mask] <= q95s[q99_mask])).mean()) if q99_mask.any() else float("nan"),
        "q95_cell_count": int(q95_mask.sum()),
        "q99_cell_count": int(q99_mask.sum()),
        "quiet_ratio": float(shape["quiet_ratio"]),
        "shoulder_ratio": float(shape["shoulder_ratio"]),
        "extreme_ratio": float(shape["extreme_ratio"]),
        "kurtosis_ratio": float(shape["kurtosis_ratio"]),
        "family_top1_mean": float(family_stats["val_family_top1_mean"]),
        "family_entropy_mean": float(family_stats["val_family_entropy_mean"]),
        "gate_prob_mean": float(family_stats["val_gate_prob_mean"]),
    }
    checks = {
        "h1_cov90_ok": 0.85 <= metrics["coverage_90"] <= 0.93,
        "h1_q99_cov_ok": metrics["realized_q99_coverage_90"] >= 0.58,
        "h1_quiet_ok": metrics["quiet_ratio"] >= 0.85,
        "h1_shoulder_ok": metrics["shoulder_ratio"] <= 1.10,
        "h1_kurtosis_ok": metrics["kurtosis_ratio"] >= 0.65,
        "family_top1_ok": metrics["family_top1_mean"] >= 0.40,
    }
    gap = (
        max(0.0, 0.85 - metrics["coverage_90"])
        + max(0.0, metrics["coverage_90"] - 0.93)
        + max(0.0, 0.58 - metrics["realized_q99_coverage_90"])
        + max(0.0, 0.85 - metrics["quiet_ratio"])
        + max(0.0, metrics["shoulder_ratio"] - 1.10)
        + max(0.0, 0.65 - metrics["kurtosis_ratio"])
        + max(0.0, 0.40 - metrics["family_top1_mean"])
    )
    metrics["smoke_gate_checks"] = checks
    metrics["smoke_gate_pass"] = bool(all(checks.values()))
    metrics["n_pass"] = int(sum(checks.values()))
    metrics["gap_score"] = float(gap)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="208b frozen event-spread calibration sweep")
    parser.add_argument("--best_checkpoint", type=str, default="models/backfill/transformer_h1_hard_separated_local_family_student_t_208b_smoke512/best_model.pt")
    parser.add_argument("--final_checkpoint", type=str, default="models/backfill/transformer_h1_hard_separated_local_family_student_t_208b_smoke512/final_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_samples", type=int, default=256)
    parser.add_argument("--noise_mult_grid", type=str, default="0.35,0.5,0.75,1.0,1.25,1.5,2.0")
    parser.add_argument("--noise_floor_grid", type=str, default="0.002,0.005,0.01,0.02")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    train_abs_delta = np.abs(np.diff(surfaces[: args.test_start], axis=0).reshape(-1))
    q95 = float(np.quantile(train_abs_delta, 0.95))
    q99 = float(np.quantile(train_abs_delta, 0.99))

    max_train_idx = args.test_start - args.history_len - 30
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    val_indices = val_indices[: args.max_val_windows]

    surf_tensor = torch.from_numpy(surfaces).to(device)
    history_01, target_01 = build_one_step_windows(val_indices, surf_tensor, args.history_len)

    mult_grid = [float(x) for x in args.noise_mult_grid.split(",") if x.strip()]
    floor_grid = [float(x) for x in args.noise_floor_grid.split(",") if x.strip()]

    all_results = []
    for ckpt_name, ckpt_path in [("best", args.best_checkpoint), ("final", args.final_checkpoint)]:
        for mult in mult_grid:
            for floor in floor_grid:
                metrics = evaluate_checkpoint_with_spread(
                    checkpoint=ckpt_path,
                    event_noise_mult=mult,
                    event_noise_floor=floor,
                    history_01=history_01,
                    target_01=target_01,
                    q95=q95,
                    q99=q99,
                    batch_size=args.batch_size,
                    eval_samples=args.eval_samples,
                    device=device,
                )
                metrics["checkpoint_name"] = ckpt_name
                metrics["checkpoint_path"] = ckpt_path
                all_results.append(metrics)

    all_results.sort(key=lambda row: (-row["n_pass"], row["gap_score"], -row["coverage_90"]))
    best_overall = all_results[0]
    best_by_checkpoint = {}
    for name in ("best", "final"):
        subset = [row for row in all_results if row["checkpoint_name"] == name]
        if subset:
            best_by_checkpoint[name] = subset[0]

    out = {
        "thresholds": {"q95": q95, "q99": q99},
        "grid": {"event_noise_mult": mult_grid, "event_noise_floor": floor_grid},
        "n_evaluated": len(all_results),
        "best_overall": best_overall,
        "best_by_checkpoint": best_by_checkpoint,
        "results": all_results,
    }

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({
        "n_evaluated": len(all_results),
        "best_overall": {
            "checkpoint_name": best_overall["checkpoint_name"],
            "event_noise_mult": best_overall["event_noise_mult"],
            "event_noise_floor": best_overall["event_noise_floor"],
            "n_pass": best_overall["n_pass"],
            "gap_score": best_overall["gap_score"],
            "coverage_90": best_overall["coverage_90"],
            "realized_q99_coverage_90": best_overall["realized_q99_coverage_90"],
            "quiet_ratio": best_overall["quiet_ratio"],
            "shoulder_ratio": best_overall["shoulder_ratio"],
            "kurtosis_ratio": best_overall["kurtosis_ratio"],
        },
    }, indent=2))


if __name__ == "__main__":
    main()
