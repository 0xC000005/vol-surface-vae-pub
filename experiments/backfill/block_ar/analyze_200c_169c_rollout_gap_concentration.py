#!/usr/bin/env python
"""
Measure whether the 169c teacher-forced vs rollout gap concentrates on sparse hard windows.
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

from experiments.backfill.block_ar.analyze_h7_mechanisms import (
    build_test_subset,
    load_model,
    make_serializable,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import denormalize_iv


def subset_stats(mask: np.ndarray, cov: np.ndarray, width: np.ndarray, mae: np.ndarray) -> dict[str, float]:
    if not np.any(mask):
        return {
            "windows": 0,
            "cov90": float("nan"),
            "width90": float("nan"),
            "mae": float("nan"),
        }
    return {
        "windows": int(mask.sum()),
        "cov90": float(cov[mask].mean()),
        "width90": float(width[mask].mean()),
        "mae": float(mae[mask].mean()),
    }


def sample_teacher_forced_horizon(
    model,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    hidx: int,
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outs = []
    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        hist = history_01[start:end].to(device)
        fut = future_01[start:end].to(device)
        if hidx > 0:
            tf_hist = torch.cat([hist[:, hidx:], fut[:, :hidx]], dim=1)
        else:
            tf_hist = hist
        samp = model.sample_next_iv(tf_hist, n_samples=n_samples).detach().cpu().numpy()
        outs.append(samp)
    return np.concatenate(outs, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rollout-gap concentration probe for 169c")
    parser.add_argument("--ckpt", type=str, default="models/backfill/student_t_169c/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--max_windows", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/200_design/200c_169c_rollout_gap_concentration",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, model_type, _payload = load_model(args.ckpt, str(device))
    if model_type != "multi_step_student_t_169c":
        raise ValueError(f"Expected 169c checkpoint, got {model_type}")

    history_norm, future_norm = build_test_subset(
        args.data_path,
        args.history_len,
        args.future_len,
        args.test_start,
        args.max_windows,
    )
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm)
    future_flat = future_01.reshape(future_01.shape[0], future_01.shape[1], -1).numpy()
    target_h30 = future_flat[:, -1, :]

    hist_mean = history_01.mean(dim=(-1, -2)).numpy()
    vov = np.diff(hist_mean, axis=1).std(axis=1)
    q80_vov = float(np.quantile(vov, 0.8))
    turb = vov >= q80_vov
    prev = history_01[:, -1].reshape(history_01.shape[0], -1).numpy()
    h30_energy = np.abs(target_h30 - prev).sum(axis=1)
    q80_h30_turb = float(np.quantile(h30_energy[turb], 0.8))
    hard_late = turb & (h30_energy >= q80_h30_turb)
    clean_late_turb = turb & ~hard_late
    calm = vov <= float(np.quantile(vov, 0.2))

    tf_samples = sample_teacher_forced_horizon(
        model,
        history_01,
        future_01,
        hidx=29,
        n_samples=args.samples,
        batch_size=args.batch_size,
        device=device,
    )
    tf_lo = np.quantile(tf_samples, 0.05, axis=1)
    tf_hi = np.quantile(tf_samples, 0.95, axis=1)
    tf_mean = tf_samples.mean(axis=1)

    rollout = model.sample_batched(
        history_norm.to(device),
        n_samples=args.samples,
        n_steps=30,
        chunk_size=min(args.batch_size, args.samples),
    ).detach().cpu().numpy()
    ro_h30 = rollout[:, :, 29].reshape(history_01.shape[0], args.samples, -1)
    ro_lo = np.quantile(ro_h30, 0.05, axis=1)
    ro_hi = np.quantile(ro_h30, 0.95, axis=1)
    ro_mean = ro_h30.mean(axis=1)

    tf_cov = ((target_h30 >= tf_lo) & (target_h30 <= tf_hi)).mean(axis=1)
    tf_width = (tf_hi - tf_lo).mean(axis=1)
    tf_mae = np.abs(tf_mean - target_h30).mean(axis=1)

    ro_cov = ((target_h30 >= ro_lo) & (target_h30 <= ro_hi)).mean(axis=1)
    ro_width = (ro_hi - ro_lo).mean(axis=1)
    ro_mae = np.abs(ro_mean - target_h30).mean(axis=1)

    subsets = {
        "all": np.ones(history_01.shape[0], dtype=bool),
        "calm": calm,
        "turb": turb,
        "hard_late": hard_late,
        "clean_late_turb": clean_late_turb,
    }

    summary: dict[str, Any] = {
        "config": {
            "ckpt": args.ckpt,
            "max_windows": int(history_01.shape[0]),
            "samples": args.samples,
        },
        "subset_thresholds": {
            "q80_vov": q80_vov,
            "q80_h30_energy_within_turb": q80_h30_turb,
        },
        "subsets": {},
        "interpretation": {
            "read": (
                "If the teacher-forced vs rollout gap is much larger on hard_late than elsewhere, "
                "rollout-consistent training is directly justified for the sparse-case AR phase."
            )
        },
    }

    for name, mask in subsets.items():
        tf_stats = subset_stats(mask, tf_cov, tf_width, tf_mae)
        ro_stats = subset_stats(mask, ro_cov, ro_width, ro_mae)
        summary["subsets"][name] = {
            "teacher_forced": tf_stats,
            "rollout": ro_stats,
            "gap_rollout_minus_teacher_forced": {
                "cov90": ro_stats["cov90"] - tf_stats["cov90"],
                "width90": ro_stats["width90"] - tf_stats["width90"],
                "mae": ro_stats["mae"] - tf_stats["mae"],
            },
        }

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(make_serializable(summary), indent=2))
    print(json.dumps(make_serializable(summary), indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
