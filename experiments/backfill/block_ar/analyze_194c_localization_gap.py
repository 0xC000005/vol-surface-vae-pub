#!/usr/bin/env python
"""
Targeted pre-implementation test for whether 194c's remaining miss is
"whole-step regime too coarse" rather than another training issue.

Questions:
  1. On turbulent h30 windows with misses, are the missed cells sparse/localized?
  2. Is the model's event uplift broad relative to the underfit concentration?
  3. Does the event uplift overlap the hard cells enough to justify staying with
     a whole-step event state, or is a localized event carrier needed?
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

from experiments.backfill.block_ar.analyze_194c_two_state_mechanism import (
    analyze_194a_teacher_forced,
    build_test_subset,
    load_194a_model,
    make_serializable,
    regime_masks_from_history,
    rollout_coverage_arrays,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import denormalize_iv


def concentration_stats(arr: np.ndarray) -> dict[str, float]:
    x = np.asarray(arr, dtype=np.float64).reshape(-1)
    x = np.clip(x, 0.0, None)
    total = float(x.sum())
    if total <= 1e-12:
        return {
            "top1_share": float("nan"),
            "top3_share": float("nan"),
            "eff_support": float("nan"),
            "norm_entropy": float("nan"),
        }
    p = x / total
    p_sorted = np.sort(p)[::-1]
    top1 = float(p_sorted[:1].sum())
    top3 = float(p_sorted[:3].sum())
    eff_support = float(1.0 / np.square(p).sum())
    entropy = float(-(p[p > 0] * np.log(p[p > 0])).sum())
    norm_entropy = float(entropy / np.log(len(p)))
    return {
        "top1_share": top1,
        "top3_share": top3,
        "eff_support": eff_support,
        "norm_entropy": norm_entropy,
    }


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def topk_overlap_score(weights: np.ndarray, hard_mask: np.ndarray, k: int) -> float:
    idx = np.argsort(np.asarray(weights, dtype=np.float64))[::-1][:k]
    return float(np.asarray(hard_mask, dtype=np.float64)[idx].mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Localization-gap analysis for 194c")
    parser.add_argument("--ckpt", type=str, default="models/backfill/regime_switching_ar_latent_factor_194c/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--max_windows", type=int, default=384)
    parser.add_argument("--rollout_samples", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/194c_localization_gap",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_norm, future_norm = build_test_subset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        max_windows=args.max_windows,
    )
    _vov, q20, q80, calm_mask, turb_mask = regime_masks_from_history(history_norm)
    model, _payload = load_194a_model(Path(args.ckpt), device)

    tf_diag = analyze_194a_teacher_forced(
        model,
        history_norm,
        future_norm,
        device=device,
        batch_size=args.batch_size,
    )
    rollout_diag = rollout_coverage_arrays(
        model,
        history_norm,
        future_norm,
        device=device,
        n_samples=args.rollout_samples,
        batch_size=args.batch_size,
    )

    future_01 = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1).cpu().numpy()
    mu_bar = tf_diag["mu_bar_iv"]
    expected_var = np.clip(tf_diag["expected_var_cells"], 1e-8, None)
    state_var = tf_diag["state_var_cells"]
    event_uplift = np.clip(state_var[:, :, 1, :] - state_var[:, :, 0, :], 0.0, None)

    hidx = 29
    hard_h30 = (~rollout_diag["inside90"][:, hidx, :]) & turb_mask[:, None]
    clean_h30 = rollout_diag["inside90"][:, hidx, :] & turb_mask[:, None]
    hard_windows = hard_h30.any(axis=1)
    late_turb_windows = turb_mask

    lo_h30 = np.quantile(rollout_diag["future_01"][:, hidx, :][None, ...], 0.0)  # placeholder to satisfy type flow
    del lo_h30
    # Use rollout miss-excess as the target localization signal because it is
    # directly aligned with the failing hard cells.
    # Reconstruct per-cell h30 miss-excess from rollout summaries.
    # We do not have lo/hi saved, so approximate hard-cell concentration from the
    # binary hard mask itself and from teacher-forced standardized underfit.
    # The binary hard mask gives the directly benchmarked localization target.
    sq_err = np.square(future_01[:, hidx, :] - mu_bar[:, hidx, :])
    underfit = np.clip(sq_err / expected_var[:, hidx, :] - 1.0e-3, 0.0, None)
    uplift_h30 = event_uplift[:, hidx, :]

    hard_counts = hard_h30[hard_windows].sum(axis=1).astype(np.float64)
    hard_top1 = []
    hard_top3 = []
    missmask_top1 = []
    missmask_top3 = []
    uplift_top1 = []
    uplift_top3 = []
    uplift_eff = []
    uplift_entropy = []
    uplift_top1_hit = []
    uplift_top3_hit = []
    missmask_top1_hit = []
    missmask_top3_hit = []
    top1_cells = []

    for i in np.where(hard_windows)[0]:
        hard_mask_i = hard_h30[i]
        count_i = int(hard_mask_i.sum())
        hard_top1.append(1.0 / count_i)
        hard_top3.append(min(3, count_i) / count_i)

        missmask_i = hard_mask_i.astype(np.float64)
        uplift_i = uplift_h30[i]

        uf_stats = concentration_stats(missmask_i)
        eu_stats = concentration_stats(uplift_i)
        missmask_top1.append(uf_stats["top1_share"])
        missmask_top3.append(uf_stats["top3_share"])
        uplift_top1.append(eu_stats["top1_share"])
        uplift_top3.append(eu_stats["top3_share"])
        uplift_eff.append(eu_stats["eff_support"])
        uplift_entropy.append(eu_stats["norm_entropy"])

        uplift_top1_hit.append(topk_overlap_score(uplift_i, hard_mask_i, k=1))
        uplift_top3_hit.append(topk_overlap_score(uplift_i, hard_mask_i, k=3))
        missmask_top1_hit.append(topk_overlap_score(missmask_i, hard_mask_i, k=1))
        missmask_top3_hit.append(topk_overlap_score(missmask_i, hard_mask_i, k=3))
        top1_cells.append(int(np.argmax(underfit[i])))

    top1_hist = np.bincount(np.asarray(top1_cells, dtype=np.int64), minlength=25).astype(np.float64)
    top1_probs = top1_hist / max(top1_hist.sum(), 1.0)
    top1_entropy = float(-(top1_probs[top1_probs > 0] * np.log(top1_probs[top1_probs > 0])).sum() / np.log(25.0))

    summary: dict[str, Any] = {
        "config": {
            "test_start": args.test_start,
            "max_windows": args.max_windows,
            "rollout_samples": args.rollout_samples,
            "batch_size": args.batch_size,
            "q20_vov": q20,
            "q80_vov": q80,
            "late_turb_windows": int(late_turb_windows.sum()),
            "hard_h30_windows": int(hard_windows.sum()),
        },
        "hard_h30_sparsity": {
            "mean_hard_cell_count": safe_mean(hard_counts.tolist()),
            "median_hard_cell_count": float(np.median(hard_counts)) if hard_counts.size else float("nan"),
            "mean_binary_top1_share": safe_mean(hard_top1),
            "mean_binary_top3_share": safe_mean(hard_top3),
        },
        "localization_gap": {
            "hard_mask_top1_share": safe_mean(missmask_top1),
            "hard_mask_top3_share": safe_mean(missmask_top3),
            "event_uplift_top1_share": safe_mean(uplift_top1),
            "event_uplift_top3_share": safe_mean(uplift_top3),
            "event_uplift_eff_support": safe_mean(uplift_eff),
            "event_uplift_norm_entropy": safe_mean(uplift_entropy),
        },
        "topk_hit_rates": {
            "hard_mask_top1_hits_hard": safe_mean(missmask_top1_hit),
            "hard_mask_top3_hits_hard": safe_mean(missmask_top3_hit),
            "event_uplift_top1_hits_hard": safe_mean(uplift_top1_hit),
            "event_uplift_top3_hits_hard": safe_mean(uplift_top3_hit),
        },
        "window_specificity": {
            "underfit_top1_cell_entropy_norm": top1_entropy,
            "top1_cell_freq": top1_hist.tolist(),
        },
        "calm_vs_turb_event_uplift_h30": {
            "mean_event_uplift_calm_h30": float(uplift_h30[calm_mask].mean()) if calm_mask.any() else float("nan"),
            "mean_event_uplift_turb_h30": float(uplift_h30[turb_mask].mean()) if turb_mask.any() else float("nan"),
        },
        "interpretation": {
            "sparse_hard_windows_mean": "If hard h30 windows involve only a modest number of cells, the remaining miss is localized rather than whole-surface.",
            "localization_gap": "If underfit concentration is sharper than event uplift, the 194c event state is too broad to carry the remaining frontier.",
            "window_specificity": "If the top underfit cell varies a lot across windows, a whole-step event state is too coarse and a localized event object is justified.",
        },
    }

    out_path = output_dir / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(make_serializable(summary), f, indent=2)

    print(json.dumps(make_serializable(summary), indent=2))
    print(f"\nSaved localization summary to {out_path}")


if __name__ == "__main__":
    main()
