#!/usr/bin/env python
"""
Pre-implementation test for a localized marked-event AR model idea.

We compare the hard-cell miss severity pattern against:
  1. a fixed global template
  2. 194c's current event uplift map
  3. the best localized single-anchor kernel

If the best localized kernel consistently fits the miss severity better than the
current whole-step event uplift/template, that supports moving to a localized
marked-event carrier.
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
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import denormalize_iv


GRID = 5


def coords(idx: int) -> tuple[int, int]:
    return divmod(idx, GRID)


def make_kernel(anchor: int, family: str) -> np.ndarray:
    ax, ay = coords(anchor)
    ker = np.zeros((GRID, GRID), dtype=np.float64)
    for i in range(GRID):
        for j in range(GRID):
            dx = abs(i - ax)
            dy = abs(j - ay)
            if family == "delta":
                val = 1.0 if (dx == 0 and dy == 0) else 0.0
            elif family == "cross1":
                val = 1.0 if (dx + dy) <= 1 else 0.0
            elif family == "square1":
                val = 1.0 if max(dx, dy) <= 1 else 0.0
            elif family == "gauss08":
                val = float(np.exp(-0.5 * (dx * dx + dy * dy) / (0.8 * 0.8)))
            elif family == "gauss14":
                val = float(np.exp(-0.5 * (dx * dx + dy * dy) / (1.4 * 1.4)))
            else:
                raise ValueError(f"unknown family {family}")
            ker[i, j] = val
    flat = ker.reshape(-1)
    s = flat.sum()
    if s <= 1e-12:
        flat[:] = 1.0 / flat.size
    else:
        flat /= s
    return flat


def overlap_score(target: np.ndarray, proposal: np.ndarray) -> float:
    t = np.asarray(target, dtype=np.float64).reshape(-1)
    p = np.asarray(proposal, dtype=np.float64).reshape(-1)
    if t.sum() <= 1e-12 or p.sum() <= 1e-12:
        return float("nan")
    t = t / t.sum()
    p = p / p.sum()
    return float(np.minimum(t, p).sum())


def norm_entropy(x: np.ndarray) -> float:
    p = np.asarray(x, dtype=np.float64).reshape(-1)
    if p.sum() <= 1e-12:
        return float("nan")
    p = p / p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(25.0))


@torch.no_grad()
def rollout_intervals(
    model,
    history_norm: torch.Tensor,
    future_len: int,
    device: torch.device,
    n_samples: int,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    lo_chunks = []
    hi_chunks = []
    n = history_norm.shape[0]
    for start in range(0, n, batch_size):
        hist_b = history_norm[start : start + batch_size].to(device)
        samples = model.sample_batched(hist_b, n_samples=n_samples, n_steps=future_len)
        lo = samples.quantile(0.05, dim=1)
        hi = samples.quantile(0.95, dim=1)
        lo_chunks.append(lo.detach().cpu().numpy().reshape(lo.shape[0], lo.shape[1], -1))
        hi_chunks.append(hi.detach().cpu().numpy().reshape(hi.shape[0], hi.shape[1], -1))
    return np.concatenate(lo_chunks, axis=0), np.concatenate(hi_chunks, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Localized event feasibility test for 195a")
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
        default="results/validations/2026-04-07/analysis/195a_localized_event_feasibility",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history_norm, future_norm = build_test_subset(
        args.data_path, args.history_len, args.future_len, args.test_start, args.max_windows
    )
    future_01 = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1).cpu().numpy()
    _vov, q20, q80, calm_mask, turb_mask = regime_masks_from_history(history_norm)

    model, _payload = load_194a_model(Path(args.ckpt), device)
    tf_diag = analyze_194a_teacher_forced(
        model, history_norm, future_norm, device=device, batch_size=args.batch_size
    )
    lo, hi = rollout_intervals(
        model,
        history_norm,
        future_len=args.future_len,
        device=device,
        n_samples=args.rollout_samples,
        batch_size=args.batch_size,
    )

    hidx = 29
    y = future_01[:, hidx, :]
    width = np.maximum(hi[:, hidx, :] - lo[:, hidx, :], 1e-6)
    severity = np.maximum(lo[:, hidx, :] - y, 0.0) + np.maximum(y - hi[:, hidx, :], 0.0)
    severity = severity / width

    event_uplift = np.clip(tf_diag["state_var_cells"][:, hidx, 1, :] - tf_diag["state_var_cells"][:, hidx, 0, :], 0.0, None)
    global_template = event_uplift[turb_mask].mean(axis=0)

    families = ["delta", "cross1", "square1", "gauss08", "gauss14"]
    kernels = {(anchor, fam): make_kernel(anchor, fam) for anchor in range(25) for fam in families}

    hard_windows = turb_mask & (severity.sum(axis=1) > 0)

    local_scores = []
    current_scores = []
    template_scores = []
    local_beats_current = []
    local_beats_template = []
    best_anchor_list = []
    best_family_list = []
    target_entropy = []
    current_entropy = []
    target_spread = []

    grid_coords = np.array([coords(i) for i in range(25)], dtype=np.float64)

    for i in np.where(hard_windows)[0]:
        target = severity[i]
        if target.sum() <= 1e-12:
            continue
        target_n = target / target.sum()
        current = event_uplift[i]
        template = global_template

        best_score = -1.0
        best_anchor = None
        best_family = None
        best_kernel = None
        for (anchor, fam), ker in kernels.items():
            score = overlap_score(target_n, ker)
            if score > best_score:
                best_score = score
                best_anchor = anchor
                best_family = fam
                best_kernel = ker

        cur_score = overlap_score(target_n, current)
        tpl_score = overlap_score(target_n, template)

        local_scores.append(best_score)
        current_scores.append(cur_score)
        template_scores.append(tpl_score)
        local_beats_current.append(float(best_score > cur_score))
        local_beats_template.append(float(best_score > tpl_score))
        best_anchor_list.append(best_anchor)
        best_family_list.append(best_family)
        target_entropy.append(norm_entropy(target_n))
        current_entropy.append(norm_entropy(current))

        anchor_xy = grid_coords[best_anchor]
        d2 = np.square(grid_coords - anchor_xy).sum(axis=1)
        target_spread.append(float((target_n * np.sqrt(d2)).sum()))

    anchor_hist = np.bincount(np.asarray(best_anchor_list, dtype=np.int64), minlength=25).astype(np.float64)
    family_counts = {fam: int(sum(1 for f in best_family_list if f == fam)) for fam in families}

    summary: dict[str, Any] = {
        "config": {
            "test_start": args.test_start,
            "max_windows": args.max_windows,
            "rollout_samples": args.rollout_samples,
            "batch_size": args.batch_size,
            "q20_vov": q20,
            "q80_vov": q80,
            "turb_windows": int(turb_mask.sum()),
            "hard_windows": int(hard_windows.sum()),
        },
        "fit_scores": {
            "best_local_kernel_overlap_mean": float(np.mean(local_scores)) if local_scores else float("nan"),
            "current_event_uplift_overlap_mean": float(np.mean(current_scores)) if current_scores else float("nan"),
            "global_template_overlap_mean": float(np.mean(template_scores)) if template_scores else float("nan"),
            "best_local_beats_current_frac": float(np.mean(local_beats_current)) if local_beats_current else float("nan"),
            "best_local_beats_template_frac": float(np.mean(local_beats_template)) if local_beats_template else float("nan"),
        },
        "localization": {
            "target_entropy_norm_mean": float(np.mean(target_entropy)) if target_entropy else float("nan"),
            "current_event_entropy_norm_mean": float(np.mean(current_entropy)) if current_entropy else float("nan"),
            "target_spread_radius_mean": float(np.mean(target_spread)) if target_spread else float("nan"),
            "best_anchor_entropy_norm": norm_entropy(anchor_hist),
            "best_anchor_freq": anchor_hist.tolist(),
            "best_family_counts": family_counts,
        },
        "calm_vs_turb_current_event": {
            "mean_current_event_uplift_calm_h30": float(event_uplift[calm_mask].mean()) if calm_mask.any() else float("nan"),
            "mean_current_event_uplift_turb_h30": float(event_uplift[turb_mask].mean()) if turb_mask.any() else float("nan"),
        },
        "interpretation": {
            "fit_read": "If the best local kernel consistently beats the current event uplift and the fixed template, a localized marked-event carrier is a better abstraction than a whole-step event regime.",
            "anchor_read": "If best anchors vary materially across windows, the event location is conditional and cannot be represented by a fixed template.",
            "spread_read": "A modest target spread radius supports an anchor-plus-kernel event object rather than a whole-surface broadening regime.",
        },
    }

    out_path = output_dir / "summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(make_serializable(summary), f, indent=2)

    print(json.dumps(make_serializable(summary), indent=2))
    print(f"\nSaved localized-event feasibility summary to {out_path}")


if __name__ == "__main__":
    main()
