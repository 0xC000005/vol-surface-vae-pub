#!/usr/bin/env python
"""
Oracle H=1 event-spread geometry audit.

Question:
  After choosing the right local severe-event family member, what residual spread
  geometry is needed to explain the remaining one-step conditional law?

We hold family selection oracle-fixed and compare a few simple spread objects:
  1. scalar isotropic
  2. shape-scaled diagonal (closest to 208b event spread)
  3. empirical diagonal
  4. local low-rank + isotropic residual
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_201b_targeting_jump_mechanism import make_serializable
from experiments.backfill.block_ar.analyze_202a_discrete_innovation_mode_pretest import (
    build_split_indices,
    build_window_metadata,
    load_model,
)
from experiments.backfill.block_ar.analyze_205a_conditional_shape_family_audit import collect_teacher_forced_records
from experiments.backfill.block_ar.analyze_206b_local_conditional_scenario_family_pretest import (
    farthest_first_subset,
    fit_feature_space,
    transform_feat,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows
from experiments.backfill.block_ar.train_208a_h1_teacher_guided_local_family_student_t import compute_h1_shape_stats


def normalize_pattern(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return x / max(np.abs(x).sum(), eps)


def assign_family_members(patterns: np.ndarray, family: np.ndarray) -> np.ndarray:
    cost = np.abs(patterns[:, None, :] - family[None, :, :]).mean(axis=-1)
    return cost.argmin(axis=1)


def amplitude_projection(delta: np.ndarray, pattern: np.ndarray, eps: float = 1e-8) -> float:
    denom = float(np.dot(pattern, pattern))
    return float(np.dot(delta, pattern) / max(denom, eps))


def build_lowrank_cov(centered: np.ndarray, rank: int, var_floor: float) -> np.ndarray:
    n, d = centered.shape
    if n <= 1:
        return np.eye(d, dtype=np.float64) * var_floor
    cov = centered.T @ centered / max(n - 1, 1)
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0.0, None)
    evecs = evecs[:, order]
    keep = min(rank, d)
    kept = evals[:keep]
    kept_vec = evecs[:, :keep]
    tail = evals[keep:]
    noise = float(np.mean(tail)) if tail.size else var_floor
    noise = max(noise, var_floor)
    return kept_vec @ np.diag(kept) @ kept_vec.T + np.eye(d, dtype=np.float64) * noise


def build_shape_scaled_diag(centered: np.ndarray, pattern: np.ndarray, var_floor: float) -> np.ndarray:
    var = centered.var(axis=0) + var_floor
    x = np.stack([np.ones_like(pattern), np.abs(pattern)], axis=1)
    coef, *_ = np.linalg.lstsq(x, var, rcond=None)
    alpha = max(float(coef[0]), 0.0)
    beta = max(float(coef[1]), 0.0)
    fitted = alpha + beta * np.abs(pattern)
    fitted = np.maximum(fitted, var_floor)
    return np.diag(fitted.astype(np.float64))


def gaussian_logpdf(x: np.ndarray, mean: np.ndarray, cov: np.ndarray, jitter: float = 1e-8) -> float:
    d = x.shape[0]
    cov = cov + np.eye(d, dtype=np.float64) * jitter
    sign, logdet = np.linalg.slogdet(cov)
    if sign <= 0:
        return float("-inf")
    diff = x - mean
    solve = np.linalg.solve(cov, diff)
    quad = float(diff @ solve)
    return float(-0.5 * (d * np.log(2.0 * np.pi) + logdet + quad))


def sample_gaussian(
    rng: np.random.Generator,
    mean: np.ndarray,
    cov: np.ndarray,
    n_samples: int,
    jitter: float = 1e-8,
) -> np.ndarray:
    cov = cov + np.eye(mean.shape[0], dtype=np.float64) * jitter
    return rng.multivariate_normal(mean=mean, cov=cov, size=n_samples).astype(np.float32)


def evaluate_subset(
    subset_name: str,
    subset_idx: np.ndarray,
    train_records: dict[str, np.ndarray],
    test_records: dict[str, np.ndarray],
    train_delta: np.ndarray,
    test_delta: np.ndarray,
    q99: float,
    feature_bundle: dict[str, Any],
    nn: NearestNeighbors,
    train_pool_idx: np.ndarray,
    family_size: int,
    knn: int,
    min_cluster: int,
    lowrank_rank: int,
    eval_samples: int,
    seed: int,
) -> dict[str, Any]:
    if subset_idx.size == 0:
        return {"subset": subset_name, "status": "empty"}

    rng = np.random.default_rng(seed)
    query_feat = transform_feat(test_records["cond_feat"][subset_idx], feature_bundle)
    _, nbr_pos = nn.kneighbors(query_feat, return_distance=True)
    nbr_idx = train_pool_idx[nbr_pos]

    geom_names = ["scalar", "shape_diag", "diag", "lowrank"]
    sample_store: dict[str, list[np.ndarray]] = {k: [] for k in geom_names}
    loglik_store: dict[str, list[float]] = {k: [] for k in geom_names}
    cluster_sizes: list[float] = []
    family_mae: list[float] = []
    resid_mean_norms: list[float] = []
    oracle_top1_hits: list[float] = []

    for local_q_pos, q_abs in enumerate(subset_idx):
        neighbors = nbr_idx[local_q_pos]
        neigh_patterns = train_records["signed_delta_norm"][neighbors]
        fam_sel = farthest_first_subset(neigh_patterns, family_size)
        family = neigh_patterns[fam_sel]

        q_pattern = test_records["signed_delta_norm"][q_abs]
        q_delta = test_delta[q_abs]
        family_cost = np.abs(family - q_pattern[None, :]).mean(axis=1)
        chosen = int(family_cost.argmin())
        pattern = normalize_pattern(family[chosen])
        family_mae.append(float(family_cost[chosen]))
        oracle_top1_hits.append(float(int(np.abs(pattern).argmax()) == int(test_records["top_cell"][q_abs])))

        neigh_assign = assign_family_members(neigh_patterns, family)
        cluster = neighbors[neigh_assign == chosen]
        if cluster.size < min_cluster:
            cluster = neighbors
        cluster_sizes.append(float(cluster.size))

        cluster_delta = train_delta[cluster]
        cluster_amp = np.array([amplitude_projection(d, pattern) for d in cluster_delta], dtype=np.float64)
        cluster_resid = cluster_delta - cluster_amp[:, None] * pattern[None, :]
        resid_mean = cluster_resid.mean(axis=0)
        centered = cluster_resid - resid_mean[None, :]
        resid_mean_norms.append(float(np.linalg.norm(resid_mean)))

        q_amp = amplitude_projection(q_delta, pattern)
        mean = q_amp * pattern + resid_mean

        d = pattern.shape[0]
        var_floor = 1e-6
        scalar_var = float(centered.var() + var_floor)
        scalar_cov = np.eye(d, dtype=np.float64) * scalar_var
        shape_diag_cov = build_shape_scaled_diag(centered, pattern, var_floor=var_floor)
        diag_cov = np.diag(centered.var(axis=0) + var_floor)
        lowrank_cov = build_lowrank_cov(centered, rank=lowrank_rank, var_floor=var_floor)
        covs = {
            "scalar": scalar_cov,
            "shape_diag": shape_diag_cov,
            "diag": diag_cov,
            "lowrank": lowrank_cov,
        }

        for name, cov in covs.items():
            loglik_store[name].append(gaussian_logpdf(q_delta.astype(np.float64), mean.astype(np.float64), cov))
            sample_store[name].append(sample_gaussian(rng, mean.astype(np.float64), cov, n_samples=eval_samples))

    out: dict[str, Any] = {
        "subset": subset_name,
        "n_queries": int(subset_idx.size),
        "mean_cluster_size": float(np.mean(cluster_sizes)),
        "mean_oracle_family_mae": float(np.mean(family_mae)),
        "mean_resid_mean_norm": float(np.mean(resid_mean_norms)),
        "oracle_pattern_top1_acc": float(np.mean(oracle_top1_hits)),
        "family_size": int(family_size),
        "knn": int(knn),
        "min_cluster": int(min_cluster),
    }

    gt_delta = test_delta[subset_idx]
    gt_abs = np.abs(gt_delta)
    q99_mask = gt_abs >= q99

    for name in geom_names:
        samples = np.stack(sample_store[name], axis=0)
        q05 = np.quantile(samples, 0.05, axis=1)
        q95v = np.quantile(samples, 0.95, axis=1)
        mean_pred = samples.mean(axis=1)
        coverage = float(((gt_delta >= q05) & (gt_delta <= q95v)).mean())
        q99_cov = (
            float(((gt_delta[q99_mask] >= q05[q99_mask]) & (gt_delta[q99_mask] <= q95v[q99_mask])).mean())
            if q99_mask.any()
            else float("nan")
        )
        shape = compute_h1_shape_stats(gt_delta, samples)
        prefix = f"{name}_"
        out[prefix + "loglik_mean"] = float(np.mean(loglik_store[name]))
        out[prefix + "coverage_90"] = coverage
        out[prefix + "realized_q99_coverage_90"] = q99_cov
        out[prefix + "mae"] = float(np.abs(mean_pred - gt_delta).mean())
        out[prefix + "width_90"] = float(np.mean(q95v - q05))
        out[prefix + "quiet_ratio"] = float(shape["quiet_ratio"])
        out[prefix + "shoulder_ratio"] = float(shape["shoulder_ratio"])
        out[prefix + "extreme_ratio"] = float(shape["extreme_ratio"])
        out[prefix + "kurtosis_ratio"] = float(shape["kurtosis_ratio"])
        out[prefix + "q95_ratio"] = float(shape["q95_ratio"])
        out[prefix + "q99_ratio"] = float(shape["q99_ratio"])

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="209a oracle H1 event spread geometry audit")
    parser.add_argument("--checkpoint", type=str, default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--knn", type=int, default=32)
    parser.add_argument("--family_size", type=int, default=3)
    parser.add_argument("--min_cluster", type=int, default=8)
    parser.add_argument("--lowrank_rank", type=int, default=3)
    parser.add_argument("--eval_samples", type=int, default=512)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-08/analysis/209_design/209a_h1_oracle_event_spread_geometry",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    surfaces = np.load(args.data_path)["surface"].astype(np.float32)
    train_abs_delta = np.abs(np.diff(surfaces[: args.test_start], axis=0).reshape(-1))
    q95 = float(np.quantile(train_abs_delta, 0.95))
    q99 = float(np.quantile(train_abs_delta, 0.99))

    split_indices = build_split_indices(
        surface_len=surfaces.shape[0],
        history_len=args.history_len,
        future_len=1,
        test_start=args.test_start,
        val_size=args.val_size,
    )

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_history, train_target = build_one_step_windows(split_indices["train"], surf_tensor, args.history_len)
    test_history, test_target = build_one_step_windows(split_indices["test"], surf_tensor, args.history_len)

    train_future = train_target.unsqueeze(1)
    test_future = test_target.unsqueeze(1)
    train_future_np = train_future.detach().cpu().numpy()
    test_future_np = test_future.detach().cpu().numpy()
    train_history_np = train_history.detach().cpu().numpy()
    test_history_np = test_history.detach().cpu().numpy()
    train_meta = build_window_metadata(train_history_np, train_future_np)
    test_meta = build_window_metadata(
        test_history_np,
        test_future_np,
        q80_vov_train=train_meta["q80_vov"],
        q80_h30_turb_train=train_meta["q80_h30_turb"],
    )

    teacher_model, payload = load_model(args.checkpoint, device)
    train_records = collect_teacher_forced_records(
        model=teacher_model,
        history_01=train_history,
        future_flat=train_future,
        window_meta=train_meta,
        split_name="train",
        q95=q95,
        q99=q99,
        batch_size=args.batch_size,
        device=device,
    )
    test_records = collect_teacher_forced_records(
        model=teacher_model,
        history_01=test_history,
        future_flat=test_future,
        window_meta=test_meta,
        split_name="test",
        q95=q95,
        q99=q99,
        batch_size=args.batch_size,
        device=device,
    )

    train_prev = train_history[:, -1].reshape(train_history.shape[0], -1).detach().cpu().numpy()
    test_prev = test_history[:, -1].reshape(test_history.shape[0], -1).detach().cpu().numpy()
    train_delta = train_target.detach().cpu().numpy() - train_prev
    test_delta = test_target.detach().cpu().numpy() - test_prev

    train_pool_mask = train_records["q99_any"] == 1
    train_pool_idx = np.flatnonzero(train_pool_mask)
    feat_bundle, train_z = fit_feature_space(train_records["cond_feat"][train_pool_idx], pca_dim=32)
    nn = NearestNeighbors(n_neighbors=min(args.knn, train_pool_idx.size), metric="euclidean")
    nn.fit(train_z)

    test_q99 = test_records["q99_any"] == 1
    subsets = {
        "all_q99": np.flatnonzero(test_q99),
        "turb_q99": np.flatnonzero(test_q99 & (test_records["turb"] == 1)),
        "hard_window_q99": np.flatnonzero(test_q99 & (test_records["hard_late_window"] == 1)),
    }

    subset_results = {}
    for i, (name, idx) in enumerate(subsets.items()):
        subset_results[name] = evaluate_subset(
            subset_name=name,
            subset_idx=idx,
            train_records=train_records,
            test_records=test_records,
            train_delta=train_delta,
            test_delta=test_delta,
            q99=q99,
            feature_bundle=feat_bundle,
            nn=nn,
            train_pool_idx=train_pool_idx,
            family_size=args.family_size,
            knn=args.knn,
            min_cluster=args.min_cluster,
            lowrank_rank=args.lowrank_rank,
            eval_samples=args.eval_samples,
            seed=args.seed + i,
        )

    summary = {
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
        "q95": q95,
        "q99": q99,
        "family_size": args.family_size,
        "knn": args.knn,
        "min_cluster": args.min_cluster,
        "lowrank_rank": args.lowrank_rank,
        "eval_samples": args.eval_samples,
        "subset_results": make_serializable(subset_results),
    }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
