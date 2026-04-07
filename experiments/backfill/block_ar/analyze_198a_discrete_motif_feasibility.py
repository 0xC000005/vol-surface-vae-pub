#!/usr/bin/env python
"""
Pretest for 198a: learned discrete future motifs / scenario tokens.

Question:
  Can a small discrete codebook of future residual blocks around the 183c anchor
  represent the held-out future motifs materially better than a single global
  residual template, and is the motif identity at least partially predictable
  from the 183c history embedding?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_170d_mechanisms import (
    build_test_subset,
    make_serializable,
    regime_masks_from_history,
)
from experiments.backfill.block_ar.analyze_183c_best_mechanism import load_model
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    unconstrained_to_iv,
)


def build_subset(
    data_path: str,
    history_len: int,
    future_len: int,
    start_idx: int,
    max_windows: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return build_test_subset(data_path, history_len, future_len, start_idx, max_windows)


@torch.no_grad()
def embed_and_residualize(
    model,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_01 = denormalize_iv(history_norm).to(device)
    future_01 = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1).to(device)

    all_embed = []
    all_resid = []
    all_mean = []

    n = history_norm.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        hist_b = history_01[start:end]
        fut_b = future_01[start:end]
        (
            mu_u,
            time_factor,
            time_diag,
            cell_factor,
            cell_diag,
            scale,
            flow_context,
            base_local_delta,
            block_logits,
        ) = model.forward_from_history(hist_b)
        det_mean = unconstrained_to_iv(mu_u, lo=model.support_lo, hi=model.support_hi).reshape(end - start, fut_b.shape[1], fut_b.shape[2])
        path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
        residual = fut_b - det_mean
        all_embed.append(path_context.detach().cpu().numpy().reshape(end - start, -1))
        all_resid.append(residual.detach().cpu().numpy())
        all_mean.append(det_mean.detach().cpu().numpy())

    return (
        np.concatenate(all_embed, axis=0),
        np.concatenate(all_resid, axis=0),
        np.concatenate(all_mean, axis=0),
    )


def overlap_score(target: np.ndarray, proposal: np.ndarray) -> float:
    t = np.abs(np.asarray(target, dtype=np.float64).reshape(-1))
    p = np.abs(np.asarray(proposal, dtype=np.float64).reshape(-1))
    if t.sum() <= 1e-12 or p.sum() <= 1e-12:
        return float("nan")
    t /= t.sum()
    p /= p.sum()
    return float(np.minimum(t, p).sum())


def norm_entropy(counts: np.ndarray) -> float:
    p = np.asarray(counts, dtype=np.float64).reshape(-1)
    if p.sum() <= 1e-12:
        return float("nan")
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(counts)))


def evaluate_codebook(
    train_embed: np.ndarray,
    train_resid: np.ndarray,
    test_embed: np.ndarray,
    test_resid: np.ndarray,
    test_turb_mask: np.ndarray,
    k: int,
    random_state: int,
) -> dict[str, Any]:
    b_train, t, c = train_resid.shape
    b_test = test_resid.shape[0]
    flat_train = train_resid.reshape(b_train, -1)
    flat_test = test_resid.reshape(b_test, -1)

    feat_mean = flat_train.mean(axis=0, keepdims=True)
    feat_std = flat_train.std(axis=0, keepdims=True) + 1e-6
    train_z = (flat_train - feat_mean) / feat_std
    test_z = (flat_test - feat_mean) / feat_std

    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=random_state,
        batch_size=min(512, b_train),
        n_init=10,
        max_iter=300,
    )
    train_labels = km.fit_predict(train_z)
    centers_z = km.cluster_centers_
    centers = centers_z * feat_std + feat_mean
    centers = centers.reshape(k, t, c)

    # Oracle nearest motif on test residuals.
    d_test = np.square(test_z[:, None, :] - centers_z[None, :, :]).sum(axis=2)
    oracle_labels = d_test.argmin(axis=1)

    # Simple conditional predictor: nearest embedding centroid by cluster.
    embed_centroids = np.zeros((k, train_embed.shape[1]), dtype=np.float64)
    for i in range(k):
        mask = train_labels == i
        if mask.any():
            embed_centroids[i] = train_embed[mask].mean(axis=0)
        else:
            embed_centroids[i] = train_embed.mean(axis=0)
    d_embed = np.square(test_embed[:, None, :] - embed_centroids[None, :, :]).sum(axis=2)
    pred_labels = d_embed.argmin(axis=1)

    global_template = train_resid.mean(axis=0)
    oracle_proto = centers[oracle_labels]
    pred_proto = centers[pred_labels]

    global_mae = float(np.mean(np.abs(test_resid - global_template[None])))
    oracle_mae = float(np.mean(np.abs(test_resid - oracle_proto)))
    pred_mae = float(np.mean(np.abs(test_resid - pred_proto)))
    oracle_rel_gain = float((global_mae - oracle_mae) / max(global_mae, 1e-12))
    pred_rel_gain = float((global_mae - pred_mae) / max(global_mae, 1e-12))

    hidx = t - 1
    energy_h30 = np.abs(test_resid[:, hidx, :]).sum(axis=1)
    turb_energy = energy_h30[test_turb_mask]
    hard_thr = float(np.quantile(turb_energy, 0.8)) if turb_energy.size > 0 else float("inf")
    hard_mask = test_turb_mask & (energy_h30 >= hard_thr)

    global_overlap = []
    oracle_overlap = []
    pred_overlap = []
    global_top1 = []
    oracle_top1 = []
    pred_top1 = []
    for i in np.where(hard_mask)[0]:
        target = np.abs(test_resid[i, hidx, :])
        global_overlap.append(overlap_score(target, np.abs(global_template[hidx])))
        oracle_overlap.append(overlap_score(target, np.abs(oracle_proto[i, hidx])))
        pred_overlap.append(overlap_score(target, np.abs(pred_proto[i, hidx])))
        target_top1 = int(target.argmax())
        global_top1.append(int(np.abs(global_template[hidx]).argmax() == target_top1))
        oracle_top1.append(int(np.abs(oracle_proto[i, hidx]).argmax() == target_top1))
        pred_top1.append(int(np.abs(pred_proto[i, hidx]).argmax() == target_top1))

    train_counts = np.bincount(train_labels, minlength=k).astype(np.float64)
    pred_counts = np.bincount(pred_labels, minlength=k).astype(np.float64)

    return {
        "k": int(k),
        "global_residual_mae": global_mae,
        "oracle_residual_mae": oracle_mae,
        "pred_residual_mae": pred_mae,
        "oracle_relative_gain": oracle_rel_gain,
        "pred_relative_gain": pred_rel_gain,
        "oracle_label_predict_match": float((pred_labels == oracle_labels).mean()),
        "train_cluster_entropy_norm": norm_entropy(train_counts),
        "pred_cluster_entropy_norm": norm_entropy(pred_counts),
        "hard_windows": int(hard_mask.sum()),
        "hard_h30_overlap": {
            "global": float(np.nanmean(global_overlap)) if global_overlap else float("nan"),
            "oracle": float(np.nanmean(oracle_overlap)) if oracle_overlap else float("nan"),
            "pred": float(np.nanmean(pred_overlap)) if pred_overlap else float("nan"),
        },
        "hard_h30_top1_hit_rate": {
            "global": float(np.mean(global_top1)) if global_top1 else float("nan"),
            "oracle": float(np.mean(oracle_top1)) if oracle_top1 else float("nan"),
            "pred": float(np.mean(pred_top1)) if pred_top1 else float("nan"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="198a discrete motif feasibility pretest")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/backfill/state_metric_transport_pathwise_residual_law_mean_reverting_covariance_mixture_structured_joint_student_t_183c/best_model.pt",
    )
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--train_start", type=int, default=0)
    parser.add_argument("--train_windows", type=int, default=4040)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--test_windows", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/198_design/198a_discrete_motif_feasibility",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_hist, train_fut = build_subset(
        args.data_path, args.history_len, args.future_len, args.train_start, args.train_windows
    )
    test_hist, test_fut = build_subset(
        args.data_path, args.history_len, args.future_len, args.test_start, args.test_windows
    )

    model, _ = load_model(args.ckpt, str(device))
    train_embed, train_resid, _ = embed_and_residualize(
        model, train_hist, train_fut, device=device, batch_size=args.batch_size
    )
    test_embed, test_resid, _ = embed_and_residualize(
        model, test_hist, test_fut, device=device, batch_size=args.batch_size
    )

    vov, q20, q80 = regime_masks_from_history(test_hist)
    turb_mask = vov >= q80

    results = {
        "config": {
            "ckpt": args.ckpt,
            "train_windows": int(train_hist.shape[0]),
            "test_windows": int(test_hist.shape[0]),
            "q20_vov": float(q20),
            "q80_vov": float(q80),
            "turb_windows": int(turb_mask.sum()),
        },
        "codebooks": {},
    }

    for k in [8, 16, 32]:
        results["codebooks"][str(k)] = evaluate_codebook(
            train_embed=train_embed,
            train_resid=train_resid,
            test_embed=test_embed,
            test_resid=test_resid,
            test_turb_mask=turb_mask,
            k=k,
            random_state=42,
        )

    best_k = max(results["codebooks"], key=lambda kk: results["codebooks"][kk]["pred_relative_gain"])
    results["interpretation"] = {
        "best_k_by_pred_gain": int(best_k),
        "read": (
            "If a small codebook materially beats the single global residual template on "
            "held-out residual reconstruction and hard-window h30 overlap, discrete future "
            "motifs are a plausible next branch."
        ),
    }

    out_path = output_dir / "summary.json"
    out_path.write_text(json.dumps(make_serializable(results), indent=2))
    print(json.dumps(make_serializable(results), indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
