#!/usr/bin/env python
"""
Stronger routing-feasibility pretest for 198a.

Question:
  If future residual motifs are representable by a small discrete codebook, can a
  small learned selector recover the right motif from history well enough to beat
  a single global residual template on held-out windows?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
def extract_features_and_residuals(
    model,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    history_01 = denormalize_iv(history_norm).to(device)
    future_01 = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1).to(device)

    all_path_context = []
    all_hist_flat = []
    all_resid = []

    n = history_norm.shape[0]
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        hist_b = history_01[start:end]
        fut_b = future_01[start:end]
        (
            mu_u,
            _time_factor,
            _time_diag,
            _cell_factor,
            _cell_diag,
            scale,
            flow_context,
            base_local_delta,
            block_logits,
        ) = model.forward_from_history(hist_b)
        det_mean = unconstrained_to_iv(mu_u, lo=model.support_lo, hi=model.support_hi).reshape(
            end - start, fut_b.shape[1], fut_b.shape[2]
        )
        path_context = model.build_path_context(flow_context, block_logits, base_local_delta, scale)
        residual = fut_b - det_mean
        all_path_context.append(path_context.detach().cpu().numpy())
        all_hist_flat.append(hist_b.detach().cpu().numpy().reshape(end - start, -1))
        all_resid.append(residual.detach().cpu().numpy())

    return (
        np.concatenate(all_path_context, axis=0),
        np.concatenate(all_hist_flat, axis=0),
        np.concatenate(all_resid, axis=0),
    )


def overlap_score(target: np.ndarray, proposal: np.ndarray) -> float:
    t = np.abs(np.asarray(target, dtype=np.float64).reshape(-1))
    p = np.abs(np.asarray(proposal, dtype=np.float64).reshape(-1))
    if t.sum() <= 1e-12 or p.sum() <= 1e-12:
        return float("nan")
    t /= t.sum()
    p /= p.sum()
    return float(np.minimum(t, p).sum())


def build_codebook(train_resid: np.ndarray, k: int, random_state: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    b_train, t, c = train_resid.shape
    flat_train = train_resid.reshape(b_train, -1)
    feat_mean = flat_train.mean(axis=0, keepdims=True)
    feat_std = flat_train.std(axis=0, keepdims=True) + 1e-6
    train_z = (flat_train - feat_mean) / feat_std
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
    return train_labels, centers, feat_mean, feat_std


def oracle_labels_from_codebook(test_resid: np.ndarray, centers: np.ndarray, feat_mean: np.ndarray, feat_std: np.ndarray) -> np.ndarray:
    flat_test = test_resid.reshape(test_resid.shape[0], -1)
    test_z = (flat_test - feat_mean) / feat_std
    centers_z = (centers.reshape(centers.shape[0], -1) - feat_mean) / feat_std
    d_test = np.square(test_z[:, None, :] - centers_z[None, :, :]).sum(axis=2)
    return d_test.argmin(axis=1)


def evaluate_prototypes(
    test_resid: np.ndarray,
    global_template: np.ndarray,
    centers: np.ndarray,
    pred_labels: np.ndarray,
    pred_probs: np.ndarray | None,
    oracle_labels: np.ndarray,
    test_turb_mask: np.ndarray,
) -> dict[str, Any]:
    oracle_proto = centers[oracle_labels]
    pred_proto = centers[pred_labels]
    global_mae = float(np.mean(np.abs(test_resid - global_template[None])))
    oracle_mae = float(np.mean(np.abs(test_resid - oracle_proto)))
    pred_mae = float(np.mean(np.abs(test_resid - pred_proto)))

    top3_proto = None
    top3_hit = float("nan")
    top3_mae = float("nan")
    if pred_probs is not None:
        topk = min(3, pred_probs.shape[1])
        top_idx = np.argsort(pred_probs, axis=1)[:, -topk:][:, ::-1]
        top_weights = np.take_along_axis(pred_probs, top_idx, axis=1)
        top_weights = top_weights / np.clip(top_weights.sum(axis=1, keepdims=True), 1e-12, None)
        top3_proto = (centers[top_idx] * top_weights[:, :, None, None]).sum(axis=1)
        top3_mae = float(np.mean(np.abs(test_resid - top3_proto)))
        top3_hit = float(np.mean((top_idx == oracle_labels[:, None]).any(axis=1)))

    hidx = test_resid.shape[1] - 1
    energy_h30 = np.abs(test_resid[:, hidx, :]).sum(axis=1)
    turb_energy = energy_h30[test_turb_mask]
    hard_thr = float(np.quantile(turb_energy, 0.8)) if turb_energy.size > 0 else float("inf")
    hard_mask = test_turb_mask & (energy_h30 >= hard_thr)
    hard_idx = np.where(hard_mask)[0]

    def _hard_stats(proto: np.ndarray | None) -> dict[str, float]:
        if proto is None or hard_idx.size == 0:
            return {"overlap": float("nan"), "top1": float("nan")}
        overlaps = []
        top1 = []
        for i in hard_idx:
            target = np.abs(test_resid[i, hidx, :])
            proposal = np.abs(proto[i, hidx])
            overlaps.append(overlap_score(target, proposal))
            top1.append(int(int(proposal.argmax()) == int(target.argmax())))
        return {
            "overlap": float(np.nanmean(overlaps)) if overlaps else float("nan"),
            "top1": float(np.mean(top1)) if top1 else float("nan"),
        }

    return {
        "global_residual_mae": global_mae,
        "oracle_residual_mae": oracle_mae,
        "pred_residual_mae": pred_mae,
        "pred_relative_gain": float((global_mae - pred_mae) / max(global_mae, 1e-12)),
        "oracle_relative_gain": float((global_mae - oracle_mae) / max(global_mae, 1e-12)),
        "top1_match_vs_oracle": float((pred_labels == oracle_labels).mean()),
        "top3_hit_vs_oracle": top3_hit,
        "top3_residual_mae": top3_mae,
        "top3_relative_gain": float((global_mae - top3_mae) / max(global_mae, 1e-12)) if np.isfinite(top3_mae) else float("nan"),
        "hard_windows": int(hard_idx.size),
        "hard_h30_global": _hard_stats(np.broadcast_to(global_template[None], test_resid.shape)),
        "hard_h30_oracle": _hard_stats(oracle_proto),
        "hard_h30_pred": _hard_stats(pred_proto),
        "hard_h30_top3": _hard_stats(top3_proto) if top3_proto is not None else {"overlap": float("nan"), "top1": float("nan")},
    }


def train_selector(name: str, x_train: np.ndarray, y_train: np.ndarray, random_state: int):
    if name == "path_context_logreg":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=2000,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    if name in {"path_context_mlp", "history_flat_mlp"}:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128) if name == "history_flat_mlp" else (192, 96),
                        activation="relu",
                        alpha=1e-4,
                        batch_size=128,
                        learning_rate_init=1e-3,
                        max_iter=300,
                        early_stopping=True,
                        validation_fraction=0.15,
                        n_iter_no_change=20,
                        random_state=random_state,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unknown selector {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description="198a routing-feasibility pretest")
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
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/198_design/198a_routing_feasibility",
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
    train_ctx, train_hist_flat, train_resid = extract_features_and_residuals(
        model, train_hist, train_fut, device=device, batch_size=args.batch_size
    )
    test_ctx, test_hist_flat, test_resid = extract_features_and_residuals(
        model, test_hist, test_fut, device=device, batch_size=args.batch_size
    )

    vov, q20, q80 = regime_masks_from_history(test_hist)
    test_turb_mask = (vov >= q80)

    results: dict[str, Any] = {
        "config": {
            "ckpt": args.ckpt,
            "train_windows": int(train_hist.shape[0]),
            "test_windows": int(test_hist.shape[0]),
            "random_state": args.random_state,
            "selectors": ["path_context_logreg", "path_context_mlp", "history_flat_mlp"],
        },
        "codebooks": {},
    }

    for k in (8, 16):
        train_labels, centers, feat_mean, feat_std = build_codebook(train_resid, k=k, random_state=args.random_state)
        oracle_test = oracle_labels_from_codebook(test_resid, centers, feat_mean, feat_std)
        global_template = train_resid.mean(axis=0)
        codebook_results: dict[str, Any] = {
            "oracle_reference": evaluate_prototypes(
                test_resid=test_resid,
                global_template=global_template,
                centers=centers,
                pred_labels=oracle_test,
                pred_probs=None,
                oracle_labels=oracle_test,
                test_turb_mask=test_turb_mask,
            )
        }

        selector_inputs = {
            "path_context_logreg": (train_ctx, test_ctx),
            "path_context_mlp": (train_ctx, test_ctx),
            "history_flat_mlp": (train_hist_flat, test_hist_flat),
        }

        for name, (x_train, x_test) in selector_inputs.items():
            clf = train_selector(name, x_train, train_labels, random_state=args.random_state)
            clf.fit(x_train, train_labels)
            pred_labels = clf.predict(x_test)
            pred_probs = clf.predict_proba(x_test) if hasattr(clf, "predict_proba") else None
            metrics = evaluate_prototypes(
                test_resid=test_resid,
                global_template=global_template,
                centers=centers,
                pred_labels=pred_labels,
                pred_probs=pred_probs,
                oracle_labels=oracle_test,
                test_turb_mask=test_turb_mask,
            )
            codebook_results[name] = metrics

        results["codebooks"][str(k)] = codebook_results

    results["interpretation"] = {
        "read": (
            "If a small learned selector from history can beat the single global residual "
            "template and recover a useful fraction of the oracle motif benefit on hard "
            "windows, 198a earns full implementation. If not, the discrete-motif branch "
            "should be dropped before building the full model."
        )
    }

    out_path = output_dir / "summary.json"
    out_path.write_text(json.dumps(make_serializable(results), indent=2))
    print(json.dumps(make_serializable(results), indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
