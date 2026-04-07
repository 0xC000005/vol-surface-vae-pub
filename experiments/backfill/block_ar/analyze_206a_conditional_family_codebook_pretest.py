#!/usr/bin/env python
"""
Pretest whether dangerous-window shape families can be represented by a reusable
global codebook plus conditional top-K routing.

This is the next step after 205a:
  - 205a showed a local small coherent family exists on dangerous severe steps
  - this script tests whether a *global* pattern dictionary can capture that family
    with history-conditioned routing

Question:
  Is a conditional top-K family over a reusable codebook plausible, or is the family
  still too local/idiosyncratic?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_201b_targeting_jump_mechanism import make_serializable
from experiments.backfill.block_ar.analyze_202a_discrete_innovation_mode_pretest import (
    build_split_indices,
    build_window_metadata,
    fit_multiclass_probe,
    load_model,
)
from experiments.backfill.block_ar.analyze_205a_conditional_shape_family_audit import (
    collect_teacher_forced_records,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows


def fit_codebook(
    x_train: np.ndarray,
    x_val: np.ndarray,
    k_values: list[int],
    seed: int,
) -> tuple[dict[int, Any], int, dict[str, Any]]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    pca = PCA(n_components=min(16, x_train.shape[1]), random_state=seed)
    z_train = pca.fit_transform(x_train_scaled)
    z_val = pca.transform(x_val_scaled)

    results: dict[int, Any] = {}
    best_k = k_values[0]
    best_score = float("inf")
    best_model = None
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        train_labels = km.fit_predict(z_train)
        centroids = km.cluster_centers_
        val_labels = np.argmin(((z_val[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1), axis=1)
        train_protos = np.stack(
            [x_train[train_labels == i].mean(axis=0) for i in range(k)],
            axis=0,
        )
        val_proto_mae = np.abs(train_protos[val_labels] - x_val).mean(axis=1).mean()
        cluster_sizes = np.bincount(train_labels, minlength=k)
        results[k] = {
            "val_proto_mae": float(val_proto_mae),
            "min_cluster_frac": float(cluster_sizes.min() / max(cluster_sizes.sum(), 1)),
            "max_cluster_frac": float(cluster_sizes.max() / max(cluster_sizes.sum(), 1)),
        }
        if val_proto_mae < best_score:
            best_score = float(val_proto_mae)
            best_k = k
            best_model = {
                "scaler": scaler,
                "pca": pca,
                "km": km,
            }
    assert best_model is not None
    return results, best_k, best_model


def transform_codebook(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    return model["pca"].transform(model["scaler"].transform(x))


def assign_codes(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    z = transform_codebook(x, model)
    centroids = model["km"].cluster_centers_
    return np.argmin(((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1), axis=1)


def build_prototypes(x: np.ndarray, labels: np.ndarray, k: int) -> np.ndarray:
    global_proto = x.mean(axis=0)
    protos = []
    for i in range(k):
        if np.any(labels == i):
            protos.append(x[labels == i].mean(axis=0))
        else:
            protos.append(global_proto)
    return np.stack(protos, axis=0)


def fit_routing_probe(
    cond_feat_train: np.ndarray,
    code_train: np.ndarray,
    cond_feat_test: np.ndarray,
    code_test: np.ndarray,
    n_classes: int,
    seed: int,
) -> tuple[dict[str, float], np.ndarray]:
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=min(32, cond_feat_train.shape[1]), random_state=seed)),
            (
                "logit",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    multi_class="multinomial",
                    random_state=seed,
                ),
            ),
        ]
    )
    pipe.fit(cond_feat_train, code_train)
    proba = pipe.predict_proba(cond_feat_test)
    pred = proba.argmax(axis=1)
    top3 = np.argsort(proba, axis=1)[:, -min(3, n_classes) :]
    majority = np.bincount(code_train, minlength=n_classes).argmax()
    metrics = {
        "test_top1_acc": float((pred == code_test).mean()),
        "test_top3_acc": float(np.mean([y in row for y, row in zip(code_test, top3)])),
        "chance_top1": 1.0 / float(n_classes),
        "majority_top1": float((code_test == majority).mean()),
        "n_train": int(cond_feat_train.shape[0]),
        "n_test": int(cond_feat_test.shape[0]),
    }
    return metrics, top3


def evaluate_family(
    test_records: dict[str, np.ndarray],
    test_idx: np.ndarray,
    prototypes: np.ndarray,
    predicted_top3_codes: np.ndarray,
    train_code_freq_order: np.ndarray,
) -> dict[str, float]:
    if test_idx.size == 0:
        return {}

    x = test_records["signed_delta_norm"][test_idx]
    top_cell = test_records["top_cell"][test_idx]

    global_template = prototypes.mean(axis=0)
    pred_top3 = predicted_top3_codes
    prior_top3 = np.tile(train_code_freq_order[:3][None, :], (test_idx.size, 1))

    pred_best_mae = []
    prior_best_mae = []
    global_template_mae = []
    pred_top1_hits = []
    pred_top3_hits = []
    prior_top1_hits = []
    prior_top3_hits = []

    proto_top_cells = np.abs(prototypes).argmax(axis=1)
    for i in range(test_idx.size):
        pred_codes = pred_top3[i]
        prior_codes = prior_top3[i]
        pred_best_mae.append(float(np.abs(prototypes[pred_codes] - x[i][None, :]).mean(axis=1).min()))
        prior_best_mae.append(float(np.abs(prototypes[prior_codes] - x[i][None, :]).mean(axis=1).min()))
        global_template_mae.append(float(np.abs(global_template - x[i]).mean()))

        pred_top1_hits.append(float(top_cell[i] == proto_top_cells[pred_codes[-1]]))
        pred_top3_hits.append(float(top_cell[i] in proto_top_cells[pred_codes]))
        prior_top1_hits.append(float(top_cell[i] == proto_top_cells[prior_codes[-1]]))
        prior_top3_hits.append(float(top_cell[i] in proto_top_cells[prior_codes]))

    return {
        "global_template_mae": float(np.mean(global_template_mae)),
        "prior_top3_proto_mae": float(np.mean(prior_best_mae)),
        "pred_top3_proto_mae": float(np.mean(pred_best_mae)),
        "pred_vs_prior_proto_improvement_pct": float(
            (np.mean(prior_best_mae) - np.mean(pred_best_mae)) / max(np.mean(prior_best_mae), 1e-8)
        ),
        "pred_vs_global_template_improvement_pct": float(
            (np.mean(global_template_mae) - np.mean(pred_best_mae)) / max(np.mean(global_template_mae), 1e-8)
        ),
        "prior_top1_acc": float(np.mean(prior_top1_hits)),
        "pred_top1_acc": float(np.mean(pred_top1_hits)),
        "prior_top3_acc": float(np.mean(prior_top3_hits)),
        "pred_top3_acc": float(np.mean(pred_top3_hits)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="206a conditional family codebook pretest")
    parser.add_argument("--checkpoint", type=str, default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--k_values", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/206_design/206a_conditional_family_codebook_pretest",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    surfaces = np.load(args.data_path)["surface"].astype(np.float32)
    train_abs_delta = np.abs(np.diff(surfaces[: args.test_start], axis=0).reshape(-1))
    q95 = float(np.quantile(train_abs_delta, 0.95))
    q99 = float(np.quantile(train_abs_delta, 0.99))
    model, payload = load_model(args.checkpoint, device)

    split_indices = build_split_indices(
        surface_len=surfaces.shape[0],
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
    )
    surf_tensor = torch.from_numpy(surfaces).to(device)

    split_records: dict[str, dict[str, np.ndarray]] = {}
    train_meta_for_thresholds = None
    for split_name in ("train", "val", "test"):
        hist, fut = build_multistep_windows(split_indices[split_name], surf_tensor, args.history_len, args.future_len)
        fut_flat_np = fut.view(fut.shape[0], fut.shape[1], -1).detach().cpu().numpy()
        hist_np = hist.detach().cpu().numpy()
        if split_name == "train":
            meta = build_window_metadata(hist_np, fut_flat_np)
            train_meta_for_thresholds = meta
        else:
            assert train_meta_for_thresholds is not None
            meta = build_window_metadata(
                hist_np,
                fut_flat_np,
                q80_vov_train=train_meta_for_thresholds["q80_vov"],
                q80_h30_turb_train=train_meta_for_thresholds["q80_h30_turb"],
            )
        split_records[split_name] = collect_teacher_forced_records(
            model=model,
            history_01=hist,
            future_flat=fut.view(fut.shape[0], fut.shape[1], -1),
            window_meta=meta,
            split_name=split_name,
            q95=q95,
            q99=q99,
            batch_size=args.batch_size,
            device=device,
        )

    # Build the global codebook on turbulent q99 severe steps.
    train_mask = (split_records["train"]["q99_any"] == 1) & (split_records["train"]["turb"] == 1)
    val_mask = (split_records["val"]["q99_any"] == 1) & (split_records["val"]["turb"] == 1)
    test_mask = (split_records["test"]["q99_any"] == 1) & (split_records["test"]["turb"] == 1)

    fit_results, best_k, codebook_model = fit_codebook(
        x_train=split_records["train"]["signed_delta_norm"][train_mask],
        x_val=split_records["val"]["signed_delta_norm"][val_mask],
        k_values=args.k_values,
        seed=args.seed,
    )

    train_codes = assign_codes(split_records["train"]["signed_delta_norm"][train_mask], codebook_model)
    val_codes = assign_codes(split_records["val"]["signed_delta_norm"][val_mask], codebook_model)
    test_codes = assign_codes(split_records["test"]["signed_delta_norm"][test_mask], codebook_model)
    prototypes = build_prototypes(split_records["train"]["signed_delta_norm"][train_mask], train_codes, best_k)
    train_code_freq_order = np.argsort(np.bincount(train_codes, minlength=best_k))[::-1]

    routing_metrics, top3_codes_test = fit_routing_probe(
        cond_feat_train=split_records["train"]["cond_feat"][train_mask],
        code_train=train_codes,
        cond_feat_test=split_records["test"]["cond_feat"][test_mask],
        code_test=test_codes,
        n_classes=best_k,
        seed=args.seed,
    )

    subset_metrics = {}
    full_test_mask = np.flatnonzero(test_mask)
    subset_defs = {
        "turb_q99_steps": full_test_mask,
        "hard_window_q99_steps": full_test_mask[split_records["test"]["hard_late_window"][test_mask] == 1],
        "hard_h30_steps": full_test_mask[split_records["test"]["hard_late_h30"][test_mask] == 1],
    }

    # Map from absolute test indices inside full record array to relative position in top3_codes_test.
    rel_pos = {abs_idx: i for i, abs_idx in enumerate(full_test_mask.tolist())}
    for name, subset_abs in subset_defs.items():
        subset_rel = np.array([rel_pos[int(i)] for i in subset_abs], dtype=np.int64)
        subset_metrics[name] = evaluate_family(
            test_records=split_records["test"],
            test_idx=subset_abs,
            prototypes=prototypes,
            predicted_top3_codes=top3_codes_test[subset_rel],
            train_code_freq_order=train_code_freq_order,
        )
        subset_metrics[name]["n_test"] = int(subset_abs.size)

    summary = {
        "checkpoint": args.checkpoint,
        "model_type": payload["config"]["type"],
        "thresholds": {"q95": q95, "q99": q99},
        "fit_results": fit_results,
        "best_k": int(best_k),
        "routing_metrics_turb_q99_test": routing_metrics,
        "subset_metrics": subset_metrics,
    }

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(make_serializable(summary), indent=2))
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
