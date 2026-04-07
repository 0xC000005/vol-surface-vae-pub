#!/usr/bin/env python
"""
Pretest whether severe 201b teacher-forced residuals contain a useful localization mode
after factoring out severity.

This is narrower than 202a:
  - keep only severe steps
  - normalize out magnitude
  - cluster spatial/sign pattern only
  - test whether oracle pattern labels help targeting and are history-predictable
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

from experiments.backfill.block_ar.analyze_202a_discrete_innovation_mode_pretest import (
    build_split_indices,
    build_window_metadata,
    load_model,
    fit_multiclass_probe,
    summarize_persistence,
)
from experiments.backfill.block_ar.analyze_201b_targeting_jump_mechanism import (
    make_serializable,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows


def collect_localization_records(
    model,
    history_01: torch.Tensor,
    future_flat: torch.Tensor,
    window_meta: dict[str, np.ndarray],
    split_name: str,
    q99: float,
    batch_size: int,
    device: torch.device,
) -> dict[str, np.ndarray]:
    out: dict[str, list[np.ndarray]] = {
        "split": [],
        "window_idx": [],
        "step": [],
        "cond_feat": [],
        "delta": [],
        "abs_delta": [],
        "abs_delta_norm": [],
        "signed_delta_norm": [],
        "q99_any": [],
        "top_cell": [],
        "effective_support": [],
        "calm": [],
        "turb": [],
        "hard_late_window": [],
        "hard_late_h30": [],
    }

    n_windows = history_01.shape[0]
    future_len = future_flat.shape[1]
    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        hist = history_01[start:end].to(device)
        fut = future_flat[start:end].to(device)
        context = hist.clone()
        batch_window_idx = np.arange(start, end)

        for step in range(future_len):
            with torch.no_grad():
                cond = model.encode(context)
                if isinstance(cond, tuple):
                    cond = cond[0]

            prev = context[:, -1].reshape(end - start, -1)
            target = fut[:, step]
            delta = target - prev
            abs_delta = delta.abs()
            abs_sum = abs_delta.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            abs_norm = abs_delta / abs_sum
            signed_norm = delta / abs_sum
            entropy = -(abs_norm * torch.log(abs_norm.clamp_min(1e-8))).sum(dim=-1)
            eff_support = torch.exp(entropy)
            q99_any = (abs_delta >= q99).any(dim=-1)

            out["split"].append(np.full(end - start, split_name))
            out["window_idx"].append(batch_window_idx.copy())
            out["step"].append(np.full(end - start, step, dtype=np.int64))
            out["cond_feat"].append(cond.detach().cpu().numpy())
            out["delta"].append(delta.detach().cpu().numpy())
            out["abs_delta"].append(abs_delta.detach().cpu().numpy())
            out["abs_delta_norm"].append(abs_norm.detach().cpu().numpy())
            out["signed_delta_norm"].append(signed_norm.detach().cpu().numpy())
            out["q99_any"].append(q99_any.detach().cpu().numpy().astype(np.int64))
            out["top_cell"].append(abs_delta.argmax(dim=-1).detach().cpu().numpy())
            out["effective_support"].append(eff_support.detach().cpu().numpy())
            out["calm"].append(window_meta["calm"][batch_window_idx].astype(np.int64))
            out["turb"].append(window_meta["turb"][batch_window_idx].astype(np.int64))
            out["hard_late_window"].append(window_meta["hard_late"][batch_window_idx].astype(np.int64))
            out["hard_late_h30"].append((window_meta["hard_late"][batch_window_idx] & (step == future_len - 1)).astype(np.int64))

            next_frame = target.view(end - start, 1, 5, 5)
            context = torch.cat([context[:, 1:], next_frame], dim=1)

    return {k: np.concatenate(v, axis=0) for k, v in out.items()}


def fit_pattern_model(
    x_train: np.ndarray,
    x_val: np.ndarray,
    x_test: np.ndarray,
    max_modes: int,
    pca_dim: int,
    seed: int,
) -> tuple[dict[int, Any], int, Any]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    pca = PCA(n_components=min(pca_dim, x_train.shape[1]), random_state=seed)
    z_train = pca.fit_transform(x_train_scaled)
    z_val = pca.transform(scaler.transform(x_val))
    z_test = pca.transform(scaler.transform(x_test))

    results: dict[int, Any] = {}
    best_k = 2
    best_score = -np.inf
    best_model = None
    for k in range(2, max_modes + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        train_labels = km.fit_predict(z_train)
        centroids = km.cluster_centers_

        def neg_dist(z: np.ndarray) -> np.ndarray:
            d = ((z[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=-1)
            return -d.min(axis=1)

        # Larger separation between assigned centroid and others is better.
        train_min = neg_dist(z_train).mean()
        val_min = neg_dist(z_val).mean()
        test_min = neg_dist(z_test).mean()
        cluster_sizes = np.bincount(train_labels, minlength=k)
        results[k] = {
            "train_negdist_mean": float(train_min),
            "val_negdist_mean": float(val_min),
            "test_negdist_mean": float(test_min),
            "min_cluster_frac": float(cluster_sizes.min() / max(cluster_sizes.sum(), 1)),
            "max_cluster_frac": float(cluster_sizes.max() / max(cluster_sizes.sum(), 1)),
        }
        if val_min > best_score:
            best_score = float(val_min)
            best_k = k
            best_model = {
                "scaler": scaler,
                "pca": pca,
                "km": km,
            }

    assert best_model is not None
    return results, best_k, best_model


def transform_pattern(x: np.ndarray, model: dict[str, Any]) -> np.ndarray:
    scaler = model["scaler"]
    pca = model["pca"]
    return pca.transform(scaler.transform(x))


def summarize_pattern_modes(
    records: dict[str, np.ndarray],
    labels: np.ndarray,
    n_modes: int,
) -> dict[str, Any]:
    global_top_counts = np.bincount(records["top_cell"], minlength=25)
    global_top1 = float(global_top_counts.max() / max(global_top_counts.sum(), 1))
    mode_summary: dict[str, Any] = {}
    for k in range(n_modes):
        mask = labels == k
        if not np.any(mask):
            mode_summary[str(k)] = {"count": 0}
            continue
        top_counts = np.bincount(records["top_cell"][mask], minlength=25)
        mode_summary[str(k)] = {
            "count": int(mask.sum()),
            "weight": float(mask.mean()),
            "effective_support_mean": float(records["effective_support"][mask].mean()),
            "hard_late_h30_rate": float(records["hard_late_h30"][mask].mean()),
            "calm_rate": float(records["calm"][mask].mean()),
            "turb_rate": float(records["turb"][mask].mean()),
            "top_cell_majority": int(top_counts.argmax()),
            "top_cell_majority_share": float(top_counts.max() / max(top_counts.sum(), 1)),
            "top3_cells": np.argsort(top_counts)[-3:][::-1].tolist(),
            "mean_abs_pattern_top3": np.argsort(records["abs_delta_norm"][mask].mean(axis=0))[-3:][::-1].tolist(),
            "mean_signed_pattern_top3_pos": np.argsort(records["signed_delta_norm"][mask].mean(axis=0))[-3:][::-1].tolist(),
            "mean_signed_pattern_top3_neg": np.argsort(records["signed_delta_norm"][mask].mean(axis=0))[:3].tolist(),
        }
    return {
        "global_top_cell_majority_share": global_top1,
        "modes": mode_summary,
    }


def oracle_pattern_usefulness(
    train_records: dict[str, np.ndarray],
    test_records: dict[str, np.ndarray],
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    n_modes: int,
) -> dict[str, Any]:
    global_abs_template = train_records["abs_delta_norm"].mean(axis=0)
    mode_abs_templates = np.stack(
        [
            train_records["abs_delta_norm"][train_labels == k].mean(axis=0)
            if np.any(train_labels == k)
            else global_abs_template
            for k in range(n_modes)
        ],
        axis=0,
    )

    global_signed_template = train_records["signed_delta_norm"].mean(axis=0)
    mode_signed_templates = np.stack(
        [
            train_records["signed_delta_norm"][train_labels == k].mean(axis=0)
            if np.any(train_labels == k)
            else global_signed_template
            for k in range(n_modes)
        ],
        axis=0,
    )

    global_top_probs = np.bincount(train_records["top_cell"], minlength=25).astype(np.float64)
    global_top_probs /= max(global_top_probs.sum(), 1.0)
    mode_top_probs = []
    for k in range(n_modes):
        mask = train_labels == k
        counts = np.bincount(train_records["top_cell"][mask], minlength=25).astype(np.float64) if np.any(mask) else global_top_probs.copy()
        counts /= max(counts.sum(), 1.0)
        mode_top_probs.append(counts)
    mode_top_probs = np.stack(mode_top_probs, axis=0)

    pred_abs_global = np.repeat(global_abs_template[None, :], test_records["abs_delta_norm"].shape[0], axis=0)
    pred_abs_mode = mode_abs_templates[test_labels]
    pred_signed_global = np.repeat(global_signed_template[None, :], test_records["signed_delta_norm"].shape[0], axis=0)
    pred_signed_mode = mode_signed_templates[test_labels]

    abs_global_mae = np.abs(pred_abs_global - test_records["abs_delta_norm"]).mean(axis=1)
    abs_mode_mae = np.abs(pred_abs_mode - test_records["abs_delta_norm"]).mean(axis=1)
    signed_global_mae = np.abs(pred_signed_global - test_records["signed_delta_norm"]).mean(axis=1)
    signed_mode_mae = np.abs(pred_signed_mode - test_records["signed_delta_norm"]).mean(axis=1)

    top_rank_global = np.argsort(global_top_probs)[::-1]
    top_rank_mode = np.argsort(mode_top_probs, axis=1)[:, ::-1]
    top_cell = test_records["top_cell"]

    def subset(mask: np.ndarray) -> dict[str, float]:
        if not np.any(mask):
            return {}
        return {
            "abs_template_mae_global": float(abs_global_mae[mask].mean()),
            "abs_template_mae_oracle": float(abs_mode_mae[mask].mean()),
            "abs_template_improvement_pct": float((abs_global_mae[mask].mean() - abs_mode_mae[mask].mean()) / max(abs_global_mae[mask].mean(), 1e-8)),
            "signed_template_mae_global": float(signed_global_mae[mask].mean()),
            "signed_template_mae_oracle": float(signed_mode_mae[mask].mean()),
            "signed_template_improvement_pct": float((signed_global_mae[mask].mean() - signed_mode_mae[mask].mean()) / max(signed_global_mae[mask].mean(), 1e-8)),
            "global_top1_acc": float((top_cell[mask] == top_rank_global[0]).mean()),
            "oracle_top1_acc": float(np.mean(top_cell[mask] == top_rank_mode[test_labels[mask], 0])),
            "global_top3_acc": float(np.mean([y in top_rank_global[:3] for y in top_cell[mask]])),
            "oracle_top3_acc": float(np.mean([y in row[:3] for y, row in zip(top_cell[mask], top_rank_mode[test_labels[mask]])])),
        }

    return {
        "all_severe": subset(np.ones(test_records["top_cell"].shape[0], dtype=bool)),
        "hard_late_h30_severe": subset(test_records["hard_late_h30"] == 1),
        "turb_severe": subset(test_records["turb"] == 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="202b severity-normalized localization pretest")
    parser.add_argument("--checkpoint", type=str, default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_modes", type=int, default=8)
    parser.add_argument("--pca_dim", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/202_design/202b_severity_normalized_localization_pretest",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    surfaces = np.load(args.data_path)["surface"].astype(np.float32)
    q99 = float(np.quantile(np.abs(np.diff(surfaces[: args.test_start], axis=0).reshape(-1)), 0.99))
    model, payload = load_model(args.checkpoint, device)
    split_indices = build_split_indices(
        surface_len=surfaces.shape[0],
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
    )
    surf_tensor = torch.from_numpy(surfaces).to(device)

    split_records = {}
    split_meta = {}
    train_thresholds = None
    for split_name in ("train", "val", "test"):
        idx = split_indices[split_name]
        hist_01, future_flat = build_multistep_windows(idx, surf_tensor, args.history_len, args.future_len)
        hist_np = hist_01.detach().cpu().numpy()
        future_np = future_flat.detach().cpu().numpy()
        if split_name == "train":
            meta = build_window_metadata(hist_np, future_np)
            train_thresholds = {
                "q80_vov": meta["q80_vov"],
                "q80_h30_turb": meta["q80_h30_turb"],
            }
        else:
            meta = build_window_metadata(
                hist_np,
                future_np,
                q80_vov_train=train_thresholds["q80_vov"],
                q80_h30_turb_train=train_thresholds["q80_h30_turb"],
            )
        split_meta[split_name] = meta
        rec = collect_localization_records(
            model=model,
            history_01=hist_01,
            future_flat=future_flat,
            window_meta=meta,
            split_name=split_name,
            q99=q99,
            batch_size=args.batch_size,
            device=device,
        )
        severe_mask = rec["q99_any"] == 1
        split_records[split_name] = {k: v[severe_mask] for k, v in rec.items()}

    pattern_results, best_k, pattern_model = fit_pattern_model(
        x_train=split_records["train"]["signed_delta_norm"],
        x_val=split_records["val"]["signed_delta_norm"],
        x_test=split_records["test"]["signed_delta_norm"],
        max_modes=args.max_modes,
        pca_dim=args.pca_dim,
        seed=args.seed,
    )

    labels = {}
    for split_name, rec in split_records.items():
        z = transform_pattern(rec["signed_delta_norm"], pattern_model)
        labels[split_name] = pattern_model["km"].predict(z)

    mode_probe = fit_multiclass_probe(
        x_train=split_records["train"]["cond_feat"],
        y_train=labels["train"],
        x_test=split_records["test"]["cond_feat"],
        y_test=labels["test"],
        max_classes=best_k,
        seed=args.seed,
    )

    persistence = {
        split_name: {
            "all_severe_steps": summarize_persistence(
                split_records[split_name],
                labels[split_name],
                split_indices[split_name].shape[0],
                args.future_len,
                subset_mask=None,
            ),
            "turb_windows": summarize_persistence(
                split_records[split_name],
                labels[split_name],
                split_indices[split_name].shape[0],
                args.future_len,
                subset_mask=split_meta[split_name]["turb"],
            ),
            "hard_late_windows": summarize_persistence(
                split_records[split_name],
                labels[split_name],
                split_indices[split_name].shape[0],
                args.future_len,
                subset_mask=split_meta[split_name]["hard_late"],
            ),
        }
        for split_name in ("train", "val", "test")
    }

    semantics = {
        split_name: summarize_pattern_modes(split_records[split_name], labels[split_name], best_k)
        for split_name in ("train", "val", "test")
    }
    oracle_use = oracle_pattern_usefulness(
        train_records=split_records["train"],
        test_records=split_records["test"],
        train_labels=labels["train"],
        test_labels=labels["test"],
        n_modes=best_k,
    )

    summary = {
        "config": {
            "checkpoint": args.checkpoint,
            "history_len": args.history_len,
            "future_len": args.future_len,
            "q99": q99,
            "best_checkpoint_epoch": int(payload.get("epoch", -1)),
            "best_k": best_k,
        },
        "counts": {
            split_name: {
                "severe_steps": int(split_records[split_name]["step"].shape[0]),
                "hard_late_h30_severe_steps": int(split_records[split_name]["hard_late_h30"].sum()),
                "turb_severe_steps": int(split_records[split_name]["turb"].sum()),
            }
            for split_name in ("train", "val", "test")
        },
        "pattern_model_selection": {
            "results": pattern_results,
            "best_k": best_k,
        },
        "history_to_pattern_predictability": mode_probe,
        "pattern_semantics": semantics,
        "pattern_persistence": persistence,
        "oracle_pattern_usefulness": oracle_use,
        "interpretation": {
            "read": (
                "A useful discrete event-pattern state is supported if severity-normalized severe-step clusters "
                "show distinct localized top-cell structure, oracle pattern labels beat the global severe template, "
                "history predicts the pattern above chance, and pattern persistence is positive on hard windows."
            )
        },
    }

    out_path = out_dir / "summary.json"
    out_path.write_text(json.dumps(make_serializable(summary), indent=2))
    print(json.dumps(make_serializable(summary), indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
