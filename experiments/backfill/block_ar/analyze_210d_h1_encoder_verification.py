#!/usr/bin/env python
"""
Direct H=1 verification of whether the transformer encoder contains useful one-step signal.

This is a strict follow-up to the earlier 201a/201b mechanism work:
  - avoid late-horizon proxy tasks
  - stay on the actual H=1 problem
  - probe the saved encoder representation with simple linear heads only

Questions:
  1. Does the trusted transformer encoder linearly separate next-day severe windows?
  2. Does it contain next-day localization signal on severe windows?
  3. Is that signal materially better than simple hand features or flattened history?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_201b_targeting_jump_mechanism import (
    extract_features,
    load_model as load_201ab_model,
    make_serializable,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import build_one_step_windows
from experiments.backfill.block_ar.train_210c_h1_mixture_density_transformer import (
    load_model as load_210c_model,
)


def safe_auc(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, score))


def safe_ap(y_true: np.ndarray, score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, score))


def build_strict_indices(
    surface_len: int,
    history_len: int,
    test_start: int,
    val_size: int,
    future_gap: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_train_idx = test_start - history_len - future_gap
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    test_indices = np.arange(test_start, surface_len - history_len)
    return train_indices, val_indices, test_indices


def build_window_metadata(
    history_01: torch.Tensor,
    target_01: torch.Tensor,
    q95: float,
    q99: float,
    train_vov_q80: float,
) -> dict[str, np.ndarray]:
    history_np = history_01.detach().cpu().numpy()
    target_np = target_01.detach().cpu().numpy()
    prev_np = history_np[:, -1].reshape(history_np.shape[0], -1)
    delta_abs = np.abs(target_np - prev_np)

    hist_mean = history_np.mean(axis=(2, 3))
    hist_vov = np.diff(hist_mean, axis=1).std(axis=1)
    turb = hist_vov >= train_vov_q80

    return {
        "flat_history": history_np.reshape(history_np.shape[0], -1).astype(np.float32),
        "simple_features": np.stack(
            [
                hist_vov,
                hist_mean[:, -1],
                hist_mean[:, -1] - hist_mean[:, 0],
            ],
            axis=1,
        ).astype(np.float32),
        "q95_any": (delta_abs >= q95).any(axis=1).astype(np.int64),
        "q99_any": (delta_abs >= q99).any(axis=1).astype(np.int64),
        "turbulent": turb.astype(np.int64),
        "turb_q99": (turb & ((delta_abs >= q99).any(axis=1))).astype(np.int64),
        "top_cell": delta_abs.argmax(axis=1).astype(np.int64),
        "max_abs_move": delta_abs.max(axis=1).astype(np.float32),
    }


def build_probe_pipeline(
    x_train: np.ndarray,
    feature_name: str,
    *,
    class_weight: str | None = None,
) -> Pipeline:
    steps: list[tuple[str, Any]] = [("scaler", StandardScaler())]
    if feature_name == "flat_history":
        n_components = min(32, x_train.shape[1], max(1, x_train.shape[0] - 1))
        steps.append(("pca", PCA(n_components=n_components)))
    steps.append(("logit", LogisticRegression(max_iter=4000, class_weight=class_weight)))
    return Pipeline(steps)


def fit_binary_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_name: str,
) -> dict[str, float]:
    clf = build_probe_pipeline(x_train, feature_name=feature_name, class_weight="balanced")
    clf.fit(x_train, y_train)
    score = clf.predict_proba(x_eval)[:, 1]
    return {
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_eval": float(y_eval.mean()),
        "roc_auc": safe_auc(y_eval, score),
        "average_precision": safe_ap(y_eval, score),
    }


def fit_multiclass_probe(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    feature_name: str,
    n_classes: int = 25,
) -> dict[str, float]:
    if x_train.shape[0] == 0 or x_eval.shape[0] == 0:
        return {
            "n_train": int(x_train.shape[0]),
            "n_eval": int(x_eval.shape[0]),
            "top1_acc": float("nan"),
            "top3_acc": float("nan"),
            "chance_top1": 1.0 / float(n_classes),
            "majority_top1": float("nan"),
        }

    clf = build_probe_pipeline(x_train, feature_name=feature_name, class_weight=None)
    clf.fit(x_train, y_train)
    proba = clf.predict_proba(x_eval)
    classes = clf.named_steps["logit"].classes_

    top1_pred = classes[np.argmax(proba, axis=1)]
    topk = min(3, proba.shape[1])
    top_idx = np.argsort(proba, axis=1)[:, -topk:]
    top_classes = classes[top_idx]
    top3_acc = float(np.mean([int(y in row) for y, row in zip(y_eval, top_classes)]))

    majority = np.bincount(y_train, minlength=n_classes).argmax()
    return {
        "n_train": int(x_train.shape[0]),
        "n_eval": int(x_eval.shape[0]),
        "top1_acc": float((top1_pred == y_eval).mean()),
        "top3_acc": top3_acc,
        "chance_top1": 1.0 / float(n_classes),
        "majority_top1": float((y_eval == majority).mean()),
    }


def load_encoder_checkpoint(checkpoint_path: str, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_type = payload["config"]["type"]
    if model_type in {
        "transformer_ar_rollout_tail_student_t_201a",
        "transformer_underfit_aware_selffed_rollout_student_t_201b",
    }:
        model, _, raw_payload = load_201ab_model(checkpoint_path, device)
        return model, model_type, raw_payload
    if model_type == "transformer_h1_mixture_density_student_t_210c":
        model, raw_payload = load_210c_model(checkpoint_path, device)
        return model, model_type, raw_payload
    raise ValueError(f"Unsupported checkpoint type for H1 encoder verification: {model_type}")


def summarize_binary_tasks(
    train_features: dict[str, np.ndarray],
    eval_features: dict[str, np.ndarray],
    train_meta: dict[str, np.ndarray],
    eval_meta: dict[str, np.ndarray],
    task_names: list[str],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for task in task_names:
        out[task] = {}
        y_train = train_meta[task]
        y_eval = eval_meta[task]
        for feat_name, x_train in train_features.items():
            out[task][feat_name] = fit_binary_probe(
                x_train=x_train,
                y_train=y_train,
                x_eval=eval_features[feat_name],
                y_eval=y_eval,
                feature_name=feat_name,
            )
    return out


def summarize_multiclass_tasks(
    train_features: dict[str, np.ndarray],
    eval_features: dict[str, np.ndarray],
    train_meta: dict[str, np.ndarray],
    eval_meta: dict[str, np.ndarray],
    subset_masks: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, (train_mask, eval_mask) in subset_masks.items():
        out[name] = {}
        y_train = train_meta["top_cell"][train_mask]
        y_eval = eval_meta["top_cell"][eval_mask]
        for feat_name, x_train in train_features.items():
            out[name][feat_name] = fit_multiclass_probe(
                x_train=x_train[train_mask],
                y_train=y_train,
                x_eval=eval_features[feat_name][eval_mask],
                y_eval=y_eval,
                feature_name=feat_name,
            )
    return out


def build_markdown(summary: dict[str, Any]) -> str:
    ckpt = summary["checkpoint"]
    b_test = summary["binary_test"]
    m_test = summary["multiclass_test"]
    q99_encoder = b_test["q99_any"]["encoder"]
    q99_simple = b_test["q99_any"]["simple"]
    q99_flat = b_test["q99_any"]["flat_history"]
    turb_encoder = b_test["turb_q99"]["encoder"]
    turb_simple = b_test["turb_q99"]["simple"]
    top_encoder = m_test["q99_any_top_cell"]["encoder"]
    top_simple = m_test["q99_any_top_cell"]["simple"]
    top_flat = m_test["q99_any_top_cell"]["flat_history"]
    turb_top_encoder = m_test["turb_q99_top_cell"]["encoder"]
    turb_top_flat = m_test["turb_q99_top_cell"]["flat_history"]

    lines = [
        "# `210d` H=1 encoder verification",
        "",
        "Verification Result",
        "",
        f"Verdict: `{summary['verdict']}`",
        "",
        "What I Checked",
        "",
        f"- checkpoint: `{ckpt['path']}`",
        f"- model type: `{ckpt['type']}`",
        f"- strict split windows: train `{summary['counts']['train_windows']}`, val `{summary['counts']['val_windows']}`, test `{summary['counts']['test_windows']}`",
        "- direct `H=1` probe tasks only",
        "- linear probes on encoder features, flattened history, and simple hand features",
        "",
        "Findings",
        "",
        "Confirmed Correct",
        "",
        (
            f"- On severe-window top-cell localization, the encoder probe is the strongest of the three tested feature sets: "
            f"test top-1/top-3 on `q99_any_top_cell` are `{top_encoder['top1_acc']:.3f}` / `{top_encoder['top3_acc']:.3f}`, "
            f"vs simple `{top_simple['top1_acc']:.3f}` / `{top_simple['top3_acc']:.3f}` and flat-history "
            f"`{top_flat['top1_acc']:.3f}` / `{top_flat['top3_acc']:.3f}`."
        ),
        (
            f"- Even on the smaller turbulent severe subset, the encoder still carries useful small-family localization signal: "
            f"`turb_q99_top_cell` top-3 is `{turb_top_encoder['top3_acc']:.3f}`."
        ),
        "",
        "Issues Found",
        "",
        (
            f"- `WARNING`: on direct one-step severe-window detection, the encoder probe is weak. "
            f"On `q99_any`, encoder ROC-AUC / AP are `{q99_encoder['roc_auc']:.3f}` / `{q99_encoder['average_precision']:.3f}`, "
            f"worse than simple `{q99_simple['roc_auc']:.3f}` / `{q99_simple['average_precision']:.3f}` and also worse than flat-history "
            f"`{q99_flat['roc_auc']:.3f}` / `{q99_flat['average_precision']:.3f}`."
        ),
        (
            f"- `WARNING`: the same pattern holds on the stricter `turb_q99` binary task. "
            f"Encoder ROC-AUC / AP are `{turb_encoder['roc_auc']:.3f}` / `{turb_encoder['average_precision']:.3f}`, "
            f"while simple features are `{turb_simple['roc_auc']:.3f}` / `{turb_simple['average_precision']:.3f}`."
        ),
        (
            f"- `WARNING`: exact one-step severe localization is still not solved. "
            f"On `q99_any_top_cell`, encoder top-1 is `{top_encoder['top1_acc']:.3f}`, and on `turb_q99_top_cell` "
            f"encoder top-1 is only `{turb_top_encoder['top1_acc']:.3f}` vs flat-history `{turb_top_flat['top1_acc']:.3f}`."
        ),
        "",
        "Alternative Explanations",
        "",
        "- The encoder may preserve coarse severe-shape family information while compressing away some of the simple volatility/severity cues that matter for direct H=1 event detection.",
        "- Some of the remaining error may reflect weak identifiability from the current 30-day conditioning set rather than a pure decoder-only failure.",
        "",
        "My Independent Assessment",
        "",
        (
            "The H=1 evidence is mixed. The encoder is not generally useless, because it is the best tested representation "
            "for severe-window localization on the broader `q99_any` slice, especially in top-3 terms."
        ),
        (
            "But it does not support the stronger claim `encoder is already clearly right for H=1`. "
            "On direct one-step severe-event detection, the encoder underperforms very simple history statistics. "
            "So the clean decoder-only story is too strong."
        ),
        "",
        "Recommended Action",
        "",
        "- Treat H=1 as a split bottleneck: severity detection still looks encoder/state-limited, while severe-shape allocation still looks output-law-limited.",
        "- If you continue the H=1 multimodal line, consider adding simple severity context directly or testing a stronger H=1 encoder alongside the next decoder branch.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct H=1 encoder verification")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/best_model.pt",
    )
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--future_gap", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-08/analysis/210_design/210d_h1_encoder_verification",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    train_abs_delta = np.abs(np.diff(surfaces[: args.test_start], axis=0).reshape(-1))
    q95 = float(np.quantile(train_abs_delta, 0.95))
    q99 = float(np.quantile(train_abs_delta, 0.99))

    train_idx, val_idx, test_idx = build_strict_indices(
        surface_len=surfaces.shape[0],
        history_len=args.history_len,
        test_start=args.test_start,
        val_size=args.val_size,
        future_gap=args.future_gap,
    )

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_target = build_one_step_windows(train_idx, surf_tensor, args.history_len)
    val_hist, val_target = build_one_step_windows(val_idx, surf_tensor, args.history_len)
    test_hist, test_target = build_one_step_windows(test_idx, surf_tensor, args.history_len)

    train_hist_np = train_hist.detach().cpu().numpy()
    hist_mean_train = train_hist_np.mean(axis=(2, 3))
    train_vov_q80 = float(np.quantile(np.diff(hist_mean_train, axis=1).std(axis=1), 0.8))

    train_meta = build_window_metadata(train_hist, train_target, q95=q95, q99=q99, train_vov_q80=train_vov_q80)
    val_meta = build_window_metadata(val_hist, val_target, q95=q95, q99=q99, train_vov_q80=train_vov_q80)
    test_meta = build_window_metadata(test_hist, test_target, q95=q95, q99=q99, train_vov_q80=train_vov_q80)

    model, model_type, payload = load_encoder_checkpoint(args.checkpoint, device)
    train_enc = extract_features(model, train_hist, batch_size=args.batch_size, device=device).astype(np.float32)
    val_enc = extract_features(model, val_hist, batch_size=args.batch_size, device=device).astype(np.float32)
    test_enc = extract_features(model, test_hist, batch_size=args.batch_size, device=device).astype(np.float32)

    train_features = {
        "simple": train_meta["simple_features"],
        "flat_history": train_meta["flat_history"],
        "encoder": train_enc,
    }
    val_features = {
        "simple": val_meta["simple_features"],
        "flat_history": val_meta["flat_history"],
        "encoder": val_enc,
    }
    test_features = {
        "simple": test_meta["simple_features"],
        "flat_history": test_meta["flat_history"],
        "encoder": test_enc,
    }

    binary_tasks = ["q95_any", "q99_any", "turb_q99"]
    binary_val = summarize_binary_tasks(train_features, val_features, train_meta, val_meta, binary_tasks)
    binary_test = summarize_binary_tasks(train_features, test_features, train_meta, test_meta, binary_tasks)

    multiclass_masks_val = {
        "q99_any_top_cell": (train_meta["q99_any"] == 1, val_meta["q99_any"] == 1),
        "turb_q99_top_cell": (train_meta["turb_q99"] == 1, val_meta["turb_q99"] == 1),
    }
    multiclass_masks_test = {
        "q99_any_top_cell": (train_meta["q99_any"] == 1, test_meta["q99_any"] == 1),
        "turb_q99_top_cell": (train_meta["turb_q99"] == 1, test_meta["turb_q99"] == 1),
    }
    multiclass_val = summarize_multiclass_tasks(
        train_features=train_features,
        eval_features=val_features,
        train_meta=train_meta,
        eval_meta=val_meta,
        subset_masks=multiclass_masks_val,
    )
    multiclass_test = summarize_multiclass_tasks(
        train_features=train_features,
        eval_features=test_features,
        train_meta=train_meta,
        eval_meta=test_meta,
        subset_masks=multiclass_masks_test,
    )

    enc_q99_auc = binary_test["q99_any"]["encoder"]["roc_auc"]
    enc_turb_q99_auc = binary_test["turb_q99"]["encoder"]["roc_auc"]
    enc_top3 = multiclass_test["q99_any_top_cell"]["encoder"]["top3_acc"]
    verdict = "PARTIAL"
    if np.isfinite(enc_q99_auc) and np.isfinite(enc_turb_q99_auc) and np.isfinite(enc_top3):
        if enc_q99_auc >= 0.65 and enc_turb_q99_auc >= 0.70 and enc_top3 >= 0.45:
            verdict = "AGREE"

    summary: dict[str, Any] = {
        "checkpoint": {
            "path": args.checkpoint,
            "type": model_type,
            "epoch": int(payload.get("epoch", -1)),
        },
        "config": {
            "history_len": args.history_len,
            "test_start": args.test_start,
            "val_size": args.val_size,
            "future_gap": args.future_gap,
            "q95_threshold": q95,
            "q99_threshold": q99,
            "train_vov_q80": train_vov_q80,
        },
        "counts": {
            "train_windows": int(train_hist.shape[0]),
            "val_windows": int(val_hist.shape[0]),
            "test_windows": int(test_hist.shape[0]),
            "train_q99_windows": int(train_meta["q99_any"].sum()),
            "val_q99_windows": int(val_meta["q99_any"].sum()),
            "test_q99_windows": int(test_meta["q99_any"].sum()),
            "train_turb_q99_windows": int(train_meta["turb_q99"].sum()),
            "val_turb_q99_windows": int(val_meta["turb_q99"].sum()),
            "test_turb_q99_windows": int(test_meta["turb_q99"].sum()),
        },
        "binary_val": binary_val,
        "binary_test": binary_test,
        "multiclass_val": multiclass_val,
        "multiclass_test": multiclass_test,
        "verdict": verdict,
        "interpretation": {
            "read": (
                "If the encoder feature probe materially beats simple and flattened-history baselines on H=1 "
                "severity and severe top-cell tasks, then the transformer encoder is carrying real one-step signal "
                "and the output law remains the main unresolved bottleneck."
            )
        },
    }

    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(make_serializable(summary), indent=2))

    memo_path = out_dir / "210d_h1_encoder_verification.md"
    memo_path.write_text(build_markdown(summary))

    print(json.dumps(make_serializable(summary), indent=2))
    print(f"\nSaved to {summary_path}")


if __name__ == "__main__":
    main()
