#!/usr/bin/env python
"""516a: audit whether observable market state predicts the remaining failures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    loc = x.mean(axis=0, keepdims=True)
    scale = x.std(axis=0, keepdims=True)
    return loc, np.maximum(scale, 1e-8)


def standardize_apply(x: np.ndarray, loc: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (x - loc) / scale


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    xtx = x.T @ x
    reg = float(alpha) * np.eye(xtx.shape[0], dtype=np.float64)
    return np.linalg.solve(xtx + reg, x.T @ y)


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-12)


def fit_eval_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    alphas: list[float],
) -> dict[str, object]:
    inner_split = max(8, int(round(0.8 * x_train.shape[0])))
    x_inner, x_val = x_train[:inner_split], x_train[inner_split:]
    y_inner, y_val = y_train[:inner_split], y_train[inner_split:]

    x_loc, x_scale = standardize_fit(x_inner)
    y_loc, y_scale = standardize_fit(y_inner)
    x_inner_z = standardize_apply(x_inner, x_loc, x_scale)
    x_val_z = standardize_apply(x_val, x_loc, x_scale)
    y_inner_z = standardize_apply(y_inner, y_loc, y_scale)
    y_val_z = standardize_apply(y_val, y_loc, y_scale)

    best_alpha = float(alphas[0])
    best_score = -np.inf
    for alpha in alphas:
        coef = ridge_fit(x_inner_z, y_inner_z, alpha)
        pred = x_val_z @ coef
        score = float(np.mean(r2_score(y_val_z, pred)))
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)

    x_loc, x_scale = standardize_fit(x_train)
    y_loc, y_scale = standardize_fit(y_train)
    x_train_z = standardize_apply(x_train, x_loc, x_scale)
    x_test_z = standardize_apply(x_test, x_loc, x_scale)
    y_train_z = standardize_apply(y_train, y_loc, y_scale)
    y_test_z = standardize_apply(y_test, y_loc, y_scale)
    coef = ridge_fit(x_train_z, y_train_z, best_alpha)
    pred = x_test_z @ coef
    r2 = r2_score(y_test_z, pred)
    return {
        "alpha": best_alpha,
        "inner_val_mean_r2": best_score,
        "holdout_mean_r2": float(np.mean(r2)),
        "holdout_median_r2": float(np.median(r2)),
        "holdout_min_r2": float(np.min(r2)),
        "holdout_max_r2": float(np.max(r2)),
        "n_targets": int(y_train.shape[1]),
    }


def build_windows(
    surfaces: np.ndarray,
    ret: np.ndarray,
    price: np.ndarray,
    slopes: np.ndarray,
    skews: np.ndarray,
    levels: np.ndarray,
    indices: np.ndarray,
    history_len: int,
    future_len: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    iv_features = []
    obs_features = []
    targets: dict[str, list[np.ndarray]] = {
        "future_mean_level_scalar": [],
        "future_mean_level_cells": [],
        "future_h30_level_cells": [],
        "future_vov_scalar": [],
        "future_max_abs_jump_scalar": [],
        "future_q90_abs_jump_cells": [],
    }
    obs_panel = np.stack(
        [
            ret,
            np.log(np.maximum(price, 1e-8)),
            slopes,
            skews,
            levels,
        ],
        axis=1,
    ).astype(np.float64)

    for idx in indices:
        hist = surfaces[idx : idx + history_len].reshape(history_len, -1).astype(np.float64)
        fut = surfaces[idx + history_len : idx + history_len + future_len].reshape(future_len, -1).astype(np.float64)
        hist_mean_path = hist.mean(axis=1)
        hist_deltas = np.diff(hist, axis=0)
        vov = np.std(np.diff(hist_mean_path))
        iv_features.append(
            np.concatenate(
                [
                    hist[-1],
                    hist.mean(axis=0),
                    hist.std(axis=0),
                    hist[-1] - hist[0],
                    np.array(
                        [
                            hist[-1].mean(),
                            hist.mean(),
                            hist.std(),
                            vov,
                            np.abs(hist_deltas).max(),
                            np.quantile(np.abs(hist_deltas), 0.9),
                        ],
                        dtype=np.float64,
                    ),
                ]
            )
        )

        obs_hist = obs_panel[idx : idx + history_len]
        obs_features.append(
            np.concatenate(
                [
                    obs_hist[-1],
                    obs_hist.mean(axis=0),
                    obs_hist.std(axis=0),
                    obs_hist[-1] - obs_hist[0],
                ]
            )
        )

        fut_mean_path = fut.mean(axis=1)
        fut_deltas = np.diff(fut, axis=0)
        abs_fut_deltas = np.abs(fut_deltas)
        targets["future_mean_level_scalar"].append(np.array([fut.mean()], dtype=np.float64))
        targets["future_mean_level_cells"].append(fut.mean(axis=0))
        targets["future_h30_level_cells"].append(fut[-1])
        targets["future_vov_scalar"].append(np.array([np.std(np.diff(fut_mean_path))], dtype=np.float64))
        targets["future_max_abs_jump_scalar"].append(np.array([abs_fut_deltas.max()], dtype=np.float64))
        targets["future_q90_abs_jump_cells"].append(np.quantile(abs_fut_deltas, 0.9, axis=0))

    y = {key: np.stack(value, axis=0) for key, value in targets.items()}
    return np.stack(iv_features, axis=0), np.stack(obs_features, axis=0), y


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--audit_windows", type=int, default=1600)
    parser.add_argument("--holdout_frac", type=float, default=0.25)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_md", required=True)
    args = parser.parse_args()

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float64)
    max_train_idx = args.test_start - args.history_len - args.future_len
    val_start = max_train_idx - args.val_size
    n = min(int(args.audit_windows), int(val_start))
    indices = np.arange(val_start - n, val_start)

    x_iv, x_obs, targets = build_windows(
        surfaces=surfaces,
        ret=raw["ret"].astype(np.float64),
        price=raw["price"].astype(np.float64),
        slopes=raw["slopes"].astype(np.float64),
        skews=raw["skews"].astype(np.float64),
        levels=raw["levels"].astype(np.float64),
        indices=indices,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    x_joint = np.concatenate([x_iv, x_obs], axis=1)
    split = int(round((1.0 - float(args.holdout_frac)) * n))
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    result_targets: dict[str, object] = {}
    rows = []
    for name, y in targets.items():
        iv_res = fit_eval_ridge(x_iv[:split], y[:split], x_iv[split:], y[split:], alphas)
        joint_res = fit_eval_ridge(x_joint[:split], y[:split], x_joint[split:], y[split:], alphas)
        delta = float(joint_res["holdout_mean_r2"] - iv_res["holdout_mean_r2"])
        result_targets[name] = {
            "iv_only": iv_res,
            "iv_plus_observable_state": joint_res,
            "delta_mean_r2": delta,
        }
        rows.append(
            (
                name,
                iv_res["holdout_mean_r2"],
                joint_res["holdout_mean_r2"],
                delta,
                joint_res["holdout_median_r2"],
            )
        )

    results = {
        "config": {
            "data_path": args.data_path,
            "history_len": int(args.history_len),
            "future_len": int(args.future_len),
            "test_start": int(args.test_start),
            "val_size": int(args.val_size),
            "val_start": int(val_start),
            "audit_windows": int(n),
            "holdout_frac": float(args.holdout_frac),
            "index_start": int(indices[0]),
            "index_end": int(indices[-1]),
            "iv_feature_dim": int(x_iv.shape[1]),
            "observable_feature_dim": int(x_obs.shape[1]),
        },
        "targets": result_targets,
    }
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    lines = [
        "# 516a Observable-State Predictability Audit",
        "",
        "This audit uses only pre-validation rolling windows. It compares ridge models using IV-history summaries alone versus IV-history plus observable market-state summaries (`ret`, `price`, `slopes`, `skews`, `levels`).",
        "",
        f"- index range: `{indices[0]}..{indices[-1]}`",
        f"- train/holdout windows: `{split}/{n - split}`",
        f"- IV feature dim: `{x_iv.shape[1]}`",
        f"- observable-state feature dim: `{x_obs.shape[1]}`",
        "",
        "| target | IV-only R2 | IV+state R2 | delta | joint median R2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, iv_r2, joint_r2, delta, median_r2 in rows:
        lines.append(
            f"| `{name}` | `{iv_r2:.4f}` | `{joint_r2:.4f}` | `{delta:+.4f}` | `{median_r2:.4f}` |"
        )
    lines.extend(
        [
            "",
            "Interpretation: positive delta means the extra observable state added out-of-sample linear signal beyond IV history. Negative or near-zero delta means the broader state does not justify a new conditioning branch for that target.",
        ]
    )
    Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(results["config"], indent=2))
    for name, iv_r2, joint_r2, delta, _median_r2 in rows:
        print(f"{name}: iv={iv_r2:.4f} joint={joint_r2:.4f} delta={delta:+.4f}")


if __name__ == "__main__":
    main()
