#!/usr/bin/env python
"""
Focused mechanistic review of 201a against the 169c AR baseline and 183c strict anchor.

Questions:
  1. Did 201a materially improve sparse-case H=1 tail fit?
  2. Did 201a reduce the hard-late teacher-forced vs rollout gap?
  3. Did tail-aware rollout training improve localization on realized h30 tail cells,
     or mostly reshuffle global width?
  4. Is the Transformer history encoder more discriminative of hard-late sparse cases?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_169a_transformed_student_t import denormalize_iv
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import ShapeScaleStudentTARModel
from experiments.backfill.block_ar.train_201a_transformer_ar_rollout_tail_student_t import (
    TransformerRolloutTailStudentTARModel,
)
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def make_serializable(obj: Any) -> Any:
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return make_serializable(obj.detach().cpu().numpy())
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    return obj


def build_test_subset(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    max_windows: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = np.load(data_path)
    dataset = VolSurfaceDataset(
        data["surface"],
        history_len,
        future_len,
        start_idx=test_start,
    )
    n = len(dataset) if max_windows is None else min(len(dataset), max_windows)
    histories = []
    futures = []
    for i in range(n):
        item = dataset[i]
        histories.append(item["history"])
        futures.append(item["future"])
    return torch.stack(histories, dim=0), torch.stack(futures, dim=0)


def regime_masks_from_history(history_norm: torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    history_01 = denormalize_iv(history_norm)
    mean_iv = history_01.mean(dim=(-1, -2))
    daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
    vov = daily_chg.std(dim=1).cpu().numpy()
    q20 = float(np.quantile(vov, 0.2))
    q80 = float(np.quantile(vov, 0.8))
    calm = vov <= q20
    turb = vov >= q80
    return vov, calm, turb, q20, q80


def safe_mean(mask: np.ndarray, values: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(values[mask].mean())


def auc_roc_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = y_true.astype(np.int64)
    scores = scores.astype(np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    pos_rank_sum = ranks[pos].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


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


def summarize_h1_subset(
    mask: np.ndarray,
    inside90: np.ndarray,
    width90: np.ndarray,
    actual_abs_delta: np.ndarray,
    sample_abs_delta: np.ndarray,
    q95: float,
    q99: float,
) -> dict[str, float]:
    realized_q95 = actual_abs_delta >= q95
    realized_q99 = actual_abs_delta >= q99
    sample_q95_prob = (sample_abs_delta >= q95).mean(axis=1)
    sample_q99_prob = (sample_abs_delta >= q99).mean(axis=1)
    q95_mask_2d = mask[:, None] & realized_q95
    q99_mask_2d = mask[:, None] & realized_q99
    return {
        "windows": int(mask.sum()),
        "cov90": float(inside90[mask].mean()) if np.any(mask) else float("nan"),
        "width90": float(width90[mask].mean()) if np.any(mask) else float("nan"),
        "empirical_q95_rate": float(realized_q95[mask].mean()) if np.any(mask) else float("nan"),
        "empirical_q99_rate": float(realized_q99[mask].mean()) if np.any(mask) else float("nan"),
        "sample_q95_rate": float(sample_q95_prob[mask].mean()) if np.any(mask) else float("nan"),
        "sample_q99_rate": float(sample_q99_prob[mask].mean()) if np.any(mask) else float("nan"),
        "coverage90_on_realized_q95_cells": safe_mean(q95_mask_2d, inside90),
        "coverage90_on_realized_q99_cells": safe_mean(q99_mask_2d, inside90),
        "width90_on_realized_q95_cells": safe_mean(q95_mask_2d, width90),
        "width90_on_realized_q99_cells": safe_mean(q99_mask_2d, width90),
        "sample_q95_prob_on_realized_q95_cells": safe_mean(q95_mask_2d, sample_q95_prob),
        "sample_q99_prob_on_realized_q99_cells": safe_mean(q99_mask_2d, sample_q99_prob),
    }


def load_model(checkpoint_path: str, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = payload["config"]
    model_type = raw_config["type"]
    if model_type == "multi_step_student_t_169c":
        model = ShapeScaleStudentTARModel(
            encoder_config=EncoderConfig(**raw_config["encoder"]),
            decoder_config=raw_config["decoder"],
            support_lo=raw_config.get("support_lo", 0.01),
            support_hi=raw_config.get("support_hi", 1.0),
            support_eps=raw_config.get("support_eps", 1e-5),
            cov_jitter=raw_config.get("cov_jitter", 1e-4),
        )
    elif model_type == "transformer_ar_rollout_tail_student_t_201a":
        model = TransformerRolloutTailStudentTARModel(
            encoder_config=raw_config["encoder"],
            decoder_config=raw_config["decoder"],
            support_lo=raw_config.get("support_lo", 0.01),
            support_hi=raw_config.get("support_hi", 1.0),
            support_eps=raw_config.get("support_eps", 1e-5),
            cov_jitter=raw_config.get("cov_jitter", 1e-4),
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, model_type, payload


def sample_next_iv_batched(
    model,
    history_01: torch.Tensor,
    n_samples: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outs = []
    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        hist = history_01[start:end].to(device)
        samp = model.sample_next_iv(hist, n_samples=n_samples).detach().cpu().numpy()
        outs.append(samp)
    return np.concatenate(outs, axis=0)


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


def fit_linear_probe_auc(
    features: np.ndarray,
    labels: np.ndarray,
    seed: int = 42,
    epochs: int = 400,
    lr: float = 5e-2,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = features.shape[0]
    order = rng.permutation(n)
    split = max(int(0.7 * n), 1)
    train_idx = order[:split]
    test_idx = order[split:]
    x = features.astype(np.float32)
    mean = x[train_idx].mean(axis=0, keepdims=True)
    std = x[train_idx].std(axis=0, keepdims=True) + 1e-6
    x = (x - mean) / std

    x_train = torch.from_numpy(x[train_idx])
    y_train = torch.from_numpy(labels[train_idx].astype(np.float32))
    x_test = torch.from_numpy(x[test_idx])
    y_test = labels[test_idx].astype(np.int64)

    probe = torch.nn.Linear(x.shape[1], 1)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-3)
    for _ in range(epochs):
        logits = probe(x_train).squeeze(-1)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_scores = probe(x_train).squeeze(-1).numpy()
        test_scores = probe(x_test).squeeze(-1).numpy()

    return {
        "train_auc": auc_roc_binary(labels[train_idx], train_scores),
        "test_auc": auc_roc_binary(y_test, test_scores),
        "train_pos_rate": float(labels[train_idx].mean()),
        "test_pos_rate": float(y_test.mean()) if y_test.size else float("nan"),
    }


def extract_features(model, history_01: torch.Tensor, batch_size: int, device: torch.device) -> np.ndarray:
    feats = []
    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        hist = history_01[start:end].to(device)
        with torch.no_grad():
            enc = model.encode(hist)
            if isinstance(enc, tuple):
                enc = enc[0]
            feats.append(enc.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def summarize_h30_tail_localization(
    window_mask: np.ndarray,
    target_prev: np.ndarray,
    target_h30: np.ndarray,
    sample_h30: np.ndarray,
    q95: float,
    q99: float,
) -> dict[str, float]:
    target_abs_delta = np.abs(target_h30 - target_prev)
    sample_abs_delta = np.abs(sample_h30 - target_prev[:, None, :])
    lo = np.quantile(sample_h30, 0.05, axis=1)
    hi = np.quantile(sample_h30, 0.95, axis=1)
    inside = (target_h30 >= lo) & (target_h30 <= hi)
    width = hi - lo
    sample_q95_prob = (sample_abs_delta >= q95).mean(axis=1)
    sample_q99_prob = (sample_abs_delta >= q99).mean(axis=1)
    realized_q95 = target_abs_delta >= q95
    realized_q99 = target_abs_delta >= q99
    q95_mask = window_mask[:, None] & realized_q95
    q99_mask = window_mask[:, None] & realized_q99
    return {
        "coverage90_on_realized_q95_cells": safe_mean(q95_mask, inside),
        "coverage90_on_realized_q99_cells": safe_mean(q99_mask, inside),
        "width90_on_realized_q95_cells": safe_mean(q95_mask, width),
        "width90_on_realized_q99_cells": safe_mean(q99_mask, width),
        "sample_q95_prob_on_realized_q95_cells": safe_mean(q95_mask, sample_q95_prob),
        "sample_q99_prob_on_realized_q99_cells": safe_mean(q99_mask, sample_q99_prob),
    }


def evaluate_model(
    name: str,
    checkpoint_path: str,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    q95: float,
    q99: float,
    calm: np.ndarray,
    turb: np.ndarray,
    hard_late: np.ndarray,
    clean_late_turb: np.ndarray,
    batch_size: int,
    h1_samples: int,
    h30_samples: int,
    device: torch.device,
) -> dict[str, Any]:
    model, model_type, payload = load_model(checkpoint_path, device)
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm)
    history_flat = history_01.reshape(history_01.shape[0], history_01.shape[1], -1).numpy()
    future_flat = future_01.reshape(future_01.shape[0], future_01.shape[1], -1).numpy()

    prev_h1 = history_flat[:, -1, :]
    target_h1 = future_flat[:, 0, :]
    h1_samples_arr = sample_next_iv_batched(model, history_01, n_samples=h1_samples, batch_size=batch_size, device=device)
    h1_lo = np.quantile(h1_samples_arr, 0.05, axis=1)
    h1_hi = np.quantile(h1_samples_arr, 0.95, axis=1)
    h1_inside = (target_h1 >= h1_lo) & (target_h1 <= h1_hi)
    h1_width = h1_hi - h1_lo
    h1_sample_abs_delta = np.abs(h1_samples_arr - prev_h1[:, None, :])
    h1_actual_abs_delta = np.abs(target_h1 - prev_h1)

    h30_prev = future_flat[:, 28, :]
    target_h30 = future_flat[:, 29, :]
    tf_h30 = sample_teacher_forced_horizon(
        model,
        history_01,
        future_01,
        hidx=29,
        n_samples=h30_samples,
        batch_size=batch_size,
        device=device,
    )
    tf_lo = np.quantile(tf_h30, 0.05, axis=1)
    tf_hi = np.quantile(tf_h30, 0.95, axis=1)
    tf_mean = tf_h30.mean(axis=1)
    tf_cov = ((target_h30 >= tf_lo) & (target_h30 <= tf_hi)).mean(axis=1)
    tf_width = (tf_hi - tf_lo).mean(axis=1)
    tf_mae = np.abs(tf_mean - target_h30).mean(axis=1)

    rollout = model.sample_batched(
        history_norm.to(device),
        n_samples=h30_samples,
        n_steps=30,
        chunk_size=min(batch_size, h30_samples),
    ).detach().cpu().numpy()
    ro_h30 = rollout[:, :, 29].reshape(history_01.shape[0], h30_samples, -1)
    ro_lo = np.quantile(ro_h30, 0.05, axis=1)
    ro_hi = np.quantile(ro_h30, 0.95, axis=1)
    ro_mean = ro_h30.mean(axis=1)
    ro_cov = ((target_h30 >= ro_lo) & (target_h30 <= ro_hi)).mean(axis=1)
    ro_width = (ro_hi - ro_lo).mean(axis=1)
    ro_mae = np.abs(ro_mean - target_h30).mean(axis=1)

    features = extract_features(model, history_01, batch_size=batch_size, device=device)
    probe = fit_linear_probe_auc(features, hard_late.astype(np.int64))

    subsets = {
        "all": np.ones(history_01.shape[0], dtype=bool),
        "calm": calm,
        "turb": turb,
        "hard_late": hard_late,
        "clean_late_turb": clean_late_turb,
    }

    h1_summary = {
        subset_name: summarize_h1_subset(
            subset_mask,
            h1_inside,
            h1_width,
            h1_actual_abs_delta,
            h1_sample_abs_delta,
            q95,
            q99,
        )
        for subset_name, subset_mask in subsets.items()
    }

    rollout_summary = {}
    for subset_name, subset_mask in subsets.items():
        tf_stats = subset_stats(subset_mask, tf_cov, tf_width, tf_mae)
        ro_stats = subset_stats(subset_mask, ro_cov, ro_width, ro_mae)
        rollout_summary[subset_name] = {
            "teacher_forced": tf_stats,
            "rollout": ro_stats,
            "gap_rollout_minus_teacher_forced": {
                "cov90": ro_stats["cov90"] - tf_stats["cov90"],
                "width90": ro_stats["width90"] - tf_stats["width90"],
                "mae": ro_stats["mae"] - tf_stats["mae"],
            },
            "teacher_forced_localization": summarize_h30_tail_localization(
                subset_mask, h30_prev, target_h30, tf_h30, q95, q99
            ),
            "rollout_localization": summarize_h30_tail_localization(
                subset_mask, h30_prev, target_h30, ro_h30, q95, q99
            ),
        }

    out = {
        "model_name": name,
        "checkpoint_path": checkpoint_path,
        "model_type": model_type,
        "epoch": int(payload.get("epoch", -1)),
        "h1_tail_fit": h1_summary,
        "h30_rollout_gap": rollout_summary,
        "encoder_probe_hard_late": probe,
    }
    if model_type == "transformer_ar_rollout_tail_student_t_201a":
        with torch.no_grad():
            attns = []
            for start in range(0, history_01.shape[0], batch_size):
                end = min(start + batch_size, history_01.shape[0])
                _, attn = model.encode(history_01[start:end].to(device), return_attention=True)
                attns.append(attn.detach().cpu().numpy())
            attn = np.concatenate(attns, axis=0)
        out["attention"] = {
            "top1_mean": float(attn.max(axis=1).mean()),
            "entropy_mean": float((-(attn * np.log(np.clip(attn, 1e-8, None))).sum(axis=1)).mean()),
            "hard_late_top1": float(attn[hard_late].max(axis=1).mean()) if np.any(hard_late) else float("nan"),
            "non_hard_top1": float(attn[~hard_late].max(axis=1).mean()) if np.any(~hard_late) else float("nan"),
        }
    return out


def load_strict_summary(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused 201a mechanism review vs 169c/183c")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4540)
    parser.add_argument("--max_windows", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--h1_samples", type=int, default=128)
    parser.add_argument("--h30_samples", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--baseline_ckpt", type=str, default="models/backfill/student_t_169c/best_model.pt")
    parser.add_argument(
        "--best_ckpt",
        type=str,
        default="models/backfill/transformer_ar_rollout_tail_student_t_201a/best_model.pt",
    )
    parser.add_argument(
        "--final_ckpt",
        type=str,
        default="models/backfill/transformer_ar_rollout_tail_student_t_201a/final_model.pt",
    )
    parser.add_argument(
        "--anchor_summary",
        type=str,
        default="results/block_ar/183c_best_v2_s3mrjspec_full_30d/summary.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/validations/2026-04-07/analysis/201a_transformer_rollout_mechanistic",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    raw = np.load(args.data_path)
    train_surface = raw["surface"][: args.test_start].astype(np.float32)
    train_abs_delta = np.abs(np.diff(train_surface, axis=0).reshape(-1))
    q95 = float(np.quantile(train_abs_delta, 0.95))
    q99 = float(np.quantile(train_abs_delta, 0.99))

    vov, calm, turb, q20_vov, q80_vov = regime_masks_from_history(history_norm)
    prev = history_01[:, -1].reshape(history_01.shape[0], -1).numpy()
    target_h30 = future_flat[:, -1, :]
    h30_energy = np.abs(target_h30 - prev).sum(axis=1)
    q80_h30_turb = float(np.quantile(h30_energy[turb], 0.8))
    hard_late = turb & (h30_energy >= q80_h30_turb)
    clean_late_turb = turb & ~hard_late

    summary = {
        "config": {
            "max_windows": int(history_norm.shape[0]),
            "h1_samples": args.h1_samples,
            "h30_samples": args.h30_samples,
            "batch_size": args.batch_size,
        },
        "thresholds": {
            "train_abs_q95": q95,
            "train_abs_q99": q99,
            "history_vov_q20": q20_vov,
            "history_vov_q80": q80_vov,
            "h30_energy_q80_within_turb": q80_h30_turb,
        },
        "subset_sizes": {
            "all": int(history_norm.shape[0]),
            "calm": int(calm.sum()),
            "turb": int(turb.sum()),
            "hard_late": int(hard_late.sum()),
            "clean_late_turb": int(clean_late_turb.sum()),
        },
        "models": {
            "169c_best": evaluate_model(
                "169c_best",
                args.baseline_ckpt,
                history_norm,
                future_norm,
                q95,
                q99,
                calm,
                turb,
                hard_late,
                clean_late_turb,
                args.batch_size,
                args.h1_samples,
                args.h30_samples,
                device,
            ),
            "201a_best": evaluate_model(
                "201a_best",
                args.best_ckpt,
                history_norm,
                future_norm,
                q95,
                q99,
                calm,
                turb,
                hard_late,
                clean_late_turb,
                args.batch_size,
                args.h1_samples,
                args.h30_samples,
                device,
            ),
            "201a_final": evaluate_model(
                "201a_final",
                args.final_ckpt,
                history_norm,
                future_norm,
                q95,
                q99,
                calm,
                turb,
                hard_late,
                clean_late_turb,
                args.batch_size,
                args.h1_samples,
                args.h30_samples,
                device,
            ),
        },
        "strict_anchor_183c": load_strict_summary(args.anchor_summary),
    }

    mech = {
        "read": (
            "If 201a materially improves H=1 realized-tail fit or reduces the hard-late rollout gap relative to "
            "169c, then the rollout-consistent tail-aware training idea has real sparse-case signal even if the "
            "strict suite still fails. If not, the branch mostly reshuffled width without solving the sparse case."
        )
    }
    summary["interpretation"] = mech

    out_path = output_dir / "mechanistic_summary.json"
    out_path.write_text(json.dumps(make_serializable(summary), indent=2))
    print(json.dumps(make_serializable(summary), indent=2))
    print(f"\nSaved mechanistic summary to {out_path}")


if __name__ == "__main__":
    main()
