#!/usr/bin/env python
"""
Fixed smoke-gate evaluation for 207a.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_201b_targeting_jump_mechanism import (
    make_serializable,
    regime_masks_from_history,
    sample_teacher_forced_horizon,
    summarize_abs_delta_shape,
    summarize_jump_distribution,
    summarize_targeting,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_201a_transformer_ar_rollout_tail_student_t import (
    evaluate_rollout_subset,
    reshape_history,
)
from experiments.backfill.block_ar.train_207a_local_conditional_family_student_t import (
    TransformerLocalConditionalFamilyARModel,
)


def load_model(checkpoint_path: str, device: torch.device):
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw_config = payload["config"]
    if raw_config["type"] != "transformer_local_conditional_family_student_t_207a":
        raise ValueError(f"Unexpected model type: {raw_config['type']}")
    model = TransformerLocalConditionalFamilyARModel(
        encoder_config=raw_config["encoder"],
        decoder_config=raw_config["decoder"],
        support_lo=raw_config.get("support_lo", 0.01),
        support_hi=raw_config.get("support_hi", 1.0),
        support_eps=raw_config.get("support_eps", 1e-5),
        cov_jitter=raw_config.get("cov_jitter", 1e-4),
    )
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.to(device).eval()
    return model, payload


@torch.no_grad()
def evaluate_family_stats(
    model: TransformerLocalConditionalFamilyARModel,
    history_01: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    total = {
        "gate_prob_mean": 0.0,
        "gate_prob_min": 0.0,
        "gate_prob_max": 0.0,
        "family_top1_mean": 0.0,
        "family_entropy_mean": 0.0,
        "family_scale_mean": 0.0,
        "family_union_size_mean": 0.0,
        "family_pairwise_cos_mean": 0.0,
    }
    count = 0
    for start in range(0, history_01.shape[0], batch_size):
        end = min(start + batch_size, history_01.shape[0])
        stats = model.family_statistics_from_history(history_01[start:end].to(device))
        bs = end - start
        for k in total:
            total[k] += float(stats[k].item()) * bs
        count += bs
    return {k: v / max(count, 1) for k, v in total.items()}


@torch.no_grad()
def evaluate_family_h30_allocation(
    model: TransformerLocalConditionalFamilyARModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    hard_late_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    future_flat = future_01 if future_01.dim() == 3 else future_01.reshape(future_01.shape[0], future_01.shape[1], -1)
    n_windows = history_01.shape[0]

    family_union_hits = []
    width_top1_hits = []
    family_union_sizes = []
    gate_probs = []
    family_top1_probs = []

    for start in range(0, n_windows, batch_size):
        end = min(start + batch_size, n_windows)
        context = history_01[start:end].to(device)
        fut = future_flat[start:end].to(device)
        for step in range(fut.shape[1]):
            mu, factor, diag, scale, nu = model.forward_from_history(context)
            if step == fut.shape[1] - 1:
                family = model.last_family_params()
                cov = model.covariance(factor, diag, scale)
                std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(1e-8))
                prev = reshape_history(context)[:, -1]
                target = fut[:, step]
                target_top = (target - prev).abs().argmax(dim=-1).detach().cpu().numpy()
                width_top = std.argmax(dim=-1).detach().cpu().numpy()
                family_top_cells = family["family_shapes"].abs().argmax(dim=-1).detach().cpu().numpy()
                gate_prob = torch.sigmoid(family["gate_logit"]).detach().cpu().numpy()
                family_prob_top1 = torch.softmax(family["family_logits"], dim=-1).max(dim=-1).values.detach().cpu().numpy()
                batch_mask = hard_late_mask[start:end]
                for i, active in enumerate(batch_mask):
                    if not active:
                        continue
                    union = np.unique(family_top_cells[i])
                    family_union_hits.append(float(target_top[i] in union))
                    width_top1_hits.append(float(target_top[i] == width_top[i]))
                    family_union_sizes.append(float(union.size))
                    gate_probs.append(float(gate_prob[i]))
                    family_top1_probs.append(float(family_prob_top1[i]))
            next_frame = fut[:, step].view(end - start, 1, 5, 5)
            context = torch.cat([context[:, 1:], next_frame], dim=1)

    if not family_union_hits:
        return {
            "n_hard_late": 0,
            "family_union_top1_acc": float("nan"),
            "width_top1_acc": float("nan"),
            "family_vs_width_top1_lift": float("nan"),
            "family_union_size_mean": float("nan"),
            "family_gate_prob_mean": float("nan"),
            "family_prob_top1_mean": float("nan"),
        }

    return {
        "n_hard_late": int(len(family_union_hits)),
        "family_union_top1_acc": float(np.mean(family_union_hits)),
        "width_top1_acc": float(np.mean(width_top1_hits)),
        "family_vs_width_top1_lift": float(np.mean(family_union_hits) - np.mean(width_top1_hits)),
        "family_union_size_mean": float(np.mean(family_union_sizes)),
        "family_gate_prob_mean": float(np.mean(gate_probs)),
        "family_prob_top1_mean": float(np.mean(family_top1_probs)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="207a smoke gate")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--h30_samples", type=int, default=48)
    parser.add_argument("--rollout_val_samples", type=int, default=8)
    parser.add_argument("--rollout_eval_limit", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    train_abs_delta = np.abs(np.diff(surfaces[: args.test_start], axis=0).reshape(-1))
    q95 = float(np.quantile(train_abs_delta, 0.95))
    q99 = float(np.quantile(train_abs_delta, 0.99))

    hist_len = args.history_len
    future_len = args.future_len
    max_train_idx = args.test_start - hist_len - future_len
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    val_indices = val_indices[: args.max_val_windows]

    surf_tensor = torch.from_numpy(surfaces).to(device)
    history_01, future_01 = build_multistep_windows(val_indices, surf_tensor, hist_len, future_len)
    history_norm = normalize_iv(history_01)
    future_grid_01 = future_01.view(future_01.shape[0], future_01.shape[1], 5, 5)

    model, payload = load_model(args.checkpoint, device)
    val_loader = DataLoader(TensorDataset(history_01, future_01), batch_size=args.batch_size, shuffle=False)
    rollout_diag = evaluate_rollout_subset(
        model,
        val_loader,
        rollout_val_samples=args.rollout_val_samples,
        rollout_eval_limit=args.rollout_eval_limit,
    )

    future_flat = future_01.reshape(future_01.shape[0], future_01.shape[1], -1).detach().cpu().numpy()
    history_flat = history_01.reshape(history_01.shape[0], history_01.shape[1], -1).detach().cpu().numpy()
    vov, calm, turb, q20_vov, q80_vov = regime_masks_from_history(history_norm.detach().cpu())
    prev = history_flat[:, -1, :]
    target_h30 = future_flat[:, -1, :]
    h30_energy = np.abs(target_h30 - prev).sum(axis=1)
    q80_h30_turb = float(np.quantile(h30_energy[turb], 0.8)) if np.any(turb) else float("nan")
    hard_late = turb & (h30_energy >= q80_h30_turb) if np.any(turb) else np.zeros_like(turb, dtype=bool)
    clean_late_turb = turb & ~hard_late

    tf_h30 = sample_teacher_forced_horizon(
        model,
        history_01,
        future_grid_01,
        hidx=29,
        n_samples=args.h30_samples,
        batch_size=args.batch_size,
        device=device,
    )
    rollout = model.sample_batched(
        history_norm.to(device),
        n_samples=args.h30_samples,
        n_steps=args.future_len,
        chunk_size=min(args.batch_size, args.h30_samples),
    ).detach().cpu().numpy()
    ro_h30 = rollout[:, :, 29].reshape(history_01.shape[0], args.h30_samples, -1)

    subsets = {
        "all": np.ones(history_01.shape[0], dtype=bool),
        "calm": calm,
        "turb": turb,
        "hard_late": hard_late,
        "clean_late_turb": clean_late_turb,
    }
    jump = summarize_jump_distribution(
        history_flat,
        future_flat,
        rollout.reshape(history_01.shape[0], args.h30_samples, args.future_len, -1),
        q95,
        q99,
        subsets,
    )
    shape = summarize_abs_delta_shape(
        history_flat,
        future_flat,
        rollout.reshape(history_01.shape[0], args.h30_samples, args.future_len, -1),
        subsets,
    )
    targeting = summarize_targeting(future_flat, tf_h30, ro_h30, hard_late, q95, q99)
    family_stats = evaluate_family_stats(model, history_01, args.batch_size, device)
    family_h30 = evaluate_family_h30_allocation(model, history_01, future_01, hard_late, args.batch_size, device)

    gate_checks = {
        "best_checkpoint_post_rollout": int(payload.get("epoch", -1)) >= 4,
        "rollout_mae_ok": rollout_diag["rollout_mae"] <= 0.060,
        "rollout_rank_ok": 0.90 <= rollout_diag["rollout_rank_ratio_h30"] <= 1.25,
        "rollout_corr_ok": 0.90 <= rollout_diag["rollout_corr_ratio_h30"] <= 1.15,
        "hard_late_top1_ok": targeting.get("rollout", {}).get("top1_matches_true_top1", float("nan")) >= 0.22,
        "hard_late_support_ok": targeting.get("rollout", {}).get("effective_support", float("inf")) <= 15.5,
        "q99_jump_count_ok": jump.get("all", {}).get("ro_q99_count_mean", float("inf")) <= 12.0,
        "quiet_ratio_ok": shape.get("all", {}).get("quiet_ratio", float("nan")) >= 0.93,
        "shoulder_ratio_ok": shape.get("all", {}).get("shoulder_ratio", float("inf")) <= 1.05,
        "family_beats_width_top1": family_h30.get("family_vs_width_top1_lift", float("-inf")) >= 0.10,
        "family_union_size_ok": family_h30.get("family_union_size_mean", float("inf")) <= 2.0,
    }

    summary = {
        "checkpoint": args.checkpoint,
        "epoch": int(payload.get("epoch", -1)),
        "n_val_windows": int(history_01.shape[0]),
        "thresholds": {
            "q95": q95,
            "q99": q99,
            "history_vov_q20": q20_vov,
            "history_vov_q80": q80_vov,
            "h30_turb_q80": q80_h30_turb,
        },
        "counts": {
            "calm_windows": int(calm.sum()),
            "turb_windows": int(turb.sum()),
            "hard_late_windows": int(hard_late.sum()),
        },
        "rollout_diag": rollout_diag,
        "targeting_h30": targeting,
        "pathwise_jump_realism": jump,
        "abs_delta_shape": shape,
        "family_stats": family_stats,
        "family_h30_allocation": family_h30,
        "smoke_gate_checks": gate_checks,
        "smoke_gate_pass": bool(all(bool(v) for v in gate_checks.values())),
    }

    out_path.write_text(json.dumps(make_serializable(summary), indent=2))
    print(json.dumps(make_serializable(summary["smoke_gate_checks"]), indent=2))
    print(f"smoke_gate_pass={summary['smoke_gate_pass']}")


if __name__ == "__main__":
    main()
