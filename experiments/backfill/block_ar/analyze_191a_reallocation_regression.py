#!/usr/bin/env python
"""
Focused mechanistic review of 191a regression vs the 169c AR anchor.

Questions:
  1. Is 191a's reallocation support aligned with model-relative underfit targets?
  2. Does the reallocation head actually concentrate variance on late turbulent
     recipient cells, or does it leak off-target?
  3. Why does 191a regress broad metrics (S2/S8/S10/S11) even though it
     improves broad coverage and slightly improves kurtosis?
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

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.analyze_170d_mechanisms import (
    build_test_subset,
    make_serializable,
    regime_masks_from_history,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    iv_to_unconstrained,
    normalize_iv,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    ShapeScaleStudentTARModel,
)
from experiments.backfill.block_ar.train_191a_ar_reallocation_student_t import (
    ReallocationStudentTARModel,
    build_target_reallocation,
)


def safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0:
        return float("nan")
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def load_169c_model(ckpt_path: Path, device: torch.device) -> tuple[ShapeScaleStudentTARModel, dict[str, Any]]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = ShapeScaleStudentTARModel(
        encoder_config=EncoderConfig(**cfg["encoder"]),
        decoder_config=cfg["decoder"],
        support_lo=cfg.get("support_lo", 0.01),
        support_hi=cfg.get("support_hi", 1.0),
        support_eps=cfg.get("support_eps", 1e-5),
        cov_jitter=cfg.get("cov_jitter", 1e-4),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model, payload


def load_191a_model(ckpt_path: Path, device: torch.device) -> tuple[ReallocationStudentTARModel, dict[str, Any]]:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    model = ReallocationStudentTARModel(
        encoder_config=EncoderConfig(**cfg["encoder"]),
        decoder_config=cfg["decoder"],
        reallocation_config=cfg["reallocation"],
        support_lo=cfg.get("support_lo", 0.01),
        support_hi=cfg.get("support_hi", 1.0),
        support_eps=cfg.get("support_eps", 1e-5),
        cov_jitter=cfg.get("cov_jitter", 1e-4),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model, payload


def cond_from_gru_outputs_169c(model: ShapeScaleStudentTARModel, gru_outputs: torch.Tensor) -> torch.Tensor:
    attn_logits = model.encoder.attn_proj(gru_outputs).squeeze(-1)
    attn_weights = torch.softmax(attn_logits, dim=1)
    pooled = (attn_weights.unsqueeze(-1) * gru_outputs).sum(dim=1)
    return model.encoder.bottleneck(pooled)


@torch.no_grad()
def analyze_169c_subset(
    model: ShapeScaleStudentTARModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1)

    scale_vals: list[float] = []
    scale_last_turb: list[float] = []
    n = history_01.shape[0]
    vov, q20, q80 = regime_masks_from_history(history_norm)
    turb_mask = vov >= q80

    for start in range(0, n, batch_size):
        hist_b = history_01[start : start + batch_size].to(device)
        future_b = future_01[start : start + batch_size].to(device)
        bsz, hist_len = hist_b.shape[:2]
        n_cells = future_b.shape[-1]

        hist_norm_b = normalize_iv(hist_b).reshape(bsz, hist_len, n_cells)
        gru_outputs, gru_state = model.encoder.gru(hist_norm_b)
        prev_01 = hist_b[:, -1].reshape(bsz, n_cells)

        batch_turb = turb_mask[start : start + bsz]
        for step in range(future_b.shape[1]):
            cond = cond_from_gru_outputs_169c(model, gru_outputs)
            prev_u = iv_to_unconstrained(prev_01, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)
            _mu, _factor, _diag, scale, _nu = model.decoder(cond, prev_u)
            scale_vals.extend(scale.detach().cpu().tolist())
            if step == future_b.shape[1] - 1 and np.any(batch_turb):
                scale_last_turb.extend(scale.detach().cpu().numpy()[batch_turb].tolist())

            next_frame = future_b[:, step, :]
            next_norm = normalize_iv(next_frame).unsqueeze(1)
            next_out, gru_state = model.encoder.gru(next_norm, gru_state)
            gru_outputs = torch.cat([gru_outputs, next_out], dim=1)
            prev_01 = next_frame

    return {
        "overall_scale_mean": safe_mean(scale_vals),
        "late_turb_scale_mean": safe_mean(scale_last_turb),
        "q20_vov": q20,
        "q80_vov": q80,
    }


@torch.no_grad()
def analyze_191a_subset(
    model: ReallocationStudentTARModel,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    device: torch.device,
    batch_size: int,
    target_clip: float,
) -> dict[str, Any]:
    history_01 = denormalize_iv(history_norm)
    future_01 = denormalize_iv(future_norm).reshape(future_norm.shape[0], future_norm.shape[1], -1)

    n = history_01.shape[0]
    future_len = future_01.shape[1]
    vov, q20, q80 = regime_masks_from_history(history_norm)
    turb_mask_full = vov >= q80

    overall: dict[str, list[float]] = {
        "budget": [],
        "scale": [],
        "pos_top1": [],
        "neg_top1": [],
        "recipient_overlap": [],
        "donor_overlap": [],
        "recipient_capture": [],
        "donor_capture": [],
        "wrong_sign_share": [],
        "raw_pos_mass": [],
        "raw_neg_mass": [],
        "scale_budget_corr_x": [],
        "scale_budget_corr_y": [],
    }
    late_turb: dict[str, list[float]] = {
        "budget": [],
        "scale": [],
        "pos_top1": [],
        "neg_top1": [],
        "recipient_overlap": [],
        "donor_overlap": [],
        "recipient_capture": [],
        "donor_capture": [],
        "wrong_sign_share": [],
        "target_recipient_mean_weight": [],
        "offtarget_positive_mean_weight": [],
        "target_donor_mean_weight": [],
        "offtarget_negative_mean_weight": [],
        "budget_underfit_corr_x": [],
        "budget_underfit_corr_y": [],
    }
    per_step = [
        {
            "budget": [],
            "scale": [],
            "recipient_overlap": [],
            "donor_overlap": [],
            "recipient_capture": [],
            "donor_capture": [],
            "wrong_sign_share": [],
        }
        for _ in range(future_len)
    ]

    for start in range(0, n, batch_size):
        hist_b = history_01[start : start + batch_size].to(device)
        future_b = future_01[start : start + batch_size].to(device)
        bsz, hist_len = hist_b.shape[:2]
        n_cells = future_b.shape[-1]

        hist_norm_b = normalize_iv(hist_b).reshape(bsz, hist_len, n_cells)
        gru_outputs, gru_state = model.encoder.gru(hist_norm_b)
        prev_01 = hist_b[:, -1].reshape(bsz, n_cells)
        batch_turb_mask = turb_mask_full[start : start + bsz]

        for step in range(future_len):
            cond = model.cond_from_gru_outputs(gru_outputs)
            prev_u = iv_to_unconstrained(prev_01, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)
            mu, factor, diag, scale, _nu, pos_scores, neg_scores, budget = model.decoder(cond, prev_u)
            factor_adj, diag_adj, _avg_var, pos_alloc, neg_alloc, weights = model.adjusted_components(
                factor, diag, pos_scores, neg_scores, budget
            )
            base_cov = model.base_covariance(factor, diag, scale)
            base_var = torch.diagonal(base_cov, dim1=-2, dim2=-1)

            target_t = future_b[:, step, :]
            target_u = iv_to_unconstrained(target_t, lo=model.support_lo, hi=model.support_hi, eps=model.support_eps)
            residual_sq = (target_u - mu).pow(2)
            target_plus, target_minus, target_w, raw = build_target_reallocation(
                residual_sq=residual_sq,
                base_var=base_var,
                k_plus=model.decoder.realloc_k_plus,
                k_minus=model.decoder.realloc_k_minus,
                target_clip=target_clip,
            )

            pos_gain = (weights - 1.0).clamp_min(0.0)
            neg_gain = (1.0 - weights).clamp_min(0.0)
            target_plus_mask = target_plus > 0
            target_minus_mask = target_minus > 0
            off_plus_mask = ~target_plus_mask
            off_minus_mask = ~target_minus_mask

            recipient_overlap = torch.minimum(pos_alloc, target_plus).sum(dim=-1)
            donor_overlap = torch.minimum(neg_alloc, target_minus).sum(dim=-1)

            recipient_capture = (pos_gain * target_plus_mask.float()).sum(dim=-1) / pos_gain.sum(dim=-1).clamp_min(1e-8)
            donor_capture = (neg_gain * target_minus_mask.float()).sum(dim=-1) / neg_gain.sum(dim=-1).clamp_min(1e-8)

            significant = (target_w - 1.0).abs() > 0.05
            wrong_sign = ((weights - 1.0) * (target_w - 1.0) < 0.0) & significant
            wrong_sign_share = wrong_sign.float().sum(dim=-1) / significant.float().sum(dim=-1).clamp_min(1.0)

            raw_pos_mass = raw.clamp_min(0.0).sum(dim=-1)
            raw_neg_mass = (-raw).clamp_min(0.0).sum(dim=-1)

            for name, vals in [
                ("budget", budget),
                ("scale", scale),
                ("pos_top1", pos_alloc.max(dim=-1).values),
                ("neg_top1", neg_alloc.max(dim=-1).values),
                ("recipient_overlap", recipient_overlap),
                ("donor_overlap", donor_overlap),
                ("recipient_capture", recipient_capture),
                ("donor_capture", donor_capture),
                ("wrong_sign_share", wrong_sign_share),
                ("raw_pos_mass", raw_pos_mass),
                ("raw_neg_mass", raw_neg_mass),
            ]:
                overall[name].extend(vals.detach().cpu().tolist())

            per_step[step]["budget"].extend(budget.detach().cpu().tolist())
            per_step[step]["scale"].extend(scale.detach().cpu().tolist())
            per_step[step]["recipient_overlap"].extend(recipient_overlap.detach().cpu().tolist())
            per_step[step]["donor_overlap"].extend(donor_overlap.detach().cpu().tolist())
            per_step[step]["recipient_capture"].extend(recipient_capture.detach().cpu().tolist())
            per_step[step]["donor_capture"].extend(donor_capture.detach().cpu().tolist())
            per_step[step]["wrong_sign_share"].extend(wrong_sign_share.detach().cpu().tolist())

            overall["scale_budget_corr_x"].extend(scale.detach().cpu().tolist())
            overall["scale_budget_corr_y"].extend(budget.detach().cpu().tolist())

            if step == future_len - 1 and np.any(batch_turb_mask):
                idx = torch.as_tensor(batch_turb_mask, device=device, dtype=torch.bool)
                late_turb["budget"].extend(budget[idx].detach().cpu().tolist())
                late_turb["scale"].extend(scale[idx].detach().cpu().tolist())
                late_turb["pos_top1"].extend(pos_alloc[idx].max(dim=-1).values.detach().cpu().tolist())
                late_turb["neg_top1"].extend(neg_alloc[idx].max(dim=-1).values.detach().cpu().tolist())
                late_turb["recipient_overlap"].extend(recipient_overlap[idx].detach().cpu().tolist())
                late_turb["donor_overlap"].extend(donor_overlap[idx].detach().cpu().tolist())
                late_turb["recipient_capture"].extend(recipient_capture[idx].detach().cpu().tolist())
                late_turb["donor_capture"].extend(donor_capture[idx].detach().cpu().tolist())
                late_turb["wrong_sign_share"].extend(wrong_sign_share[idx].detach().cpu().tolist())

                tgt_plus = target_plus_mask[idx]
                tgt_minus = target_minus_mask[idx]
                w_idx = weights[idx]
                pos_idx = pos_gain[idx]
                neg_idx = neg_gain[idx]
                late_turb["target_recipient_mean_weight"].extend(
                    torch.where(tgt_plus, w_idx, torch.full_like(w_idx, float("nan"))).nanmean(dim=-1).detach().cpu().tolist()
                )
                late_turb["offtarget_positive_mean_weight"].extend(
                    torch.where(off_plus_mask[idx], pos_idx, torch.full_like(pos_idx, float("nan"))).nanmean(dim=-1).detach().cpu().tolist()
                )
                late_turb["target_donor_mean_weight"].extend(
                    torch.where(tgt_minus, w_idx, torch.full_like(w_idx, float("nan"))).nanmean(dim=-1).detach().cpu().tolist()
                )
                late_turb["offtarget_negative_mean_weight"].extend(
                    torch.where(off_minus_mask[idx], neg_idx, torch.full_like(neg_idx, float("nan"))).nanmean(dim=-1).detach().cpu().tolist()
                )
                late_turb["budget_underfit_corr_x"].extend(budget[idx].detach().cpu().tolist())
                late_turb["budget_underfit_corr_y"].extend(raw_pos_mass[idx].detach().cpu().tolist())

            next_frame = future_b[:, step, :]
            next_norm = normalize_iv(next_frame).unsqueeze(1)
            next_out, gru_state = model.encoder.gru(next_norm, gru_state)
            gru_outputs = torch.cat([gru_outputs, next_out], dim=1)
            prev_01 = next_frame

    return {
        "overall": {
            "budget_mean": safe_mean(overall["budget"]),
            "scale_mean": safe_mean(overall["scale"]),
            "pos_top1_mean": safe_mean(overall["pos_top1"]),
            "neg_top1_mean": safe_mean(overall["neg_top1"]),
            "recipient_overlap_mean": safe_mean(overall["recipient_overlap"]),
            "donor_overlap_mean": safe_mean(overall["donor_overlap"]),
            "recipient_capture_mean": safe_mean(overall["recipient_capture"]),
            "donor_capture_mean": safe_mean(overall["donor_capture"]),
            "wrong_sign_share_mean": safe_mean(overall["wrong_sign_share"]),
            "raw_pos_mass_mean": safe_mean(overall["raw_pos_mass"]),
            "raw_neg_mass_mean": safe_mean(overall["raw_neg_mass"]),
            "scale_budget_corr": corr(
                np.asarray(overall["scale_budget_corr_x"]),
                np.asarray(overall["scale_budget_corr_y"]),
            ),
        },
        "late_turbulent_h30": {
            "budget_mean": safe_mean(late_turb["budget"]),
            "scale_mean": safe_mean(late_turb["scale"]),
            "pos_top1_mean": safe_mean(late_turb["pos_top1"]),
            "neg_top1_mean": safe_mean(late_turb["neg_top1"]),
            "recipient_overlap_mean": safe_mean(late_turb["recipient_overlap"]),
            "donor_overlap_mean": safe_mean(late_turb["donor_overlap"]),
            "recipient_capture_mean": safe_mean(late_turb["recipient_capture"]),
            "donor_capture_mean": safe_mean(late_turb["donor_capture"]),
            "wrong_sign_share_mean": safe_mean(late_turb["wrong_sign_share"]),
            "target_recipient_mean_weight": safe_mean(late_turb["target_recipient_mean_weight"]),
            "offtarget_positive_mean_weight": safe_mean(late_turb["offtarget_positive_mean_weight"]),
            "target_donor_mean_weight": safe_mean(late_turb["target_donor_mean_weight"]),
            "offtarget_negative_mean_weight": safe_mean(late_turb["offtarget_negative_mean_weight"]),
            "budget_underfit_corr": corr(
                np.asarray(late_turb["budget_underfit_corr_x"]),
                np.asarray(late_turb["budget_underfit_corr_y"]),
            ),
        },
        "per_step": [
            {
                "horizon": step + 1,
                "budget_mean": safe_mean(row["budget"]),
                "scale_mean": safe_mean(row["scale"]),
                "recipient_overlap_mean": safe_mean(row["recipient_overlap"]),
                "donor_overlap_mean": safe_mean(row["donor_overlap"]),
                "recipient_capture_mean": safe_mean(row["recipient_capture"]),
                "donor_capture_mean": safe_mean(row["donor_capture"]),
                "wrong_sign_share_mean": safe_mean(row["wrong_sign_share"]),
            }
            for step, row in enumerate(per_step)
        ],
        "q20_vov": q20,
        "q80_vov": q80,
    }


def compare(best: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "budget_mean",
        "scale_mean",
        "pos_top1_mean",
        "neg_top1_mean",
        "recipient_overlap_mean",
        "donor_overlap_mean",
        "recipient_capture_mean",
        "donor_capture_mean",
        "wrong_sign_share_mean",
    ]
    out: dict[str, Any] = {}
    for section in ["overall", "late_turbulent_h30"]:
        out[section] = {}
        for key in keys:
            if key in best[section] and key in final[section]:
                out[section][key] = float(final[section][key] - best[section][key])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze 191a AR reallocation regression.")
    parser.add_argument(
        "--base_ckpt",
        type=str,
        default="models/backfill/student_t_169c/best_model.pt",
    )
    parser.add_argument(
        "--best_ckpt",
        type=str,
        default="models/backfill/ar_reallocation_student_t_191a/best_model.pt",
    )
    parser.add_argument(
        "--final_ckpt",
        type=str,
        default="models/backfill/ar_reallocation_student_t_191a/final_model.pt",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max_windows", type=int, default=384)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--output_json",
        type=str,
        default="results/validations/2026-04-06/analysis/191a_reallocation_mechanistic/mechanistic_summary.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    base_model, base_payload = load_169c_model(Path(args.base_ckpt), device)
    best_model, best_payload = load_191a_model(Path(args.best_ckpt), device)
    final_model, final_payload = load_191a_model(Path(args.final_ckpt), device)

    hist_len = int(best_payload["config"]["history_len"])
    future_len = int(best_payload["config"]["future_len"])
    history_norm, future_norm = build_test_subset(
        data_path="data/vol_surface_with_ret.npz",
        history_len=hist_len,
        future_len=future_len,
        test_start=4511,
        max_windows=args.max_windows,
    )

    base_stats = analyze_169c_subset(base_model, history_norm, future_norm, device, args.batch_size)
    target_clip = float(best_payload["config"]["reallocation"].get("target_clip", 1.0))
    # In 191a configs target_clip is not inside reallocation block, so fall back to spec default.
    if not np.isfinite(target_clip):
        target_clip = 1.0
    best_stats = analyze_191a_subset(best_model, history_norm, future_norm, device, args.batch_size, target_clip=1.0)
    final_stats = analyze_191a_subset(final_model, history_norm, future_norm, device, args.batch_size, target_clip=1.0)

    summary = {
        "subset": {
            "n_windows": int(history_norm.shape[0]),
            "history_len": hist_len,
            "future_len": future_len,
        },
        "169c_reference": base_stats,
        "191a_best": best_stats,
        "191a_final": final_stats,
        "best_to_final_delta": compare(best_stats, final_stats),
        "checkpoint_epochs": {
            "169c_best": int(base_payload.get("epoch", -1)),
            "191a_best": int(best_payload.get("epoch", -1)),
            "191a_final": int(final_payload.get("epoch", -1)),
        },
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(make_serializable(summary), indent=2))
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
