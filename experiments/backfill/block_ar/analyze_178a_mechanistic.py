#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.regime_state_space_modules import (
    RegimeCoupledStateSpaceModel,
    aggregate_slope_ratio,
)
from experiments.backfill.block_ar.support_transforms import build_support_transform
from experiments.backfill.block_ar.train_169a_transformed_student_t import make_serializable, normalize_iv
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows


def load_model(model_path: str, device: str) -> tuple[RegimeCoupledStateSpaceModel, dict]:
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    enc_cfg = EncoderConfig(**cfg["encoder"])
    model = RegimeCoupledStateSpaceModel(
        encoder_config=enc_cfg,
        support_transform=build_support_transform(cfg["support_transform"]),
        base_nu=cfg.get("base_nu", 8.0),
        **cfg["model"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    return model, ckpt


def bucket_masks(vov: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    q20 = torch.quantile(vov, 0.2)
    q80 = torch.quantile(vov, 0.8)
    calm = vov <= q20
    turb = vov >= q80
    return calm, turb


def main():
    parser = argparse.ArgumentParser(description="178a mechanistic analysis")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--subset_windows", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["select", "val", "test"])
    args = parser.parse_args()

    device = args.device
    model, ckpt = load_model(args.model_path, device)
    cfg = ckpt["config"]
    history_len = cfg["history_len"]
    future_len = cfg["future_len"]

    surfaces = np.load("data/vol_surface_with_ret.npz")["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    test_start = 4511
    max_train_idx = test_start - history_len - future_len
    total_holdout = 441
    selection_size = total_holdout // 2
    monitor_size = total_holdout - selection_size
    if args.split == "select":
        indices = np.arange(max_train_idx - total_holdout, max_train_idx - monitor_size)
    elif args.split == "val":
        indices = np.arange(max_train_idx - monitor_size, max_train_idx)
    else:
        indices = np.arange(test_start, len(surfaces) - history_len - future_len + 1)
    indices = indices[: args.subset_windows]

    hist, future = build_multistep_windows(indices, surf_tensor, history_len, future_len)
    loader = DataLoader(TensorDataset(hist, future), batch_size=args.batch_size, shuffle=False)

    prior_probs_all = []
    post_probs_all = []
    post_idx_all = []
    vov_all = []
    kappa_all = []
    local_min_all = []
    local_max_all = []
    correction_rms_all = []
    prev_all = []
    gt_next_all = []
    det_next_all = []
    samp_next_all = []
    pred_std_h = {1: [], 7: [], 14: [], 30: []}
    abs_err_h = {1: [], 7: [], 14: [], 30: []}
    cover_h = {1: [], 7: [], 14: [], 30: []}

    with torch.no_grad():
        for history_01, future_01 in loader:
            outputs = model.rollout(
                history_01=history_01,
                future_01=future_01,
                temperature=0.0,
                hard_regimes=True,
                use_posterior=True,
            )
            prior_probs = torch.softmax(outputs.prior_logits, dim=-1)
            post_probs = torch.softmax(outputs.posterior_logits, dim=-1)
            post_idx = post_probs.argmax(dim=-1)
            prior_probs_all.append(prior_probs.cpu())
            post_probs_all.append(post_probs.cpu())
            post_idx_all.append(post_idx.cpu())

            mean_iv = history_01.mean(dim=(-1, -2))
            daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
            vov = daily_chg.std(dim=1)
            vov_all.append(vov.cpu())

            kappa_all.append(outputs.aux["kappa_mean"].cpu())
            local_min_all.append(outputs.aux["local_scale_min"].cpu())
            local_max_all.append(outputs.aux["local_scale_max"].cpu())
            correction_rms_all.append(outputs.aux["correction_rms"].cpu())

            det_u = outputs.mu_u
            det_native = model.support_transform.inverse(det_u.reshape(-1, det_u.shape[-1]))[0].view_as(det_u)
            prev_flat = history_01[:, -1].reshape(history_01.shape[0], -1)
            gt_flat = future_01.reshape(future_01.shape[0], future_01.shape[1], -1)
            prev_all.append(prev_flat.cpu())
            gt_next_all.append(gt_flat[:, 0].cpu())
            det_next_all.append(det_native[:, 0].cpu())

            samples = model.sample_batched(normalize_iv(history_01), n_samples=args.n_samples)
            samples_flat = samples.view(history_01.shape[0], args.n_samples, future_len, -1)
            samp_next_all.append(samples_flat[:, :, 0].mean(dim=1).cpu())

            for h in [1, 7, 14, 30]:
                t = h - 1
                lo = samples_flat[:, :, t].quantile(0.05, dim=1)
                hi = samples_flat[:, :, t].quantile(0.95, dim=1)
                med = samples_flat[:, :, t].median(dim=1).values
                cover = ((gt_flat[:, t].cpu() >= lo.cpu()) & (gt_flat[:, t].cpu() <= hi.cpu())).float()
                width = (hi - lo).cpu()
                approx_std = width / (2.0 * 1.645)
                abs_err = (med.cpu() - gt_flat[:, t].cpu()).abs()
                pred_std_h[h].append(approx_std)
                abs_err_h[h].append(abs_err)
                cover_h[h].append(cover)

    prior_probs = torch.cat(prior_probs_all, dim=0)
    post_probs = torch.cat(post_probs_all, dim=0)
    post_idx = torch.cat(post_idx_all, dim=0)
    vov = torch.cat(vov_all, dim=0)
    calm_mask, turb_mask = bucket_masks(vov)
    kappa = torch.cat(kappa_all, dim=0)
    local_min = torch.cat(local_min_all, dim=0)
    local_max = torch.cat(local_max_all, dim=0)
    correction_rms = torch.cat(correction_rms_all, dim=0)
    prev = torch.cat(prev_all, dim=0)
    gt_next = torch.cat(gt_next_all, dim=0)
    det_next = torch.cat(det_next_all, dim=0)
    samp_next = torch.cat(samp_next_all, dim=0)

    persistence = (post_idx[:, 1:] == post_idx[:, :-1]).float().mean().item()

    summary = {
        "checkpoint_epoch": ckpt.get("epoch"),
        "split": args.split,
        "n_windows": int(len(indices)),
        "overall": {
            "det_mr_ratio": aggregate_slope_ratio(prev, gt_next, det_next),
            "sample_mr_ratio": aggregate_slope_ratio(prev, gt_next, samp_next),
            "prior_usage_mean": prior_probs.mean(dim=(0, 1)).tolist(),
            "posterior_usage_mean": post_probs.mean(dim=(0, 1)).tolist(),
            "posterior_hard_usage_mean": torch.nn.functional.one_hot(post_idx, num_classes=model.n_regimes).float().mean(dim=(0, 1)).tolist(),
            "posterior_persistence": persistence,
            "prior_entropy_mean": (-(prior_probs * prior_probs.clamp_min(1e-8).log()).sum(dim=-1)).mean().item(),
            "posterior_entropy_mean": (-(post_probs * post_probs.clamp_min(1e-8).log()).sum(dim=-1)).mean().item(),
        },
        "by_bucket": {},
        "kappa_by_block": kappa.mean(dim=0).tolist(),
        "local_scale_min_by_block": local_min.mean(dim=0).tolist(),
        "local_scale_max_by_block": local_max.mean(dim=0).tolist(),
        "correction_rms_by_block": correction_rms.mean(dim=0).tolist(),
        "coverage": {},
    }

    for name, mask in {"calm": calm_mask, "turb": turb_mask}.items():
        if mask.any():
            summary["by_bucket"][name] = {
                "n_windows": int(mask.sum().item()),
                "prior_usage_mean": prior_probs[mask].mean(dim=(0, 1)).tolist(),
                "posterior_usage_mean": post_probs[mask].mean(dim=(0, 1)).tolist(),
                "posterior_hard_usage_mean": torch.nn.functional.one_hot(post_idx[mask], num_classes=model.n_regimes).float().mean(dim=(0, 1)).tolist(),
                "posterior_persistence": (post_idx[mask][:, 1:] == post_idx[mask][:, :-1]).float().mean().item(),
                "kappa_by_block": kappa[mask].mean(dim=0).tolist(),
                "local_scale_min_by_block": local_min[mask].mean(dim=0).tolist(),
                "local_scale_max_by_block": local_max[mask].mean(dim=0).tolist(),
                "correction_rms_by_block": correction_rms[mask].mean(dim=0).tolist(),
                "det_mr_ratio": aggregate_slope_ratio(prev[mask], gt_next[mask], det_next[mask]),
                "sample_mr_ratio": aggregate_slope_ratio(prev[mask], gt_next[mask], samp_next[mask]),
            }

    top_cells = {}
    for h in [1, 7, 14, 30]:
        cov = torch.cat(cover_h[h], dim=0)
        err = torch.cat(abs_err_h[h], dim=0)
        std = torch.cat(pred_std_h[h], dim=0)
        mean_cov = cov.mean(dim=0).view(5, 5)
        worst_idx = torch.argmin(mean_cov.view(-1)).item()
        best_idx = torch.argmax(mean_cov.view(-1)).item()
        wi = (worst_idx // 5, worst_idx % 5)
        bi = (best_idx // 5, best_idx % 5)
        summary["coverage"][f"h{h}"] = {
            "overall_90": cov.mean().item(),
            "worst_cell": {"cell": wi, "coverage": mean_cov.view(-1)[worst_idx].item()},
            "best_cell": {"cell": bi, "coverage": mean_cov.view(-1)[best_idx].item()},
            "underwide_cells": int((mean_cov.view(-1) < 0.70).sum().item()),
            "overwide_cells": int((mean_cov.view(-1) > 0.95).sum().item()),
        }
        top_cells[h] = wi

        for bucket_name, mask in {"calm": calm_mask, "turb": turb_mask}.items():
            if mask.any():
                cell_flat = wi[0] * 5 + wi[1]
                summary["coverage"][f"h{h}"][bucket_name] = {
                    "worst_cell_cov": cov[mask][:, cell_flat].mean().item(),
                    "worst_cell_abs_err": err[mask][:, cell_flat].mean().item(),
                    "worst_cell_pred_std": std[mask][:, cell_flat].mean().item(),
                    "worst_cell_err_to_std": (err[mask][:, cell_flat].mean() / std[mask][:, cell_flat].mean().clamp_min(1e-8)).item(),
                }

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(make_serializable(summary), indent=2))
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
