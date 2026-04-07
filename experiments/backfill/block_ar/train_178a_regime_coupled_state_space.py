#!/usr/bin/env python
"""
178a: Support-aware sticky-regime mean-reverting state-space generator.

First implementation of the new research phase:
  - persistent blockwise latent regimes
  - explicit mean-reverting latent dynamics
  - regime-conditioned local covariance residuals
  - support-aware transformed-space observation law
  - Student-t residual law in v0
"""

from __future__ import annotations

import argparse
import json
import time
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
    effective_rank,
)
from experiments.backfill.block_ar.support_transforms import TransformConfig, build_support_transform
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    make_serializable,
    normalize_iv,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
)


def joint_loss(
    model: RegimeCoupledStateSpaceModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    beta_regime: float,
    lambda_usage: float,
    lambda_cov: float,
    lambda_mr: float,
    regime_temperature: float,
) -> tuple[torch.Tensor, dict]:
    logprob, outputs, aux = model.log_prob_future(
        history_01=history_01,
        future_01=future_01,
        temperature=regime_temperature,
    )
    nll = (-logprob).mean()
    kl_regime = aux["kl_regime"].mean()

    target_usage = torch.full_like(aux["regime_usage"], 1.0 / aux["regime_usage"].numel())
    usage_penalty = (aux["regime_usage"] - target_usage).pow(2).mean()
    cov_penalty = (
        model.observation.regime_factor_bank.pow(2).mean()
        + model.observation.regime_diag_bank.pow(2).mean()
        + model.observation.regime_local_bank.pow(2).mean()
    )
    mr_penalty = aux["mr_penalty"]

    loss = nll + beta_regime * kl_regime + lambda_usage * usage_penalty + lambda_cov * cov_penalty + lambda_mr * mr_penalty

    future_flat = future_01.reshape(future_01.shape[0], future_01.shape[1], -1)
    pred_01 = model.support_transform.inverse(outputs.mu_u.reshape(-1, outputs.mu_u.shape[-1]))[0].view_as(outputs.mu_u)
    det_next = pred_01[:, 0]
    prev_native = history_01[:, -1].reshape(history_01.shape[0], -1)
    gt_next = future_flat[:, 0]
    det_mr_ratio = aggregate_slope_ratio(prev_native, gt_next, det_next)

    mean_cov = aux["cov"].mean(dim=1)
    metrics = {
        "total_loss": loss,
        "joint_nll": nll,
        "kl_regime": kl_regime,
        "usage_penalty": usage_penalty,
        "cov_penalty": cov_penalty,
        "mr_penalty": mr_penalty,
        "joint_mae": (pred_01 - future_flat).abs().mean(),
        "joint_det_mr_ratio": torch.tensor(det_mr_ratio, device=history_01.device),
        "regime_usage_max": aux["regime_usage"].max(),
        "prior_entropy": aux["prior_entropy"].mean(),
        "posterior_entropy": aux["posterior_entropy"].mean(),
        "kappa_mean": outputs.aux["kappa_mean"].mean(),
        "local_scale_min": outputs.aux["local_scale_min"].mean(),
        "local_scale_max": outputs.aux["local_scale_max"].mean(),
        "cell_eff_rank": effective_rank(mean_cov).mean(),
        "white_std_mean": aux["white_std"].mean(),
    }
    return loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: RegimeCoupledStateSpaceModel,
    loader: DataLoader,
    beta_regime: float,
    lambda_usage: float,
    lambda_cov: float,
    lambda_mr: float,
    regime_temperature: float,
) -> dict:
    model.eval()
    totals = {
        "val_total_loss": 0.0,
        "val_joint_nll": 0.0,
        "val_kl_regime": 0.0,
        "val_usage_penalty": 0.0,
        "val_cov_penalty": 0.0,
        "val_mr_penalty": 0.0,
        "val_joint_mae": 0.0,
        "val_joint_det_mr_ratio": 0.0,
        "val_regime_usage_max": 0.0,
        "val_prior_entropy": 0.0,
        "val_posterior_entropy": 0.0,
        "val_kappa_mean": 0.0,
        "val_local_scale_min": 0.0,
        "val_local_scale_max": 0.0,
        "val_cell_eff_rank": 0.0,
        "val_white_std_mean": 0.0,
    }
    total_count = 0
    for history_01, future_01 in loader:
        _loss, metrics = joint_loss(
            model,
            history_01,
            future_01,
            beta_regime=beta_regime,
            lambda_usage=lambda_usage,
            lambda_cov=lambda_cov,
            lambda_mr=lambda_mr,
            regime_temperature=regime_temperature,
        )
        batch_size = history_01.shape[0]
        for key in totals:
            metric_key = key.replace("val_", "")
            totals[key] += float(metrics[metric_key].item()) * batch_size
        total_count += batch_size
    if total_count == 0:
        return {k: float("nan") for k in totals}
    return {k: v / total_count for k, v in totals.items()}


@torch.no_grad()
def evaluate_joint_subset(
    model: RegimeCoupledStateSpaceModel,
    loader: DataLoader,
    joint_samples: int,
    eval_limit: int,
) -> dict:
    model.eval()
    total_cov = 0.0
    total_width = 0.0
    total_mae = 0.0
    total_support_viol = 0.0
    total_count = 0
    all_window_widths = []
    all_vov = []
    all_sample_eff_rank = []
    prev_chunks = []
    gt_next_chunks = []
    det_next_chunks = []
    sample_next_chunks = []
    h30_worst_covs = []
    h30_best_covs = []
    window_floor_bad = []

    for history_01, future_01 in loader:
        if total_count >= eval_limit:
            break
        if total_count + history_01.shape[0] > eval_limit:
            keep = eval_limit - total_count
            history_01 = history_01[:keep]
            future_01 = future_01[:keep]

        history_norm = normalize_iv(history_01)
        samples = model.sample_batched(history_norm, n_samples=joint_samples)
        future_grid = future_01.view(history_01.shape[0], future_01.shape[1], 5, 5)
        lo = samples.quantile(0.05, dim=1)
        hi = samples.quantile(0.95, dim=1)
        median = samples.median(dim=1).values

        coverage = ((future_grid >= lo) & (future_grid <= hi)).float()
        width = (hi - lo)
        mae = (median - future_grid).abs()
        support_viol = ((samples < 0.0) | (samples > 1.0)).float().mean()

        mean_iv = history_01.mean(dim=(-1, -2))
        daily_chg = mean_iv[:, 1:] - mean_iv[:, :-1]
        vov = daily_chg.std(dim=1)
        window_width = width.mean(dim=(1, 2, 3))
        window_cov = coverage.mean(dim=(1, 2, 3))

        first_sample = samples[:, 0].reshape(history_01.shape[0], future_01.shape[1], -1)
        changes = first_sample[:, 1:] - first_sample[:, :-1]
        flat = changes.reshape(-1, changes.shape[-1]).cpu().numpy()
        corr = np.corrcoef(flat.T)
        eigvals = np.linalg.eigvalsh(corr)
        eigvals = np.maximum(eigvals, 1e-10)
        probs = eigvals / eigvals.sum()
        all_sample_eff_rank.append(float(np.exp(-(probs * np.log(probs)).sum())))

        det_outputs = model.rollout(history_01, future_01=None, temperature=0.0, hard_regimes=True, use_posterior=False)
        det_next = model.support_transform.inverse(det_outputs.mu_u[:, 0])[0].reshape(history_01.shape[0], 5, 5)
        prev_chunks.append(history_01[:, -1].detach().cpu())
        gt_next_chunks.append(future_01[:, 0].detach().cpu())
        det_next_chunks.append(det_next.detach().cpu())
        sample_next_chunks.append(samples[:, :, 0].mean(dim=1).detach().cpu())

        h30_cov = coverage[:, -1].reshape(history_01.shape[0], -1)
        h30_worst_covs.append(h30_cov.min(dim=1).values.cpu())
        h30_best_covs.append(h30_cov.max(dim=1).values.cpu())
        window_floor_bad.append((window_cov < 0.5).float().cpu())

        total_cov += coverage.mean().item() * history_01.shape[0]
        total_width += width.mean().item() * history_01.shape[0]
        total_mae += mae.mean().item() * history_01.shape[0]
        total_support_viol += support_viol.item() * history_01.shape[0]
        total_count += history_01.shape[0]

        all_vov.append(vov.detach().cpu())
        all_window_widths.append(window_width.detach().cpu())

    if total_count == 0:
        return {
            "joint_cov90": float("nan"),
            "joint_width90": float("nan"),
            "joint_mae": float("nan"),
            "joint_support_violation_rate": float("nan"),
            "joint_turb_calm_ratio": float("nan"),
            "joint_sample_eff_rank": float("nan"),
            "joint_det_mr_ratio": float("nan"),
            "joint_sample_mr_ratio": float("nan"),
            "joint_h30_worst_cov90": float("nan"),
            "joint_h30_best_cov90": float("nan"),
            "joint_window_floor_bad_rate": float("nan"),
        }

    vov = torch.cat(all_vov)
    widths = torch.cat(all_window_widths)
    q20 = torch.quantile(vov, 0.2)
    q80 = torch.quantile(vov, 0.8)
    calm_mask = vov <= q20
    turb_mask = vov >= q80
    turb_calm_ratio = (
        (widths[turb_mask].mean() / widths[calm_mask].mean()).item()
        if calm_mask.any() and turb_mask.any()
        else float("nan")
    )

    prev = torch.cat(prev_chunks, dim=0)
    gt_next = torch.cat(gt_next_chunks, dim=0)
    det_next = torch.cat(det_next_chunks, dim=0)
    sample_next = torch.cat(sample_next_chunks, dim=0)

    return {
        "joint_cov90": total_cov / total_count,
        "joint_width90": total_width / total_count,
        "joint_mae": total_mae / total_count,
        "joint_support_violation_rate": total_support_viol / total_count,
        "joint_turb_calm_ratio": turb_calm_ratio,
        "joint_sample_eff_rank": float(np.mean(all_sample_eff_rank)),
        "joint_det_mr_ratio": aggregate_slope_ratio(prev, gt_next, det_next),
        "joint_sample_mr_ratio": aggregate_slope_ratio(prev, gt_next, sample_next),
        "joint_h30_worst_cov90": torch.cat(h30_worst_covs).mean().item(),
        "joint_h30_best_cov90": torch.cat(h30_best_covs).mean().item(),
        "joint_window_floor_bad_rate": torch.cat(window_floor_bad).mean().item(),
    }


def _finite_or_inf(x: float) -> float:
    return x if np.isfinite(x) else float("inf")


def selection_key(val_metrics: dict, joint_metrics: dict) -> tuple[float, float, float, float]:
    mr_pen = _finite_or_inf(abs(joint_metrics["joint_det_mr_ratio"] - 1.0) + abs(joint_metrics["joint_sample_mr_ratio"] - 1.0))
    cov_pen = _finite_or_inf(
        abs(joint_metrics["joint_cov90"] - 0.90)
        + max(0.70 - joint_metrics["joint_h30_worst_cov90"], 0.0)
        + max(joint_metrics["joint_h30_best_cov90"] - 0.95, 0.0)
        + joint_metrics["joint_window_floor_bad_rate"]
    )
    tc_pen = _finite_or_inf(max(1.15 - joint_metrics["joint_turb_calm_ratio"], 0.0))
    nll_pen = _finite_or_inf(val_metrics["val_joint_nll"])
    return (mr_pen, cov_pen, tc_pen, nll_pen)


def main():
    parser = argparse.ArgumentParser(description="178a: regime-coupled state-space generator")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_model", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_model", type=float, default=0.01)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--n_blocks", type=int, default=5)
    parser.add_argument("--n_regimes", type=int, default=3)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--regime_dim", type=int, default=16)
    parser.add_argument("--time_emb_dim", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--cell_rank", type=int, default=5)
    parser.add_argument("--cov_resid_rank", type=int, default=3)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--local_scale_clip", type=float, default=0.35)
    parser.add_argument("--base_nu", type=float, default=8.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-5)
    parser.add_argument("--support_kind", type=str, default="bounded_logit")
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--beta_regime", type=float, default=0.05)
    parser.add_argument("--lambda_usage", type=float, default=2.0)
    parser.add_argument("--lambda_cov", type=float, default=0.5)
    parser.add_argument("--lambda_mr", type=float, default=0.10)
    parser.add_argument("--train_regime_temperature", type=float, default=0.5)
    parser.add_argument("--joint_select_samples", type=int, default=8)
    parser.add_argument("--joint_select_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_select_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    hist_len = args.history_len
    future_len = args.future_len

    test_start = 4511
    max_train_idx = test_start - hist_len - future_len
    total_holdout = 441
    selection_size = total_holdout // 2
    monitor_size = total_holdout - selection_size
    train_indices = np.arange(0, max_train_idx - total_holdout)
    select_indices = np.arange(max_train_idx - total_holdout, max_train_idx - monitor_size)
    val_indices = np.arange(max_train_idx - monitor_size, max_train_idx)

    if args.max_train_windows is not None:
        train_indices = train_indices[: args.max_train_windows]
    if args.max_select_windows is not None:
        select_indices = select_indices[: args.max_select_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    print(f"Train: {len(train_indices)}, Select: {len(select_indices)}, Val: {len(val_indices)}")

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, hist_len, future_len)
    select_hist, select_future = build_multistep_windows(select_indices, surf_tensor, hist_len, future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, hist_len, future_len)

    train_loader = DataLoader(TensorDataset(train_hist, train_future), batch_size=args.batch_size, shuffle=True, drop_last=True)
    select_loader = DataLoader(TensorDataset(select_hist, select_future), batch_size=args.batch_size, shuffle=False)
    val_loader = DataLoader(TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False)

    encoder_config = EncoderConfig(
        input_dim=25,
        gru_hidden_dim=64,
        bottleneck_dim=128,
        dropout=0.1,
        cond_aug_sigma=0.0,
    )
    support_transform = build_support_transform(
        TransformConfig(
            kind=args.support_kind,
            lo=args.support_lo,
            hi=args.support_hi,
            eps=args.support_eps,
        )
    )
    model = RegimeCoupledStateSpaceModel(
        encoder_config=encoder_config,
        support_transform=support_transform,
        future_len=future_len,
        n_cells=25,
        n_blocks=args.n_blocks,
        n_regimes=args.n_regimes,
        latent_dim=args.latent_dim,
        regime_dim=args.regime_dim,
        time_emb_dim=args.time_emb_dim,
        cell_rank=args.cell_rank,
        cov_resid_rank=args.cov_resid_rank,
        hidden_dim=args.hidden_dim,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        local_scale_clip=args.local_scale_clip,
        base_nu=args.base_nu,
        cov_jitter=args.cov_jitter,
    ).to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_model = sum(p.numel() for n, p in model.named_parameters() if not n.startswith("encoder."))
    print(f"\n{'=' * 64}")
    print("178a: regime-coupled state-space generator")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Core params:    {n_model:,}")
    print(f"  Total params:   {n_enc + n_model:,}")
    print(f"  Blocks={args.n_blocks} | regimes={args.n_regimes} | latent_dim={args.latent_dim}")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr_encoder, "weight_decay": args.weight_decay_encoder},
            {
                "params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")],
                "lr": args.lr_model,
                "weight_decay": args.weight_decay_model,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_key = None
    best_metrics = None
    history = []
    history_path = Path(args.output_dir) / "training_history.json"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        totals = {
            "train_total_loss": 0.0,
            "train_joint_nll": 0.0,
            "train_kl_regime": 0.0,
            "train_usage_penalty": 0.0,
            "train_cov_penalty": 0.0,
            "train_mr_penalty": 0.0,
            "train_joint_mae": 0.0,
            "train_joint_det_mr_ratio": 0.0,
            "train_regime_usage_max": 0.0,
            "train_prior_entropy": 0.0,
            "train_posterior_entropy": 0.0,
            "train_kappa_mean": 0.0,
            "train_local_scale_min": 0.0,
            "train_local_scale_max": 0.0,
            "train_cell_eff_rank": 0.0,
            "train_white_std_mean": 0.0,
        }
        nb = 0

        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = joint_loss(
                model,
                history_01,
                future_01,
                beta_regime=args.beta_regime,
                lambda_usage=args.lambda_usage,
                lambda_cov=args.lambda_cov,
                lambda_mr=args.lambda_mr,
                regime_temperature=args.train_regime_temperature,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for key in totals:
                metric_key = key.replace("train_", "")
                totals[key] += float(metrics[metric_key].item())
            nb += 1

        scheduler.step()
        train_metrics = {k: v / max(nb, 1) for k, v in totals.items()}
        val_metrics = evaluate_teacher_forced(
            model,
            val_loader,
            beta_regime=args.beta_regime,
            lambda_usage=args.lambda_usage,
            lambda_cov=args.lambda_cov,
            lambda_mr=args.lambda_mr,
            regime_temperature=0.0,
        )
        select_metrics = evaluate_joint_subset(
            model,
            select_loader,
            joint_samples=args.joint_select_samples,
            eval_limit=args.joint_select_limit,
        )
        key = selection_key(val_metrics, select_metrics)
        is_best = best_key is None or key < best_key
        if is_best:
            best_key = key
            best_metrics = {**val_metrics, **select_metrics}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_key": best_key,
                    "config": {
                        "type": "regime_coupled_state_space_student_t_178a",
                        "encoder": vars(encoder_config),
                        "model": {
                            "future_len": future_len,
                            "n_cells": 25,
                            "n_blocks": args.n_blocks,
                            "n_regimes": args.n_regimes,
                            "latent_dim": args.latent_dim,
                            "regime_dim": args.regime_dim,
                            "time_emb_dim": args.time_emb_dim,
                            "cell_rank": args.cell_rank,
                            "cov_resid_rank": args.cov_resid_rank,
                            "hidden_dim": args.hidden_dim,
                            "diag_floor": args.diag_floor,
                            "scale_floor": args.scale_floor,
                            "local_scale_clip": args.local_scale_clip,
                            "cov_jitter": args.cov_jitter,
                        },
                        "support_transform": {
                            "kind": args.support_kind,
                            "lo": args.support_lo,
                            "hi": args.support_hi,
                            "eps": args.support_eps,
                        },
                        "base_nu": args.base_nu,
                        "history_len": hist_len,
                        "future_len": future_len,
                        "train_windows": len(train_indices),
                        "select_windows": len(select_indices),
                        "val_windows": len(val_indices),
                        "beta_regime": args.beta_regime,
                        "lambda_usage": args.lambda_usage,
                        "lambda_cov": args.lambda_cov,
                        "lambda_mr": args.lambda_mr,
                    },
                    "best_metrics": best_metrics,
                },
                f"{args.output_dir}/best_model.pt",
            )

        row = {"epoch": epoch, **train_metrics, **val_metrics, **select_metrics}
        history.append(row)
        history_path.write_text(json.dumps(make_serializable(history), indent=2))

        elapsed = time.time() - t0
        print(
            f"Ep {epoch:3d}  "
            f"train_loss={train_metrics['train_total_loss']:.4f}  "
            f"val_nll={val_metrics['val_joint_nll']:.4f}  "
            f"sel_cov90={select_metrics['joint_cov90']:.3f}  "
            f"sel_tc={select_metrics['joint_turb_calm_ratio']:.3f}  "
            f"sel_mr_det={select_metrics['joint_det_mr_ratio']:.3f}  "
            f"sel_mr_samp={select_metrics['joint_sample_mr_ratio']:.3f}  "
            f"h30=[{select_metrics['joint_h30_worst_cov90']:.3f},{select_metrics['joint_h30_best_cov90']:.3f}]  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "selection_key": best_key,
        "config": {
            "type": "regime_coupled_state_space_student_t_178a",
            "encoder": vars(encoder_config),
            "model": {
                "future_len": future_len,
                "n_cells": 25,
                "n_blocks": args.n_blocks,
                "n_regimes": args.n_regimes,
                "latent_dim": args.latent_dim,
                "regime_dim": args.regime_dim,
                "time_emb_dim": args.time_emb_dim,
                "cell_rank": args.cell_rank,
                "cov_resid_rank": args.cov_resid_rank,
                "hidden_dim": args.hidden_dim,
                "diag_floor": args.diag_floor,
                "scale_floor": args.scale_floor,
                "local_scale_clip": args.local_scale_clip,
                "cov_jitter": args.cov_jitter,
            },
            "support_transform": {
                "kind": args.support_kind,
                "lo": args.support_lo,
                "hi": args.support_hi,
                "eps": args.support_eps,
            },
            "base_nu": args.base_nu,
            "history_len": hist_len,
            "future_len": future_len,
            "train_windows": len(train_indices),
            "select_windows": len(select_indices),
            "val_windows": len(val_indices),
            "beta_regime": args.beta_regime,
            "lambda_usage": args.lambda_usage,
            "lambda_cov": args.lambda_cov,
            "lambda_mr": args.lambda_mr,
        },
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    history_path.write_text(json.dumps(make_serializable(history), indent=2))

    if best_metrics is not None:
        print(f"\nBest selection key: {best_key}")
        print(
            "Best diagnostics: "
            f"cov90={best_metrics['joint_cov90']:.3f}, "
            f"tc={best_metrics['joint_turb_calm_ratio']:.3f}, "
            f"mr_det={best_metrics['joint_det_mr_ratio']:.3f}, "
            f"mr_samp={best_metrics['joint_sample_mr_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
