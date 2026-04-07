#!/usr/bin/env python
"""
175a: keep the 173a architecture fixed, change only the objective.

Add a weak teacher-forced local calibration regularizer on top of the
173a realism-first backbone:
  - exact joint NLL remains the main loss
  - local variance shrinkage remains
  - new penalty encourages transformed residuals to have the right
    second moment and 90% exceedance rate by:
      * regime (calm / turbulent)
      * horizon bucket (early / mid / late)
      * cell
"""

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
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    iv_to_unconstrained,
    make_serializable,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_173a_local_var_residual_flow_structured_joint_student_t import (
    LocalVarianceResidualFlowStructuredJointStudentTModel,
    evaluate_joint_subset,
)


BASE_TAIL_Q_90_CENTRAL_DF8 = 1.8595480375228424


def _history_vol_of_vol(history_01: torch.Tensor) -> torch.Tensor:
    mean_iv = history_01.mean(dim=(-1, -2))
    daily = mean_iv[:, 1:] - mean_iv[:, :-1]
    return daily.std(dim=1)


def local_calibration_penalty(
    z_tensor: torch.Tensor,
    history_01: torch.Tensor,
    base_nu: float,
    regime_quantile: float,
    smooth_beta: float,
    var_weight: float,
    exceed_weight: float,
    z_clip: float,
) -> tuple[torch.Tensor, dict]:
    """Small regime x horizon-group x cell calibration penalty in z-space.

    z_tensor is the transformed residual after the conditional flow. If the
    model is locally calibrated, each dimension should match the iid Student-t
    base law regardless of regime / horizon / cell.
    """
    batch, future_len, n_cells = z_tensor.shape
    device = z_tensor.device
    dtype = z_tensor.dtype

    vov = _history_vol_of_vol(history_01).detach()
    q_lo = torch.quantile(vov, regime_quantile)
    q_hi = torch.quantile(vov, 1.0 - regime_quantile)
    masks = {
        "calm": vov <= q_lo,
        "turb": vov >= q_hi,
    }
    groups = (
        ("early", 0, min(6, future_len)),
        ("mid", min(6, future_len), min(15, future_len)),
        ("late", min(15, future_len), future_len),
    )

    target_second = z_tensor.new_tensor(base_nu / (base_nu - 2.0))
    target_std = target_second.sqrt()
    target_exceed = z_tensor.new_tensor(0.10)
    tail_q = z_tensor.new_tensor(BASE_TAIL_Q_90_CENTRAL_DF8)

    terms = []
    var_terms = []
    exceed_terms = []
    active_slices = 0

    for mask in masks.values():
        if mask.sum().item() < 2:
            continue
        z_reg = z_tensor[mask]
        for _, start, end in groups:
            if end <= start:
                continue
            z_slice = z_reg[:, start:end, :].reshape(-1, n_cells)
            if z_slice.shape[0] < 8:
                continue

            z_clip_slice = z_slice.clamp(min=-z_clip, max=z_clip)
            second = z_clip_slice.pow(2).mean(dim=0)
            smooth_exceed = torch.sigmoid(smooth_beta * (z_clip_slice.abs() - tail_q)).mean(dim=0)

            var_pen = (second.sqrt() - target_std).pow(2).mean()
            exceed_pen = (smooth_exceed - target_exceed).pow(2).mean()
            term = var_weight * var_pen + exceed_weight * exceed_pen

            terms.append(term)
            var_terms.append(var_pen)
            exceed_terms.append(exceed_pen)
            active_slices += 1

    if terms:
        total = torch.stack(terms).mean()
        var_mean = torch.stack(var_terms).mean()
        exceed_mean = torch.stack(exceed_terms).mean()
    else:
        total = torch.zeros((), device=device, dtype=dtype)
        var_mean = torch.zeros((), device=device, dtype=dtype)
        exceed_mean = torch.zeros((), device=device, dtype=dtype)

    with torch.no_grad():
        q_lo_std = torch.quantile(vov, regime_quantile)
        q_hi_std = torch.quantile(vov, 1.0 - regime_quantile)
        calm_mask = vov <= q_lo_std
        turb_mask = vov >= q_hi_std
        calm_z_std = z_tensor[calm_mask].std() if calm_mask.any() else torch.zeros((), device=device, dtype=dtype)
        turb_z_std = z_tensor[turb_mask].std() if turb_mask.any() else torch.zeros((), device=device, dtype=dtype)

    return total, {
        "calib_penalty": total,
        "calib_var_penalty": var_mean,
        "calib_exceed_penalty": exceed_mean,
        "calib_active_slices": torch.tensor(float(active_slices), device=device, dtype=dtype),
        "calib_calm_z_std": calm_z_std,
        "calib_turb_z_std": turb_z_std,
    }


def joint_nll_loss(
    model: LocalVarianceResidualFlowStructuredJointStudentTModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    local_var_penalty: float,
    calibration_penalty: float,
    calibration_regime_quantile: float,
    calibration_smooth_beta: float,
    calibration_var_weight: float,
    calibration_exceed_weight: float,
    calibration_z_clip: float,
) -> tuple[torch.Tensor, dict]:
    target_u = iv_to_unconstrained(
        future_01,
        lo=model.support_lo,
        hi=model.support_hi,
        eps=model.support_eps,
    )
    (
        mu,
        time_factor,
        time_diag,
        cell_factor,
        cell_diag,
        scale,
        flow_context,
        local_delta,
    ) = model.forward_from_history(history_01)

    batch, n_frames, n_cells = target_u.shape
    cov_t, cov_c = model.covariance_parts(time_factor, time_diag, cell_factor, cell_diag, scale)
    chol_t = torch.linalg.cholesky(cov_t)
    chol_c = torch.linalg.cholesky(cov_c)

    local_scale = torch.exp(0.5 * local_delta)
    diff = (target_u - mu) / local_scale
    white_t = torch.linalg.solve_triangular(chol_t, diff, upper=False)
    white = torch.linalg.solve_triangular(chol_c, white_t.transpose(1, 2), upper=False).transpose(1, 2)
    white_flat = white.reshape(batch, n_frames * n_cells)

    z_flat, flow_logdet = model.flow(white_flat, flow_context)
    z_tensor = z_flat.view(batch, n_frames, n_cells)

    logdet_t = 2.0 * torch.log(torch.diagonal(chol_t, dim1=-2, dim2=-1)).sum(dim=-1)
    logdet_c = 2.0 * torch.log(torch.diagonal(chol_c, dim1=-2, dim2=-1)).sum(dim=-1)
    logdet_cov = n_cells * logdet_t + n_frames * logdet_c
    logdet_local = 2.0 * torch.log(local_scale).sum(dim=(1, 2))

    base_logprob = model._base_logprob(z_flat)
    logprob = base_logprob + flow_logdet - 0.5 * (logdet_cov + logdet_local)

    nll = (-logprob).mean()
    local_pen = local_delta.pow(2).mean()
    calib_pen, calib_metrics = local_calibration_penalty(
        z_tensor=z_tensor,
        history_01=history_01,
        base_nu=model.base_nu,
        regime_quantile=calibration_regime_quantile,
        smooth_beta=calibration_smooth_beta,
        var_weight=calibration_var_weight,
        exceed_weight=calibration_exceed_weight,
        z_clip=calibration_z_clip,
    )

    loss = nll + local_var_penalty * local_pen + calibration_penalty * calib_pen
    metrics = {
        "joint_nll": nll,
        "total_loss": loss,
        "local_var_penalty": local_pen,
        "calibration_penalty": calib_pen,
        "calibration_var_penalty": calib_metrics["calib_var_penalty"],
        "calibration_exceed_penalty": calib_metrics["calib_exceed_penalty"],
        "calibration_active_slices": calib_metrics["calib_active_slices"],
        "calm_z_std": calib_metrics["calib_calm_z_std"],
        "turb_z_std": calib_metrics["calib_turb_z_std"],
        "joint_mae": (mu - target_u).abs().mean(),
        "time_eff_rank": torch.linalg.matrix_rank(cov_t).float().mean(),
        "cell_eff_rank": torch.linalg.matrix_rank(cov_c).float().mean(),
        "scale_mean": scale.mean(),
        "flow_logdet_mean": flow_logdet.mean(),
        "white_std_mean": white_flat.std(dim=-1).mean(),
        "z_std_mean": z_flat.std(dim=-1).mean(),
        "local_delta_rms": local_delta.pow(2).mean(dim=(1, 2)).sqrt().mean(),
        "local_scale_min": local_scale.amin(dim=(1, 2)).mean(),
        "local_scale_max": local_scale.amax(dim=(1, 2)).mean(),
    }
    return loss, metrics


@torch.no_grad()
def evaluate_teacher_forced(
    model: LocalVarianceResidualFlowStructuredJointStudentTModel,
    val_loader: DataLoader,
    local_var_penalty: float,
    calibration_penalty: float,
    calibration_regime_quantile: float,
    calibration_smooth_beta: float,
    calibration_var_weight: float,
    calibration_exceed_weight: float,
    calibration_z_clip: float,
) -> dict:
    model.eval()
    totals = {
        "val_total_loss": 0.0,
        "val_joint_nll": 0.0,
        "val_local_var_penalty": 0.0,
        "val_calibration_penalty": 0.0,
        "val_calibration_var_penalty": 0.0,
        "val_calibration_exceed_penalty": 0.0,
        "val_calibration_active_slices": 0.0,
        "val_joint_mae": 0.0,
        "val_time_eff_rank": 0.0,
        "val_cell_eff_rank": 0.0,
        "val_scale_mean": 0.0,
        "val_flow_logdet_mean": 0.0,
        "val_white_std_mean": 0.0,
        "val_z_std_mean": 0.0,
        "val_calm_z_std": 0.0,
        "val_turb_z_std": 0.0,
        "val_local_delta_rms": 0.0,
        "val_local_scale_min": 0.0,
        "val_local_scale_max": 0.0,
    }
    total_count = 0

    for history_01, future_01 in val_loader:
        _, metrics = joint_nll_loss(
            model,
            history_01,
            future_01,
            local_var_penalty=local_var_penalty,
            calibration_penalty=calibration_penalty,
            calibration_regime_quantile=calibration_regime_quantile,
            calibration_smooth_beta=calibration_smooth_beta,
            calibration_var_weight=calibration_var_weight,
            calibration_exceed_weight=calibration_exceed_weight,
            calibration_z_clip=calibration_z_clip,
        )
        batch_size = history_01.shape[0]
        totals["val_total_loss"] += metrics["total_loss"].item() * batch_size
        totals["val_joint_nll"] += metrics["joint_nll"].item() * batch_size
        totals["val_local_var_penalty"] += metrics["local_var_penalty"].item() * batch_size
        totals["val_calibration_penalty"] += metrics["calibration_penalty"].item() * batch_size
        totals["val_calibration_var_penalty"] += metrics["calibration_var_penalty"].item() * batch_size
        totals["val_calibration_exceed_penalty"] += metrics["calibration_exceed_penalty"].item() * batch_size
        totals["val_calibration_active_slices"] += metrics["calibration_active_slices"].item() * batch_size
        totals["val_joint_mae"] += metrics["joint_mae"].item() * batch_size
        totals["val_time_eff_rank"] += metrics["time_eff_rank"].item() * batch_size
        totals["val_cell_eff_rank"] += metrics["cell_eff_rank"].item() * batch_size
        totals["val_scale_mean"] += metrics["scale_mean"].item() * batch_size
        totals["val_flow_logdet_mean"] += metrics["flow_logdet_mean"].item() * batch_size
        totals["val_white_std_mean"] += metrics["white_std_mean"].item() * batch_size
        totals["val_z_std_mean"] += metrics["z_std_mean"].item() * batch_size
        totals["val_calm_z_std"] += metrics["calm_z_std"].item() * batch_size
        totals["val_turb_z_std"] += metrics["turb_z_std"].item() * batch_size
        totals["val_local_delta_rms"] += metrics["local_delta_rms"].item() * batch_size
        totals["val_local_scale_min"] += metrics["local_scale_min"].item() * batch_size
        totals["val_local_scale_max"] += metrics["local_scale_max"].item() * batch_size
        total_count += batch_size

    return {k: (float("nan") if total_count == 0 else v / total_count) for k, v in totals.items()}


def main():
    parser = argparse.ArgumentParser(description="175a: 173a + weak local calibration regularizer")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--lr_flow", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_decoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_flow", type=float, default=0.0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--time_rank", type=int, default=6)
    parser.add_argument("--cell_rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.05)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--flow_context_dim", type=int, default=256)
    parser.add_argument("--flow_hidden_dim", type=int, default=256)
    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--flow_scale_clip", type=float, default=2.0)
    parser.add_argument("--base_nu", type=float, default=8.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-5)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--joint_val_samples", type=int, default=8)
    parser.add_argument("--joint_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--local_delta_clip", type=float, default=0.35)
    parser.add_argument("--local_var_penalty", type=float, default=1.0)
    parser.add_argument("--calibration_penalty", type=float, default=1.0)
    parser.add_argument("--calibration_regime_quantile", type=float, default=0.3)
    parser.add_argument("--calibration_smooth_beta", type=float, default=6.0)
    parser.add_argument("--calibration_var_weight", type=float, default=1.0)
    parser.add_argument("--calibration_exceed_weight", type=float, default=1.0)
    parser.add_argument("--calibration_z_clip", type=float, default=4.0)
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
    val_size = 441
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)

    if args.max_train_windows is not None:
        train_indices = train_indices[: args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, hist_len, future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, hist_len, future_len)

    train_loader = DataLoader(TensorDataset(train_hist, train_future), batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False)

    encoder_config = EncoderConfig(
        input_dim=25,
        gru_hidden_dim=64,
        bottleneck_dim=128,
        dropout=0.1,
        cond_aug_sigma=0.0,
    )
    decoder_config = dict(
        n_frames=future_len,
        n_cells=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        cond_dim=128,
        time_rank=args.time_rank,
        cell_rank=args.cell_rank,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        flow_context_dim=args.flow_context_dim,
        local_delta_clip=args.local_delta_clip,
    )
    flow_config = dict(
        dim=future_len * 25,
        context_dim=args.flow_context_dim,
        hidden_dim=args.flow_hidden_dim,
        n_layers=args.flow_layers,
        scale_clip=args.flow_scale_clip,
    )

    model = LocalVarianceResidualFlowStructuredJointStudentTModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
        base_nu=args.base_nu,
    ).to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    n_flow = sum(p.numel() for p in model.flow.parameters())
    print(f"\n{'=' * 64}")
    print("175a: 173a + weak teacher-forced local calibration regularizer")
    print(f"{'=' * 64}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Flow params:    {n_flow:,}")
    print(f"  Total params:   {n_enc + n_dec + n_flow:,}")
    print(f"  History len: {hist_len} | future len: {future_len}")
    print(f"  Time rank={args.time_rank} | Cell rank={args.cell_rank}")
    print(f"  Local penalty={args.local_var_penalty} | calib penalty={args.calibration_penalty}")
    print(f"  Calib regime quantile={args.calibration_regime_quantile} | smooth beta={args.calibration_smooth_beta}")
    print("  Objective: exact joint likelihood + local variance shrinkage + weak local calibration")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr_encoder, "weight_decay": args.weight_decay_encoder},
            {"params": model.decoder.parameters(), "lr": args.lr_decoder, "weight_decay": args.weight_decay_decoder},
            {"params": model.flow.parameters(), "lr": args.lr_flow, "weight_decay": args.weight_decay_flow},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val = float("inf")
    best_metrics = None
    history = []
    history_path = Path(args.output_dir) / "training_history.json"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        totals = {
            "train_total_loss": 0.0,
            "train_joint_nll": 0.0,
            "train_local_var_penalty": 0.0,
            "train_calibration_penalty": 0.0,
            "train_calibration_var_penalty": 0.0,
            "train_calibration_exceed_penalty": 0.0,
            "train_calibration_active_slices": 0.0,
            "train_joint_mae": 0.0,
            "train_time_eff_rank": 0.0,
            "train_cell_eff_rank": 0.0,
            "train_scale_mean": 0.0,
            "train_flow_logdet_mean": 0.0,
            "train_white_std_mean": 0.0,
            "train_z_std_mean": 0.0,
            "train_calm_z_std": 0.0,
            "train_turb_z_std": 0.0,
            "train_local_delta_rms": 0.0,
            "train_local_scale_min": 0.0,
            "train_local_scale_max": 0.0,
        }
        nb = 0

        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = joint_nll_loss(
                model,
                history_01,
                future_01,
                local_var_penalty=args.local_var_penalty,
                calibration_penalty=args.calibration_penalty,
                calibration_regime_quantile=args.calibration_regime_quantile,
                calibration_smooth_beta=args.calibration_smooth_beta,
                calibration_var_weight=args.calibration_var_weight,
                calibration_exceed_weight=args.calibration_exceed_weight,
                calibration_z_clip=args.calibration_z_clip,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for key, value in metrics.items():
                totals[f"train_{key}"] += value.item()
            nb += 1

        scheduler.step()
        train_metrics = {k: v / max(nb, 1) for k, v in totals.items()}
        val_metrics = evaluate_teacher_forced(
            model,
            val_loader,
            local_var_penalty=args.local_var_penalty,
            calibration_penalty=args.calibration_penalty,
            calibration_regime_quantile=args.calibration_regime_quantile,
            calibration_smooth_beta=args.calibration_smooth_beta,
            calibration_var_weight=args.calibration_var_weight,
            calibration_exceed_weight=args.calibration_exceed_weight,
            calibration_z_clip=args.calibration_z_clip,
        )
        joint_metrics = evaluate_joint_subset(
            model,
            val_loader,
            joint_val_samples=args.joint_val_samples,
            eval_limit=args.joint_eval_limit,
        )

        elapsed = time.time() - t0
        is_best = val_metrics["val_total_loss"] < best_val
        if is_best:
            best_val = val_metrics["val_total_loss"]
            best_metrics = {**val_metrics, **joint_metrics}
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_total_loss": best_val,
                    "config": {
                        "type": "local_var_residual_flow_structured_joint_student_t_175a",
                        "encoder": vars(encoder_config),
                        "decoder": decoder_config,
                        "flow": flow_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "base_nu": args.base_nu,
                        "history_len": hist_len,
                        "future_len": future_len,
                        "train_windows": len(train_indices),
                        "val_windows": len(val_indices),
                        "local_var_penalty": args.local_var_penalty,
                        "calibration_penalty": args.calibration_penalty,
                        "calibration_regime_quantile": args.calibration_regime_quantile,
                        "calibration_smooth_beta": args.calibration_smooth_beta,
                        "calibration_var_weight": args.calibration_var_weight,
                        "calibration_exceed_weight": args.calibration_exceed_weight,
                        "calibration_z_clip": args.calibration_z_clip,
                    },
                    "best_metrics": best_metrics,
                },
                f"{args.output_dir}/best_model.pt",
            )

        row = {"epoch": epoch, **train_metrics, **val_metrics, **joint_metrics}
        history.append(row)
        history_path.write_text(json.dumps(make_serializable(history), indent=2))

        print(
            f"Ep {epoch:3d}  "
            f"train_loss={train_metrics['train_total_loss']:.4f}  "
            f"val_loss={val_metrics['val_total_loss']:.4f}  "
            f"val_nll={val_metrics['val_joint_nll']:.4f}  "
            f"cal={val_metrics['val_calibration_penalty']:.4f}  "
            f"cov90={joint_metrics['joint_cov90']:.4f}  "
            f"tc={joint_metrics['joint_turb_calm_ratio']:.3f}  "
            f"zstd(c/t)={val_metrics['val_calm_z_std']:.3f}/{val_metrics['val_turb_z_std']:.3f}  "
            f"viol={joint_metrics['joint_support_violation_rate']:.4e}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    final_payload = {
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs,
        "val_total_loss": history[-1]["val_total_loss"] if history else float("nan"),
        "config": {
            "type": "local_var_residual_flow_structured_joint_student_t_175a",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "base_nu": args.base_nu,
            "history_len": hist_len,
            "future_len": future_len,
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "local_var_penalty": args.local_var_penalty,
            "calibration_penalty": args.calibration_penalty,
            "calibration_regime_quantile": args.calibration_regime_quantile,
            "calibration_smooth_beta": args.calibration_smooth_beta,
            "calibration_var_weight": args.calibration_var_weight,
            "calibration_exceed_weight": args.calibration_exceed_weight,
            "calibration_z_clip": args.calibration_z_clip,
        },
        "best_val_total_loss": best_val,
        "best_metrics": best_metrics,
    }
    torch.save(final_payload, f"{args.output_dir}/final_model.pt")
    history_path.write_text(json.dumps(make_serializable(history), indent=2))

    print(f"\nBest val total loss: {best_val:.4f}")
    if best_metrics is not None:
        print(
            "Best diagnostics: "
            f"joint_cov90={best_metrics['joint_cov90']:.4f}, "
            f"joint_turb_calm_ratio={best_metrics['joint_turb_calm_ratio']:.3f}, "
            f"joint_support_violation_rate={best_metrics['joint_support_violation_rate']:.4e}"
        )


if __name__ == "__main__":
    main()
