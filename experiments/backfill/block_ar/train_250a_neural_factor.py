#!/usr/bin/env python
"""
250a: Stage A — minimal non-AR neural factor generator, trained from scratch.

Fresh minimal trainer. Does NOT clone 241b's 183c-specific plumbing.
Reuses only: afcrps_loss / variogram_score / energy_score from single_pass_ar, and
the build_multistep_windows helper.

Loss = lambda_cell * afCRPS_frame_sum  +  lambda_vs * VS(p=0.5)  +  lambda_es * ES_25d
With --lambda_vs auto, lambda_vs is calibrated from mean_ES / mean_VS over a single
warmup epoch (dualGNN-style scale matching).

Inputs: raw [0,1] IV surfaces. Outputs: raw [0,1] IV surfaces. No normalize_iv wrap.
Stack-agnostic: D=n_cells is whatever the caller configures; 5x5 reshape happens only
where the loss helpers expect (B,K,T,H,W) shape.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.neural_factor import NeuralFactorConfig, NeuralFactorModel
from diffusion.block_ar.single_pass_ar import (
    afcrps_loss, energy_score, normalize_iv, variogram_score,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
)


def build_losses(
    samples_btd: torch.Tensor,
    gt_btd: torch.Tensor,
    lambda_cell: float,
    lambda_vs: float,
    lambda_es: float,
    H: int,
    W: int,
    lambda_pmax: float = 0.0,
    lambda_chg: float = 0.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """samples_btd: (B, K, T, D). gt_btd: (B, T, D). Reshape once to (B,K,T,H,W) for loss helpers.

    Optional tail-attack terms:
      lambda_pmax: afCRPS on pathwise max-|Δ| per cell. Directly attacks under-generation of extreme jumps.
      lambda_chg:  afCRPS on per-cell daily changes (frame_sum). Attacks chg_KS and kurtosis.
    """
    B, K, T, D = samples_btd.shape
    assert D == H * W, f"loss reshape requires D=H*W, got D={D}, H*W={H*W}"
    samples_grid = samples_btd.view(B, K, T, H, W)
    gt_grid = gt_btd.view(B, T, H, W)

    cell_crps, _mae, _spread = afcrps_loss(
        samples_grid, gt_grid, alpha=0.95, reduction="frame_sum"
    )
    vs = variogram_score(samples_grid, gt_grid, p=0.5)
    es = energy_score(samples_grid, gt_grid)
    total = lambda_cell * cell_crps + lambda_vs * vs + lambda_es * es
    metrics = {
        "cell_crps": cell_crps.detach(),
        "variogram_score": vs.detach(),
        "energy_score": es.detach(),
    }

    if lambda_pmax > 0.0 and T >= 2:
        # Pathwise max-|Δ| per cell: reduces (B,K,T,H,W) -> (B,K,1,H,W), (B,T,H,W) -> (B,1,H,W)
        s_chg = samples_grid[:, :, 1:] - samples_grid[:, :, :-1]       # (B, K, T-1, H, W)
        g_chg = gt_grid[:, 1:] - gt_grid[:, :-1]                        # (B, T-1, H, W)
        s_pmax = s_chg.abs().amax(dim=2, keepdim=True)                  # (B, K, 1, H, W)
        g_pmax = g_chg.abs().amax(dim=1, keepdim=True)                  # (B, 1, H, W)
        pmax_crps, _, _ = afcrps_loss(
            s_pmax, g_pmax, alpha=0.95, reduction="frame_sum"
        )
        total = total + lambda_pmax * pmax_crps
        metrics["pmax_crps"] = pmax_crps.detach()

    if lambda_chg > 0.0 and T >= 2:
        s_chg = samples_grid[:, :, 1:] - samples_grid[:, :, :-1]       # (B, K, T-1, H, W)
        g_chg = gt_grid[:, 1:] - gt_grid[:, :-1]                        # (B, T-1, H, W)
        chg_crps, _, _ = afcrps_loss(
            s_chg, g_chg, alpha=0.95, reduction="frame_sum"
        )
        total = total + lambda_chg * chg_crps
        metrics["chg_crps"] = chg_crps.detach()

    metrics["total"] = total.detach()
    return total, metrics


def estimate_lambda_vs_auto(
    model: NeuralFactorModel,
    loader: DataLoader,
    device: str,
    K: int,
    H: int,
    W: int,
    max_batches: int = 40,
) -> float:
    """Compute lambda_vs = mean_ES / mean_VS over one warmup pass for scale matching."""
    model.eval()
    es_sum = 0.0
    vs_sum = 0.0
    n = 0
    with torch.no_grad():
        for i, (hist_01, fut_flat) in enumerate(loader):
            if i >= max_batches:
                break
            hist_01 = hist_01.to(device, non_blocking=True)
            fut_flat = fut_flat.to(device, non_blocking=True)
            B = hist_01.shape[0]
            hist_norm = normalize_iv(hist_01).view(B, hist_01.shape[1], -1)
            samples_btd, _ = model(hist_norm, n_samples=K)  # (B, K, T, D)
            samples_grid = samples_btd.view(B, K, samples_btd.shape[2], H, W)
            gt_grid = fut_flat.view(B, fut_flat.shape[1], H, W)
            es_sum += energy_score(samples_grid, gt_grid).item()
            vs_sum += variogram_score(samples_grid, gt_grid, p=0.5).item()
            n += 1
    model.train()
    if n == 0 or vs_sum <= 0.0:
        return 0.5
    return float(es_sum) / float(vs_sum)


def make_dataset(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    device: str,
) -> tuple[DataLoader, DataLoader, int, int, int]:
    data = np.load(data_path)
    surfaces = data["surface"].astype(np.float32)  # (N, H, W)
    _, H, W = surfaces.shape
    max_train_idx = test_start - history_len - future_len
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(
        train_indices, surf_tensor, history_len, future_len
    )
    val_hist, val_future = build_multistep_windows(
        val_indices, surf_tensor, history_len, future_len
    )
    D = H * W
    return (train_hist, train_future, val_hist, val_future), H, W, D


def main() -> None:
    parser = argparse.ArgumentParser(description="250a: Stage A Neural Factor Model")
    parser.add_argument("--L", type=int, default=8, help="latent dim")
    parser.add_argument("--K", type=int, default=8, help="ensemble members")
    parser.add_argument("--encoder_hidden", type=int, default=64)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--idio_scale_clip", type=float, default=0.20)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)

    # Losses
    parser.add_argument("--lambda_cell", type=float, default=1.0)
    parser.add_argument(
        "--lambda_vs",
        type=str,
        default="auto",
        help="Variogram-score weight. 'auto' calibrates from mean_ES/mean_VS on a warmup pass.",
    )
    parser.add_argument("--lambda_es", type=float, default=1.0)
    parser.add_argument("--lambda_pmax", type=float, default=0.0,
                        help="afCRPS on pathwise max-|Δ| per cell. Attacks tail under-generation.")
    parser.add_argument("--lambda_chg", type=float, default=0.0,
                        help="afCRPS on per-cell daily changes (frame_sum). Attacks chg_KS and kurtosis.")
    parser.add_argument("--ortho_reg_weight", type=float, default=0.0)

    # Optim
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "none"])
    parser.add_argument("--clip_grad", type=float, default=1.0)

    # Data
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)

    # Infra
    parser.add_argument("--disable_early_stop", action="store_true", default=True)
    parser.add_argument(
        "--bf16", dest="bf16", action="store_true",
        help="Enable bf16 mixed precision (default on CUDA).",
    )
    parser.add_argument(
        "--no_bf16", dest="bf16", action="store_false",
        help="Disable bf16 mixed precision (fall back to fp32 autocast disabled).",
    )
    parser.set_defaults(bf16=True)
    parser.add_argument(
        "--compile", dest="use_compile", action="store_true",
        help="Compile the model with torch.compile (default off; adds ~1min warmup).",
    )
    parser.set_defaults(use_compile=False)
    parser.add_argument("--no_ema", action="store_true", default=True)
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "args.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device(args.device)

    # ---- Data ----
    tensors, H, W, D = make_dataset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        device=args.device,
    )
    train_hist, train_future, val_hist, val_future = tensors
    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size, shuffle=False, num_workers=0,
    )
    print(f"Train windows: {len(train_loader.dataset)}  Val windows: {len(val_loader.dataset)}")
    print(f"Grid: H={H}  W={W}  D={D}")

    # ---- Model ----
    cfg = NeuralFactorConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=D,
        latent_dim=args.L,
        encoder_hidden=args.encoder_hidden,
        bottleneck_dim=args.bottleneck_dim,
        head_hidden=args.head_hidden,
        head_layers=args.head_layers,
        head_dropout=args.head_dropout,
        encoder_dropout=args.encoder_dropout,
        idio_scale_clip=args.idio_scale_clip,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        ortho_reg_weight=args.ortho_reg_weight,
        use_marginal_head=False,
    )
    model = NeuralFactorModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # ---- lambda_vs auto-calibration (fp32 — stable stats) ----
    if args.lambda_vs == "auto":
        print("Calibrating lambda_vs (dualGNN-style scale matching) ...")
        lambda_vs = estimate_lambda_vs_auto(model, train_loader, args.device, args.K, H, W)
        print(f"  lambda_vs = {lambda_vs:.4f}")
    else:
        lambda_vs = float(args.lambda_vs)

    # ---- Optimizer / scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    if args.lr_schedule == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    # ---- Mixed precision: bf16 has fp32 dynamic range, no GradScaler needed ----
    device_is_cuda = device.type == "cuda"
    use_bf16 = bool(args.bf16) and device_is_cuda
    if use_bf16:
        print("Mixed precision: bf16 autocast on forward+loss (optimizer stays fp32)")
    else:
        print("Mixed precision: disabled (fp32 throughout)")

    def autocast_ctx():
        if use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cuda", enabled=False) if device_is_cuda \
            else torch.autocast(device_type="cpu", enabled=False)

    # ---- Optional torch.compile (skip for smoke tests; enable for long runs) ----
    if args.use_compile:
        print("Compiling model with torch.compile ...")
        model = torch.compile(model, mode="reduce-overhead")

    # ---- Training loop ----
    history_path = Path(args.output_dir) / "training_history.json"
    history: list[dict] = []
    best_val = float("inf")
    best_path = Path(args.output_dir) / "best_model.pt"
    final_path = Path(args.output_dir) / "final_model.pt"
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_sums: dict[str, float] = {}
        nb = 0
        optimizer.zero_grad()
        for batch_idx, (hist_01, fut_flat) in enumerate(train_loader):
            hist_01 = hist_01.to(device, non_blocking=True)
            fut_flat = fut_flat.to(device, non_blocking=True)
            B = hist_01.shape[0]
            hist_norm = normalize_iv(hist_01).view(B, hist_01.shape[1], -1)  # (B,T_hist,D)

            with autocast_ctx():
                samples, aux = model(hist_norm, n_samples=args.K)  # (B, K, T, D)
                loss, metrics = build_losses(
                    samples, fut_flat,
                    lambda_cell=args.lambda_cell, lambda_vs=lambda_vs, lambda_es=args.lambda_es,
                    H=H, W=W,
                    lambda_pmax=args.lambda_pmax, lambda_chg=args.lambda_chg,
                )
                if args.ortho_reg_weight > 0.0:
                    loss = loss + args.ortho_reg_weight * model.orthogonality_penalty(aux["Lambda"])

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch} batch {batch_idx}")

            (loss / args.grad_accum).backward()
            if (batch_idx + 1) % args.grad_accum == 0:
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                optimizer.step()
                optimizer.zero_grad()

            for k, v in metrics.items():
                ep_sums[k] = ep_sums.get(k, 0.0) + (v.item() if torch.is_tensor(v) else float(v))
            nb += 1
            global_step += 1

            if global_step % args.log_every_n_steps == 0:
                print(
                    f"  [ep {epoch} step {global_step}] "
                    f"cell={metrics['cell_crps'].item():.4f} "
                    f"vs={metrics['variogram_score'].item():.4f} "
                    f"es={metrics['energy_score'].item():.4f} "
                    f"loss={metrics['total'].item():.4f}"
                )

        if scheduler is not None:
            scheduler.step()
        train_avg = {k: v / nb for k, v in ep_sums.items()}

        # ---- Validation ----
        model.eval()
        val_sums: dict[str, float] = {}
        vb = 0
        with torch.no_grad():
            for hist_01, fut_flat in val_loader:
                hist_01 = hist_01.to(device, non_blocking=True)
                fut_flat = fut_flat.to(device, non_blocking=True)
                B = hist_01.shape[0]
                hist_norm = normalize_iv(hist_01).view(B, hist_01.shape[1], -1)
                with autocast_ctx():
                    samples, _ = model(hist_norm, n_samples=args.K)
                    _loss, metrics = build_losses(
                        samples, fut_flat,
                        lambda_cell=args.lambda_cell, lambda_vs=lambda_vs, lambda_es=args.lambda_es,
                        H=H, W=W,
                        lambda_pmax=args.lambda_pmax, lambda_chg=args.lambda_chg,
                    )
                for k, v in metrics.items():
                    val_sums[k] = val_sums.get(k, 0.0) + (v.item() if torch.is_tensor(v) else float(v))
                vb += 1
        val_avg = {k: v / vb for k, v in val_sums.items()} if vb > 0 else {}

        dt = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_cell={train_avg.get('cell_crps', 0):.4f}  "
            f"train_vs={train_avg.get('variogram_score', 0):.4f}  "
            f"train_es={train_avg.get('energy_score', 0):.4f}  "
            f"val_cell={val_avg.get('cell_crps', 0):.4f}  "
            f"val_total={val_avg.get('total', 0):.4f}  "
            f"({dt:.1f}s)"
        )

        # ---- Checkpoint selection — val cell_crps ----
        raw_state = (
            model._orig_mod.state_dict()
            if hasattr(model, "_orig_mod")
            else model.state_dict()
        )
        payload_best = {
            "config": asdict(cfg),
            "model_state_dict": raw_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_metrics": train_avg,
            "val_metrics": val_avg,
            "lambda_vs": lambda_vs,
        }
        if val_avg.get("cell_crps", float("inf")) < best_val:
            best_val = val_avg["cell_crps"]
            torch.save(payload_best, best_path)
            print(f"    best (val_cell={best_val:.4f}) saved to {best_path}")

        history.append({
            "epoch": epoch,
            "train": train_avg,
            "val": val_avg,
            "time_sec": dt,
        })
        history_path.write_text(json.dumps(history, indent=2))

    # ---- Final checkpoint ----
    raw_state = (
        model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    )
    payload_final = {
        "config": asdict(cfg),
        "model_state_dict": raw_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.epochs,
        "lambda_vs": lambda_vs,
    }
    torch.save(payload_final, final_path)
    print(f"\nFinal checkpoint saved to {final_path}")
    print(f"Best val cell_crps: {best_val:.4f}")


if __name__ == "__main__":
    main()
