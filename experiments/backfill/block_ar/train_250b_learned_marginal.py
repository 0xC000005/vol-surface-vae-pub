#!/usr/bin/env python
"""
250b: Stage B — add LearnedMarginalHead on top of 250a's factor generator.

Warm-starts from a 250a best_model.pt, enables cfg.use_marginal_head=True, instantiates
the marginal head (Choice A — conditional spline head, all knots learnable).

Schedule:
  - first `--freeze_backbone_epochs` epochs: train ONLY the marginal head (backbone frozen)
  - remaining epochs: unfreeze backbone and co-train

Loss stack is unchanged from 250a.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.neural_factor import (
    LearnedMarginalHead,
    NeuralFactorConfig,
    NeuralFactorModel,
    load_model,
)
from diffusion.block_ar.single_pass_ar import (
    afcrps_loss, energy_score, normalize_iv, variogram_score,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
)


def build_losses(
    samples_btd: torch.Tensor, gt_btd: torch.Tensor,
    lambda_cell: float, lambda_vs: float, lambda_es: float, H: int, W: int,
) -> tuple[torch.Tensor, dict]:
    B, K, T, D = samples_btd.shape
    samples_grid = samples_btd.view(B, K, T, H, W)
    gt_grid = gt_btd.view(B, T, H, W)
    cell_crps, _mae, _spread = afcrps_loss(
        samples_grid, gt_grid, alpha=0.95, reduction="frame_sum"
    )
    vs = variogram_score(samples_grid, gt_grid, p=0.5)
    es = energy_score(samples_grid, gt_grid)
    total = lambda_cell * cell_crps + lambda_vs * vs + lambda_es * es
    return total, {
        "cell_crps": cell_crps.detach(),
        "variogram_score": vs.detach(),
        "energy_score": es.detach(),
        "total": total.detach(),
    }


def estimate_lambda_vs_auto(
    model: NeuralFactorModel, loader: DataLoader, device: str, K: int, H: int, W: int,
    max_batches: int = 40,
) -> float:
    model.eval()
    es_sum = vs_sum = 0.0
    n = 0
    with torch.no_grad():
        for i, (hist_01, fut_flat) in enumerate(loader):
            if i >= max_batches:
                break
            hist_01 = hist_01.to(device, non_blocking=True)
            fut_flat = fut_flat.to(device, non_blocking=True)
            B = hist_01.shape[0]
            hist_norm = normalize_iv(hist_01).view(B, hist_01.shape[1], -1)
            s, _ = model(hist_norm, n_samples=K)
            s_grid = s.view(B, K, s.shape[2], H, W)
            g_grid = fut_flat.view(B, fut_flat.shape[1], H, W)
            es_sum += energy_score(s_grid, g_grid).item()
            vs_sum += variogram_score(s_grid, g_grid, p=0.5).item()
            n += 1
    model.train()
    if n == 0 or vs_sum <= 0.0:
        return 0.5
    return float(es_sum) / float(vs_sum)


def attach_marginal_head(model: NeuralFactorModel, K_knots: int, device: torch.device) -> NeuralFactorModel:
    """Create a fresh marginal head on the model and update config."""
    model.cfg.use_marginal_head = True
    model.cfg.marginal_knots = K_knots
    model.marginal_head = LearnedMarginalHead(model.cfg).to(device)
    return model


def set_backbone_trainable(model: NeuralFactorModel, trainable: bool) -> None:
    for name, p in model.named_parameters():
        if name.startswith("marginal_head."):
            p.requires_grad_(True)
        else:
            p.requires_grad_(trainable)


def main() -> None:
    parser = argparse.ArgumentParser(description="250b: Stage B Learned Marginal Head")
    parser.add_argument("--warm_start", type=str, required=True,
                        help="Path to a 250a best_model.pt (Stage A output)")
    parser.add_argument("--K_knots", type=int, default=12)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=10)
    parser.add_argument("--total_epochs", type=int, default=20)
    parser.add_argument("--K", type=int, default=8, help="ensemble members")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--lambda_cell", type=float, default=1.0)
    parser.add_argument("--lambda_vs", type=str, default="auto")
    parser.add_argument("--lambda_es", type=float, default=1.0)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--bf16", dest="bf16", action="store_true")
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.set_defaults(bf16=True)
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

    # ---- Warm-start from Stage A ----
    model, payload = load_model(args.warm_start, device)
    print(f"Warm-started from {args.warm_start} (epoch {payload.get('epoch', -1)})")
    print(f"  config L={model.cfg.latent_dim}  D={model.cfg.n_cells}  T={model.cfg.future_len}")

    # ---- Attach marginal head ----
    model = attach_marginal_head(model, args.K_knots, device)
    n_params = sum(p.numel() for p in model.parameters())
    n_head = sum(p.numel() for p in model.marginal_head.parameters())
    print(f"Total params: {n_params:,}   Marginal head params: {n_head:,}")

    # ---- Data ----
    data = np.load(args.data_path)
    surfaces = data["surface"].astype(np.float32)
    _, H, W = surfaces.shape
    max_train_idx = args.test_start - args.history_len - args.future_len
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, args.history_len, args.future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, args.history_len, args.future_len)
    train_loader = DataLoader(
        TensorDataset(train_hist, train_future), batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False,
    )
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Grid: {H}x{W} D={H*W}")

    # ---- lambda_vs calibration (fresh — head changes the scale) ----
    if args.lambda_vs == "auto":
        lambda_vs = estimate_lambda_vs_auto(model, train_loader, args.device, args.K, H, W)
        print(f"lambda_vs = {lambda_vs:.4f}")
    else:
        lambda_vs = float(args.lambda_vs)

    # ---- Mixed precision ----
    device_is_cuda = device.type == "cuda"
    use_bf16 = bool(args.bf16) and device_is_cuda

    def autocast_ctx():
        if use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cuda", enabled=False) if device_is_cuda \
            else torch.autocast(device_type="cpu", enabled=False)

    # ---- Phase 1: freeze backbone, train head only ----
    set_backbone_trainable(model, trainable=False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    print(f"\nPhase 1: freeze backbone, train head only. epochs 1..{args.freeze_backbone_epochs}")

    history: list[dict] = []
    best_val = float("inf")
    best_path = Path(args.output_dir) / "best_model.pt"
    final_path = Path(args.output_dir) / "final_model.pt"
    history_path = Path(args.output_dir) / "training_history.json"
    global_step = 0

    def _epoch(model, loader, train_mode: bool, epoch: int) -> dict:
        nonlocal global_step
        model.train() if train_mode else model.eval()
        ep_sums: dict[str, float] = {}
        nb = 0
        with torch.set_grad_enabled(train_mode):
            for batch_idx, (hist_01, fut_flat) in enumerate(loader):
                hist_01 = hist_01.to(device, non_blocking=True)
                fut_flat = fut_flat.to(device, non_blocking=True)
                B = hist_01.shape[0]
                hist_norm = normalize_iv(hist_01).view(B, hist_01.shape[1], -1)
                with autocast_ctx():
                    samples, _ = model(hist_norm, n_samples=args.K)
                    loss, metrics = build_losses(
                        samples, fut_flat,
                        lambda_cell=args.lambda_cell, lambda_vs=lambda_vs, lambda_es=args.lambda_es,
                        H=H, W=W,
                    )
                if train_mode:
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at epoch {epoch} batch {batch_idx}")
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad],
                            args.clip_grad,
                        )
                    optimizer.step()
                    optimizer.zero_grad()
                for k, v in metrics.items():
                    ep_sums[k] = ep_sums.get(k, 0.0) + (v.item() if torch.is_tensor(v) else float(v))
                nb += 1
                global_step += 1
        return {k: v / max(nb, 1) for k, v in ep_sums.items()}

    for epoch in range(1, args.total_epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            print(f"\nPhase 2: unfreeze backbone, co-train. epochs {epoch}..{args.total_epochs}")
            set_backbone_trainable(model, trainable=True)
            # Rebuild optimizer so all params get an entry
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            )

        t0 = time.time()
        train_avg = _epoch(model, train_loader, train_mode=True, epoch=epoch)
        val_avg = _epoch(model, val_loader, train_mode=False, epoch=epoch)
        dt = time.time() - t0

        phase = "frozen" if epoch <= args.freeze_backbone_epochs else "cotrain"
        print(
            f"Epoch {epoch:3d}/{args.total_epochs} [{phase}] "
            f"train_cell={train_avg.get('cell_crps', 0):.4f}  "
            f"val_cell={val_avg.get('cell_crps', 0):.4f}  "
            f"val_total={val_avg.get('total', 0):.4f}  ({dt:.1f}s)"
        )

        raw_state = (
            model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
        )
        payload_ckpt = {
            "config": asdict(model.cfg),
            "model_state_dict": raw_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_metrics": train_avg,
            "val_metrics": val_avg,
            "lambda_vs": lambda_vs,
            "phase": phase,
            "warm_start": args.warm_start,
        }
        if val_avg.get("cell_crps", float("inf")) < best_val:
            best_val = val_avg["cell_crps"]
            torch.save(payload_ckpt, best_path)
            print(f"    best (val_cell={best_val:.4f}) saved to {best_path}")
        history.append({
            "epoch": epoch, "phase": phase,
            "train": train_avg, "val": val_avg, "time_sec": dt,
        })
        history_path.write_text(json.dumps(history, indent=2))

    raw_state = (
        model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
    )
    payload_final = {
        "config": asdict(model.cfg),
        "model_state_dict": raw_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": args.total_epochs,
        "lambda_vs": lambda_vs,
        "warm_start": args.warm_start,
    }
    torch.save(payload_final, final_path)
    print(f"\nFinal checkpoint saved to {final_path}")
    print(f"Best val cell_crps: {best_val:.4f}")


if __name__ == "__main__":
    main()
