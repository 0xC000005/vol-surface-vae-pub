#!/usr/bin/env python
"""
251a — DLFM (Dynamical Latent Factor Model).

Replaces the static latent z (drawn once per trajectory) with a time-indexed z_t
evolving as a learned neural SDE:

    z_0         ~ p_0(· | h)                                    # inherit LatentFM
    z_{t+1}     = z_t + f_θ(z_t, t, h) · Δt + g_θ(z_t, t, h) · ε_t · √Δt
    Δsurface[t] = Λ(h)[t] · z_t + D(h)[t] · η[t]

One new MLP head f_θ / g_θ (~120k params at L=16). Hypothesis-class upgrade from
"static factor readout" to "neural SDE" — proper scoring rules now have a real
chance of converging to the true distribution.

Warm-start from 250c (primary) to isolate the SDE contribution; optional
secondary warm-start from 250ac for an optimistic stacked test.
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
    LatentFM,
    LatentSDE,
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
    lambda_pmax: float = 0.0, lambda_chg: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    B, K, T, D = samples_btd.shape
    samples_grid = samples_btd.view(B, K, T, H, W)
    gt_grid = gt_btd.view(B, T, H, W)
    cell_crps, _mae, _spread = afcrps_loss(samples_grid, gt_grid, alpha=0.95, reduction="frame_sum")
    vs = variogram_score(samples_grid, gt_grid, p=0.5)
    es = energy_score(samples_grid, gt_grid)
    total = lambda_cell * cell_crps + lambda_vs * vs + lambda_es * es
    metrics = {
        "cell_crps": cell_crps.detach(),
        "variogram_score": vs.detach(),
        "energy_score": es.detach(),
    }
    if lambda_pmax > 0.0 and T >= 2:
        s_chg = samples_grid[:, :, 1:] - samples_grid[:, :, :-1]
        g_chg = gt_grid[:, 1:] - gt_grid[:, :-1]
        s_pmax = s_chg.abs().amax(dim=2, keepdim=True)
        g_pmax = g_chg.abs().amax(dim=1, keepdim=True)
        pmax_crps, _, _ = afcrps_loss(s_pmax, g_pmax, alpha=0.95, reduction="frame_sum")
        total = total + lambda_pmax * pmax_crps
        metrics["pmax_crps"] = pmax_crps.detach()
    if lambda_chg > 0.0 and T >= 2:
        s_chg = samples_grid[:, :, 1:] - samples_grid[:, :, :-1]
        g_chg = gt_grid[:, 1:] - gt_grid[:, :-1]
        chg_crps, _, _ = afcrps_loss(s_chg, g_chg, alpha=0.95, reduction="frame_sum")
        total = total + lambda_chg * chg_crps
        metrics["chg_crps"] = chg_crps.detach()
    metrics["total"] = total.detach()
    return total, metrics


def estimate_lambda_vs_auto(model, loader, device, K, H, W, max_batches=40):
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
    return float(es_sum) / float(vs_sum) if n > 0 and vs_sum > 0 else 0.5


def set_backbone_trainable(model: NeuralFactorModel, trainable: bool) -> None:
    """Freeze backbone — ALWAYS keep LatentSDE and LatentFM (p_0) trainable.

    Phase 1 (trainable=False): only the new dynamics (latent_sde) and p_0
    (latent_fm) receive gradients. Lambda, D, encoder, latent_encoder frozen so
    the SDE has to do its own work.

    Phase 2 (trainable=True): everything co-trains.
    """
    for name, p in model.named_parameters():
        if name.startswith("latent_sde.") or name.startswith("latent_fm."):
            p.requires_grad_(True)
        else:
            p.requires_grad_(trainable)


def main() -> None:
    parser = argparse.ArgumentParser(description="251a: DLFM dynamical latent factor model")
    parser.add_argument("--warm_start", type=str, required=True,
                        help="Path to 250c (primary) or 250ac (secondary) best_model.pt")
    # LatentSDE hparams
    parser.add_argument("--use_latent_sde", dest="use_latent_sde", action="store_true", default=True)
    parser.add_argument("--no_latent_sde", dest="use_latent_sde", action="store_false")
    parser.add_argument("--latent_sde_hidden", type=int, default=256)
    parser.add_argument("--latent_sde_time_embed", type=int, default=32)
    parser.add_argument("--latent_sde_eps_floor", type=float, default=1e-4)
    # LatentFM (p_0) hparams — inherited from 250c/ac warm-start, but overridable
    parser.add_argument("--fm_steps", type=int, default=4)
    parser.add_argument("--fm_hidden", type=int, default=256)
    parser.add_argument("--fm_time_embed", type=int, default=32)
    # Training phases
    parser.add_argument("--freeze_backbone_epochs", type=int, default=10)
    parser.add_argument("--total_epochs", type=int, default=20)
    parser.add_argument("--K", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    # Loss stack (match 250ac recipe by default)
    parser.add_argument("--lambda_cell", type=float, default=1.0)
    parser.add_argument("--lambda_vs", type=str, default="auto")
    parser.add_argument("--lambda_es", type=float, default=1.0)
    parser.add_argument("--lambda_pmax", type=float, default=0.5)
    parser.add_argument("--lambda_chg", type=float, default=0.2)
    # Data / misc
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

    # Warm-start backbone (load whatever 250 variant was passed)
    model, payload = load_model(args.warm_start, device)
    print(f"Warm-started from {args.warm_start} (epoch {payload.get('epoch', -1)})")

    # Ensure LatentFM is present as p_0 (may already be there if warm-start is 250c/ac).
    if not model.cfg.use_latent_fm:
        print("  warm-start has no LatentFM; attaching fresh p_0")
        model.cfg.use_latent_fm = True
        model.cfg.latent_fm_steps = args.fm_steps
        model.cfg.latent_fm_hidden = args.fm_hidden
        model.cfg.latent_fm_time_embed = args.fm_time_embed
        model.latent_fm = LatentFM(model.cfg).to(device)
    else:
        print(f"  inherited LatentFM (steps={model.cfg.latent_fm_steps}, hidden={model.cfg.latent_fm_hidden})")

    # Attach fresh LatentSDE on top.
    if args.use_latent_sde:
        model.cfg.use_latent_sde = True
        model.cfg.latent_sde_hidden = args.latent_sde_hidden
        model.cfg.latent_sde_time_embed = args.latent_sde_time_embed
        model.cfg.latent_sde_eps_floor = args.latent_sde_eps_floor
        model.latent_sde = LatentSDE(model.cfg).to(device)
        print(f"  attached fresh LatentSDE (hidden={args.latent_sde_hidden}, eps_floor={args.latent_sde_eps_floor})")
    else:
        print("  --no_latent_sde set; this is a degenerate case (should match 250c)")

    n_params = sum(p.numel() for p in model.parameters())
    n_sde = sum(p.numel() for p in model.latent_sde.parameters()) if model.latent_sde is not None else 0
    n_fm = sum(p.numel() for p in model.latent_fm.parameters()) if model.latent_fm is not None else 0
    print(f"Total params: {n_params:,}   LatentSDE: {n_sde:,}   LatentFM: {n_fm:,}")

    # Data
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

    if args.lambda_vs == "auto":
        lambda_vs = estimate_lambda_vs_auto(model, train_loader, args.device, args.K, H, W)
        print(f"lambda_vs (auto) = {lambda_vs:.4f}")
    else:
        lambda_vs = float(args.lambda_vs)

    device_is_cuda = device.type == "cuda"
    use_bf16 = bool(args.bf16) and device_is_cuda

    def autocast_ctx():
        if use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return torch.autocast(device_type="cuda", enabled=False) if device_is_cuda \
            else torch.autocast(device_type="cpu", enabled=False)

    # Phase 1: freeze backbone, train LatentSDE + LatentFM (p_0) only
    set_backbone_trainable(model, trainable=False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    n_train_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPhase 1: freeze backbone, train LatentSDE + LatentFM only ({n_train_now:,} trainable params). "
          f"epochs 1..{args.freeze_backbone_epochs}")

    history: list[dict] = []
    best_val = float("inf")
    best_path = Path(args.output_dir) / "best_model.pt"
    final_path = Path(args.output_dir) / "final_model.pt"
    history_path = Path(args.output_dir) / "training_history.json"
    global_step = 0

    def _epoch(loader, train_mode: bool, epoch: int) -> dict:
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
                        lambda_pmax=args.lambda_pmax, lambda_chg=args.lambda_chg,
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
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            )

        t0 = time.time()
        train_avg = _epoch(train_loader, train_mode=True, epoch=epoch)
        val_avg = _epoch(val_loader, train_mode=False, epoch=epoch)
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
