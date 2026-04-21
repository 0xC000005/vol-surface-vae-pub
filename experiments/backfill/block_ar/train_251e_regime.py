#!/usr/bin/env python
"""
251e — H1: Regime-aware factor modulation.

Attaches RegimeEncoder(history) → r_slow ∈ R^regime_dim and concatenates to h
before LoadingHead and IdiosyncraticScaleHead. Λ(h, r_slow) and D(h, r_slow)
become regime-conditional while the latent SDE and loss stack stay unchanged.

Warm-start path requires rebuilding LoadingHead and IdiosyncraticScaleHead with
larger input_dim (bottleneck_dim + regime_dim). Zero-pad init preserves the
warm-start Λ/D behavior at t=0; gradient then unlocks regime routing.

Primary warm-start: 250c (cleaner — no SDE pre-training confound lets regime
modulate Λ/D from a static-z baseline).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.neural_factor import (
    IdiosyncraticScaleHead,
    LatentFM,
    LatentSDE,
    LoadingHead,
    NeuralFactorConfig,
    NeuralFactorModel,
    RegimeEncoder,
    load_model,
)
from diffusion.block_ar.single_pass_ar import (
    afcrps_loss, energy_score, normalize_iv, variogram_score,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
)


def _regime_width_loss(
    samples_grid: torch.Tensor,   # (B, K, T, H, W)
    hist_01: torch.Tensor,        # (B, T_hist, H, W) raw [0,1] IV
    target_ratio: float = 1.15,
) -> torch.Tensor:
    """H5: penalize ensembles for being regime-invariant in width.
    Classify each window calm/turb by HISTORY realized vol (RV). Compute per-window
    ensemble width (std over K samples, mean over cells/horizons). Regularizer:
    relu(target_ratio - width_turb/width_calm)^2.
    """
    B, K, T, H, W = samples_grid.shape
    # History RV per window — use history_01 shape (B, T_hist, D) or (B, T_hist, H, W)
    h_flat = hist_01.reshape(B, -1, H * W) if hist_01.dim() == 4 else hist_01
    # Realized vol proxy: mean absolute daily change over history
    rv = (h_flat[:, 1:] - h_flat[:, :-1]).abs().mean(dim=(1, 2))  # (B,)
    # Batch-median split
    median = rv.median()
    is_turb = (rv > median).float()
    is_calm = 1.0 - is_turb
    # Per-window ensemble width: std over K samples, mean over cells/horizons
    widths = samples_grid.std(dim=1).mean(dim=(1, 2, 3))  # (B,)
    # Per-regime mean width (weighted)
    w_turb = (widths * is_turb).sum() / is_turb.sum().clamp(min=1.0)
    w_calm = (widths * is_calm).sum() / is_calm.sum().clamp(min=1.0)
    # Target: turb width should exceed calm width by at least target_ratio
    actual_ratio = w_turb / (w_calm + 1e-8)
    # Penalize when actual < target (relu^2)
    penalty = torch.nn.functional.relu(target_ratio - actual_ratio).pow(2)
    return penalty


def build_losses(
    samples_btd: torch.Tensor, gt_btd: torch.Tensor,
    lambda_cell: float, lambda_vs: float, lambda_es: float, H: int, W: int,
    lambda_pmax: float = 0.0, lambda_chg: float = 0.0,
    lambda_regwidth: float = 0.0, hist_01_for_rv: torch.Tensor | None = None,
    regwidth_target: float = 1.15,
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
    if lambda_regwidth > 0.0 and hist_01_for_rv is not None:
        regwidth_loss = _regime_width_loss(samples_grid, hist_01_for_rv, target_ratio=regwidth_target)
        total = total + lambda_regwidth * regwidth_loss
        metrics["regwidth_loss"] = regwidth_loss.detach()
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
    """Keep LatentSDE, LatentFM, AND regime_encoder + heads trainable. Freeze other backbone."""
    # H1: "regime track" = regime_encoder + the two heads that consume r_slow
    regime_prefixes = ("latent_sde.", "latent_fm.", "regime_encoder.", "loading_head.", "idio_head.")
    for name, p in model.named_parameters():
        if any(name.startswith(pref) for pref in regime_prefixes):
            p.requires_grad_(True)
        else:
            p.requires_grad_(trainable)


def _rebuild_head_with_regime(old_head: nn.Module, new_cfg: NeuralFactorConfig, HeadClass, device):
    """Rebuild LoadingHead or IdiosyncraticScaleHead with extended input_dim.
    Zero-pad the regime columns so initial behavior matches the warm-start.
    """
    new_head = HeadClass(new_cfg).to(device)
    # Shape audit
    old_first = old_head.net[0]
    new_first = new_head.net[0]
    regime_cols = new_first.in_features - old_first.in_features
    assert regime_cols == new_cfg.regime_dim, (
        f"rebuild mismatch: new_in={new_first.in_features}, old_in={old_first.in_features}, "
        f"regime_dim={new_cfg.regime_dim}"
    )
    with torch.no_grad():
        # Copy original weights into the first old_in columns; zero-init regime columns.
        new_first.weight[:, :old_first.in_features].copy_(old_first.weight)
        new_first.weight[:, old_first.in_features:].zero_()
        new_first.bias.copy_(old_first.bias)
        # Deeper layers: copy state_dict for each sub-module after the first Linear.
        for i in range(1, len(new_head.net)):
            src_mod = old_head.net[i]
            dst_mod = new_head.net[i]
            if hasattr(src_mod, "state_dict"):
                try:
                    dst_mod.load_state_dict(src_mod.state_dict())
                except (RuntimeError, ValueError):
                    pass  # ReLU/SiLU/Dropout have no params
    return new_head


def main() -> None:
    parser = argparse.ArgumentParser(description="251e: H1 regime-aware modulation")
    parser.add_argument("--warm_start", type=str, required=True,
                        help="Path to 250c_best_model.pt (preferred), 251b_final, or 251c_best")
    # LatentSDE / OU hparams (inherit from warm-start unless explicitly overridden)
    parser.add_argument("--use_latent_sde", dest="use_latent_sde", action="store_true", default=True)
    parser.add_argument("--no_latent_sde", dest="use_latent_sde", action="store_false")
    parser.add_argument("--latent_sde_hidden", type=int, default=256)
    parser.add_argument("--latent_sde_time_embed", type=int, default=0)
    parser.add_argument("--latent_sde_eps_floor", type=float, default=1e-4)
    parser.add_argument("--use_ou_prior", dest="use_ou_prior", action="store_true", default=False)
    parser.add_argument("--no_ou_prior", dest="use_ou_prior", action="store_false")
    parser.add_argument("--lambda_ou", type=float, default=0.0)
    parser.add_argument("--ou_alpha_init", type=float, default=-4.0)
    # H1 Regime
    parser.add_argument("--use_regime", dest="use_regime", action="store_true", default=True)
    parser.add_argument("--no_regime", dest="use_regime", action="store_false")
    parser.add_argument("--regime_dim", type=int, default=8)
    parser.add_argument("--regime_hidden", type=int, default=64)
    # LatentFM
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
    # Loss
    parser.add_argument("--lambda_cell", type=float, default=1.0)
    parser.add_argument("--lambda_vs", type=str, default="auto")
    parser.add_argument("--lambda_es", type=float, default=1.0)
    parser.add_argument("--lambda_pmax", type=float, default=0.5)
    parser.add_argument("--lambda_chg", type=float, default=0.2)
    # H5 width-regularizer: penalize regime-invariant widths
    parser.add_argument("--lambda_regwidth", type=float, default=0.0,
                        help="H5: penalize when turb/calm ensemble width ratio < target (batch RV-split)")
    parser.add_argument("--regwidth_target", type=float, default=1.15,
                        help="Target ratio turb/calm width (default 1.15 matches conditionality gate)")
    # Data / misc
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--bf16", dest="bf16", action="store_true")
    parser.add_argument("--no_bf16", dest="bf16", action="store_false")
    parser.set_defaults(bf16=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "args.json").write_text(json.dumps(vars(args), indent=2))

    device = torch.device(args.device)

    # Warm-start
    model, payload = load_model(args.warm_start, device)
    print(f"Warm-started from {args.warm_start} (epoch {payload.get('epoch', -1)})")

    # Ensure LatentFM is present
    if not model.cfg.use_latent_fm:
        print("  warm-start has no LatentFM; attaching fresh p_0")
        model.cfg.use_latent_fm = True
        model.cfg.latent_fm_steps = args.fm_steps
        model.cfg.latent_fm_hidden = args.fm_hidden
        model.cfg.latent_fm_time_embed = args.fm_time_embed
        model.latent_fm = LatentFM(model.cfg).to(device)
    else:
        print(f"  inherited LatentFM (steps={model.cfg.latent_fm_steps})")

    # Attach / keep LatentSDE
    if args.use_latent_sde:
        if model.latent_sde is None:
            print(f"  attaching fresh LatentSDE (hidden={args.latent_sde_hidden}, time_embed={args.latent_sde_time_embed})")
            model.cfg.use_latent_sde = True
            model.cfg.latent_sde_hidden = args.latent_sde_hidden
            model.cfg.latent_sde_time_embed = args.latent_sde_time_embed
            model.cfg.latent_sde_eps_floor = args.latent_sde_eps_floor
            model.latent_sde = LatentSDE(model.cfg).to(device)
        else:
            print(f"  inherited LatentSDE")

    # OU prior (optional)
    if args.use_ou_prior:
        if model.latent_sde is None:
            raise ValueError("OU prior requires LatentSDE")
        if model.latent_sde.alpha_raw is None:
            model.cfg.use_ou_prior = True
            model.cfg.ou_alpha_init = args.ou_alpha_init
            model.latent_sde.alpha_raw = nn.Parameter(
                torch.tensor(float(args.ou_alpha_init), device=device)
            )

    # H1: Attach RegimeEncoder + rebuild LoadingHead and IdiosyncraticScaleHead with extended input
    if args.use_regime and model.regime_encoder is None:
        print(f"  H1: enabling regime-aware modulation (regime_dim={args.regime_dim})")
        model.cfg.use_regime = True
        model.cfg.regime_dim = args.regime_dim
        model.cfg.regime_hidden = args.regime_hidden
        # Fresh RegimeEncoder
        model.regime_encoder = RegimeEncoder(model.cfg).to(device)
        # Rebuild heads with regime-extended input
        model.loading_head = _rebuild_head_with_regime(
            model.loading_head, model.cfg, LoadingHead, device
        )
        model.idio_head = _rebuild_head_with_regime(
            model.idio_head, model.cfg, IdiosyncraticScaleHead, device
        )
        # Verify
        names = {n for n, _ in model.named_parameters()}
        assert any(n.startswith("regime_encoder.") for n in names), "regime_encoder not in parameters"
        print(f"  LoadingHead in_dim: {model.loading_head.net[0].in_features}  (was {model.cfg.bottleneck_dim})")
        print(f"  IdiosyncraticScaleHead in_dim: {model.idio_head.net[0].in_features}")

    n_params = sum(p.numel() for p in model.parameters())
    n_regime = sum(p.numel() for p in model.regime_encoder.parameters()) if model.regime_encoder is not None else 0
    print(f"Total params: {n_params:,}   RegimeEncoder: {n_regime:,}")

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

    # Phase 1
    set_backbone_trainable(model, trainable=False)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=args.weight_decay,
    )
    n_train_now = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPhase 1: freeze backbone, train regime+SDE+FM+heads ({n_train_now:,} trainable params). "
          f"epochs 1..{args.freeze_backbone_epochs}")

    history: list[dict] = []
    best_val = float("inf")
    best_path = Path(args.output_dir) / "best_model.pt"
    final_path = Path(args.output_dir) / "final_model.pt"
    history_path = Path(args.output_dir) / "training_history.json"

    def _epoch(loader, train_mode: bool, epoch: int) -> dict:
        model.train() if train_mode else model.eval()
        ep_sums: dict[str, float] = {}
        r_slow_norms = []
        nb = 0
        with torch.set_grad_enabled(train_mode):
            for batch_idx, (hist_01, fut_flat) in enumerate(loader):
                hist_01 = hist_01.to(device, non_blocking=True)
                fut_flat = fut_flat.to(device, non_blocking=True)
                B = hist_01.shape[0]
                hist_norm = normalize_iv(hist_01).view(B, hist_01.shape[1], -1)
                with autocast_ctx():
                    samples, aux = model(hist_norm, n_samples=args.K)
                    loss, metrics = build_losses(
                        samples, fut_flat,
                        lambda_cell=args.lambda_cell, lambda_vs=lambda_vs, lambda_es=args.lambda_es,
                        H=H, W=W,
                        lambda_pmax=args.lambda_pmax, lambda_chg=args.lambda_chg,
                        lambda_regwidth=args.lambda_regwidth, hist_01_for_rv=hist_01,
                        regwidth_target=args.regwidth_target,
                    )
                if aux.get("r_slow") is not None:
                    r_slow_norms.append(aux["r_slow"].detach().float().std().item())
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
        result = {k: v / max(nb, 1) for k, v in ep_sums.items()}
        if r_slow_norms:
            result["r_slow_std"] = sum(r_slow_norms) / len(r_slow_norms)
        return result

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
        r_slow_log = f"  r_slow_std={train_avg.get('r_slow_std', 0):.4f}"
        print(
            f"Epoch {epoch:3d}/{args.total_epochs} [{phase}] "
            f"train_cell={train_avg.get('cell_crps', 0):.4f}  "
            f"val_cell={val_avg.get('cell_crps', 0):.4f}  "
            f"val_total={val_avg.get('total', 0):.4f}{r_slow_log}  ({dt:.1f}s)"
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
