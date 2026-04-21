#!/usr/bin/env python
"""
257c: latent future-token VAE with multi-sample scenario objective.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.latent_future_token_vae import (
    LatentFutureTokenVAE,
    LatentFutureTokenVAEConfig,
    config_to_dict,
)
from diffusion.block_ar.single_pass_ar import normalize_iv
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
)


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none")
    if weights is not None:
        loss = loss * weights
    return loss.mean()


def sample_crps(samples: torch.Tensor, target: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    target_term = torch.abs(samples - target.unsqueeze(1))
    if weights is not None:
        target_term = target_term * weights.view(1, 1, weights.shape[1], 1)
    term1 = target_term.mean()

    pairwise = torch.abs(samples.unsqueeze(2) - samples.unsqueeze(1))
    if weights is not None:
        pairwise = pairwise * weights.view(1, 1, 1, weights.shape[1], 1)
    term2 = 0.5 * pairwise.mean()
    return term1 - term2


def temporal_covariance(x: torch.Tensor) -> torch.Tensor:
    xc = x - x.mean(dim=1, keepdim=True)
    denom = max(x.shape[1] - 1, 1)
    return torch.einsum("btd,bte->bde", xc, xc) / float(denom)


def covariance_to_correlation(cov: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    diag = torch.diagonal(cov, dim1=-2, dim2=-1).clamp_min(eps).sqrt()
    denom = diag.unsqueeze(-1) * diag.unsqueeze(-2)
    return cov / denom.clamp_min(eps)


def corr_structure_loss(gen_path: torch.Tensor, gt_path: torch.Tensor) -> torch.Tensor:
    gen_cov = temporal_covariance(gen_path)
    gt_cov = temporal_covariance(gt_path)
    gen_corr = covariance_to_correlation(gen_cov)
    gt_corr = covariance_to_correlation(gt_cov)
    return F.smooth_l1_loss(gen_corr, gt_corr)


def spectrum_structure_loss(gen_path: torch.Tensor, gt_path: torch.Tensor, topk: int = 8) -> torch.Tensor:
    gen_cov = temporal_covariance(gen_path)
    gt_cov = temporal_covariance(gt_path)
    gen_eigs = torch.linalg.eigvalsh(gen_cov).flip(dims=[-1]).clamp_min(0.0)
    gt_eigs = torch.linalg.eigvalsh(gt_cov).flip(dims=[-1]).clamp_min(0.0)
    gen_eigs = gen_eigs / gen_eigs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    gt_eigs = gt_eigs / gt_eigs.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    k = min(topk, gen_eigs.shape[-1], gt_eigs.shape[-1])
    return F.smooth_l1_loss(gen_eigs[:, :k], gt_eigs[:, :k])


def make_dataset(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    device: str,
) -> tuple[tuple[torch.Tensor, ...], int, int, int]:
    data = np.load(data_path)
    surfaces = data["surface"].astype(np.float32)
    _, H, W = surfaces.shape
    max_train_idx = test_start - history_len - future_len
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, history_len, future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, history_len, future_len)
    D = H * W
    return (train_hist, train_future, val_hist, val_future), H, W, D


def build_losses(
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    pred_level: torch.Tensor,
    aux: dict[str, torch.Tensor],
    ms_level_samples: torch.Tensor,
    ms_change_samples: torch.Tensor,
    lambda_change_corr: float,
    lambda_change_spec: float,
    lambda_level_corr: float,
    lambda_level_spec: float,
    structure_topk: int,
    lambda_level: float,
    lambda_change: float,
    lambda_jump: float,
    lambda_terminal: float,
    lambda_resid: float,
    lambda_kl: float,
    lambda_kl_floor: float,
    kl_floor: float,
    lambda_ms_level: float,
    lambda_ms_change: float,
    terminal_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    _, T, _ = pred_level.shape
    h_weights = torch.linspace(
        1.0, terminal_weight, T, device=pred_level.device, dtype=pred_level.dtype
    ).view(1, T, 1)

    pred_change = aux["mean_change"]
    last_hist = history_norm[:, -1:, :]
    gt_prev = torch.cat([last_hist, future_norm[:, :-1]], dim=1)
    gt_change = future_norm - gt_prev

    level_loss = weighted_smooth_l1(pred_level, future_norm, h_weights)
    change_loss = weighted_smooth_l1(pred_change, gt_change, h_weights)
    pred_pmax = pred_change.abs().amax(dim=1)
    gt_pmax = gt_change.abs().amax(dim=1)
    jump_loss = F.smooth_l1_loss(pred_pmax, gt_pmax)
    terminal_loss = F.smooth_l1_loss(pred_level[:, -1, :], future_norm[:, -1, :])

    ms_level_loss = sample_crps(ms_level_samples, future_norm, h_weights)
    ms_change_loss = sample_crps(ms_change_samples, gt_change, h_weights)
    ensemble_mean_level = ms_level_samples.mean(dim=1)
    ensemble_mean_change = ms_change_samples.mean(dim=1)
    change_corr_loss = corr_structure_loss(ensemble_mean_change, gt_change)
    change_spec_loss = spectrum_structure_loss(ensemble_mean_change, gt_change, topk=structure_topk)
    level_corr_loss = corr_structure_loss(ensemble_mean_level, future_norm)
    level_spec_loss = spectrum_structure_loss(ensemble_mean_level, future_norm, topk=structure_topk)

    resid_rms = aux["mean_resid"].pow(2).mean().sqrt()
    kl = aux["kl"]
    kl_floor_penalty = torch.relu(torch.as_tensor(kl_floor, device=kl.device, dtype=kl.dtype) - kl).square()
    attn = aux["attn_weights"].clamp_min(1e-8)
    attn_entropy = -(attn * attn.log()).sum(dim=-1).mean()
    token_std = aux["post_logvar"].exp().sqrt().mean()
    token_top1 = attn.max(dim=-1).values.mean()

    total = (
        lambda_level * level_loss
        + lambda_change * change_loss
        + lambda_jump * jump_loss
        + lambda_terminal * terminal_loss
        + lambda_resid * resid_rms
        + lambda_kl * kl
        + lambda_kl_floor * kl_floor_penalty
        + lambda_ms_level * ms_level_loss
        + lambda_ms_change * ms_change_loss
        + lambda_change_corr * change_corr_loss
        + lambda_change_spec * change_spec_loss
        + lambda_level_corr * level_corr_loss
        + lambda_level_spec * level_spec_loss
    )
    metrics = {
        "level_loss": level_loss.detach(),
        "change_loss": change_loss.detach(),
        "jump_loss": jump_loss.detach(),
        "terminal_loss": terminal_loss.detach(),
        "resid_rms": resid_rms.detach(),
        "kl": kl.detach(),
        "kl_floor_penalty": kl_floor_penalty.detach(),
        "ms_level_loss": ms_level_loss.detach(),
        "ms_change_loss": ms_change_loss.detach(),
        "change_corr_loss": change_corr_loss.detach(),
        "change_spec_loss": change_spec_loss.detach(),
        "level_corr_loss": level_corr_loss.detach(),
        "level_spec_loss": level_spec_loss.detach(),
        "attn_entropy": attn_entropy.detach(),
        "token_std": token_std.detach(),
        "token_top1": token_top1.detach(),
        "total": total.detach(),
    }
    return total, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="257c latent future-token VAE with multi-sample objective")
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--token_dim", type=int, default=64)
    parser.add_argument("--n_tokens", type=int, default=4)
    parser.add_argument("--encoder_hidden", type=int, default=64)
    parser.add_argument("--bottleneck_dim", type=int, default=128)
    parser.add_argument("--encoder_dropout", type=float, default=0.1)
    parser.add_argument("--head_hidden", type=int, default=256)
    parser.add_argument("--head_layers", type=int, default=2)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--obs_hidden", type=int, default=64)
    parser.add_argument("--time_embed_dim", type=int, default=16)
    parser.add_argument("--max_resid_ratio", type=float, default=0.30)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)

    parser.add_argument("--lambda_level", type=float, default=0.75)
    parser.add_argument("--lambda_change", type=float, default=1.25)
    parser.add_argument("--lambda_jump", type=float, default=0.50)
    parser.add_argument("--lambda_terminal", type=float, default=0.75)
    parser.add_argument("--lambda_resid", type=float, default=0.05)
    parser.add_argument("--lambda_kl", type=float, default=0.01)
    parser.add_argument("--lambda_kl_floor", type=float, default=20.0)
    parser.add_argument("--kl_floor", type=float, default=0.01)
    parser.add_argument("--kl_warmup_epochs", type=int, default=8)
    parser.add_argument("--lambda_ms_level", type=float, default=0.10)
    parser.add_argument("--lambda_ms_change", type=float, default=0.20)
    parser.add_argument("--lambda_change_corr", type=float, default=0.0)
    parser.add_argument("--lambda_change_spec", type=float, default=0.0)
    parser.add_argument("--lambda_level_corr", type=float, default=0.0)
    parser.add_argument("--lambda_level_spec", type=float, default=0.0)
    parser.add_argument("--structure_topk", type=int, default=8)
    parser.add_argument("--ms_samples", type=int, default=4)
    parser.add_argument("--terminal_weight", type=float, default=2.0)

    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--lr_schedule", type=str, default="cosine", choices=["cosine", "none"])

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
    tensors, H, W, D = make_dataset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        device=args.device,
    )
    train_hist, train_future, val_hist, val_future = tensors
    train_loader = DataLoader(TensorDataset(train_hist, train_future), batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"Train windows: {len(train_loader.dataset)}  Val windows: {len(val_loader.dataset)}")
    print(f"Grid: H={H}  W={W}  D={D}")

    cfg = LatentFutureTokenVAEConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=D,
        latent_dim=args.latent_dim,
        token_dim=args.token_dim,
        n_tokens=args.n_tokens,
        encoder_hidden=args.encoder_hidden,
        bottleneck_dim=args.bottleneck_dim,
        encoder_dropout=args.encoder_dropout,
        head_hidden=args.head_hidden,
        head_layers=args.head_layers,
        head_dropout=args.head_dropout,
        obs_hidden=args.obs_hidden,
        time_embed_dim=args.time_embed_dim,
        max_resid_ratio=args.max_resid_ratio,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        query_use_history=False,
        factor_use_history=False,
        resid_use_history=False,
        token_dependent_loadings=True,
        loading_delta_scale=0.5,
    )
    model = LatentFutureTokenVAE(cfg).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
        if args.lr_schedule == "cosine"
        else None
    )

    device_is_cuda = device.type == "cuda"
    use_bf16 = bool(args.bf16) and device_is_cuda

    def autocast_ctx():
        if use_bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return (
            torch.autocast(device_type="cuda", enabled=False)
            if device_is_cuda
            else torch.autocast(device_type="cpu", enabled=False)
        )

    history_path = Path(args.output_dir) / "training_history.json"
    history: list[dict] = []
    best_val = float("inf")
    best_path = Path(args.output_dir) / "best_model.pt"
    final_path = Path(args.output_dir) / "final_model.pt"
    global_step = 0

    def draw_posterior_samples(model: LatentFutureTokenVAE, history_norm: torch.Tensor, aux: dict[str, torch.Tensor], n_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        h = aux["h"]
        post_mu, post_logvar = aux["post_mu"], aux["post_logvar"]
        level_samples = []
        change_samples = []
        for _ in range(n_samples):
            z = model._sample_latents(post_mu, post_logvar)
            level_i, aux_i = model._decode(history_norm, h, z)
            level_samples.append(level_i)
            change_samples.append(aux_i["mean_change"])
        return torch.stack(level_samples, dim=1), torch.stack(change_samples, dim=1)

    def run_epoch(loader: DataLoader, train_mode: bool, epoch: int) -> dict[str, float]:
        nonlocal global_step
        model.train() if train_mode else model.eval()
        sums: dict[str, float] = {}
        nb = 0
        if train_mode:
            optimizer.zero_grad()
        kl_scale = args.lambda_kl * min(1.0, epoch / max(args.kl_warmup_epochs, 1))
        with torch.set_grad_enabled(train_mode):
            for batch_idx, (hist_01, fut_flat) in enumerate(loader):
                hist_01 = hist_01.to(device, non_blocking=True)
                fut_flat = fut_flat.to(device, non_blocking=True)
                B = hist_01.shape[0]
                hist_norm = normalize_iv(hist_01).view(B, hist_01.shape[1], -1)
                future_norm = fut_flat
                with autocast_ctx():
                    pred, aux = model(hist_norm, future=future_norm)
                    ms_level_samples, ms_change_samples = draw_posterior_samples(model, hist_norm, aux, args.ms_samples)
                    loss, metrics = build_losses(
                        history_norm=hist_norm,
                        future_norm=future_norm,
                        pred_level=pred,
                        aux=aux,
                        ms_level_samples=ms_level_samples,
                        ms_change_samples=ms_change_samples,
                        lambda_change_corr=args.lambda_change_corr,
                        lambda_change_spec=args.lambda_change_spec,
                        lambda_level_corr=args.lambda_level_corr,
                        lambda_level_spec=args.lambda_level_spec,
                        structure_topk=args.structure_topk,
                        lambda_level=args.lambda_level,
                        lambda_change=args.lambda_change,
                        lambda_jump=args.lambda_jump,
                        lambda_terminal=args.lambda_terminal,
                        lambda_resid=args.lambda_resid,
                        lambda_kl=kl_scale,
                        lambda_kl_floor=args.lambda_kl_floor,
                        kl_floor=args.kl_floor,
                        lambda_ms_level=args.lambda_ms_level,
                        lambda_ms_change=args.lambda_ms_change,
                        terminal_weight=args.terminal_weight,
                    )
                if train_mode:
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"Non-finite loss at epoch {epoch} batch {batch_idx}")
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
                    optimizer.zero_grad()
                for k, v in metrics.items():
                    sums[k] = sums.get(k, 0.0) + (v.item() if torch.is_tensor(v) else float(v))
                nb += 1
                global_step += 1
                if train_mode and global_step % args.log_every_n_steps == 0:
                    print(
                        f"  [ep {epoch} step {global_step}] "
                        f"level={metrics['level_loss'].item():.4f} "
                        f"chg={metrics['change_loss'].item():.4f} "
                        f"msL={metrics['ms_level_loss'].item():.4f} "
                        f"msC={metrics['ms_change_loss'].item():.4f} "
                        f"cCorr={metrics['change_corr_loss'].item():.4f} "
                        f"lCorr={metrics['level_corr_loss'].item():.4f} "
                        f"kl={metrics['kl'].item():.4f} "
                        f"top1={metrics['token_top1'].item():.4f} "
                        f"H={metrics['attn_entropy'].item():.4f} "
                        f"loss={metrics['total'].item():.4f}"
                    )
        avg = {k: v / max(nb, 1) for k, v in sums.items()}
        avg["kl_scale"] = kl_scale
        return avg

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True, epoch=epoch)
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            val_avg = run_epoch(val_loader, train_mode=False, epoch=epoch)
        dt = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train_level={train_avg.get('level_loss', 0):.4f}  "
            f"train_chg={train_avg.get('change_loss', 0):.4f}  "
            f"train_msC={train_avg.get('ms_change_loss', 0):.4f}  "
            f"train_cCorr={train_avg.get('change_corr_loss', 0):.4f}  "
            f"train_kl={train_avg.get('kl', 0):.4f}  "
            f"val_level={val_avg.get('level_loss', 0):.4f}  "
            f"val_chg={val_avg.get('change_loss', 0):.4f}  "
            f"val_msC={val_avg.get('ms_change_loss', 0):.4f}  "
            f"val_cCorr={val_avg.get('change_corr_loss', 0):.4f}  "
            f"val_jump={val_avg.get('jump_loss', 0):.4f}  "
            f"val_kl={val_avg.get('kl', 0):.4f}  "
            f"val_top1={val_avg.get('token_top1', 0):.4f}  "
            f"val_H={val_avg.get('attn_entropy', 0):.4f}  "
            f"val_total={val_avg.get('total', 0):.4f}  "
            f"time={dt:.1f}s"
        )
        rec = {"epoch": epoch, "train": train_avg, "val": val_avg, "lr": float(optimizer.param_groups[0]["lr"]), "time_sec": dt}
        history.append(rec)
        history_path.write_text(json.dumps(history, indent=2))
        if val_avg["total"] < best_val:
            best_val = val_avg["total"]
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_total": best_val,
                    "config": config_to_dict(cfg),
                },
                best_path,
            )

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_total": best_val,
            "config": config_to_dict(cfg),
        },
        final_path,
    )
    print(f"Saved final checkpoint to {final_path}")


if __name__ == "__main__":
    main()
