#!/usr/bin/env python
"""476a: learned future-path latent manifold plus conditional latent flow."""

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

from diffusion.block_ar.latent_path_manifold_flow import (  # noqa: E402
    LatentPathManifoldFlow,
    LatentPathManifoldFlowConfig,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402
from experiments.backfill.block_ar.train_330a_causal_future_memory_transition_flow import (  # noqa: E402
    make_dataset,
)
from experiments.backfill.block_ar.train_340a_empirical_normal_score_causal_memory_transition_flow import (  # noqa: E402
    compute_shared_level_quantiles,
)


def _run_epoch(
    model: LatentPathManifoldFlow,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    stage: str,
    max_batches: int,
    clip_grad: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    sums: dict[str, float] = {}
    n_batches = 0
    for hist_01, fut_01 in loader:
        if max_batches > 0 and n_batches >= max_batches:
            break
        hist_norm = normalize_iv(hist_01.to(device, non_blocking=True)).view(
            hist_01.shape[0],
            hist_01.shape[1],
            -1,
        )
        fut_norm = normalize_iv(fut_01.to(device, non_blocking=True)).view(
            fut_01.shape[0],
            fut_01.shape[1],
            -1,
        )
        with torch.set_grad_enabled(train_mode):
            if stage == "ae":
                loss, metrics = model.autoencoder_loss(fut_norm)
            elif stage == "flow":
                loss, metrics = model.flow_matching_loss(hist_norm, fut_norm)
            else:
                raise ValueError(f"unknown stage {stage}")
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        clip_grad,
                    )
                optimizer.step()
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value.item())
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent_dim", type=int, default=96)
    parser.add_argument("--ae_hidden", type=int, default=512)
    parser.add_argument("--ae_layers", type=int, default=3)
    parser.add_argument("--history_hidden", type=int, default=160)
    parser.add_argument("--history_layers", type=int, default=2)
    parser.add_argument("--flow_hidden", type=int, default=384)
    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--time_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--n_quantiles", type=int, default=401)
    parser.add_argument("--cdf_eps", type=float, default=1e-4)
    parser.add_argument("--prefix_feature_mode", choices=["basic", "scale"], default="scale")
    parser.add_argument("--flow_steps", type=int, default=32)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--max_sample_chunk", type=int, default=8)
    parser.add_argument("--recon_change_weight", type=float, default=0.5)

    parser.add_argument("--ae_epochs", type=int, default=60)
    parser.add_argument("--flow_epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--ae_lr", type=float, default=5e-4)
    parser.add_argument("--flow_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)

    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    tensors, h, w, d = make_dataset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        device=device,
    )
    train_hist, train_future, val_hist, val_future = tensors
    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    cfg = LatentPathManifoldFlowConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=d,
        latent_dim=args.latent_dim,
        ae_hidden=args.ae_hidden,
        ae_layers=args.ae_layers,
        history_hidden=args.history_hidden,
        history_layers=args.history_layers,
        flow_hidden=args.flow_hidden,
        flow_layers=args.flow_layers,
        time_dim=args.time_dim,
        dropout=args.dropout,
        n_quantiles=args.n_quantiles,
        cdf_eps=args.cdf_eps,
        prefix_feature_mode=args.prefix_feature_mode,
        flow_steps=args.flow_steps,
        sample_temperature=args.sample_temperature,
        max_sample_chunk=args.max_sample_chunk,
        recon_change_weight=args.recon_change_weight,
    )
    model = LatentPathManifoldFlow(cfg).to(device)
    quantiles, quantile_levels = compute_shared_level_quantiles(
        train_hist,
        train_future,
        n_quantiles=args.n_quantiles,
    )
    model.set_empirical_quantiles(quantiles, quantile_levels)

    print(f"Train windows: {len(train_loader.dataset)}  Val windows: {len(val_loader.dataset)}")
    print(f"Grid: H={h} W={w} D={d}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    records: list[dict[str, float | str | int]] = []

    ae_optimizer = torch.optim.AdamW(
        list(model.future_encoder.parameters()) + list(model.future_decoder.parameters()),
        lr=args.ae_lr,
        weight_decay=args.weight_decay,
    )
    ae_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(ae_optimizer, T_max=args.ae_epochs)
    best_ae = float("inf")
    best_ae_epoch = -1
    for epoch in range(1, args.ae_epochs + 1):
        t0 = time.time()
        train_avg = _run_epoch(
            model,
            train_loader,
            ae_optimizer,
            device,
            "ae",
            args.max_train_batches,
            args.clip_grad,
        )
        val_avg = _run_epoch(
            model,
            val_loader,
            None,
            device,
            "ae",
            args.max_val_batches,
            args.clip_grad,
        )
        ae_scheduler.step()
        rec = {
            "stage": "ae",
            "epoch": epoch,
            "train_total": train_avg["total"],
            "val_total": val_avg["total"],
            "val_recon_level_mse": val_avg["recon_level_mse"],
            "val_recon_change_mse": val_avg["recon_change_mse"],
            "val_latent_std": val_avg["latent_std"],
            "val_target_score_std": val_avg["target_score_std"],
            "val_recon_score_std": val_avg["recon_score_std"],
            "lr": ae_optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ae {epoch:03d}] train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} level={rec['val_recon_level_mse']:.5f} "
            f"chg={rec['val_recon_change_mse']:.5f} "
            f"std={rec['val_recon_score_std']:.3f}/{rec['val_target_score_std']:.3f} "
            f"latent={rec['val_latent_std']:.3f} lr={rec['lr']:.2e} "
            f"time={rec['sec']:.1f}s"
        )
        if val_avg["total"] < best_ae:
            best_ae = val_avg["total"]
            best_ae_epoch = epoch
            save_checkpoint(str(out_dir / "best_ae_model.pt"), model, cfg, epoch, best_ae, "ae")

    model.freeze_autoencoder()
    flow_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.flow_lr,
        weight_decay=args.weight_decay,
    )
    flow_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(flow_optimizer, T_max=args.flow_epochs)
    best_flow = float("inf")
    best_flow_epoch = -1
    for epoch in range(1, args.flow_epochs + 1):
        t0 = time.time()
        train_avg = _run_epoch(
            model,
            train_loader,
            flow_optimizer,
            device,
            "flow",
            args.max_train_batches,
            args.clip_grad,
        )
        val_avg = _run_epoch(
            model,
            val_loader,
            None,
            device,
            "flow",
            args.max_val_batches,
            args.clip_grad,
        )
        flow_scheduler.step()
        rec = {
            "stage": "flow",
            "epoch": epoch,
            "train_total": train_avg["total"],
            "val_total": val_avg["total"],
            "val_latent_std": val_avg["latent_std"],
            "val_target_velocity_std": val_avg["target_velocity_std"],
            "val_context_abs": val_avg["context_abs"],
            "lr": flow_optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[flow {epoch:03d}] train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} latent={rec['val_latent_std']:.3f} "
            f"vel={rec['val_target_velocity_std']:.3f} ctx={rec['val_context_abs']:.3f} "
            f"lr={rec['lr']:.2e} time={rec['sec']:.1f}s"
        )
        if val_avg["total"] < best_flow:
            best_flow = val_avg["total"]
            best_flow_epoch = epoch
            save_checkpoint(
                str(out_dir / "best_model.pt"),
                model,
                cfg,
                epoch,
                best_flow,
                "flow",
            )

    save_checkpoint(str(out_dir / "final_model.pt"), model, cfg, args.flow_epochs, best_flow, "flow")
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {
        "best_ae_epoch": best_ae_epoch,
        "best_ae_val_total": best_ae,
        "best_flow_epoch": best_flow_epoch,
        "best_flow_val_total": best_flow,
        "n_train": len(train_loader.dataset),
        "n_val": len(val_loader.dataset),
        "config": vars(args),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
