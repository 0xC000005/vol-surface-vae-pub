#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.deterministic_learned_retrieval_rich_target_backbone import (
    load_model as load_277d_model,
)
from diffusion.block_ar.probabilistic_structural_embedding_center_model import (
    StructuralEmbeddingCenterConfig,
    StructuralEmbeddingDensity,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_277d_deterministic_learned_retrieval_rich_target_backbone import (
    build_future_representation,
    build_library_future_embeddings,
    build_windows,
)


class StructuralEmbeddingDataset(Dataset):
    def __init__(self, history_embedding: torch.Tensor, future_embedding: torch.Tensor):
        self.history_embedding = history_embedding
        self.future_embedding = future_embedding

    def __len__(self) -> int:
        return int(self.history_embedding.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "history_embedding": self.history_embedding[idx],
            "future_embedding": self.future_embedding[idx],
        }


@torch.no_grad()
def build_history_embeddings(
    model: torch.nn.Module,
    history_norm: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    outs: list[torch.Tensor] = []
    for start in range(0, history_norm.shape[0], batch_size):
        chunk = history_norm[start : start + batch_size].to(device)
        outs.append(model.encode_history(chunk).cpu())
    return torch.cat(outs, dim=0)


def run_epoch(
    model: StructuralEmbeddingDensity,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    clip_grad: float,
) -> dict[str, float]:
    train_mode = optimizer is not None
    model.train(train_mode)
    sums: dict[str, float] = {}
    n_batches = 0
    for batch in loader:
        history_embedding = batch["history_embedding"].to(device)
        future_embedding = batch["future_embedding"].to(device)
        with torch.set_grad_enabled(train_mode):
            loss, metrics = model.training_loss(history_embedding, future_embedding)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                optimizer.step()
        for key, value in metrics.items():
            sums[key] = sums.get(key, 0.0) + float(value.item())
        n_batches += 1
    return {key: value / max(n_batches, 1) for key, value in sums.items()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "298a-v0 conditional density over frozen 277d structural future embeddings, "
            "decoded through the training future library"
        )
    )
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument(
        "--backbone_checkpoint",
        type=str,
        default="models/backfill/277d_v0_s42/best_model.pt",
    )

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--scale_floor", type=float, default=0.03)
    parser.add_argument("--scale_ceiling", type=float, default=0.80)
    parser.add_argument("--sample_scale_mult", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    windows = build_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        device=str(device),
    )

    backbone, _ = load_277d_model(args.backbone_checkpoint, device)
    backbone.eval()

    print("Building frozen 277d structural embeddings...")
    train_history_embeddings = build_history_embeddings(
        backbone,
        windows["train_hist_norm"],
        batch_size=args.batch_size,
        device=device,
    )
    val_history_embeddings = build_history_embeddings(
        backbone,
        windows["val_hist_norm"],
        batch_size=args.batch_size,
        device=device,
    )
    train_future_embeddings = build_library_future_embeddings(
        backbone,
        windows["train_future_repr"],
        batch_size=args.batch_size,
        device=device,
    )
    val_future_embeddings = build_library_future_embeddings(
        backbone,
        windows["val_future_repr"],
        batch_size=args.batch_size,
        device=device,
    )

    torch.save(train_history_embeddings, out_dir / "train_history_embeddings.pt")
    torch.save(train_future_embeddings, out_dir / "train_future_embeddings.pt")
    torch.save(val_history_embeddings, out_dir / "val_history_embeddings.pt")
    torch.save(val_future_embeddings, out_dir / "val_future_embeddings.pt")

    cfg = StructuralEmbeddingCenterConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=windows["train_hist_norm"].shape[-1],
        embed_dim=train_history_embeddings.shape[-1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        scale_floor=args.scale_floor,
        scale_ceiling=args.scale_ceiling,
        sample_scale_mult=args.sample_scale_mult,
    )
    model = StructuralEmbeddingDensity(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_loader = DataLoader(
        StructuralEmbeddingDataset(train_history_embeddings, train_future_embeddings),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        StructuralEmbeddingDataset(val_history_embeddings, val_future_embeddings),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    best_val = float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []
    best_path = out_dir / "best_model.pt"
    final_path = out_dir / "final_model.pt"

    print(f"Train windows: {len(train_loader.dataset)}  Val windows: {len(val_loader.dataset)}")
    print(f"Embedding dim: {cfg.embed_dim}; params: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.clip_grad)
        val_metrics = run_epoch(model, val_loader, None, device, args.clip_grad)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "train_scale_mean": train_metrics["scale_mean"],
            "train_z_abs_mean": train_metrics["z_abs_mean"],
            "train_mu_target_cos": train_metrics["mu_target_cos"],
            "val_total": val_metrics["total"],
            "val_scale_mean": val_metrics["scale_mean"],
            "val_z_abs_mean": val_metrics["z_abs_mean"],
            "val_mu_target_cos": val_metrics["mu_target_cos"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        history.append(row)
        print(json.dumps(row))
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            best_epoch = epoch
            save_checkpoint(
                str(best_path),
                model,
                cfg,
                epoch,
                best_val,
                backbone_checkpoint_path=args.backbone_checkpoint,
                library_future_embeddings=train_future_embeddings,
                library_last_level_01=windows["train_last_level"],
                library_future_01=windows["train_future_01"],
            )

    save_checkpoint(
        str(final_path),
        model,
        cfg,
        args.epochs,
        best_val,
        backbone_checkpoint_path=args.backbone_checkpoint,
        library_future_embeddings=train_future_embeddings,
        library_last_level_01=windows["train_last_level"],
        library_future_01=windows["train_future_01"],
    )
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    summary = {
        "config": asdict(cfg),
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "backbone_checkpoint": args.backbone_checkpoint,
        "n_train": int(train_history_embeddings.shape[0]),
        "n_val": int(val_history_embeddings.shape[0]),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
