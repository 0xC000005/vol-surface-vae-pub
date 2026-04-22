#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.deterministic_learned_retrieval_change_anchor_backbone import (
    LearnedRetrievalChangeAnchorBackbone,
    LearnedRetrievalChangeAnchorConfig,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)


class HistoryFutureChangeDataset(Dataset):
    def __init__(self, history_norm: torch.Tensor, future_changes_01: torch.Tensor):
        self.history_norm = history_norm
        self.future_changes_01 = future_changes_01

    def __len__(self) -> int:
        return int(self.history_norm.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "history_norm": self.history_norm[idx],
            "future_changes_01": self.future_changes_01[idx],
        }


def build_future_changes(train_hist: torch.Tensor, future_01: torch.Tensor) -> torch.Tensor:
    if train_hist.ndim == 4:
        train_hist = train_hist.view(train_hist.shape[0], train_hist.shape[1], -1)
    if future_01.ndim == 4:
        future_01 = future_01.view(future_01.shape[0], future_01.shape[1], -1)
    prev = torch.cat([train_hist[:, -1:].contiguous(), future_01[:, :-1]], dim=1)
    return future_01 - prev


def build_windows(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    device: str,
):
    data = np.load(data_path)
    surfaces = data["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    max_train_idx = test_start - history_len - future_len
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, history_len, future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, history_len, future_len)
    train_hist_norm = normalize_iv(train_hist).view(train_hist.shape[0], train_hist.shape[1], -1)
    val_hist_norm = normalize_iv(val_hist).view(val_hist.shape[0], val_hist.shape[1], -1)
    train_changes = build_future_changes(train_hist, train_future).view(train_future.shape[0], train_future.shape[1], -1)
    val_changes = build_future_changes(val_hist, val_future).view(val_future.shape[0], val_future.shape[1], -1)
    if train_hist.ndim == 3:
        train_last_level = train_hist[:, -1].view(train_hist.shape[0], 5, 5)
    else:
        train_last_level = train_hist[:, -1]
    if val_hist.ndim == 3:
        val_last_level = val_hist[:, -1].view(val_hist.shape[0], 5, 5)
    else:
        val_last_level = val_hist[:, -1]
    if train_future.ndim == 3:
        train_future_01 = train_future.view(train_future.shape[0], train_future.shape[1], 5, 5)
    else:
        train_future_01 = train_future
    if val_future.ndim == 3:
        val_future_01 = val_future.view(val_future.shape[0], val_future.shape[1], 5, 5)
    else:
        val_future_01 = val_future
    return {
        "train_hist_norm": train_hist_norm,
        "train_last_level": train_last_level,
        "train_future_01": train_future_01,
        "train_changes_01": train_changes,
        "val_hist_norm": val_hist_norm,
        "val_last_level": val_last_level,
        "val_future_01": val_future_01,
        "val_changes_01": val_changes,
    }


def symmetric_infonce_loss(logits: torch.Tensor) -> tuple[torch.Tensor, float]:
    labels = torch.arange(logits.shape[0], device=logits.device)
    loss_hf = F.cross_entropy(logits, labels)
    loss_fh = F.cross_entropy(logits.transpose(0, 1), labels)
    loss = 0.5 * (loss_hf + loss_fh)
    acc = (logits.argmax(dim=1) == labels).float().mean().item()
    return loss, acc


def run_epoch(
    model: LearnedRetrievalChangeAnchorBackbone,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    total_loss = 0.0
    total_acc = 0.0
    total_count = 0
    for batch in loader:
        history_norm = batch["history_norm"].to(device)
        future_changes_01 = batch["future_changes_01"].to(device)
        logits, _, _ = model.contrastive_logits(history_norm, future_changes_01)
        loss, acc = symmetric_infonce_loss(logits)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        bs = history_norm.shape[0]
        total_loss += loss.item() * bs
        total_acc += acc * bs
        total_count += bs
    return {
        "loss": total_loss / max(total_count, 1),
        "top1_acc": total_acc / max(total_count, 1),
    }


@torch.no_grad()
def build_library_future_embeddings(
    model: LearnedRetrievalChangeAnchorBackbone,
    future_changes_01: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    outs: list[torch.Tensor] = []
    for start in range(0, future_changes_01.shape[0], batch_size):
        chunk = future_changes_01[start : start + batch_size].to(device)
        outs.append(model.encode_future_changes(chunk).cpu())
    return torch.cat(outs, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="277c-v0 learned retrieval backbone for deterministic Stage A")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--embed_dim", type=int, default=96)
    parser.add_argument("--rnn_hidden", type=int, default=128)
    parser.add_argument("--rnn_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

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

    cfg = LearnedRetrievalChangeAnchorConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=windows["train_hist_norm"].shape[-1],
        embed_dim=args.embed_dim,
        rnn_hidden=args.rnn_hidden,
        rnn_layers=args.rnn_layers,
        dropout=args.dropout,
        temperature=args.temperature,
    )
    model = LearnedRetrievalChangeAnchorBackbone(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    train_loader = DataLoader(
        HistoryFutureChangeDataset(windows["train_hist_norm"], windows["train_changes_01"]),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        HistoryFutureChangeDataset(windows["val_hist_norm"], windows["val_changes_01"]),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    best_val = float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []
    best_payload: dict | None = None

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device)
        val_metrics = run_epoch(model, val_loader, None, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1_acc": train_metrics["top1_acc"],
            "val_loss": val_metrics["loss"],
            "val_top1_acc": val_metrics["top1_acc"],
        }
        history.append(row)
        print(json.dumps(row))

        library_future_embeddings = build_library_future_embeddings(
            model,
            windows["train_changes_01"],
            batch_size=args.batch_size,
            device=device,
        )
        extra = {
            "best_epoch": epoch,
            "val_loss": val_metrics["loss"],
            "val_top1_acc": val_metrics["top1_acc"],
        }
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            best_payload = {
                "library_future_embeddings": library_future_embeddings,
                "extra": extra,
            }
            save_checkpoint(
                str(out_dir / "best_model.pt"),
                cfg,
                model,
                library_future_embeddings=library_future_embeddings,
                library_last_level_01=windows["train_last_level"],
                library_future_01=windows["train_future_01"],
                extra=extra,
            )

    final_library_future_embeddings = build_library_future_embeddings(
        model,
        windows["train_changes_01"],
        batch_size=args.batch_size,
        device=device,
    )
    save_checkpoint(
        str(out_dir / "final_model.pt"),
        cfg,
        model,
        library_future_embeddings=final_library_future_embeddings,
        library_last_level_01=windows["train_last_level"],
        library_future_01=windows["train_future_01"],
        extra={
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
        },
    )
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    summary = {
        "config": asdict(cfg),
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "n_train": int(windows["train_hist_norm"].shape[0]),
        "n_val": int(windows["val_hist_norm"].shape[0]),
    }
    if best_payload is not None:
        summary["best_val_top1_acc"] = best_payload["extra"]["val_top1_acc"]
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
