#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.deterministic_learned_retrieval_rich_target_backbone import (
    LearnedRetrievalRichTargetBackbone,
    LearnedRetrievalRichTargetConfig,
)
from diffusion.block_ar.hierarchical_retrieval_reweighted_scenario_generator import (
    HierarchicalReweightedRetrievalConfig,
    HierarchicalReweightedRetrievalScenarioGenerator,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    denormalize_iv,
    normalize_iv,
)
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows


class HistoryFutureDataset(Dataset):
    def __init__(self, history_norm: torch.Tensor, future_01: torch.Tensor):
        self.history_norm = history_norm
        self.future_01 = future_01

    def __len__(self) -> int:
        return int(self.history_norm.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "history_norm": self.history_norm[idx],
            "future_01": self.future_01[idx],
        }


def build_windows(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    device: str,
) -> dict[str, torch.Tensor]:
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
    train_future_01 = train_future.view(train_future.shape[0], train_future.shape[1], 5, 5)
    val_future_01 = val_future.view(val_future.shape[0], val_future.shape[1], 5, 5)
    return {
        "train_hist_norm": train_hist_norm,
        "train_future_01": train_future_01,
        "val_hist_norm": val_hist_norm,
        "val_future_01": val_future_01,
    }


@torch.no_grad()
def build_topk_batch(
    model: HierarchicalReweightedRetrievalScenarioGenerator,
    history_norm: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    top_idx, top_scores, query_z = model.topk_candidates(history_norm)
    retrieved_future = model.library_future_01[top_idx]
    retrieved_prev = torch.cat(
        [model.library_last_level_01[top_idx].unsqueeze(2), retrieved_future[:, :, :-1]], dim=2
    )
    deltas = retrieved_future - retrieved_prev
    history_01 = denormalize_iv(history_norm).view(history_norm.shape[0], history_norm.shape[1], 5, 5)
    query_last = history_01[:, -1][:, None, None]
    anchored_candidates = torch.clamp(query_last + torch.cumsum(deltas, dim=2), 0.0, 1.0)

    center = model.base_model.sample_batched(
        history_norm,
        n_samples=1,
        n_steps=model.base_model.cfg.future_len,
        history_is_normalized=True,
    ).squeeze(1)
    half_centered = center.unsqueeze(1) + alpha * (anchored_candidates - center.unsqueeze(1))
    half_centered = torch.clamp(half_centered, 0.0, 1.0)
    return top_idx, top_scores, query_z, half_centered


def weighted_crps(
    candidates_flat: torch.Tensor,
    weights: torch.Tensor,
    target_flat: torch.Tensor,
) -> torch.Tensor:
    diff_to_target = torch.abs(candidates_flat - target_flat.unsqueeze(1))
    term1 = (weights.unsqueeze(-1) * diff_to_target).sum(dim=1)

    pairwise = torch.abs(candidates_flat.unsqueeze(2) - candidates_flat.unsqueeze(1))
    weight_outer = weights.unsqueeze(2) * weights.unsqueeze(1)
    term2 = 0.5 * (weight_outer.unsqueeze(-1) * pairwise).sum(dim=(1, 2))
    return (term1 - term2).mean()


def run_epoch(
    model: HierarchicalReweightedRetrievalScenarioGenerator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    alpha: float,
) -> dict[str, float]:
    train = optimizer is not None
    model.train(train)
    model.base_model.eval()
    total_loss = 0.0
    total_entropy = 0.0
    total_count = 0
    total_top1_mass = 0.0
    for batch in loader:
        history_norm = batch["history_norm"].to(device)
        future_01 = batch["future_01"].to(device)
        with torch.no_grad():
            top_idx, top_scores, query_z, candidates = build_topk_batch(model, history_norm, alpha)
        logits = model.candidate_logits(query_z, top_idx, top_scores)
        weights = torch.softmax(logits / model.cfg.sample_temperature, dim=-1)
        candidates_flat = candidates.view(candidates.shape[0], candidates.shape[1], -1)
        target_flat = future_01.view(future_01.shape[0], -1)
        loss = weighted_crps(candidates_flat, weights, target_flat)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.scorer.parameters(), 1.0)
            optimizer.step()
        bs = history_norm.shape[0]
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1).mean()
        total_loss += float(loss.detach()) * bs
        total_entropy += float(entropy.detach()) * bs
        total_top1_mass += float(weights.max(dim=-1).values.mean().detach()) * bs
        total_count += bs
    return {
        "loss": total_loss / max(total_count, 1),
        "entropy": total_entropy / max(total_count, 1),
        "top1_mass": total_top1_mass / max(total_count, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="283c-v0 half-centered anchored future-path reweighting trained with weighted CRPS")
    parser.add_argument("--base_checkpoint", type=str, default="models/backfill/277d_v0_s42/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--sample_temperature", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    base_payload = torch.load(args.base_checkpoint, map_location=device, weights_only=False)
    base_cfg = LearnedRetrievalRichTargetConfig(**base_payload["config"])
    base_model = LearnedRetrievalRichTargetBackbone(
        base_cfg,
        library_future_embeddings=base_payload["library_future_embeddings"].to(device),
        library_last_level_01=base_payload["library_last_level_01"].to(device),
        library_future_01=base_payload["library_future_01"].to(device),
    )
    base_model.load_state_dict(base_payload["model_state_dict"], strict=False)
    base_model.to(device).eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    cfg = HierarchicalReweightedRetrievalConfig(
        top_k=args.top_k,
        hidden_dim=args.hidden_dim,
        label_temperature=0.25,
        sample_temperature=args.sample_temperature,
    )
    model = HierarchicalReweightedRetrievalScenarioGenerator(base_model, cfg).to(device)
    optimizer = torch.optim.AdamW(model.scorer.parameters(), lr=args.lr, weight_decay=1e-4)

    windows = build_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        device=str(device),
    )
    train_loader = DataLoader(
        HistoryFutureDataset(windows["train_hist_norm"], windows["train_future_01"]),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        HistoryFutureDataset(windows["val_hist_norm"], windows["val_future_01"]),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    best_val = float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, device, args.alpha)
        val_metrics = run_epoch(model, val_loader, None, device, args.alpha)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_entropy": train_metrics["entropy"],
            "train_top1_mass": train_metrics["top1_mass"],
            "val_loss": val_metrics["loss"],
            "val_entropy": val_metrics["entropy"],
            "val_top1_mass": val_metrics["top1_mass"],
        }
        history.append(row)
        print(json.dumps(row))
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(
                str(out_dir / "best_model.pt"),
                model,
                extra={
                    "best_epoch": epoch,
                    "best_val_loss": val_metrics["loss"],
                    "source_checkpoint": args.base_checkpoint,
                    "training_objective": "weighted_crps_half_centered_anchored_support",
                    "alpha": args.alpha,
                },
            )

    save_checkpoint(
        str(out_dir / "final_model.pt"),
        model,
        extra={
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "source_checkpoint": args.base_checkpoint,
            "training_objective": "weighted_crps_half_centered_anchored_support",
            "alpha": args.alpha,
        },
    )
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
        "source_checkpoint": args.base_checkpoint,
        "alpha": args.alpha,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
