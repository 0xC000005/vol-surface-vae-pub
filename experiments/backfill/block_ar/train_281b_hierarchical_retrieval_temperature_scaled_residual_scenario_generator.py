#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.hierarchical_retrieval_temperature_scaled_residual_scenario_generator import (
    HierarchicalTemperatureScaledResidualRetrievalScenarioGenerator,
)
from diffusion.block_ar.deterministic_learned_retrieval_rich_target_backbone import (
    LearnedRetrievalRichTargetBackbone,
    LearnedRetrievalRichTargetConfig,
)
from experiments.backfill.block_ar._rollout_220_utils import build_rollout_windows


def make_model(base_checkpoint: str, device: torch.device, args: argparse.Namespace):
    payload = torch.load(base_checkpoint, map_location=device, weights_only=False)
    base_cfg = LearnedRetrievalRichTargetConfig(**payload["config"])
    base_model = LearnedRetrievalRichTargetBackbone(
        base_cfg,
        library_future_embeddings=payload["library_future_embeddings"].to(device),
        library_last_level_01=payload["library_last_level_01"].to(device),
        library_future_01=payload["library_future_01"].to(device),
    )
    base_model.load_state_dict(payload["model_state_dict"], strict=False)
    base_model.to(device).eval()
    model = HierarchicalTemperatureScaledResidualRetrievalScenarioGenerator(
        base_model=base_model,
        top_k=args.top_k,
        tau_min=args.tau_min,
        tau_max=args.tau_max,
        min_scale=args.min_scale,
        hidden_dim=args.hidden_dim,
    )
    model.to(device)
    return model, payload


def epoch_pass(
    model: HierarchicalTemperatureScaledResidualRetrievalScenarioGenerator,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    total_loss = 0.0
    total_tau = 0.0
    total_scale = 0.0
    total_dist = 0.0
    total_n = 0
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    for history_norm, future_01 in loader:
        history_norm = history_norm.to(device)
        future_01 = future_01.to(device)
        loss, metrics = model.training_loss(history_norm, future_01)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        batch_n = int(history_norm.shape[0])
        total_loss += float(loss.detach()) * batch_n
        total_tau += float(metrics["tau_mean"].detach()) * batch_n
        total_scale += float(metrics["scale_mean"].detach()) * batch_n
        total_dist += float(metrics["dist_mean"].detach()) * batch_n
        total_n += batch_n
    return {
        "loss": total_loss / max(total_n, 1),
        "tau_mean": total_tau / max(total_n, 1),
        "scale_mean": total_scale / max(total_n, 1),
        "dist_mean": total_dist / max(total_n, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="281b-v0 residual temperature plus scale hierarchy")
    parser.add_argument("--base_checkpoint", type=str, default="models/backfill/277d_v0_s42/best_model.pt")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--top_k", type=int, default=16)
    parser.add_argument("--tau_min", type=float, default=0.05)
    parser.add_argument("--tau_max", type=float, default=1.5)
    parser.add_argument("--min_scale", type=float, default=0.25)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=None,
        device=device,
        split="train",
    )
    val_batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=192,
        device=device,
        split="val",
    )

    model, payload = make_model(args.base_checkpoint, device, args)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.lr)

    train_loader = DataLoader(
        TensorDataset(train_batch.history_norm, train_batch.future_01),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_batch.history_norm, val_batch.future_01),
        batch_size=args.batch_size,
        shuffle=False,
    )

    history = []
    best_val = float("inf")
    best_state = None
    best_epoch = -1
    for epoch in range(1, args.epochs + 1):
        train_metrics = epoch_pass(model, train_loader, optimizer)
        with torch.no_grad():
            val_metrics = epoch_pass(model, val_loader, optimizer=None)
        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_tau_mean": train_metrics["tau_mean"],
            "train_scale_mean": train_metrics["scale_mean"],
            "train_dist_mean": train_metrics["dist_mean"],
            "val_loss": val_metrics["loss"],
            "val_tau_mean": val_metrics["tau_mean"],
            "val_scale_mean": val_metrics["scale_mean"],
            "val_dist_mean": val_metrics["dist_mean"],
        }
        history.append(record)
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.head.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No best state captured during training")

    save_payload = {
        "base_config": payload["config"],
        "model_state_dict": payload["model_state_dict"],
        "library_future_embeddings": payload["library_future_embeddings"],
        "library_last_level_01": payload["library_last_level_01"],
        "library_future_01": payload["library_future_01"],
        "top_k": args.top_k,
        "tau_min": args.tau_min,
        "tau_max": args.tau_max,
        "min_scale": args.min_scale,
        "hidden_dim": args.hidden_dim,
        "head_state_dict": best_state,
        "source_checkpoint": args.base_checkpoint,
        "best_epoch": best_epoch,
        "best_val_loss": best_val,
    }
    torch.save(save_payload, out_dir / "best_model.pt")
    torch.save(save_payload, out_dir / "final_model.pt")
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps({"best_epoch": best_epoch, "best_val_loss": best_val, "history_len": len(history)}, indent=2))


if __name__ == "__main__":
    main()
