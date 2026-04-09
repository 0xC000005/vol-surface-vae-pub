#!/usr/bin/env python
"""
212l: 212k ablation with raw linear delta output, no tanh, no clipping.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    build_one_step_windows,
    make_serializable,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    build_history_features,
)
from experiments.backfill.block_ar.train_212k_h1_deterministic_direct_delta_mse import (
    evaluate_point_mse,
)


class DeterministicDirectDeltaRawMSEModel(nn.Module):
    def __init__(
        self,
        n_cells: int = 25,
        history_feat_dim: int = 50,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        head_hidden: int = 128,
        delta_scale: torch.Tensor | None = None,
    ):
        super().__init__()
        self.n_cells = n_cells
        self.hidden_dim = hidden_dim
        self.gru = nn.GRU(
            input_size=history_feat_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, head_hidden),
            nn.SiLU(),
            nn.Linear(head_hidden, n_cells),
        )
        if delta_scale is None:
            delta_scale = torch.ones(n_cells, dtype=torch.float32) * 0.05
        self.register_buffer("delta_scale", delta_scale.float())
        self._init_parameters()

    def _init_parameters(self) -> None:
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if name.endswith("head.2"):
                    continue
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def encode(self, history_01: torch.Tensor) -> torch.Tensor:
        feat = build_history_features(history_01, self.delta_scale)
        _out, h_n = self.gru(feat)
        return h_n[-1]

    def predict_delta(self, history_01: torch.Tensor) -> torch.Tensor:
        state = self.encode(history_01)
        raw = self.head(state)
        return raw

    def predict_next_iv(self, history_01: torch.Tensor) -> torch.Tensor:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells)
        return prev + self.predict_delta(history_01)

    def training_loss(self, history_01: torch.Tensor, target_01: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells)
        target_delta = target_01 - prev
        pred_delta = self.predict_delta(history_01)
        norm = self.delta_scale.view(1, self.n_cells).clamp_min(1e-6)
        per_window_mse = (((pred_delta - target_delta) / norm) ** 2).mean(dim=1)
        mean_mse = per_window_mse.mean()
        metrics = {
            "mean_mse": mean_mse.detach(),
            "loss": mean_mse.detach(),
            "mae": (pred_delta - target_delta).abs().mean().detach(),
            "rmse": ((pred_delta - target_delta) ** 2).mean().sqrt().detach(),
            "pred_abs_mean": pred_delta.abs().mean().detach(),
        }
        return mean_mse, metrics


def load_model(checkpoint_path: str, device: torch.device) -> tuple[DeterministicDirectDeltaRawMSEModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "minimal_h1_deterministic_direct_delta_212l":
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    model = DeterministicDirectDeltaRawMSEModel(
        n_cells=cfg["n_cells"],
        history_feat_dim=cfg["history_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        gru_layers=cfg["gru_layers"],
        gru_dropout=cfg["gru_dropout"],
        head_hidden=cfg["head_hidden"],
        delta_scale=torch.tensor(cfg["delta_scale"], dtype=torch.float32),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="212l H=1 deterministic direct delta raw linear + MSE only")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=4010)
    parser.add_argument("--max_val_windows", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--gru_layers", type=int, default=2)
    parser.add_argument("--gru_dropout", type=float, default=0.1)
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = np.load(args.data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)

    max_train_idx = args.test_start - args.history_len - 30
    train_indices = np.arange(0, max_train_idx - args.val_size)
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    train_indices = train_indices[: args.max_train_windows]
    val_indices = val_indices[: args.max_val_windows]

    train_hist, train_target = build_one_step_windows(train_indices, surf_tensor, args.history_len)
    val_hist, val_target = build_one_step_windows(val_indices, surf_tensor, args.history_len)

    train_delta_np = np.diff(surfaces[: args.test_start].reshape(-1, 25), axis=0)
    delta_scale = np.quantile(np.abs(train_delta_np), 0.99, axis=0).astype(np.float32)
    delta_scale = np.clip(delta_scale, 1e-3, None)
    q95_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.95))
    q99_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.99))

    train_loader = DataLoader(TensorDataset(train_hist, train_target), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_target), batch_size=args.batch_size, shuffle=False)

    model = DeterministicDirectDeltaRawMSEModel(
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        head_hidden=args.head_hidden,
        delta_scale=torch.from_numpy(delta_scale),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("212l H=1 deterministic direct delta raw linear + MSE only")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(f"  delta_scale median={float(np.median(delta_scale)):.6f}")
    print(f"  q95={q95_threshold:.5f} q99={q99_threshold:.5f}")

    history: list[dict[str, Any]] = []
    best_score = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = {
            "loss": 0.0,
            "mean_mse": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "pred_abs_mean": 0.0,
        }
        count = 0

        for history_01, target_01 in train_loader:
            loss, metrics = model.training_loss(history_01=history_01, target_01=target_01)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["mean_mse"] += float(metrics["mean_mse"].item()) * batch
            running["mae"] += float(metrics["mae"].item()) * batch
            running["rmse"] += float(metrics["rmse"].item()) * batch
            running["pred_abs_mean"] += float(metrics["pred_abs_mean"].item()) * batch
            count += batch

        train_metrics = {f"train_{k}": v / max(count, 1) for k, v in running.items()}
        val_metrics = evaluate_point_mse(
            model,
            val_loader,
            q95_threshold=q95_threshold,
            q99_threshold=q99_threshold,
        )
        score = float(val_metrics["val_loss"])

        record = {
            "epoch": epoch,
            "elapsed_sec": time.time() - t0,
            **train_metrics,
            **val_metrics,
            "selection_score": score,
        }
        history.append(make_serializable(record))

        payload = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "config": {
                "type": "minimal_h1_deterministic_direct_delta_212l",
                "n_cells": 25,
                "history_feat_dim": 50,
                "hidden_dim": args.hidden_dim,
                "gru_layers": args.gru_layers,
                "gru_dropout": args.gru_dropout,
                "head_hidden": args.head_hidden,
                "delta_scale": delta_scale.tolist(),
                "seed": args.seed,
                "train_windows": int(train_hist.shape[0]),
                "val_windows": int(val_hist.shape[0]),
            },
        }

        torch.save(payload, out_dir / "final_model.pt")
        if score < best_score:
            best_score = score
            torch.save(payload, out_dir / "best_model.pt")

        (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
        print(
            f"[{epoch:02d}/{args.epochs}] "
            f"loss={train_metrics['train_loss']:.4f}  "
            f"MSE={train_metrics['train_mean_mse']:.4f}  "
            f"val_loss={val_metrics['val_loss']:.4f}  "
            f"mae={val_metrics['val_mae']:.4f}  "
            f"q95mae={val_metrics['val_q95_mae']:.4f}  "
            f"q99mae={val_metrics['val_q99_mae']:.4f}  "
            f"sign={val_metrics['val_sign_acc']:.3f}"
        )

    print(f"Finished. Best selection score={best_score:.4f}")
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
