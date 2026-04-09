#!/usr/bin/env python
"""
212p: deterministic H=1 point forecaster with relative-move output and dual MSE loss.

Model outputs a per-cell simple-return-like relative move.
Training uses:
  - raw delta MSE
  - relative move MSE
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
from experiments.backfill.block_ar.train_212n_h1_deterministic_direct_delta_fully_raw_mse import (
    build_history_features_raw,
)


class DeterministicRelativeMoveDualMSEModel(nn.Module):
    def __init__(
        self,
        n_cells: int = 25,
        history_feat_dim: int = 50,
        hidden_dim: int = 128,
        gru_layers: int = 2,
        gru_dropout: float = 0.1,
        head_hidden: int = 128,
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
        feat = build_history_features_raw(history_01)
        _out, h_n = self.gru(feat)
        return h_n[-1]

    def predict_relative_move(self, history_01: torch.Tensor) -> torch.Tensor:
        state = self.encode(history_01)
        return self.head(state)

    def predict_delta(self, history_01: torch.Tensor) -> torch.Tensor:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells)
        rel_move = self.predict_relative_move(history_01)
        return prev * rel_move

    def predict_next_iv(self, history_01: torch.Tensor) -> torch.Tensor:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells)
        return prev + self.predict_delta(history_01)

    def training_loss(
        self,
        history_01: torch.Tensor,
        target_01: torch.Tensor,
        delta_loss_weight: float,
        return_loss_weight: float,
        prev_floor: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prev = history_01[:, -1].reshape(history_01.shape[0], self.n_cells)
        prev_safe = prev.clamp_min(prev_floor)
        target_delta = target_01 - prev
        target_return = target_delta / prev_safe

        pred_return = self.predict_relative_move(history_01)
        pred_delta = prev * pred_return

        per_window_delta_mse = ((pred_delta - target_delta) ** 2).mean(dim=1)
        per_window_return_mse = ((pred_return - target_return) ** 2).mean(dim=1)
        delta_mse = per_window_delta_mse.mean()
        return_mse = per_window_return_mse.mean()
        loss = delta_loss_weight * delta_mse + return_loss_weight * return_mse

        metrics = {
            "delta_mse": delta_mse.detach(),
            "return_mse": return_mse.detach(),
            "loss": loss.detach(),
            "mae": (pred_delta - target_delta).abs().mean().detach(),
            "rmse": ((pred_delta - target_delta) ** 2).mean().sqrt().detach(),
            "pred_abs_mean": pred_delta.abs().mean().detach(),
            "pred_abs_rel_move": pred_return.abs().mean().detach(),
        }
        return loss, metrics


@torch.no_grad()
def evaluate_point_dual_mse(
    model: DeterministicRelativeMoveDualMSEModel,
    loader: DataLoader,
    q95_threshold: float,
    q99_threshold: float,
    delta_loss_weight: float,
    return_loss_weight: float,
    prev_floor: float,
) -> dict[str, float]:
    model.eval()
    total = {
        "val_delta_mse": 0.0,
        "val_return_mse": 0.0,
        "val_loss": 0.0,
        "val_mae": 0.0,
        "val_rmse": 0.0,
        "val_pred_abs_mean": 0.0,
        "val_pred_abs_rel_move": 0.0,
        "val_sign_acc": 0.0,
    }
    q95_mae_sum = 0.0
    q99_mae_sum = 0.0
    q95_count = 0
    q99_count = 0
    total_count = 0

    for history_01, target_01 in loader:
        prev = history_01[:, -1].reshape(history_01.shape[0], model.n_cells)
        prev_safe = prev.clamp_min(prev_floor)
        target_delta = target_01 - prev
        target_return = target_delta / prev_safe
        pred_return = model.predict_relative_move(history_01)
        pred_delta = prev * pred_return

        per_window_delta_mse = ((pred_delta - target_delta) ** 2).mean(dim=1)
        per_window_return_mse = ((pred_return - target_return) ** 2).mean(dim=1)
        delta_mse = per_window_delta_mse.mean()
        return_mse = per_window_return_mse.mean()
        loss = delta_loss_weight * delta_mse + return_loss_weight * return_mse

        abs_err = (pred_delta - target_delta).abs()
        rmse = ((pred_delta - target_delta) ** 2).mean().sqrt()
        sign_acc = (torch.sign(pred_delta) == torch.sign(target_delta)).float().mean()
        target_abs = target_delta.abs()
        q95_mask = target_abs >= q95_threshold
        q99_mask = target_abs >= q99_threshold

        batch_size = history_01.shape[0]
        total["val_delta_mse"] += float(delta_mse.item()) * batch_size
        total["val_return_mse"] += float(return_mse.item()) * batch_size
        total["val_loss"] += float(loss.item()) * batch_size
        total["val_mae"] += float(abs_err.mean().item()) * batch_size
        total["val_rmse"] += float(rmse.item()) * batch_size
        total["val_pred_abs_mean"] += float(pred_delta.abs().mean().item()) * batch_size
        total["val_pred_abs_rel_move"] += float(pred_return.abs().mean().item()) * batch_size
        total["val_sign_acc"] += float(sign_acc.item()) * batch_size
        total_count += batch_size

        if q95_mask.any():
            q95_mae_sum += float(abs_err[q95_mask].mean().item()) * int(q95_mask.sum().item())
            q95_count += int(q95_mask.sum().item())
        if q99_mask.any():
            q99_mae_sum += float(abs_err[q99_mask].mean().item()) * int(q99_mask.sum().item())
            q99_count += int(q99_mask.sum().item())

    out = {k: v / max(total_count, 1) for k, v in total.items()}
    out["val_q95_mae"] = q95_mae_sum / max(q95_count, 1)
    out["val_q99_mae"] = q99_mae_sum / max(q99_count, 1)
    out["val_q95_cell_count"] = q95_count
    out["val_q99_cell_count"] = q99_count
    return out


def load_model(checkpoint_path: str, device: torch.device) -> tuple[DeterministicRelativeMoveDualMSEModel, dict[str, Any]]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    if cfg["type"] != "minimal_h1_deterministic_relative_move_212p":
        raise ValueError(f"Unexpected model type: {cfg['type']}")
    model = DeterministicRelativeMoveDualMSEModel(
        n_cells=cfg["n_cells"],
        history_feat_dim=cfg["history_feat_dim"],
        hidden_dim=cfg["hidden_dim"],
        gru_layers=cfg["gru_layers"],
        gru_dropout=cfg["gru_dropout"],
        head_hidden=cfg["head_hidden"],
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    return model, payload


def main() -> None:
    parser = argparse.ArgumentParser(description="212p H=1 deterministic relative-move output + dual MSE")
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
    parser.add_argument("--delta_loss_weight", type=float, default=1.0)
    parser.add_argument("--return_loss_weight", type=float, default=1.0)
    parser.add_argument("--prev_floor", type=float, default=1e-4)
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
    q95_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.95))
    q99_threshold = float(np.quantile(np.abs(train_delta_np.reshape(-1)), 0.99))

    train_loader = DataLoader(TensorDataset(train_hist, train_target), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_hist, val_target), batch_size=args.batch_size, shuffle=False)

    model = DeterministicRelativeMoveDualMSEModel(
        hidden_dim=args.hidden_dim,
        gru_layers=args.gru_layers,
        gru_dropout=args.gru_dropout,
        head_hidden=args.head_hidden,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    print("212p H=1 deterministic relative-move output + dual MSE")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(f"  q95={q95_threshold:.5f} q99={q99_threshold:.5f}")
    print(f"  delta_loss_weight={args.delta_loss_weight:.2f} return_loss_weight={args.return_loss_weight:.2f}")

    history: list[dict[str, Any]] = []
    best_score = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        running = {
            "loss": 0.0,
            "delta_mse": 0.0,
            "return_mse": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "pred_abs_mean": 0.0,
            "pred_abs_rel_move": 0.0,
        }
        count = 0

        for history_01, target_01 in train_loader:
            loss, metrics = model.training_loss(
                history_01=history_01,
                target_01=target_01,
                delta_loss_weight=args.delta_loss_weight,
                return_loss_weight=args.return_loss_weight,
                prev_floor=args.prev_floor,
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            batch = history_01.shape[0]
            running["loss"] += float(loss.item()) * batch
            running["delta_mse"] += float(metrics["delta_mse"].item()) * batch
            running["return_mse"] += float(metrics["return_mse"].item()) * batch
            running["mae"] += float(metrics["mae"].item()) * batch
            running["rmse"] += float(metrics["rmse"].item()) * batch
            running["pred_abs_mean"] += float(metrics["pred_abs_mean"].item()) * batch
            running["pred_abs_rel_move"] += float(metrics["pred_abs_rel_move"].item()) * batch
            count += batch

        train_metrics = {f"train_{k}": v / max(count, 1) for k, v in running.items()}
        val_metrics = evaluate_point_dual_mse(
            model,
            val_loader,
            q95_threshold=q95_threshold,
            q99_threshold=q99_threshold,
            delta_loss_weight=args.delta_loss_weight,
            return_loss_weight=args.return_loss_weight,
            prev_floor=args.prev_floor,
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
                "type": "minimal_h1_deterministic_relative_move_212p",
                "n_cells": 25,
                "history_feat_dim": 50,
                "hidden_dim": args.hidden_dim,
                "gru_layers": args.gru_layers,
                "gru_dropout": args.gru_dropout,
                "head_hidden": args.head_hidden,
                "delta_loss_weight": args.delta_loss_weight,
                "return_loss_weight": args.return_loss_weight,
                "prev_floor": args.prev_floor,
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
            f"loss={train_metrics['train_loss']:.6f}  "
            f"delta={train_metrics['train_delta_mse']:.6f}  "
            f"ret={train_metrics['train_return_mse']:.6f}  "
            f"val_loss={val_metrics['val_loss']:.6f}  "
            f"mae={val_metrics['val_mae']:.4f}  "
            f"q95mae={val_metrics['val_q95_mae']:.4f}  "
            f"q99mae={val_metrics['val_q99_mae']:.4f}  "
            f"sign={val_metrics['val_sign_acc']:.3f}"
        )

    print(f"Finished. Best selection score={best_score:.6f}")
    print(f"Artifacts saved to: {out_dir}")


if __name__ == "__main__":
    main()
