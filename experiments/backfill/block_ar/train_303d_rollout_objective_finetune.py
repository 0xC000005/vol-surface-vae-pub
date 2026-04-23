#!/usr/bin/env python
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

from diffusion.block_ar.logit_level_flow_matching import logit_to_iv
from diffusion.block_ar.recurrent_logit_transition_token_flow_matching import (
    RecurrentLogitTransitionTokenFlowMatching,
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_212b_h1_minimal_direct_stochastic_delta import (
    energy_score,
)


def make_dataset(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    max_train_windows: int,
    max_val_windows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    raw = np.load(data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    max_train_idx = test_start - history_len - future_len
    train_indices = np.arange(0, max_train_idx - val_size)[:max_train_windows]
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)[:max_val_windows]
    train_hist, train_future = build_multistep_windows(
        train_indices, surf_tensor, history_len, future_len
    )
    val_hist, val_future = build_multistep_windows(
        val_indices, surf_tensor, history_len, future_len
    )
    return train_hist, train_future, val_hist, val_future


def parse_horizons(raw: str, future_len: int) -> list[int]:
    horizons = []
    for item in raw.split(","):
        h = int(item.strip())
        if 1 <= h <= future_len:
            horizons.append(h)
    if not horizons:
        raise ValueError("At least one rollout horizon must be valid")
    return sorted(set(horizons))


def rollout_samples_grad(
    model: RecurrentLogitTransitionTokenFlowMatching,
    history_norm: torch.Tensor,
    n_samples: int,
    n_steps: int,
    flow_steps: int,
    temperature: float,
) -> torch.Tensor:
    history_norm = model._flatten(history_norm)
    base_state, base_logit = model.encode_history(history_norm)
    bsz = history_norm.shape[0]
    state_stack = (
        base_state.unsqueeze(2)
        .expand(model.cfg.gru_layers, bsz, n_samples, model.cfg.hidden_dim)
        .reshape(model.cfg.gru_layers, bsz * n_samples, model.cfg.hidden_dim)
        .clone()
    )
    current_logit = (
        base_logit.unsqueeze(1)
        .expand(bsz, n_samples, model.cfg.n_cells)
        .reshape(bsz * n_samples, model.cfg.n_cells)
        .clone()
    )
    dt = 1.0 / float(flow_steps)
    frames: list[torch.Tensor] = []
    for _ in range(n_steps):
        x = temperature * torch.randn(
            bsz * n_samples,
            model.cfg.n_cells,
            device=history_norm.device,
            dtype=history_norm.dtype,
        )
        for step in range(flow_steps):
            t = torch.full(
                (bsz * n_samples,),
                (step + 0.5) * dt,
                device=history_norm.device,
                dtype=history_norm.dtype,
            )
            x = x + dt * model.predict_velocity(x, current_logit, state_stack, t)
        next_logit = current_logit + x
        next_iv = logit_to_iv(next_logit)
        frames.append(next_iv.view(bsz, n_samples, model.cfg.n_cells))
        state_stack = model.recurrent_step(state_stack, current_logit, next_logit)
        current_logit = next_logit
    return torch.stack(frames, dim=2)


def rollout_energy(
    samples_01: torch.Tensor,
    future_01: torch.Tensor,
    horizons: list[int],
) -> torch.Tensor:
    target = future_01.view(future_01.shape[0], future_01.shape[1], -1)
    losses = [
        energy_score(samples_01[:, :, h - 1], target[:, h - 1])
        for h in horizons
    ]
    return torch.stack(losses).mean()


def main() -> None:
    parser = argparse.ArgumentParser(description="303d rollout-objective fine-tune from 303b")
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=512)
    parser.add_argument("--max_val_windows", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--rollout_weight", type=float, default=0.5)
    parser.add_argument("--rollout_samples", type=int, default=2)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--rollout_horizons", type=str, default="1,7,14,30")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    model, payload = load_model(args.init_checkpoint, device)
    cfg = model.cfg
    horizons = parse_horizons(args.rollout_horizons, args.future_len)
    train_hist, train_future, val_hist, val_future = make_dataset(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_train_windows=args.max_train_windows,
        max_val_windows=args.max_val_windows,
        device=device,
    )
    train_loader = DataLoader(
        TensorDataset(train_hist, train_future),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val = float("inf")
    best_epoch = -1
    history: list[dict[str, float]] = []

    print("303d rollout-objective fine-tune")
    print(f"  init={args.init_checkpoint} epoch={payload.get('epoch', -1)}")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(
        f"  rollout_samples={args.rollout_samples} flow_steps={args.rollout_flow_steps} "
        f"horizons={horizons} weight={args.rollout_weight}"
    )

    def run_epoch(loader: DataLoader, train_mode: bool) -> dict[str, float]:
        model.train(train_mode)
        totals = {"total": 0.0, "fm": 0.0, "rollout": 0.0}
        count = 0
        for hist_01, fut_01 in loader:
            hist_norm = normalize_iv(hist_01.to(device)).view(hist_01.shape[0], hist_01.shape[1], -1)
            fut_norm = normalize_iv(fut_01.to(device)).view(fut_01.shape[0], fut_01.shape[1], -1)
            fut_01 = fut_01.to(device)
            with torch.set_grad_enabled(train_mode):
                fm_loss, _metrics = model.training_loss(hist_norm, fut_norm)
                samples = rollout_samples_grad(
                    model,
                    hist_norm,
                    n_samples=args.rollout_samples,
                    n_steps=args.future_len,
                    flow_steps=args.rollout_flow_steps,
                    temperature=args.temperature,
                )
                roll_loss = rollout_energy(samples, fut_01, horizons)
                loss = fm_loss + args.rollout_weight * roll_loss
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
            bs = hist_01.shape[0]
            totals["total"] += float(loss.detach().item()) * bs
            totals["fm"] += float(fm_loss.detach().item()) * bs
            totals["rollout"] += float(roll_loss.detach().item()) * bs
            count += bs
        return {k: v / max(count, 1) for k, v in totals.items()}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(train_loader, train_mode=True)
        val_metrics = run_epoch(val_loader, train_mode=False)
        rec = {
            "epoch": epoch,
            "train_total": train_metrics["total"],
            "train_fm": train_metrics["fm"],
            "train_rollout": train_metrics["rollout"],
            "val_total": val_metrics["total"],
            "val_fm": val_metrics["fm"],
            "val_rollout": val_metrics["rollout"],
            "sec": time.time() - t0,
        }
        history.append(rec)
        print(
            f"[ep {epoch:02d}] "
            f"train={rec['train_total']:.5f} fm={rec['train_fm']:.5f} roll={rec['train_rollout']:.5f} "
            f"val={rec['val_total']:.5f} vfm={rec['val_fm']:.5f} vroll={rec['val_rollout']:.5f} "
            f"time={rec['sec']:.1f}s"
        )
        if rec["val_total"] < best_val:
            best_val = rec["val_total"]
            best_epoch = epoch
            save_checkpoint(str(out_dir / "best_model.pt"), model, cfg, epoch, best_val)

    save_checkpoint(str(out_dir / "final_model.pt"), model, cfg, args.epochs, best_val)
    (out_dir / "training_history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "train_summary.json").write_text(
        json.dumps({"best_epoch": best_epoch, "best_val_total": best_val}, indent=2)
    )


if __name__ == "__main__":
    main()
