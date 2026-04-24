#!/usr/bin/env python
"""377a: recent-window FM adaptation for the 340c backbone.

This keeps the 340c architecture and vanilla teacher-forced flow-matching
objective unchanged. The only change is data framing: adapt the checkpoint on
the immediately preceding pre-validation window, then evaluate forward.
"""

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

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (  # noqa: E402
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar._rollout_220_utils import build_multistep_windows  # noqa: E402
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402


def build_recent_block(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    adaptation_windows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    raw = np.load(data_path)
    surfaces = raw["surface"].astype(np.float32)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    max_train_idx = test_start - history_len - future_len
    val_start = max_train_idx - val_size
    n = min(int(adaptation_windows), int(val_start))
    indices = np.arange(val_start - n, val_start)
    history_01, future_01 = build_multistep_windows(indices, surf_tensor, history_len, future_len)
    future_01 = future_01.view(history_01.shape[0], future_len, 5, 5)
    return history_01, future_01, indices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="models/backfill/340c_v0_s42/best_model.pt")
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--adaptation_windows", type=int, default=441)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument(
        "--anchor_weight",
        type=float,
        default=0.0,
        help="Optional L2 penalty that keeps trainable weights near the source checkpoint.",
    )
    parser.add_argument(
        "--trainable_scope",
        choices=["all", "conditioning", "conditioning_memory_proj"],
        default="all",
        help="Use conditioning to adapt only feature/memory conditioning parameters.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    model.train()
    if model.cfg.history_len != args.history_len or model.cfg.future_len != args.future_len:
        raise ValueError("Checkpoint horizon configuration does not match requested data")
    if args.trainable_scope == "conditioning":
        trainable_prefixes = ("feature_proj.", "pos_embed.", "memory.", "memory_norm.")
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(trainable_prefixes)
    elif args.trainable_scope == "conditioning_memory_proj":
        trainable_prefixes = (
            "feature_proj.",
            "pos_embed.",
            "memory.",
            "memory_norm.",
            "velocity.memory_proj.",
        )
        for name, param in model.named_parameters():
            param.requires_grad = name.startswith(trainable_prefixes)
    trainable_named_params = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for _, p in trainable_named_params)
    anchor_params = {
        name: p.detach().clone()
        for name, p in trainable_named_params
    } if args.anchor_weight > 0 else {}

    hist_01, fut_01, indices = build_recent_block(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        adaptation_windows=args.adaptation_windows,
        device=device,
    )
    loader = DataLoader(
        TensorDataset(hist_01, fut_01),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_loss = float("inf")
    best_epoch = -1
    records: list[dict[str, float]] = []

    print(f"Source checkpoint: {args.checkpoint}")
    print(f"Source epoch: {payload.get('epoch')}  source best_val: {payload.get('best_val')}")
    print(f"Adaptation windows: {len(loader.dataset)}  index range: {indices[0]}..{indices[-1]}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params: {n_trainable:,}  scope={args.trainable_scope}")
    print(f"Anchor weight: {args.anchor_weight:.3g}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        sums: dict[str, float] = {}
        n_batches = 0
        model.train()
        for hist_batch, fut_batch in loader:
            hist_norm = normalize_iv(hist_batch.to(device, non_blocking=True)).view(
                hist_batch.shape[0], hist_batch.shape[1], -1
            )
            fut_norm = normalize_iv(fut_batch.to(device, non_blocking=True)).view(
                fut_batch.shape[0], fut_batch.shape[1], -1
            )
            loss, metrics = model.training_loss(hist_norm, fut_norm)
            anchor_penalty = loss.new_zeros(())
            if anchor_params:
                for name, param in trainable_named_params:
                    anchor_penalty = anchor_penalty + torch.sum((param - anchor_params[name]) ** 2)
                loss = loss + args.anchor_weight * anchor_penalty
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.item())
            sums["anchor_penalty"] = sums.get("anchor_penalty", 0.0) + float(anchor_penalty.item())
            n_batches += 1
        scheduler.step()
        avg = {key: value / max(n_batches, 1) for key, value in sums.items()}
        rec = {
            "epoch": epoch,
            "adapt_total": avg["total"],
            "transition_std": avg["transition_std"],
            "transition_abs": avg["transition_abs"],
            "target_velocity_std": avg["target_velocity_std"],
            "memory_abs": avg["memory_abs"],
            "anchor_penalty": avg["anchor_penalty"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] adapt={rec['adapt_total']:.5f} "
            f"trans_std={rec['transition_std']:.3f} trans_abs={rec['transition_abs']:.3f} "
            f"vel_std={rec['target_velocity_std']:.3f} anchor={rec['anchor_penalty']:.3e} "
            f"lr={rec['lr']:.2e} "
            f"time={rec['sec']:.1f}s"
        )
        if rec["adapt_total"] < best_loss:
            best_loss = rec["adapt_total"]
            best_epoch = epoch
            save_checkpoint(str(out_dir / "best_model.pt"), model, model.cfg, epoch, best_loss)

    save_checkpoint(str(out_dir / "final_model.pt"), model, model.cfg, args.epochs, best_loss)
    summary = {
        "source_checkpoint": args.checkpoint,
        "source_epoch": payload.get("epoch"),
        "source_best_val": payload.get("best_val"),
        "adaptation_start_index": int(indices[0]),
        "adaptation_end_index": int(indices[-1]),
        "best_epoch_by_adaptation_loss": best_epoch,
        "best_adaptation_loss": best_loss,
        "trainable_scope": args.trainable_scope,
        "n_trainable_params": n_trainable,
        "anchor_weight": args.anchor_weight,
        "config": payload["config"],
    }
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
