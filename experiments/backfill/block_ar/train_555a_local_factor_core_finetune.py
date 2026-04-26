#!/usr/bin/env python
"""555a: local-factor conditioned core fine-tune from the 510a frontier."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (  # noqa: E402
    EmpiricalNormalScoreCausalMemoryTransitionFMConfig,
    EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
)
from experiments.backfill.block_ar._factor_conditioning_525_utils import (  # noqa: E402
    official_train_val_indices,
    recent_prevalidation_indices,
)
from experiments.backfill.block_ar._local_factor_conditioning_555_utils import (  # noqa: E402
    build_local_factor_history_block,
    tensor_dict_summary,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402


def load_local_factor_extended_checkpoint(
    checkpoint: str,
    factor_dim: int,
    device: torch.device,
) -> tuple[EmpiricalNormalScoreCausalMemoryTransitionFlowMatching, dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg_dict = dict(payload["config"])
    cfg_dict["factor_dim"] = int(factor_dim)
    cfg = EmpiricalNormalScoreCausalMemoryTransitionFMConfig(**cfg_dict)
    model = EmpiricalNormalScoreCausalMemoryTransitionFlowMatching(cfg)
    result = model.load_state_dict(payload["model_state_dict"], strict=False)
    allowed_missing = {
        "factor_encoder.weight_ih_l0",
        "factor_encoder.weight_hh_l0",
        "factor_encoder.bias_ih_l0",
        "factor_encoder.bias_hh_l0",
        "factor_context_norm.weight",
        "factor_context_norm.bias",
        "factor_context_scale",
    }
    missing = set(result.missing_keys) - allowed_missing
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )
    model.to(device)
    return model, payload


def save_local_factor_checkpoint(
    path: Path,
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    epoch: int,
    best_train: float,
    factor_mean: np.ndarray,
    factor_std: np.ndarray,
    factor_columns: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.cfg),
            "epoch": int(epoch),
            "best_val": float(best_train),
            "model_state_dict": model.state_dict(),
            "factor_mean": factor_mean.astype(np.float32),
            "factor_std": factor_std.astype(np.float32),
            "factor_columns": list(factor_columns),
            "factor_source": "vol_surface_with_ret.npz",
            "train_scope": "all_core_parameters",
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/509a_recent_patch_energy_l5_w005_s42/final_model.pt",
    )
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--adaptation_windows", type=int, default=441)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=555)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    train_indices, _ = official_train_val_indices(
        test_start=args.test_start,
        val_size=args.val_size,
        history_len=args.history_len,
        future_len=args.future_len,
    )
    adapt_indices = recent_prevalidation_indices(
        test_start=args.test_start,
        val_size=args.val_size,
        history_len=args.history_len,
        future_len=args.future_len,
        adaptation_windows=args.adaptation_windows,
    )
    train_block = build_local_factor_history_block(
        data_path=args.data_path,
        indices=adapt_indices,
        history_len=args.history_len,
        future_len=args.future_len,
        device=device,
        fit_indices=train_indices,
    )
    factor_dim = int(train_block.factor_history.shape[-1])
    model, payload = load_local_factor_extended_checkpoint(args.checkpoint, factor_dim, device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(
        TensorDataset(train_block.history_01, train_block.future_01, train_block.factor_history),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    records: list[dict[str, float]] = []
    best_loss = float("inf")
    best_epoch = -1
    print(f"Source checkpoint: {args.checkpoint}")
    print(f"Source epoch: {payload.get('epoch')}  source best_val: {payload.get('best_val')}")
    print(json.dumps(tensor_dict_summary(train_block), indent=2))
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        sums: dict[str, float] = {}
        n_batches = 0
        for hist_01, fut_01, factor_history in loader:
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
            loss, metrics = model.training_loss(
                hist_norm,
                fut_norm,
                factor_history=factor_history.to(device, non_blocking=True),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
            optimizer.step()
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.item())
            n_batches += 1
        scheduler.step()
        avg = {key: value / max(n_batches, 1) for key, value in sums.items()}
        rec = {
            "epoch": float(epoch),
            "train_total": avg["total"],
            "fm_loss": avg["fm_loss"],
            "factor_context_abs": avg.get("factor_context_abs", 0.0),
            "factor_context_scale_abs": avg.get("factor_context_scale_abs", 0.0),
            "transition_std": avg["transition_std"],
            "target_velocity_std": avg["target_velocity_std"],
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] loss={rec['train_total']:.5f} "
            f"factor_scale={rec['factor_context_scale_abs']:.5f} "
            f"factor_ctx={rec['factor_context_abs']:.3f} "
            f"lr={rec['lr']:.2e} time={rec['sec']:.1f}s",
            flush=True,
        )
        if rec["train_total"] < best_loss:
            best_loss = rec["train_total"]
            best_epoch = epoch
            save_local_factor_checkpoint(
                out_dir / "best_model.pt",
                model,
                epoch,
                best_loss,
                train_block.factor_mean,
                train_block.factor_std,
                train_block.factor_columns,
            )

    save_local_factor_checkpoint(
        out_dir / "final_model.pt",
        model,
        args.epochs,
        best_loss,
        train_block.factor_mean,
        train_block.factor_std,
        train_block.factor_columns,
    )
    summary = {
        "best_epoch": int(best_epoch),
        "best_train_total": float(best_loss),
        "source_checkpoint": args.checkpoint,
        "factor_dim": int(factor_dim),
        "train_scope": "all_core_parameters",
        "train_block": tensor_dict_summary(train_block),
        "config": asdict(model.cfg),
    }
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
