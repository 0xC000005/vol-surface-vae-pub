#!/usr/bin/env python
"""540a: factor-conditioned patch-energy fine-tune from the 525/392 frontier line."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.empirical_normal_score_causal_memory_transition_flow_matching import (  # noqa: E402
    EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    load_model,
)
from experiments.backfill.block_ar._factor_conditioning_525_utils import (  # noqa: E402
    build_factor_history_block,
    recent_prevalidation_indices,
    tensor_dict_summary,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402
from experiments.backfill.block_ar.train_509a_recent_patch_energy_finetune import (  # noqa: E402
    patch_energy_score,
)
from experiments.backfill.block_ar.train_525a_factor_conditioned_surface_fm import (  # noqa: E402
    save_factor_checkpoint,
)


def sample_rollout_scores_with_grad_factor(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    history_norm: torch.Tensor,
    factor_history: torch.Tensor | None,
    n_samples: int,
    n_steps: int,
    flow_steps: int,
) -> torch.Tensor:
    """Differentiable free rollout in score space, preserving factor conditioning."""
    history_scores = model.history_scores(history_norm)
    factor_context = model._factor_context(
        factor_history,
        device=history_scores.device,
        dtype=history_scores.dtype,
    )
    bsz = history_scores.shape[0]
    k = int(n_samples)
    n_flow = max(1, int(flow_steps))
    dt = 1.0 / float(n_flow)
    prefix = (
        history_scores.unsqueeze(1)
        .expand(bsz, k, model.cfg.history_len, model.cfg.n_cells)
        .reshape(bsz * k, model.cfg.history_len, model.cfg.n_cells)
        .clone()
    )
    chunk_factor_context = None
    if factor_context is not None:
        chunk_factor_context = (
            factor_context.unsqueeze(1)
            .expand(bsz, k, model.cfg.memory_dim)
            .reshape(bsz * k, model.cfg.memory_dim)
        )

    rho = float(max(0.0, min(0.999, model.cfg.path_source_corr)))
    ar_rho = float(max(0.0, min(0.999, model.cfg.path_source_ar)))
    temporal_source = model._ar1_source_noise(
        torch.Size((bsz * k, n_steps, model.cfg.n_cells)),
        ar_rho,
        history_scores.device,
        history_scores.dtype,
    )
    path_source = None
    if rho > 0.0:
        path_source = torch.randn(
            bsz * k,
            model.cfg.n_cells,
            device=history_scores.device,
            dtype=history_scores.dtype,
        )

    frames: list[torch.Tensor] = []
    for step in range(n_steps):
        memory_state = model._encode_prefix_scores(
            prefix,
            factor_context=chunk_factor_context,
        )[:, -1]
        current_score = prefix[:, -1]
        if path_source is None:
            x = model.cfg.sample_temperature * temporal_source[:, step]
        else:
            x = model.cfg.sample_temperature * (
                math.sqrt(rho) * path_source
                + math.sqrt(1.0 - rho) * temporal_source[:, step]
            )
        noise_scale = model._conditional_noise_scale(memory_state)
        if noise_scale is not None:
            x = x * noise_scale
        for flow_step in range(n_flow):
            t = torch.full(
                (bsz * k,),
                (flow_step + 0.5) * dt,
                device=history_scores.device,
                dtype=history_scores.dtype,
            )
            x = x + dt * model.predict_velocity(x, current_score, memory_state, t)
        next_score = current_score + x
        frames.append(next_score.view(bsz, k, model.cfg.n_cells))
        prefix = torch.cat([prefix, next_score[:, None, :]], dim=1)
    return torch.stack(frames, dim=2)


def combined_factor_patch_loss(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    factor_history: torch.Tensor | None,
    train_sample_count: int,
    rollout_flow_steps: int,
    patch_len: int,
    patch_energy_weight: float,
    fm_anchor_weight: float,
    energy_eps: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fm_loss, fm_metrics = model.training_loss(
        history_norm,
        future_norm,
        factor_history=factor_history,
    )
    target_scores = model.target_future_scores(future_norm)
    sampled_scores = sample_rollout_scores_with_grad_factor(
        model=model,
        history_norm=history_norm,
        factor_history=factor_history,
        n_samples=train_sample_count,
        n_steps=target_scores.shape[1],
        flow_steps=rollout_flow_steps,
    )
    patch_loss, target_dist, pair_dist = patch_energy_score(
        samples=sampled_scores,
        target=target_scores,
        patch_len=patch_len,
        eps=energy_eps,
    )
    total = float(fm_anchor_weight) * fm_loss + float(patch_energy_weight) * patch_loss
    metrics = {
        "total": total.detach(),
        "fm_loss": fm_loss.detach(),
        "patch_energy": patch_loss.detach(),
        "patch_target_dist": target_dist.detach(),
        "patch_pair_dist": pair_dist.detach(),
        "transition_std": fm_metrics["transition_std"].detach(),
        "sample_score_std": sampled_scores.std(unbiased=False).detach(),
        "target_score_std": target_scores.std(unbiased=False).detach(),
        "sample_h1_std": sampled_scores[:, :, 0].std(unbiased=False).detach(),
        "sample_h30_std": sampled_scores[:, :, -1].std(unbiased=False).detach(),
    }
    if "factor_context_abs" in fm_metrics:
        metrics["factor_context_abs"] = fm_metrics["factor_context_abs"].detach()
        metrics["factor_context_scale_abs"] = fm_metrics["factor_context_scale_abs"].detach()
    return total, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/525a_factor_conditioned_surface_fm_e4_s525/best_model.pt",
    )
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--adaptation_windows", type=int, default=441)
    parser.add_argument("--holdout_frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--train_sample_count", type=int, default=4)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--patch_len", type=int, default=5)
    parser.add_argument("--energy_eps", type=float, default=1e-6)
    parser.add_argument("--patch_energy_weight", type=float, default=0.05)
    parser.add_argument("--fm_anchor_weight", type=float, default=1.0)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=540)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_model(args.checkpoint, device)
    if model.cfg.factor_dim <= 0:
        raise ValueError("540a requires a factor-conditioned checkpoint with factor_dim > 0")
    factor_mean = payload.get("factor_mean")
    factor_std = payload.get("factor_std")
    factor_columns = payload.get("factor_columns")
    if factor_mean is None or factor_std is None or factor_columns is None:
        raise ValueError("Factor checkpoint must include factor_mean, factor_std, and factor_columns")
    if model.cfg.history_len != args.history_len or model.cfg.future_len != args.future_len:
        raise ValueError("Checkpoint horizon configuration does not match requested data")

    adapt_indices = recent_prevalidation_indices(
        test_start=args.test_start,
        val_size=args.val_size,
        history_len=args.history_len,
        future_len=args.future_len,
        adaptation_windows=args.adaptation_windows,
    )
    block = build_factor_history_block(
        data_path=args.data_path,
        indices=adapt_indices,
        history_len=args.history_len,
        future_len=args.future_len,
        device=device,
        factor_mean=factor_mean,
        factor_std=factor_std,
    )
    n_total = int(block.history_01.shape[0])
    n_val = max(1, int(round(n_total * float(args.holdout_frac))))
    n_train = n_total - n_val
    train_tensors = (
        block.history_01[:n_train],
        block.future_01[:n_train],
        block.factor_history[:n_train],
    )
    val_tensors = (
        block.history_01[n_train:],
        block.future_01[n_train:],
        block.factor_history[n_train:],
    )
    train_loader = DataLoader(
        TensorDataset(*train_tensors),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(*val_tensors),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def run_epoch(loader: DataLoader, train_mode: bool, max_batches: int) -> dict[str, float]:
        model.train(train_mode)
        sums: dict[str, float] = {}
        n_batches = 0
        for hist_01, fut_01, factor_history in loader:
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
                loss, metrics = combined_factor_patch_loss(
                    model=model,
                    history_norm=hist_norm,
                    future_norm=fut_norm,
                    factor_history=factor_history.to(device, non_blocking=True),
                    train_sample_count=args.train_sample_count,
                    rollout_flow_steps=args.rollout_flow_steps,
                    patch_len=args.patch_len,
                    patch_energy_weight=args.patch_energy_weight,
                    fm_anchor_weight=args.fm_anchor_weight,
                    energy_eps=args.energy_eps,
                )
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.item())
            n_batches += 1
        return {key: value / max(n_batches, 1) for key, value in sums.items()}

    print(f"Source checkpoint: {args.checkpoint}")
    print(f"Source epoch: {payload.get('epoch')}  source best_val: {payload.get('best_val')}")
    print(json.dumps(tensor_dict_summary(block), indent=2))
    print(f"Train/holdout: {n_train}/{n_val}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    best_val = float("inf")
    best_epoch = -1
    records: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True, max_batches=args.max_train_batches)
        val_avg = run_epoch(val_loader, train_mode=False, max_batches=args.max_val_batches)
        scheduler.step()
        rec = {
            "epoch": epoch,
            "train_total": train_avg["total"],
            "train_fm_loss": train_avg["fm_loss"],
            "train_patch_energy": train_avg["patch_energy"],
            "val_total": val_avg["total"],
            "val_fm_loss": val_avg["fm_loss"],
            "val_patch_energy": val_avg["patch_energy"],
            "val_patch_target_dist": val_avg["patch_target_dist"],
            "val_patch_pair_dist": val_avg["patch_pair_dist"],
            "val_sample_score_std": val_avg["sample_score_std"],
            "val_target_score_std": val_avg["target_score_std"],
            "val_sample_h1_std": val_avg["sample_h1_std"],
            "val_sample_h30_std": val_avg["sample_h30_std"],
            "val_factor_context_scale_abs": val_avg.get("factor_context_scale_abs", 0.0),
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} fm={rec['val_fm_loss']:.5f} "
            f"patch={rec['val_patch_energy']:.5f} "
            f"target={rec['val_patch_target_dist']:.3f} pair={rec['val_patch_pair_dist']:.3f} "
            f"std={rec['val_sample_score_std']:.3f}/{rec['val_target_score_std']:.3f} "
            f"h1/h30={rec['val_sample_h1_std']:.3f}/{rec['val_sample_h30_std']:.3f} "
            f"fscale={rec['val_factor_context_scale_abs']:.5f} "
            f"lr={rec['lr']:.2e} time={rec['sec']:.1f}s",
            flush=True,
        )
        if rec["val_total"] < best_val:
            best_val = rec["val_total"]
            best_epoch = epoch
            save_factor_checkpoint(
                out_dir / "best_model.pt",
                model,
                epoch,
                best_val,
                np.asarray(factor_mean, dtype=np.float32),
                np.asarray(factor_std, dtype=np.float32),
                list(factor_columns),
            )

    save_factor_checkpoint(
        out_dir / "final_model.pt",
        model,
        args.epochs,
        best_val,
        np.asarray(factor_mean, dtype=np.float32),
        np.asarray(factor_std, dtype=np.float32),
        list(factor_columns),
    )
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {
        "source_checkpoint": args.checkpoint,
        "source_epoch": payload.get("epoch"),
        "source_best_val": payload.get("best_val"),
        "adaptation_start_index": int(block.indices[0]),
        "adaptation_end_index": int(block.indices[-1]),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "objective": {
            "fm_anchor_weight": args.fm_anchor_weight,
            "patch_energy_weight": args.patch_energy_weight,
            "patch_len": args.patch_len,
            "train_sample_count": args.train_sample_count,
            "rollout_flow_steps": args.rollout_flow_steps,
            "loss": "factor_conditioned_overlapping_patch_energy",
        },
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

