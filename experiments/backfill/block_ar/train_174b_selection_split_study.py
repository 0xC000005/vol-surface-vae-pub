#!/usr/bin/env python
"""
174b: Held-out selection-split study on the 173a backbone.

Keep the 173a model and loss unchanged.
Change only the checkpoint-selection protocol:
  - hold out a separate model-selection split from the training slice
  - keep the original validation split for NLL monitoring
  - average proxy metrics across repeated seeds
  - save best_nll and best_select checkpoints separately
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from experiments.backfill.block_ar.train_169a_transformed_student_t import (
    make_serializable,
)
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
)
from experiments.backfill.block_ar.train_173a_local_var_residual_flow_structured_joint_student_t import (
    LocalVarianceResidualFlowStructuredJointStudentTModel,
    evaluate_joint_subset,
    evaluate_teacher_forced,
    joint_nll_loss,
)
from experiments.backfill.block_ar.train_174a_selection_study import (
    evaluate_proxy_subset,
    save_checkpoint,
)


def parse_seed_list(seed_list: str) -> list[int]:
    seeds = []
    for part in seed_list.split(","):
        part = part.strip()
        if part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("proxy_seeds must contain at least one integer")
    return seeds


def aggregate_proxy_metrics(
    model: LocalVarianceResidualFlowStructuredJointStudentTModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    seeds: list[int],
    n_samples: int,
    batch_size: int,
) -> dict:
    per_seed = []
    for seed in seeds:
        metrics = evaluate_proxy_subset(
            model=model,
            history_01=history_01,
            future_01=future_01,
            n_samples=n_samples,
            batch_size=batch_size,
            seed=seed,
        )
        metrics["seed"] = seed
        per_seed.append(metrics)

    numeric_keys = [
        "proxy_overall_cov90",
        "proxy_calibration_error",
        "proxy_horizon_pass_count",
        "proxy_cell_low_pass_count",
        "proxy_cell_high_pass_count",
        "proxy_worst_h1",
        "proxy_worst_h7",
        "proxy_worst_h14",
        "proxy_worst_h30",
        "proxy_best_h1",
        "proxy_best_h7",
        "proxy_best_h14",
        "proxy_best_h30",
        "proxy_turb_calm_ratio",
        "proxy_layer1_pass_count",
        "proxy_layer2_n_passing",
        "proxy_layer2_n_total",
        "proxy_layer3_catastrophic_rate",
        "proxy_window_floor_pct_bad",
        "proxy_gate_units",
        "proxy_violation",
    ]
    bool_keys = [
        "proxy_turb_calm_pass",
        "proxy_layer3_pass",
        "proxy_window_floor_pass",
    ]

    agg = {"proxy_seed_metrics": per_seed}
    for key in numeric_keys:
        vals = np.array([float(row[key]) for row in per_seed], dtype=float)
        agg[f"select_{key[6:]}_mean"] = float(vals.mean())
        agg[f"select_{key[6:]}_std"] = float(vals.std(ddof=0))
    for key in bool_keys:
        vals = np.array([1.0 if row[key] else 0.0 for row in per_seed], dtype=float)
        agg[f"select_{key[6:]}_rate"] = float(vals.mean())
    return agg


def selection_key(row: dict) -> tuple[float, float, float, float, float, float]:
    return (
        float(row["select_gate_units_mean"]),
        -float(row["select_violation_mean"]),
        -float(row["select_violation_std"]),
        -float(row["select_layer3_catastrophic_rate_mean"]),
        -float(row["select_window_floor_pct_bad_mean"]),
        -float(row["val_total_loss"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="174b: held-out selection-split study on 173a")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=1e-3)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--lr_flow", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_decoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_flow", type=float, default=0.0)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--time_rank", type=int, default=6)
    parser.add_argument("--cell_rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.05)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--flow_context_dim", type=int, default=256)
    parser.add_argument("--flow_hidden_dim", type=int, default=256)
    parser.add_argument("--flow_layers", type=int, default=4)
    parser.add_argument("--flow_scale_clip", type=float, default=2.0)
    parser.add_argument("--base_nu", type=float, default=8.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-5)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--joint_val_samples", type=int, default=8)
    parser.add_argument("--joint_eval_limit", type=int, default=64)
    parser.add_argument("--selection_split_size", type=int, default=256)
    parser.add_argument("--selection_samples", type=int, default=8)
    parser.add_argument("--selection_batch_size", type=int, default=32)
    parser.add_argument("--proxy_seeds", type=str, default="20260404,20260405")
    parser.add_argument("--proxy_eval_every", type=int, default=1)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--local_delta_clip", type=float, default=0.35)
    parser.add_argument("--local_var_penalty", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    epoch_dir = outdir / "epoch_checkpoints"
    epoch_dir.mkdir(parents=True, exist_ok=True)

    proxy_seeds = parse_seed_list(args.proxy_seeds)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)

    hist_len = args.history_len
    future_len = args.future_len
    test_start = 4511
    max_train_idx = test_start - hist_len - future_len
    val_size = 441

    base_train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)

    if args.max_train_windows is not None:
        base_train_indices = base_train_indices[: args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    selection_size = min(args.selection_split_size, max(1, len(base_train_indices) // 8))
    if selection_size >= len(base_train_indices):
        raise ValueError("selection_split_size leaves no training windows")

    selection_indices = base_train_indices[-selection_size:]
    train_indices = base_train_indices[:-selection_size]

    print(f"Train: {len(train_indices)}, Selection: {len(selection_indices)}, Val: {len(val_indices)}")

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, hist_len, future_len)
    selection_hist, selection_future = build_multistep_windows(selection_indices, surf_tensor, hist_len, future_len)
    val_hist, val_future = build_multistep_windows(val_indices, surf_tensor, hist_len, future_len)

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
    )

    encoder_config = EncoderConfig(
        input_dim=25,
        gru_hidden_dim=64,
        bottleneck_dim=128,
        dropout=0.1,
        cond_aug_sigma=0.0,
    )
    decoder_config = dict(
        n_frames=future_len,
        n_cells=25,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        cond_dim=128,
        time_rank=args.time_rank,
        cell_rank=args.cell_rank,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        flow_context_dim=args.flow_context_dim,
        local_delta_clip=args.local_delta_clip,
    )
    flow_config = dict(
        dim=future_len * 25,
        context_dim=args.flow_context_dim,
        hidden_dim=args.flow_hidden_dim,
        n_layers=args.flow_layers,
        scale_clip=args.flow_scale_clip,
    )
    model = LocalVarianceResidualFlowStructuredJointStudentTModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
        base_nu=args.base_nu,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print("\n" + "=" * 64)
    print("174b: held-out selection split on 173a backbone")
    print("=" * 64)
    print(f"  Total params: {n_params:,}")
    print(f"  Selection seeds: {proxy_seeds}")
    print(f"  Selection samples/window: {args.selection_samples}")
    print("  Selection objective: gate_units_mean -> violation_mean -> violation_std")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.encoder.parameters(),
                "lr": args.lr_encoder,
                "weight_decay": args.weight_decay_encoder,
            },
            {
                "params": model.decoder.parameters(),
                "lr": args.lr_decoder,
                "weight_decay": args.weight_decay_decoder,
            },
            {
                "params": model.flow.parameters(),
                "lr": args.lr_flow,
                "weight_decay": args.weight_decay_flow,
            },
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_nll = float("inf")
    best_nll_metrics = None
    best_select_key = None
    best_select_metrics = None
    history = []
    history_path = outdir / "training_history.json"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep_total = 0.0
        ep_nll = 0.0
        ep_pen = 0.0
        ep_mae = 0.0
        ep_time_rank = 0.0
        ep_cell_rank = 0.0
        ep_scale = 0.0
        ep_flow_logdet = 0.0
        ep_white_std = 0.0
        ep_z_std = 0.0
        ep_local_delta_rms = 0.0
        ep_local_scale_min = 0.0
        ep_local_scale_max = 0.0
        nb = 0

        for history_01, future_01 in train_loader:
            optimizer.zero_grad()
            loss, metrics = joint_nll_loss(
                model,
                history_01,
                future_01,
                local_var_penalty=args.local_var_penalty,
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_total += metrics["total_loss"].item()
            ep_nll += metrics["joint_nll"].item()
            ep_pen += metrics["local_var_penalty"].item()
            ep_mae += metrics["joint_mae"].item()
            ep_time_rank += metrics["time_eff_rank"].item()
            ep_cell_rank += metrics["cell_eff_rank"].item()
            ep_scale += metrics["scale_mean"].item()
            ep_flow_logdet += metrics["flow_logdet_mean"].item()
            ep_white_std += metrics["white_std_mean"].item()
            ep_z_std += metrics["z_std_mean"].item()
            ep_local_delta_rms += metrics["local_delta_rms"].item()
            ep_local_scale_min += metrics["local_scale_min"].item()
            ep_local_scale_max += metrics["local_scale_max"].item()
            nb += 1

        scheduler.step()

        train_metrics = {
            "train_total_loss": ep_total / max(nb, 1),
            "train_joint_nll": ep_nll / max(nb, 1),
            "train_local_var_penalty": ep_pen / max(nb, 1),
            "train_joint_mae": ep_mae / max(nb, 1),
            "train_time_eff_rank": ep_time_rank / max(nb, 1),
            "train_cell_eff_rank": ep_cell_rank / max(nb, 1),
            "train_scale_mean": ep_scale / max(nb, 1),
            "train_flow_logdet_mean": ep_flow_logdet / max(nb, 1),
            "train_white_std_mean": ep_white_std / max(nb, 1),
            "train_z_std_mean": ep_z_std / max(nb, 1),
            "train_local_delta_rms": ep_local_delta_rms / max(nb, 1),
            "train_local_scale_min": ep_local_scale_min / max(nb, 1),
            "train_local_scale_max": ep_local_scale_max / max(nb, 1),
        }
        val_metrics = evaluate_teacher_forced(
            model,
            val_loader,
            local_var_penalty=args.local_var_penalty,
        )
        joint_metrics = evaluate_joint_subset(
            model,
            val_loader,
            joint_val_samples=args.joint_val_samples,
            eval_limit=args.joint_eval_limit,
        )
        if epoch % args.proxy_eval_every == 0:
            select_metrics = aggregate_proxy_metrics(
                model=model,
                history_01=selection_hist,
                future_01=selection_future,
                seeds=proxy_seeds,
                n_samples=args.selection_samples,
                batch_size=args.selection_batch_size,
            )
        else:
            select_metrics = {}

        row = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            **joint_metrics,
            **select_metrics,
        }
        history.append(row)
        history_path.write_text(json.dumps(make_serializable(history), indent=2))

        extra = {
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "selection_windows": len(selection_indices),
            "selection_metrics": row,
        }
        save_checkpoint(
            path=epoch_dir / f"epoch_{epoch:03d}.pt",
            model=model,
            epoch=epoch,
            encoder_config=encoder_config,
            decoder_config=decoder_config,
            flow_config=flow_config,
            args=args,
            extra={**extra, **row},
        )

        is_best_nll = val_metrics["val_total_loss"] < best_nll
        if is_best_nll:
            best_nll = val_metrics["val_total_loss"]
            best_nll_metrics = row
            save_checkpoint(
                path=outdir / "best_model.pt",
                model=model,
                epoch=epoch,
                encoder_config=encoder_config,
                decoder_config=decoder_config,
                flow_config=flow_config,
                args=args,
                extra={**extra, **row, "best_metrics": row},
            )

        is_best_select = False
        if select_metrics:
            key = selection_key(row)
            if best_select_key is None or key > best_select_key:
                best_select_key = key
                best_select_metrics = row
                is_best_select = True
                save_checkpoint(
                    path=outdir / "best_select_model.pt",
                    model=model,
                    epoch=epoch,
                    encoder_config=encoder_config,
                    decoder_config=decoder_config,
                    flow_config=flow_config,
                    args=args,
                    extra={
                        **extra,
                        **row,
                        "selection_key": list(key),
                        "best_select_metrics": row,
                    },
                )

        elapsed = time.time() - t0
        select_str = ""
        if select_metrics:
            select_str = (
                f"  sel_gates={select_metrics['select_gate_units_mean']:.2f}"
                f"  sel_violation={select_metrics['select_violation_mean']:.3f}"
                f"  sel_violation_std={select_metrics['select_violation_std']:.3f}"
                f"  sel_tc={select_metrics['select_turb_calm_ratio_mean']:.3f}"
                f"  sel_l2={select_metrics['select_layer2_n_passing_mean']:.2f}"
                f"  sel_l3={select_metrics['select_layer3_catastrophic_rate_mean']:.3f}"
                f"  sel_wf={select_metrics['select_window_floor_pct_bad_mean']:.3f}"
            )

        print(
            f"Ep {epoch:3d}  "
            f"train_loss={train_metrics['train_total_loss']:.4f}  "
            f"val_loss={val_metrics['val_total_loss']:.4f}  "
            f"cov90={joint_metrics['joint_cov90']:.4f}  "
            f"tc={joint_metrics['joint_turb_calm_ratio']:.3f}  "
            f"ld_rms={val_metrics['val_local_delta_rms']:.3f}"
            f"{select_str}  "
            f"({elapsed:.1f}s)"
            + ("  *best_nll" if is_best_nll else "")
            + ("  *best_select" if is_best_select else "")
        )

    summary = {
        "best_nll_epoch": None if best_nll_metrics is None else best_nll_metrics["epoch"],
        "best_select_epoch": None if best_select_metrics is None else best_select_metrics["epoch"],
        "best_nll_val_total_loss": best_nll,
        "best_select_key": None if best_select_key is None else list(best_select_key),
        "selection_split_size": len(selection_indices),
        "proxy_seeds": proxy_seeds,
    }
    (outdir / "study_summary.json").write_text(json.dumps(make_serializable(summary), indent=2))

    save_checkpoint(
        path=outdir / "final_model.pt",
        model=model,
        epoch=args.epochs,
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        flow_config=flow_config,
        args=args,
        extra={
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
            "selection_windows": len(selection_indices),
            **history[-1],
            "best_val_total_loss": best_nll,
            "best_nll_metrics": best_nll_metrics,
            "best_select_metrics": best_select_metrics,
            "best_select_key": None if best_select_key is None else list(best_select_key),
        },
    )

    print(f"\nBest val total loss: {best_nll:.4f}")
    if best_nll_metrics is not None:
        print(
            f"Best NLL epoch {best_nll_metrics['epoch']}: "
            f"sel_gates={best_nll_metrics.get('select_gate_units_mean')} "
            f"sel_violation={best_nll_metrics.get('select_violation_mean')}"
        )
    if best_select_metrics is not None:
        print(
            f"Best select epoch {best_select_metrics['epoch']}: "
            f"sel_gates={best_select_metrics['select_gate_units_mean']:.2f} "
            f"sel_violation={best_select_metrics['select_violation_mean']:.3f}"
        )


if __name__ == "__main__":
    main()
