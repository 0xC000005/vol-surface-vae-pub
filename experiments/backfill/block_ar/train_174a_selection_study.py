#!/usr/bin/env python
"""
174a: Checkpoint-selection study on top of the 173a backbone.

Goal:
  determine whether the current H7 frontier is capped by model capacity or by
  selecting checkpoints using validation likelihood alone.

Keep the 173a model and loss exactly the same.
Change only:
  - save every epoch
  - run a small fixed validation proxy each epoch aligned with S2/S3/S7/S8
  - save best_nll and best_proxy checkpoints separately
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
    normalize_iv,
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


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def set_eval_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_proxy_samples(
    model: LocalVarianceResidualFlowStructuredJointStudentTModel,
    history_01: torch.Tensor,
    n_samples: int,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    batches = []
    for start in range(0, history_01.shape[0], batch_size):
        hist = history_01[start : start + batch_size]
        hist_norm = normalize_iv(hist)
        samples = model.sample_batched(hist_norm, n_samples=n_samples)
        batches.append(samples.detach().cpu().numpy())
    return np.concatenate(batches, axis=0)


def compute_proxy_metrics(
    cond_samples: np.ndarray,
    ground_truth: np.ndarray,
    history: np.ndarray,
) -> dict:
    horizons = [1, 7, 14, 30]
    horizon_targets = {1: 0.80, 7: 0.75, 14: 0.70, 30: 0.65}

    lower_90 = np.quantile(cond_samples, 0.05, axis=1)
    upper_90 = np.quantile(cond_samples, 0.95, axis=1)
    covered_90 = (ground_truth >= lower_90) & (ground_truth <= upper_90)

    overall_cov = {}
    for level in [0.5, 0.8, 0.9, 0.95]:
        alpha = (1.0 - level) / 2.0
        lo = np.quantile(cond_samples, alpha, axis=1)
        hi = np.quantile(cond_samples, 1.0 - alpha, axis=1)
        overall_cov[level] = float(((ground_truth >= lo) & (ground_truth <= hi)).mean())

    calib_nominal = []
    calib_empirical = []
    for p in np.linspace(0.1, 0.9, 9):
        alpha = (1.0 - p) / 2.0
        lo = np.quantile(cond_samples, alpha, axis=1)
        hi = np.quantile(cond_samples, 1.0 - alpha, axis=1)
        calib_nominal.append(float(p))
        calib_empirical.append(float(((ground_truth >= lo) & (ground_truth <= hi)).mean()))
    calibration_error = float(
        np.mean(np.abs(np.array(calib_nominal) - np.array(calib_empirical)))
    )

    per_h_cov = {}
    worst_cell = {}
    best_cell = {}
    horizon_pass_count = 0
    cell_low_pass_count = 0
    cell_high_pass_count = 0
    for h in horizons:
        t = h - 1
        cov_h = float(covered_90[:, t].mean())
        per_h_cov[h] = cov_h
        horizon_pass_count += int(cov_h > horizon_targets[h])
        cell_cov = covered_90[:, t].mean(axis=0)
        worst = float(cell_cov.min())
        best = float(cell_cov.max())
        worst_cell[h] = worst
        best_cell[h] = best
        cell_low_pass_count += int(worst >= 0.70)
        cell_high_pass_count += int(best <= 0.95)

    ci_width = upper_90 - lower_90
    mean_iv = history.mean(axis=(2, 3))
    daily_changes = np.diff(mean_iv, axis=1)
    vol_of_vol = daily_changes.std(axis=1)
    q20 = np.quantile(vol_of_vol, 0.20)
    q80 = np.quantile(vol_of_vol, 0.80)
    calm_mask = vol_of_vol <= q20
    turb_mask = vol_of_vol >= q80

    per_window_width = ci_width.mean(axis=(1, 2, 3))
    if calm_mask.any() and turb_mask.any():
        turb_calm_ratio = float(per_window_width[turb_mask].mean() / per_window_width[calm_mask].mean())
    else:
        turb_calm_ratio = float("nan")
    turb_calm_pass = turb_calm_ratio > 1.15

    layer1_pass_count = 0
    layer2_n_passing = 0
    layer2_n_total = 0
    layer1_results = {}
    for regime_name, regime_mask in [("calm", calm_mask), ("turb", turb_mask)]:
        layer1_results[regime_name] = {}
        for h in horizons:
            t = h - 1
            cov = float(covered_90[regime_mask, t].mean()) if regime_mask.any() else float("nan")
            layer1_results[regime_name][h] = cov
            passed = cov > 0.65 if np.isfinite(cov) else False
            layer1_pass_count += int(passed)

            layer2_n_total += 1
            cell_cov = covered_90[regime_mask, t].mean(axis=0)
            worst = float(cell_cov.min())
            best = float(cell_cov.max())
            if worst >= 0.70 and best <= 0.95:
                layer2_n_passing += 1

    window_cell_cov = covered_90.mean(axis=1)
    catastrophic_rate = float((window_cell_cov < 0.30).mean())
    layer3_pass = catastrophic_rate < 0.05

    per_window_cov = covered_90.mean(axis=(1, 2, 3))
    window_floor_pct = float((per_window_cov < 0.50).mean())
    window_floor_pass = window_floor_pct < 0.05

    gate_units = (
        horizon_pass_count
        + cell_low_pass_count
        + cell_high_pass_count
        + int(turb_calm_pass)
        + layer1_pass_count
        + layer2_n_passing
        + int(layer3_pass)
        + int(window_floor_pass)
    )

    violation = 0.0
    for h, target in horizon_targets.items():
        violation += max(0.0, target - per_h_cov[h]) / target
        violation += max(0.0, 0.70 - worst_cell[h]) / 0.70
        violation += max(0.0, best_cell[h] - 0.95) / 0.05
    violation += max(0.0, 1.15 - turb_calm_ratio) / 1.15
    for regime_name in ["calm", "turb"]:
        for h in horizons:
            violation += max(0.0, 0.65 - layer1_results[regime_name][h]) / 0.65
    violation += (layer2_n_total - layer2_n_passing) / max(layer2_n_total, 1)
    violation += max(0.0, catastrophic_rate - 0.05) / 0.05
    violation += max(0.0, window_floor_pct - 0.05) / 0.05
    violation += calibration_error / 0.10

    return {
        "proxy_overall_cov90": overall_cov[0.9],
        "proxy_calibration_error": calibration_error,
        "proxy_horizon_pass_count": horizon_pass_count,
        "proxy_cell_low_pass_count": cell_low_pass_count,
        "proxy_cell_high_pass_count": cell_high_pass_count,
        "proxy_worst_h1": worst_cell[1],
        "proxy_worst_h7": worst_cell[7],
        "proxy_worst_h14": worst_cell[14],
        "proxy_worst_h30": worst_cell[30],
        "proxy_best_h1": best_cell[1],
        "proxy_best_h7": best_cell[7],
        "proxy_best_h14": best_cell[14],
        "proxy_best_h30": best_cell[30],
        "proxy_turb_calm_ratio": turb_calm_ratio,
        "proxy_turb_calm_pass": turb_calm_pass,
        "proxy_layer1_pass_count": layer1_pass_count,
        "proxy_layer2_n_passing": layer2_n_passing,
        "proxy_layer2_n_total": layer2_n_total,
        "proxy_layer3_catastrophic_rate": catastrophic_rate,
        "proxy_layer3_pass": layer3_pass,
        "proxy_window_floor_pct_bad": window_floor_pct,
        "proxy_window_floor_pass": window_floor_pass,
        "proxy_gate_units": int(gate_units),
        "proxy_violation": float(violation),
    }


@torch.no_grad()
def evaluate_proxy_subset(
    model: LocalVarianceResidualFlowStructuredJointStudentTModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    n_samples: int,
    batch_size: int,
    seed: int,
) -> dict:
    rng_state = capture_rng_state()
    try:
        set_eval_seed(seed)
        cond_samples = generate_proxy_samples(
            model=model,
            history_01=history_01,
            n_samples=n_samples,
            batch_size=batch_size,
        )
    finally:
        restore_rng_state(rng_state)

    ground_truth = future_01.view(future_01.shape[0], future_01.shape[1], 5, 5).detach().cpu().numpy()
    history_np = history_01.detach().cpu().numpy()
    return compute_proxy_metrics(cond_samples, ground_truth, history_np)


def proxy_selection_key(row: dict) -> tuple[float, float, float]:
    return (
        float(row["proxy_gate_units"]),
        -float(row["proxy_violation"]),
        -float(row["val_total_loss"]),
    )


def save_checkpoint(
    path: Path,
    model: LocalVarianceResidualFlowStructuredJointStudentTModel,
    epoch: int,
    encoder_config: EncoderConfig,
    decoder_config: dict,
    flow_config: dict,
    args: argparse.Namespace,
    extra: dict,
) -> None:
    payload = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "val_total_loss": extra.get("val_total_loss"),
        "config": {
            "type": "local_var_residual_flow_structured_joint_student_t_173a",
            "encoder": vars(encoder_config),
            "decoder": decoder_config,
            "flow": flow_config,
            "support_lo": args.support_lo,
            "support_hi": args.support_hi,
            "support_eps": args.support_eps,
            "base_nu": args.base_nu,
            "history_len": args.history_len,
            "future_len": args.future_len,
            "train_windows": int(extra["train_windows"]),
            "val_windows": int(extra["val_windows"]),
            "local_var_penalty": args.local_var_penalty,
        },
        **extra,
    }
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="174a: checkpoint-selection study on 173a backbone")
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
    parser.add_argument("--proxy_val_windows", type=int, default=192)
    parser.add_argument("--proxy_val_samples", type=int, default=8)
    parser.add_argument("--proxy_batch_size", type=int, default=32)
    parser.add_argument("--proxy_eval_every", type=int, default=1)
    parser.add_argument("--proxy_seed", type=int, default=20260404)
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

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)

    hist_len = args.history_len
    future_len = args.future_len
    test_start = 4511
    max_train_idx = test_start - hist_len - future_len
    val_size = 441
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    if args.max_train_windows is not None:
        train_indices = train_indices[: args.max_train_windows]
    if args.max_val_windows is not None:
        val_indices = val_indices[: args.max_val_windows]

    print(f"Train: {len(train_indices)}, Val: {len(val_indices)}")

    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(train_indices, surf_tensor, hist_len, future_len)
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

    proxy_n = min(args.proxy_val_windows, val_hist.shape[0])
    proxy_history = val_hist[:proxy_n]
    proxy_future = val_future[:proxy_n]

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
    print("174a: 173a backbone with checkpoint-selection study")
    print("=" * 64)
    print(f"  Total params: {n_params:,}")
    print(f"  Proxy validation windows: {proxy_n}")
    print(f"  Proxy samples/window: {args.proxy_val_samples}")
    print("  Selection objective: gate_units -> violation -> val_total_loss")

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
    best_proxy_key = None
    best_proxy_metrics = None
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
            proxy_metrics = evaluate_proxy_subset(
                model,
                history_01=proxy_history,
                future_01=proxy_future,
                n_samples=args.proxy_val_samples,
                batch_size=args.proxy_batch_size,
                seed=args.proxy_seed,
            )
        else:
            proxy_metrics = {}

        row = {
            "epoch": epoch,
            **train_metrics,
            **val_metrics,
            **joint_metrics,
            **proxy_metrics,
        }
        history.append(row)
        history_path.write_text(json.dumps(make_serializable(history), indent=2))

        extra = {
            "train_windows": len(train_indices),
            "val_windows": len(val_indices),
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

        is_best_proxy = False
        if proxy_metrics:
            current_key = proxy_selection_key(row)
            if best_proxy_key is None or current_key > best_proxy_key:
                best_proxy_key = current_key
                best_proxy_metrics = row
                is_best_proxy = True
                save_checkpoint(
                    path=outdir / "best_proxy_model.pt",
                    model=model,
                    epoch=epoch,
                    encoder_config=encoder_config,
                    decoder_config=decoder_config,
                    flow_config=flow_config,
                    args=args,
                    extra={
                        **extra,
                        **row,
                        "proxy_selection_key": list(current_key),
                        "best_proxy_metrics": row,
                    },
                )

        elapsed = time.time() - t0
        proxy_str = ""
        if proxy_metrics:
            proxy_str = (
                f"  proxy_gates={proxy_metrics['proxy_gate_units']:2d}"
                f"  proxy_violation={proxy_metrics['proxy_violation']:.3f}"
                f"  proxy_tc={proxy_metrics['proxy_turb_calm_ratio']:.3f}"
                f"  proxy_l2={proxy_metrics['proxy_layer2_n_passing']}/{proxy_metrics['proxy_layer2_n_total']}"
                f"  proxy_l3={proxy_metrics['proxy_layer3_catastrophic_rate']:.3f}"
                f"  proxy_wf={proxy_metrics['proxy_window_floor_pct_bad']:.3f}"
            )

        print(
            f"Ep {epoch:3d}  "
            f"train_loss={train_metrics['train_total_loss']:.4f}  "
            f"val_loss={val_metrics['val_total_loss']:.4f}  "
            f"cov90={joint_metrics['joint_cov90']:.4f}  "
            f"tc={joint_metrics['joint_turb_calm_ratio']:.3f}  "
            f"ld_rms={val_metrics['val_local_delta_rms']:.3f}"
            f"{proxy_str}  "
            f"({elapsed:.1f}s)"
            + ("  *best_nll" if is_best_nll else "")
            + ("  *best_proxy" if is_best_proxy else "")
        )

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
            **history[-1],
            "best_val_total_loss": best_nll,
            "best_nll_metrics": best_nll_metrics,
            "best_proxy_metrics": best_proxy_metrics,
            "best_proxy_key": list(best_proxy_key) if best_proxy_key is not None else None,
        },
    )

    print(f"\nBest val total loss: {best_nll:.4f}")
    if best_nll_metrics is not None:
        print(
            f"Best NLL epoch {best_nll_metrics['epoch']}: "
            f"proxy_gates={best_nll_metrics.get('proxy_gate_units')} "
            f"proxy_violation={best_nll_metrics.get('proxy_violation')}"
        )
    if best_proxy_metrics is not None:
        print(
            f"Best proxy epoch {best_proxy_metrics['epoch']}: "
            f"proxy_gates={best_proxy_metrics['proxy_gate_units']} "
            f"proxy_violation={best_proxy_metrics['proxy_violation']:.3f}"
        )


if __name__ == "__main__":
    main()
