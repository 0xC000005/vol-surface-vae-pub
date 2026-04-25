#!/usr/bin/env python
"""448a: train only conditional source-noise scale from the 392a core."""

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
    EmpiricalNormalScoreCausalMemoryTransitionFMConfig,
    EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    save_checkpoint,
)
from experiments.backfill.block_ar._rollout_220_utils import build_multistep_windows  # noqa: E402
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv  # noqa: E402
from experiments.backfill.block_ar.train_353a_340c_full_rollout_energy_finetune import (  # noqa: E402
    path_energy_score,
)


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


def load_with_conditional_noise_scale(
    checkpoint: str,
    device: torch.device,
    noise_scale_min: float,
    noise_scale_max: float,
    noise_scale_init: float,
) -> tuple[EmpiricalNormalScoreCausalMemoryTransitionFlowMatching, dict]:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg_dict = dict(payload["config"])
    cfg_dict.update(
        {
            "conditional_noise_scale": True,
            "noise_scale_min": float(noise_scale_min),
            "noise_scale_max": float(noise_scale_max),
        }
    )
    cfg = EmpiricalNormalScoreCausalMemoryTransitionFMConfig(**cfg_dict)
    model = EmpiricalNormalScoreCausalMemoryTransitionFlowMatching(cfg)
    result = model.load_state_dict(payload["model_state_dict"], strict=False)
    allowed_missing = {
        "noise_log_scale.0.weight",
        "noise_log_scale.0.bias",
        "noise_log_scale.1.weight",
        "noise_log_scale.1.bias",
    }
    missing = set(result.missing_keys) - allowed_missing
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint state mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(result.unexpected_keys)}"
        )
    init = float(noise_scale_init)
    if init <= 0.0:
        raise ValueError("--noise_scale_init must be positive")
    with torch.no_grad():
        model.noise_log_scale[-1].bias.fill_(math.log(init))
    model.to(device)
    return model, payload


def select_trainable_parameters(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    trainable_scope: str,
) -> list[torch.nn.Parameter]:
    trainable: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if trainable_scope == "noise_log_scale_only":
            param.requires_grad = name.startswith("noise_log_scale.")
        elif trainable_scope == "all":
            param.requires_grad = True
        else:
            raise ValueError(f"Unknown trainable_scope: {trainable_scope}")
        if param.requires_grad:
            trainable.append(param)
    if not trainable:
        raise RuntimeError("No parameters are trainable")
    return trainable


def sample_rollout_scores_with_conditional_scale(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    history_norm: torch.Tensor,
    n_samples: int,
    n_steps: int,
    flow_steps: int,
) -> torch.Tensor:
    history_scores = model.history_scores(history_norm)
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
    frames: list[torch.Tensor] = []
    for _step in range(n_steps):
        memory_state = model._encode_prefix_scores(prefix)[:, -1]
        current_score = prefix[:, -1]
        x = model.cfg.sample_temperature * torch.randn_like(current_score)
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


def combined_loss(
    model: EmpiricalNormalScoreCausalMemoryTransitionFlowMatching,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    train_sample_count: int,
    rollout_flow_steps: int,
    energy_eps: float,
    energy_weight: float,
    fm_anchor_weight: float,
    scale_l2_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    fm_loss, fm_metrics = model.training_loss(history_norm, future_norm)
    target_scores = model.target_future_scores(future_norm)
    sampled_scores = sample_rollout_scores_with_conditional_scale(
        model=model,
        history_norm=history_norm,
        n_samples=train_sample_count,
        n_steps=target_scores.shape[1],
        flow_steps=rollout_flow_steps,
    )
    energy, target_dist, pair_dist = path_energy_score(
        sampled_scores,
        target_scores,
        eps=energy_eps,
    )
    scale_penalty = torch.zeros((), device=history_norm.device, dtype=history_norm.dtype)
    for param in model.noise_log_scale.parameters():
        scale_penalty = scale_penalty + param.square().mean()
    total = (
        float(fm_anchor_weight) * fm_loss
        + float(energy_weight) * energy
        + float(scale_l2_weight) * scale_penalty
    )
    with torch.no_grad():
        history_scores = model.history_scores(history_norm)
        memory = model._encode_prefix_scores(history_scores)[:, -1]
        noise_scale = model._conditional_noise_scale(memory)
    metrics = {
        "total": total.detach(),
        "fm_loss": fm_loss.detach(),
        "energy": energy.detach(),
        "energy_target_dist": target_dist.detach(),
        "energy_pair_dist": pair_dist.detach(),
        "scale_l2": scale_penalty.detach(),
        "transition_std": fm_metrics["transition_std"].detach(),
        "sample_score_std": sampled_scores.std(unbiased=False).detach(),
        "target_score_std": target_scores.std(unbiased=False).detach(),
        "sample_h1_std": sampled_scores[:, :, 0].std(unbiased=False).detach(),
        "sample_h30_std": sampled_scores[:, :, -1].std(unbiased=False).detach(),
        "noise_scale_mean": noise_scale.mean().detach(),
        "noise_scale_std": noise_scale.std(unbiased=False).detach(),
        "noise_scale_min": noise_scale.min().detach(),
        "noise_scale_max": noise_scale.max().detach(),
    }
    return total, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/392a_recent_rollout_energy_w005_s42/best_model.pt",
    )
    parser.add_argument("--data_path", default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--adaptation_windows", type=int, default=441)
    parser.add_argument("--holdout_frac", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--train_sample_count", type=int, default=4)
    parser.add_argument("--rollout_flow_steps", type=int, default=4)
    parser.add_argument("--energy_eps", type=float, default=1e-6)
    parser.add_argument("--energy_weight", type=float, default=0.05)
    parser.add_argument("--fm_anchor_weight", type=float, default=1.0)
    parser.add_argument("--scale_l2_weight", type=float, default=0.001)
    parser.add_argument("--noise_scale_min", type=float, default=0.5)
    parser.add_argument("--noise_scale_max", type=float, default=2.0)
    parser.add_argument("--noise_scale_init", type=float, default=1.0)
    parser.add_argument(
        "--trainable_scope",
        choices=("noise_log_scale_only", "all"),
        default="noise_log_scale_only",
    )
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
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
    model, payload = load_with_conditional_noise_scale(
        args.checkpoint,
        device,
        noise_scale_min=args.noise_scale_min,
        noise_scale_max=args.noise_scale_max,
        noise_scale_init=args.noise_scale_init,
    )
    trainable = select_trainable_parameters(model, args.trainable_scope)
    model.train()
    if model.cfg.history_len != args.history_len or model.cfg.future_len != args.future_len:
        raise ValueError("Checkpoint horizon configuration does not match requested data")

    hist_01, fut_01, indices = build_recent_block(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        adaptation_windows=args.adaptation_windows,
        device=device,
    )
    n_total = hist_01.shape[0]
    n_val = max(1, int(round(n_total * float(args.holdout_frac))))
    n_train = n_total - n_val
    train_hist, val_hist = hist_01[:n_train], hist_01[n_train:]
    train_fut, val_fut = fut_01[:n_train], fut_01[n_train:]
    train_loader = DataLoader(
        TensorDataset(train_hist, train_fut),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        TensorDataset(val_hist, val_fut),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    def run_epoch(loader: DataLoader, train_mode: bool, max_batches: int) -> dict[str, float]:
        model.train(train_mode)
        sums: dict[str, float] = {}
        n_batches = 0
        for hist_batch, fut_batch in loader:
            if max_batches > 0 and n_batches >= max_batches:
                break
            hist_norm = normalize_iv(hist_batch.to(device, non_blocking=True)).view(
                hist_batch.shape[0],
                hist_batch.shape[1],
                -1,
            )
            fut_norm = normalize_iv(fut_batch.to(device, non_blocking=True)).view(
                fut_batch.shape[0],
                fut_batch.shape[1],
                -1,
            )
            with torch.set_grad_enabled(train_mode):
                loss, metrics = combined_loss(
                    model=model,
                    history_norm=hist_norm,
                    future_norm=fut_norm,
                    train_sample_count=args.train_sample_count,
                    rollout_flow_steps=args.rollout_flow_steps,
                    energy_eps=args.energy_eps,
                    energy_weight=args.energy_weight,
                    fm_anchor_weight=args.fm_anchor_weight,
                    scale_l2_weight=args.scale_l2_weight,
                )
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
                    optimizer.step()
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.item())
            n_batches += 1
        return {key: value / max(n_batches, 1) for key, value in sums.items()}

    print(f"Source checkpoint: {args.checkpoint}")
    print(f"Source epoch: {payload.get('epoch')}  source best_val: {payload.get('best_val')}")
    print(f"Recent windows: {n_total} index range: {indices[0]}..{indices[-1]}")
    print(f"Train/holdout: {n_train}/{n_val}")
    print(f"Trainable params: {sum(p.numel() for p in trainable):,}")
    print(f"Trainable scope: {args.trainable_scope}")
    print(f"Noise scale clamp: [{args.noise_scale_min}, {args.noise_scale_max}]")
    print(f"Noise scale init: {args.noise_scale_init}")

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
            **{f"train_{k}": v for k, v in train_avg.items()},
            **{f"val_{k}": v for k, v in val_avg.items()},
            "lr": optimizer.param_groups[0]["lr"],
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:03d}] train={rec['train_total']:.5f} "
            f"val={rec['val_total']:.5f} fm={rec['val_fm_loss']:.5f} "
            f"energy={rec['val_energy']:.5f} "
            f"scale={rec['val_noise_scale_mean']:.3f}+/-{rec['val_noise_scale_std']:.3f} "
            f"range={rec['val_noise_scale_min']:.3f}/{rec['val_noise_scale_max']:.3f} "
            f"h1/h30={rec['val_sample_h1_std']:.3f}/{rec['val_sample_h30_std']:.3f} "
            f"lr={rec['lr']:.2e} time={rec['sec']:.1f}s"
        )
        if rec["val_total"] < best_val:
            best_val = rec["val_total"]
            best_epoch = epoch
            save_checkpoint(str(out_dir / "best_model.pt"), model, model.cfg, epoch, best_val)

    save_checkpoint(str(out_dir / "final_model.pt"), model, model.cfg, args.epochs, best_val)
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
    summary = {
        "source_checkpoint": args.checkpoint,
        "source_epoch": payload.get("epoch"),
        "source_best_val": payload.get("best_val"),
        "adaptation_start_index": int(indices[0]),
        "adaptation_end_index": int(indices[-1]),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "trainable_scope": args.trainable_scope,
        "objective": {
            "fm_anchor_weight": args.fm_anchor_weight,
            "energy_weight": args.energy_weight,
            "scale_l2_weight": args.scale_l2_weight,
            "train_sample_count": args.train_sample_count,
            "rollout_flow_steps": args.rollout_flow_steps,
            "energy_eps": args.energy_eps,
            "noise_scale_min": args.noise_scale_min,
            "noise_scale_max": args.noise_scale_max,
            "noise_scale_init": args.noise_scale_init,
        },
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
