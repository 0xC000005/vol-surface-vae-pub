#!/usr/bin/env python
"""
207b: teacher-guided local-family AR.

Keep the 207a architecture fixed and only add offline teacher supervision for the
danger gate and local family branch.
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar.analyze_202a_discrete_innovation_mode_pretest import (
    build_window_metadata,
    load_model as load_teacher_model,
)
from experiments.backfill.block_ar.analyze_205a_conditional_shape_family_audit import (
    collect_teacher_forced_records,
)
from experiments.backfill.block_ar.analyze_206b_local_conditional_scenario_family_pretest import (
    farthest_first_subset,
    fit_feature_space,
    transform_feat,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import make_serializable
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_201a_transformer_ar_rollout_tail_student_t import (
    evaluate_rollout_subset,
    maybe_load_decoder_warm_start,
)
from experiments.backfill.block_ar.train_201b_underfit_aware_selffed_rollout_student_t import (
    evaluate_selffed_rollout,
    evaluate_teacher_forced,
    selffed_rollout_likelihood_objective,
    teacher_forced_multistep_objective,
)
from experiments.backfill.block_ar.train_207a_local_conditional_family_student_t import (
    TransformerLocalConditionalFamilyARModel,
    evaluate_family_statistics,
)


PERMUTATIONS_3 = list(itertools.permutations(range(3)))


def softmax_np(x: np.ndarray, temp: float) -> np.ndarray:
    z = -x / max(temp, 1e-6)
    z = z - z.max()
    e = np.exp(z)
    return e / np.clip(e.sum(), 1e-8, None)


def pad_family(fam: np.ndarray, family_size: int) -> np.ndarray:
    if fam.shape[0] >= family_size:
        return fam[:family_size]
    if fam.shape[0] == 0:
        return np.zeros((family_size, 25), dtype=np.float32)
    pad = np.repeat(fam[-1:], family_size - fam.shape[0], axis=0)
    return np.concatenate([fam, pad], axis=0)


def build_teacher_targets_for_split(
    train_records: dict[str, np.ndarray],
    split_records: dict[str, np.ndarray],
    n_windows: int,
    future_len: int,
    family_size: int,
    knn: int,
    target_temp: float,
    exclude_self: bool,
) -> dict[str, torch.Tensor]:
    gate_targets = np.zeros((n_windows, future_len), dtype=np.float32)
    family_mask = np.zeros((n_windows, future_len), dtype=np.float32)
    family_shapes = np.zeros((n_windows, future_len, family_size, 25), dtype=np.float32)
    family_probs = np.zeros((n_windows, future_len, family_size), dtype=np.float32)

    train_pool_mask = train_records["q99_any"] == 1
    train_pool_idx = np.flatnonzero(train_pool_mask)
    if train_pool_idx.size == 0:
        return {
            "gate_targets": torch.from_numpy(gate_targets),
            "family_mask": torch.from_numpy(family_mask),
            "family_shapes": torch.from_numpy(family_shapes),
            "family_probs": torch.from_numpy(family_probs),
        }

    feat_bundle, train_z = fit_feature_space(train_records["cond_feat"][train_pool_idx], pca_dim=32)
    nn = NearestNeighbors(n_neighbors=min(knn + int(exclude_self), train_pool_idx.size), metric="euclidean")
    nn.fit(train_z)
    split_z = transform_feat(split_records["cond_feat"], feat_bundle)
    _, nbr_pos = nn.kneighbors(split_z, return_distance=True)
    nbr_idx = train_pool_idx[nbr_pos]

    for q_abs in range(split_records["step"].shape[0]):
        window_idx = int(split_records["window_idx"][q_abs])
        step_idx = int(split_records["step"][q_abs])
        gate_targets[window_idx, step_idx] = float(split_records["q95_any"][q_abs])

        if split_records["q99_any"][q_abs] != 1:
            continue

        neighbors = nbr_idx[q_abs]
        if exclude_self:
            neighbors = neighbors[neighbors != q_abs]
        if neighbors.size == 0:
            neighbors = nbr_idx[q_abs][:1]

        fam = train_records["signed_delta_norm"][neighbors]
        fam = pad_family(fam, family_size=max(family_size, 1))
        fam_sel = farthest_first_subset(fam, family_size)
        fam = pad_family(fam[fam_sel], family_size)

        query = split_records["signed_delta_norm"][q_abs]
        mae = np.abs(fam - query[None, :]).mean(axis=1)
        order = np.argsort(mae)
        fam = fam[order]
        mae = mae[order]
        probs = softmax_np(mae, temp=target_temp).astype(np.float32)

        family_mask[window_idx, step_idx] = 1.0
        family_shapes[window_idx, step_idx] = fam.astype(np.float32)
        family_probs[window_idx, step_idx] = probs

    return {
        "gate_targets": torch.from_numpy(gate_targets),
        "family_mask": torch.from_numpy(family_mask),
        "family_shapes": torch.from_numpy(family_shapes),
        "family_probs": torch.from_numpy(family_probs),
    }


def build_teacher_guidance_targets(
    teacher_checkpoint: str,
    train_history: torch.Tensor,
    train_future: torch.Tensor,
    val_history: torch.Tensor,
    val_future: torch.Tensor,
    q95_threshold: float,
    q99_threshold: float,
    batch_size: int,
    device: torch.device,
    family_size: int,
    knn: int,
    target_temp: float,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    teacher_model, _payload = load_teacher_model(teacher_checkpoint, device)
    train_future_flat = train_future.view(train_future.shape[0], train_future.shape[1], -1).detach().cpu().numpy()
    val_future_flat = val_future.view(val_future.shape[0], val_future.shape[1], -1).detach().cpu().numpy()
    train_history_np = train_history.detach().cpu().numpy()
    val_history_np = val_history.detach().cpu().numpy()

    train_meta = build_window_metadata(train_history_np, train_future_flat)
    val_meta = build_window_metadata(
        val_history_np,
        val_future_flat,
        q80_vov_train=train_meta["q80_vov"],
        q80_h30_turb_train=train_meta["q80_h30_turb"],
    )

    train_records = collect_teacher_forced_records(
        model=teacher_model,
        history_01=train_history,
        future_flat=train_future.view(train_future.shape[0], train_future.shape[1], -1),
        window_meta=train_meta,
        split_name="train",
        q95=q95_threshold,
        q99=q99_threshold,
        batch_size=batch_size,
        device=device,
    )
    val_records = collect_teacher_forced_records(
        model=teacher_model,
        history_01=val_history,
        future_flat=val_future.view(val_future.shape[0], val_future.shape[1], -1),
        window_meta=val_meta,
        split_name="val",
        q95=q95_threshold,
        q99=q99_threshold,
        batch_size=batch_size,
        device=device,
    )

    train_targets = build_teacher_targets_for_split(
        train_records=train_records,
        split_records=train_records,
        n_windows=train_history.shape[0],
        future_len=train_future.shape[1],
        family_size=family_size,
        knn=knn,
        target_temp=target_temp,
        exclude_self=True,
    )
    val_targets = build_teacher_targets_for_split(
        train_records=train_records,
        split_records=val_records,
        n_windows=val_history.shape[0],
        future_len=val_future.shape[1],
        family_size=family_size,
        knn=knn,
        target_temp=target_temp,
        exclude_self=False,
    )
    return train_targets, val_targets


def family_teacher_guidance_objective(
    model: TransformerLocalConditionalFamilyARModel,
    history_01: torch.Tensor,
    future_01: torch.Tensor,
    gate_targets: torch.Tensor,
    family_mask: torch.Tensor,
    family_shapes_target: torch.Tensor,
    family_probs_target: torch.Tensor,
    gate_pos_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    context = history_01.clone()
    gate_losses = []
    shape_losses = []
    prob_losses = []
    top1_matches = []
    family_active = []

    family_size = family_shapes_target.shape[2]
    if family_size != 3:
        raise ValueError("207b currently expects family_size=3")

    pos_weight = history_01.new_tensor(gate_pos_weight)

    for step in range(future_01.shape[1]):
        _mu, _factor, _diag, _scale, _nu = model.forward_from_history(context)
        family = model.last_family_params()

        gate_t = gate_targets[:, step].to(history_01.device)
        gate_loss = F.binary_cross_entropy_with_logits(
            family["gate_logit"],
            gate_t,
            pos_weight=pos_weight,
            reduction="mean",
        )
        gate_losses.append(gate_loss)

        mask = family_mask[:, step].to(history_01.device) > 0.5
        family_active.append(mask.float().mean())
        if mask.any():
            pred_shapes = family["family_shapes"][mask]
            pred_logits = family["family_logits"][mask]
            tgt_shapes = family_shapes_target[:, step].to(history_01.device)[mask]
            tgt_probs = family_probs_target[:, step].to(history_01.device)[mask]

            shape_loss_batch = []
            prob_loss_batch = []
            top1_batch = []
            for i in range(pred_shapes.shape[0]):
                best_cost = None
                best_probs = None
                best_top1 = None
                best_cost_value = None
                for perm in PERMUTATIONS_3:
                    perm_idx = list(perm)
                    aligned_shapes = tgt_shapes[i, perm_idx]
                    aligned_probs = tgt_probs[i, perm_idx]
                    shape_cost = (aligned_probs * (pred_shapes[i] - aligned_shapes).abs().mean(dim=-1)).sum()
                    shape_cost_value = float(shape_cost.detach().item())
                    if best_cost is None or shape_cost_value < best_cost_value:
                        best_cost = shape_cost
                        best_probs = aligned_probs
                        best_top1 = float(pred_logits[i].argmax().item() == int(aligned_probs.argmax().item()))
                        best_cost_value = shape_cost_value
                shape_loss_batch.append(best_cost)
                prob_loss_batch.append(-(best_probs * F.log_softmax(pred_logits[i], dim=-1)).sum())
                top1_batch.append(best_top1)

            shape_losses.append(torch.stack(shape_loss_batch).mean())
            prob_losses.append(torch.stack(prob_loss_batch).mean())
            top1_matches.append(history_01.new_tensor(top1_batch).mean())

        next_frame = future_01[:, step : step + 1]
        if context.dim() == 4 and next_frame.dim() == 3:
            next_frame = next_frame.view(next_frame.shape[0], 1, *context.shape[2:])
        context = torch.cat([context[:, 1:], next_frame], dim=1)

    zero = history_01.new_tensor(0.0)
    total_gate = torch.stack(gate_losses).mean() if gate_losses else zero
    total_shape = torch.stack(shape_losses).mean() if shape_losses else zero
    total_prob = torch.stack(prob_losses).mean() if prob_losses else zero
    total_top1 = torch.stack(top1_matches).mean() if top1_matches else zero
    total_family_active = torch.stack(family_active).mean() if family_active else zero

    total = total_gate + total_shape + total_prob
    metrics = {
        "teacher_guidance_total_loss": total.detach(),
        "teacher_gate_bce": total_gate.detach(),
        "teacher_family_shape_loss": total_shape.detach(),
        "teacher_family_prob_loss": total_prob.detach(),
        "teacher_gate_target_rate": gate_targets.mean().detach(),
        "teacher_family_mask_rate": family_mask.mean().detach(),
        "teacher_family_top1_match": total_top1.detach(),
        "teacher_family_active_rate": total_family_active.detach(),
    }
    return total, metrics


@torch.no_grad()
def evaluate_teacher_guidance(
    model: TransformerLocalConditionalFamilyARModel,
    loader: DataLoader,
    gate_pos_weight: float,
) -> dict[str, float]:
    model.eval()
    totals = {
        "val_teacher_guidance_total_loss": 0.0,
        "val_teacher_gate_bce": 0.0,
        "val_teacher_family_shape_loss": 0.0,
        "val_teacher_family_prob_loss": 0.0,
        "val_teacher_gate_target_rate": 0.0,
        "val_teacher_family_mask_rate": 0.0,
        "val_teacher_family_top1_match": 0.0,
        "val_teacher_family_active_rate": 0.0,
    }
    count = 0
    for history_01, future_01, gate_targets, family_mask, family_shapes, family_probs in loader:
        loss, metrics = family_teacher_guidance_objective(
            model,
            history_01,
            future_01,
            gate_targets,
            family_mask,
            family_shapes,
            family_probs,
            gate_pos_weight=gate_pos_weight,
        )
        bs = history_01.shape[0]
        totals["val_teacher_guidance_total_loss"] += float(loss.item()) * bs
        totals["val_teacher_gate_bce"] += float(metrics["teacher_gate_bce"].item()) * bs
        totals["val_teacher_family_shape_loss"] += float(metrics["teacher_family_shape_loss"].item()) * bs
        totals["val_teacher_family_prob_loss"] += float(metrics["teacher_family_prob_loss"].item()) * bs
        totals["val_teacher_gate_target_rate"] += float(metrics["teacher_gate_target_rate"].item()) * bs
        totals["val_teacher_family_mask_rate"] += float(metrics["teacher_family_mask_rate"].item()) * bs
        totals["val_teacher_family_top1_match"] += float(metrics["teacher_family_top1_match"].item()) * bs
        totals["val_teacher_family_active_rate"] += float(metrics["teacher_family_active_rate"].item()) * bs
        count += bs
    return {k: v / max(count, 1) for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="207b teacher-guided local-family Student-t AR")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr_encoder", type=float, default=3e-4)
    parser.add_argument("--lr_decoder", type=float, default=1e-3)
    parser.add_argument("--weight_decay_encoder", type=float, default=0.01)
    parser.add_argument("--weight_decay_decoder", type=float, default=0.01)
    parser.add_argument("--enc_d_model", type=int, default=128)
    parser.add_argument("--enc_heads", type=int, default=4)
    parser.add_argument("--enc_layers", type=int, default=4)
    parser.add_argument("--enc_dropout", type=float, default=0.1)
    parser.add_argument("--dec_d_model", type=int, default=128)
    parser.add_argument("--dec_heads", type=int, default=4)
    parser.add_argument("--dec_layers", type=int, default=4)
    parser.add_argument("--rank", type=int, default=5)
    parser.add_argument("--diag_floor", type=float, default=1e-3)
    parser.add_argument("--scale_floor", type=float, default=1e-4)
    parser.add_argument("--init_diag", type=float, default=0.10)
    parser.add_argument("--init_scale", type=float, default=0.10)
    parser.add_argument("--nu_floor", type=float, default=2.1)
    parser.add_argument("--nu_init", type=float, default=8.0)
    parser.add_argument("--nu_max", type=float, default=100.0)
    parser.add_argument("--fixed_nu", type=float, default=8.0)
    parser.add_argument("--cov_jitter", type=float, default=1e-4)
    parser.add_argument("--support_lo", type=float, default=0.01)
    parser.add_argument("--support_hi", type=float, default=1.0)
    parser.add_argument("--support_eps", type=float, default=1e-5)
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--tail_weight", type=float, default=6.0)
    parser.add_argument("--tail_mae_weight", type=float, default=0.05)
    parser.add_argument("--underfit_z_gate", type=float, default=1.8)
    parser.add_argument("--overwidth_weight", type=float, default=0.02)
    parser.add_argument("--rollout_weight", type=float, default=0.25)
    parser.add_argument("--rollout_steps", type=int, default=5)
    parser.add_argument("--rollout_warmup_epochs", type=int, default=3)
    parser.add_argument("--rollout_ramp_epochs", type=int, default=3)
    parser.add_argument("--rollout_val_samples", type=int, default=8)
    parser.add_argument("--rollout_eval_limit", type=int, default=64)
    parser.add_argument("--max_train_windows", type=int, default=None)
    parser.add_argument("--max_val_windows", type=int, default=None)
    parser.add_argument("--decoder_warm_start", type=str, default="models/backfill/student_t_169c/best_model.pt")
    parser.add_argument("--teacher_checkpoint", type=str, default="models/backfill/transformer_underfit_aware_selffed_rollout_student_t_201b/best_model.pt")
    parser.add_argument("--n_family", type=int, default=3)
    parser.add_argument("--family_scale_floor", type=float, default=5e-4)
    parser.add_argument("--init_gate_prob", type=float, default=0.10)
    parser.add_argument("--init_family_scale", type=float, default=0.05)
    parser.add_argument("--family_shape_scale", type=float, default=0.05)
    parser.add_argument("--family_nu", type=float, default=4.0)
    parser.add_argument("--gate_temperature", type=float, default=0.5)
    parser.add_argument("--family_temperature", type=float, default=0.6)
    parser.add_argument("--gate_prob_eps", type=float, default=1e-4)
    parser.add_argument("--teacher_guidance_weight", type=float, default=0.25)
    parser.add_argument("--teacher_target_temp", type=float, default=0.05)
    parser.add_argument("--teacher_knn", type=int, default=32)
    parser.add_argument("--gate_pos_weight", type=float, default=4.0)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    train_abs_delta = np.abs(np.diff(surfaces[:4511], axis=0).reshape(-1))
    q95_threshold = float(np.quantile(train_abs_delta, 0.95))
    q99_threshold = float(np.quantile(train_abs_delta, 0.99))

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

    train_targets, val_targets = build_teacher_guidance_targets(
        teacher_checkpoint=args.teacher_checkpoint,
        train_history=train_hist,
        train_future=train_future,
        val_history=val_hist,
        val_future=val_future,
        q95_threshold=q95_threshold,
        q99_threshold=q99_threshold,
        batch_size=args.batch_size,
        device=device,
        family_size=args.n_family,
        knn=args.teacher_knn,
        target_temp=args.teacher_target_temp,
    )

    train_loader = DataLoader(
        TensorDataset(
            train_hist,
            train_future,
            train_targets["gate_targets"].to(device),
            train_targets["family_mask"].to(device),
            train_targets["family_shapes"].to(device),
            train_targets["family_probs"].to(device),
        ),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(
            val_hist,
            val_future,
            val_targets["gate_targets"].to(device),
            val_targets["family_mask"].to(device),
            val_targets["family_shapes"].to(device),
            val_targets["family_probs"].to(device),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    rollout_val_loader = DataLoader(TensorDataset(val_hist, val_future), batch_size=args.batch_size, shuffle=False)

    encoder_config = dict(
        input_dim=25,
        d_model=args.enc_d_model,
        n_heads=args.enc_heads,
        n_layers=args.enc_layers,
        dropout=args.enc_dropout,
        bottleneck_dim=128,
        max_len=max(hist_len, 64),
    )
    decoder_config = dict(
        n_cells=25,
        d_model=args.dec_d_model,
        n_heads=args.dec_heads,
        n_layers=args.dec_layers,
        cond_dim=128,
        rank=args.rank,
        diag_floor=args.diag_floor,
        scale_floor=args.scale_floor,
        nu_floor=args.nu_floor,
        nu_max=args.nu_max,
        init_diag=args.init_diag,
        init_scale=args.init_scale,
        init_nu=args.nu_init,
        fixed_nu=args.fixed_nu,
        n_family=args.n_family,
        family_scale_floor=args.family_scale_floor,
        init_gate_prob=args.init_gate_prob,
        init_family_scale=args.init_family_scale,
        family_shape_scale=args.family_shape_scale,
        family_nu=args.family_nu,
        gate_temperature=args.gate_temperature,
        family_temperature=args.family_temperature,
        gate_prob_eps=args.gate_prob_eps,
    )
    objective_config = dict(
        q95_threshold=q95_threshold,
        q99_threshold=q99_threshold,
        tail_weight=args.tail_weight,
        tail_mae_weight=args.tail_mae_weight,
        underfit_z_gate=args.underfit_z_gate,
        overwidth_weight=args.overwidth_weight,
        rollout_steps=args.rollout_steps,
    )

    model = TransformerLocalConditionalFamilyARModel(
        encoder_config=encoder_config,
        decoder_config=decoder_config,
        support_lo=args.support_lo,
        support_hi=args.support_hi,
        support_eps=args.support_eps,
        cov_jitter=args.cov_jitter,
    ).to(device)
    maybe_load_decoder_warm_start(model, args.decoder_warm_start)

    n_enc = sum(p.numel() for p in model.encoder.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print(f"\n{'=' * 76}")
    print("207b: Transformer AR Student-t with teacher-guided local family")
    print(f"{'=' * 76}")
    print(f"  Encoder params: {n_enc:,}")
    print(f"  Decoder params: {n_dec:,}")
    print(f"  Total params:   {n_enc + n_dec:,}")
    print(f"  Tail thresholds: q95={q95_threshold:.5f}, q99={q99_threshold:.5f}")
    print(f"  Self-fed rollout steps: {args.rollout_steps}, weight={args.rollout_weight}")
    print(f"  Teacher checkpoint: {args.teacher_checkpoint}")
    print(f"  Teacher KNN / family size: {args.teacher_knn} / {args.n_family}")
    print(f"  Teacher guidance weight: {args.teacher_guidance_weight}")

    optimizer = torch.optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": args.lr_encoder, "weight_decay": args.weight_decay_encoder},
            {"params": model.decoder.parameters(), "lr": args.lr_decoder, "weight_decay": args.weight_decay_decoder},
        ]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_score = float("inf")
    best_metrics = None
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        ep = {
            "teacher_total_loss": 0.0,
            "rollout_total_loss": 0.0,
            "teacher_guidance_total_loss": 0.0,
            "teacher_gate_bce": 0.0,
            "teacher_family_shape_loss": 0.0,
            "teacher_family_prob_loss": 0.0,
            "teacher_gate_target_rate": 0.0,
            "teacher_family_mask_rate": 0.0,
            "teacher_family_top1_match": 0.0,
            "teacher_family_active_rate": 0.0,
        }
        nb = 0

        if epoch <= args.rollout_warmup_epochs:
            rollout_scale = 0.0
        else:
            progress = (epoch - args.rollout_warmup_epochs) / max(args.rollout_ramp_epochs, 1)
            rollout_scale = float(min(max(progress, 0.0), 1.0))

        for history_01, future_01, gate_targets, family_mask, family_shapes, family_probs in train_loader:
            optimizer.zero_grad()

            teacher_loss, teacher_metrics = teacher_forced_multistep_objective(
                model, history_01, future_01, objective_config
            )
            guidance_loss, guidance_metrics = family_teacher_guidance_objective(
                model,
                history_01,
                future_01,
                gate_targets,
                family_mask,
                family_shapes,
                family_probs,
                gate_pos_weight=args.gate_pos_weight,
            )

            loss = teacher_loss + args.teacher_guidance_weight * guidance_loss

            rollout_metrics = {
                "rollout_total_loss": torch.tensor(0.0, device=history_01.device),
            }
            if rollout_scale > 0.0 and args.rollout_weight > 0.0 and args.rollout_steps > 0:
                rollout_loss, rollout_full_metrics = selffed_rollout_likelihood_objective(
                    model, history_01, future_01, objective_config
                )
                loss = loss + (args.rollout_weight * rollout_scale) * rollout_loss
                rollout_metrics["rollout_total_loss"] = rollout_full_metrics["rollout_total_loss"]

            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss detected at epoch {epoch}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep["teacher_total_loss"] += float(teacher_metrics["teacher_total_loss"].item())
            ep["rollout_total_loss"] += float(rollout_metrics["rollout_total_loss"].item())
            for k in [
                "teacher_guidance_total_loss",
                "teacher_gate_bce",
                "teacher_family_shape_loss",
                "teacher_family_prob_loss",
                "teacher_gate_target_rate",
                "teacher_family_mask_rate",
                "teacher_family_top1_match",
                "teacher_family_active_rate",
            ]:
                ep[k] += float(guidance_metrics[k].item())
            nb += 1

        scheduler.step()

        train_metrics = {f"train_{k}": v / max(nb, 1) for k, v in ep.items()}
        train_metrics["train_rollout_scale"] = rollout_scale

        val_teacher = evaluate_teacher_forced(model, rollout_val_loader, objective_config)
        val_rollout = evaluate_selffed_rollout(model, rollout_val_loader, objective_config)
        val_guidance = evaluate_teacher_guidance(model, val_loader, gate_pos_weight=args.gate_pos_weight)
        rollout_diag = evaluate_rollout_subset(
            model,
            rollout_val_loader,
            rollout_val_samples=args.rollout_val_samples,
            rollout_eval_limit=args.rollout_eval_limit,
        )
        family_diag = evaluate_family_statistics(model, rollout_val_loader)

        selection_score = (
            val_teacher["val_teacher_total_loss"]
            + args.rollout_weight * val_rollout["val_rollout_total_loss"]
            + args.teacher_guidance_weight * val_guidance["val_teacher_guidance_total_loss"]
        )

        elapsed = time.time() - t0
        is_best = selection_score < best_score
        if is_best:
            best_score = selection_score
            best_metrics = {
                **val_teacher,
                **val_rollout,
                **val_guidance,
                **rollout_diag,
                **family_diag,
                "selection_score": selection_score,
            }
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "selection_score": best_score,
                    "config": {
                        "type": "transformer_teacher_guided_local_family_student_t_207b",
                        "encoder": encoder_config,
                        "decoder": decoder_config,
                        "support_lo": args.support_lo,
                        "support_hi": args.support_hi,
                        "support_eps": args.support_eps,
                        "cov_jitter": args.cov_jitter,
                        "history_len": hist_len,
                        "future_len": future_len,
                        "objective": objective_config,
                        "teacher_guidance": {
                            "teacher_checkpoint": args.teacher_checkpoint,
                            "teacher_guidance_weight": args.teacher_guidance_weight,
                            "teacher_target_temp": args.teacher_target_temp,
                            "teacher_knn": args.teacher_knn,
                            "gate_pos_weight": args.gate_pos_weight,
                        },
                    },
                    "metrics": best_metrics,
                },
                Path(args.output_dir) / "best_model.pt",
            )

        row = {
            "epoch": epoch,
            **train_metrics,
            **val_teacher,
            **val_rollout,
            **val_guidance,
            **rollout_diag,
            **family_diag,
            "selection_score": selection_score,
            "elapsed_sec": elapsed,
        }
        history.append(row)
        with open(Path(args.output_dir) / "training_history.json", "w") as f:
            json.dump(make_serializable(history), f, indent=2)

        print(
            f"Ep {epoch:>3d}  "
            f"train_tf={train_metrics['train_teacher_total_loss']:.4f}  "
            f"train_tg={train_metrics['train_teacher_guidance_total_loss']:.4f}  "
            f"train_ro={train_metrics['train_rollout_total_loss']:.4f}  "
            f"val_tf={val_teacher['val_teacher_total_loss']:.4f}  "
            f"val_tg={val_guidance['val_teacher_guidance_total_loss']:.4f}  "
            f"val_ro={val_rollout['val_rollout_total_loss']:.4f}  "
            f"roll_cov90={rollout_diag['rollout_cov90']:.4f}  "
            f"roll_mae={rollout_diag['rollout_mae']:.4f}  "
            f"rank={rollout_diag['rollout_rank_ratio_h30']:.2f}  "
            f"gate={family_diag['val_gate_prob_mean']:.3f}  "
            f"fam_top1={family_diag['val_family_top1_mean']:.3f}  "
            f"tg_top1={val_guidance['val_teacher_family_top1_match']:.3f}  "
            f"rs={rollout_scale:.2f}  "
            f"({elapsed:.1f}s)"
            + ("  *best" if is_best else "")
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "epoch": args.epochs,
            "selection_score": best_score,
            "config": {
                "type": "transformer_teacher_guided_local_family_student_t_207b",
                "encoder": encoder_config,
                "decoder": decoder_config,
                "support_lo": args.support_lo,
                "support_hi": args.support_hi,
                "support_eps": args.support_eps,
                "cov_jitter": args.cov_jitter,
                "history_len": hist_len,
                "future_len": future_len,
                "objective": objective_config,
                "teacher_guidance": {
                    "teacher_checkpoint": args.teacher_checkpoint,
                    "teacher_guidance_weight": args.teacher_guidance_weight,
                    "teacher_target_temp": args.teacher_target_temp,
                    "teacher_knn": args.teacher_knn,
                    "gate_pos_weight": args.gate_pos_weight,
                },
            },
            "metrics": best_metrics,
        },
        Path(args.output_dir) / "final_model.pt",
    )


if __name__ == "__main__":
    main()
