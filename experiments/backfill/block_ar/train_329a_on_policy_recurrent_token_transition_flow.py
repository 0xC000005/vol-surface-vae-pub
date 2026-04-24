#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.recurrent_logit_transition_token_flow_matching import (
    RecurrentLogitTransitionTokenFlowMatching,
    load_model,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
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
    train_indices = np.arange(0, max_train_idx - val_size)
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    if max_train_windows > 0:
        train_indices = train_indices[-max_train_windows:]
    if max_val_windows > 0:
        val_indices = val_indices[:max_val_windows]
    train_hist, train_future = build_multistep_windows(
        train_indices, surf_tensor, history_len, future_len
    )
    val_hist, val_future = build_multistep_windows(
        val_indices, surf_tensor, history_len, future_len
    )
    return train_hist, train_future, val_hist, val_future


@torch.no_grad()
def collect_rollin_states(
    model: RecurrentLogitTransitionTokenFlowMatching,
    history_norm: torch.Tensor,
    n_steps: int,
    flow_steps: int,
    temperature: float,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Sample one free-running path and return the states seen before each step."""
    history_norm = model._flatten(history_norm)
    was_training = model.training
    model.eval()
    state_stack, current_logit = model.encode_history(history_norm)
    states: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    dt = 1.0 / float(flow_steps)
    for _step in range(n_steps):
        states.append(state_stack.detach().clone())
        logits.append(current_logit.detach().clone())
        x = temperature * torch.randn_like(current_logit)
        for flow_step in range(flow_steps):
            t = torch.full(
                (history_norm.shape[0],),
                (flow_step + 0.5) * dt,
                device=history_norm.device,
                dtype=history_norm.dtype,
            )
            x = x + dt * model.predict_velocity(x, current_logit, state_stack, t)
        next_logit = current_logit + x
        state_stack = model.recurrent_step(state_stack, current_logit, next_logit)
        current_logit = next_logit
    model.train(was_training)
    return states, logits


def transition_fm_loss_from_states(
    model: RecurrentLogitTransitionTokenFlowMatching,
    future_logits: torch.Tensor,
    true_prev_logits: torch.Tensor,
    states: list[torch.Tensor],
    current_logits: list[torch.Tensor],
    target_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    bsz = future_logits.shape[0]
    losses: list[torch.Tensor] = []
    transition_abs = torch.tensor(0.0, device=future_logits.device, dtype=future_logits.dtype)
    transition_std = torch.tensor(0.0, device=future_logits.device, dtype=future_logits.dtype)
    rollin_error = torch.tensor(0.0, device=future_logits.device, dtype=future_logits.dtype)
    for step, (state_stack, current_logit) in enumerate(zip(states, current_logits)):
        next_logit = future_logits[:, step]
        if target_mode == "level":
            x1 = next_logit - current_logit
        elif target_mode == "innovation":
            x1 = next_logit - true_prev_logits[:, step]
        else:
            raise ValueError(f"Unknown roll-in target mode: {target_mode}")
        x0 = torch.randn_like(x1)
        t = torch.rand(bsz, device=future_logits.device, dtype=future_logits.dtype)
        x_t = (1.0 - t)[:, None] * x0 + t[:, None] * x1
        target_velocity = x1 - x0
        pred_velocity = model.predict_velocity(x_t, current_logit, state_stack, t)
        losses.append(F.mse_loss(pred_velocity, target_velocity))
        transition_abs = transition_abs + x1.abs().mean()
        transition_std = transition_std + x1.std(unbiased=False)
        rollin_error = rollin_error + (current_logit - next_logit).abs().mean()
    horizon = float(len(losses))
    total = torch.stack(losses).mean()
    metrics = {
        "on_policy_fm": total.detach(),
        "on_policy_transition_abs": (transition_abs / horizon).detach(),
        "on_policy_transition_std": (transition_std / horizon).detach(),
        "on_policy_to_target_abs": (rollin_error / horizon).detach(),
    }
    return total, metrics


def on_policy_training_loss(
    model: RecurrentLogitTransitionTokenFlowMatching,
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    rollin_flow_steps: int,
    rollin_temperature: float,
    rollin_target_mode: str,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    teacher_loss, teacher_metrics = model.training_loss(history_norm, future_norm)
    future_logits = model.target_future_logits(future_norm)
    with torch.no_grad():
        _state_stack, history_last_logit = model.encode_history(history_norm)
        true_prev_logits = torch.cat(
            [history_last_logit[:, None, :], future_logits[:, :-1]],
            dim=1,
        ).detach()
    states, current_logits = collect_rollin_states(
        model,
        history_norm,
        n_steps=future_logits.shape[1],
        flow_steps=rollin_flow_steps,
        temperature=rollin_temperature,
    )
    on_policy_loss, on_policy_metrics = transition_fm_loss_from_states(
        model,
        future_logits,
        true_prev_logits,
        states,
        current_logits,
        target_mode=rollin_target_mode,
    )
    total = 0.5 * (teacher_loss + on_policy_loss)
    metrics = {
        "total": total.detach(),
        "teacher_fm": teacher_loss.detach(),
        "on_policy_fm": on_policy_loss.detach(),
        "teacher_transition_std": teacher_metrics["transition_std"].detach(),
        "teacher_state_logit_abs": teacher_metrics["state_logit_abs"].detach(),
        **on_policy_metrics,
    }
    return total, metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="329a on-policy recurrent token transition flow fine-tune"
    )
    parser.add_argument("--init_checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_train_windows", type=int, default=2048)
    parser.add_argument("--max_val_windows", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--rollin_flow_steps", type=int, default=8)
    parser.add_argument("--rollin_temperature", type=float, default=1.0)
    parser.add_argument(
        "--rollin_target_mode",
        choices=["level", "innovation"],
        default="level",
        help=(
            "level reproduces 329a's corrective target true_next - generated_current; "
            "innovation uses the observed one-day transition true_next - true_prev."
        ),
    )
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
    records: list[dict[str, float]] = []

    print("329a on-policy recurrent token transition fine-tune")
    print(f"  init={args.init_checkpoint} epoch={payload.get('epoch', -1)}")
    print(f"  train_windows={train_hist.shape[0]} val_windows={val_hist.shape[0]}")
    print(
        f"  rollin_flow_steps={args.rollin_flow_steps} "
        f"rollin_temperature={args.rollin_temperature} "
        f"rollin_target_mode={args.rollin_target_mode}"
    )

    def run_epoch(loader: DataLoader, train_mode: bool) -> dict[str, float]:
        model.train(train_mode)
        sums: dict[str, float] = {}
        count = 0
        for hist_01, fut_01 in loader:
            hist_norm = normalize_iv(hist_01.to(device, non_blocking=True)).view(
                hist_01.shape[0], hist_01.shape[1], -1
            )
            fut_norm = normalize_iv(fut_01.to(device, non_blocking=True)).view(
                fut_01.shape[0], fut_01.shape[1], -1
            )
            with torch.set_grad_enabled(train_mode):
                loss, metrics = on_policy_training_loss(
                    model,
                    hist_norm,
                    fut_norm,
                    rollin_flow_steps=args.rollin_flow_steps,
                    rollin_temperature=args.rollin_temperature,
                    rollin_target_mode=args.rollin_target_mode,
                )
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    if args.clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    optimizer.step()
            bs = hist_01.shape[0]
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.item()) * bs
            count += bs
        return {key: value / max(count, 1) for key, value in sums.items()}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_avg = run_epoch(train_loader, train_mode=True)
        val_avg = run_epoch(val_loader, train_mode=False)
        rec = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_avg.items()},
            **{f"val_{k}": v for k, v in val_avg.items()},
            "sec": time.time() - t0,
        }
        records.append(rec)
        print(
            f"[ep {epoch:02d}] "
            f"train={rec['train_total']:.5f} tfm={rec['train_teacher_fm']:.5f} "
            f"ofm={rec['train_on_policy_fm']:.5f} "
            f"val={rec['val_total']:.5f} vtfm={rec['val_teacher_fm']:.5f} "
            f"vofm={rec['val_on_policy_fm']:.5f} "
            f"roll_abs={rec['val_on_policy_to_target_abs']:.3f} "
            f"time={rec['sec']:.1f}s"
        )
        if rec["val_total"] < best_val:
            best_val = rec["val_total"]
            best_epoch = epoch
            save_checkpoint(str(out_dir / "best_model.pt"), model, cfg, epoch, best_val)

    save_checkpoint(str(out_dir / "final_model.pt"), model, cfg, args.epochs, best_val)
    (out_dir / "training_history.json").write_text(json.dumps(records, indent=2))
    summary = {
        "best_epoch": best_epoch,
        "best_val_total": best_val,
        "init_checkpoint": args.init_checkpoint,
        "init_epoch": int(payload.get("epoch", -1)),
        "rollin_flow_steps": args.rollin_flow_steps,
        "rollin_temperature": args.rollin_temperature,
        "rollin_target_mode": args.rollin_target_mode,
        "config": asdict(cfg),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
