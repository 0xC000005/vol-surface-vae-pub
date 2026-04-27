#!/usr/bin/env python
"""584a: fine-tune unified flow with generic reconstructed-state realism losses."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import make_serializable  # noqa: E402
from experiments.backfill.block_ar.train_577a_unified_increment_flow import (  # noqa: E402
    UnifiedIncrementFlow,
    UnifiedIncrementFlowConfig,
    _fit_mean_std,
    _make_train_val_blocks,
    _standardize,
)


def encode_state_torch(state: torch.Tensor, transforms: list[str], eps: float = 1e-8) -> torch.Tensor:
    parts = []
    for idx, transform in enumerate(transforms):
        values = state[..., idx]
        if transform == "log_level":
            parts.append(torch.log(values.clamp_min(eps)))
        elif transform == "diff_level":
            parts.append(values)
        else:
            raise ValueError(f"unknown transform {transform}")
    return torch.stack(parts, dim=-1)


def transformed_increment_to_encoded_state(
    history_state: torch.Tensor,
    transformed_increment: torch.Tensor,
    *,
    inc_mean: torch.Tensor,
    inc_std: torch.Tensor,
    transforms: list[str],
) -> torch.Tensor:
    raw_increment = transformed_increment * inc_std + inc_mean
    last_encoded = encode_state_torch(history_state, transforms)[:, -1, :][:, None, :]
    return last_encoded + torch.cumsum(raw_increment, dim=1)


def range_and_coverage_loss(
    encoded_samples: torch.Tensor,
    encoded_target: torch.Tensor,
    *,
    encoded_min: torch.Tensor,
    encoded_max: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Generic state-space realism loss.

    Shapes:
    - encoded_samples: (B,K,T,D)
    - encoded_target: (B,T,D)
    """
    span = (encoded_max - encoded_min).clamp_min(1e-6)
    below = torch.relu(encoded_min - encoded_samples) / span
    above = torch.relu(encoded_samples - encoded_max) / span
    range_loss = (below.square() + above.square()).mean()
    q05 = torch.quantile(encoded_samples, 0.05, dim=1)
    q95 = torch.quantile(encoded_samples, 0.95, dim=1)
    miss_low = torch.relu(q05 - encoded_target) / span
    miss_high = torch.relu(encoded_target - q95) / span
    coverage_loss = (miss_low.square() + miss_high.square()).mean()
    coverage = ((encoded_target >= q05) & (encoded_target <= q95)).float().mean()
    exceedance = ((encoded_samples < encoded_min) | (encoded_samples > encoded_max)).float().mean()
    return range_loss + coverage_loss, {
        "range_loss": range_loss.detach(),
        "coverage_loss": coverage_loss.detach(),
        "encoded_coverage": coverage.detach(),
        "encoded_exceedance": exceedance.detach(),
    }


def differentiable_sample(
    model: UnifiedIncrementFlow,
    history: torch.Tensor,
    *,
    n_samples: int,
    n_steps: int,
) -> torch.Tensor:
    bsz = history.shape[0]
    repeated_history = history.repeat_interleave(int(n_samples), dim=0)
    x = model.draw_source(
        bsz * int(n_samples),
        device=history.device,
        dtype=history.dtype,
        history=repeated_history,
    )
    dt = 1.0 / float(n_steps)
    for step in range(int(n_steps)):
        t_value = torch.full((x.shape[0],), (step + 0.5) * dt, device=x.device, dtype=x.dtype)
        x = x + dt * model.forward(repeated_history, x, t_value)
    return x.reshape(bsz, int(n_samples), model.cfg.future_len, model.cfg.n_vars)


@torch.no_grad()
def sample_audit(
    model: UnifiedIncrementFlow,
    val_hist: torch.Tensor,
    val_history_state: torch.Tensor,
    val_future_state: torch.Tensor,
    *,
    inc_mean: torch.Tensor,
    inc_std: torch.Tensor,
    transforms: list[str],
    encoded_min: torch.Tensor,
    encoded_max: torch.Tensor,
    n_samples: int,
    sample_steps: int,
    sample_windows: int,
) -> dict[str, Any]:
    n = min(int(sample_windows), int(val_hist.shape[0]))
    samples = model.sample(val_hist[:n], n_samples=n_samples, n_steps=sample_steps)
    flat = samples.reshape(n * int(n_samples), model.cfg.future_len, model.cfg.n_vars)
    repeated_history = val_history_state[:n].repeat_interleave(int(n_samples), dim=0)
    encoded = transformed_increment_to_encoded_state(
        repeated_history,
        flat,
        inc_mean=inc_mean,
        inc_std=inc_std,
        transforms=transforms,
    ).reshape(n, int(n_samples), model.cfg.future_len, model.cfg.n_vars)
    encoded_target = encode_state_torch(val_future_state[:n], transforms)
    q05 = torch.quantile(encoded, 0.05, dim=1)
    q95 = torch.quantile(encoded, 0.95, dim=1)
    coverage = ((encoded_target >= q05) & (encoded_target <= q95)).float().mean()
    exceedance = ((encoded < encoded_min) | (encoded > encoded_max)).float().mean()
    return {
        "sample_windows": int(n),
        "n_samples": int(n_samples),
        "sample_steps": int(sample_steps),
        "encoded_90_coverage": float(coverage.detach().cpu().item()),
        "encoded_train_range_exceedance": float(exceedance.detach().cpu().item()),
        "transformed_increment_std": float(samples.detach().std().cpu().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default="models/backfill/582c_clean_conditional_empirical_source_top4_flow_s586/best_model.pt",
    )
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--iv_count", type=int, default=25)
    parser.add_argument("--clean_nonpositive_log_levels", action="store_true", default=True)
    parser.add_argument("--max_train_windows", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--clip_grad", type=float, default=1.0)
    parser.add_argument("--lambda_realism", type=float, default=0.25)
    parser.add_argument("--realism_samples", type=int, default=4)
    parser.add_argument("--realism_steps", type=int, default=4)
    parser.add_argument("--sample_windows", type=int, default=128)
    parser.add_argument("--audit_samples", type=int, default=16)
    parser.add_argument("--audit_steps", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=584)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    _panel, _columns, train_block, val_block = _make_train_val_blocks(args)
    hist_mean = payload["history_mean"]
    hist_std = payload["history_std"]
    inc_mean_np = payload["increment_mean"]
    inc_std_np = payload["increment_std"]
    if payload.get("increment_transform", "standard") != "standard":
        raise ValueError("584a currently expects the standard increment coordinate")

    train_hist = torch.from_numpy(_standardize(train_block.history_state, hist_mean, hist_std)).to(device)
    val_hist = torch.from_numpy(_standardize(val_block.history_state, hist_mean, hist_std)).to(device)
    train_inc = torch.from_numpy(_standardize(train_block.future_increment, inc_mean_np, inc_std_np)).to(device)
    val_inc = torch.from_numpy(_standardize(val_block.future_increment, inc_mean_np, inc_std_np)).to(device)
    train_history_state = torch.from_numpy(train_block.history_state).to(device)
    train_future_state = torch.from_numpy(train_block.future_state).to(device)
    val_history_state = torch.from_numpy(val_block.history_state).to(device)
    val_future_state = torch.from_numpy(val_block.future_state).to(device)
    transforms = [spec.transform for spec in train_block.specs]
    train_encoded = torch.cat(
        [encode_state_torch(train_history_state, transforms), encode_state_torch(train_future_state, transforms)],
        dim=1,
    )
    encoded_min = train_encoded.amin(dim=(0, 1))
    encoded_max = train_encoded.amax(dim=(0, 1))
    inc_mean = torch.from_numpy(inc_mean_np).to(device)
    inc_std = torch.from_numpy(inc_std_np).to(device)

    cfg = UnifiedIncrementFlowConfig(**payload["config"])
    model = UnifiedIncrementFlow(cfg).to(device)
    if cfg.source_mode in {"empirical_path", "empirical_conditional"}:
        bank = train_inc.reshape(train_inc.shape[0], -1)
        keys = train_hist[:, -1, :] if cfg.source_mode == "empirical_conditional" else None
        model.set_source_bank(bank, keys)
    model.load_state_dict(payload["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = DataLoader(
        TensorDataset(train_hist, train_inc, train_history_state, train_future_state),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    val_loader = DataLoader(TensorDataset(val_hist, val_inc), batch_size=args.batch_size, shuffle=False)

    def val_loss() -> float:
        model.eval()
        losses = []
        for hist, inc in val_loader:
            loss, _metrics = model.training_loss(hist, inc)
            losses.append(float(loss.detach().cpu().item()))
        return float(np.mean(losses))

    best_metric = float("inf")
    best_epoch = -1
    records: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        t0 = time.time()
        model.train()
        sums: dict[str, float] = {}
        n_batches = 0
        for hist, inc, hist_state, fut_state in loader:
            fm_loss, fm_metrics = model.training_loss(hist, inc)
            samples = differentiable_sample(
                model,
                hist,
                n_samples=int(args.realism_samples),
                n_steps=int(args.realism_steps),
            )
            flat_samples = samples.reshape(-1, model.cfg.future_len, model.cfg.n_vars)
            repeated_state = hist_state.repeat_interleave(int(args.realism_samples), dim=0)
            encoded_samples = transformed_increment_to_encoded_state(
                repeated_state,
                flat_samples,
                inc_mean=inc_mean,
                inc_std=inc_std,
                transforms=transforms,
            ).reshape(hist.shape[0], int(args.realism_samples), model.cfg.future_len, model.cfg.n_vars)
            encoded_target = encode_state_torch(fut_state, transforms)
            realism_loss, realism_metrics = range_and_coverage_loss(
                encoded_samples,
                encoded_target,
                encoded_min=encoded_min,
                encoded_max=encoded_max,
            )
            loss = fm_loss + float(args.lambda_realism) * realism_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            optimizer.step()
            metrics = {
                "loss": loss.detach(),
                "fm_loss": fm_loss.detach(),
                **fm_metrics,
                **realism_metrics,
            }
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach().cpu().item())
            n_batches += 1
        avg = {key: value / max(n_batches, 1) for key, value in sums.items()}
        audit = sample_audit(
            model,
            val_hist,
            val_history_state,
            val_future_state,
            inc_mean=inc_mean,
            inc_std=inc_std,
            transforms=transforms,
            encoded_min=encoded_min,
            encoded_max=encoded_max,
            n_samples=args.audit_samples,
            sample_steps=args.audit_steps,
            sample_windows=args.sample_windows,
        )
        vloss = val_loss()
        rec = {
            "epoch": epoch,
            "val_fm_loss": vloss,
            "sec": time.time() - t0,
            **avg,
            **{f"audit_{key}": value for key, value in audit.items()},
        }
        records.append(rec)
        print(json.dumps(make_serializable(rec), sort_keys=True), flush=True)
        selection_metric = vloss + 0.25 * float(audit["encoded_train_range_exceedance"])
        if selection_metric < best_metric:
            best_metric = selection_metric
            best_epoch = epoch
            torch.save(
                {
                    **payload,
                    "epoch": int(epoch),
                    "best_val": float(best_metric),
                    "model_state_dict": model.state_dict(),
                    "finetune_config": vars(args),
                },
                out_dir / "best_model.pt",
            )

    torch.save(
        {
            **payload,
            "epoch": int(args.epochs),
            "best_val": float(best_metric),
            "model_state_dict": model.state_dict(),
            "finetune_config": vars(args),
        },
        out_dir / "final_model.pt",
    )
    summary = {
        "best_epoch": int(best_epoch),
        "best_selection_metric": float(best_metric),
        "source_checkpoint": args.checkpoint,
        "records": records,
        "config": vars(args),
        "base_config": asdict(cfg),
    }
    (out_dir / "training_history.json").write_text(json.dumps(make_serializable(records), indent=2), encoding="utf-8")
    (out_dir / "train_summary.json").write_text(json.dumps(make_serializable(summary), indent=2), encoding="utf-8")
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
