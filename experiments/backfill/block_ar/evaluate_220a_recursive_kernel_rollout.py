#!/usr/bin/env python
"""
220a: Recursive multi-day rollout smoke test for frozen one-day kernels.

Purpose:
  - turn a 1-day conditional model into a 30-day path generator by recursion
  - measure whether width grows sensibly, whether floor/ceiling mass stays local,
    and whether calm/turbulent widening persists over horizon
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

import sys

sys.path.insert(0, ".")

from experiments.backfill.block_ar._rollout_220_utils import (
    OneDayKernelRolloutWrapper,
    build_rollout_windows,
    evaluate_rollout_subset,
    load_one_day_kernel,
    make_serializable,
    write_markdown_summary,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="220a recursive rollout smoke test")
    parser.add_argument("--model_type", type=str, default="212ai")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--max_windows", type=int, default=192)
    parser.add_argument("--samples", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--chunk_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model, payload = load_one_day_kernel(args.model_type, args.checkpoint, device)
    wrapper = OneDayKernelRolloutWrapper(model).eval()
    batch = build_rollout_windows(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        max_windows=args.max_windows,
        device=device,
        split="val",
    )
    summary = evaluate_rollout_subset(
        wrapper=wrapper,
        history_norm=batch.history_norm,
        future_01=batch.future_01,
        batch_size=args.batch_size,
        n_samples=args.samples,
        n_steps=args.future_len,
        chunk_size=args.chunk_size,
    )
    summary["config"] = {
        "model_type": args.model_type,
        "checkpoint": args.checkpoint,
        "samples": args.samples,
        "n_windows": int(batch.history_norm.shape[0]),
        "history_len": args.history_len,
        "future_len": args.future_len,
        "checkpoint_epoch": int(payload.get("epoch", -1)),
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(make_serializable(summary), indent=2))

    lines = [
        f"- model: `{args.model_type}`",
        f"- checkpoint: `{args.checkpoint}`",
        f"- windows: `{batch.history_norm.shape[0]}`",
        f"- samples per window: `{args.samples}`",
        "",
        "**Coverage 90%**",
        f"- h1: `{summary.get('cov90_h1', float('nan')):.3f}`",
        f"- h5: `{summary.get('cov90_h5', float('nan')):.3f}`",
        f"- h10: `{summary.get('cov90_h10', float('nan')):.3f}`",
        f"- h20: `{summary.get('cov90_h20', float('nan')):.3f}`",
        f"- h30: `{summary.get('cov90_h30', float('nan')):.3f}`",
        "",
        "**Width 90%**",
        f"- h1: `{summary.get('width90_h1', float('nan')):.4f}`",
        f"- h5: `{summary.get('width90_h5', float('nan')):.4f}`",
        f"- h10: `{summary.get('width90_h10', float('nan')):.4f}`",
        f"- h20: `{summary.get('width90_h20', float('nan')):.4f}`",
        f"- h30: `{summary.get('width90_h30', float('nan')):.4f}`",
        "",
        "**Regime Widening**",
        f"- turb/calm h1: `{summary.get('turb_calm_ratio_h1', float('nan')):.3f}`",
        f"- turb/calm h10: `{summary.get('turb_calm_ratio_h10', float('nan')):.3f}`",
        f"- turb/calm h30: `{summary.get('turb_calm_ratio_h30', float('nan')):.3f}`",
        "",
        "**Boundary Occupancy**",
        f"- at_floor h1: `{summary.get('at_floor_h1', float('nan')):.3%}`",
        f"- at_floor h30: `{summary.get('at_floor_h30', float('nan')):.3%}`",
        f"- at_ceiling h1: `{summary.get('at_ceiling_h1', float('nan')):.3%}`",
        f"- at_ceiling h30: `{summary.get('at_ceiling_h30', float('nan')):.3%}`",
    ]
    write_markdown_summary(args.output_md, "220a Recursive Rollout Smoke", lines)
    print(json.dumps(make_serializable(summary), indent=2))


if __name__ == "__main__":
    main()
