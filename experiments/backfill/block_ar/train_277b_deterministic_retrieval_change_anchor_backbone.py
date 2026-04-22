#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.deterministic_retrieval_change_anchor_backbone import (
    DeterministicRetrievalChangeAnchorConfig,
    save_checkpoint,
)
from experiments.backfill.block_ar.train_169a_transformed_student_t import normalize_iv
from experiments.backfill.block_ar.train_169c_shape_scale_student_t import (
    build_multistep_windows,
)


def make_library(
    data_path: str,
    history_len: int,
    future_len: int,
    test_start: int,
    val_size: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int, int]:
    data = np.load(data_path)
    surfaces = data["surface"].astype(np.float32)
    _, h, w = surfaces.shape
    max_train_idx = test_start - history_len - future_len
    train_indices = np.arange(0, max_train_idx - val_size)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    train_hist, train_future = build_multistep_windows(
        train_indices, surf_tensor, history_len, future_len
    )
    hist_norm = normalize_iv(train_hist).view(train_hist.shape[0], train_hist.shape[1], -1)
    last_level = train_hist[:, -1]
    return hist_norm.cpu(), last_level.cpu(), train_future.cpu(), h, w, h * w


def main() -> None:
    parser = argparse.ArgumentParser(description="277b-v0 deterministic anchored retrieval Stage A backbone")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    library_history_norm, library_last_level_01, library_future_01, h, w, d = make_library(
        data_path=args.data_path,
        history_len=args.history_len,
        future_len=args.future_len,
        test_start=args.test_start,
        val_size=args.val_size,
        device=args.device,
    )
    cfg = DeterministicRetrievalChangeAnchorConfig(
        history_len=args.history_len,
        future_len=args.future_len,
        n_cells=d,
        distance="l2",
    )
    save_checkpoint(
        str(out_dir / "best_model.pt"),
        cfg,
        library_history_norm,
        library_last_level_01,
        library_future_01,
    )
    save_checkpoint(
        str(out_dir / "final_model.pt"),
        cfg,
        library_history_norm,
        library_last_level_01,
        library_future_01,
    )
    summary = {
        "library_windows": int(library_history_norm.shape[0]),
        "grid_h": h,
        "grid_w": w,
        "n_cells": d,
        "config": asdict(cfg),
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
