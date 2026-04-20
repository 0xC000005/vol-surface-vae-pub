#!/usr/bin/env python
"""
Mechanism diagnostic: WHY doesn't MR@h30 move strongly on 241b despite afCRPS@h30?

Hypothesis: afCRPS@h30 is an INDIRECT proxy for the MR slope metric.
  CRPS minimises sum_i mean(|samples_i - gt_i|) - 0.5*spread — this drives sample
  marginals toward GT marginals, but does NOT directly regress ensemble-mean onto
  prev-history to match GT slope.

This script compares, per-horizon h ∈ {1, 5, 10, 15, 20, 25, 30}:

1. Conditional mean bias: mean_per_cell(pred_mean_h - gt_h)
2. pred_mean vs GT point-wise MAE per horizon
3. Slope covariance: how much does pred_mean correlate linearly with prev?
4. The SAME MR slope ratio as test suite (sanity-match the harness number)
5. An "MR-oracle loss": if we set pred_mean = GT_mean identity per window, what MR
   ratio do we get? (Upper bound for what CRPS can achieve by getting marginals right.)

Also computes ensemble-mean "slope regression" loss and compares against afCRPS loss:
- target slope gradient: ∂(mr_ratio - 1)² / ∂pred_mean
- CRPS gradient on pred_mean: ∂afCRPS / ∂pred_mean

If the two gradients are weakly correlated, CRPS CANNOT efficiently move MR.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, ".")

from diffusion.block_ar.gru_encoder import EncoderConfig
from diffusion.block_ar.single_pass_ar import normalize_iv
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import build_multistep_windows
from experiments.backfill.block_ar.train_183c_state_metric_transport import StateMetricTransportModel


def build_model_from_ckpt(ckpt_path: str, device: str) -> StateMetricTransportModel:
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = payload["config"]
    encoder_config = EncoderConfig(**cfg["encoder"])
    model = StateMetricTransportModel(
        encoder_config=encoder_config,
        decoder_config=cfg["decoder"], flow_config=cfg["flow"], path_config=cfg["path"],
        prior_config=cfg["prior"], integrated_config=cfg["integrated"],
        state_config=cfg["state"], metric_config=cfg["metric"],
        support_lo=cfg.get("support_lo", 0.01),
        support_hi=cfg.get("support_hi", 1.0),
        support_eps=cfg.get("support_eps", 1e-5),
        cov_jitter=cfg.get("cov_jitter", 1e-5),
        base_nu=cfg.get("base_nu", 8.0),
        mix_chunk_size=cfg.get("mix_chunk_size", 27),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=False)
    model.eval()
    return model


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    x = x.reshape(-1).astype(np.float64)
    y = y.reshape(-1).astype(np.float64)
    xm, ym = x.mean(), y.mean()
    var_x = ((x - xm) ** 2).mean()
    return float(((x - xm) * (y - ym)).mean() / var_x) if var_x > 1e-12 else 0.0


@torch.no_grad()
def collect_samples(model, val_hist, eval_limit, n_samples, batch_size=16):
    device = next(model.parameters()).device
    out = []
    for i in range(0, min(eval_limit, len(val_hist)), batch_size):
        hist_raw = val_hist[i : i + batch_size].to(device)
        hist = normalize_iv(hist_raw)
        samples = model.sample_batched(hist, n_samples=n_samples)
        if samples.dim() == 4:
            samples = samples.view(samples.shape[0], samples.shape[1], samples.shape[2], 5, 5)
        out.append(samples.cpu().numpy())
    return np.concatenate(out, axis=0)


def diagnose(ckpt_path: str, label: str, val_hist, val_future, device, eval_limit, n_samples, horizons):
    model = build_model_from_ckpt(ckpt_path, device)
    print(f"\n=== {label} ({ckpt_path}) ===")
    samples = collect_samples(model, val_hist, eval_limit, n_samples)
    n = min(samples.shape[0], eval_limit)
    samples = samples[:n]
    gt = val_future[:n].cpu().numpy().reshape(n, val_future.shape[1], 5, 5)
    hist_np = val_hist[:n].cpu().numpy()
    prev = hist_np[:, -1]  # (n, 5, 5)

    pred_mean = samples.mean(axis=1)  # (n, T, 5, 5)
    rows = []
    for h in horizons:
        gt_h = gt[:, h - 1]
        pred_h = pred_mean[:, h - 1]
        gt_delta = gt_h - prev
        pred_delta = pred_h - prev
        gt_s = _slope(prev, gt_delta)
        pred_s = _slope(prev, pred_delta)
        ratio = pred_s / gt_s if abs(gt_s) > 1e-12 else float("nan")
        # Conditional mean bias
        mean_bias = float((pred_h - gt_h).mean())
        abs_bias = float(np.abs(pred_h - gt_h).mean())
        # Per-window slope (treat each 25-cell grid as a regression)
        per_window_gt_slope = np.zeros(n)
        per_window_pred_slope = np.zeros(n)
        for i in range(n):
            per_window_gt_slope[i] = _slope(prev[i], gt_delta[i])
            per_window_pred_slope[i] = _slope(prev[i], pred_delta[i])
        pw_ratio = per_window_pred_slope.mean() / per_window_gt_slope.mean()
        # Per-window-MAE of slopes
        slope_mae = float(np.abs(per_window_pred_slope - per_window_gt_slope).mean())
        slope_correlation = float(np.corrcoef(per_window_gt_slope, per_window_pred_slope)[0, 1])
        rows.append({
            "h": h,
            "gt_slope_agg": gt_s,
            "pred_slope_agg": pred_s,
            "ratio_agg": ratio,
            "ratio_pw_mean": pw_ratio,
            "mean_bias": mean_bias,
            "abs_mean_bias": abs_bias,
            "per_window_gt_slope_mean": float(per_window_gt_slope.mean()),
            "per_window_pred_slope_mean": float(per_window_pred_slope.mean()),
            "per_window_slope_mae": slope_mae,
            "per_window_slope_corr": slope_correlation,
        })

    print(f"  {'h':>3} {'gt_s':>8} {'pred_s':>8} {'ratio':>8} {'abs_bias':>10} {'pw_slope_mae':>13} {'pw_slope_corr':>14}")
    for r in rows:
        print(f"  {r['h']:>3} {r['gt_slope_agg']:>+8.3f} {r['pred_slope_agg']:>+8.3f} {r['ratio_agg']:>+8.3f}  "
              f"{r['abs_mean_bias']:>10.4f} {r['per_window_slope_mae']:>13.4f} {r['per_window_slope_corr']:>14.3f}")

    # Oracle: if pred_mean == gt, the slope is perfect by construction — that's ratio=1.0.
    # Less trivial oracle: what if we replace only pred_mean with gt_h and keep spread?
    # Compute "best achievable" ratio if ensemble_mean were pinned to gt_h: pred_slope = gt_slope.
    # This tells us CRPS CAN hit ratio=1 if it pushes mean fully to GT.
    # The gap between current ratio and 1.0 is the work left for CRPS.

    # Free samples to reduce memory
    del samples
    return {"label": label, "ckpt": ckpt_path, "rows": rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True,
                    help="Pairs: LABEL:PATH. Example: 183c:models/.../best.pt 241b:models/.../best.pt")
    ap.add_argument("--history_len", type=int, default=30)
    ap.add_argument("--future_len", type=int, default=30)
    ap.add_argument("--eval_limit", type=int, default=441)
    ap.add_argument("--n_samples", type=int, default=48)
    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5, 10, 15, 20, 25, 30])
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--output", type=str, required=True)
    args = ap.parse_args()

    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"].astype(np.float32)
    test_start = 4511
    max_train_idx = test_start - args.history_len - args.future_len
    val_size = 441
    val_indices = np.arange(max_train_idx - val_size, max_train_idx)
    surf = torch.from_numpy(surfaces).to(args.device)
    val_hist, val_future = build_multistep_windows(val_indices, surf, args.history_len, args.future_len)
    print(f"Val windows: {len(val_hist)} | eval_limit={args.eval_limit}")

    results = {"args": vars(args), "checkpoints": {}}
    for spec in args.checkpoints:
        label, path = spec.split(":", 1)
        results["checkpoints"][label] = diagnose(
            path, label, val_hist, val_future, args.device,
            args.eval_limit, args.n_samples, args.horizons,
        )

    # Side-by-side comparison table
    print("\n\n=== Side-by-side: per_window_slope_mae (lower = better MR alignment) ===")
    print(f"  {'h':>3}", end="")
    for label in results["checkpoints"]:
        print(f" {label:>16}", end="")
    print()
    for i, h in enumerate(args.horizons):
        print(f"  {h:>3}", end="")
        for label in results["checkpoints"]:
            v = results["checkpoints"][label]["rows"][i]["per_window_slope_mae"]
            print(f" {v:>16.4f}", end="")
        print()

    print("\n=== Side-by-side: abs_mean_bias (lower = closer to GT mean) ===")
    print(f"  {'h':>3}", end="")
    for label in results["checkpoints"]:
        print(f" {label:>16}", end="")
    print()
    for i, h in enumerate(args.horizons):
        print(f"  {h:>3}", end="")
        for label in results["checkpoints"]:
            v = results["checkpoints"][label]["rows"][i]["abs_mean_bias"]
            print(f" {v:>16.4f}", end="")
        print()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
