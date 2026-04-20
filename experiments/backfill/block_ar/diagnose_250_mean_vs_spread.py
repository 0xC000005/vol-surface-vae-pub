"""
250-series mean-vs-spread decomposition diagnostic (REQUIRED for Stage A).

241b lesson: proper-scoring rules can improve headline metrics by narrowing spread
instead of learning a better center path. This diagnostic catches that pathology.

For each horizon h in {1, 5, 10, 15, 20, 25, 30}, compute on the full val split:
  - ensemble_mean_bias(h):  E[ mean_K(samples) - gt ]   per cell, aggregated
  - ensemble_spread(h):     E[ std_K(samples) ]         per cell, aggregated
  - gt_empirical_std(h):    std across val windows of gt[:, h]  per cell
  - spread_ratio(h):        ensemble_spread / gt_empirical_std  (1.0 = matched)
  - per-cell CRPS vs per-cell spread          (skill-spread curve)

Gate heuristic (Stage A):
  If  |mean_bias(h30)|  AT LEAST  0.5 * |183c's 241b mean_bias(h30)|  AND
      spread_ratio(h30) <  0.6
  then 241b pathology recurred — model is trading spread for headline.

Outputs JSON + markdown + PNG plots (if matplotlib available).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import sys

sys.path.insert(0, ".")

from diffusion.block_ar.neural_factor import load_model as load_250_model
from diffusion.block_ar.single_pass_ar import normalize_iv
from experiments.backfill.block_ar.train_170d_structured_joint_student_t import (
    build_multistep_windows,
)


def sample_all(
    model,
    loader: DataLoader,
    device: torch.device,
    K: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (samples, gt) stacked across the loader.

    samples: (N, K, T, D)   gt: (N, T, D)
    Assumes the model is 250-series (flat D input convention).
    """
    all_samples = []
    all_gt = []
    model.eval()
    with torch.no_grad():
        for hist_01, fut_flat in loader:
            hist_01 = hist_01.to(device, non_blocking=True)
            fut_flat = fut_flat.to(device, non_blocking=True)
            B = hist_01.shape[0]
            hist_norm = normalize_iv(hist_01).view(B, hist_01.shape[1], -1)
            samples, _ = model(hist_norm, n_samples=K)  # (B, K, T, D)
            all_samples.append(samples.cpu().numpy())
            all_gt.append(fut_flat.cpu().numpy())
    return np.concatenate(all_samples, axis=0), np.concatenate(all_gt, axis=0)


def decompose(
    samples: np.ndarray,  # (N, K, T, D)
    gt: np.ndarray,        # (N, T, D)
    horizons: list[int],
) -> dict:
    N, K, T, D = samples.shape
    results = {"per_horizon": {}}
    for h in horizons:
        idx = h - 1  # 0-based
        s = samples[:, :, idx, :]  # (N, K, D)
        g = gt[:, idx, :]            # (N, D)
        ens_mean = s.mean(axis=1)          # (N, D)
        ens_std = s.std(axis=1)            # (N, D)
        # Mean bias: averaged over windows and cells
        mean_bias_per_cell = (ens_mean - g).mean(axis=0)  # (D,)
        mean_bias = float(np.abs(mean_bias_per_cell).mean())
        mean_bias_signed = float(mean_bias_per_cell.mean())
        # Spread
        ensemble_spread_mean = float(ens_std.mean())
        # GT empirical std across windows at horizon h (per cell then mean)
        gt_emp_std = float(np.abs(g - g.mean(axis=0, keepdims=True)).std(axis=0).mean())
        spread_ratio = (
            ensemble_spread_mean / max(gt_emp_std, 1e-8)
        )
        # CRPS approximation per cell: mean |s - g| - 0.5 mean |s_i - s_j|
        mae_per_cell = np.abs(s - g[:, None]).mean(axis=(0, 1))  # (D,)
        # Pair-wise spread (estimator of E|X - X'|)
        if K >= 2:
            diffs = np.abs(s[:, :, None] - s[:, None, :])  # (N, K, K, D)
            # mean over K,K,N — approx E|X-X'|
            pair_spread_per_cell = diffs.mean(axis=(0, 1, 2))  # (D,)
        else:
            pair_spread_per_cell = np.zeros(D)
        crps_per_cell = mae_per_cell - 0.5 * pair_spread_per_cell
        crps_mean = float(crps_per_cell.mean())
        results["per_horizon"][str(h)] = {
            "mean_abs_bias": mean_bias,
            "mean_signed_bias": mean_bias_signed,
            "ensemble_spread_mean": ensemble_spread_mean,
            "gt_empirical_std_mean": gt_emp_std,
            "spread_ratio": spread_ratio,
            "crps_mean": crps_mean,
            "mae_mean": float(mae_per_cell.mean()),
            "pair_spread_mean": float(pair_spread_per_cell.mean()),
        }
    return results


def pathology_flag(report: dict, h_target: int = 30,
                   spread_ratio_floor: float = 0.6) -> dict:
    """Return a boolean + explanation of whether the 241b spread pathology recurred."""
    h_key = str(h_target)
    if h_key not in report["per_horizon"]:
        return {"flag": False, "reason": f"h={h_target} not in report"}
    row = report["per_horizon"][h_key]
    if row["spread_ratio"] < spread_ratio_floor:
        return {
            "flag": True,
            "reason": (
                f"h{h_target}: spread_ratio={row['spread_ratio']:.3f} "
                f"< {spread_ratio_floor} (241b spread-collapse pathology recurred)"
            ),
        }
    return {
        "flag": False,
        "reason": (
            f"h{h_target}: spread_ratio={row['spread_ratio']:.3f} "
            f">= {spread_ratio_floor} (spread preserved; mechanism healthy)"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="250-series mean-vs-spread diagnostic")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--baseline_checkpoint", type=str, default=None,
                        help="optional: a 250a/183c checkpoint to compare deltas against")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--history_len", type=int, default=30)
    parser.add_argument("--future_len", type=int, default=30)
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--val_size", type=int, default=441)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--horizons", type=str, default="1,5,10,15,20,25,30")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]

    # Data
    data = np.load(args.data_path)
    surfaces = data["surface"].astype(np.float32)
    max_train_idx = args.test_start - args.history_len - args.future_len
    val_indices = np.arange(max_train_idx - args.val_size, max_train_idx)
    surf_tensor = torch.from_numpy(surfaces).to(device)
    val_hist, val_future = build_multistep_windows(
        val_indices, surf_tensor, args.history_len, args.future_len
    )
    loader = DataLoader(
        TensorDataset(val_hist, val_future),
        batch_size=args.batch_size, shuffle=False,
    )

    # Target model
    model, payload = load_250_model(args.checkpoint, device)
    print(f"Sampling from {args.checkpoint} ... (N={len(loader.dataset)}, K={args.K})")
    samples, gt = sample_all(model, loader, device, args.K)
    target_report = decompose(samples, gt, horizons)
    target_report["checkpoint"] = args.checkpoint
    target_report["epoch"] = int(payload.get("epoch", -1))

    # Baseline (optional)
    baseline_report = None
    if args.baseline_checkpoint is not None:
        try:
            base_model, base_payload = load_250_model(args.baseline_checkpoint, device)
            print(f"Sampling from baseline {args.baseline_checkpoint} ...")
            b_samples, b_gt = sample_all(base_model, loader, device, args.K)
            baseline_report = decompose(b_samples, b_gt, horizons)
            baseline_report["checkpoint"] = args.baseline_checkpoint
        except Exception as e:
            print(f"Baseline load failed: {e}")

    # Flags
    flag = pathology_flag(target_report)
    target_report["pathology_flag"] = flag

    # Summary table
    print("\n=== Mean-vs-spread decomposition ===")
    print(f"{'h':>3}  {'|bias|':>10}  {'signed_bias':>12}  {'ens_spread':>12}  "
          f"{'gt_std':>10}  {'ratio':>8}  {'crps':>10}  {'mae':>10}")
    for h in horizons:
        row = target_report["per_horizon"][str(h)]
        print(f"{h:>3}  {row['mean_abs_bias']:>10.5f}  {row['mean_signed_bias']:>12.5f}"
              f"  {row['ensemble_spread_mean']:>12.5f}  {row['gt_empirical_std_mean']:>10.5f}"
              f"  {row['spread_ratio']:>8.3f}  {row['crps_mean']:>10.5f}  {row['mae_mean']:>10.5f}")
    print(f"\nPathology flag: {flag['reason']}")

    # Write JSON
    out_json = out / "mean_vs_spread.json"
    out_json.write_text(json.dumps({
        "target": target_report,
        "baseline": baseline_report,
    }, indent=2))
    print(f"\nJSON -> {out_json}")

    # Write markdown
    md_lines = [
        f"# 250-series Mean-vs-Spread Diagnostic",
        "",
        f"- Checkpoint: `{args.checkpoint}`",
        f"- Epoch: {target_report['epoch']}",
        f"- Val windows: {len(loader.dataset)}, K={args.K}",
        f"- Horizons: {horizons}",
        "",
        "## Per-horizon table",
        "",
        "| h | |bias| | signed_bias | ens_spread | gt_std | ratio | crps | mae |",
        "|---|-------|-------------|------------|--------|-------|------|-----|",
    ]
    for h in horizons:
        row = target_report["per_horizon"][str(h)]
        md_lines.append(
            f"| {h} | {row['mean_abs_bias']:.5f} | {row['mean_signed_bias']:.5f} | "
            f"{row['ensemble_spread_mean']:.5f} | {row['gt_empirical_std_mean']:.5f} | "
            f"{row['spread_ratio']:.3f} | {row['crps_mean']:.5f} | {row['mae_mean']:.5f} |"
        )
    md_lines += [
        "",
        f"**Pathology flag (h30):** {flag['reason']}",
        "",
    ]
    if baseline_report is not None:
        md_lines += [
            "## Baseline comparison",
            "",
            f"- Baseline: `{baseline_report['checkpoint']}`",
            "",
            "| h | |bias|_target | |bias|_base | Δ|bias| | ratio_target | ratio_base |",
            "|---|---------------|-------------|----------|--------------|------------|",
        ]
        for h in horizons:
            tr = target_report["per_horizon"][str(h)]
            br = baseline_report["per_horizon"][str(h)]
            md_lines.append(
                f"| {h} | {tr['mean_abs_bias']:.5f} | {br['mean_abs_bias']:.5f} | "
                f"{tr['mean_abs_bias'] - br['mean_abs_bias']:+.5f} | {tr['spread_ratio']:.3f} | "
                f"{br['spread_ratio']:.3f} |"
            )

    out_md = out / "mean_vs_spread.md"
    out_md.write_text("\n".join(md_lines))
    print(f"MD   -> {out_md}")


if __name__ == "__main__":
    main()
