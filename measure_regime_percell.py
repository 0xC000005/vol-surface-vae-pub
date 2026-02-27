"""Measure per-cell calm vs turbulent regime CI coverage for model comparison.

Reports per-cell (5x5 grid) coverage broken down by regime and horizon.
Allows fair comparison between models with different history_len by using
a FIXED vol_of_vol threshold (from h=30 test set) rather than per-model quintiles.

Usage:
    PYTHONPATH=. python measure_regime_percell.py <model_path> [--n_windows N] [--n_samples S]
    PYTHONPATH=. python measure_regime_percell.py <model_path> --fixed_q20 0.01246 --fixed_q80 0.01884
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path

from diffusion.block_ar.block_ar_ddpm import ConditionalBlockARDDPM, BlockARConfig, denormalize_iv
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", type=str)
    parser.add_argument("--n_windows", type=int, default=400)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--fixed_q20", type=float, default=None,
                        help="Fixed Q20 threshold for regime classification (for cross-model comparison)")
    parser.add_argument("--fixed_q80", type=float, default=None,
                        help="Fixed Q80 threshold for regime classification")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    device = args.device

    # Load model
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    import dataclasses
    cfg = checkpoint["config"]
    if dataclasses.is_dataclass(cfg):
        cfg = dataclasses.asdict(cfg)
    config = BlockARConfig(**{k: v for k, v in cfg.items() if k in BlockARConfig.__dataclass_fields__})
    model = ConditionalBlockARDDPM(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded {args.model_path} (epoch {checkpoint.get('epoch', '?')})")
    print(f"  ratio_target_mode={config.ratio_target_mode}, history_len={config.history_len}")

    # Load data (test set)
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    history_len = config.history_len
    test_ds = VolSurfaceDataset(surfaces, start_idx=4540, end_idx=len(surfaces),
                                history_len=history_len, future_len=30)
    print(f"Test set: {len(test_ds)} windows")

    n_windows = min(args.n_windows, len(test_ds))
    indices = np.linspace(0, len(test_ds) - 1, n_windows, dtype=int)

    # Compute vol_of_vol for all test windows
    all_vov = []
    for i in range(len(test_ds)):
        batch = test_ds[i]
        hist = denormalize_iv(batch["history"]).numpy()
        mean_iv = hist.mean(axis=(-2, -1))
        daily_diff = np.diff(mean_iv)
        all_vov.append(daily_diff.std())
    all_vov = np.array(all_vov)

    # Quintile thresholds
    if args.fixed_q20 is not None and args.fixed_q80 is not None:
        q20, q80 = args.fixed_q20, args.fixed_q80
        print(f"Using FIXED thresholds: Q20={q20:.5f}, Q80={q80:.5f}")
    else:
        q20 = np.percentile(all_vov, 20)
        q80 = np.percentile(all_vov, 80)
        print(f"Computed thresholds: Q20={q20:.5f}, Q80={q80:.5f}")

    # Per-cell coverage storage: regime -> horizon -> (5,5) arrays of [covered, total]
    horizons = [1, 7, 14, 30]
    H, W = 5, 5
    cell_covered = {}  # regime -> horizon -> (5,5) sum of covered
    cell_total = {}    # regime -> horizon -> (5,5) count
    cell_width = {}    # regime -> horizon -> (5,5) sum of widths

    for regime in ["calm", "turb", "mid"]:
        cell_covered[regime] = {h: np.zeros((H, W)) for h in horizons}
        cell_total[regime] = {h: np.zeros((H, W)) for h in horizons}
        cell_width[regime] = {h: np.zeros((H, W)) for h in horizons}

    for wi, idx in enumerate(indices):
        batch = test_ds[idx]
        history = batch["history"].unsqueeze(0).to(device)
        future_gt = denormalize_iv(batch["future"].to(device)).unsqueeze(0)

        with torch.no_grad():
            samples = model.sample_batched(history, n_samples=args.n_samples)

        samples = samples.squeeze(0)  # (S, 30, 5, 5)
        gt = future_gt.squeeze(0)     # (30, 5, 5)

        vov = all_vov[idx]
        regime = "calm" if vov <= q20 else ("turb" if vov >= q80 else "mid")

        for h in horizons:
            t = h - 1
            gt_slice = gt[t]  # (5, 5)
            sample_slice = samples[:, t, :, :]  # (S, 5, 5)
            lo = torch.quantile(sample_slice, 0.05, dim=0)  # (5, 5)
            hi = torch.quantile(sample_slice, 0.95, dim=0)  # (5, 5)
            covered = ((gt_slice >= lo) & (gt_slice <= hi)).float().cpu().numpy()  # (5, 5) of 0/1
            width = (hi - lo).cpu().numpy()  # (5, 5)

            cell_covered[regime][h] += covered
            cell_total[regime][h] += 1.0
            cell_width[regime][h] += width

        if (wi + 1) % 50 == 0:
            print(f"  {wi+1}/{n_windows} windows processed")

    # Report
    print("\n" + "=" * 70)
    print("PER-CELL REGIME COVERAGE (90% CI)")
    print("=" * 70)

    results_json = {}

    for regime in ["calm", "turb", "mid"]:
        n_wins = int(cell_total[regime][1].max()) if cell_total[regime][1].max() > 0 else 0
        if n_wins == 0:
            continue
        print(f"\n{'='*40}")
        print(f"{regime.upper()} regime ({n_wins} windows)")
        print(f"{'='*40}")

        regime_results = {}
        for h in horizons:
            cov_grid = cell_covered[regime][h] / np.maximum(cell_total[regime][h], 1)
            width_grid = cell_width[regime][h] / np.maximum(cell_total[regime][h], 1)

            print(f"\n  h={h} coverage (%):")
            for r in range(H):
                row_str = "    " + " ".join(f"{cov_grid[r,c]*100:5.1f}" for c in range(W))
                print(row_str)
            mean_cov = cov_grid.mean()
            min_cov = cov_grid.min()
            below_70 = (cov_grid < 0.70).sum()
            below_75 = (cov_grid < 0.75).sum()
            print(f"    Mean={mean_cov*100:.1f}%, Min={min_cov*100:.1f}%, "
                  f"Cells<70%={below_70}, Cells<75%={below_75}")

            print(f"  h={h} width:")
            for r in range(H):
                row_str = "    " + " ".join(f"{width_grid[r,c]:.4f}" for c in range(W))
                print(row_str)

            regime_results[str(h)] = {
                "coverage": cov_grid.tolist(),
                "width": width_grid.tolist(),
                "mean_coverage": float(mean_cov),
                "min_coverage": float(min_cov),
                "cells_below_70": int(below_70),
                "cells_below_75": int(below_75),
            }

        results_json[regime] = {"n_windows": n_wins, "horizons": regime_results}

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for regime in ["calm", "turb"]:
        if regime not in results_json:
            continue
        for h in [7, 14, 30]:
            r = results_json[regime]["horizons"][str(h)]
            print(f"  {regime.upper()} h={h}: mean={r['mean_coverage']*100:.1f}%, "
                  f"min={r['min_coverage']*100:.1f}%, cells<75%={r['cells_below_75']}")

    # Width ratio per cell
    if "calm" in results_json and "turb" in results_json:
        print(f"\n  Width ratio turb/calm (h=7):")
        calm_w = np.array(results_json["calm"]["horizons"]["7"]["width"])
        turb_w = np.array(results_json["turb"]["horizons"]["7"]["width"])
        ratio_grid = turb_w / np.maximum(calm_w, 1e-6)
        for r in range(H):
            row_str = "    " + " ".join(f"{ratio_grid[r,c]:5.3f}" for c in range(W))
            print(row_str)
        print(f"    Mean ratio: {ratio_grid.mean():.3f}x")

    results_json["model"] = args.model_path
    results_json["q20"] = float(q20)
    results_json["q80"] = float(q80)
    results_json["n_windows"] = n_windows

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results_json, f, indent=2)
        print(f"\nSaved to {args.output_json}")

    print(f"\nModel: {args.model_path}")


if __name__ == "__main__":
    main()
