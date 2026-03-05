"""Fit per-cell quantile mapping from generated to GT daily changes.

Generates samples on TRAINING data, computes per-cell quantile functions
for both generated and GT daily changes, saves as .npz for use with
QuantileMapper at inference time.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/fit_quantile_map.py \
        --model_path models/backfill/afcrps_90d/best_model.pt \
        --no_ema --n_samples 50 --batch_size 16 --max_batches 100 \
        --output models/backfill/afcrps_90d/quantile_map.npz --device cuda
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats as sp_stats
from torch.utils.data import DataLoader
from tqdm import tqdm

from diffusion.block_ar.single_pass_ar import SinglePassBlockAR, SinglePassConfig, denormalize_iv
from experiments.backfill.block_ar.quantile_mapper import QuantileMapper
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


def load_model(args):
    checkpoint = torch.load(args.model_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if isinstance(cfg, dict):
        cfg = SinglePassConfig(**cfg)
    model = SinglePassBlockAR(cfg)
    key = "ema_state_dict" if not args.no_ema and "ema_state_dict" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[key])
    model.eval().to(args.device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Fit per-cell quantile mapping")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_batches", type=int, default=100,
                        help="Max training batches to use (0=all)")
    parser.add_argument("--output", type=str, default="quantile_map.npz")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--shape_only", action="store_true",
                        help="Standardize before mapping — only fix shape (kurtosis/skewness), preserve scale")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    args.device = device

    print(f"Model: {args.model_path}")
    print(f"Samples per window: {args.n_samples}")
    print(f"Max batches: {args.max_batches or 'all'}")
    print(f"Output: {args.output}")

    # Load model
    model = load_model(args)

    # Load training data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    train_dataset = VolSurfaceDataset(surfaces, 30, 30, start_idx=0, end_idx=4040)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False)

    total_batches = len(train_loader)
    if args.max_batches > 0:
        total_batches = min(total_batches, args.max_batches)
    print(f"Training windows: {len(train_dataset)}, batches: {total_batches}")

    # Collect generated daily changes
    all_gen_changes = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(train_loader, desc="Generating", total=total_batches)):
            if args.max_batches > 0 and i >= args.max_batches:
                break
            history = batch["history"].to(device)
            samples = model.sample(history, n_samples=args.n_samples)
            # samples: (B, S, 30, 5, 5) in IV [0, 1]
            history_denorm = denormalize_iv(history).cpu().numpy()
            samples_np = samples.cpu().numpy()

            B, S, T, H, W = samples_np.shape
            anchor = history_denorm[:, -1:]  # (B, 1, 5, 5)
            anchor_exp = np.broadcast_to(
                anchor[:, np.newaxis, :, :, :], (B, S, 1, H, W)
            ).copy()
            full_traj = np.concatenate([anchor_exp, samples_np], axis=2)
            changes = np.diff(full_traj, axis=2)  # (B, S, T, 5, 5)
            all_gen_changes.append(changes.reshape(-1, H, W))

    gen_changes = np.concatenate(all_gen_changes, axis=0)  # (M, 5, 5)
    print(f"Generated changes collected: {gen_changes.shape[0]:,}")

    # GT daily changes from raw training surfaces
    gt_changes = np.diff(surfaces[:4040], axis=0)  # (4039, 5, 5)
    print(f"GT daily changes: {gt_changes.shape[0]:,}")

    # Compute quantiles
    n_quantiles = 199
    quantile_levels = np.linspace(0.005, 0.995, n_quantiles)
    gen_quantiles = np.zeros((25, n_quantiles))
    gt_quantiles = np.zeros((25, n_quantiles))
    gen_mean = np.zeros(25)
    gen_std_arr = np.zeros(25)

    print(f"\nMode: {'shape_only' if args.shape_only else 'full'}")
    print("\nPer-cell statistics (before mapping):")
    print(f"{'Cell':>8} {'GT std':>8} {'Gen std':>8} {'GT kurt':>8} {'Gen kurt':>8} {'GT skew':>8} {'Gen skew':>8}")
    for r in range(5):
        for c in range(5):
            idx = r * 5 + c
            gv = gt_changes[:, r, c]
            ev = gen_changes[:, r, c]
            gen_mean[idx] = ev.mean()
            gen_std_arr[idx] = ev.std()
            if args.shape_only:
                # Standardize both to unit variance before computing quantiles
                ev_z = (ev - ev.mean()) / max(ev.std(), 1e-10)
                gv_z = (gv - gv.mean()) / max(gv.std(), 1e-10)
                gen_quantiles[idx] = np.quantile(ev_z, quantile_levels)
                gt_quantiles[idx] = np.quantile(gv_z, quantile_levels)
            else:
                gen_quantiles[idx] = np.quantile(ev, quantile_levels)
                gt_quantiles[idx] = np.quantile(gv, quantile_levels)
            print(f"  ({r},{c})  {gv.std():8.5f} {ev.std():8.5f} "
                  f"{sp_stats.kurtosis(gv, fisher=True):8.2f} {sp_stats.kurtosis(ev, fisher=True):8.2f} "
                  f"{sp_stats.skew(gv):8.3f} {sp_stats.skew(ev):8.3f}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_dict = dict(
        gen_quantiles=gen_quantiles,
        gt_quantiles=gt_quantiles,
        quantile_levels=quantile_levels,
        model_path=str(args.model_path),
        n_gen_changes=gen_changes.shape[0],
        n_gt_changes=gt_changes.shape[0],
        shape_only=args.shape_only,
    )
    if args.shape_only:
        save_dict["gen_mean"] = gen_mean
        save_dict["gen_std"] = gen_std_arr
    np.savez(args.output, **save_dict)
    print(f"\nSaved quantile map to {args.output}")

    # Verify: apply mapping to gen changes and recompute stats
    print("\nPer-cell statistics (after mapping):")
    print(f"{'Cell':>8} {'GT std':>8} {'Map std':>8} {'GT kurt':>8} {'Map kurt':>8} {'GT skew':>8} {'Map skew':>8}")

    gt_stds, gen_stds, map_stds = [], [], []
    gt_kurts, gen_kurts, map_kurts = [], [], []
    gt_skews, gen_skews, map_skews = [], [], []

    for r in range(5):
        for c in range(5):
            idx = r * 5 + c
            gv = gt_changes[:, r, c]
            ev = gen_changes[:, r, c]
            if args.shape_only:
                z = (ev - gen_mean[idx]) / max(gen_std_arr[idx], 1e-10)
                z_mapped = QuantileMapper._interp_with_extrapolation(
                    z, gen_quantiles[idx], gt_quantiles[idx]
                )
                mv = z_mapped * gen_std_arr[idx] + gen_mean[idx]
            else:
                mv = QuantileMapper._interp_with_extrapolation(
                    ev, gen_quantiles[idx], gt_quantiles[idx]
                )
            gt_stds.append(gv.std())
            gen_stds.append(ev.std())
            map_stds.append(mv.std())
            gt_kurts.append(sp_stats.kurtosis(gv, fisher=True))
            gen_kurts.append(sp_stats.kurtosis(ev, fisher=True))
            map_kurts.append(sp_stats.kurtosis(mv, fisher=True))
            gt_skews.append(sp_stats.skew(gv))
            gen_skews.append(sp_stats.skew(ev))
            map_skews.append(sp_stats.skew(mv))
            print(f"  ({r},{c})  {gv.std():8.5f} {mv.std():8.5f} "
                  f"{sp_stats.kurtosis(gv, fisher=True):8.2f} {sp_stats.kurtosis(mv, fisher=True):8.2f} "
                  f"{sp_stats.skew(gv):8.3f} {sp_stats.skew(mv):8.3f}")

    gt_stds, gen_stds, map_stds = np.array(gt_stds), np.array(gen_stds), np.array(map_stds)
    gt_kurts, gen_kurts, map_kurts = np.array(gt_kurts), np.array(gen_kurts), np.array(map_kurts)
    gt_skews, gen_skews, map_skews = np.array(gt_skews), np.array(gen_skews), np.array(map_skews)

    # Correlations
    from scipy.stats import pearsonr
    print(f"\nCorrelations (GT vs Gen → GT vs Mapped):")
    print(f"  Std:      r={pearsonr(gt_stds, gen_stds)[0]:.3f} → r={pearsonr(gt_stds, map_stds)[0]:.3f}")
    print(f"  Kurtosis: r={pearsonr(gt_kurts, gen_kurts)[0]:.3f} → r={pearsonr(gt_kurts, map_kurts)[0]:.3f}")
    print(f"  Skewness: r={pearsonr(gt_skews, gen_skews)[0]:.3f} → r={pearsonr(gt_skews, map_skews)[0]:.3f}")

    # Diagnostic scatter plot
    output_dir = Path(args.output).parent
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (gt_v, gen_v, map_v, label) in zip(axes, [
        (gt_stds, gen_stds, map_stds, "Std"),
        (gt_kurts, gen_kurts, map_kurts, "Excess Kurtosis"),
        (gt_skews, gen_skews, map_skews, "Skewness"),
    ]):
        ax.scatter(gt_v, gen_v, c="tab:red", alpha=0.7, s=60, label="Before mapping", zorder=3)
        ax.scatter(gt_v, map_v, c="tab:blue", alpha=0.7, s=60, label="After mapping", zorder=4)
        lims = [min(gt_v.min(), gen_v.min(), map_v.min()), max(gt_v.max(), gen_v.max(), map_v.max())]
        margin = (lims[1] - lims[0]) * 0.1
        ax.plot([lims[0]-margin, lims[1]+margin], [lims[0]-margin, lims[1]+margin],
                "k--", alpha=0.3, label="y=x")
        ax.set_xlabel(f"GT {label}")
        ax.set_ylabel(f"Generated {label}")
        ax.set_title(f"Per-Cell {label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Quantile Mapping: Before vs After", fontsize=14)
    plt.tight_layout()
    plot_path = output_dir / "quantile_map_diagnostic.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Diagnostic plot saved to {plot_path}")


if __name__ == "__main__":
    main()
