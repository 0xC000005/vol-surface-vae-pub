#!/usr/bin/env python
"""Compare cross-cell correlation structure between models (90d, 91c, 91e) and GT.

Generates daily changes from each model at 30-day horizon, computes 25x25 correlation
matrix, and compares to GT daily-change correlation.

Usage:
    PYTHONPATH=. python experiments/backfill/block_ar/compare_cross_cell_corr.py \
        --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.block_ar.single_pass_ar import SinglePassBlockAR, SinglePassConfig
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset

LABELS_K = ["K70", "K85", "K100", "K115", "K130"]
LABELS_T = ["1M", "3M", "6M", "1Y", "2Y"]
CELL_NAMES = [f"{t}/{k}" for t in LABELS_T for k in LABELS_K]

MODELS = {
    "90d": ("models/backfill/afcrps_90d/best_model.pt", {"ar_frame_floor_clamp": 0.01}),
    "91c": ("models/backfill/afcrps_91c/best_model.pt", {}),
    "91e": ("models/backfill/afcrps_91e/best_model.pt", {}),
}


def load_model(path, device, overrides=None):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", ckpt.get("model_config", {}))
    if isinstance(cfg_dict, dict):
        cfg = SinglePassConfig(**{k: v for k, v in cfg_dict.items() if hasattr(SinglePassConfig, k)})
    else:
        cfg = cfg_dict
    if overrides:
        for k, v in overrides.items():
            setattr(cfg, k, v)
    model = SinglePassBlockAR(cfg)
    sd = ckpt.get("model_state_dict", ckpt.get("state_dict"))
    model.load_state_dict(sd, strict=False)
    model.to(device).eval()
    return model


def get_daily_changes(model, dataset, device, n_windows=50, n_samples=50):
    """Generate samples and compute daily changes (N_total, 25)."""
    all_changes = []
    rng = np.random.RandomState(42)
    indices = rng.choice(len(dataset), min(n_windows, len(dataset)), replace=False)

    for i, idx in enumerate(indices):
        item = dataset[idx]
        hist = item["history"]
        if isinstance(hist, np.ndarray):
            hist_t = torch.from_numpy(hist).unsqueeze(0).to(device)
        else:
            hist_t = hist.unsqueeze(0).to(device)
            hist = hist.numpy()

        with torch.no_grad():
            samples = model.sample(hist_t, n_samples=n_samples)  # (1, S, T, 5, 5)

        samples_np = samples.cpu().numpy()[0]  # (S, T, 5, 5)
        # Prepend last history frame as anchor
        anchor = hist[-1:]  # (1, 5, 5)
        anchored = np.concatenate([
            np.broadcast_to(anchor, (n_samples, 1, 5, 5)),
            samples_np
        ], axis=1)  # (S, T+1, 5, 5)
        changes = np.diff(anchored, axis=1)  # (S, T, 5, 5)
        changes_flat = changes.reshape(-1, 25)  # (S*T, 25)
        all_changes.append(changes_flat)

        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(indices)} windows done")

    return np.concatenate(all_changes, axis=0)  # (N_total, 25)


def get_gt_changes(surfaces):
    """GT daily changes from raw surfaces."""
    changes = np.diff(surfaces, axis=0)  # (N-1, 5, 5)
    return changes.reshape(-1, 25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n_windows", type=int, default=50)
    parser.add_argument("--n_samples", type=int, default=50)
    parser.add_argument("--output_dir", default="results/block_ar/cross_cell_corr_compare")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data
    data = np.load("data/vol_surface_with_ret.npz")
    surfaces = data["surface"]
    n_train = 4040
    test_surfaces = surfaces[n_train:]
    dataset = VolSurfaceDataset(surfaces, 30, 30, start_idx=n_train, end_idx=len(surfaces))

    # GT correlation
    gt_changes = get_gt_changes(test_surfaces)
    gt_corr = np.corrcoef(gt_changes.T)  # (25, 25)

    # Model correlations
    model_corrs = {}
    model_stats = {}
    for name, (path, overrides) in MODELS.items():
        print(f"\n=== {name} ===")
        model = load_model(path, device, overrides)
        changes = get_daily_changes(model, dataset, device, args.n_windows, args.n_samples)
        corr = np.corrcoef(changes.T)
        model_corrs[name] = corr
        model_stats[name] = {
            "mean_corr": float(np.mean(corr[np.triu_indices(25, k=1)])),
            "std_corr": float(np.std(corr[np.triu_indices(25, k=1)])),
            "min_corr": float(np.min(corr[np.triu_indices(25, k=1)])),
            "max_corr": float(np.max(corr[np.triu_indices(25, k=1)])),
            "frobenius_to_gt": float(np.linalg.norm(corr - gt_corr, 'fro')),
            "n_changes": len(changes),
        }
        del model
        torch.cuda.empty_cache()

    # GT stats
    gt_triu = gt_corr[np.triu_indices(25, k=1)]
    print(f"\n=== GT ===")
    print(f"  Mean corr: {np.mean(gt_triu):.4f}, Std: {np.std(gt_triu):.4f}")
    print(f"  Range: [{np.min(gt_triu):.4f}, {np.max(gt_triu):.4f}]")

    for name, st in model_stats.items():
        print(f"\n=== {name} ===")
        for k, v in st.items():
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Within-tenor vs across-tenor correlation
    print("\n=== Within-tenor vs Across-tenor Correlation ===")
    within_mask = np.zeros((25, 25), dtype=bool)
    for t in range(5):
        for k1 in range(5):
            for k2 in range(k1+1, 5):
                within_mask[t*5+k1, t*5+k2] = True

    across_mask = np.triu(np.ones((25, 25), dtype=bool), k=1) & ~within_mask

    print(f"  GT:  within={np.mean(gt_corr[within_mask]):.4f}, across={np.mean(gt_corr[across_mask]):.4f}, "
          f"ratio={np.mean(gt_corr[within_mask])/np.mean(gt_corr[across_mask]):.2f}")
    for name, corr in model_corrs.items():
        w = np.mean(corr[within_mask])
        a = np.mean(corr[across_mask])
        print(f"  {name}: within={w:.4f}, across={a:.4f}, ratio={w/a:.2f}")

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    vmin, vmax = 0, 1
    titles = ["GT", "90d (baseline)", "91c (factor)", "91e (norm factor)"]
    mats = [gt_corr, model_corrs["90d"], model_corrs["91c"], model_corrs["91e"]]

    for ax, title, mat in zip(axes, titles, mats):
        im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="RdYlBu_r")
        ax.set_title(title, fontsize=12)
        ax.set_xticks(range(0, 25, 5))
        ax.set_xticklabels(LABELS_T, fontsize=8)
        ax.set_yticks(range(0, 25, 5))
        ax.set_yticklabels(LABELS_T, fontsize=8)
        # Add tenor block boundaries
        for b in [5, 10, 15, 20]:
            ax.axhline(b-0.5, color='black', linewidth=0.5)
            ax.axvline(b-0.5, color='black', linewidth=0.5)

    fig.colorbar(im, ax=axes[-1], shrink=0.8)
    fig.suptitle("Cross-Cell Daily Change Correlation: GT vs Models", fontsize=14)
    plt.tight_layout()
    plt.savefig(outdir / "cross_cell_correlation.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved to {outdir / 'cross_cell_correlation.png'}")

    # Error heatmaps (model - GT)
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
    for ax, name in zip(axes2, ["90d", "91c", "91e"]):
        diff = model_corrs[name] - gt_corr
        im = ax.imshow(diff, vmin=-0.3, vmax=0.3, cmap="RdBu_r")
        ax.set_title(f"{name} - GT (Frob={model_stats[name]['frobenius_to_gt']:.2f})", fontsize=12)
        ax.set_xticks(range(0, 25, 5))
        ax.set_xticklabels(LABELS_T, fontsize=8)
        ax.set_yticks(range(0, 25, 5))
        ax.set_yticklabels(LABELS_T, fontsize=8)
        for b in [5, 10, 15, 20]:
            ax.axhline(b-0.5, color='black', linewidth=0.5)
            ax.axvline(b-0.5, color='black', linewidth=0.5)
    fig2.colorbar(im, ax=axes2[-1], shrink=0.8)
    fig2.suptitle("Correlation Error (Model - GT)", fontsize=14)
    plt.tight_layout()
    plt.savefig(outdir / "correlation_error.png", dpi=150, bbox_inches="tight")
    print(f"Saved to {outdir / 'correlation_error.png'}")


if __name__ == "__main__":
    main()
