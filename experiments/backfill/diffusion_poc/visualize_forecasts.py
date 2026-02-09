#!/usr/bin/env python
"""
Visualization of Block-AR diffusion model volatility surface forecasts.

Produces three types of publication-quality plots:

1. Conditional Trajectory Fan Charts (diverse conditioning histories)
2. Surface Evolution Heatmaps (generated vs GT at selected horizons)
3. Term Structure & Vol Smile Cross-Sections (generated vs GT spatial structure)

Usage:
    PYTHONPATH=. python experiments/backfill/diffusion_poc/visualize_forecasts.py \
        --no_ema --n_samples 30
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from diffusion.block_ar.block_ar_ddpm import (
    BlockARConfig,
    ConditionalBlockARDDPM,
    denormalize_iv,
)
from experiments.backfill.diffusion_poc.config_block_ar import get_default_config
from experiments.backfill.diffusion_poc.train_ddpm_poc import VolSurfaceDataset


TENOR_LABELS = ["1mo", "2mo", "4mo", "8mo", "12mo"]
MONEYNESS_LABELS = ["ITM", "ITM-", "ATM", "OTM-", "OTM"]

SELECTED_CELLS = [
    (0, 2, "ATM / 1mo"),
    (4, 2, "ATM / 12mo"),
    (0, 4, "OTM / 1mo"),
    (4, 0, "ITM / 12mo"),
]


# ============================================================================
# Plot Type 1: Fan Charts with diverse conditioning + uncertainty annotation
# ============================================================================

def plot_fan_charts(
    histories, futures_gt, all_samples, output_path,
    history_len=30, future_len=30,
):
    n_examples = len(histories)
    n_cells = len(SELECTED_CELLS)

    fig, axes = plt.subplots(
        n_cells, n_examples,
        figsize=(6.0 * n_examples, 3.5 * n_cells),
        squeeze=False,
    )

    time_hist = np.arange(history_len)
    time_fut = np.arange(history_len, history_len + future_len)

    # First pass: compute shared y-limits per row (same grid cell across examples)
    row_ylims = {}
    for row_idx, (r, c, label) in enumerate(SELECTED_CELLS):
        all_vals = []
        for col_idx in range(n_examples):
            hist = histories[col_idx]
            gt = futures_gt[col_idx]
            samples = all_samples[col_idx]
            sample_series = samples[:, :, r, c]
            q05, q95 = np.quantile(sample_series, [0.05, 0.95], axis=0)
            all_vals.extend([q05.min(), q95.max(), gt[:, r, c].min(), gt[:, r, c].max(),
                             hist[-5:, r, c].min(), hist[-5:, r, c].max()])
        ymin, ymax = min(all_vals), max(all_vals)
        ypad = (ymax - ymin) * 0.12
        row_ylims[row_idx] = (ymin - ypad, ymax + ypad)

    # Second pass: plot with shared y-axes per row
    for col_idx in range(n_examples):
        hist = histories[col_idx]
        gt = futures_gt[col_idx]
        samples = all_samples[col_idx]  # (n_samples, future_len, 5, 5)

        for row_idx, (r, c, label) in enumerate(SELECTED_CELLS):
            ax = axes[row_idx, col_idx]

            hist_series = hist[:, r, c]
            gt_series = gt[:, r, c]
            sample_series = samples[:, :, r, c]  # (n_samples, future_len)

            q05, q10, q25, q50, q75, q90, q95 = np.quantile(
                sample_series, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95], axis=0
            )

            # History
            ax.plot(time_hist, hist_series, color="black", linewidth=1.5, label="History")

            # Ground truth
            ax.plot(
                [time_hist[-1], time_fut[0]], [hist_series[-1], gt_series[0]],
                color="#d62728", linewidth=1.2, linestyle="--",
            )
            ax.plot(time_fut, gt_series, color="#d62728", linewidth=1.2, linestyle="--", label="GT")

            # Fan bands
            ax.fill_between(time_fut, q05, q95, color="#2166ac", alpha=0.12, label="5-95%")
            ax.fill_between(time_fut, q10, q90, color="#4393c3", alpha=0.18, label="10-90%")
            ax.fill_between(time_fut, q25, q75, color="#92c5de", alpha=0.28, label="25-75%")
            ax.plot(time_fut, q50, color="#2166ac", linewidth=1.0, label="Median")

            # Forecast boundary
            ax.axvline(x=history_len - 0.5, color="gray", linewidth=0.8, linestyle=":", alpha=0.7)

            # Annotate 90% CI width at h=1 and h=30
            w1 = q95[0] - q05[0]
            w30 = q95[-1] - q05[-1]
            ax.annotate(
                f"w={w1:.3f}", xy=(time_fut[0], q95[0]),
                fontsize=7, color="#2166ac", ha="left", va="bottom",
            )
            ax.annotate(
                f"w={w30:.3f}", xy=(time_fut[-1], q95[-1]),
                fontsize=7, color="#2166ac", ha="right", va="bottom",
            )

            # Shared y-axis per row
            ax.set_ylim(row_ylims[row_idx])

            if row_idx == 0:
                ax.set_title(f"Example {col_idx + 1}", fontsize=12, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{label}\nIV", fontsize=10)
            if row_idx == n_cells - 1:
                ax.set_xlabel("Day", fontsize=10)

            ax.tick_params(labelsize=8)
            ax.set_xlim(0, history_len + future_len - 1)

            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7, loc="upper left", framealpha=0.9)

    fig.suptitle(
        "Block-AR Diffusion: Conditional Forecast Fan Charts",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved fan charts to {output_path}")


# ============================================================================
# Plot Type 2: Surface Evolution Heatmaps
# ============================================================================

def plot_surface_evolution(
    history, future_gt, sample_path, output_path, history_len=30,
):
    forecast_steps = [0, 4, 9, 19, 29]
    step_labels = ["Last History"] + [f"h={s+1}" for s in forecast_steps]
    n_cols = 1 + len(forecast_steps)

    gen_surfaces = [history[-1]]
    gt_surfaces = [history[-1]]
    for s in forecast_steps:
        gen_surfaces.append(sample_path[s])
        gt_surfaces.append(future_gt[s])

    all_vals = np.concatenate([s.ravel() for s in gen_surfaces + gt_surfaces])
    vmin = float(np.percentile(all_vals, 1))
    vmax = float(np.percentile(all_vals, 99))

    fig, axes = plt.subplots(2, n_cols, figsize=(3.2 * n_cols, 7.0), squeeze=False)

    for row_idx, (surfaces, row_label) in enumerate(
        [(gen_surfaces, "Generated"), (gt_surfaces, "Ground Truth")]
    ):
        for col_idx, (surf, slabel) in enumerate(zip(surfaces, step_labels)):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(surf, cmap="RdYlBu_r", vmin=vmin, vmax=vmax, aspect="equal", origin="upper")
            ax.set_xticks(range(5))
            ax.set_xticklabels(MONEYNESS_LABELS, fontsize=7, rotation=45, ha="right")
            ax.set_yticks(range(5))
            ax.set_yticklabels(TENOR_LABELS, fontsize=7)
            if row_idx == 0:
                ax.set_title(slabel, fontsize=9, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\nTenor", fontsize=9, fontweight="bold")
            for i in range(5):
                for j in range(5):
                    val = surf[i, j]
                    text_color = "white" if val > (vmin + vmax) / 2 else "black"
                    ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=6, color=text_color)

    fig.suptitle("Surface Evolution (Generated vs Ground Truth)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.91, 0.95])
    cbar_ax = fig.add_axes([0.93, 0.12, 0.015, 0.75])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Implied Volatility", fontsize=10)
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved surface evolution to {output_path}")


# ============================================================================
# Plot Type 3: Term Structure & Vol Smile Cross-Sections
# ============================================================================

def plot_cross_sections(
    future_gt, samples, output_path, future_len=30,
):
    """
    Plot term structure (IV vs tenor) and vol smile (IV vs moneyness)
    for generated median + sample envelope vs ground truth at selected horizons.

    Args:
        future_gt: (future_len, 5, 5) ground truth
        samples: (n_samples, future_len, 5, 5) generated samples
        output_path: save path
    """
    horizons = [0, 4, 9, 19, 29]  # h=1, 5, 10, 20, 30
    horizon_labels = [f"h={h+1}" for h in horizons]
    n_h = len(horizons)
    tenor_x = np.array([1, 2, 4, 8, 12])
    money_x = np.arange(5)

    fig, axes = plt.subplots(2, n_h, figsize=(4.0 * n_h, 7.0), squeeze=False)

    for col_idx, (h_idx, h_label) in enumerate(zip(horizons, horizon_labels)):
        gt_surf = future_gt[h_idx]  # (5, 5)
        gen_surfs = samples[:, h_idx]  # (n_samples, 5, 5)
        gen_median = np.median(gen_surfs, axis=0)
        gen_q10 = np.quantile(gen_surfs, 0.10, axis=0)
        gen_q90 = np.quantile(gen_surfs, 0.90, axis=0)

        # --- Top row: Term structure (IV vs tenor) at ATM (col=2) ---
        ax = axes[0, col_idx]
        atm_col = 2
        ax.plot(tenor_x, gt_surf[:, atm_col], "o-", color="#d62728", linewidth=1.5, markersize=5, label="GT")
        ax.plot(tenor_x, gen_median[:, atm_col], "s-", color="#2166ac", linewidth=1.5, markersize=5, label="Median")
        ax.fill_between(
            tenor_x, gen_q10[:, atm_col], gen_q90[:, atm_col],
            color="#2166ac", alpha=0.2, label="10-90%",
        )
        ax.set_title(h_label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Tenor (months)", fontsize=9)
        if col_idx == 0:
            ax.set_ylabel("ATM Term Structure\nIV", fontsize=10)
        ax.set_xticks(tenor_x)
        ax.tick_params(labelsize=8)
        if col_idx == 0:
            ax.legend(fontsize=7, framealpha=0.9)

        # --- Bottom row: Vol smile (IV vs moneyness) at short tenor (row=0) ---
        ax = axes[1, col_idx]
        tenor_row = 0
        ax.plot(money_x, gt_surf[tenor_row, :], "o-", color="#d62728", linewidth=1.5, markersize=5, label="GT")
        ax.plot(money_x, gen_median[tenor_row, :], "s-", color="#2166ac", linewidth=1.5, markersize=5, label="Median")
        ax.fill_between(
            money_x, gen_q10[tenor_row, :], gen_q90[tenor_row, :],
            color="#2166ac", alpha=0.2, label="10-90%",
        )
        ax.set_xlabel("Moneyness", fontsize=9)
        ax.set_xticks(money_x)
        ax.set_xticklabels(MONEYNESS_LABELS, fontsize=8)
        if col_idx == 0:
            ax.set_ylabel("1mo Vol Smile\nIV", fontsize=10)
        ax.tick_params(labelsize=8)
        if col_idx == 0:
            ax.legend(fontsize=7, framealpha=0.9)

    fig.suptitle(
        "Term Structure (ATM) & Vol Smile (1mo) — Generated vs Ground Truth",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved cross-sections to {output_path}")


def plot_cross_sections_multi_tenor(
    future_gt, samples, output_path, future_len=30,
):
    """
    Vol smile at multiple tenors (rows 0, 2, 4) at h=1, h=10, h=30.
    """
    horizons = [0, 9, 29]
    horizon_labels = ["h=1", "h=10", "h=30"]
    tenor_rows = [0, 2, 4]
    tenor_names = ["1mo", "4mo", "12mo"]
    money_x = np.arange(5)

    fig, axes = plt.subplots(
        len(tenor_rows), len(horizons),
        figsize=(5.0 * len(horizons), 3.5 * len(tenor_rows)),
        squeeze=False,
    )

    for col_idx, (h_idx, h_label) in enumerate(zip(horizons, horizon_labels)):
        gt_surf = future_gt[h_idx]
        gen_surfs = samples[:, h_idx]
        gen_median = np.median(gen_surfs, axis=0)
        gen_q10 = np.quantile(gen_surfs, 0.10, axis=0)
        gen_q90 = np.quantile(gen_surfs, 0.90, axis=0)
        gen_q25 = np.quantile(gen_surfs, 0.25, axis=0)
        gen_q75 = np.quantile(gen_surfs, 0.75, axis=0)

        for row_idx, (tr, tn) in enumerate(zip(tenor_rows, tenor_names)):
            ax = axes[row_idx, col_idx]

            # Individual samples as thin lines
            for s_idx in range(min(20, samples.shape[0])):
                ax.plot(money_x, gen_surfs[s_idx, tr, :], color="#2166ac", alpha=0.08, linewidth=0.5)

            # Envelope
            ax.fill_between(
                money_x, gen_q10[tr, :], gen_q90[tr, :],
                color="#2166ac", alpha=0.15,
            )
            ax.fill_between(
                money_x, gen_q25[tr, :], gen_q75[tr, :],
                color="#2166ac", alpha=0.2,
            )

            # Median and GT
            ax.plot(money_x, gen_median[tr, :], "s-", color="#2166ac", linewidth=2, markersize=5, label="Gen median", zorder=5)
            ax.plot(money_x, gt_surf[tr, :], "o-", color="#d62728", linewidth=2, markersize=5, label="GT", zorder=6)

            ax.set_xticks(money_x)
            ax.set_xticklabels(MONEYNESS_LABELS, fontsize=8)
            ax.tick_params(labelsize=8)

            if col_idx == 0:
                ax.set_ylabel(f"{tn}\nIV", fontsize=10, fontweight="bold")
            if row_idx == 0:
                ax.set_title(h_label, fontsize=11, fontweight="bold")
            if row_idx == len(tenor_rows) - 1:
                ax.set_xlabel("Moneyness", fontsize=9)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7, framealpha=0.9)

    fig.suptitle(
        "Vol Smile by Tenor — Generated Samples vs Ground Truth",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved multi-tenor smiles to {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize Block-AR forecasts")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--n_samples", type=int, default=30)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config = get_default_config()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"

    model_path = args.model_path
    if model_path is None:
        for path in [
            "models/backfill/block_ar_dual_path/best_coverage_model.pt",
            f"{config.output_dir}/best_coverage_model.pt",
        ]:
            if Path(path).exists():
                model_path = path
                break

    if model_path is None or not Path(model_path).exists():
        print(f"ERROR: No model found at {model_path}")
        return

    output_dir = args.output_dir or "results/block_ar_dual_path"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    print("=" * 60)
    print("Block-AR Diffusion: Forecast Visualization")
    print("=" * 60)

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint["config"]
    if isinstance(model_config, dict):
        model_config = BlockARConfig(**model_config)

    model = ConditionalBlockARDDPM(model_config)
    if "ema_params" in checkpoint and not args.no_ema:
        state_dict = model.state_dict()
        for name in state_dict:
            if name in checkpoint["ema_params"]:
                state_dict[name] = checkpoint["ema_params"][name]
        model.load_state_dict(state_dict)
        print("  Loaded EMA parameters")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
        print("  Loaded regular weights (no EMA)")

    model = model.to(device)
    model.eval()
    print(f"  Epoch {checkpoint.get('epoch', '?')}, {sum(p.numel() for p in model.parameters()):,} params")

    # Load val data
    data = np.load(config.data_path)
    surfaces = data["surface"]
    val_dataset = VolSurfaceDataset(
        surfaces, config.history_len, config.future_len,
        start_idx=config.val_start, end_idx=config.val_end,
    )

    # Pick DIVERSE examples — spread across the val set to show different regimes
    n_val = len(val_dataset)
    example_indices = [0, n_val // 4, n_val // 2]  # ~0, ~110, ~220
    print(f"\n  Val set size: {n_val}")
    print(f"  Using examples at indices: {example_indices}")

    histories_list = []
    futures_list = []
    samples_list = []

    for idx in example_indices:
        item = val_dataset[idx]
        h = item["history"].unsqueeze(0).to(device)  # (1, 30, 5, 5)
        f = item["future"].unsqueeze(0).to(device)

        print(f"\n  Generating {args.n_samples} samples for example {idx}...")
        with torch.no_grad():
            s = model.sample_batched(h, n_samples=args.n_samples)  # (1, S, 30, 5, 5)

        histories_list.append(denormalize_iv(h).squeeze(0).cpu().numpy())
        futures_list.append(denormalize_iv(f).squeeze(0).cpu().numpy())
        samples_list.append(s.squeeze(0).cpu().numpy())  # (S, 30, 5, 5)

    # ---- Plot 1: Fan Charts ----
    print("\nPlot 1: Fan charts...")
    plot_fan_charts(
        histories_list, futures_list, samples_list,
        f"{output_dir}/fan_charts.png",
        history_len=config.history_len, future_len=config.future_len,
    )

    # ---- Plot 2: Surface Evolution ----
    print("\nPlot 2: Surface evolution...")
    plot_surface_evolution(
        histories_list[0], futures_list[0], samples_list[0][0],
        f"{output_dir}/surface_evolution.png",
        history_len=config.history_len,
    )

    # ---- Plot 3: Cross-sections (term structure + smile) ----
    print("\nPlot 3: Term structure & smile cross-sections...")
    plot_cross_sections(
        futures_list[0], samples_list[0],
        f"{output_dir}/cross_sections.png",
        future_len=config.future_len,
    )

    # ---- Plot 4: Vol smile at multiple tenors ----
    print("\nPlot 4: Vol smile by tenor...")
    plot_cross_sections_multi_tenor(
        futures_list[0], samples_list[0],
        f"{output_dir}/vol_smiles_by_tenor.png",
        future_len=config.future_len,
    )

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
