#!/usr/bin/env python
"""
Plot unconditional one-day per-cell marginals under different target transforms:
  1. delta normalized by per-cell q99(|delta|)
  2. simple return = delta / previous_level
  3. log return = log(next_level) - log(previous_level)
  4. log return normalized by per-cell max(|log return|)
  5. log return normalized by per-cell q99(|log return|)
  6. log return normalized by per-cell MAD(log return)

This is intended to audit the normalization assumptions in the minimal
deterministic H=1 direct-delta models.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _pooled_stats(x: np.ndarray) -> dict[str, float]:
    flat = x.reshape(-1)
    abs_flat = np.abs(flat)
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "median": float(np.median(flat)),
        "abs_q50": float(np.quantile(abs_flat, 0.50)),
        "abs_q90": float(np.quantile(abs_flat, 0.90)),
        "abs_q95": float(np.quantile(abs_flat, 0.95)),
        "abs_q99": float(np.quantile(abs_flat, 0.99)),
        "abs_q995": float(np.quantile(abs_flat, 0.995)),
        "abs_max": float(np.max(abs_flat)),
    }


def _cell_stats(x: np.ndarray) -> list[list[dict[str, float]]]:
    out: list[list[dict[str, float]]] = []
    for r in range(5):
        row = []
        for c in range(5):
            cell = x[:, r, c]
            abs_cell = np.abs(cell)
            row.append({
                "mean": float(np.mean(cell)),
                "std": float(np.std(cell)),
                "median": float(np.median(cell)),
                "abs_q90": float(np.quantile(abs_cell, 0.90)),
                "abs_q95": float(np.quantile(abs_cell, 0.95)),
                "abs_q99": float(np.quantile(abs_cell, 0.99)),
                "abs_max": float(np.max(abs_cell)),
            })
        out.append(row)
    return out


def _plot_grid(
    x: np.ndarray,
    title: str,
    out_path: Path,
    bins: int,
    clip_quantile: float,
) -> dict[str, float]:
    pooled_abs = np.abs(x.reshape(-1))
    xlim = float(np.quantile(pooled_abs, clip_quantile))
    xlim = max(xlim, 1e-6)

    fig, axes = plt.subplots(5, 5, figsize=(16, 14), sharex=True, sharey=True)
    edges = np.linspace(-xlim, xlim, bins + 1)

    for r in range(5):
        for c in range(5):
            ax = axes[r, c]
            cell = x[:, r, c]
            ax.hist(cell, bins=edges, density=True, histtype="step", linewidth=1.1, color="#1f77b4")
            ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.6)
            ax.set_title(f"({r},{c})", fontsize=9)
            ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=15)
    fig.text(0.5, 0.03, f"value (clipped to pooled |x| q{int(clip_quantile * 1000) / 10:.1f})", ha="center")
    fig.text(0.02, 0.5, "density", va="center", rotation="vertical")
    fig.tight_layout(rect=[0.03, 0.05, 1.0, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return {"plot_abs_clip_quantile": clip_quantile, "plot_abs_xlim": xlim}


def _plot_pooled_overlay(
    transforms: dict[str, np.ndarray],
    out_path: Path,
    bins: int,
    clip_quantile: float,
) -> dict[str, dict[str, float]]:
    clip_info: dict[str, dict[str, float]] = {}
    n = len(transforms)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.5))
    if n == 1:
        axes = [axes]
    colors = {
        "normalized_delta": "#1f77b4",
        "simple_return": "#d62728",
        "log_return": "#2ca02c",
        "normalized_log_return_max": "#9467bd",
        "normalized_log_return_q99": "#8c564b",
        "normalized_log_return_mad": "#e377c2",
    }
    labels = {
        "normalized_delta": "delta / q99_cell",
        "simple_return": "simple return",
        "log_return": "log return",
        "normalized_log_return_max": "log return / max|log return|_cell",
        "normalized_log_return_q99": "log return / q99|log return|_cell",
        "normalized_log_return_mad": "log return / MAD_cell",
    }

    for ax, (name, values) in zip(axes, transforms.items()):
        flat = values.reshape(-1)
        abs_flat = np.abs(flat)
        xlim = float(np.quantile(abs_flat, clip_quantile))
        xlim = max(xlim, 1e-6)
        edges = np.linspace(-xlim, xlim, bins + 1)
        ax.hist(flat, bins=edges, density=True, histtype="step", linewidth=1.6, color=colors[name])
        ax.axvline(0.0, color="black", linewidth=0.7, alpha=0.6)
        ax.set_title(labels[name])
        ax.set_xlabel("value")
        clip_info[name] = {"plot_abs_clip_quantile": clip_quantile, "plot_abs_xlim": xlim}

    axes[0].set_ylabel("density")
    fig.suptitle("Pooled unconditional one-day marginals by transform", fontsize=14)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    return clip_info


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze unconditional one-day marginals under different transforms")
    parser.add_argument("--data_path", type=str, default="data/vol_surface_with_ret.npz")
    parser.add_argument("--test_start", type=int, default=4511)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--bins", type=int, default=120)
    parser.add_argument("--clip_quantile", type=float, default=0.995)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.data_path)
    surface = data["surface"].astype(np.float64)
    prev = surface[: args.test_start - 1]
    nxt = surface[1 : args.test_start]
    delta = nxt - prev

    flat_delta = delta.reshape(-1, 25)
    delta_scale = np.quantile(np.abs(flat_delta), 0.99, axis=0)
    delta_scale = np.clip(delta_scale, 1e-3, None).reshape(1, 5, 5)

    prev_safe = np.clip(prev, args.eps, None)
    nxt_safe = np.clip(nxt, args.eps, None)

    normalized_delta = delta / delta_scale
    simple_return = delta / prev_safe
    log_return = np.log(nxt_safe) - np.log(prev_safe)
    log_return_scale_max = np.max(np.abs(log_return.reshape(-1, 25)), axis=0)
    log_return_scale_max = np.clip(log_return_scale_max, 1e-6, None).reshape(1, 5, 5)
    normalized_log_return_max = log_return / log_return_scale_max
    log_return_scale_q99 = np.quantile(np.abs(log_return.reshape(-1, 25)), 0.99, axis=0)
    log_return_scale_q99 = np.clip(log_return_scale_q99, 1e-6, None).reshape(1, 5, 5)
    log_return_flat = log_return.reshape(-1, 25)
    log_return_median = np.median(log_return_flat, axis=0, keepdims=True)
    log_return_scale_mad = np.median(np.abs(log_return_flat - log_return_median), axis=0)
    log_return_scale_mad = np.clip(log_return_scale_mad, 1e-6, None).reshape(1, 5, 5)
    normalized_log_return_q99 = log_return / log_return_scale_q99
    normalized_log_return_mad = log_return / log_return_scale_mad

    transforms = {
        "normalized_delta": normalized_delta,
        "simple_return": simple_return,
        "log_return": log_return,
        "normalized_log_return_max": normalized_log_return_max,
        "normalized_log_return_q99": normalized_log_return_q99,
        "normalized_log_return_mad": normalized_log_return_mad,
    }

    overlay_info = _plot_pooled_overlay(
        transforms=transforms,
        out_path=out_dir / "pooled_transform_overlay.png",
        bins=args.bins,
        clip_quantile=args.clip_quantile,
    )

    summary: dict[str, object] = {
        "data_path": args.data_path,
        "test_start": args.test_start,
        "n_days_used": int(prev.shape[0]),
        "eps": args.eps,
        "bins": args.bins,
        "clip_quantile": args.clip_quantile,
        "delta_scale_stats": {
            "min": float(delta_scale.min()),
            "median": float(np.median(delta_scale)),
            "max": float(delta_scale.max()),
        },
        "log_return_scale_max_stats": {
            "min": float(log_return_scale_max.min()),
            "median": float(np.median(log_return_scale_max)),
            "max": float(log_return_scale_max.max()),
        },
        "log_return_scale_q99_stats": {
            "min": float(log_return_scale_q99.min()),
            "median": float(np.median(log_return_scale_q99)),
            "max": float(log_return_scale_q99.max()),
        },
        "log_return_scale_mad_stats": {
            "min": float(log_return_scale_mad.min()),
            "median": float(np.median(log_return_scale_mad)),
            "max": float(log_return_scale_mad.max()),
        },
        "transforms": {},
    }

    titles = {
        "normalized_delta": "Per-cell unconditional marginals: delta / q99_cell",
        "simple_return": "Per-cell unconditional marginals: simple return",
        "log_return": "Per-cell unconditional marginals: log return",
        "normalized_log_return_max": "Per-cell unconditional marginals: log return / max|log return|_cell",
        "normalized_log_return_q99": "Per-cell unconditional marginals: log return / q99|log return|_cell",
        "normalized_log_return_mad": "Per-cell unconditional marginals: log return / MAD_cell",
    }

    for name, values in transforms.items():
        grid_info = _plot_grid(
            x=values,
            title=titles[name],
            out_path=out_dir / f"{name}_grid.png",
            bins=args.bins,
            clip_quantile=args.clip_quantile,
        )
        summary["transforms"][name] = {
            "pooled_stats": _pooled_stats(values),
            "cell_stats": _cell_stats(values),
            "plot_info": {
                "grid": grid_info,
                "overlay": overlay_info[name],
            },
        }

    md_lines = [
        "# Transform Marginal Audit",
        "",
        f"- Data: `{args.data_path}`",
        f"- Pre-test days used: `{prev.shape[0]}`",
        f"- `delta_scale` is per-cell q99(|delta|) over `surface[:{args.test_start}]`",
        f"- Plot clipping uses pooled `|x|` q{int(args.clip_quantile * 1000) / 10:.1f} for readability",
        "",
        "## Pooled Stats",
        "",
        "| Transform | mean | std | abs q50 | abs q90 | abs q95 | abs q99 | abs max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, info in summary["transforms"].items():
        stats = info["pooled_stats"]  # type: ignore[index]
        md_lines.append(
            f"| `{name}` | {stats['mean']:.5f} | {stats['std']:.5f} | {stats['abs_q50']:.5f} | "
            f"{stats['abs_q90']:.5f} | {stats['abs_q95']:.5f} | {stats['abs_q99']:.5f} | {stats['abs_max']:.5f} |"
        )

    (out_dir / "transform_marginal_audit.md").write_text("\n".join(md_lines))
    (out_dir / "transform_marginal_audit.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({
        "output_dir": str(out_dir),
        "delta_scale_median": summary["delta_scale_stats"]["median"],
        "normalized_delta_abs_q99": summary["transforms"]["normalized_delta"]["pooled_stats"]["abs_q99"],
        "simple_return_abs_q99": summary["transforms"]["simple_return"]["pooled_stats"]["abs_q99"],
        "log_return_abs_q99": summary["transforms"]["log_return"]["pooled_stats"]["abs_q99"],
    }, indent=2))


if __name__ == "__main__":
    main()
