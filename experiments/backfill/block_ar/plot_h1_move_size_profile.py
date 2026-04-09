#!/usr/bin/env python
"""
Plot management-facing H=1 move-size profile from a smoke summary JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot H=1 move-size profile")
    parser.add_argument("--summary", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--title", type=str, default="H=1 Move-Size Profile")
    args = parser.parse_args()

    summary_path = Path(args.summary)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_path) as f:
        payload = json.load(f)

    msm = payload["move_size_metrics"]
    labels = [
        "<= 0.005",
        "<= 0.010",
        "<= 0.020",
        "<= 0.050",
    ]
    keys = [
        "very_small_lte_0p005",
        "small_lte_0p010",
        "moderate_lte_0p020",
        "large_lte_0p050",
    ]
    gt = np.array([msm[k]["gt_share"] for k in keys], dtype=float)
    sample = np.array([msm[k]["sample_share"] for k in keys], dtype=float)

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(x - width / 2, gt, width=width, label="GT", color="#2b6cb0")
    ax.bar(x + width / 2, sample, width=width, label="Generated", color="#d97706")

    for i, (g, s) in enumerate(zip(gt, sample)):
        ax.text(i - width / 2, g + 0.01, f"{g:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(i + width / 2, s + 0.01, f"{s:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Share Of Moves")
    ax.set_xlabel("|delta| threshold")
    ax.set_title(args.title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
