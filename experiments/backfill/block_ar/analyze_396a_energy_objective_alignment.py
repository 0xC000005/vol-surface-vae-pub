#!/usr/bin/env python
"""396a: compare path-energy holdout objective to official 11-suite behavior."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RUNS = {
    "391a_w020_s2": {
        "model_dir": Path("models/backfill/391a_recent_rollout_energy_w02_s42"),
        "result": Path("results/block_ar/391a_recent_rollout_energy_w02_s42/full11.json"),
    },
    "392a_w005_s2": {
        "model_dir": Path("models/backfill/392a_recent_rollout_energy_w005_s42"),
        "result": Path("results/block_ar/392a_recent_rollout_energy_w005_s42/full11.json"),
    },
    "393a_w010_s2": {
        "model_dir": Path("models/backfill/393a_recent_rollout_energy_w01_s42"),
        "result": Path("results/block_ar/393a_recent_rollout_energy_w01_s42/full11.json"),
    },
    "395a_w005_s4": {
        "model_dir": Path("models/backfill/395a_recent_rollout_energy_w005_s4_s42"),
        "result": Path("results/block_ar/395a_recent_rollout_energy_w005_s4_s42/full11.json"),
    },
}
OUT_DIR = Path("results/block_ar/396a_energy_objective_alignment")


def load_json(path: Path) -> dict[str, Any] | list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def best_epoch(history: list[dict[str, Any]]) -> dict[str, Any]:
    return min(history, key=lambda row: row["val_total"])


def summarize_run(name: str, meta: dict[str, Path]) -> dict[str, Any]:
    args = load_json(meta["model_dir"] / "args.json")
    history = load_json(meta["model_dir"] / "training_history.json")
    if not isinstance(history, list):
        raise TypeError("training_history.json must contain a list")
    best = best_epoch(history)
    result = load_json(meta["result"])
    if not isinstance(result, dict):
        raise TypeError("full11.json must contain an object")
    dist = result["distributional_fidelity"]
    return {
        "name": name,
        "energy_weight": args["energy_weight"],
        "train_sample_count": args["train_sample_count"],
        "best_epoch": best["epoch"],
        "val_total": best["val_total"],
        "val_fm_loss": best["val_fm_loss"],
        "val_energy": best["val_energy"],
        "val_energy_target_dist": best["val_energy_target_dist"],
        "val_energy_pair_dist": best["val_energy_pair_dist"],
        "val_sample_score_std": best["val_sample_score_std"],
        "val_target_score_std": best["val_target_score_std"],
        "val_std_gap": best["val_sample_score_std"] - best["val_target_score_std"],
        "val_sample_h1_std": best["val_sample_h1_std"],
        "val_sample_h30_std": best["val_sample_h30_std"],
        "score": result["summary"]["n_pass"],
        "failed": result["summary"]["failed_suites"],
        "coverage_90": result["coverage"]["overall"]["0.9"],
        "conditional_mae_reduction": result["conditionality"]["mae_reduction_pct"],
        "cointegration_worst": result["cointegration"]["worst_cell_ratio"],
        "level_ks_pass": dist["ks_level_test"]["n_pass"],
        "level_ks_median": dist["ks_level_test"]["median_stat"],
        "regime_layer2_pass": result["regime_coverage"]["layer2_n_passing"],
    }


def rank(rows: list[dict[str, Any]], field: str, reverse: bool = False) -> list[str]:
    return [
        row["name"]
        for row in sorted(rows, key=lambda row: row[field], reverse=reverse)
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [summarize_run(name, meta) for name, meta in RUNS.items()]
    rows_by_name = {row["name"]: row for row in rows}
    payload = {
        "runs": rows_by_name,
        "rankings": {
            "best_internal_val_total_lowest": rank(rows, "val_total"),
            "best_internal_energy_lowest": rank(rows, "val_energy"),
            "best_official_score_highest": rank(rows, "score", reverse=True),
            "best_conditionality_highest": rank(rows, "conditional_mae_reduction", reverse=True),
            "best_level_ks_highest": rank(rows, "level_ks_pass", reverse=True),
            "best_cointegration_worst_highest": rank(rows, "cointegration_worst", reverse=True),
        },
        "key_falsifier": (
            "395a has the lowest internal validation total and lowest validation energy, "
            "and its sample standard deviation is closest to the target standard deviation, "
            "but it has the worst official score and fails conditionality plus worst-cell "
            "cointegration. The holdout energy objective is therefore not aligned enough "
            "with the target 11-suite."
        ),
        "decision": (
            "Close the path-energy fine-tune family as a primary route. The next move "
            "should be research ideation for a target-aligned but still clean objective: "
            "directly train or select for conditional calibration/occupancy constraints "
            "instead of adding more architecture."
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# 396a Energy Objective Alignment Audit",
        "",
        "| run | w | samples | val total | val energy | std gap | score | failed | cov90 | cond MAE | level KS | coint worst |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: item["name"]):
        lines.append(
            f"| {row['name']} | {row['energy_weight']:.2f} | "
            f"{row['train_sample_count']} | {row['val_total']:.4f} | "
            f"{row['val_energy']:.4f} | {row['val_std_gap']:.3f} | "
            f"{row['score']}/11 | {', '.join(row['failed'])} | "
            f"{row['coverage_90']:.3f} | {row['conditional_mae_reduction']:.2f}% | "
            f"{row['level_ks_pass']}/25 | {row['cointegration_worst']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Key Falsifier",
            "",
            payload["key_falsifier"],
            "",
            "## Decision",
            "",
            payload["decision"],
            "",
        ]
    )
    (OUT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
