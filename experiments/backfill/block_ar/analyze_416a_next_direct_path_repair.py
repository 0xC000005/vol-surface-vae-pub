#!/usr/bin/env python
"""416a: select the next repair after the failed 392a/413a mixture."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


RUNS = {
    "339a_axial_old": Path("results/block_ar/339a_v0_s42/full11.json"),
    "339b_transformer_old": Path("results/block_ar/339b_v0_s42/full11.json"),
    "413a_axial_recent": Path("results/block_ar/413a_recent_score_path_fm_s42/full11.json"),
    "415a_mix": Path("results/block_ar/415a_392a_413a_mix0125/full11.json"),
}
OUT_DIR = Path("results/block_ar/416a_next_direct_path_repair")


def load(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "score": int(result["summary"]["n_pass"]),
        "failed": list(result["summary"]["failed_suites"]),
        "conditionality": float(result["conditionality"]["mae_reduction_pct"]),
        "time_series_pass": bool(result["time_series"]["overall_pass"]),
        "kurtosis_ratio": float(result["time_series"]["kurtosis"]["kurtosis_ratio"]),
        "cointegration_worst": float(result["cointegration"]["worst_cell_ratio"]),
        "daily_ks": int(result["distributional_fidelity"]["ks_test"]["n_pass"]),
        "level_ks": int(result["distributional_fidelity"]["ks_level_test"]["n_pass"]),
        "median_fraction": int(result["distributional_fidelity"]["median_bias"]["n_pass"]),
        "bias_magnitude": int(result["distributional_fidelity"]["median_bias"]["n_mag_pass"]),
        "corr_ratio": float(result["cross_cell_correlation"]["corr_ratio"]),
        "rank_ratio": float(result["cross_cell_correlation"]["rank_ratio"]),
        "mean_reversion_pass": bool(result["mean_reversion"]["overall_pass"]),
        "pathwise_ks": float(result["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"]),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = {name: summarize(load(path)) for name, path in RUNS.items()}
    payload = {
        "runs": rows,
        "mechanism_read": (
            "The mixture diagnostic should be closed, but 413a should not be discarded "
            "as mere failure. Recent quantile framing plus direct path FM solved level "
            "occupancy. The missing piece is structural coupling. The old 339b "
            "Transformer path mixer had better cointegration and cross-cell structure "
            "than old 339a, but it never received the recent-score framing that made "
            "413a's level KS jump to 20/25."
        ),
        "decision": (
            "Run one final direct-path repair: recent-quantile adaptation of the 339b "
            "Transformer path-flow checkpoint. This is not a depth/head sweep; it is the "
            "single missing cross of two known mechanisms: 339b's stronger joint mixer "
            "and 413a's recent score framing. If it fails to beat the frontier or at "
            "least preserve distributional fidelity while improving structural suites, "
            "close direct path flow."
        ),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# 416a Next Direct-Path Repair",
        "",
        "| run | score | failed | cond | coint worst | level KS | corr ratio | MR | path KS |",
        "|---|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for name, row in rows.items():
        lines.append(
            f"| {name} | {row['score']}/11 | {', '.join(row['failed'])} | "
            f"{row['conditionality']:.2f}% | {row['cointegration_worst']:.3f} | "
            f"{row['level_ks']}/25 | {row['corr_ratio']:.3f} | "
            f"{row['mean_reversion_pass']} | {row['pathwise_ks']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Mechanism Read",
            "",
            payload["mechanism_read"],
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
