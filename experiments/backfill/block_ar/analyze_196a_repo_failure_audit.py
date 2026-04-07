#!/usr/bin/env python
"""
Repo-level failure audit across the current best one-shot and AR branches.

Focus:
  - 183c best strict anchor
  - 169c best clean AR baseline
  - 194c best 2-state quiet/event AR regime model
  - 195a best localized marked-event AR model

The audit normalizes suite outcomes, extracts high-signal metrics, and writes a
compact JSON artifact for downstream memo generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path("/home/max/Documents/vol-surface-vae-pub")


MODELS = {
    "183c_best": {
        "summary": ROOT / "results/block_ar/183c_best_v2_s3mrjspec_full_30d/summary.json",
        "family": "one_shot_transport",
        "role": "strict_overall_anchor",
    },
    "169c_best": {
        "summary": ROOT / "results/block_ar/169c_best_v2_s3mrjspec_full_30d/summary.json",
        "family": "ar_student_t",
        "role": "clean_ar_baseline",
    },
    "194c_best": {
        "summary": ROOT / "results/block_ar/194c_best_v2_s3mrjspec_full_30d/summary.json",
        "family": "ar_regime_switching",
        "role": "two_state_quiet_event_ar",
    },
    "195a_best": {
        "summary": ROOT / "results/block_ar/195a_best_v2_s3mrjspec_full_30d/summary.json",
        "family": "ar_localized_event",
        "role": "localized_marked_event_ar",
    },
}


MECHANISM = {
    "183c_best": ROOT / "results/validations/2026-04-05/analysis/183c_best_mechanistic/mechanistic_summary.json",
    "194c_best": ROOT / "results/validations/2026-04-07/analysis/194c_two_state_mechanistic/mechanistic_summary.json",
    "194c_split": ROOT / "results/validations/2026-04-07/analysis/194c_split_generalization/summary.json",
    "195a_feasibility": ROOT / "results/validations/2026-04-07/analysis/195a_localized_event_feasibility/summary.json",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def suite_passes(summary: dict[str, Any]) -> dict[str, bool]:
    return {
        "S1": bool(summary["surface"]["overall_pass"]),
        "S2": bool(summary["coverage"]["overall_pass"]),
        "S3": bool(summary["conditionality"]["overall_pass"]),
        "S4": bool(summary["time_series"]["overall_pass"]),
        "S5": bool(summary["block_ar"]["overall_pass"]),
        "S6": bool(summary["cointegration"]["overall_pass"]),
        "S7": bool(summary["regime_coverage"]["overall_pass"]),
        "S8": bool(summary["distributional"]["overall_pass"]),
        "S9": bool(summary["cross_cell_correlation"]["overall_pass"]),
        "S10": bool(summary["mean_reversion"]["overall_pass"]),
        "S11": bool(summary["pathwise_jump_realism"]["overall_pass"]),
    }


def extract_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "coverage_90": float(summary["coverage"]["overall"]["0.9"]),
        "calibration_error": float(summary["coverage"]["calibration_error"]),
        "turb_calm_ratio": float(summary["conditionality"]["turb_calm_ratio"]),
        "worst_cell_width_ratio": float(summary["conditionality"]["worst_cell_width_ratio"]),
        "kurtosis_ratio": float(summary["time_series"]["kurtosis"]["kurtosis_ratio"]),
        "quiet_ratio": float(summary["time_series"]["exceedance_spectrum"]["quiet_mass"]["ratio"]),
        "shoulder_ratio": float(summary["time_series"]["exceedance_spectrum"]["shoulder_mass"]["ratio"]),
        "extreme_ratio": float(summary["time_series"]["exceedance_spectrum"]["extreme_mass"]["ratio"]),
        "regime_layer2_pass": int(summary["regime_coverage"]["layer2_n_passing"]),
        "regime_layer2_total": int(summary["regime_coverage"]["layer2_n_total"]),
        "catastrophic_rate": float(summary["regime_coverage"]["layer3_catastrophic_rate"]),
        "dist_frac_pass": int(summary["distributional"]["median_bias"]["n_pass"]),
        "dist_mag_pass": int(summary["distributional"]["median_bias"]["n_mag_pass"]),
        "corr_ratio": float(summary["cross_cell_correlation"]["corr_ratio"]),
        "rank_ratio": float(summary["cross_cell_correlation"]["rank_ratio"]),
        "mr_ratio": float(summary["mean_reversion"]["mr_gt_ratio"]),
        "mr_full_pass": bool(summary["mean_reversion"]["full_horizon"]["overall_pass"]),
        "jump_ks": float(summary["pathwise_jump_realism"]["pathwise_max_jump"]["ks_stat"]),
    }


def main() -> None:
    out_dir = ROOT / "results/validations/2026-04-07/analysis/196_design"
    out_dir.mkdir(parents=True, exist_ok=True)

    models: dict[str, Any] = {}
    suite_matrix: dict[str, list[str]] = {f"S{i}": [] for i in range(1, 12)}
    fail_matrix: dict[str, list[str]] = {f"S{i}": [] for i in range(1, 12)}

    for name, cfg in MODELS.items():
        summary = load_json(cfg["summary"])
        passes = suite_passes(summary)
        metrics = extract_metrics(summary)
        models[name] = {
            "family": cfg["family"],
            "role": cfg["role"],
            "suite_passes": passes,
            "pass_count": int(sum(int(v) for v in passes.values())),
            "metrics": metrics,
        }
        for suite, ok in passes.items():
            (suite_matrix if ok else fail_matrix)[suite].append(name)

    shared_passes = [suite for suite, names in suite_matrix.items() if len(names) == len(MODELS)]
    shared_failures = [suite for suite, names in fail_matrix.items() if len(names) == len(MODELS)]

    conflict_notes = {
        "shared_passes": shared_passes,
        "shared_failures": shared_failures,
        "tradeoff_suites": [suite for suite in suite_matrix if suite not in shared_passes and suite not in shared_failures],
        "mechanism_refs": {k: str(v.relative_to(ROOT)) for k, v in MECHANISM.items()},
    }

    out = {
        "models": models,
        "shared_structure": conflict_notes,
    }

    out_path = out_dir / "196a_repo_failure_audit.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))
    print(f"\nSaved repo failure audit to {out_path}")


if __name__ == "__main__":
    main()
