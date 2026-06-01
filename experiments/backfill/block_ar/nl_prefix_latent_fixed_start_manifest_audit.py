#!/usr/bin/env python
"""Audit fixed-start narrative conditionality across a cached manifest grid."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_fixed_start_shape_audit import (  # noqa: E402
    _bootstrap_pairs,
    _load_json,
    _max_start_difference,
    _operational_variant_index,
    _pairwise,
    _ratio,
    _summarize_pair_rows,
    _write_json,
    pair_distribution_metrics,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    scale_delta_samples_around_mean,
)


DEFAULT_OBSERVED_ROOT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_control_suite_865a_full_narrative_s192"
)
DEFAULT_REPEAT_ROOT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_control_suite_865b_full_s192_repeat"
)
DEFAULT_START_ONLY_ROOT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_control_suite_862d_full_start_only_s96"
)
DEFAULT_VARIANT_DIR = "decoder_soft_topk_narrative_start_checked_gen_temp_0p50"
DEFAULT_START_ONLY_VARIANT_DIR = "decoder_soft_topk_start_only_gen_temp_0p50"
CASE_RE = re.compile(r"^(?P<narrative>.+)_start(?P<start>\d+)$")


def parse_case_slug(slug: str) -> tuple[str, int]:
    match = CASE_RE.match(str(slug))
    if not match:
        raise ValueError(f"Cannot parse narrative/start slug: {slug}")
    return str(match.group("narrative")), int(match.group("start"))


def _load_case_dir(
    *,
    case_dir: Path,
    case_name: str,
    narrative: str,
    start_index: int,
    label: str,
    fan_scale: float,
    seed: int | None = None,
) -> dict[str, Any]:
    report = _load_json(case_dir / "prefix_latent_story_smoke_report.json")
    arrays = np.load(case_dir / "prefix_latent_story_smoke_arrays.npz")
    generated = np.asarray(arrays["generated_states"], dtype=np.float32)
    requested_raw = np.asarray(arrays["requested_raw"], dtype=np.float32)
    variant_idx = _operational_variant_index(report, generated.shape[0])
    states = np.asarray(generated[variant_idx], dtype=np.float32)
    start = np.asarray(requested_raw[variant_idx], dtype=np.float32)
    if abs(float(fan_scale) - 1.0) > 1e-8:
        deltas = (states - start[None, None, :]).astype(np.float32)
        scaled = scale_delta_samples_around_mean(deltas[None, ...], float(fan_scale))
        states = (start[None, None, :] + scaled[0]).astype(np.float32)
    operational = _operational_validation_case(report)
    return {
        "case_name": str(case_name),
        "narrative": str(narrative),
        "start_index": int(start_index),
        "seed": None if seed is None else int(seed),
        "label": str(label),
        "color": "#1565C0",
        "states": states,
        "start": start,
        "sample_count": int(states.shape[0]),
        "run_report": str(case_dir / "prefix_latent_story_smoke_report.json"),
        "validation_operational_status": operational.get("status", ""),
        "validation_operational_warnings": operational.get("warnings", []),
        "validation_operational_failures": operational.get("failures", []),
        "validation_start_distance_z": operational.get("start_distance_z"),
    }


def _operational_validation_case(report: dict[str, Any]) -> dict[str, Any]:
    gate = report.get("validation_gate", {})
    if not isinstance(gate, dict):
        return {}
    for row in gate.get("cases", []):
        if isinstance(row, dict) and bool(row.get("is_operational")):
            return row
    return {}


def discover_observed_cases(
    observed_root: Path,
    *,
    variant_dir: str,
    fan_scale: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case_root in sorted(Path(observed_root).iterdir()):
        if not case_root.is_dir():
            continue
        try:
            narrative, start_index = parse_case_slug(case_root.name)
        except ValueError:
            continue
        case_dir = case_root / f"fixed_start_{start_index}" / variant_dir
        if not (case_dir / "prefix_latent_story_smoke_report.json").exists():
            continue
        rows.append(
            _load_case_dir(
                case_dir=case_dir,
                case_name=case_root.name,
                narrative=narrative,
                start_index=start_index,
                label=narrative.replace("_", " "),
                fan_scale=fan_scale,
            )
        )
    return rows


def discover_start_only_cases(
    start_only_root: Path,
    *,
    variant_dir: str,
    fan_scale: float,
) -> list[dict[str, Any]]:
    if not Path(start_only_root).exists():
        return []
    return discover_observed_cases(
        Path(start_only_root),
        variant_dir=variant_dir,
        fan_scale=fan_scale,
    )


def _split_roots(root_arg: str | Path) -> list[Path]:
    return [
        Path(item.strip())
        for item in str(root_arg).split(",")
        if item.strip()
    ]


def discover_repeat_cases(
    repeat_root: str | Path,
    *,
    variant_dir: str,
    fan_scale: float,
) -> list[dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    roots = _split_roots(repeat_root)
    if len(roots) > 1:
        for seed_index, root in enumerate(roots):
            for case in discover_observed_cases(
                root,
                variant_dir=variant_dir,
                fan_scale=fan_scale,
            ):
                seed = seed_index + 1
                case["seed"] = int(seed)
                case["case_name"] = f"{case['case_name']}#seed_{seed}"
                case["label"] = f"{case['label']} seed {seed}"
                all_rows.append(case)
        return all_rows
    base = roots[0] if roots else Path(repeat_root)
    if (base / "repeat_controls").exists():
        base = base / "repeat_controls"
    rows: list[dict[str, Any]] = []
    for case_root in sorted(base.iterdir()):
        if not case_root.is_dir():
            continue
        try:
            narrative, start_index = parse_case_slug(case_root.name)
        except ValueError:
            continue
        fixed_dir = case_root / f"fixed_start_{start_index}"
        for seed_dir in sorted(fixed_dir.glob("seed_*")):
            if not seed_dir.is_dir():
                continue
            try:
                seed = int(seed_dir.name.replace("seed_", ""))
            except ValueError:
                continue
            if not (seed_dir / "prefix_latent_story_smoke_report.json").exists():
                continue
            rows.append(
                _load_case_dir(
                    case_dir=seed_dir,
                    case_name=f"{case_root.name}#seed_{seed}",
                    narrative=narrative,
                    start_index=start_index,
                    label=f"{narrative.replace('_', ' ')} seed {seed}",
                    fan_scale=fan_scale,
                    seed=seed,
                )
            )
    return rows


def _by_start(cases: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[int(case["start_index"])].append(case)
    return dict(sorted(grouped.items()))


def _repeat_pairwise(repeat_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for case in repeat_cases:
        grouped[(str(case["narrative"]), int(case["start_index"]))].append(case)
    rows: list[dict[str, Any]] = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        rows.extend(_pairwise(group, "same_narrative_repeat"))
    return rows


def _observed_pairwise_by_start(
    observed_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start_index, group in _by_start(observed_cases).items():
        for left, right in combinations(group, 2):
            row = pair_distribution_metrics(left, right)
            row["control"] = "observed_narrative"
            row["start_index"] = int(start_index)
            rows.append(row)
    return rows


def _case_half(case: dict[str, Any], half_index: int) -> dict[str, Any] | None:
    states = np.asarray(case["states"], dtype=np.float32)
    split = states.shape[0] // 2
    if split < 2:
        return None
    if int(half_index) == 0:
        subset = states[:split]
        suffix = "left"
    else:
        subset = states[split:]
        suffix = "right"
    return {
        **case,
        "case_name": f"{case['case_name']}#{suffix}",
        "label": f"{case['label']} {suffix}",
        "states": subset,
        "sample_count": int(subset.shape[0]),
    }


def _sample_matched_observed_pairwise_by_start(
    observed_cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for start_index, group in _by_start(observed_cases).items():
        for left, right in combinations(group, 2):
            for left_half in [0, 1]:
                for right_half in [0, 1]:
                    left_case = _case_half(left, left_half)
                    right_case = _case_half(right, right_half)
                    if left_case is None or right_case is None:
                        continue
                    row = pair_distribution_metrics(left_case, right_case)
                    row["control"] = "observed_narrative_sample_matched"
                    row["start_index"] = int(start_index)
                    row["left_half"] = int(left_half)
                    row["right_half"] = int(right_half)
                    rows.append(row)
    return rows


def _per_start_summaries(
    observed_cases: list[dict[str, Any]],
    repeat_cases: list[dict[str, Any]],
    start_only_cases: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    repeat_by_start = _by_start(repeat_cases)
    start_only_by_start = _by_start(start_only_cases)
    summaries: dict[str, dict[str, Any]] = {}
    for start_index, observed_group in _by_start(observed_cases).items():
        observed_rows = _pairwise(observed_group, "observed_narrative")
        repeat_rows = _repeat_pairwise(repeat_by_start.get(start_index, []))
        bootstrap_rows = _bootstrap_pairs(observed_group)
        start_only_rows = _pairwise(
            start_only_by_start.get(start_index, []),
            "start_only_null",
        )
        summaries[str(start_index)] = {
            "observed_narrative": _summarize_pair_rows(observed_rows),
            "same_narrative_repeat": _summarize_pair_rows(repeat_rows),
            "within_run_bootstrap": _summarize_pair_rows(bootstrap_rows),
            "start_only_null": _summarize_pair_rows(start_only_rows),
            "max_start_abs_diff": _max_start_difference(observed_group),
            "observed_case_count": int(len(observed_group)),
            "repeat_case_count": int(len(repeat_by_start.get(start_index, []))),
            "start_only_case_count": int(len(start_only_by_start.get(start_index, []))),
        }
    return summaries


def _safe_median(values: list[float]) -> float | None:
    finite = [float(value) for value in values if np.isfinite(float(value))]
    return float(median(finite)) if finite else None


def build_manifest_audit(
    observed_cases: list[dict[str, Any]],
    repeat_cases: list[dict[str, Any]],
    start_only_cases: list[dict[str, Any]] | None = None,
    *,
    max_start_abs_diff: float,
    max_start_only_ratio: float,
    max_repeat_ratio: float,
    max_bootstrap_ratio: float,
    enforce_start_reliability: bool = False,
) -> dict[str, Any]:
    start_only_cases = list(start_only_cases or [])
    observed = _observed_pairwise_by_start(observed_cases)
    observed_sample_matched = _sample_matched_observed_pairwise_by_start(
        observed_cases
    )
    repeat = _repeat_pairwise(repeat_cases)
    bootstrap = _bootstrap_pairs(observed_cases)
    start_only = _observed_pairwise_by_start(start_only_cases)
    for row in start_only:
        row["control"] = "start_only_null"
    summaries = {
        "observed_narrative": _summarize_pair_rows(observed),
        "observed_narrative_sample_matched": _summarize_pair_rows(
            observed_sample_matched
        ),
        "same_narrative_repeat": _summarize_pair_rows(repeat),
        "within_run_bootstrap": _summarize_pair_rows(bootstrap),
        "start_only_null": _summarize_pair_rows(start_only),
    }
    ratios = {
        "repeat_to_observed_path_wasserstein": _ratio(
            summaries["same_narrative_repeat"],
            summaries["observed_narrative"],
            "path_wasserstein_z_mean_median",
        ),
        "bootstrap_to_observed_path_wasserstein": _ratio(
            summaries["within_run_bootstrap"],
            summaries["observed_narrative"],
            "path_wasserstein_z_mean_median",
        ),
        "repeat_to_observed_path_variance": _ratio(
            summaries["same_narrative_repeat"],
            summaries["observed_narrative"],
            "path_std_log_ratio_mean_median",
        ),
        "bootstrap_to_observed_path_variance": _ratio(
            summaries["within_run_bootstrap"],
            summaries["observed_narrative"],
            "path_std_log_ratio_mean_median",
        ),
        "repeat_to_observed_path_energy": _ratio(
            summaries["same_narrative_repeat"],
            summaries["observed_narrative"],
            "path_energy_distance",
        ),
        "bootstrap_to_observed_path_energy": _ratio(
            summaries["within_run_bootstrap"],
            summaries["observed_narrative"],
            "path_energy_distance",
        ),
        "start_only_to_observed_path_wasserstein": _ratio(
            summaries["start_only_null"],
            summaries["observed_narrative"],
            "path_wasserstein_z_mean_median",
        ),
        "start_only_to_observed_path_variance": _ratio(
            summaries["start_only_null"],
            summaries["observed_narrative"],
            "path_std_log_ratio_mean_median",
        ),
        "start_only_to_observed_path_energy": _ratio(
            summaries["start_only_null"],
            summaries["observed_narrative"],
            "path_energy_distance",
        ),
        "bootstrap_to_sample_matched_observed_path_wasserstein": _ratio(
            summaries["within_run_bootstrap"],
            summaries["observed_narrative_sample_matched"],
            "path_wasserstein_z_mean_median",
        ),
        "bootstrap_to_sample_matched_observed_path_variance": _ratio(
            summaries["within_run_bootstrap"],
            summaries["observed_narrative_sample_matched"],
            "path_std_log_ratio_mean_median",
        ),
        "bootstrap_to_sample_matched_observed_path_energy": _ratio(
            summaries["within_run_bootstrap"],
            summaries["observed_narrative_sample_matched"],
            "path_energy_distance",
        ),
    }
    per_start = _per_start_summaries(observed_cases, repeat_cases, start_only_cases)
    max_start_diff = max(
        [float(row["max_start_abs_diff"]) for row in per_start.values()] or [0.0]
    )
    per_start_observed_path_energy = [
        float(row["observed_narrative"].get("path_energy_distance_median_across_pairs"))
        for row in per_start.values()
        if row["observed_narrative"].get("path_energy_distance_median_across_pairs")
        is not None
    ]
    failures: list[str] = []
    warnings: list[str] = []
    if not start_only:
        warnings.append("start_only_null_absent")
    if max_start_diff > float(max_start_abs_diff):
        failures.append("per_start_fixed_level_not_identical")
    start_reliability = _start_reliability_summary(observed_cases)
    start_reliability_warnings: list[str] = []
    if bool(enforce_start_reliability):
        warning_count = int(
            start_reliability.get("status_counts", {}).get("warning", 0)
        )
        fail_count = int(start_reliability.get("status_counts", {}).get("fail", 0))
        if fail_count:
            failures.append("observed_operational_start_failures")
        if warning_count:
            start_reliability_warnings.append("observed_operational_start_warnings")
    for key, name in [
        ("repeat_to_observed_path_wasserstein", "repeat_path_too_close_to_observed"),
        ("repeat_to_observed_path_variance", "repeat_variance_too_close_to_observed"),
        ("repeat_to_observed_path_energy", "repeat_energy_too_close_to_observed"),
    ]:
        if ratios[key] is not None and float(ratios[key]) > float(max_repeat_ratio):
            failures.append(name)
    for key, name in [
        (
            "start_only_to_observed_path_wasserstein",
            "start_only_path_too_close_to_observed",
        ),
        (
            "start_only_to_observed_path_variance",
            "start_only_variance_too_close_to_observed",
        ),
        (
            "start_only_to_observed_path_energy",
            "start_only_energy_too_close_to_observed",
        ),
    ]:
        if ratios[key] is not None and float(ratios[key]) > float(max_start_only_ratio):
            failures.append(name)
    for key, name in [
        (
            "bootstrap_to_observed_path_wasserstein",
            "bootstrap_path_noise_close_to_observed",
        ),
        (
            "bootstrap_to_observed_path_variance",
            "bootstrap_variance_noise_close_to_observed",
        ),
        (
            "bootstrap_to_observed_path_energy",
            "bootstrap_energy_noise_close_to_observed",
        ),
    ]:
        if ratios[key] is not None and float(ratios[key]) > float(max_bootstrap_ratio):
            warnings.append(name)
    status = "fail" if failures else "warning" if warnings else "pass"
    promotion_warnings = list(warnings) + start_reliability_warnings
    promotion_status = (
        "fail" if failures else "warning" if promotion_warnings else "pass"
    )
    return {
        "status": status,
        "promotion_status": promotion_status,
        "scope_note": (
            "Broad cached fixed-start manifest audit. Cross-narrative pairs are "
            "formed only within the same selected historical start; same-narrative "
            "repeat seeds and within-run bootstraps are controls. This artifact "
            "does not include start-only null controls, so it cannot by itself "
            "promote a production conditionality claim."
        ),
        "case_counts": {
            "observed_case_count": int(len(observed_cases)),
            "repeat_case_count": int(len(repeat_cases)),
            "start_only_case_count": int(len(start_only_cases)),
            "start_count": int(len(_by_start(observed_cases))),
            "narrative_count": int(
                len({str(case["narrative"]) for case in observed_cases})
            ),
            "observed_pair_count": int(len(observed)),
            "observed_sample_matched_pair_count": int(len(observed_sample_matched)),
            "repeat_pair_count": int(len(repeat)),
            "bootstrap_pair_count": int(len(bootstrap)),
            "start_only_pair_count": int(len(start_only)),
        },
        "thresholds": {
            "max_start_abs_diff": float(max_start_abs_diff),
            "max_start_only_ratio": float(max_start_only_ratio),
            "max_repeat_ratio": float(max_repeat_ratio),
            "max_bootstrap_ratio": float(max_bootstrap_ratio),
            "enforce_start_reliability": bool(enforce_start_reliability),
        },
        "start_reliability": start_reliability,
        "max_per_start_abs_diff": float(max_start_diff),
        "per_start_observed_path_energy_median": _safe_median(
            per_start_observed_path_energy
        ),
        "summaries": summaries,
        "ratios": ratios,
        "per_start_summaries": per_start,
        "warnings": warnings,
        "promotion_warnings": promotion_warnings,
        "failures": failures,
        "observed_pairwise": observed,
        "observed_sample_matched_pairwise": observed_sample_matched,
        "repeat_pairwise": repeat,
        "bootstrap_pairwise": bootstrap,
        "start_only_pairwise": start_only,
        "interpretation": [
            "A low repeat-to-observed ratio means narrative changes are larger than seed noise.",
            "A bootstrap warning means within-run sampling noise is large relative to narrative separation.",
            "Sample-matched observed ratios compare bootstrap halves against narrative pairs at the same half-sample size; they diagnose finite-sample readout bias and do not by themselves change promotion status.",
            "A low start-only ratio means fixed-start compatibility alone does not reproduce the narrative-conditioned path differences.",
            "This is a generalization audit for conditionality, not a realized-future backtest.",
        ],
    }


def _start_reliability_summary(cases: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    warning_counts: dict[str, int] = {}
    failure_counts: dict[str, int] = {}
    distances: list[float] = []
    rows: list[dict[str, Any]] = []
    for case in cases:
        status = str(case.get("validation_operational_status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
        for warning in case.get("validation_operational_warnings") or []:
            warning = str(warning)
            warning_counts[warning] = warning_counts.get(warning, 0) + 1
        for failure in case.get("validation_operational_failures") or []:
            failure = str(failure)
            failure_counts[failure] = failure_counts.get(failure, 0) + 1
        value = case.get("validation_start_distance_z")
        if value is not None:
            distances.append(float(value))
        rows.append(
            {
                "case_name": case.get("case_name"),
                "start_index": case.get("start_index"),
                "status": status,
                "warnings": list(case.get("validation_operational_warnings") or []),
                "failures": list(case.get("validation_operational_failures") or []),
                "start_distance_z": value,
            }
        )
    return {
        "status_counts": status_counts,
        "warning_counts": warning_counts,
        "failure_counts": failure_counts,
        "start_distance_z_median": _safe_median(distances),
        "start_distance_z_max": max(distances) if distances else None,
        "case_rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-root", default=str(DEFAULT_OBSERVED_ROOT))
    parser.add_argument("--repeat-root", default=str(DEFAULT_REPEAT_ROOT))
    parser.add_argument("--start-only-root", default=str(DEFAULT_START_ONLY_ROOT))
    parser.add_argument("--variant-dir", default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--start-only-variant-dir", default=DEFAULT_START_ONLY_VARIANT_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fan-scale", type=float, default=1.0)
    parser.add_argument("--max-start-abs-diff", type=float, default=1e-5)
    parser.add_argument("--max-start-only-ratio", type=float, default=0.50)
    parser.add_argument("--max-repeat-ratio", type=float, default=0.75)
    parser.add_argument("--max-bootstrap-ratio", type=float, default=0.75)
    parser.add_argument("--enforce-start-reliability", action="store_true")
    args = parser.parse_args()

    observed_cases = discover_observed_cases(
        Path(args.observed_root),
        variant_dir=str(args.variant_dir),
        fan_scale=float(args.fan_scale),
    )
    repeat_cases = discover_repeat_cases(
        str(args.repeat_root),
        variant_dir=str(args.variant_dir),
        fan_scale=float(args.fan_scale),
    )
    start_only_cases = discover_start_only_cases(
        Path(args.start_only_root),
        variant_dir=str(args.start_only_variant_dir),
        fan_scale=float(args.fan_scale),
    )
    report = build_manifest_audit(
        observed_cases,
        repeat_cases,
        start_only_cases,
        max_start_abs_diff=float(args.max_start_abs_diff),
        max_start_only_ratio=float(args.max_start_only_ratio),
        max_repeat_ratio=float(args.max_repeat_ratio),
        max_bootstrap_ratio=float(args.max_bootstrap_ratio),
        enforce_start_reliability=bool(args.enforce_start_reliability),
    )
    output_dir = Path(args.output_dir)
    report["observed_root"] = str(args.observed_root)
    report["repeat_root"] = str(args.repeat_root)
    report["start_only_root"] = str(args.start_only_root)
    report["variant_dir"] = str(args.variant_dir)
    report["start_only_variant_dir"] = str(args.start_only_variant_dir)
    report["fan_scale"] = float(args.fan_scale)
    report["artifact_paths"] = {
        "report": str(output_dir / "fixed_start_manifest_audit.json")
    }
    _write_json(report["artifact_paths"]["report"], report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "promotion_status": report["promotion_status"],
                "report": report["artifact_paths"]["report"],
                "case_counts": report["case_counts"],
                "ratios": report["ratios"],
                "warnings": report["warnings"],
                "promotion_warnings": report["promotion_warnings"],
                "failures": report["failures"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
