#!/usr/bin/env python
"""Fixed-start control suite for narrative conditionality.

This script is a diagnostic control, not a new product path. It asks whether
the promoted fixed-start narrative gaps are larger than no-narrative start-only
support and same-narrative seed-repeat noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_fixed_start_narrative_contrast import (  # noqa: E402
    DEFAULT_MARKETS,
    build_fixed_start_contrast,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    KEY_FACTOR_NAMES,
)
from experiments.backfill.block_ar.nl_prefix_latent_analogue_mixture_prior import (  # noqa: E402
    load_state_spec_names,
)
from experiments.backfill.block_ar.nl_prefix_latent_start_conditioned_bakeoff import (  # noqa: E402
    row_from_report,
    selected_historical_cases,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    run_prefix_latent_story_smoke,
)
from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (  # noqa: E402
    build_prefix_latent_run_args,
)


DEFAULT_CASE_SPEC = Path(
    "docs/research_protocols/nl_prefix_latent_promoted_specs/"
    "fixed_start_narrative_matrix_858b_cases.json"
)
DEFAULT_OBSERVED_BAKEOFF = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_narrative_matrix_858b/start_conditioned_bakeoff.json"
)
DEFAULT_OBSERVED_CONTRAST = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_narrative_contrast_858b/"
    "fixed_start_narrative_contrast.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_fixed_start_control_suite_862a"
)

REPEAT_VARIANT = {
    "variant_name": "decoder_soft_topk_narrative_start_checked_gen_temp_0p50",
    "memory_prior_mode": "soft_topk_narrative_start_checked",
    "prefix_prior_mode": "decoder",
    "top_k": 8,
    "temperature": 0.2,
    "generator_temperature": 0.5,
}


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.quantile(arr, float(q)))


def _gap_summary_from_pairwise(pairwise_rows: list[dict[str, Any]]) -> dict[str, Any]:
    gaps = [_as_float(row.get("standardized_l2_gap")) for row in pairwise_rows]
    by_start: dict[str, list[float]] = {}
    for row in pairwise_rows:
        by_start.setdefault(str(row.get("start_name", "")), []).append(
            _as_float(row.get("standardized_l2_gap"))
        )
    start_rows = []
    for start_name, start_gaps in sorted(by_start.items()):
        start_rows.append(
            {
                "start_name": start_name,
                "pair_count": int(len(start_gaps)),
                "median_gap": float(median(start_gaps)) if start_gaps else 0.0,
                "mean_gap": float(np.mean(start_gaps)) if start_gaps else 0.0,
                "p90_gap": _quantile(start_gaps, 0.90),
                "max_gap": max(start_gaps) if start_gaps else 0.0,
            }
        )
    return {
        "pair_count": int(len(gaps)),
        "overall_median_gap": float(median(gaps)) if gaps else 0.0,
        "overall_mean_gap": float(np.mean(gaps)) if gaps else 0.0,
        "overall_p90_gap": _quantile(gaps, 0.90),
        "overall_max_gap": max(gaps) if gaps else 0.0,
        "by_start": start_rows,
    }


def _summary_std(summary: dict[str, Any]) -> float:
    if summary.get("std") is not None:
        return _as_float(summary.get("std"))
    width = _as_float(summary.get("p90")) - _as_float(summary.get("p10"))
    return width / (2.0 * 1.281551565545)


def _terminal_summary_by_market(report: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows = report.get("generation", {}).get("terminal_delta_summary", [])
    if not isinstance(rows, list):
        raise ValueError("report generation.terminal_delta_summary is missing")
    output: dict[str, dict[str, float]] = {}
    for row in rows:
        row = _as_dict(row)
        market = str(row.get("market", "")).upper()
        if not market:
            continue
        summary = {
            "mean": _as_float(row.get("mean_terminal_delta")),
            "p10": _as_float(row.get("p10")),
            "p50": _as_float(row.get("p50", row.get("mean_terminal_delta"))),
            "p90": _as_float(row.get("p90")),
        }
        summary["std"] = _summary_std(summary)
        output[market] = summary
    return output


def _standardized_terminal_gap(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
    *,
    markets: list[str],
) -> tuple[float, list[dict[str, Any]]]:
    squared = 0.0
    market_rows: list[dict[str, Any]] = []
    for market in markets:
        if market not in left or market not in right:
            continue
        mean_delta = float(left[market]["mean"] - right[market]["mean"])
        pooled = float(
            np.sqrt((_summary_std(left[market]) ** 2 + _summary_std(right[market]) ** 2) / 2.0)
        )
        standardized = mean_delta / pooled if pooled > 1e-12 else 0.0
        squared += standardized * standardized
        market_rows.append(
            {
                "market": market,
                "left_mean_minus_right_mean": mean_delta,
                "pooled_std": pooled,
                "standardized_mean_gap": standardized,
            }
        )
    return float(np.sqrt(squared)), sorted(
        market_rows,
        key=lambda row: abs(float(row["standardized_mean_gap"])),
        reverse=True,
    )[:5]


def run_repeat_controls(
    *,
    cases: list[dict[str, Any]],
    seeds: list[int],
    output_dir: Path,
    samples: int,
    steps: int,
    chunk_size: int,
    device: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for case in cases:
        for seed in seeds:
            case_dir = (
                output_dir
                / str(case["case_name"])
                / str(case["start_name"])
                / f"seed_{int(seed)}"
            )
            run_args = build_prefix_latent_run_args(
                start_mode="explicit_start_window",
                samples=int(samples),
                condition_report=str(case["condition_report"]),
                explicit_start_window_index=int(case["candidate_index"]),
                output_dir=str(case_dir),
            )
            run_args.steps = int(steps)
            run_args.chunk_size = int(chunk_size)
            run_args.device = str(device)
            run_args.seed = int(seed)
            run_args.memory_prior_mode = str(REPEAT_VARIANT["memory_prior_mode"])
            run_args.memory_prior_top_k = int(REPEAT_VARIANT["top_k"])
            run_args.memory_prior_temperature = float(REPEAT_VARIANT["temperature"])
            run_args.prefix_prior_mode = str(REPEAT_VARIANT["prefix_prior_mode"])
            run_args.temperature = float(REPEAT_VARIANT["generator_temperature"])
            report = run_prefix_latent_story_smoke(run_args)
            row = row_from_report(case=case, variant=REPEAT_VARIANT, report=report)
            row["repeat_seed"] = int(seed)
            rows.append(row)
    output = {
        "status": "pass" if rows else "fail",
        "scope_note": (
            "Same-narrative fixed-start repeat controls. Repeats change only the "
            "random seed, so resulting gaps estimate sampling/training noise."
        ),
        "seed_count": int(len(seeds)),
        "case_count": int(len(cases)),
        "run_count": int(len(rows)),
        "seeds": [int(seed) for seed in seeds],
        "rows": rows,
        "artifact_paths": {
            "report": str(output_dir / "fixed_start_repeat_controls.json")
        },
    }
    _write_json(output["artifact_paths"]["report"], output)
    return output


def repeat_gap_report(
    repeat_report: dict[str, Any],
    *,
    markets: list[str],
) -> dict[str, Any]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in _as_list(repeat_report.get("rows")):
        row = _as_dict(row)
        key = (
            str(row.get("case_name", "")),
            str(row.get("start_name", "")),
            int(row.get("candidate_index", -1)),
        )
        groups.setdefault(key, []).append(row)
    pairwise: list[dict[str, Any]] = []
    for key, group in sorted(groups.items()):
        if len(group) < 2:
            continue
        loaded = []
        for row in group:
            report = _load_json(str(row.get("run_report", "")))
            loaded.append((row, _terminal_summary_by_market(report)))
        for (left_row, left_summary), (right_row, right_summary) in combinations(
            loaded, 2
        ):
            gap, market_gaps = _standardized_terminal_gap(
                left_summary,
                right_summary,
                markets=markets,
            )
            pairwise.append(
                {
                    "case_name": key[0],
                    "start_name": key[1],
                    "candidate_index": key[2],
                    "left_seed": int(left_row.get("repeat_seed", -1)),
                    "right_seed": int(right_row.get("repeat_seed", -1)),
                    "standardized_l2_gap": gap,
                    "largest_abs_market_gaps": market_gaps,
                }
            )
    return {
        "status": "pass" if pairwise else "warning",
        "scope_note": (
            "Same-narrative repeat gap report. These gaps should be materially "
            "smaller than different-narrative fixed-start gaps."
        ),
        "pairwise_contrasts": sorted(
            pairwise,
            key=lambda row: float(row.get("standardized_l2_gap", 0.0)),
            reverse=True,
        ),
        "gap_summary": _gap_summary_from_pairwise(pairwise),
    }


def _operational_sample_index(report: dict[str, Any]) -> int:
    rows = report.get("variant_rows", [])
    if not isinstance(rows, list):
        return 0
    for idx, row in enumerate(rows):
        if isinstance(row, dict) and bool(row.get("is_operational")):
            return int(idx)
    return 0


def _terminal_sample_values_by_market(
    *,
    run_report: str | Path,
    markets: list[str],
    spec_names: list[str],
) -> dict[str, np.ndarray]:
    report = _load_json(run_report)
    arrays_path = Path(run_report).with_name("prefix_latent_story_smoke_arrays.npz")
    arrays = np.load(arrays_path)
    samples = np.asarray(arrays["samples"], dtype=np.float32)
    if samples.ndim != 4:
        raise ValueError(f"{arrays_path}: expected samples with shape [K,S,T,C]")
    op_idx = min(_operational_sample_index(report), samples.shape[0] - 1)
    terminal = samples[op_idx, :, -1, :]
    index = {name: col for col, name in enumerate(spec_names)}
    output: dict[str, np.ndarray] = {}
    for market in markets:
        if market == "IV_SURFACE":
            output[market] = np.nanmean(terminal[:, :25], axis=1).astype(np.float32)
            continue
        spec_name = KEY_FACTOR_NAMES.get(market)
        if spec_name is None:
            continue
        col = index.get(spec_name)
        if col is None:
            continue
        output[market] = terminal[:, col].astype(np.float32)
    return output


def _sample_summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "mean": float(np.mean(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    }


def bootstrap_gap_report(
    bakeoff_report: dict[str, Any],
    *,
    markets: list[str],
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
) -> dict[str, Any]:
    """Estimate within-run sampling noise from saved generated samples."""

    spec_names = load_state_spec_names(checkpoint)
    pairwise: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for row in _as_list(bakeoff_report.get("rows")):
        row = _as_dict(row)
        run_report = str(row.get("run_report", ""))
        try:
            samples_by_market = _terminal_sample_values_by_market(
                run_report=run_report,
                markets=markets,
                spec_names=spec_names,
            )
        except (FileNotFoundError, KeyError, ValueError) as exc:
            skipped.append(
                {
                    "case_name": row.get("case_name"),
                    "start_name": row.get("start_name"),
                    "reason": str(exc),
                }
            )
            continue
        if not samples_by_market:
            continue
        sample_count = min(values.shape[0] for values in samples_by_market.values())
        if sample_count < 4:
            skipped.append(
                {
                    "case_name": row.get("case_name"),
                    "start_name": row.get("start_name"),
                    "reason": f"sample_count_too_small:{sample_count}",
                }
            )
            continue
        split = sample_count // 2
        left = {
            market: _sample_summary(values[:split])
            for market, values in samples_by_market.items()
        }
        right = {
            market: _sample_summary(values[split:sample_count])
            for market, values in samples_by_market.items()
        }
        gap, market_gaps = _standardized_terminal_gap(left, right, markets=markets)
        pairwise.append(
            {
                "case_name": str(row.get("case_name", "")),
                "start_name": str(row.get("start_name", "")),
                "candidate_index": int(row.get("candidate_index", -1)),
                "sample_count": int(sample_count),
                "left_sample_count": int(split),
                "right_sample_count": int(sample_count - split),
                "standardized_l2_gap": gap,
                "largest_abs_market_gaps": market_gaps,
            }
        )
    return {
        "status": "pass" if pairwise else "warning",
        "scope_note": (
            "Within-run bootstrap control. It splits saved generated paths from "
            "the same narrative/start/run to estimate Monte Carlo fan-chart noise."
        ),
        "pairwise_contrasts": sorted(
            pairwise,
            key=lambda item: float(item.get("standardized_l2_gap", 0.0)),
            reverse=True,
        ),
        "gap_summary": _gap_summary_from_pairwise(pairwise),
        "skipped": skipped,
    }


def _control_contrast(
    *,
    bakeoff_report_path: str | Path,
    markets: list[str],
    output_dir: Path,
    label: str,
) -> dict[str, Any]:
    report = build_fixed_start_contrast(
        _load_json(bakeoff_report_path),
        markets=markets,
    )
    output = output_dir / f"{label}_fixed_start_contrast.json"
    _write_json(output, report)
    report["artifact_path"] = str(output)
    report["gap_summary"] = _gap_summary_from_pairwise(
        [_as_dict(row) for row in _as_list(report.get("pairwise_contrasts"))]
    )
    _write_json(output, report)
    return report


def _ratio(numerator: float, denominator: float) -> float | None:
    if abs(float(denominator)) < 1e-12:
        return None
    return float(numerator) / float(denominator)


def _by_start(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("start_name", "")): _as_dict(row)
        for row in _as_list(summary.get("by_start"))
    }


def _per_start_controls(
    *,
    observed_summary: dict[str, Any],
    start_only_summary: dict[str, Any] | None,
    bootstrap_summary: dict[str, Any] | None,
    repeat_summary: dict[str, Any] | None,
    max_start_only_ratio: float,
    max_bootstrap_ratio: float,
    max_repeat_ratio: float,
) -> list[dict[str, Any]]:
    observed_by_start = _by_start(observed_summary)
    start_only_by_start = _by_start(start_only_summary or {})
    bootstrap_by_start = _by_start(bootstrap_summary or {})
    repeat_by_start = _by_start(repeat_summary or {})
    rows: list[dict[str, Any]] = []
    for start_name, observed in sorted(observed_by_start.items()):
        observed_gap = _as_float(observed.get("median_gap"))
        start_only_gap = _as_float(
            start_only_by_start.get(start_name, {}).get("median_gap")
        )
        bootstrap_gap = _as_float(
            bootstrap_by_start.get(start_name, {}).get("median_gap")
        )
        repeat_gap = _as_float(repeat_by_start.get(start_name, {}).get("median_gap"))
        start_only_ratio = _ratio(start_only_gap, observed_gap)
        bootstrap_ratio = (
            None if start_name not in bootstrap_by_start else _ratio(bootstrap_gap, observed_gap)
        )
        repeat_ratio = (
            None if start_name not in repeat_by_start else _ratio(repeat_gap, observed_gap)
        )
        failures: list[str] = []
        warnings: list[str] = []
        if (
            start_only_ratio is not None
            and start_only_ratio > float(max_start_only_ratio)
        ):
            failures.append("start_only_gap_too_close")
        if (
            bootstrap_ratio is not None
            and bootstrap_ratio > float(max_bootstrap_ratio)
        ):
            warnings.append("bootstrap_noise_close")
        if repeat_ratio is not None and repeat_ratio > float(max_repeat_ratio):
            failures.append("repeat_noise_close")
        rows.append(
            {
                "start_name": start_name,
                "status": "fail" if failures else "warning" if warnings else "pass",
                "observed_median_gap": observed_gap,
                "start_only_median_gap": start_only_gap,
                "bootstrap_median_gap": bootstrap_gap if bootstrap_ratio is not None else None,
                "repeat_median_gap": repeat_gap if repeat_ratio is not None else None,
                "start_only_ratio": start_only_ratio,
                "bootstrap_ratio": bootstrap_ratio,
                "repeat_ratio": repeat_ratio,
                "warnings": warnings,
                "failures": failures,
            }
        )
    return rows


def build_control_suite(
    *,
    observed_contrast: dict[str, Any],
    start_only_contrast: dict[str, Any] | None,
    bootstrap_report: dict[str, Any] | None,
    repeat_report: dict[str, Any] | None,
    max_start_only_ratio: float,
    max_bootstrap_ratio: float,
    max_repeat_ratio: float,
) -> dict[str, Any]:
    observed_summary = _gap_summary_from_pairwise(
        [_as_dict(row) for row in _as_list(observed_contrast.get("pairwise_contrasts"))]
    )
    observed_gap = float(observed_summary["overall_median_gap"])
    controls: dict[str, Any] = {}
    failures: list[str] = []
    warnings: list[str] = []

    if start_only_contrast is None:
        warnings.append("start_only_control_missing")
    else:
        start_only_summary = _as_dict(start_only_contrast.get("gap_summary"))
        start_only_gap = _as_float(start_only_summary.get("overall_median_gap"))
        start_only_ratio = _ratio(start_only_gap, observed_gap)
        controls["start_only"] = {
            "gap_summary": start_only_summary,
            "ratio_to_observed_median_gap": start_only_ratio,
            "artifact_path": start_only_contrast.get("artifact_path"),
        }
        if start_only_ratio is not None and start_only_ratio > max_start_only_ratio:
            failures.append("start_only_gap_too_close_to_narrative_gap")

    if repeat_report is None:
        warnings.append("same_narrative_repeat_control_missing")
    else:
        repeat_summary = _as_dict(repeat_report.get("gap_summary"))
        repeat_gap = _as_float(repeat_summary.get("overall_median_gap"))
        repeat_ratio = _ratio(repeat_gap, observed_gap)
        controls["same_narrative_repeat"] = {
            "gap_summary": repeat_summary,
            "ratio_to_observed_median_gap": repeat_ratio,
            "artifact_path": repeat_report.get("artifact_path"),
        }
        if repeat_ratio is not None and repeat_ratio > max_repeat_ratio:
            failures.append("repeat_noise_gap_too_close_to_narrative_gap")
        elif _as_float(repeat_summary.get("overall_p90_gap")) >= observed_gap:
            warnings.append("repeat_noise_p90_overlaps_observed_median_gap")

    if bootstrap_report is None:
        warnings.append("within_run_bootstrap_control_missing")
    else:
        bootstrap_summary = _as_dict(bootstrap_report.get("gap_summary"))
        bootstrap_gap = _as_float(bootstrap_summary.get("overall_median_gap"))
        bootstrap_ratio = _ratio(bootstrap_gap, observed_gap)
        controls["within_run_bootstrap"] = {
            "gap_summary": bootstrap_summary,
            "ratio_to_observed_median_gap": bootstrap_ratio,
            "artifact_path": bootstrap_report.get("artifact_path"),
            "skipped": bootstrap_report.get("skipped", []),
        }
        if bootstrap_ratio is not None and bootstrap_ratio > max_bootstrap_ratio:
            warnings.append("bootstrap_noise_gap_close_to_narrative_gap")

    per_start = _per_start_controls(
        observed_summary=observed_summary,
        start_only_summary=_as_dict(controls.get("start_only", {})).get("gap_summary"),
        bootstrap_summary=_as_dict(controls.get("within_run_bootstrap", {})).get(
            "gap_summary"
        ),
        repeat_summary=_as_dict(controls.get("same_narrative_repeat", {})).get(
            "gap_summary"
        ),
        max_start_only_ratio=float(max_start_only_ratio),
        max_bootstrap_ratio=float(max_bootstrap_ratio),
        max_repeat_ratio=float(max_repeat_ratio),
    )
    if any(_as_list(row.get("failures")) for row in per_start):
        failures.append("per_start_control_failure")
    if any(_as_list(row.get("warnings")) for row in per_start):
        warnings.append("per_start_control_warning")

    status = "fail" if failures else "warning" if warnings else "pass"
    interpretation = [
        (
            "The observed gap is the distributional gap across different "
            "narratives while the starting level is fixed."
        ),
        (
            "The start-only control removes narrative ranking. If its gap is "
            "small, then the fixed-start narrative gap is not just start geometry."
        ),
        (
            "The same-narrative repeat control changes only seed. If repeat gaps "
            "are small, then narrative gaps are not just sampling noise."
        ),
    ]
    return {
        "status": status,
        "scope_note": (
            "Fixed-start narrative control suite for the production support prior. "
            "This is a promotion guardrail, not a product-time model variant."
        ),
        "thresholds": {
            "max_start_only_ratio": float(max_start_only_ratio),
            "max_bootstrap_ratio": float(max_bootstrap_ratio),
            "max_repeat_ratio": float(max_repeat_ratio),
        },
        "observed": {
            "gap_summary": observed_summary,
            "artifact_path": observed_contrast.get("artifact_path"),
        },
        "controls": controls,
        "per_start_controls": per_start,
        "warnings": warnings,
        "failures": failures,
        "interpretation": interpretation,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fixed-Start Narrative Control Suite",
        "",
        str(report.get("scope_note", "")),
        "",
        f"- Status: `{report.get('status')}`",
        f"- Failures: `{', '.join(report.get('failures', [])) or 'none'}`",
        f"- Warnings: `{', '.join(report.get('warnings', [])) or 'none'}`",
        "",
        "## Interpretation",
        "",
    ]
    for item in _as_list(report.get("interpretation")):
        lines.append(f"- {item}")
    observed = _as_dict(report.get("observed")).get("gap_summary", {})
    lines.extend(
        [
            "",
            "## Gap Summary",
            "",
            "| Source | Pairs | Median Gap | Mean Gap | P90 Gap | Max Gap | Ratio to Observed |",
            "|---|---:|---:|---:|---:|---:|---:|",
            _summary_table_row("observed_narrative", observed, 1.0),
        ]
    )
    for name, control in _as_dict(report.get("controls")).items():
        control = _as_dict(control)
        lines.append(
            _summary_table_row(
                str(name),
                _as_dict(control.get("gap_summary")),
                control.get("ratio_to_observed_median_gap"),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Start Control Status",
            "",
            "| Start | Status | Observed Median | Start-Only Ratio | Bootstrap Ratio | Repeat Ratio | Warnings | Failures |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in _as_list(report.get("per_start_controls")):
        row = _as_dict(row)
        lines.append(
            f"| `{row.get('start_name')}` | `{row.get('status')}` | "
            f"`{_as_float(row.get('observed_median_gap')):.3f}` | "
            f"`{_format_ratio(row.get('start_only_ratio'))}` | "
            f"`{_format_ratio(row.get('bootstrap_ratio'))}` | "
            f"`{_format_ratio(row.get('repeat_ratio'))}` | "
            f"`{', '.join(str(x) for x in _as_list(row.get('warnings'))) or 'none'}` | "
            f"`{', '.join(str(x) for x in _as_list(row.get('failures'))) or 'none'}` |"
        )
    return "\n".join(lines) + "\n"


def _format_ratio(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.3f}"


def _summary_table_row(name: str, summary: dict[str, Any], ratio: Any) -> str:
    return (
        f"| `{name}` | "
        f"`{int(summary.get('pair_count', 0) or 0)}` | "
        f"`{_as_float(summary.get('overall_median_gap')):.3f}` | "
        f"`{_as_float(summary.get('overall_mean_gap')):.3f}` | "
        f"`{_as_float(summary.get('overall_p90_gap')):.3f}` | "
        f"`{_as_float(summary.get('overall_max_gap')):.3f}` | "
        f"`{'' if ratio is None else f'{float(ratio):.3f}'}` |"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observed-contrast", type=Path, default=DEFAULT_OBSERVED_CONTRAST)
    parser.add_argument("--observed-bakeoff-report", type=Path, default=DEFAULT_OBSERVED_BAKEOFF)
    parser.add_argument("--case-spec-json", type=Path, default=DEFAULT_CASE_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-only-bakeoff-report", type=Path)
    parser.add_argument("--repeat-report", type=Path)
    parser.add_argument("--run-repeat-controls", action="store_true")
    parser.add_argument("--repeat-case-count", type=int, default=4)
    parser.add_argument("--repeat-seeds", default="791,792")
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-start-only-ratio", type=float, default=0.50)
    parser.add_argument("--max-bootstrap-ratio", type=float, default=0.75)
    parser.add_argument("--max-repeat-ratio", type=float, default=0.75)
    parser.add_argument(
        "--markets",
        default=",".join(DEFAULT_MARKETS),
        help="Comma-separated market names to compare.",
    )
    args = parser.parse_args()

    markets = [
        item.strip().upper() for item in str(args.markets).split(",") if item.strip()
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observed = _load_json(args.observed_contrast)
    observed["artifact_path"] = str(args.observed_contrast)
    bootstrap = bootstrap_gap_report(
        _load_json(args.observed_bakeoff_report),
        markets=markets,
    )
    bootstrap["artifact_path"] = str(args.output_dir / "within_run_bootstrap_gap_report.json")
    _write_json(bootstrap["artifact_path"], bootstrap)

    start_only_contrast = None
    if args.start_only_bakeoff_report:
        start_only_contrast = _control_contrast(
            bakeoff_report_path=args.start_only_bakeoff_report,
            markets=markets,
            output_dir=args.output_dir,
            label="start_only",
        )

    repeat_gap = None
    if args.run_repeat_controls:
        seeds = [
            int(item.strip())
            for item in str(args.repeat_seeds).split(",")
            if item.strip()
        ]
        cases = selected_historical_cases(
            int(args.repeat_case_count),
            case_spec_json=args.case_spec_json,
        )
        repeat = run_repeat_controls(
            cases=cases,
            seeds=seeds,
            output_dir=args.output_dir / "repeat_controls",
            samples=int(args.samples),
            steps=int(args.steps),
            chunk_size=int(args.chunk_size),
            device=str(args.device),
        )
        repeat_gap = repeat_gap_report(repeat, markets=markets)
        repeat_gap["artifact_path"] = str(
            args.output_dir / "same_narrative_repeat_gap_report.json"
        )
        _write_json(repeat_gap["artifact_path"], repeat_gap)
    elif args.repeat_report:
        repeat_gap = repeat_gap_report(_load_json(args.repeat_report), markets=markets)
        repeat_gap["artifact_path"] = str(args.repeat_report)

    report = build_control_suite(
        observed_contrast=observed,
        start_only_contrast=start_only_contrast,
        bootstrap_report=bootstrap,
        repeat_report=repeat_gap,
        max_start_only_ratio=float(args.max_start_only_ratio),
        max_bootstrap_ratio=float(args.max_bootstrap_ratio),
        max_repeat_ratio=float(args.max_repeat_ratio),
    )
    report["inputs"] = {
        "observed_contrast": str(args.observed_contrast),
        "observed_bakeoff_report": str(args.observed_bakeoff_report),
        "case_spec_json": str(args.case_spec_json),
        "start_only_bakeoff_report": (
            str(args.start_only_bakeoff_report)
            if args.start_only_bakeoff_report
            else ""
        ),
        "repeat_report": str(args.repeat_report) if args.repeat_report else "",
        "repeat_case_count": int(args.repeat_case_count),
        "repeat_seeds": str(args.repeat_seeds),
    }
    report["artifact_paths"] = {
        "report": str(args.output_dir / "fixed_start_control_suite.json"),
        "markdown": str(args.output_dir / "fixed_start_control_suite.md"),
    }
    _write_json(report["artifact_paths"]["report"], report)
    Path(report["artifact_paths"]["markdown"]).write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "warnings": report["warnings"],
                "failures": report["failures"],
                "report": report["artifact_paths"]["report"],
                "markdown": report["artifact_paths"]["markdown"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
