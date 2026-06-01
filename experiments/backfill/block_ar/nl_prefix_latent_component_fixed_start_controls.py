#!/usr/bin/env python
"""Component-mixture fixed-start null and repeat controls."""

from __future__ import annotations

import argparse
import json
import re
import sys
from itertools import combinations
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    run_prefix_latent_story_smoke,
)
from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (  # noqa: E402
    build_prefix_latent_run_args,
)
from experiments.backfill.block_ar.plot_narrative_casebook_backtest import (  # noqa: E402
    CONTRAST_CASES,
    MARKET_INDEX,
)


DEFAULT_ROOT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_component_mixture_fixed_start_900a_s96"
)
DEFAULT_VARIANT_DIR = "decoder_component_topk_narrative_start_checked_gen_temp_0p50"
DEFAULT_MEMORY_PRIOR_MODE = "diverse_topk_narrative_start_checked"
DEFAULT_CONDITION_REPORTS = {
    "fragile_risk_on_start18": (
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_809a_fragile/condition_only_report.json"
    ),
    "defensive_risk_off_start18": (
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_809b_defensive/condition_only_report.json"
    ),
    "rates_selloff_start18": (
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_809c_rates/condition_only_report.json"
    ),
    "commodity_inflation_start18": (
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_822a_commodity/condition_only_report.json"
    ),
    "dollar_liquidity_start18": (
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_823a_dollar/condition_only_report.json"
    ),
    "safe_haven_gold_start18": (
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "prefix_latent_condition_only_report_823b_safe_haven/condition_only_report.json"
    ),
}
MARKETS = [
    ("SPX", MARKET_INDEX["SPX"]),
    ("Crude oil", 31),
    ("BBB OAS", 35),
    ("VIX", MARKET_INDEX["VIX"]),
    ("Gold", 37),
    ("1Y ATM IV", MARKET_INDEX["IV_ATM_1Y"]),
]


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


def _case_dir(root: Path, case_name: str, variant_dir: str) -> Path:
    return root / case_name / f"fixed_start_{_case_start_index(case_name)}" / variant_dir


def _case_start_index(case_name: str) -> int:
    match = re.search(r"_start(\d+)(?:#.*)?$", str(case_name))
    return int(match.group(1)) if match else 18


def _base_condition_key(case_name: str) -> str:
    return re.sub(r"_start\d+(?:#.*)?$", "_start18", str(case_name))


def _condition_report_for_case_name(
    case_name: str, condition_report_root: str | Path | None = None
) -> str:
    key = str(case_name).split("#seed_")[0]
    if condition_report_root:
        root = Path(condition_report_root)
        for candidate_key in (key, _base_condition_key(key)):
            candidate = root / candidate_key / "condition_only_report.json"
            if candidate.exists():
                return str(candidate)
    if key in DEFAULT_CONDITION_REPORTS:
        return DEFAULT_CONDITION_REPORTS[key]
    base_key = _base_condition_key(key)
    if base_key in DEFAULT_CONDITION_REPORTS:
        return DEFAULT_CONDITION_REPORTS[base_key]
    raise KeyError(f"no default condition report for {case_name!r}")


def _condition_report_from_report(report: dict[str, Any], case_name: str) -> str:
    metadata = (
        report.get("cached_query", {})
        .get("embedding_metadata", {})
    )
    path = metadata.get("condition_report") if isinstance(metadata, dict) else None
    if path:
        return str(path)
    return _condition_report_for_case_name(case_name)


def _case_specs_for_root(root: Path, variant_dir: str) -> list[tuple[str, str, str]]:
    default_cases = list(CONTRAST_CASES)
    if all((_case_dir(root, case_name, variant_dir) / "prefix_latent_story_smoke_report.json").exists() for _label, case_name, _color in default_cases):
        return default_cases
    rows: list[tuple[str, str, str]] = []
    for case_root in sorted(Path(root).iterdir()):
        if not case_root.is_dir():
            continue
        case_name = case_root.name
        case_dir = _case_dir(root, case_name, variant_dir)
        if (case_dir / "prefix_latent_story_smoke_report.json").exists():
            label = re.sub(r"_start\d+$", "", case_name).replace("_", " ")
            rows.append((label, case_name, "#666666"))
    if not rows:
        raise FileNotFoundError(f"no component cases found under {root}")
    return rows


def _operational_variant_index(report: dict[str, Any]) -> int:
    selected = report.get("selected_start_state", {})
    if isinstance(selected, dict) and selected.get("variant_index") is not None:
        return int(selected["variant_index"])
    for idx, row in enumerate(report.get("variant_rows", [])):
        if isinstance(row, dict) and bool(row.get("is_operational")):
            return int(idx)
    return 0


def _load_case(
    root: Path, case: tuple[str, str, str], variant_dir: str
) -> dict[str, Any]:
    label, case_name, color = case
    case_dir = _case_dir(root, case_name, variant_dir)
    report = _load_json(case_dir / "prefix_latent_story_smoke_report.json")
    arrays = np.load(case_dir / "prefix_latent_story_smoke_arrays.npz")
    idx = _operational_variant_index(report)
    states = np.asarray(arrays["generated_states"][idx], dtype=np.float32)
    start = np.asarray(arrays["requested_raw"][idx], dtype=np.float32)
    return {
        "label": label,
        "case_name": case_name,
        "color": color,
        "report": report,
        "condition_report": _condition_report_from_report(report, case_name),
        "start_index": _case_start_index(case_name),
        "states": states,
        "start": start,
        "run_report": str(case_dir / "prefix_latent_story_smoke_report.json"),
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
    }


def _terminal_summary(case: dict[str, Any]) -> dict[str, dict[str, float]]:
    terminal = np.asarray(case["states"], dtype=np.float32)[:, -1, :]
    return {market: _summary(terminal[:, idx]) for market, idx in MARKETS}


def _gap(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
) -> tuple[float, list[dict[str, Any]]]:
    squared = 0.0
    rows = []
    for market, _idx in MARKETS:
        l = left[market]
        r = right[market]
        pooled = float(np.sqrt((l["std"] ** 2 + r["std"] ** 2) / 2.0))
        mean_gap = float(l["mean"] - r["mean"])
        standardized = mean_gap / pooled if pooled > 1e-12 else 0.0
        squared += standardized * standardized
        rows.append(
            {
                "market": market,
                "left_mean_minus_right_mean": mean_gap,
                "pooled_std": pooled,
                "standardized_mean_gap": standardized,
            }
        )
    return (
        float(np.sqrt(squared)),
        sorted(
            rows,
            key=lambda row: abs(float(row["standardized_mean_gap"])),
            reverse=True,
        )[:5],
    )


def _pairwise_gaps(cases: list[dict[str, Any]], label: str) -> list[dict[str, Any]]:
    summaries = {case["case_name"]: _terminal_summary(case) for case in cases}
    rows = []
    for left, right in combinations(cases, 2):
        gap, market_rows = _gap(
            summaries[left["case_name"]], summaries[right["case_name"]]
        )
        rows.append(
            {
                "control": label,
                "left_case": left["case_name"],
                "right_case": right["case_name"],
                "standardized_l2_gap": gap,
                "largest_abs_market_gaps": market_rows,
            }
        )
    return sorted(rows, key=lambda row: float(row["standardized_l2_gap"]), reverse=True)


def _gap_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    values = [float(row.get("standardized_l2_gap", 0.0)) for row in rows]
    return {
        "pair_count": int(len(values)),
        "median_gap": float(median(values)) if values else 0.0,
        "mean_gap": float(np.mean(values)) if values else 0.0,
        "p90_gap": float(np.quantile(values, 0.90)) if values else 0.0,
        "max_gap": max(values) if values else 0.0,
    }


def _bootstrap_cases(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for case in cases:
        states = np.asarray(case["states"], dtype=np.float32)
        split = states.shape[0] // 2
        if split < 2:
            continue
        left = {
            **case,
            "case_name": f"{case['case_name']}#left",
            "states": states[:split],
        }
        right = {
            **case,
            "case_name": f"{case['case_name']}#right",
            "states": states[split:],
        }
        gap, market_rows = _gap(_terminal_summary(left), _terminal_summary(right))
        rows.append(
            {
                "control": "within_run_bootstrap",
                "case_name": case["case_name"],
                "left_sample_count": int(split),
                "right_sample_count": int(states.shape[0] - split),
                "standardized_l2_gap": gap,
                "largest_abs_market_gaps": market_rows,
            }
        )
    return sorted(rows, key=lambda row: float(row["standardized_l2_gap"]), reverse=True)


def _run_story_case(
    *,
    case_name: str,
    condition_report: str,
    output_dir: Path,
    samples: int,
    steps: int,
    seed: int,
    decoder_seed: int | None,
    memory_prior_mode: str,
    memory_prior_top_k: int,
    memory_prior_temperature: float,
    device: str,
    rollout_temperature: float,
    bridge_report: str | None = None,
    bridge_arrays: str | None = None,
    pipeline_report: str | None = None,
    rollout_fan_scale: float = 1.0,
    explicit_start_window_index: int = 18,
) -> dict[str, Any]:
    args = build_prefix_latent_run_args(
        start_mode="explicit_start_window",
        samples=int(samples),
        condition_report=str(condition_report),
        explicit_start_window_index=int(explicit_start_window_index),
        skip_rollout=False,
        output_dir=str(output_dir),
    )
    if bridge_report:
        args.bridge_report = str(bridge_report)
    if bridge_arrays:
        args.bridge_arrays = str(bridge_arrays)
    if pipeline_report:
        args.pipeline_report = str(pipeline_report)
    args.include_original_baseline = False
    args.rollout_mixture_mode = "component_prefix_mixture"
    args.memory_prior_mode = str(memory_prior_mode)
    args.memory_prior_top_k = int(memory_prior_top_k)
    args.memory_prior_temperature = float(memory_prior_temperature)
    args.steps = int(steps)
    args.seed = int(seed)
    args.decoder_seed = None if decoder_seed is None else int(decoder_seed)
    args.rollout_seed = int(seed)
    args.device = str(device)
    args.chunk_size = 16
    args.temperature = float(rollout_temperature)
    args.rollout_fan_scale = float(rollout_fan_scale)
    report = run_prefix_latent_story_smoke(args)
    arrays = np.load(Path(report["artifact_paths"]["arrays"]))
    return {
        "label": case_name,
        "case_name": case_name,
        "report": report,
        "states": np.asarray(arrays["generated_states"][0], dtype=np.float32),
        "start": np.asarray(arrays["requested_raw"][0], dtype=np.float32),
        "run_report": report["artifact_paths"]["report"],
    }


def _run_observed_cases(
    *,
    component_root: Path,
    variant_dir: str,
    samples: int,
    steps: int,
    seed: int,
    decoder_seed: int | None,
    memory_prior_mode: str,
    memory_prior_top_k: int,
    memory_prior_temperature: float,
    device: str,
    rollout_temperature: float,
    bridge_report: str | None,
    bridge_arrays: str | None,
    pipeline_report: str | None,
    condition_report_root: str | Path | None,
    rollout_fan_scale: float,
) -> list[dict[str, Any]]:
    output = []
    for _label, case_name, _color in CONTRAST_CASES:
        condition_report = _condition_report_for_case_name(
            case_name, condition_report_root
        )
        start_index = _case_start_index(case_name)
        output.append(
            _run_story_case(
                case_name=case_name,
                condition_report=condition_report,
                output_dir=component_root / case_name / f"fixed_start_{start_index}" / variant_dir,
                samples=samples,
                steps=steps,
                seed=seed,
                decoder_seed=decoder_seed,
                memory_prior_mode=memory_prior_mode,
                memory_prior_top_k=memory_prior_top_k,
                memory_prior_temperature=memory_prior_temperature,
                device=device,
                rollout_temperature=rollout_temperature,
                bridge_report=bridge_report,
                bridge_arrays=bridge_arrays,
                pipeline_report=pipeline_report,
                rollout_fan_scale=rollout_fan_scale,
                explicit_start_window_index=start_index,
            )
        )
    return output


def _run_start_only_controls(
    *,
    cases: list[dict[str, Any]],
    output_dir: Path,
    samples: int,
    steps: int,
    seed: int,
    decoder_seed: int | None,
    device: str,
    rollout_temperature: float,
    bridge_report: str | None,
    bridge_arrays: str | None,
    pipeline_report: str | None,
    condition_report_root: str | Path | None,
    rollout_fan_scale: float,
    memory_prior_top_k: int,
    memory_prior_temperature: float,
) -> list[dict[str, Any]]:
    output = []
    for case in cases:
        condition_report = case.get("condition_report") or _condition_report_for_case_name(
            case["case_name"], condition_report_root
        )
        output.append(
            _run_story_case(
                case_name=case["case_name"],
                condition_report=condition_report,
                output_dir=output_dir / case["case_name"],
                samples=samples,
                steps=steps,
                seed=seed,
                decoder_seed=decoder_seed,
                memory_prior_mode="soft_topk_start_only",
                memory_prior_top_k=memory_prior_top_k,
                memory_prior_temperature=memory_prior_temperature,
                device=device,
                rollout_temperature=rollout_temperature,
                bridge_report=bridge_report,
                bridge_arrays=bridge_arrays,
                pipeline_report=pipeline_report,
                rollout_fan_scale=rollout_fan_scale,
                explicit_start_window_index=int(case.get("start_index", 18)),
            )
        )
    return output


def _run_repeat_controls(
    *,
    cases: list[dict[str, Any]],
    output_dir: Path,
    samples: int,
    steps: int,
    seeds: list[int],
    decoder_seed: int | None,
    repeat_case_count: int,
    device: str,
    rollout_temperature: float,
    bridge_report: str | None,
    bridge_arrays: str | None,
    pipeline_report: str | None,
    condition_report_root: str | Path | None,
    rollout_fan_scale: float,
    memory_prior_mode: str,
    memory_prior_top_k: int,
    memory_prior_temperature: float,
) -> list[dict[str, Any]]:
    repeat_cases = cases[: int(repeat_case_count)]
    output = []
    for case in repeat_cases:
        condition_report = case.get("condition_report") or _condition_report_for_case_name(
            case["case_name"], condition_report_root
        )
        for seed in seeds:
            output.append(
                _run_story_case(
                    case_name=f"{case['case_name']}#seed_{seed}",
                    condition_report=condition_report,
                    output_dir=output_dir / case["case_name"] / f"seed_{seed}",
                    samples=samples,
                    steps=steps,
                    seed=seed,
                    decoder_seed=decoder_seed,
                    memory_prior_mode=str(memory_prior_mode),
                    memory_prior_top_k=memory_prior_top_k,
                    memory_prior_temperature=memory_prior_temperature,
                    device=device,
                    rollout_temperature=rollout_temperature,
                    bridge_report=bridge_report,
                    bridge_arrays=bridge_arrays,
                    pipeline_report=pipeline_report,
                    rollout_fan_scale=rollout_fan_scale,
                    explicit_start_window_index=int(case.get("start_index", 18)),
                )
            )
    return output


def _repeat_gaps(repeats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for case in repeats:
        base = str(case["case_name"]).split("#seed_")[0]
        grouped.setdefault(base, []).append(case)
    rows = []
    for base, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        for left, right in combinations(group, 2):
            gap, market_rows = _gap(_terminal_summary(left), _terminal_summary(right))
            rows.append(
                {
                    "control": "same_narrative_repeat",
                    "case_name": base,
                    "left_case": left["case_name"],
                    "right_case": right["case_name"],
                    "standardized_l2_gap": gap,
                    "largest_abs_market_gaps": market_rows,
                }
            )
    return sorted(rows, key=lambda row: float(row["standardized_l2_gap"]), reverse=True)


def build_control_report(args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.run_observed):
        _run_observed_cases(
            component_root=Path(args.component_root),
            variant_dir=str(args.variant_dir),
            samples=int(args.samples),
            steps=int(args.steps),
            seed=int(args.null_seed),
            decoder_seed=args.decoder_seed,
            memory_prior_mode=str(args.memory_prior_mode),
            memory_prior_top_k=int(args.memory_prior_top_k),
            memory_prior_temperature=float(args.memory_prior_temperature),
            device=str(args.device),
            rollout_temperature=float(args.rollout_temperature),
            bridge_report=args.bridge_report,
            bridge_arrays=args.bridge_arrays,
            pipeline_report=args.pipeline_report,
            condition_report_root=args.condition_report_root,
            rollout_fan_scale=float(args.rollout_fan_scale),
        )
    case_specs = _case_specs_for_root(Path(args.component_root), str(args.variant_dir))
    cases = [
        _load_case(Path(args.component_root), case, str(args.variant_dir))
        for case in case_specs
    ]
    observed = _pairwise_gaps(cases, "observed_narrative")
    bootstrap = _bootstrap_cases(cases)
    start_only_cases = (
        _run_start_only_controls(
            cases=cases,
            output_dir=Path(args.output_dir) / "start_only_controls",
            samples=int(args.samples),
            steps=int(args.steps),
            seed=int(args.null_seed),
            decoder_seed=args.decoder_seed,
            device=str(args.device),
            rollout_temperature=float(args.rollout_temperature),
            bridge_report=args.bridge_report,
            bridge_arrays=args.bridge_arrays,
            pipeline_report=args.pipeline_report,
            condition_report_root=args.condition_report_root,
            rollout_fan_scale=float(args.rollout_fan_scale),
            memory_prior_top_k=int(args.memory_prior_top_k),
            memory_prior_temperature=float(args.memory_prior_temperature),
        )
        if bool(args.run_start_only)
        else []
    )
    start_only = (
        _pairwise_gaps(start_only_cases, "start_only_null") if start_only_cases else []
    )
    seeds = [int(item) for item in str(args.repeat_seeds).split(",") if item.strip()]
    repeat_cases = (
        _run_repeat_controls(
            cases=cases,
            output_dir=Path(args.output_dir) / "repeat_controls",
            samples=int(args.samples),
            steps=int(args.steps),
            seeds=seeds,
            decoder_seed=args.decoder_seed,
            repeat_case_count=int(args.repeat_case_count),
            device=str(args.device),
            rollout_temperature=float(args.rollout_temperature),
            bridge_report=args.bridge_report,
            bridge_arrays=args.bridge_arrays,
            pipeline_report=args.pipeline_report,
            condition_report_root=args.condition_report_root,
            rollout_fan_scale=float(args.rollout_fan_scale),
            memory_prior_mode=str(args.memory_prior_mode),
            memory_prior_top_k=int(args.memory_prior_top_k),
            memory_prior_temperature=float(args.memory_prior_temperature),
        )
        if bool(args.run_repeat)
        else []
    )
    repeat = _repeat_gaps(repeat_cases) if repeat_cases else []

    observed_summary = _gap_summary(observed)
    bootstrap_summary = _gap_summary(bootstrap)
    start_only_summary = _gap_summary(start_only)
    repeat_summary = _gap_summary(repeat)
    observed_gap = max(float(observed_summary["median_gap"]), 1e-12)
    ratios = {
        "bootstrap_to_observed_median": float(
            bootstrap_summary["median_gap"] / observed_gap
        ),
        "start_only_to_observed_median": (
            float(start_only_summary["median_gap"] / observed_gap)
            if start_only
            else None
        ),
        "repeat_to_observed_median": (
            float(repeat_summary["median_gap"] / observed_gap) if repeat else None
        ),
    }
    warnings = []
    failures = []
    if ratios["bootstrap_to_observed_median"] > float(args.max_bootstrap_ratio):
        warnings.append("bootstrap_noise_close_to_observed")
    if ratios["start_only_to_observed_median"] is not None and ratios[
        "start_only_to_observed_median"
    ] > float(args.max_start_only_ratio):
        failures.append("start_only_gap_too_close_to_observed")
    if ratios["repeat_to_observed_median"] is not None and ratios[
        "repeat_to_observed_median"
    ] > float(args.max_repeat_ratio):
        failures.append("repeat_gap_too_close_to_observed")
    status = "fail" if failures else "warning" if warnings else "pass"
    return {
        "status": status,
        "scope_note": (
            "Component-mixture fixed-start control report. Observed gaps compare "
            "different narratives at the same raw start. Null controls remove "
            "narrative ranking or repeat the same narrative with new seeds."
        ),
        "component_root": str(args.component_root),
        "variant_dir": str(args.variant_dir),
        "memory_prior_mode": str(args.memory_prior_mode),
        "memory_prior_top_k": int(args.memory_prior_top_k),
        "memory_prior_temperature": float(args.memory_prior_temperature),
        "bridge_report": str(args.bridge_report) if args.bridge_report else None,
        "bridge_arrays": str(args.bridge_arrays) if args.bridge_arrays else None,
        "pipeline_report": str(args.pipeline_report) if args.pipeline_report else None,
        "condition_report_root": (
            str(args.condition_report_root) if args.condition_report_root else None
        ),
        "rollout_temperature": float(args.rollout_temperature),
        "thresholds": {
            "max_start_only_ratio": float(args.max_start_only_ratio),
            "max_bootstrap_ratio": float(args.max_bootstrap_ratio),
            "max_repeat_ratio": float(args.max_repeat_ratio),
        },
        "rollout_fan_scale": float(args.rollout_fan_scale),
        "run_observed": bool(args.run_observed),
        "summaries": {
            "observed_narrative": observed_summary,
            "within_run_bootstrap": bootstrap_summary,
            "start_only_null": start_only_summary if start_only else None,
            "same_narrative_repeat": repeat_summary if repeat else None,
        },
        "ratios": ratios,
        "observed_pairwise": observed,
        "bootstrap_pairwise": bootstrap,
        "start_only_pairwise": start_only,
        "repeat_pairwise": repeat,
        "warnings": warnings,
        "failures": failures,
        "interpretation": [
            "A useful narrative conditioner should have observed narrative gaps larger than bootstrap/repeat noise.",
            "The start-only null should be small because it removes narrative support ranking while keeping the same start.",
            "This report tests conditionality mechanics, not realized future accuracy.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-root", default=str(DEFAULT_ROOT))
    parser.add_argument("--variant-dir", default=DEFAULT_VARIANT_DIR)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--memory-prior-mode", default=DEFAULT_MEMORY_PRIOR_MODE)
    parser.add_argument("--memory-prior-top-k", type=int, default=8)
    parser.add_argument("--memory-prior-temperature", type=float, default=0.2)
    parser.add_argument("--bridge-report")
    parser.add_argument("--bridge-arrays")
    parser.add_argument("--pipeline-report")
    parser.add_argument(
        "--condition-report-root",
        help=(
            "Optional root containing <case_name>/condition_only_report.json "
            "condition-memory reports. Falls back to built-in case reports."
        ),
    )
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--run-observed", action="store_true")
    parser.add_argument("--run-start-only", action="store_true")
    parser.add_argument("--run-repeat", action="store_true")
    parser.add_argument("--rollout-temperature", type=float, default=0.5)
    parser.add_argument("--rollout-fan-scale", type=float, default=1.0)
    parser.add_argument("--repeat-case-count", type=int, default=3)
    parser.add_argument("--repeat-seeds", default="941,942")
    parser.add_argument("--null-seed", type=int, default=941)
    parser.add_argument(
        "--decoder-seed",
        type=int,
        default=None,
        help=(
            "Optional fixed decoder seed. Repeat controls should set this to "
            "hold decoded prefixes stable while varying rollout seeds."
        ),
    )
    parser.add_argument("--max-start-only-ratio", type=float, default=0.50)
    parser.add_argument("--max-bootstrap-ratio", type=float, default=0.75)
    parser.add_argument("--max-repeat-ratio", type=float, default=0.75)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    report = build_control_report(args)
    report["artifact_paths"] = {
        "report": str(output_dir / "component_fixed_start_controls.json")
    }
    _write_json(report["artifact_paths"]["report"], report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": report["artifact_paths"]["report"],
                "ratios": report["ratios"],
                "warnings": report["warnings"],
                "failures": report["failures"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
