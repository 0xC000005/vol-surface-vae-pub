#!/usr/bin/env python
"""TestFlight for response-aware support weighting.

This script tests the next NL prefix-latent objective without new OpenAI or
generator calls. It consumes the cached 932a component-preserving rollouts,
scores each support component by the generator response it produces in the
current narrative's risk channels, reweights support components, and reruns the
fixed-start conditionality stress test on the reweighted sample deck.

The method is intentionally bounded: it keeps the same fixed start, the same
candidate support components, the same frozen SNI rollout samples, and only
changes the support weights used to pool component paths.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_conditionality_stress_test import (  # noqa: E402
    CASE_LABELS,
    CASE_RELEVANT_FACTORS,
    _policy_scorecard,
    _portfolio_summary,
    plot_portfolio_delta,
    plot_reference_contrasts,
    plot_relevant_factor_panels,
    plot_terminal_overlays,
)
from experiments.backfill.block_ar.nl_fixed_start_rollout_policy_comparison import (  # noqa: E402
    CASE_NAMES,
    FACTOR_INDEX,
    SUMMARY_FACTORS,
    _case_arrays_path,
    _case_report_path,
    _direction_status,
    _load_json,
    _operational_variant_index,
    _portfolio_stats,
    _support_rows,
    load_case_result,
)


DEFAULT_INPUT_ROOT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "fixed_start_rollout_policy_comparison_932a_s384"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_response_aware_support_weighting_934a"
)
BASELINE_POLICY = "current_start_checked_gap30"
START_ONLY_POLICY = "start_only_topk"
RESPONSE_POLICY = "response_aware_support_weighting"

MARKET_TO_FACTOR = {
    "SPX": "SPX",
    "VIX": "VIX",
    "DXY": "DXY",
    "USDJPY": "DXY",
    "CRUDE": "Crude",
    "CRUDE_OIL": "Crude",
    "OIL": "Crude",
    "US10Y": "US10Y",
    "TREASURY_10Y": "US10Y",
    "UST10Y": "US10Y",
    "RATES": "US10Y",
    "BBB_OAS": "BBB_OAS",
    "CREDIT_SPREADS": "BBB_OAS",
    "SPREADS": "BBB_OAS",
    "GOLD": "Gold",
    "IV_ATM_1Y": "IV_ATM_1Y",
}
DIRECTION_SIGN = {
    "up": 1.0,
    "higher": 1.0,
    "wider": 1.0,
    "steeper": 1.0,
    "down": -1.0,
    "lower": -1.0,
    "tighter": -1.0,
    "narrower": -1.0,
    "flatter": -1.0,
    "flat": 0.0,
    "mixed": 0.0,
    "unchanged": 0.0,
}
CONFIDENCE_WEIGHT = {"high": 1.0, "medium": 0.7, "low": 0.4}
MAGNITUDE_WEIGHT = {"large": 1.2, "medium": 1.0, "small": 0.75}


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n")


def _write_text(path: str | Path, text: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text.rstrip() + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def _normal_market(value: Any) -> str:
    return str(value or "").strip().upper().replace(" ", "_")


def implication_rows(grounding: dict[str, Any]) -> list[dict[str, Any]]:
    """Return current/recent implications from both live and cached schemas."""

    if "condition_only_grounding" in grounding and isinstance(
        grounding["condition_only_grounding"], dict
    ):
        grounding = grounding["condition_only_grounding"]
    rows = grounding.get("current_market_state_implications", [])
    return [row for row in rows if isinstance(row, dict)]


def response_channels(
    grounding: dict[str, Any],
    *,
    fallback_factors: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Convert grounded current-market implications into response channels."""

    channels: list[dict[str, Any]] = []
    for row in implication_rows(grounding):
        factor = MARKET_TO_FACTOR.get(_normal_market(row.get("market")))
        if factor not in FACTOR_INDEX:
            continue
        direction = str(row.get("direction", "")).lower()
        confidence = str(row.get("confidence", "medium")).lower()
        magnitude = str(row.get("magnitude", "medium")).lower()
        channels.append(
            {
                "factor": factor,
                "sign": float(DIRECTION_SIGN.get(direction, 0.0)),
                "direction": direction,
                "weight": float(
                    CONFIDENCE_WEIGHT.get(confidence, 0.6)
                    * MAGNITUDE_WEIGHT.get(magnitude, 1.0)
                ),
                "confidence": confidence,
                "magnitude": magnitude,
            }
        )
    seen = {str(row["factor"]) for row in channels}
    for factor in fallback_factors:
        if factor not in seen and factor in FACTOR_INDEX:
            channels.append(
                {
                    "factor": factor,
                    "sign": 0.0,
                    "direction": "activation",
                    "weight": 0.35,
                    "confidence": "fallback",
                    "magnitude": "fallback",
                }
            )
    return channels


def _softmax(values: np.ndarray, *, temperature: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    temp = max(float(temperature), 1.0e-6)
    shifted = arr / temp - float(np.max(arr / temp))
    exp = np.exp(shifted)
    total = float(np.sum(exp))
    if total <= 0.0 or not math.isfinite(total):
        return np.ones(arr.shape[0], dtype=np.float64) / float(arr.shape[0])
    return exp / total


def _component_response_score(
    *,
    component_states: np.ndarray,
    start: np.ndarray,
    channels: list[dict[str, Any]],
) -> float:
    """Score generator response in narrative-relevant channels.

    The signed term measures whether the generated component moves in the
    direction implied by the current-market channel. The activation and width
    terms prevent the scorer from becoming a deterministic sign-following
    forecast; components can also score through risk-channel dispersion.
    """

    states = np.asarray(component_states, dtype=np.float64)
    start_arr = np.asarray(start, dtype=np.float64).reshape(-1)
    pieces: list[float] = []
    weights: list[float] = []
    for channel in channels:
        factor = str(channel["factor"])
        idx = FACTOR_INDEX[factor]
        scale = max(abs(float(start_arr[idx])), 1.0)
        terminal = (states[:, -1, idx] - float(start_arr[idx])) / scale
        path = (states[:, :, idx] - float(start_arr[idx])) / scale
        sign = float(channel.get("sign", 0.0))
        signed = sign * float(np.median(terminal)) if sign != 0.0 else 0.0
        activation = abs(float(np.median(terminal)))
        width = float(np.percentile(path, 90) - np.percentile(path, 10))
        pieces.append(0.55 * signed + 0.25 * activation + 0.20 * width)
        weights.append(float(channel.get("weight", 1.0)))
    if not pieces:
        return 0.0
    w = np.asarray(weights, dtype=np.float64)
    v = np.asarray(pieces, dtype=np.float64)
    if float(w.sum()) <= 0.0:
        return float(np.mean(v))
    return float(np.sum(v * w) / np.sum(w))


def _allocate_counts(weights: np.ndarray, samples: int) -> np.ndarray:
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
    if float(weights.sum()) <= 0.0:
        weights = np.ones_like(weights)
    weights = weights / float(weights.sum())
    raw = weights * int(samples)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(samples) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts


def _preview_and_pool_states(
    component_states: np.ndarray,
    *,
    pilot_samples: int,
    min_pool_samples: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Split component samples into a response-preview set and a pooling set.

    With ``pilot_samples <= 0`` the historical 934a behavior is preserved:
    score and pool from the same cached component samples. With a positive
    pilot size, the score is computed from a small shuffled preview subset and
    final pooling samples are drawn from the remaining component samples when
    possible. This emulates an operational two-stage frozen-generator preview
    without using realized futures.
    """

    states = np.asarray(component_states, dtype=np.float32)
    if states.ndim != 3:
        raise ValueError("component_states must have shape [S,T,C]")
    total = int(states.shape[0])
    if total <= 0:
        raise ValueError("component_states must contain at least one sample")
    if int(pilot_samples) <= 0:
        return (
            states,
            states,
            {
                "preview_sample_count": total,
                "pool_sample_count": total,
                "pool_reuses_preview": True,
            },
        )
    order = rng.permutation(total)
    min_pool = max(int(min_pool_samples), 1)
    max_preview = max(total - min_pool, 1)
    preview_count = min(int(pilot_samples), max_preview)
    preview_idx = order[:preview_count]
    pool_idx = order[preview_count:]
    pool_reuses_preview = False
    if pool_idx.size <= 0:
        pool_idx = order
        pool_reuses_preview = True
    return (
        states[preview_idx],
        states[pool_idx],
        {
            "preview_sample_count": int(preview_idx.size),
            "pool_sample_count": int(pool_idx.size),
            "pool_reuses_preview": bool(pool_reuses_preview),
        },
    )


def _component_slices(
    *,
    arrays: dict[str, np.ndarray],
    op_idx: int,
    states: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    cursor = 0
    variant = np.asarray(arrays["rollout_component_variant_index"], dtype=np.int64)
    windows = np.asarray(arrays["rollout_component_window_index"], dtype=np.int64)
    weights = np.asarray(arrays["rollout_component_weight"], dtype=np.float64)
    counts = np.asarray(arrays["rollout_component_sample_count"], dtype=np.int64)
    for variant_idx, window_idx, weight, count in zip(
        variant, windows, weights, counts, strict=True
    ):
        if int(variant_idx) != int(op_idx):
            continue
        count = int(count)
        rows.append(
            {
                "window_index": int(window_idx),
                "base_weight": float(weight),
                "sample_count": count,
                "states": states[cursor : cursor + count],
            }
        )
        cursor += count
    if cursor != int(states.shape[0]):
        raise ValueError(
            f"component sample counts sum to {cursor}, expected {states.shape[0]}"
        )
    return rows


def _load_grounding(report: dict[str, Any]) -> dict[str, Any]:
    cached = report.get("cached_query", {})
    return cached.get("grounding", {}) if isinstance(cached, dict) else {}


def build_response_aware_case(
    *,
    input_root: str | Path,
    case_name: str,
    baseline_policy: str,
    response_alpha: float,
    response_temperature: float,
    pilot_samples_per_component: int,
    min_pool_samples_per_component: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_path = _case_report_path(input_root, case_name, baseline_policy)
    arrays_path = _case_arrays_path(input_root, case_name, baseline_policy)
    report = _load_json(report_path)
    arrays = dict(np.load(arrays_path))
    op_idx = _operational_variant_index(report, arrays)
    states = np.asarray(arrays["generated_states"], dtype=np.float32)[op_idx]
    start = np.asarray(arrays["requested_raw"], dtype=np.float32)[op_idx]
    support = _support_rows(report)
    components = _component_slices(arrays=arrays, op_idx=op_idx, states=states)
    grounding = _load_grounding(report)
    channels = response_channels(
        grounding,
        fallback_factors=CASE_RELEVANT_FACTORS[str(case_name)],
    )
    rng = np.random.default_rng(int(seed))
    scored_components: list[dict[str, Any]] = []
    pool_components: list[np.ndarray] = []
    split_details: list[dict[str, Any]] = []
    for component in components:
        score_states, pool_states, split = _preview_and_pool_states(
            np.asarray(component["states"], dtype=np.float32),
            pilot_samples=int(pilot_samples_per_component),
            min_pool_samples=int(min_pool_samples_per_component),
            rng=rng,
        )
        scored_components.append({**component, "states": score_states})
        pool_components.append(pool_states)
        split_details.append(split)
    scores = np.asarray(
        [
            _component_response_score(
                component_states=component["states"],
                start=start,
                channels=channels,
            )
            for component in scored_components
        ],
        dtype=np.float64,
    )
    base_weights = np.asarray([component["base_weight"] for component in components])
    base_weights = np.maximum(base_weights, 1.0e-8)
    if scores.size > 1 and float(np.std(scores)) > 1.0e-9:
        scaled_scores = (scores - float(np.mean(scores))) / float(np.std(scores))
    else:
        scaled_scores = scores * 0.0
    logits = np.log(base_weights) + float(response_alpha) * scaled_scores
    response_weights = _softmax(logits, temperature=float(response_temperature))
    counts = _allocate_counts(response_weights, int(states.shape[0]))
    pieces = []
    support_by_window = {int(row["window_index"]): dict(row) for row in support}
    new_support = []
    for component, pool_states, split, weight, score, scaled, count in zip(
        components,
        pool_components,
        split_details,
        response_weights,
        scores,
        scaled_scores,
        counts,
        strict=True,
    ):
        component_states = np.asarray(pool_states, dtype=np.float32)
        if int(count) > 0:
            choice = rng.choice(
                component_states.shape[0], size=int(count), replace=True
            )
            pieces.append(component_states[choice])
        row = support_by_window.get(int(component["window_index"]), {})
        row.update(
            {
                "window_index": int(component["window_index"]),
                "weight": float(weight),
                "base_weight": float(component["base_weight"]),
                "response_score": float(score),
                "response_score_z": float(scaled),
                "response_sample_count": int(count),
                **split,
            }
        )
        new_support.append(row)
    if not pieces:
        raise RuntimeError(
            f"{case_name}: response-aware reweighting produced no samples"
        )
    reweighted_states = np.concatenate(pieces, axis=0).astype(np.float32)
    if reweighted_states.shape != states.shape:
        raise RuntimeError(
            f"{case_name}: expected states {states.shape}, got {reweighted_states.shape}"
        )
    case = {
        "case": str(case_name),
        "policy": RESPONSE_POLICY,
        "report_path": str(report_path),
        "arrays_path": str(arrays_path),
        "generated_shape": [int(value) for value in reweighted_states.shape],
        "start": start,
        "states": reweighted_states,
        "support": new_support,
        "support_count": int(len(new_support)),
        "direction_status": _direction_status(report),
        "terminal_raw_levels": {
            name: {
                "p10": float(
                    np.percentile(reweighted_states[:, -1, FACTOR_INDEX[name]], 10)
                ),
                "p50": float(
                    np.percentile(reweighted_states[:, -1, FACTOR_INDEX[name]], 50)
                ),
                "p90": float(
                    np.percentile(reweighted_states[:, -1, FACTOR_INDEX[name]], 90)
                ),
                "mean": float(np.mean(reweighted_states[:, -1, FACTOR_INDEX[name]])),
                "std": float(np.std(reweighted_states[:, -1, FACTOR_INDEX[name]])),
            }
            for name in SUMMARY_FACTORS
        },
        "portfolio_stats": _portfolio_stats(reweighted_states, start),
    }
    detail = {
        "case": str(case_name),
        "channels": channels,
        "components": [
            {
                "window_index": int(component["window_index"]),
                "base_weight": float(component["base_weight"]),
                "response_weight": float(weight),
                "response_score": float(score),
                "response_score_z": float(scaled),
                "base_sample_count": int(component["sample_count"]),
                "response_sample_count": int(count),
                **split,
            }
            for component, split, weight, score, scaled, count in zip(
                components,
                split_details,
                response_weights,
                scores,
                scaled_scores,
                counts,
                strict=True,
            )
        ],
        "weight_l1_delta": float(np.sum(np.abs(response_weights - base_weights))),
        "effective_support_count": float(1.0 / np.sum(response_weights**2)),
        "pilot_samples_per_component": int(pilot_samples_per_component),
        "min_pool_samples_per_component": int(min_pool_samples_per_component),
    }
    return case, detail


def _case_map(
    *,
    input_root: str | Path,
    cases: tuple[str, ...],
    policy: str,
) -> dict[str, dict[str, Any]]:
    return {
        case: load_case_result(
            output_root=input_root,
            case_name=case,
            policy_name=policy,
        )
        for case in cases
    }


def _plot_artifacts(
    *,
    output_dir: Path,
    cases: dict[str, dict[str, Any]],
    reference_case: str,
    prefix: str,
) -> dict[str, str]:
    paths = {
        "narrative_relevant_factor_panels": str(
            output_dir / f"{prefix}_narrative_relevant_factor_panels.png"
        ),
        "reference_contrast_panels": str(
            output_dir / f"{prefix}_reference_contrast_panels.png"
        ),
        "terminal_distribution_overlays": str(
            output_dir / f"{prefix}_terminal_distribution_overlays.png"
        ),
        "portfolio_tail_deltas": str(
            output_dir / f"{prefix}_portfolio_tail_deltas.png"
        ),
    }
    portfolio_rows = _portfolio_summary(cases, reference_case=reference_case)
    plot_relevant_factor_panels(
        cases=cases,
        output=paths["narrative_relevant_factor_panels"],
        reference_case=reference_case,
    )
    plot_reference_contrasts(
        cases=cases,
        output=paths["reference_contrast_panels"],
        reference_case=reference_case,
    )
    plot_terminal_overlays(cases=cases, output=paths["terminal_distribution_overlays"])
    plot_portfolio_delta(
        portfolio_rows=portfolio_rows, output=paths["portfolio_tail_deltas"]
    )
    return paths


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Response-Aware Support Weighting TestFlight",
        "",
        "This TestFlight reweights cached component-preserving support rollouts by "
        "their generator response in narrative-relevant risk channels. It uses no "
        "new OpenAI calls and no new generator calls.",
        "",
        "## Scorecard",
        "",
        "| Policy | Status | Gates | Support Jaccard | Relevant KS | Path Energy | Portfolio KS | VaR95 Range |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["scorecards"]:
        pairwise = row["pairwise"]
        lines.append(
            f"| {row['label']} | {row['status']} | "
            f"{row['pass_count']}/{row['gate_count']} | "
            f"{pairwise['mean_support_jaccard']:.3f} | "
            f"{pairwise['mean_relevant_terminal_ks']:.3f} | "
            f"{pairwise['mean_relevant_path_energy']:.4f} | "
            f"{pairwise['mean_portfolio_terminal_ks']:.3f} | "
            f"{row['portfolio_var95_loss_range']:.3f} |"
        )
    lines.extend(["", "## Interpretation", ""])
    response = report["scorecards_by_policy"][RESPONSE_POLICY]
    baseline = report["scorecards_by_policy"][BASELINE_POLICY]
    method = report.get("method", {})
    if int(method.get("pilot_samples_per_component", 0) or 0) > 0:
        lines.append(
            "This run uses a preview split: response scores are estimated from "
            f"{int(method['pilot_samples_per_component'])} pilot samples per "
            "component when available, and final pooling draws from held-out "
            "cached component samples."
        )
        lines.append("")
    lines.append(
        "The response-aware candidate is useful only if it improves relevant-factor "
        "or portfolio-tail separation versus the current component-preserving "
        "baseline while the start-only null stays flat."
    )
    lines.append("")
    lines.append(
        f"Relevant KS delta vs baseline: "
        f"{response['pairwise']['mean_relevant_terminal_ks'] - baseline['pairwise']['mean_relevant_terminal_ks']:.4f}."
    )
    lines.append(
        f"Portfolio KS delta vs baseline: "
        f"{response['pairwise']['mean_portfolio_terminal_ks'] - baseline['pairwise']['mean_portfolio_terminal_ks']:.4f}."
    )
    lines.append(
        f"VaR95 range delta vs baseline: "
        f"{response['portfolio_var95_loss_range'] - baseline['portfolio_var95_loss_range']:.4f}."
    )
    lines.extend(["", "## Artifacts", ""])
    for key, value in report["artifact_paths"].items():
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                lines.append(f"- {key}.{subkey}: `{subvalue}`")
        else:
            lines.append(f"- {key}: `{value}`")
    return "\n".join(lines)


def run_testflight(args: argparse.Namespace) -> dict[str, Any]:
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = tuple(args.case or CASE_NAMES)
    baseline_cases = _case_map(
        input_root=input_root, cases=cases, policy=BASELINE_POLICY
    )
    start_only_cases = _case_map(
        input_root=input_root, cases=cases, policy=START_ONLY_POLICY
    )
    response_cases: dict[str, dict[str, Any]] = {}
    response_details = []
    for i, case in enumerate(cases):
        response_case, detail = build_response_aware_case(
            input_root=input_root,
            case_name=case,
            baseline_policy=BASELINE_POLICY,
            response_alpha=float(args.response_alpha),
            response_temperature=float(args.response_temperature),
            pilot_samples_per_component=int(args.pilot_samples_per_component),
            min_pool_samples_per_component=int(args.min_pool_samples_per_component),
            seed=int(args.seed) + i,
        )
        response_cases[case] = response_case
        response_details.append(detail)

    policy_cases = {
        BASELINE_POLICY: baseline_cases,
        RESPONSE_POLICY: response_cases,
        START_ONLY_POLICY: start_only_cases,
    }
    labels = {
        BASELINE_POLICY: "Current component baseline",
        RESPONSE_POLICY: "Response-aware candidate",
        START_ONLY_POLICY: "Start-only null",
    }
    scorecards = []
    for policy, case_map in policy_cases.items():
        scorecard = _policy_scorecard(
            policy=policy,
            cases=case_map,
            reference_case=str(args.reference_case),
        )
        scorecard["label"] = labels[policy]
        scorecards.append(scorecard)

    artifact_paths: dict[str, Any] = {
        "json": str(output_dir / "response_aware_support_weighting_report.json"),
        "markdown": str(output_dir / "response_aware_support_weighting_report.md"),
        "baseline_plots": _plot_artifacts(
            output_dir=output_dir,
            cases=baseline_cases,
            reference_case=str(args.reference_case),
            prefix="baseline",
        ),
        "response_aware_plots": _plot_artifacts(
            output_dir=output_dir,
            cases=response_cases,
            reference_case=str(args.reference_case),
            prefix="response_aware",
        ),
    }
    report = {
        "status": "ok",
        "research_lane": "candidate",
        "result_status": "testflight_completed",
        "benchmark_floor_status": "not_tested",
        "scope_note": (
            "Offline response-aware weighting over cached component rollouts; "
            "no OpenAI calls and no generator calls."
        ),
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "cases": list(cases),
        "reference_case": str(args.reference_case),
        "method": {
            "name": RESPONSE_POLICY,
            "response_alpha": float(args.response_alpha),
            "response_temperature": float(args.response_temperature),
            "pilot_samples_per_component": int(args.pilot_samples_per_component),
            "min_pool_samples_per_component": int(args.min_pool_samples_per_component),
            "score_contract": (
                "log(base support weight) plus standardized generator-response "
                "score in grounded narrative channels"
            ),
        },
        "scorecards": scorecards,
        "scorecards_by_policy": {str(row["policy"]): row for row in scorecards},
        "response_details": response_details,
        "artifact_paths": artifact_paths,
    }
    _write_json(artifact_paths["json"], report)
    _write_text(artifact_paths["markdown"], _markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", action="append", choices=list(CASE_NAMES))
    parser.add_argument("--reference-case", default="fragile_risk_on_start18")
    parser.add_argument("--response-alpha", type=float, default=0.75)
    parser.add_argument("--response-temperature", type=float, default=1.0)
    parser.add_argument(
        "--pilot-samples-per-component",
        type=int,
        default=0,
        help=(
            "If positive, estimate response scores from at most this many "
            "component samples and pool final scenarios from the remaining "
            "cached samples. The default 0 preserves the original full-sample "
            "response-aware upper-bound diagnostic."
        ),
    )
    parser.add_argument(
        "--min-pool-samples-per-component",
        type=int,
        default=1,
        help="Minimum component samples to reserve for final pooling when possible.",
    )
    parser.add_argument("--seed", type=int, default=934)
    args = parser.parse_args()
    report = run_testflight(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "output_dir": report["output_dir"],
                "scorecards": [
                    {
                        "policy": row["policy"],
                        "status": row["status"],
                        "gates": f"{row['pass_count']}/{row['gate_count']}",
                        "relevant_ks": row["pairwise"]["mean_relevant_terminal_ks"],
                        "path_energy": row["pairwise"]["mean_relevant_path_energy"],
                        "portfolio_ks": row["pairwise"]["mean_portfolio_terminal_ks"],
                        "var95_range": row["portfolio_var95_loss_range"],
                    }
                    for row in report["scorecards"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
