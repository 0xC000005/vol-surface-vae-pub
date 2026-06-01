#!/usr/bin/env python
"""Narrative-conditioned calibration over support-grounded SNI ensembles.

This is a bounded TestFlight for the current NL prefix-latent objective. It
keeps the incumbent support-grounded frozen SNI rollout as the base ensemble,
then learns one small directional calibration parameter from historical
backtest rows. The purpose is to test whether a narrative-conditioned
postprocessor can strengthen start-normalized narrative response without
repeating the prior support-ranker failure pattern.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_fixed_start_rollout_policy_comparison import (  # noqa: E402
    CASE_NAMES,
    CASE_SPECS,
    FACTOR_INDEX,
    SUMMARY_FACTORS,
    _ks_statistic,
    _portfolio_pnl,
    _portfolio_stats,
    _support_jaccard,
    load_case_result,
)
from experiments.backfill.block_ar.nl_prefix_latent_component_global_calibration import (  # noqa: E402
    _selected_history_block,
)
from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: E402
    DEFAULT_CHECKPOINT,
)
from experiments.backfill.block_ar.nl_scenario_level_evaluation import (  # noqa: E402
    score_sample_distribution,
    summarize_method_scores,
)
from experiments.backfill.block_ar.nl_start_narrative_attribution import (  # noqa: E402
    DEFAULT_PAPER_FIGURE as DEFAULT_ATTRIBUTION_PAPER_FIGURE,
    DEFAULT_PAPER_TABLE as DEFAULT_ATTRIBUTION_PAPER_TABLE,
    DEFAULT_START_ROOTS,
    analyze_policy,
    cell_feature_vector,
    render_latex_table,
    render_markdown,
    two_way_feature_attribution,
    _pairwise_distribution_metrics,
    _pairwise_feature_distances,
    _result_to_dict,
)


DEFAULT_BACKTEST_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_component_backtest_940d_current_66w/component_backtest_report.json"
)
DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "nl_narrative_ensemble_calibration_945a"
)
DEFAULT_PAPER_TABLE = (
    "paper/narrative_grounded_scenarios/generated_tables/"
    "table_narrative_ensemble_calibration_attribution.tex"
)
DEFAULT_PAPER_FIGURE = (
    "paper/narrative_grounded_scenarios/figures/"
    "narrative_ensemble_calibration_attribution.png"
)
CASE_RELEVANT_FACTORS = {
    "fragile_risk_on": ("SPX", "VIX", "BBB_OAS", "IV_ATM_1Y"),
    "defensive_risk_off": ("SPX", "VIX", "BBB_OAS", "IV_ATM_1Y"),
    "commodity_inflation": ("Crude", "US10Y", "SPX", "IV_ATM_1Y"),
    "dollar_liquidity": ("DXY", "BBB_OAS", "VIX", "SPX"),
    "rates_selloff": ("US10Y", "SPX", "DXY", "IV_ATM_1Y"),
    "safe_haven_gold": ("Gold", "US10Y", "VIX", "SPX"),
}

MARKET_INDEX = {
    **{name.upper(): index for name, index in FACTOR_INDEX.items()},
    "EQUITY": FACTOR_INDEX["SPX"],
    "EQUITIES": FACTOR_INDEX["SPX"],
    "STOCKS": FACTOR_INDEX["SPX"],
    "S&P": FACTOR_INDEX["SPX"],
    "SPREADS": FACTOR_INDEX["BBB_OAS"],
    "CREDIT": FACTOR_INDEX["BBB_OAS"],
    "CREDIT_SPREADS": FACTOR_INDEX["BBB_OAS"],
    "DOLLAR": FACTOR_INDEX["DXY"],
    "USD": FACTOR_INDEX["DXY"],
    "OIL": FACTOR_INDEX["Crude"],
    "CRUDE_OIL": FACTOR_INDEX["Crude"],
    "VOL": FACTOR_INDEX["VIX"],
    "VOLATILITY": FACTOR_INDEX["VIX"],
}

POSITIVE_DIRECTIONS = {
    "up",
    "higher",
    "rise",
    "rising",
    "wider",
    "wide",
    "stronger",
    "strong",
    "elevated",
    "firm",
    "firmer",
    "bid",
}
NEGATIVE_DIRECTIONS = {
    "down",
    "lower",
    "fall",
    "falling",
    "tighter",
    "tight",
    "weaker",
    "weak",
    "compressed",
    "compressing",
    "compression",
    "lowered",
}


def _parse_start_root(raw: str) -> tuple[str, str]:
    if "=" not in raw:
        raise ValueError("--start-root must be LABEL=PATH")
    label, path = raw.split("=", 1)
    label = label.strip()
    path = path.strip()
    if not label or not path:
        raise ValueError("--start-root must be LABEL=PATH")
    return label, path


def _resolve_start_roots(raw_roots: list[str] | None) -> dict[str, str]:
    if not raw_roots:
        return dict(DEFAULT_START_ROOTS)
    roots = dict(_parse_start_root(item) for item in raw_roots)
    if len(roots) < 2:
        raise ValueError("at least two --start-root entries are required")
    missing = [path for path in roots.values() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(
            "start-root path(s) not found: " + ", ".join(sorted(missing))
        )
    return roots


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: str | Path, text: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text.rstrip() + "\n", encoding="utf-8")


def _direction_sign(raw: Any) -> float:
    direction = str(raw or "").strip().lower().replace("-", "_")
    if direction in POSITIVE_DIRECTIONS:
        return 1.0
    if direction in NEGATIVE_DIRECTIONS:
        return -1.0
    return 0.0


def _market_index(raw: Any) -> int | None:
    market = str(raw or "").strip().upper().replace(" ", "_").replace("-", "_")
    return MARKET_INDEX.get(market)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _implication_rows(grounding: dict[str, Any]) -> list[dict[str, Any]]:
    rows = _as_list(grounding.get("market_implications"))
    if rows:
        return [_as_dict(row) for row in rows]
    nested = _as_dict(grounding.get("condition_only_grounding"))
    rows = _as_list(nested.get("current_market_state_implications"))
    rows += _as_list(nested.get("recent_regime_implications"))
    return [_as_dict(row) for row in rows]


def _support_evidence_gate_from_report(
    report: dict[str, Any],
    *,
    mode: str = "direction_status",
) -> float:
    """Return whether the support evidence allows narrative calibration."""

    if mode == "none":
        return 1.0
    if mode != "direction_status":
        raise ValueError(f"unknown support gate mode {mode!r}")
    cached = _as_dict(report.get("cached_query"))
    prior = _as_dict(cached.get("memory_prior"))
    prior_mode = str(prior.get("mode", "")).strip().lower()
    if prior_mode == "soft_topk_start_only":
        return 0.0
    check = _as_dict(prior.get("direction_check"))
    status = str(check.get("status", "")).strip().lower()
    if not status:
        rows = _as_list(report.get("variant_rows"))
        for row in rows:
            variant = _as_dict(row)
            if bool(variant.get("is_operational", False)):
                variant_mode = str(
                    variant.get("memory_prior_mode", "")
                ).strip().lower()
                if variant_mode == "soft_topk_start_only":
                    return 0.0
                status = str(
                    variant.get("memory_prior_direction_status", "")
                ).strip().lower()
                break
        if not status and rows:
            tail = _as_dict(rows[-1])
            variant_mode = str(tail.get("memory_prior_mode", "")).strip().lower()
            if variant_mode == "soft_topk_start_only":
                return 0.0
            status = str(tail.get("memory_prior_direction_status", "")).strip().lower()
    if status in {"reject", "rejected", "fail", "failed"}:
        return 0.0
    return 1.0


def direction_vector_from_text(text: str, *, factor_count: int = 39) -> np.ndarray:
    """Best-effort deterministic direction fallback for cached historical text."""

    lower = str(text or "").lower()
    direction = np.zeros(int(factor_count), dtype=np.float32)

    def set_if_any(index: int, sign: float, phrases: Iterable[str]) -> None:
        if any(phrase in lower for phrase in phrases):
            direction[index] = float(sign)

    set_if_any(FACTOR_INDEX["SPX"], 1.0, ("spx up", "equities are recovering", "equities recovering", "equities up", "risk-on"))
    set_if_any(FACTOR_INDEX["SPX"], -1.0, ("spx down", "equities down", "stocks down", "risk-off", "equity selloff"))
    set_if_any(FACTOR_INDEX["VIX"], 1.0, ("vix up", "volatility up", "volatility higher", "vol spike"))
    set_if_any(FACTOR_INDEX["VIX"], -1.0, ("vix down", "volatility compress", "volatility down", "vol compression"))
    set_if_any(FACTOR_INDEX["BBB_OAS"], 1.0, ("spreads wider", "credit wider", "credit spreads wider"))
    set_if_any(FACTOR_INDEX["BBB_OAS"], -1.0, ("spreads tighter", "credit tighter", "credit spreads tighter", "spreads stabilizing"))
    set_if_any(FACTOR_INDEX["DXY"], 1.0, ("dxy up", "dollar stronger", "dollar strong", "usd stronger"))
    set_if_any(FACTOR_INDEX["DXY"], -1.0, ("dxy down", "dollar weaker", "dollar weak", "usd weaker"))
    set_if_any(FACTOR_INDEX["Crude"], 1.0, ("crude up", "oil up", "crude higher", "oil higher"))
    set_if_any(FACTOR_INDEX["Crude"], -1.0, ("crude down", "oil down", "crude lower", "oil lower"))
    set_if_any(FACTOR_INDEX["Gold"], 1.0, ("gold up", "gold higher", "gold bid", "safe-haven gold"))
    set_if_any(FACTOR_INDEX["Gold"], -1.0, ("gold down", "gold lower"))
    set_if_any(FACTOR_INDEX["US10Y"], 1.0, ("us10y up", "10y up", "rates up", "yields higher", "rates selloff"))
    set_if_any(FACTOR_INDEX["US10Y"], -1.0, ("us10y down", "10y down", "rates down", "yields lower"))
    set_if_any(FACTOR_INDEX["IV_ATM_1Y"], 1.0, ("iv up", "implied vol up", "implied volatility up"))
    set_if_any(FACTOR_INDEX["IV_ATM_1Y"], -1.0, ("iv down", "implied vol down", "implied volatility down"))
    return direction


def direction_vector_from_grounding(
    grounding: dict[str, Any] | None,
    *,
    factor_count: int = 39,
    fallback_text: str | None = None,
) -> np.ndarray:
    """Extract a signed factor vector from grounding sidecar claims."""

    vector = np.zeros(int(factor_count), dtype=np.float32)
    for row in _implication_rows(_as_dict(grounding)):
        index = _market_index(row.get("market"))
        sign = _direction_sign(row.get("direction"))
        if index is not None and sign != 0.0:
            vector[index] = float(sign)
    if np.count_nonzero(vector) == 0 and fallback_text:
        vector = direction_vector_from_text(fallback_text, factor_count=factor_count)
    return vector


def apply_directional_delta_calibration(
    samples: np.ndarray,
    *,
    delta_scale: np.ndarray,
    direction_vector: np.ndarray,
    beta: float,
    alpha: float = 1.0,
    beta_bound: float = 0.35,
) -> np.ndarray:
    """Apply bounded direction-conditioned location/dispersion calibration."""

    arr = np.asarray(samples, dtype=np.float32)
    scale = np.asarray(delta_scale, dtype=np.float32)
    direction = np.asarray(direction_vector, dtype=np.float32).reshape(-1)
    if arr.ndim != 3:
        raise ValueError("samples must have shape [S,T,C]")
    if scale.shape != arr.shape[1:]:
        raise ValueError("delta_scale must have shape [T,C]")
    if direction.shape[0] != arr.shape[-1]:
        raise ValueError("direction_vector length must equal channel count")
    clipped_beta = float(np.clip(float(beta), -float(beta_bound), float(beta_bound)))
    mean = np.mean(arr, axis=0, keepdims=True)
    centered = mean + float(alpha) * (arr - mean)
    time = np.linspace(1.0 / arr.shape[1], 1.0, arr.shape[1], dtype=np.float32)
    shift = clipped_beta * time[:, None] * scale * direction[None, :]
    return (centered + shift[None, :, :]).astype(np.float32)


def path_with_start(
    states: np.ndarray,
    start: np.ndarray,
    *,
    factor_index: int,
) -> np.ndarray:
    """Return raw-level paths with the approved starting level prepended."""

    arr = np.asarray(states, dtype=np.float64)
    start_arr = np.asarray(start, dtype=np.float64).reshape(-1)
    if arr.ndim != 3:
        raise ValueError("states must have shape [S,T,C]")
    idx = int(factor_index)
    return np.concatenate(
        [
            np.full((arr.shape[0], 1), float(start_arr[idx]), dtype=np.float64),
            arr[:, :, idx],
        ],
        axis=1,
    )


def _factor_quantiles(
    result: dict[str, Any],
    factor: str,
    *,
    quantiles: tuple[float, ...] = (10.0, 50.0, 90.0),
) -> tuple[np.ndarray, ...]:
    paths = path_with_start(
        result["states"],
        result["start"],
        factor_index=FACTOR_INDEX[factor],
    )
    return tuple(np.percentile(paths, quantiles, axis=0))


def _case_label(case: str, result: dict[str, Any] | None = None) -> str:
    if result and result.get("case_label"):
        return str(result["case_label"])
    return str(CASE_SPECS.get(case, {}).get("label", case.replace("_", " ").title()))


def _case_color(case: str) -> str:
    return str(CASE_SPECS.get(case, {}).get("color", "#455A64"))


def _support_ids(result: dict[str, Any]) -> list[Any]:
    ids = []
    for row in _as_list(result.get("support")):
        support = _as_dict(row)
        if "window_index" in support:
            ids.append(support["window_index"])
        elif "bridge_index" in support:
            ids.append(support["bridge_index"])
        elif "window" in support:
            ids.append(support["window"])
    return ids


def _summarize_qualitative_response(
    *,
    candidate_cases: dict[str, dict[str, Any]],
    null_cases: dict[str, dict[str, Any]],
    relevant_factors: dict[str, tuple[str, ...]] = CASE_RELEVANT_FACTORS,
    reference_case: str = "fragile_risk_on",
) -> dict[str, Any]:
    """Summarize raw-level narrative response against a start-only null."""

    if reference_case not in candidate_cases:
        raise ValueError("reference_case must exist in candidate_cases")
    reference = candidate_cases[reference_case]
    case_rows = []
    max_abs_candidate_minus_null = 0.0
    max_abs_reference_contrast = 0.0
    for case, result in candidate_cases.items():
        factors = tuple(relevant_factors.get(case, SUMMARY_FACTORS[:4]))
        null = null_cases.get(case)
        factor_rows = []
        for factor in factors:
            idx = FACTOR_INDEX[factor]
            q10, q50, q90 = _factor_quantiles(result, factor)
            ref_q50 = _factor_quantiles(reference, factor)[1]
            null_q50 = _factor_quantiles(null, factor)[1] if null else q50
            start_level = float(np.asarray(result["start"], dtype=np.float64)[idx])
            terminal_median = float(q50[-1])
            candidate_minus_null = float(q50[-1] - null_q50[-1])
            reference_contrast = float(q50[-1] - ref_q50[-1])
            max_abs_candidate_minus_null = max(
                max_abs_candidate_minus_null, abs(candidate_minus_null)
            )
            max_abs_reference_contrast = max(
                max_abs_reference_contrast, abs(reference_contrast)
            )
            factor_rows.append(
                {
                    "factor": factor,
                    "start_level": start_level,
                    "terminal_p10": float(q10[-1]),
                    "terminal_median": terminal_median,
                    "terminal_p90": float(q90[-1]),
                    "terminal_median_vs_start": float(q50[-1] - start_level),
                    "terminal_median_vs_reference": reference_contrast,
                    "candidate_minus_null_terminal_median": candidate_minus_null,
                }
            )
        calibration = _as_dict(result.get("calibration"))
        case_rows.append(
            {
                "case": case,
                "label": _case_label(case, result),
                "factors": list(factors),
                "support_count": int(len(_as_list(result.get("support")))),
                "support_ids": _support_ids(result),
                "effective_beta": float(calibration.get("effective_beta", 0.0)),
                "support_evidence_gate": float(
                    calibration.get("support_evidence_gate", 1.0)
                ),
                "factor_summaries": factor_rows,
            }
        )
    return {
        "reference_case": reference_case,
        "case_relevant_factors": {
            key: list(value) for key, value in relevant_factors.items()
        },
        "headline": {
            "case_count": int(len(case_rows)),
            "max_abs_candidate_minus_null_terminal_median": float(
                max_abs_candidate_minus_null
            ),
            "max_abs_terminal_median_vs_reference": float(max_abs_reference_contrast),
        },
        "case_summaries": case_rows,
    }


def _render_qualitative_markdown(summary: dict[str, Any]) -> str:
    headline = _as_dict(summary.get("headline"))
    lines = [
        "# Support-Gated Qualitative Review",
        "",
        "This artifact inspects calibrated scenario fans in raw market levels and compares them with the start-only null control.",
        "",
        "## Headline",
        "",
        f"- Cases: `{int(headline.get('case_count', 0))}`",
        "- Max terminal median difference versus start-only null: "
        f"`{float(headline.get('max_abs_candidate_minus_null_terminal_median', 0.0)):.4f}`",
        "- Max terminal median difference versus reference narrative: "
        f"`{float(headline.get('max_abs_terminal_median_vs_reference', 0.0)):.4f}`",
        "",
        "## Narrative-Relevant Factors",
        "",
        "| Narrative | Support | Effective beta | Factor | Start level | Terminal median | Vs start-only null | Vs reference |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for case in _as_list(summary.get("case_summaries")):
        row = _as_dict(case)
        label = str(row.get("label", row.get("case", "")))
        support_count = int(row.get("support_count", 0))
        beta = float(row.get("effective_beta", 0.0))
        for factor_row in _as_list(row.get("factor_summaries")):
            factor = _as_dict(factor_row)
            lines.append(
                "| {label} | {support} | {beta:.3f} | {factor} | {start:.4f} | "
                "{terminal:.4f} | {null_delta:.4f} | {ref_delta:.4f} |".format(
                    label=label,
                    support=support_count,
                    beta=beta,
                    factor=str(factor.get("factor", "")),
                    start=float(factor.get("start_level", 0.0)),
                    terminal=float(factor.get("terminal_median", 0.0)),
                    null_delta=float(
                        factor.get("candidate_minus_null_terminal_median", 0.0)
                    ),
                    ref_delta=float(factor.get("terminal_median_vs_reference", 0.0)),
                )
            )
    return "\n".join(lines)


def _score_rows(
    rows: list[dict[str, Any]],
    *,
    beta: float,
    alpha: float,
    beta_bound: float,
) -> list[dict[str, Any]]:
    scored = []
    for idx, row in enumerate(rows):
        samples = np.asarray(row["samples"], dtype=np.float32)
        target = np.asarray(row["target"], dtype=np.float32)
        scale = np.asarray(row["delta_scale"], dtype=np.float32)
        direction = np.asarray(row["direction_vector"], dtype=np.float32)
        support_gate = float(row.get("support_evidence_gate", 1.0))
        calibrated = apply_directional_delta_calibration(
            samples,
            delta_scale=scale,
            direction_vector=direction,
            beta=float(beta) * support_gate,
            alpha=alpha,
            beta_bound=beta_bound,
        )
        methods = {
            "identity": score_sample_distribution(samples, target, scale=scale),
            "narrative_calibrated": score_sample_distribution(
                calibrated, target, scale=scale
            ),
        }
        if isinstance(row.get("persistence"), dict):
            methods["persistence"] = row["persistence"]
        scored.append({"row_no": int(row.get("row_no", idx)), "methods": methods})
    return scored


def _select_beta_candidate(
    candidates: list[dict[str, Any]],
    *,
    selection_objective: str,
    max_quality_regression_pct: float,
    min_coverage_delta: float,
) -> dict[str, Any]:
    if not candidates:
        raise ValueError("at least one beta candidate is required")
    if selection_objective == "quality":
        return min(
            candidates,
            key=lambda item: (
                float(item.get("ensemble_crps_z_mean", np.inf)),
                float(item.get("energy_score_z_mean", np.inf)),
                abs(float(item["beta"])),
            ),
        )
    if selection_objective != "quality_constrained_response":
        raise ValueError(f"unknown selection_objective {selection_objective!r}")

    feasible = []
    for candidate in candidates:
        identity = _as_dict(candidate.get("identity"))
        if not identity:
            continue
        identity_crps = float(identity.get("ensemble_crps_z_mean", np.inf))
        identity_energy = float(identity.get("energy_score_z_mean", np.inf))
        identity_coverage = float(identity.get("coverage_80_mean", -np.inf))
        crps = float(candidate.get("ensemble_crps_z_mean", np.inf))
        energy = float(candidate.get("energy_score_z_mean", np.inf))
        coverage = float(candidate.get("coverage_80_mean", -np.inf))
        if crps > identity_crps * (1.0 + float(max_quality_regression_pct)):
            continue
        if energy > identity_energy * (1.0 + float(max_quality_regression_pct)):
            continue
        if coverage < identity_coverage + float(min_coverage_delta):
            continue
        feasible.append(candidate)
    if not feasible:
        feasible = candidates
    return max(
        feasible,
        key=lambda item: (
            float(item["beta"]),
            -float(item.get("ensemble_crps_z_mean", np.inf)),
            -float(item.get("energy_score_z_mean", np.inf)),
        ),
    )


def fit_directional_beta(
    rows: list[dict[str, Any]],
    *,
    beta_grid: Iterable[float],
    alpha: float = 1.0,
    beta_bound: float = 0.35,
    selection_objective: str = "quality",
    max_quality_regression_pct: float = 0.02,
    min_coverage_delta: float = -0.03,
) -> dict[str, Any]:
    """Fit beta by historical backtest score on a calibration split."""

    candidates = []
    effective_betas = sorted(
        {
            float(np.clip(float(beta), -float(beta_bound), float(beta_bound)))
            for beta in beta_grid
        }
    )
    for beta in effective_betas:
        scored = _score_rows(
            rows,
            beta=float(beta),
            alpha=float(alpha),
            beta_bound=float(beta_bound),
        )
        summary = summarize_method_scores(scored, baseline="identity")
        block = dict(summary["narrative_calibrated"])
        block["beta"] = float(beta)
        block["identity"] = summary["identity"]
        candidates.append(block)
    selected = _select_beta_candidate(
        candidates,
        selection_objective=selection_objective,
        max_quality_regression_pct=max_quality_regression_pct,
        min_coverage_delta=min_coverage_delta,
    )
    return {
        "selected_beta": float(selected["beta"]),
        "selection_objective": str(selection_objective),
        "max_quality_regression_pct": float(max_quality_regression_pct),
        "min_coverage_delta": float(min_coverage_delta),
        "selected": selected,
        "candidates": candidates,
    }


def split_rows(
    rows: list[dict[str, Any]],
    *,
    split_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if len(rows) < 4:
        raise ValueError("at least four rows are required")
    if split_mode == "even_odd":
        return rows[::2], rows[1::2]
    if split_mode == "odd_even":
        return rows[1::2], rows[::2]
    if split_mode == "chronological":
        midpoint = max(1, len(rows) // 2)
        return rows[:midpoint], rows[midpoint:]
    raise ValueError(f"unknown split_mode {split_mode!r}")


def _target_delta_for_row(
    row: dict[str, Any],
    *,
    history_raw: np.ndarray,
    future_raw: np.ndarray,
) -> np.ndarray:
    start_idx = int(row.get("start_window_index", -1))
    if start_idx < 0 or start_idx >= future_raw.shape[0]:
        raise IndexError(f"start_window_index {start_idx} outside future_raw")
    start = np.asarray(history_raw[start_idx, -1, :], dtype=np.float32)
    future = np.asarray(future_raw[start_idx], dtype=np.float32)
    return (future - start[None, :]).astype(np.float32)


def _load_component_arrays(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    path = Path(str(row["artifacts"]["component_prefix_mixture"]["arrays"]))
    arrays = np.load(path)
    samples = np.asarray(arrays["samples"], dtype=np.float32)[0]
    scale = np.asarray(arrays["delta_scale"], dtype=np.float32)
    return samples, scale


def _story_report_context(
    row: dict[str, Any],
    *,
    support_gate_mode: str,
) -> tuple[np.ndarray, float]:
    report_path = Path(str(row["artifacts"]["component_prefix_mixture"]["report"]))
    report = _load_json(report_path)
    cached = _as_dict(report.get("cached_query"))
    grounding = cached.get("grounding")
    text = cached.get("query_text") or cached.get("narrative_text") or ""
    direction = direction_vector_from_grounding(
        _as_dict(grounding), factor_count=39, fallback_text=str(text)
    )
    support_gate = _support_evidence_gate_from_report(
        report, mode=support_gate_mode
    )
    return direction, support_gate


def load_backtest_training_rows(
    *,
    backtest_report: str | Path,
    checkpoint: str | Path,
    bridge_report: str | Path | None,
    device: str,
    support_gate_mode: str = "none",
) -> list[dict[str, Any]]:
    backtest = _load_json(backtest_report)
    rows = backtest.get("window_scores", [])
    if not isinstance(rows, list) or len(rows) < 4:
        raise ValueError("backtest report needs at least four window_scores")
    selected_bridge_report = str(backtest.get("bridge_report") or bridge_report or "")
    if not selected_bridge_report:
        raise ValueError("bridge_report must be supplied or recorded in backtest")
    block = _selected_history_block(
        checkpoint=str(backtest.get("checkpoint") or checkpoint),
        bridge_report=selected_bridge_report,
        device=torch.device(device),
    )
    history_raw = np.asarray(block["history_raw"], dtype=np.float32)
    future_raw = np.asarray(block["future_raw"], dtype=np.float32)
    out = []
    for pos, row in enumerate(rows):
        samples, scale = _load_component_arrays(row)
        direction, support_gate = _story_report_context(
            row, support_gate_mode=support_gate_mode
        )
        out.append(
            {
                "row_no": int(row.get("row_no", pos)),
                "window_id": str(row.get("window_id", "")),
                "samples": samples,
                "target": _target_delta_for_row(
                    row,
                    history_raw=history_raw,
                    future_raw=future_raw,
                ),
                "delta_scale": scale,
                "direction_vector": direction,
                "support_evidence_gate": float(support_gate),
                "persistence": row.get("methods", {}).get("persistence"),
            }
        )
    return out


def _score_eval_rows(
    rows: list[dict[str, Any]],
    *,
    beta: float,
    alpha: float,
    beta_bound: float,
    support_gate_mode: str = "none",
) -> dict[str, Any]:
    scored = _score_rows(rows, beta=beta, alpha=alpha, beta_bound=beta_bound)
    return {
        "window_scores": scored,
        "summary": summarize_method_scores(scored, baseline="persistence"),
    }


def _direction_from_case_report(result: dict[str, Any]) -> np.ndarray:
    report = _load_json(result["report_path"])
    cached = _as_dict(report.get("cached_query"))
    return direction_vector_from_grounding(
        _as_dict(cached.get("grounding")),
        factor_count=39,
        fallback_text=str(cached.get("query_text") or cached.get("narrative_text") or ""),
    )


def calibrate_case_result(
    result: dict[str, Any],
    *,
    beta: float,
    alpha: float,
    beta_bound: float,
    support_gate_mode: str = "none",
) -> dict[str, Any]:
    arrays = np.load(str(result["arrays_path"]))
    scale = np.asarray(arrays["delta_scale"], dtype=np.float32)
    start = np.asarray(result["start"], dtype=np.float32)
    states = np.asarray(result["states"], dtype=np.float32)
    samples = states - start[None, None, :]
    report = _load_json(result["report_path"])
    cached = _as_dict(report.get("cached_query"))
    direction = direction_vector_from_grounding(
        _as_dict(cached.get("grounding")),
        factor_count=39,
        fallback_text=str(cached.get("query_text") or cached.get("narrative_text") or ""),
    )
    support_gate = _support_evidence_gate_from_report(
        report, mode=support_gate_mode
    )
    calibrated_delta = apply_directional_delta_calibration(
        samples,
        delta_scale=scale,
        direction_vector=direction,
        beta=float(beta) * support_gate,
        alpha=alpha,
        beta_bound=beta_bound,
    )
    out = dict(result)
    out["states"] = (start[None, None, :] + calibrated_delta).astype(np.float32)
    out["portfolio_stats"] = _portfolio_stats(out["states"], start)
    out["calibration"] = {
        "beta": float(beta),
        "effective_beta": float(beta) * float(support_gate),
        "alpha": float(alpha),
        "active_direction_count": int(np.count_nonzero(direction)),
        "support_evidence_gate": float(support_gate),
    }
    return out


def _analyze_grid(
    grid: dict[str, dict[str, dict[str, Any]]],
    *,
    starts: tuple[str, ...],
    cases: tuple[str, ...],
    policy: str,
    feature_space: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    vectors = []
    feature_names: list[str] | None = None
    for start in starts:
        row = []
        for case in cases:
            vector, names = cell_feature_vector(
                grid[start][case], feature_space=feature_space
            )
            feature_names = names if feature_names is None else feature_names
            row.append(vector)
        vectors.append(row)
    cube = np.asarray(vectors, dtype=np.float64)
    decomp = two_way_feature_attribution(cube)
    feature_dist = _pairwise_feature_distances(cube)
    dist = _pairwise_distribution_metrics(grid, starts=starts, cases=cases)
    result = {
        "feature_space": feature_space,
        "policy": policy,
        "start_share": float(decomp["start_share"]),
        "narrative_share": float(decomp["narrative_share"]),
        "interaction_share": float(decomp["interaction_share"]),
        "total_ss": float(decomp["total_ss"]),
        "mean_feature_distance_same_start_narrative": float(
            feature_dist["same_start_narrative"]
        ),
        "mean_feature_distance_same_narrative_start": float(
            feature_dist["same_narrative_start"]
        ),
        **dist,
    }
    details = {
        "starts": list(starts),
        "cases": list(cases),
        "feature_names": feature_names or [],
        "decomposition": decomp,
        "pairwise_feature_distances": feature_dist,
        "pairwise_distribution_metrics": dist,
    }
    return result, details


def _per_start_narrative_metrics(
    grid: dict[str, dict[str, dict[str, Any]]],
    *,
    starts: tuple[str, ...],
    cases: tuple[str, ...],
) -> dict[str, Any]:
    """Report narrative separation separately for each fixed start."""

    rows = []
    for start in starts:
        factor_ks_values = []
        portfolio_ks_values = []
        support_jaccard_values = []
        for i, left_case in enumerate(cases):
            for right_case in cases[i + 1 :]:
                left = grid[start][left_case]
                right = grid[start][right_case]
                left_pnl = _portfolio_pnl(left["states"], left["start"])[:, -1]
                right_pnl = _portfolio_pnl(right["states"], right["start"])[:, -1]
                portfolio_ks_values.append(_ks_statistic(left_pnl, right_pnl))
                factor_ks = []
                for factor in SUMMARY_FACTORS:
                    idx = FACTOR_INDEX[factor]
                    factor_ks.append(
                        _ks_statistic(
                            np.asarray(left["states"])[:, -1, idx],
                            np.asarray(right["states"])[:, -1, idx],
                        )
                    )
                factor_ks_values.append(float(np.mean(factor_ks)))
                support_jaccard_values.append(
                    _support_jaccard(left["support"], right["support"])
                )
        rows.append(
            {
                "start": start,
                "pair_count": int(len(factor_ks_values)),
                "mean_factor_terminal_ks": float(np.mean(factor_ks_values))
                if factor_ks_values
                else 0.0,
                "mean_portfolio_terminal_ks": float(np.mean(portfolio_ks_values))
                if portfolio_ks_values
                else 0.0,
                "mean_support_jaccard": float(np.mean(support_jaccard_values))
                if support_jaccard_values
                else 0.0,
            }
        )
    return {
        "rows": rows,
        "min_mean_factor_terminal_ks": float(
            min((row["mean_factor_terminal_ks"] for row in rows), default=0.0)
        ),
        "min_mean_portfolio_terminal_ks": float(
            min((row["mean_portfolio_terminal_ks"] for row in rows), default=0.0)
        ),
        "max_mean_support_jaccard": float(
            max((row["mean_support_jaccard"] for row in rows), default=0.0)
        ),
    }


def _per_start_promotion_gates(
    per_start_metrics: dict[str, Any],
    *,
    min_factor_ks: float = 0.20,
    min_portfolio_ks: float = 0.20,
    max_support_jaccard: float = 0.25,
) -> dict[str, bool]:
    """Return broad fixed-start gates for candidate promotion checks."""

    return {
        "per_start_factor_ks_min_ge_0p20": bool(
            float(per_start_metrics.get("min_mean_factor_terminal_ks", 0.0))
            >= float(min_factor_ks)
        ),
        "per_start_portfolio_ks_min_ge_0p20": bool(
            float(per_start_metrics.get("min_mean_portfolio_terminal_ks", 0.0))
            >= float(min_portfolio_ks)
        ),
        "per_start_support_jaccard_max_le_0p25": bool(
            float(per_start_metrics.get("max_mean_support_jaccard", 1.0))
            <= float(max_support_jaccard)
        ),
    }


def run_fixed_start_attribution(
    *,
    start_roots: dict[str, str],
    cases: tuple[str, ...],
    policy: str,
    beta: float,
    alpha: float,
    beta_bound: float,
    support_gate_mode: str = "none",
) -> dict[str, Any]:
    starts = tuple(start_roots)
    baseline_results = []
    calibrated_results = []
    details: dict[str, Any] = {"baseline": {}, "calibrated": {}}
    calibrated_grid: dict[str, dict[str, dict[str, Any]]] = {}
    for start, root in start_roots.items():
        calibrated_grid[start] = {}
        for case in cases:
            loaded = load_case_result(
                output_root=root,
                case_name=case,
                policy_name=policy,
            )
            calibrated_grid[start][case] = calibrate_case_result(
                loaded,
                beta=beta,
                alpha=alpha,
                beta_bound=beta_bound,
                support_gate_mode=support_gate_mode,
            )
    for feature_space in ("raw_level", "start_normalized"):
        baseline, baseline_detail = analyze_policy(
            start_roots=start_roots,
            cases=cases,
            policy=policy,
            feature_space=feature_space,
        )
        calibrated, calibrated_detail = _analyze_grid(
            calibrated_grid,
            starts=starts,
            cases=cases,
            policy=f"{policy}_narrative_calibrated",
            feature_space=feature_space,
        )
        baseline_results.append(_result_to_dict(baseline))
        calibrated_results.append(calibrated)
        details["baseline"][feature_space] = baseline_detail
        details["calibrated"][feature_space] = calibrated_detail
    return {
        "starts": list(starts),
        "cases": list(cases),
        "policy": policy,
        "baseline_results": baseline_results,
        "calibrated_results": calibrated_results,
        "per_start_narrative_metrics": _per_start_narrative_metrics(
            calibrated_grid,
            starts=starts,
            cases=cases,
        ),
        "details": details,
    }


def _render_markdown(payload: dict[str, Any]) -> str:
    eval_summary = payload["heldout_quality"]["evaluation_summary"]
    identity = eval_summary["identity"]
    calibrated = eval_summary["narrative_calibrated"]
    fixed = payload["fixed_start_attribution"]
    rows = []
    for row in fixed["calibrated_results"]:
        if row["feature_space"] == "start_normalized":
            rows.append(row)
    lines = [
        "# Narrative-Conditioned Ensemble Calibration TestFlight",
        "",
        "This TestFlight keeps the incumbent support-grounded SNI rollout and fits one bounded narrative-direction calibration parameter on historical backtest rows.",
        "",
        f"- Selected beta: `{payload['calibration_fit']['selected_beta']:.4f}`",
        f"- Evaluation CRPS identity/calibrated: `{identity.get('ensemble_crps_z_mean')}` / `{calibrated.get('ensemble_crps_z_mean')}`",
        f"- Evaluation energy identity/calibrated: `{identity.get('energy_score_z_mean')}` / `{calibrated.get('energy_score_z_mean')}`",
        f"- Evaluation coverage identity/calibrated: `{identity.get('coverage_80_mean')}` / `{calibrated.get('coverage_80_mean')}`",
        "",
        "## Fixed-Start Attribution",
        "",
        "| Policy | Feature space | Start share | Narrative share | Interaction | Factor KS same start | Portfolio KS same start |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in fixed["baseline_results"] + fixed["calibrated_results"]:
        lines.append(
            "| {policy} | {space} | {start:.1%} | {narr:.1%} | {inter:.1%} | {factor:.3f} | {portfolio:.3f} |".format(
                policy=row["policy"],
                space=row["feature_space"],
                start=row["start_share"],
                narr=row["narrative_share"],
                inter=row["interaction_share"],
                factor=row["mean_factor_ks_same_start_narrative"],
                portfolio=row["mean_portfolio_ks_same_start_narrative"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-Start Narrative Gate",
            "",
            "| Start | Factor KS | Portfolio KS | Support Jaccard |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in _as_list(fixed.get("per_start_narrative_metrics", {}).get("rows")):
        item = _as_dict(row)
        lines.append(
            "| {start} | {factor:.3f} | {portfolio:.3f} | {support:.3f} |".format(
                start=str(item.get("start", "")),
                factor=float(item.get("mean_factor_terminal_ks", 0.0)),
                portfolio=float(item.get("mean_portfolio_terminal_ks", 0.0)),
                support=float(item.get("mean_support_jaccard", 0.0)),
            )
        )
    lines.extend(
        [
            "",
            "## Mistake Gates",
            "",
            "- This is not a support-ranker-only variant.",
            "- The support mixture and frozen SNI rollout remain the base ensemble.",
            "- Promotion still requires independent verification and stronger qualitative plots.",
        ]
    )
    return "\n".join(lines)


def plot_fixed_start_calibrated_fans(
    *,
    start_root: str | Path,
    cases: tuple[str, ...],
    policy: str,
    beta: float,
    alpha: float,
    beta_bound: float,
    output_path: str | Path,
    support_gate_mode: str = "none",
    factors: tuple[str, ...] = ("SPX", "VIX", "Crude", "Gold"),
) -> None:
    """Plot baseline versus calibrated raw-level fans for one fixed start."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(factors), 1, figsize=(9.5, 2.4 * len(factors)), sharex=True)
    if len(factors) == 1:
        axes = [axes]
    colors = ["#1565C0", "#C62828", "#EF6C00", "#6A1B9A", "#00838F", "#2E7D32"]
    for ax, factor in zip(axes, factors, strict=True):
        idx = FACTOR_INDEX[factor]
        for pos, case in enumerate(cases):
            base = load_case_result(
                output_root=start_root,
                case_name=case,
                policy_name=policy,
            )
            cal = calibrate_case_result(
                base,
                beta=beta,
                alpha=alpha,
                beta_bound=beta_bound,
                support_gate_mode=support_gate_mode,
            )
            color = colors[pos % len(colors)]
            base_path = path_with_start(base["states"], base["start"], factor_index=idx)
            cal_path = path_with_start(cal["states"], cal["start"], factor_index=idx)
            x = np.arange(base_path.shape[1])
            base_q10, base_q50, base_q90 = np.percentile(base_path, [10, 50, 90], axis=0)
            cal_q10, cal_q50, cal_q90 = np.percentile(cal_path, [10, 50, 90], axis=0)
            if pos == 0:
                ax.fill_between(
                    x,
                    base_q10,
                    base_q90,
                    color="#9E9E9E",
                    alpha=0.12,
                    label="Incumbent fan",
                )
                ax.plot(x, base_q50, color="#616161", linewidth=1.0, linestyle="--", label="Incumbent median")
            ax.fill_between(x, cal_q10, cal_q90, color=color, alpha=0.07)
            ax.plot(x, cal_q50, color=color, linewidth=1.35, label=case.replace("_", " "))
        ax.set_title(f"{factor} raw-level fan after narrative calibration")
        ax.grid(alpha=0.18)
        ax.set_ylabel("raw level")
    axes[-1].set_xlabel("days from approved start")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center")
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_qualitative_response_panels(
    *,
    candidate_cases: dict[str, dict[str, Any]],
    null_cases: dict[str, dict[str, Any]],
    output_path: str | Path,
    relevant_factors: dict[str, tuple[str, ...]] = CASE_RELEVANT_FACTORS,
    reference_case: str = "fragile_risk_on",
) -> None:
    """Plot raw-level narrative-relevant fans with start-only null medians."""

    import matplotlib.pyplot as plt

    cases = list(candidate_cases)
    if not cases:
        raise ValueError("candidate_cases must not be empty")
    n_cols = max(len(relevant_factors.get(case, SUMMARY_FACTORS[:4])) for case in cases)
    fig, axes = plt.subplots(
        len(cases),
        n_cols,
        figsize=(4.1 * n_cols, 2.25 * len(cases)),
        squeeze=False,
    )
    fig.suptitle(
        "Support-gated narrative response in raw market levels",
        fontsize=13,
        fontweight="bold",
    )
    reference = candidate_cases.get(reference_case)
    for row_no, case in enumerate(cases):
        result = candidate_cases[case]
        null = null_cases.get(case)
        factors = tuple(relevant_factors.get(case, SUMMARY_FACTORS[:4]))
        color = _case_color(case)
        for col_no in range(n_cols):
            ax = axes[row_no, col_no]
            if col_no >= len(factors):
                ax.axis("off")
                continue
            factor = factors[col_no]
            q10, q50, q90 = _factor_quantiles(result, factor)
            x = np.arange(q50.shape[0])
            ax.fill_between(x, q10, q90, color=color, alpha=0.20)
            ax.plot(x, q50, color=color, linewidth=1.7, label="candidate")
            ax.scatter([0], [q50[0]], color="#111111", s=12, zorder=3)
            if null is not None:
                null_q50 = _factor_quantiles(null, factor)[1]
                ax.plot(
                    x,
                    null_q50,
                    color="#757575",
                    linewidth=0.95,
                    linestyle=":",
                    label="start-only null" if row_no == 0 and col_no == 0 else None,
                )
            if reference is not None and case != reference_case:
                ref_q50 = _factor_quantiles(reference, factor)[1]
                ax.plot(
                    x,
                    ref_q50,
                    color="#424242",
                    linewidth=0.9,
                    linestyle="--",
                    alpha=0.85,
                    label="reference" if row_no == 0 and col_no == 0 else None,
                )
            if row_no == 0:
                ax.set_title(factor, fontsize=9, fontweight="bold")
            if col_no == 0:
                ax.set_ylabel(_case_label(case, result), fontsize=8)
            ax.grid(alpha=0.16)
            ax.set_xlim(0, q50.shape[0] - 1)
    axes[-1, 0].set_xlabel("forward day from fixed start")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, ncol=3, loc="lower center")
        bottom = 0.06
    else:
        bottom = 0.01
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, bottom, 1, 0.96))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_qualitative_null_contrasts(
    *,
    candidate_cases: dict[str, dict[str, Any]],
    null_cases: dict[str, dict[str, Any]],
    output_path: str | Path,
    relevant_factors: dict[str, tuple[str, ...]] = CASE_RELEVANT_FACTORS,
) -> None:
    """Plot raw-level quantile contrasts versus start-only null."""

    import matplotlib.pyplot as plt

    cases = [case for case in candidate_cases if case in null_cases]
    if not cases:
        raise ValueError("at least one candidate case needs a null case")
    n_cols = max(len(relevant_factors.get(case, SUMMARY_FACTORS[:4])) for case in cases)
    fig, axes = plt.subplots(
        len(cases),
        n_cols,
        figsize=(4.1 * n_cols, 2.1 * len(cases)),
        squeeze=False,
    )
    fig.suptitle(
        "Raw-level quantile contrast versus start-only null",
        fontsize=13,
        fontweight="bold",
    )
    for row_no, case in enumerate(cases):
        result = candidate_cases[case]
        null = null_cases[case]
        factors = tuple(relevant_factors.get(case, SUMMARY_FACTORS[:4]))
        color = _case_color(case)
        for col_no in range(n_cols):
            ax = axes[row_no, col_no]
            if col_no >= len(factors):
                ax.axis("off")
                continue
            factor = factors[col_no]
            q10, q50, q90 = _factor_quantiles(result, factor)
            n10, n50, n90 = _factor_quantiles(null, factor)
            x = np.arange(q50.shape[0])
            ax.fill_between(x, q10 - n10, q90 - n90, color=color, alpha=0.18)
            ax.plot(x, q50 - n50, color=color, linewidth=1.6)
            ax.axhline(0.0, color="#757575", linestyle=":", linewidth=0.8)
            if row_no == 0:
                ax.set_title(factor, fontsize=9, fontweight="bold")
            if col_no == 0:
                ax.set_ylabel(_case_label(case, result), fontsize=8)
            ax.grid(alpha=0.16)
            ax.set_xlim(0, q50.shape[0] - 1)
    axes[-1, 0].set_xlabel("forward day from fixed start")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_qualitative_review_artifacts(
    *,
    start_root: str | Path,
    cases: tuple[str, ...],
    policy: str,
    beta: float,
    alpha: float,
    beta_bound: float,
    output_dir: str | Path,
    support_gate_mode: str = "none",
    null_policy: str = "start_only_topk",
    reference_case: str = "fragile_risk_on",
) -> dict[str, Any]:
    """Write support-gated qualitative review plots and summaries."""

    candidate_cases = {}
    null_cases = {}
    for case in cases:
        loaded = load_case_result(
            output_root=start_root,
            case_name=case,
            policy_name=policy,
        )
        candidate_cases[case] = calibrate_case_result(
            loaded,
            beta=beta,
            alpha=alpha,
            beta_bound=beta_bound,
            support_gate_mode=support_gate_mode,
        )
        null_loaded = load_case_result(
            output_root=start_root,
            case_name=case,
            policy_name=null_policy,
        )
        null_cases[case] = calibrate_case_result(
            null_loaded,
            beta=beta,
            alpha=alpha,
            beta_bound=beta_bound,
            support_gate_mode=support_gate_mode,
        )
    summary = _summarize_qualitative_response(
        candidate_cases=candidate_cases,
        null_cases=null_cases,
        reference_case=reference_case,
    )
    out = Path(output_dir)
    artifact_paths = {
        "qualitative_review_json": str(out / "support_gated_qualitative_review.json"),
        "qualitative_review_markdown": str(
            out / "support_gated_qualitative_review.md"
        ),
        "narrative_relevant_raw_panels": str(
            out / "support_gated_narrative_relevant_raw_panels.png"
        ),
        "start_only_null_contrasts": str(
            out / "support_gated_start_only_null_contrasts.png"
        ),
    }
    summary["artifact_paths"] = artifact_paths
    plot_qualitative_response_panels(
        candidate_cases=candidate_cases,
        null_cases=null_cases,
        output_path=artifact_paths["narrative_relevant_raw_panels"],
        reference_case=reference_case,
    )
    plot_qualitative_null_contrasts(
        candidate_cases=candidate_cases,
        null_cases=null_cases,
        output_path=artifact_paths["start_only_null_contrasts"],
    )
    _write_json(artifact_paths["qualitative_review_json"], summary)
    _write_text(
        artifact_paths["qualitative_review_markdown"],
        _render_qualitative_markdown(summary),
    )
    return summary


def run_testflight(args: argparse.Namespace) -> dict[str, Any]:
    start_roots = _resolve_start_roots(args.start_root)
    qualitative_start_label = (
        "start22" if "start22" in start_roots else next(iter(start_roots))
    )
    rows = load_backtest_training_rows(
        backtest_report=args.backtest_report,
        checkpoint=args.checkpoint,
        bridge_report=args.bridge_report,
        device=args.device,
        support_gate_mode=str(args.support_gate_mode),
    )
    calibration_rows, evaluation_rows = split_rows(rows, split_mode=args.split_mode)
    beta_grid = [float(value) for value in str(args.beta_grid).split(",") if value.strip()]
    fit = fit_directional_beta(
        calibration_rows,
        beta_grid=beta_grid,
        alpha=float(args.alpha),
        beta_bound=float(args.beta_bound),
        selection_objective=str(args.selection_objective),
        max_quality_regression_pct=float(args.max_quality_regression_pct),
        min_coverage_delta=float(args.min_coverage_delta),
    )
    selected_beta = float(fit["selected_beta"])
    evaluation = _score_eval_rows(
        evaluation_rows,
        beta=selected_beta,
        alpha=float(args.alpha),
        beta_bound=float(args.beta_bound),
        support_gate_mode=str(args.support_gate_mode),
    )
    fixed_start = run_fixed_start_attribution(
        start_roots=start_roots,
        cases=tuple(args.case or CASE_NAMES),
        policy=str(args.policy),
        beta=selected_beta,
        alpha=float(args.alpha),
        beta_bound=float(args.beta_bound),
        support_gate_mode=str(args.support_gate_mode),
    )
    eval_summary = evaluation["summary"]
    identity = eval_summary["identity"]
    calibrated = eval_summary["narrative_calibrated"]
    crps_regression = float(calibrated["ensemble_crps_z_mean"] - identity["ensemble_crps_z_mean"])
    energy_regression = float(calibrated["energy_score_z_mean"] - identity["energy_score_z_mean"])
    coverage_delta = float(calibrated["coverage_80_mean"] - identity["coverage_80_mean"])
    baseline_norm = next(
        row
        for row in fixed_start["baseline_results"]
        if row["feature_space"] == "start_normalized"
    )
    calibrated_norm = next(
        row
        for row in fixed_start["calibrated_results"]
        if row["feature_space"] == "start_normalized"
    )
    baseline_share = float(baseline_norm["narrative_share"] + baseline_norm["interaction_share"])
    calibrated_share = float(
        calibrated_norm["narrative_share"] + calibrated_norm["interaction_share"]
    )
    gates = {
        "crps_regression_within_2pct": bool(
            crps_regression <= 0.02 * float(identity["ensemble_crps_z_mean"])
        ),
        "energy_regression_within_2pct": bool(
            energy_regression <= 0.02 * float(identity["energy_score_z_mean"])
        ),
        "coverage_not_down_more_than_0p03": bool(coverage_delta >= -0.03),
        "start_normalized_narrative_share_improved": bool(
            calibrated_share > baseline_share
        ),
        "selected_beta_nonzero": bool(abs(selected_beta) > 1.0e-12),
    }
    gates.update(
        _per_start_promotion_gates(fixed_start["per_start_narrative_metrics"])
    )
    status = "candidate" if all(gates.values()) else "diagnostic"
    payload = {
        "status": status,
        "research_lane": "candidate",
        "result_status": "candidate" if status == "candidate" else "candidate_rejected",
        "scope_note": (
            "Offline bounded TestFlight. The incumbent support-grounded SNI "
            "ensemble is not replaced; a single narrative-direction calibration "
            "parameter is fit on calibration rows and evaluated on held-out rows."
        ),
        "backtest_report": str(args.backtest_report),
        "split_mode": str(args.split_mode),
        "selection_objective": str(args.selection_objective),
        "max_quality_regression_pct": float(args.max_quality_regression_pct),
        "min_coverage_delta": float(args.min_coverage_delta),
        "support_gate_mode": str(args.support_gate_mode),
        "start_roots": start_roots,
        "qualitative_start_label": qualitative_start_label,
        "alpha": float(args.alpha),
        "beta_bound": float(args.beta_bound),
        "calibration_row_count": int(len(calibration_rows)),
        "evaluation_row_count": int(len(evaluation_rows)),
        "calibration_fit": fit,
        "heldout_quality": {
            "evaluation_summary": eval_summary,
            "comparison": {
                "calibrated_minus_identity_crps": crps_regression,
                "calibrated_minus_identity_energy": energy_regression,
                "calibrated_minus_identity_coverage": coverage_delta,
            },
        },
        "fixed_start_attribution": fixed_start,
        "promotion_gates": gates,
        "mistake_gate_recap": [
            "No cached-only promotion: this uses split historical rows and remains diagnostic until independently verified.",
            "No ranker-only variant: support selection is unchanged; calibration is applied after incumbent rollout.",
            "No single-metric promotion: CRPS, energy, coverage, attribution, and fixed-start metrics are reported together.",
            "No hidden start: fixed-start attribution uses the resolved explicit start roots recorded in this report.",
        ],
    }
    output_dir = Path(args.output_dir)
    report_path = output_dir / "narrative_ensemble_calibration_report.json"
    md_path = output_dir / "narrative_ensemble_calibration_report.md"
    fan_path = output_dir / "fixed_start_calibrated_factor_fans.png"
    plot_fixed_start_calibrated_fans(
        start_root=start_roots[qualitative_start_label],
        cases=tuple(args.case or CASE_NAMES),
        policy=str(args.policy),
        beta=selected_beta,
        alpha=float(args.alpha),
        beta_bound=float(args.beta_bound),
        output_path=fan_path,
        support_gate_mode=str(args.support_gate_mode),
    )
    qualitative = write_qualitative_review_artifacts(
        start_root=start_roots[qualitative_start_label],
        cases=tuple(args.case or CASE_NAMES),
        policy=str(args.policy),
        beta=selected_beta,
        alpha=float(args.alpha),
        beta_bound=float(args.beta_bound),
        output_dir=output_dir,
        support_gate_mode=str(args.support_gate_mode),
        null_policy="start_only_topk",
    )
    _write_json(report_path, payload)
    _write_text(md_path, _render_markdown(payload))
    payload["artifact_paths"] = {
        "json": str(report_path),
        "markdown": str(md_path),
        "fixed_start_fans": str(fan_path),
        **qualitative["artifact_paths"],
    }
    payload["qualitative_review"] = {
        "headline": qualitative["headline"],
        "case_relevant_factors": qualitative["case_relevant_factors"],
        "reference_case": qualitative["reference_case"],
    }
    _write_json(report_path, payload)
    if not args.no_paper:
        class ResultProxy:
            def __init__(self, row: dict[str, Any]) -> None:
                self.__dict__.update(row)

        table_rows = [ResultProxy(row) for row in fixed_start["calibrated_results"]]
        _write_text(args.paper_table, render_latex_table(table_rows))
        shutil.copyfile(
            DEFAULT_ATTRIBUTION_PAPER_FIGURE,
            args.paper_figure,
        )
        payload["paper_artifacts"] = {
            "table": str(args.paper_table),
            "figure": str(args.paper_figure),
        }
        _write_json(report_path, payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backtest-report", default=DEFAULT_BACKTEST_REPORT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--bridge-report", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--split-mode", choices=["even_odd", "odd_even", "chronological"], default="even_odd")
    parser.add_argument(
        "--selection-objective",
        choices=["quality", "quality_constrained_response"],
        default="quality",
    )
    parser.add_argument("--max-quality-regression-pct", type=float, default=0.02)
    parser.add_argument("--min-coverage-delta", type=float, default=-0.03)
    parser.add_argument(
        "--support-gate-mode",
        choices=["none", "direction_status"],
        default="none",
    )
    parser.add_argument("--beta-grid", default="0.0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.3")
    parser.add_argument("--beta-bound", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--policy", default="current_start_checked_gap30")
    parser.add_argument("--case", action="append")
    parser.add_argument(
        "--start-root",
        action="append",
        help=(
            "LABEL=rollout-root. Use this to run fixed-start attribution and "
            "qualitative plots against a matched rollout deck instead of the "
            "default current-support roots."
        ),
    )
    parser.add_argument("--no-paper", action="store_true")
    parser.add_argument("--paper-table", default=DEFAULT_PAPER_TABLE)
    parser.add_argument("--paper-figure", default=DEFAULT_PAPER_FIGURE)
    args = parser.parse_args()
    payload = run_testflight(args)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "selected_beta": payload["calibration_fit"]["selected_beta"],
                "support_gate_mode": payload["support_gate_mode"],
                "heldout_quality": payload["heldout_quality"]["comparison"],
                "promotion_gates": payload["promotion_gates"],
                "artifact_paths": payload["artifact_paths"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
