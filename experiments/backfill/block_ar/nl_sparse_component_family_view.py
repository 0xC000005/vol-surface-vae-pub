#!/usr/bin/env python
"""Build sparse component-family views for narrative-conditioned live decks.

This artifact-only utility consumes saved component-preserving rollout arrays.
It does not call OpenAI, train a model, or rerun the frozen SNI generator.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_component_pooling_diagnostic import (  # noqa: E402
    component_slices_for_variant,
)

DEFAULT_SUMMARY = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_gradio_live_story_multistart_950c_5start_default_deck_warning_aware/"
    "start_22/gradio_live_api_casebook_summary.json"
)
DEFAULT_OUTPUT = DEFAULT_SUMMARY.parent / "sparse_component_family_view.json"
DEFAULT_PLOT_RAW = DEFAULT_SUMMARY.parent / "sparse_component_family_raw_levels.png"
DEFAULT_PLOT_STD = (
    DEFAULT_SUMMARY.parent / "sparse_component_family_standardized_moves.png"
)
DEFAULT_MARKETS: list[tuple[str, int]] = [
    ("SPX", 25),
    ("VIX", 38),
    ("BBB_OAS", 35),
    ("US10Y", 33),
    ("DXY", 28),
    ("GOLD", 37),
    ("CRUDE_OIL", 31),
]
COLORS = [
    "#1565C0",
    "#EF6C00",
    "#6A1B9A",
    "#2E7D32",
    "#C62828",
    "#00838F",
    "#5D4037",
]


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


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
        return value if math.isfinite(value) else None
    return value


def _case_label(case_name: str) -> str:
    text = str(case_name)
    parts = text.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        text = parts[0]
    text = text.replace("_", " ").strip().capitalize()
    return text.replace("risk on", "risk-on").replace("risk off", "risk-off")


def _variant_index(report: dict[str, Any], variant_count: int) -> int:
    generation = report.get("generation")
    calibration = (
        generation.get("narrative_ensemble_calibration")
        if isinstance(generation, dict)
        else {}
    )
    value = (
        calibration.get("operational_variant_index")
        if isinstance(calibration, dict)
        else None
    )
    if value is None:
        selected = report.get("selected_start_state")
        value = selected.get("variant_index") if isinstance(selected, dict) else None
    try:
        index = int(value)
    except (TypeError, ValueError):
        index = variant_count - 1
    if not 0 <= index < variant_count:
        raise IndexError(f"variant index {index} outside 0..{variant_count - 1}")
    return index


def _quantile_path(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "days": list(range(int(values.shape[1]))),
        "p10": np.nanpercentile(values, 10, axis=0).astype(float).tolist(),
        "p50": np.nanpercentile(values, 50, axis=0).astype(float).tolist(),
        "p90": np.nanpercentile(values, 90, axis=0).astype(float).tolist(),
    }


def _terminal_quantiles(values: np.ndarray) -> dict[str, float]:
    terminal = values[:, -1]
    return {
        "p10": float(np.nanpercentile(terminal, 10)),
        "p50": float(np.nanpercentile(terminal, 50)),
        "p90": float(np.nanpercentile(terminal, 90)),
    }


def _prepend_day0(paths: np.ndarray, day0: np.ndarray) -> np.ndarray:
    head = np.broadcast_to(day0[None, None, :], (paths.shape[0], 1, paths.shape[2]))
    return np.concatenate([head, paths], axis=1)


def select_sparse_components(
    components: list[dict[str, Any]],
    *,
    max_components: int = 2,
    min_cumulative_weight: float = 0.80,
) -> list[dict[str, Any]]:
    """Select a small support family and renormalize its weights."""

    if max_components < 1:
        raise ValueError("max_components must be at least 1")
    ranked = sorted(
        components,
        key=lambda item: (float(item.get("weight", 0.0)), -int(item.get("component_no", 0))),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    cumulative = 0.0
    for component in ranked:
        if len(selected) >= int(max_components):
            break
        selected.append(dict(component))
        cumulative += max(float(component.get("weight", 0.0)), 0.0)
        if cumulative >= float(min_cumulative_weight):
            break
    total = sum(max(float(item.get("weight", 0.0)), 0.0) for item in selected)
    if total <= 0.0 and selected:
        total = float(len(selected))
        for item in selected:
            item["sparse_weight"] = 1.0 / total
    else:
        for item in selected:
            item["sparse_weight"] = max(float(item.get("weight", 0.0)), 0.0) / total
    return sorted(selected, key=lambda item: int(item.get("component_no", 0)))


def _load_case(
    case: dict[str, Any],
    *,
    markets: list[tuple[str, int]],
    max_components: int,
    min_cumulative_weight: float,
) -> dict[str, Any]:
    arrays_path = Path(str(case.get("prefix_arrays_snapshot_path", "")))
    report_path = Path(str(case.get("prefix_report_snapshot_path", "")))
    if not arrays_path.exists():
        raise FileNotFoundError(arrays_path)
    if not report_path.exists():
        raise FileNotFoundError(report_path)

    report = _load_json(report_path)
    with np.load(arrays_path, allow_pickle=True) as arrays:
        samples = np.asarray(arrays["samples"], dtype=np.float64)
        generated = np.asarray(arrays["generated_states"], dtype=np.float64)
        requested_raw = np.asarray(arrays["requested_raw"], dtype=np.float64)
        scale = np.maximum(
            np.abs(np.asarray(arrays["delta_scale"], dtype=np.float64)), 1e-8
        )
        variant = _variant_index(report, int(samples.shape[0]))
        moves_v = samples[variant]
        states_v = generated[variant]
        start_v = requested_raw[variant]
        standardized_v = moves_v / scale[None, :, :]
        components = component_slices_for_variant(
            variant_index=variant,
            component_variant_index=np.asarray(
                arrays["rollout_component_variant_index"], dtype=np.int64
            ),
            component_window_index=np.asarray(
                arrays["rollout_component_window_index"], dtype=np.int64
            ),
            component_weight=np.asarray(
                arrays["rollout_component_weight"], dtype=np.float64
            ),
            component_sample_count=np.asarray(
                arrays["rollout_component_sample_count"], dtype=np.int64
            ),
            sample_count=int(samples.shape[1]),
        )

    component_rows: list[dict[str, Any]] = []
    for component_no, row in enumerate(components):
        start, stop = row["sample_slice"]
        raw = _prepend_day0(states_v[start:stop], start_v)
        standardized = _prepend_day0(
            standardized_v[start:stop], np.zeros_like(start_v)
        )
        component_rows.append(
            {
                **row,
                "component_no": int(component_no),
                "_raw_level": raw,
                "_standardized_move": standardized,
            }
        )

    selected = select_sparse_components(
        component_rows,
        max_components=max_components,
        min_cumulative_weight=min_cumulative_weight,
    )
    selected_numbers = {int(item["component_no"]) for item in selected}
    selected_components = [
        component for component in component_rows if component["component_no"] in selected_numbers
    ]
    raw_sparse = np.concatenate(
        [component["_raw_level"] for component in selected_components], axis=0
    )
    standardized_sparse = np.concatenate(
        [component["_standardized_move"] for component in selected_components], axis=0
    )
    full_raw = _prepend_day0(states_v, start_v)
    full_standardized = _prepend_day0(standardized_v, np.zeros_like(start_v))

    sparse_by_no = {int(item["component_no"]): item for item in selected}
    public_components: list[dict[str, Any]] = []
    for component in selected_components:
        sparse_meta = sparse_by_no[int(component["component_no"])]
        terminal_raw: dict[str, dict[str, float]] = {}
        terminal_standardized: dict[str, dict[str, float]] = {}
        fan_raw: dict[str, dict[str, list[float]]] = {}
        fan_standardized: dict[str, dict[str, list[float]]] = {}
        for label, index in markets:
            idx = int(index)
            terminal_raw[label] = _terminal_quantiles(component["_raw_level"][:, :, idx])
            terminal_standardized[label] = _terminal_quantiles(
                component["_standardized_move"][:, :, idx]
            )
            fan_raw[label] = _quantile_path(component["_raw_level"][:, :, idx])
            fan_standardized[label] = _quantile_path(
                component["_standardized_move"][:, :, idx]
            )
        public_components.append(
            {
                "component_no": int(component["component_no"]),
                "window_index": int(component["window_index"]),
                "original_weight": float(component["weight"]),
                "sparse_weight": float(sparse_meta["sparse_weight"]),
                "sample_count": int(component["sample_count"]),
                "terminal_raw": terminal_raw,
                "terminal_standardized": terminal_standardized,
                "fan_raw": fan_raw,
                "fan_standardized": fan_standardized,
            }
        )

    sparse_terminal_raw: dict[str, dict[str, float]] = {}
    sparse_terminal_standardized: dict[str, dict[str, float]] = {}
    sparse_fan_raw: dict[str, dict[str, list[float]]] = {}
    sparse_fan_standardized: dict[str, dict[str, list[float]]] = {}
    full_terminal_raw: dict[str, dict[str, float]] = {}
    full_terminal_standardized: dict[str, dict[str, float]] = {}
    for label, index in markets:
        idx = int(index)
        full_terminal_raw[label] = _terminal_quantiles(full_raw[:, :, idx])
        full_terminal_standardized[label] = _terminal_quantiles(
            full_standardized[:, :, idx]
        )
        sparse_terminal_raw[label] = _terminal_quantiles(raw_sparse[:, :, idx])
        sparse_terminal_standardized[label] = _terminal_quantiles(
            standardized_sparse[:, :, idx]
        )
        sparse_fan_raw[label] = _quantile_path(raw_sparse[:, :, idx])
        sparse_fan_standardized[label] = _quantile_path(standardized_sparse[:, :, idx])

    kept_weight = sum(float(component["weight"]) for component in selected_components)
    sparse_weights = [float(component["sparse_weight"]) for component in public_components]
    effective_components = (
        1.0 / sum(weight * weight for weight in sparse_weights)
        if sparse_weights
        else 0.0
    )
    return {
        "case_name": str(case.get("case_name", "")),
        "label": _case_label(str(case.get("case_name", ""))),
        "variant_index": int(variant),
        "arrays_snapshot": str(arrays_path),
        "report_snapshot": str(report_path),
        "original_component_count": int(len(component_rows)),
        "sparse_component_count": int(len(public_components)),
        "kept_original_weight": float(kept_weight),
        "effective_sparse_component_count": float(effective_components),
        "selected_components": public_components,
        "full_pooled_terminal_raw": full_terminal_raw,
        "full_pooled_terminal_standardized": full_terminal_standardized,
        "sparse_pooled_terminal_raw": sparse_terminal_raw,
        "sparse_pooled_terminal_standardized": sparse_terminal_standardized,
        "sparse_pooled_fan_raw": sparse_fan_raw,
        "sparse_pooled_fan_standardized": sparse_fan_standardized,
    }


def _terminal_p50_range_comparison(
    cases: list[dict[str, Any]],
    *,
    markets: list[tuple[str, int]],
) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for label, _index in markets:
        full_values = [
            float(case["full_pooled_terminal_standardized"][label]["p50"])
            for case in cases
        ]
        sparse_values = [
            float(case["sparse_pooled_terminal_standardized"][label]["p50"])
            for case in cases
        ]
        component_values = [
            float(component["terminal_standardized"][label]["p50"])
            for case in cases
            for component in case["selected_components"]
        ]
        rows[label] = {
            "full_pooled": float(max(full_values) - min(full_values)),
            "sparse_pooled": float(max(sparse_values) - min(sparse_values)),
            "selected_component": float(max(component_values) - min(component_values)),
        }
    return rows


def build_sparse_component_family_view(
    summary_path: str | Path,
    *,
    markets: list[tuple[str, int]] | None = None,
    max_components: int = 2,
    min_cumulative_weight: float = 0.80,
) -> dict[str, Any]:
    """Build a sparse component-preserving product view from saved rollouts."""

    summary = _load_json(summary_path)
    market_specs = list(markets or DEFAULT_MARKETS)
    cases = [
        _load_case(
            case,
            markets=market_specs,
            max_components=max_components,
            min_cumulative_weight=min_cumulative_weight,
        )
        for case in summary.get("cases", [])
        if isinstance(case, dict)
    ]
    return {
        "source_summary": str(summary_path),
        "fixed_start_index": summary.get("fixed_start_index"),
        "case_count": len(cases),
        "total_selected_components": int(
            sum(case["sparse_component_count"] for case in cases)
        ),
        "selection_policy": {
            "max_components": int(max_components),
            "min_cumulative_weight": float(min_cumulative_weight),
            "description": (
                "Keep the highest-weight support components up to max_components "
                "or until their original weights cover min_cumulative_weight, "
                "then expose them separately and renormalize sparse weights."
            ),
        },
        "market_specs": [
            {"label": label, "index": int(index)} for label, index in market_specs
        ],
        "terminal_p50_range_comparison": _terminal_p50_range_comparison(
            cases,
            markets=market_specs,
        ),
        "cases": cases,
        "scope_note": (
            "Sparse component-family view over saved component-preserving rollouts. "
            "The pooled sparse fan is a summary; selected component fans remain "
            "visible so narrative response is not hidden by averaging."
        ),
    }


def plot_sparse_component_families(
    report: dict[str, Any],
    output_path: str | Path,
    *,
    view: str = "standardized",
    max_markets: int = 3,
) -> str:
    """Plot selected component families and sparse pooled medians."""

    if view not in {"standardized", "raw"}:
        raise ValueError("view must be 'standardized' or 'raw'")
    cases = [case for case in report.get("cases", []) if isinstance(case, dict)]
    markets = [
        str(spec["label"])
        for spec in report.get("market_specs", [])
        if isinstance(spec, dict)
    ][: int(max_markets)]
    if not cases or not markets:
        return ""

    fig, axes = plt.subplots(
        len(cases),
        len(markets),
        figsize=(4.2 * len(markets), max(3.0, 1.65 * len(cases) + 1.4)),
        sharex=True,
    )
    if len(cases) == 1:
        axes = np.asarray([axes])
    if len(markets) == 1:
        axes = axes[:, None]

    fan_key = "fan_standardized" if view == "standardized" else "fan_raw"
    pooled_key = (
        "sparse_pooled_fan_standardized" if view == "standardized" else "sparse_pooled_fan_raw"
    )
    for row, case in enumerate(cases):
        for col, market in enumerate(markets):
            ax = axes[row, col]
            for component_idx, component in enumerate(case["selected_components"]):
                fan = component[fan_key][market]
                days = np.asarray(fan["days"], dtype=np.float64)
                p10 = np.asarray(fan["p10"], dtype=np.float64)
                p50 = np.asarray(fan["p50"], dtype=np.float64)
                p90 = np.asarray(fan["p90"], dtype=np.float64)
                color = COLORS[component_idx % len(COLORS)]
                weight = float(component["sparse_weight"])
                ax.fill_between(days, p10, p90, color=color, alpha=0.11, linewidth=0)
                ax.plot(
                    days,
                    p50,
                    color=color,
                    linewidth=1.0 + 1.4 * weight,
                    alpha=0.95,
                    label=f"component {component_idx + 1}",
                )
            pooled = case[pooled_key][market]
            ax.plot(
                np.asarray(pooled["days"], dtype=np.float64),
                np.asarray(pooled["p50"], dtype=np.float64),
                color="#111111",
                linewidth=1.7,
                linestyle="--",
                label="sparse pooled p50",
            )
            if view == "standardized":
                ax.axhline(0.0, color="#444444", linestyle=":", linewidth=0.7)
            if row == 0:
                ax.set_title(market, fontsize=10)
            if col == 0:
                supports = ", ".join(
                    f"{component['window_index']}:{component['sparse_weight']:.0%}"
                    for component in case["selected_components"]
                )
                ax.set_ylabel(
                    f"{case['label']}\nwindows {supports}",
                    fontsize=7,
                )
            ax.grid(axis="y", alpha=0.14)
    for ax in axes[-1, :]:
        ax.set_xlabel("Forecast day")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(3, len(labels)),
            fontsize=7,
            frameon=False,
        )
    title = (
        "Sparse component-family scenario view"
        if view == "standardized"
        else "Sparse component-family raw-level view"
    )
    fig.suptitle(
        f"{title} (fixed start {report.get('fixed_start_index')})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def _parse_market_specs(values: Iterable[str] | None) -> list[tuple[str, int]]:
    if not values:
        return DEFAULT_MARKETS
    specs: list[tuple[str, int]] = []
    for value in values:
        if ":" not in value:
            raise ValueError(f"market spec {value!r} must be Label:index")
        label, raw_index = value.rsplit(":", 1)
        specs.append((label.strip(), int(raw_index)))
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plot-raw", type=Path, default=DEFAULT_PLOT_RAW)
    parser.add_argument("--plot-standardized", type=Path, default=DEFAULT_PLOT_STD)
    parser.add_argument("--max-components", type=int, default=2)
    parser.add_argument("--min-cumulative-weight", type=float, default=0.80)
    parser.add_argument("--max-markets", type=int, default=3)
    parser.add_argument(
        "--market",
        action="append",
        default=None,
        help="Market label/index pair, for example SPX:25. Repeatable.",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    report = build_sparse_component_family_view(
        args.summary,
        markets=_parse_market_specs(args.market),
        max_components=args.max_components,
        min_cumulative_weight=args.min_cumulative_weight,
    )
    if not args.no_plot:
        report["artifact_paths"] = {
            "raw_component_family_plot": plot_sparse_component_families(
                report,
                args.plot_raw,
                view="raw",
                max_markets=args.max_markets,
            ),
            "standardized_component_family_plot": plot_sparse_component_families(
                report,
                args.plot_standardized,
                view="standardized",
                max_markets=args.max_markets,
            ),
        }
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "case_count": report["case_count"],
                "total_selected_components": report["total_selected_components"],
                "plots": report.get("artifact_paths", {}),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
