#!/usr/bin/env python
"""Build raw, start-relative, and standardized fan-chart views for live story decks.

This is an artifact-only plotting utility. It consumes cached live story-deck
snapshots and does not call OpenAI, train, or rerun the frozen SNI generator.
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

DEFAULT_SUMMARY = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_gradio_live_story_multistart_950c_5start_default_deck_warning_aware/"
    "start_22/gradio_live_api_casebook_summary.json"
)
DEFAULT_OUTPUT_DIR = Path("paper/narrative_grounded_scenarios/figures")
DEFAULT_MARKETS: list[tuple[str, int]] = [
    ("SPX", 25),
    ("VIX", 38),
    ("BBB OAS", 35),
    ("US10Y", 33),
    ("DXY", 28),
    ("Gold", 37),
    ("Crude oil", 31),
]
VIEW_SPECS = {
    "raw_level": {
        "title": "Raw market levels",
        "suffix": "raw_levels",
        "ylabel": "raw level",
    },
    "start_relative_move": {
        "title": "Move from accepted start",
        "suffix": "start_relative_moves",
        "ylabel": "change from start",
    },
    "delta_scale_standardized_move": {
        "title": "Delta-scale standardized move",
        "suffix": "delta_scale_standardized_moves",
        "ylabel": "standardized change",
    },
}
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
    calibration = generation.get("narrative_ensemble_calibration") if isinstance(generation, dict) else {}
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


def _require_array(arrays: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
    if key not in arrays:
        raise KeyError(f"{key} missing from prefix arrays snapshot")
    return np.asarray(arrays[key], dtype=np.float64)


def _quantile_fan(values: np.ndarray) -> dict[str, list[float]]:
    return {
        "days": list(range(int(values.shape[1]))),
        "p10": np.nanpercentile(values, 10, axis=0).astype(float).tolist(),
        "p50": np.nanpercentile(values, 50, axis=0).astype(float).tolist(),
        "p90": np.nanpercentile(values, 90, axis=0).astype(float).tolist(),
    }


def _prepend_day0(paths: np.ndarray, day0: np.ndarray) -> np.ndarray:
    head = np.broadcast_to(day0[None, None, :], (paths.shape[0], 1, paths.shape[2]))
    return np.concatenate([head, paths], axis=1)


def _load_case_views(
    case: dict[str, Any],
    *,
    markets: list[tuple[str, int]],
) -> dict[str, Any]:
    arrays_path = Path(str(case.get("prefix_arrays_snapshot_path", "")))
    report_path = Path(str(case.get("prefix_report_snapshot_path", "")))
    if not arrays_path.exists():
        raise FileNotFoundError(arrays_path)
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = _load_json(report_path)
    with np.load(arrays_path, allow_pickle=True) as arrays:
        generated = _require_array(arrays, "generated_states")
        requested_raw = _require_array(arrays, "requested_raw")
        moves = (
            _require_array(arrays, "samples")
            if "samples" in arrays
            else generated - requested_raw[:, None, None, :]
        )
        variant = _variant_index(report, int(generated.shape[0]))
        states_v = np.asarray(generated[variant], dtype=np.float64)
        moves_v = np.asarray(moves[variant], dtype=np.float64)
        start_v = np.asarray(requested_raw[variant], dtype=np.float64)
        if "delta_scale" in arrays:
            scale = np.maximum(np.abs(np.asarray(arrays["delta_scale"], dtype=np.float64)), 1e-8)
        else:
            scale = np.ones_like(moves_v[0], dtype=np.float64)
        standardized = moves_v / scale[None, :, :]

    view_arrays = {
        "raw_level": _prepend_day0(states_v, start_v),
        "start_relative_move": _prepend_day0(moves_v, np.zeros_like(start_v)),
        "delta_scale_standardized_move": _prepend_day0(
            standardized, np.zeros_like(start_v)
        ),
    }
    views: dict[str, dict[str, dict[str, list[float]]]] = {}
    for view_name, arr in view_arrays.items():
        by_market: dict[str, dict[str, list[float]]] = {}
        for label, index in markets:
            if not 0 <= int(index) < arr.shape[-1]:
                raise IndexError(f"market {label} index {index} outside width {arr.shape[-1]}")
            by_market[label] = _quantile_fan(arr[:, :, int(index)])
        views[view_name] = by_market
    return {
        "case_name": str(case.get("case_name", "")),
        "label": _case_label(str(case.get("case_name", ""))),
        "variant_index": variant,
        "arrays_snapshot": str(arrays_path),
        "report_snapshot": str(report_path),
        "views": views,
    }


def build_fan_chart_view_data(
    summary_path: str | Path,
    *,
    markets: list[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Load live story-deck arrays and compute fan quantiles for each view."""

    summary = _load_json(summary_path)
    market_specs = list(markets or DEFAULT_MARKETS)
    cases = [
        _load_case_views(case, markets=market_specs)
        for case in summary.get("cases", [])
        if isinstance(case, dict)
    ]
    return {
        "source_summary": str(summary_path),
        "fixed_start_index": summary.get("fixed_start_index"),
        "case_count": len(cases),
        "market_specs": [
            {"label": label, "index": int(index)} for label, index in market_specs
        ],
        "view_specs": VIEW_SPECS,
        "cases": cases,
        "scope_note": (
            "Raw level, start-relative move, and delta-scale standardized fan "
            "views built from cached live story-deck generated paths. The "
            "standardized view divides start-relative raw moves by the saved "
            "horizon/factor delta_scale; it is a normalized move view, not a "
            "claim that the full SNI innovation tensor was persisted."
        ),
    }


def _plot_view(
    view_data: dict[str, Any],
    view_name: str,
    output_path: str | Path,
    *,
    max_rows: int | None = None,
) -> str:
    cases = [case for case in view_data.get("cases", []) if isinstance(case, dict)]
    market_specs = [
        spec for spec in view_data.get("market_specs", []) if isinstance(spec, dict)
    ]
    if max_rows is not None:
        market_specs = market_specs[: int(max_rows)]
    if not cases or not market_specs:
        return ""

    spec = VIEW_SPECS[view_name]
    fig, axes = plt.subplots(
        len(market_specs),
        1,
        figsize=(11.5, max(2.0, 1.55 * len(market_specs) + 1.6)),
        sharex=True,
    )
    if len(market_specs) == 1:
        axes = [axes]

    for ax, market in zip(axes, market_specs, strict=True):
        label = str(market["label"])
        for case_idx, case in enumerate(cases):
            fan = case["views"][view_name][label]
            days = np.asarray(fan["days"], dtype=np.float64)
            p10 = np.asarray(fan["p10"], dtype=np.float64)
            p50 = np.asarray(fan["p50"], dtype=np.float64)
            p90 = np.asarray(fan["p90"], dtype=np.float64)
            color = COLORS[case_idx % len(COLORS)]
            ax.fill_between(days, p10, p90, color=color, alpha=0.11, linewidth=0)
            ax.plot(days, p50, color=color, linewidth=1.8, label=case["label"])
        if view_name != "raw_level":
            ax.axhline(0.0, color="#333333", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.18)

    axes[-1].set_xlabel("Forecast day")
    title = (
        f"{spec['title']} by professional narrative"
        f" (fixed start {view_data.get('fixed_start_index')})"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(3, len(labels)),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.96])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return str(output)


def plot_fan_chart_views(
    view_data: dict[str, Any],
    output_dir: str | Path,
    *,
    prefix: str = "live_story_fan_views_start22",
    max_rows: int | None = None,
) -> dict[str, str]:
    """Write one fan-chart figure for each view."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for view_name, spec in VIEW_SPECS.items():
        paths[view_name] = _plot_view(
            view_data,
            view_name,
            output / f"{prefix}_{spec['suffix']}.png",
            max_rows=max_rows,
        )
    return paths


def plot_combined_fan_chart_views(
    view_data: dict[str, Any],
    output_path: str | Path,
    *,
    max_markets: int = 4,
) -> str:
    """Write one compact grid with raw, start-relative, and standardized views."""

    cases = [case for case in view_data.get("cases", []) if isinstance(case, dict)]
    market_specs = [
        spec for spec in view_data.get("market_specs", []) if isinstance(spec, dict)
    ][: int(max_markets)]
    view_names = list(VIEW_SPECS)
    if not cases or not market_specs:
        return ""

    fig, axes = plt.subplots(
        len(market_specs),
        len(view_names),
        figsize=(15, max(3.5, 1.9 * len(market_specs) + 1.4)),
        sharex=True,
    )
    if len(market_specs) == 1:
        axes = np.asarray([axes])

    for col, view_name in enumerate(view_names):
        axes[0, col].set_title(VIEW_SPECS[view_name]["title"], fontsize=11)
    for row, market in enumerate(market_specs):
        label = str(market["label"])
        for col, view_name in enumerate(view_names):
            ax = axes[row, col]
            for case_idx, case in enumerate(cases):
                fan = case["views"][view_name][label]
                days = np.asarray(fan["days"], dtype=np.float64)
                p10 = np.asarray(fan["p10"], dtype=np.float64)
                p50 = np.asarray(fan["p50"], dtype=np.float64)
                p90 = np.asarray(fan["p90"], dtype=np.float64)
                color = COLORS[case_idx % len(COLORS)]
                ax.fill_between(days, p10, p90, color=color, alpha=0.09, linewidth=0)
                ax.plot(days, p50, color=color, linewidth=1.35, label=case["label"])
            if view_name != "raw_level":
                ax.axhline(
                    0.0, color="#333333", linestyle="--", linewidth=0.7, alpha=0.7
                )
            if col == 0:
                ax.set_ylabel(label)
            ax.grid(axis="y", alpha=0.14)

    for ax in axes[-1, :]:
        ax.set_xlabel("Forecast day")
    fig.suptitle(
        f"Fixed-start narrative fan-chart views (start {view_data.get('fixed_start_index')})",
        fontsize=14,
        fontweight="bold",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=min(3, len(labels)),
        fontsize=8,
        frameon=False,
    )
    fig.tight_layout(rect=[0, 0.12, 1, 0.95])
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
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--prefix", default="live_story_fan_views_start22")
    parser.add_argument(
        "--market",
        action="append",
        default=None,
        help="Market label/index pair, for example SPX:25. Repeatable.",
    )
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    view_data = build_fan_chart_view_data(
        args.summary,
        markets=_parse_market_specs(args.market),
    )
    figure_paths = plot_fan_chart_views(
        view_data,
        args.output_dir,
        prefix=args.prefix,
        max_rows=args.max_rows,
    )
    figure_paths["combined_views"] = plot_combined_fan_chart_views(
        view_data,
        args.output_dir / f"{args.prefix}_combined_views.png",
    )
    summary_json = args.summary_json or args.output_dir / f"{args.prefix}_summary.json"
    output = {
        "view_data": view_data,
        "figure_paths": figure_paths,
        "summary_json": str(summary_json),
    }
    _write_json(summary_json, output)
    print(
        json.dumps(
            {
                "summary_json": str(summary_json),
                "figure_paths": figure_paths,
                "case_count": view_data["case_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
