#!/usr/bin/env python
"""Diagnose where narrative conditionality is lost: support, rollout, or pooling.

This artifact-only script consumes saved live story-deck snapshots. It does not
call OpenAI, train a model, or rerun the frozen SNI generator.
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
DEFAULT_OUTPUT = (
    DEFAULT_SUMMARY.parent / "component_pooling_conditionality_diagnostic.json"
)
DEFAULT_PLOT = DEFAULT_SUMMARY.parent / "component_pooling_terminal_medians.png"
DEFAULT_MARKETS: list[tuple[str, int]] = [
    ("SPX", 25),
    ("VIX", 38),
    ("BBB_OAS", 35),
    ("US10Y", 33),
    ("DXY", 28),
    ("GOLD", 37),
    ("CRUDE_OIL", 31),
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


def component_slices_for_variant(
    *,
    variant_index: int,
    component_variant_index: np.ndarray,
    component_window_index: np.ndarray,
    component_weight: np.ndarray,
    component_sample_count: np.ndarray,
    sample_count: int,
) -> list[dict[str, Any]]:
    """Return sample slices for one rollout variant from component metadata."""

    rows: list[dict[str, Any]] = []
    cursor = 0
    for variant, window, weight, count in zip(
        component_variant_index,
        component_window_index,
        component_weight,
        component_sample_count,
        strict=True,
    ):
        if int(variant) != int(variant_index):
            continue
        start = cursor
        stop = cursor + int(count)
        rows.append(
            {
                "window_index": int(window),
                "weight": float(weight),
                "sample_count": int(count),
                "sample_slice": [int(start), int(stop)],
            }
        )
        cursor = stop
    if cursor != int(sample_count):
        raise ValueError(
            f"component sample counts sum to {cursor}, expected {int(sample_count)}"
        )
    return rows


def _flatten_paths(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr.reshape((arr.shape[0], -1))


def _energy_distance(left: np.ndarray, right: np.ndarray) -> float:
    x = _flatten_paths(left)
    y = _flatten_paths(right)
    if x.shape[0] == 0 or y.shape[0] == 0:
        return float("nan")
    dxy = np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1).mean()
    dxx = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=-1).mean()
    dyy = np.linalg.norm(y[:, None, :] - y[None, :, :], axis=-1).mean()
    return float(2.0 * dxy - dxx - dyy)


def _median(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.median(finite)) if finite else None


def _support_ids(case: dict[str, Any], report: dict[str, Any]) -> set[str]:
    rows = case.get("support_top_candidates")
    if not isinstance(rows, list):
        memory_prior = report.get("cached_query", {})
        if isinstance(memory_prior, dict):
            memory_prior = memory_prior.get("memory_prior", {})
        rows = memory_prior.get("candidate_details", []) if isinstance(memory_prior, dict) else []
    return {
        str(row.get("window_id") or row.get("window_index"))
        for row in rows
        if isinstance(row, dict)
    }


def _terminal_median(samples: np.ndarray, index: int) -> float:
    return float(np.nanmedian(samples[:, -1, int(index)]))


def _case_from_summary(
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
        samples = np.asarray(arrays["samples"], dtype=np.float64)
        delta_scale = np.maximum(
            np.abs(np.asarray(arrays["delta_scale"], dtype=np.float64)), 1e-8
        )
        variant = _variant_index(report, int(samples.shape[0]))
        standardized = samples[variant] / delta_scale[None, :, :]
        components = component_slices_for_variant(
            variant_index=variant,
            component_variant_index=np.asarray(
                arrays["rollout_component_variant_index"], dtype=np.int64
            ),
            component_window_index=np.asarray(
                arrays["rollout_component_window_index"], dtype=np.int64
            ),
            component_weight=np.asarray(arrays["rollout_component_weight"], dtype=np.float64),
            component_sample_count=np.asarray(
                arrays["rollout_component_sample_count"], dtype=np.int64
            ),
            sample_count=int(standardized.shape[0]),
        )

    component_rows: list[dict[str, Any]] = []
    for component_no, row in enumerate(components):
        start, stop = row["sample_slice"]
        component_samples = standardized[start:stop]
        terminal = {
            label: _terminal_median(component_samples, index)
            for label, index in markets
        }
        component_rows.append(
            {
                **row,
                "component_no": component_no,
                "terminal_median_standardized": terminal,
                "_samples": component_samples,
            }
        )
    pooled_terminal = {
        label: _terminal_median(standardized, index) for label, index in markets
    }
    return {
        "case_name": str(case.get("case_name", "")),
        "label": _case_label(str(case.get("case_name", ""))),
        "variant_index": variant,
        "support_ids": _support_ids(case, report),
        "arrays_snapshot": str(arrays_path),
        "report_snapshot": str(report_path),
        "pooled_terminal_median_standardized": pooled_terminal,
        "components": component_rows,
        "_pooled_samples": standardized,
    }


def _pairwise_support_jaccard(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(cases):
        for right in cases[i + 1 :]:
            a = set(left["support_ids"])
            b = set(right["support_ids"])
            union = a | b
            rows.append(
                {
                    "left": left["label"],
                    "right": right["label"],
                    "jaccard": float(len(a & b) / len(union)) if union else 0.0,
                }
            )
    return rows


def _path_energy_rows(
    cases: list[dict[str, Any]],
    *,
    market_indices: list[int],
) -> dict[str, Any]:
    pooled_cross: list[float] = []
    within_case_component: list[float] = []
    cross_case_component: list[float] = []
    within_component_split: list[float] = []

    for i, left in enumerate(cases):
        for left_idx, left_component in enumerate(left["components"]):
            for right_component in left["components"][left_idx + 1 :]:
                within_case_component.append(
                    _energy_distance(
                        left_component["_samples"][:, :, market_indices],
                        right_component["_samples"][:, :, market_indices],
                    )
                )
        for right in cases[i + 1 :]:
            pooled_cross.append(
                _energy_distance(
                    left["_pooled_samples"][:, :, market_indices],
                    right["_pooled_samples"][:, :, market_indices],
                )
            )
            for left_component in left["components"]:
                for right_component in right["components"]:
                    cross_case_component.append(
                        _energy_distance(
                            left_component["_samples"][:, :, market_indices],
                            right_component["_samples"][:, :, market_indices],
                        )
                    )
        for component in left["components"]:
            samples = component["_samples"]
            if samples.shape[0] < 4:
                continue
            split = samples.shape[0] // 2
            within_component_split.append(
                _energy_distance(
                    samples[:split, :, market_indices],
                    samples[split:, :, market_indices],
                )
            )

    return {
        "pooled_cross_narrative": {
            "path_energy_median": _median(pooled_cross),
            "path_energy_values": pooled_cross,
        },
        "component_cross_narrative": {
            "path_energy_median": _median(cross_case_component),
            "path_energy_values": cross_case_component,
        },
        "within_narrative_component": {
            "path_energy_median": _median(within_case_component),
            "path_energy_values": within_case_component,
        },
        "within_component_split": {
            "path_energy_median": _median(within_component_split),
            "path_energy_values": within_component_split,
            "usable_component_count": len(within_component_split),
        },
    }


def _pooling_diagnosis(
    cases: list[dict[str, Any]],
    *,
    markets: list[tuple[str, int]],
) -> dict[str, Any]:
    component_vs_pooled: dict[str, float | None] = {}
    component_range: dict[str, float] = {}
    pooled_range: dict[str, float] = {}
    for label, _index in markets:
        component_values = [
            float(component["terminal_median_standardized"][label])
            for case in cases
            for component in case["components"]
        ]
        pooled_values = [
            float(case["pooled_terminal_median_standardized"][label]) for case in cases
        ]
        c_range = float(max(component_values) - min(component_values))
        p_range = float(max(pooled_values) - min(pooled_values))
        component_range[label] = c_range
        pooled_range[label] = p_range
        component_vs_pooled[label] = c_range / p_range if p_range > 1e-12 else None
    return {
        "terminal_median_component_range": component_range,
        "terminal_median_pooled_range": pooled_range,
        "terminal_median_component_vs_pooled_ratio": component_vs_pooled,
        "interpretation": (
            "Ratios above 1 mean individual component rollouts span more "
            "terminal standardized response than the pooled story medians. "
            "Large ratios with weak pooled fans point to mixture smoothing; "
            "small component ranges point to generator insensitivity."
        ),
    }


def build_component_pooling_diagnostic(
    summary_path: str | Path,
    *,
    markets: list[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Build component-level support/rollout/pooling diagnostics."""

    summary = _load_json(summary_path)
    market_specs = list(markets or DEFAULT_MARKETS)
    cases = [
        _case_from_summary(case, markets=market_specs)
        for case in summary.get("cases", [])
        if isinstance(case, dict)
    ]
    support_rows = _pairwise_support_jaccard(cases)
    support_values = [float(row["jaccard"]) for row in support_rows]
    energy = _path_energy_rows(
        cases,
        market_indices=[int(index) for _label, index in market_specs],
    )
    component_count = sum(len(case["components"]) for case in cases)
    public_cases = []
    for case in cases:
        public_cases.append(
            {
                key: value
                for key, value in case.items()
                if key not in {"_pooled_samples", "support_ids"}
            }
        )
        for component in public_cases[-1]["components"]:
            component.pop("_samples", None)
    return {
        "source_summary": str(summary_path),
        "fixed_start_index": summary.get("fixed_start_index"),
        "case_count": len(cases),
        "component_count": int(component_count),
        "market_specs": [
            {"label": label, "index": int(index)} for label, index in market_specs
        ],
        "support_jaccard": {
            "mean": float(np.mean(support_values)) if support_values else None,
            "max": float(np.max(support_values)) if support_values else None,
            "pairs": support_rows,
        },
        "pooled_cross_narrative": energy["pooled_cross_narrative"],
        "component_cross_narrative": energy["component_cross_narrative"],
        "within_narrative_component": energy["within_narrative_component"],
        "within_component_split": energy["within_component_split"],
        "pooling_diagnosis": _pooling_diagnosis(cases, markets=market_specs),
        "cases": public_cases,
        "scope_note": (
            "Component-level diagnostic over delta-scale standardized generated "
            "moves. It tests whether narrative support differs, whether each "
            "support component produces distinct future responses, and whether "
            "pooling dampens those responses."
        ),
    }


def plot_component_terminal_medians(
    report: dict[str, Any],
    output_path: str | Path,
    *,
    max_markets: int = 7,
) -> str:
    """Plot component terminal medians and pooled medians by narrative."""

    markets = [
        str(spec["label"])
        for spec in report.get("market_specs", [])
        if isinstance(spec, dict)
    ][: int(max_markets)]
    cases = [case for case in report.get("cases", []) if isinstance(case, dict)]
    if not markets or not cases:
        return ""

    fig, axes = plt.subplots(
        len(markets),
        1,
        figsize=(12, max(3.2, 1.55 * len(markets) + 1.7)),
        sharex=True,
    )
    if len(markets) == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.linspace(0, 1, len(cases)))
    x_base = np.arange(len(cases), dtype=np.float64)
    for ax, market in zip(axes, markets, strict=True):
        for case_idx, case in enumerate(cases):
            x = x_base[case_idx]
            components = [
                component
                for component in case.get("components", [])
                if isinstance(component, dict)
            ]
            component_values = [
                float(component["terminal_median_standardized"][market])
                for component in components
            ]
            if component_values:
                offsets = np.linspace(-0.18, 0.18, len(component_values))
                ax.scatter(
                    x + offsets,
                    component_values,
                    color=colors[case_idx],
                    s=28,
                    alpha=0.65,
                    marker="o",
                )
            pooled = float(case["pooled_terminal_median_standardized"][market])
            ax.scatter(
                [x],
                [pooled],
                color=colors[case_idx],
                s=80,
                edgecolor="black",
                linewidth=0.7,
                marker="D",
            )
        ax.axhline(0.0, color="#444444", linestyle="--", linewidth=0.8)
        ax.set_ylabel(market)
        ax.grid(axis="y", alpha=0.18)
    axes[-1].set_xticks(
        x_base,
        [str(case.get("label", "")) for case in cases],
        rotation=30,
        ha="right",
    )
    fig.suptitle(
        "Component versus pooled terminal medians, delta-scale standardized moves",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
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
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument(
        "--market",
        action="append",
        default=None,
        help="Market label/index pair, for example SPX:25. Repeatable.",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    report = build_component_pooling_diagnostic(
        args.summary,
        markets=_parse_market_specs(args.market),
    )
    if not args.no_plot:
        report["artifact_paths"] = {
            "component_terminal_plot": plot_component_terminal_medians(
                report, args.plot
            )
        }
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "plot": report.get("artifact_paths", {}).get("component_terminal_plot", ""),
                "case_count": report["case_count"],
                "component_count": report["component_count"],
                "support_jaccard_max": report["support_jaccard"]["max"],
                "pooled_path_energy_median": report["pooled_cross_narrative"][
                    "path_energy_median"
                ],
                "component_path_energy_median": report["component_cross_narrative"][
                    "path_energy_median"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
