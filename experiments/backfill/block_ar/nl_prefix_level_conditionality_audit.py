#!/usr/bin/env python
"""Audit narrative conditionality at the selected historical-prefix layer.

This script checks whether selected support prefixes themselves match the
current/recent narrative before any frozen-generator rollout or pooling.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
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
DEFAULT_OUTPUT = DEFAULT_SUMMARY.parent / "prefix_level_conditionality_audit.json"
DEFAULT_PLOT = DEFAULT_SUMMARY.parent / "prefix_level_selected_support_paths.png"
DEFAULT_MARKETS: list[tuple[str, int]] = [
    ("SPX", 25),
    ("VIX", 38),
    ("BBB_OAS", 35),
    ("US10Y", 33),
    ("DXY", 28),
    ("GOLD", 37),
    ("CRUDE_OIL", 31),
]
MARKET_ALIASES = {
    "CRUDE": "CRUDE_OIL",
    "CRUDE OIL": "CRUDE_OIL",
    "OIL": "CRUDE_OIL",
    "GOLD": "GOLD",
    "SPREADS": "BBB_OAS",
    "CREDIT_SPREADS": "BBB_OAS",
    "CREDIT": "BBB_OAS",
    "RATES": "US10Y",
    "US 10Y": "US10Y",
}
COLORS = ["#1565C0", "#EF6C00", "#6A1B9A", "#2E7D32", "#C62828"]


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


def _norm_market(market: Any) -> str:
    raw = str(market or "").strip().upper().replace("-", "_").replace("/", "")
    raw = raw.replace(" ", "_")
    return MARKET_ALIASES.get(raw, raw)


def direction_expected_sign(direction: Any) -> int | None:
    """Map narrative direction words to recent-prefix terminal-delta signs."""

    text = str(direction or "").strip().lower()
    if text in {
        "up",
        "higher",
        "increase",
        "increasing",
        "wider",
        "steeper",
        "stronger",
    }:
        return 1
    if text in {
        "down",
        "lower",
        "decrease",
        "decreasing",
        "tighter",
        "compressed",
        "compressing",
        "weaker",
    }:
        return -1
    if text in {"flat", "stable", "unchanged", "stabilizing", "neutral"}:
        return 0
    return None


def _observed_sign(value: float, *, tolerance: float = 1e-8) -> int:
    if abs(float(value)) <= float(tolerance):
        return 0
    return 1 if float(value) > 0.0 else -1


def _align_delta(
    delta_by_market: dict[str, float],
    *,
    implications: list[dict[str, Any]],
) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for implication in implications:
        if not isinstance(implication, dict):
            continue
        market = _norm_market(implication.get("market"))
        expected = direction_expected_sign(implication.get("direction"))
        if expected is None or market not in delta_by_market:
            skipped.append(
                {
                    "market": market,
                    "direction": implication.get("direction"),
                    "reason": "unsupported_direction_or_market",
                }
            )
            continue
        observed_delta = float(delta_by_market[market])
        observed = _observed_sign(observed_delta)
        aligned = bool(observed == expected) if expected != 0 else bool(observed == 0)
        checked.append(
            {
                "market": market,
                "direction": implication.get("direction"),
                "expected_sign": int(expected),
                "observed_sign": int(observed),
                "terminal_delta": observed_delta,
                "aligned": aligned,
            }
        )
    match_count = sum(1 for row in checked if bool(row["aligned"]))
    return {
        "checked": checked,
        "skipped": skipped,
        "checked_count": int(len(checked)),
        "match_count": int(match_count),
        "mismatch_count": int(len(checked) - match_count),
        "skipped_count": int(len(skipped)),
        "match_rate": float(match_count / len(checked)) if checked else None,
        "status": "pass" if checked and match_count == len(checked) else "warning",
    }


def prefix_direction_alignment(
    prefix: np.ndarray,
    *,
    implications: list[dict[str, Any]],
    market_to_index: dict[str, int],
) -> dict[str, Any]:
    """Check implication directions against the raw recent-prefix terminal delta."""

    arr = np.asarray(prefix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError("prefix must have shape [history_len, channels]")
    delta_by_market = {
        _norm_market(market): float(arr[-1, int(index)] - arr[0, int(index)])
        for market, index in market_to_index.items()
    }
    return _align_delta(delta_by_market, implications=implications)


def _weighted_prefix(paths: list[np.ndarray], weights: list[float]) -> np.ndarray:
    arr = np.stack([np.asarray(path, dtype=np.float64) for path in paths], axis=0)
    w = np.asarray(weights, dtype=np.float64)
    if arr.shape[0] != w.shape[0]:
        raise ValueError("paths and weights must have the same length")
    total = float(w.sum())
    if total <= 0.0:
        w = np.ones_like(w) / float(len(w))
    else:
        w = w / total
    return np.sum(arr * w[:, None, None], axis=0)


def _support_candidates(
    report: dict[str, Any], fallback_case: dict[str, Any]
) -> list[dict[str, Any]]:
    prior = report.get("cached_query", {})
    if isinstance(prior, dict):
        prior = prior.get("memory_prior", {})
    rows = prior.get("candidate_details", []) if isinstance(prior, dict) else []
    if not rows:
        rows = fallback_case.get("support_top_candidates", [])
    return [dict(row) for row in rows if isinstance(row, dict)]


def _grounding_implications(report: dict[str, Any]) -> list[dict[str, Any]]:
    grounding = report.get("cached_query", {})
    grounding = grounding.get("grounding", {}) if isinstance(grounding, dict) else {}
    rows = (
        grounding.get("market_implications", []) if isinstance(grounding, dict) else []
    )
    return [dict(row) for row in rows if isinstance(row, dict)]


def _candidate_index(row: dict[str, Any]) -> int:
    for key in ("bridge_local_index", "window_index", "index"):
        if row.get(key) is None:
            continue
        return int(row[key])
    raise KeyError(f"candidate row has no bridge/local window index: {row}")


def _candidate_weight(row: dict[str, Any]) -> float:
    try:
        return max(float(row.get("weight", 0.0)), 0.0)
    except (TypeError, ValueError):
        return 0.0


def _metadata_for_index(metadata: list[dict[str, Any]], index: int) -> dict[str, Any]:
    if 0 <= int(index) < len(metadata):
        row = metadata[int(index)]
        if isinstance(row, dict):
            return row
    return {"window_id": f"window_{int(index):04d}", "window_index": int(index)}


def _load_npz_support_bank(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=True) as arrays:
        history_raw = np.asarray(arrays["history_raw"], dtype=np.float32)
        if "metadata" in arrays:
            raw_metadata = arrays["metadata"].tolist()
            metadata = (
                list(raw_metadata) if isinstance(raw_metadata, list) else raw_metadata
            )
        else:
            metadata = [
                {"window_id": f"window_{idx:04d}", "window_index": idx}
                for idx in range(int(history_raw.shape[0]))
            ]
    return {"history_raw": history_raw, "metadata": metadata}


def _load_default_bridge_support_bank(
    *,
    bridge_report_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    from diffusion.block_ar.generic_state_aware_normalized_innovation_flow_matching import (  # noqa: WPS433
        load_model,
    )
    from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: WPS433
        build_val_block,
    )
    from experiments.backfill.block_ar.nl_prefix_latent_oracle_autoencoder import (  # noqa: WPS433
        selected_bridge_window_indices,
    )
    from experiments.backfill.block_ar.nl_prefix_latent_story_smoke import (  # noqa: WPS433
        DEFAULT_BRIDGE_REPORT,
        DEFAULT_CHECKPOINT,
        window_metadata_by_bridge_local_index,
    )
    import torch  # noqa: WPS433

    bridge_path = Path(bridge_report_path or DEFAULT_BRIDGE_REPORT)
    checkpoint = Path(checkpoint_path or DEFAULT_CHECKPOINT)
    bridge_report = _load_json(bridge_path)
    selected_windows = selected_bridge_window_indices(bridge_report)
    args = SimpleNamespace(
        state_scope="joint38",
        test_start=4511,
        val_size=441,
        max_windows=441,
        eval_split="val",
        iv_count=25,
        clean_nonpositive_log_levels=True,
        positive_level_policy="reference_based",
        iv_transform="log_level",
        iv_lower_bound=1e-4,
        iv_upper_bound=1.0,
        scale_half_life=0.0,
        scale_floor=1e-4,
        center_mode="zero",
        drift_feature_mode="none",
    )
    _model, payload = load_model(str(checkpoint), torch.device("cpu"))
    (
        _history_level,
        _history_norm,
        _center,
        _scale,
        _drift,
        all_history_raw,
        _specs,
        _block,
    ) = build_val_block(args, payload)
    metadata_map = window_metadata_by_bridge_local_index(bridge_report)
    metadata = [
        metadata_map.get(
            int(idx),
            {"window_id": f"bridge_window_{idx:04d}", "window_index": int(idx)},
        )
        for idx in range(len(selected_windows))
    ]
    return {
        "history_raw": np.asarray(all_history_raw[selected_windows], dtype=np.float32),
        "metadata": metadata,
        "bridge_report": str(bridge_path),
        "checkpoint": str(checkpoint),
    }


def _case_audit(
    case: dict[str, Any],
    *,
    history_raw: np.ndarray,
    metadata: list[dict[str, Any]],
    market_to_index: dict[str, int],
) -> dict[str, Any]:
    report_path = Path(str(case.get("prefix_report_snapshot_path", "")))
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    report = _load_json(report_path)
    implications = _grounding_implications(report)
    candidates = _support_candidates(report, case)
    components: list[dict[str, Any]] = []
    paths: list[np.ndarray] = []
    weights: list[float] = []
    support_ids: list[str] = []
    for rank, row in enumerate(candidates, 1):
        index = _candidate_index(row)
        if not 0 <= index < int(history_raw.shape[0]):
            raise IndexError(f"support index {index} outside history bank")
        prefix = np.asarray(history_raw[index], dtype=np.float64)
        weight = _candidate_weight(row)
        paths.append(prefix)
        weights.append(weight)
        meta = _metadata_for_index(metadata, index)
        support_id = str(row.get("window_id") or meta.get("window_id") or index)
        support_ids.append(support_id)
        deltas = {
            market: float(prefix[-1, idx] - prefix[0, idx])
            for market, idx in market_to_index.items()
        }
        components.append(
            {
                "rank": int(row.get("rank", rank)),
                "bridge_local_index": int(index),
                "window_id": support_id,
                "weight": float(weight),
                "calendar_start_date": meta.get("calendar", {}).get(
                    "calendar_start_date", meta.get("calendar_start_date")
                ),
                "calendar_end_date": meta.get("calendar", {}).get(
                    "calendar_end_date", meta.get("calendar_end_date")
                ),
                "terminal_delta": deltas,
                "alignment": prefix_direction_alignment(
                    prefix,
                    implications=implications,
                    market_to_index=market_to_index,
                ),
            }
        )
    weighted = _weighted_prefix(paths, weights) if paths else np.empty((0, 0))
    weighted_delta = (
        {
            market: float(weighted[-1, idx] - weighted[0, idx])
            for market, idx in market_to_index.items()
        }
        if paths
        else {}
    )
    return {
        "case_name": str(case.get("case_name", "")),
        "label": _case_label(str(case.get("case_name", ""))),
        "report_snapshot": str(report_path),
        "implications": implications,
        "support_ids": support_ids,
        "selected_support_count": int(len(components)),
        "weighted_terminal_delta": weighted_delta,
        "weighted_alignment": _align_delta(weighted_delta, implications=implications),
        "components": components,
        "_weighted_prefix": weighted,
    }


def _support_jaccard(cases: list[dict[str, Any]]) -> dict[str, Any]:
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
    values = [float(row["jaccard"]) for row in rows]
    return {
        "mean": float(np.mean(values)) if values else None,
        "max": float(np.max(values)) if values else None,
        "pairs": rows,
    }


def _pairwise_signature_distance(
    cases: list[dict[str, Any]],
    markets: list[str],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    values: list[float] = []
    for i, left in enumerate(cases):
        left_vec = np.asarray(
            [left["weighted_terminal_delta"].get(market, 0.0) for market in markets],
            dtype=np.float64,
        )
        for right in cases[i + 1 :]:
            right_vec = np.asarray(
                [
                    right["weighted_terminal_delta"].get(market, 0.0)
                    for market in markets
                ],
                dtype=np.float64,
            )
            distance = float(np.linalg.norm(left_vec - right_vec))
            values.append(distance)
            rows.append(
                {"left": left["label"], "right": right["label"], "distance": distance}
            )
    finite = [value for value in values if math.isfinite(value)]
    return {
        "median": float(np.median(finite)) if finite else None,
        "mean": float(np.mean(finite)) if finite else None,
        "max": float(np.max(finite)) if finite else None,
        "pairs": rows,
    }


def build_prefix_level_conditionality_audit(
    summary_path: str | Path,
    *,
    support_bank_npz: str | Path | None = None,
    bridge_report: str | Path | None = None,
    checkpoint: str | Path | None = None,
    markets: list[tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Build support-prefix conditionality diagnostics before generator rollout."""

    summary = _load_json(summary_path)
    market_specs = [
        (_norm_market(label), int(index))
        for label, index in (markets or DEFAULT_MARKETS)
    ]
    market_to_index = {label: index for label, index in market_specs}
    if support_bank_npz is not None:
        bank = _load_npz_support_bank(support_bank_npz)
        bank_source = str(support_bank_npz)
    else:
        bank = _load_default_bridge_support_bank(
            bridge_report_path=bridge_report,
            checkpoint_path=checkpoint,
        )
        bank_source = "rebuilt_bridge_validation_history"
    history_raw = np.asarray(bank["history_raw"], dtype=np.float32)
    metadata = [dict(row) for row in list(bank.get("metadata", []))]
    cases = [
        _case_audit(
            case,
            history_raw=history_raw,
            metadata=metadata,
            market_to_index=market_to_index,
        )
        for case in summary.get("cases", [])
        if isinstance(case, dict)
    ]
    public_cases = []
    for case in cases:
        public_cases.append(
            {key: value for key, value in case.items() if key != "_weighted_prefix"}
        )
    market_labels = [label for label, _index in market_specs]
    return {
        "source_summary": str(summary_path),
        "support_bank_source": bank_source,
        "fixed_start_index": summary.get("fixed_start_index"),
        "case_count": len(cases),
        "market_specs": [
            {"label": label, "index": int(index)} for label, index in market_specs
        ],
        "support_jaccard": _support_jaccard(cases),
        "pairwise_prefix_signature_distance": _pairwise_signature_distance(
            cases,
            market_labels,
        ),
        "cases": public_cases,
        "scope_note": (
            "Prefix-level audit over selected historical 30-day supports. It "
            "checks recent-prefix mechanics before frozen-generator rollout, so "
            "future generated behavior is not used as a selection target."
        ),
    }


def plot_selected_prefix_paths(
    report: dict[str, Any],
    output_path: str | Path,
    *,
    support_bank_npz: str | Path | None = None,
    bridge_report: str | Path | None = None,
    checkpoint: str | Path | None = None,
    max_markets: int = 4,
) -> str:
    """Plot raw selected-support prefix paths by narrative and market."""

    if support_bank_npz is not None:
        bank = _load_npz_support_bank(support_bank_npz)
    else:
        bank = _load_default_bridge_support_bank(
            bridge_report_path=bridge_report,
            checkpoint_path=checkpoint,
        )
    history_raw = np.asarray(bank["history_raw"], dtype=np.float32)
    cases = [case for case in report.get("cases", []) if isinstance(case, dict)]
    markets = [
        (str(spec["label"]), int(spec["index"]))
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
    days = np.arange(history_raw.shape[1])
    for row, case in enumerate(cases):
        for col, (market, index) in enumerate(markets):
            ax = axes[row, col]
            for comp_idx, component in enumerate(case.get("components", [])):
                if not isinstance(component, dict):
                    continue
                bridge_idx = int(component["bridge_local_index"])
                weight = float(component.get("weight", 0.0))
                prefix = history_raw[bridge_idx, :, index]
                ax.plot(
                    days,
                    prefix,
                    color=COLORS[comp_idx % len(COLORS)],
                    linewidth=1.0 + 2.0 * weight,
                    alpha=0.9,
                    label=f"{component.get('window_id', bridge_idx)} ({weight:.0%})",
                )
            if row == 0:
                ax.set_title(market, fontsize=10)
            if col == 0:
                ax.set_ylabel(str(case.get("label", "")), fontsize=8)
            ax.grid(axis="y", alpha=0.14)
    for ax in axes[-1, :]:
        ax.set_xlabel("Recent-prefix day")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles[:4],
            labels[:4],
            loc="lower center",
            ncol=min(4, len(labels)),
            fontsize=7,
            frameon=False,
        )
    fig.suptitle(
        f"Selected historical 30-day prefixes before rollout (start {report.get('fixed_start_index')})",
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
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--support-bank-npz", type=Path, default=None)
    parser.add_argument("--bridge-report", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--max-markets", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument(
        "--market",
        action="append",
        default=None,
        help="Market label/index pair, for example SPX:25. Repeatable.",
    )
    args = parser.parse_args()

    report = build_prefix_level_conditionality_audit(
        args.summary,
        support_bank_npz=args.support_bank_npz,
        bridge_report=args.bridge_report,
        checkpoint=args.checkpoint,
        markets=_parse_market_specs(args.market),
    )
    if not args.no_plot:
        report["artifact_paths"] = {
            "selected_prefix_plot": plot_selected_prefix_paths(
                report,
                args.plot,
                support_bank_npz=args.support_bank_npz,
                bridge_report=args.bridge_report,
                checkpoint=args.checkpoint,
                max_markets=args.max_markets,
            )
        }
    _write_json(args.output, report)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "plot": report.get("artifact_paths", {}).get(
                    "selected_prefix_plot", ""
                ),
                "case_count": report["case_count"],
                "support_jaccard_max": report["support_jaccard"]["max"],
                "prefix_signature_distance_median": report[
                    "pairwise_prefix_signature_distance"
                ]["median"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
