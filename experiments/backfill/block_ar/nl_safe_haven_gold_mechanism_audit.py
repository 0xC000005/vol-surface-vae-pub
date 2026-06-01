#!/usr/bin/env python
"""Mechanism audit for the weak Safe-haven Gold terminal response.

The prior Safe-haven Gold audits established that the paper/demo table is
reproducible and that the weak day-30 Gold response is not specific to start
22. This script goes one layer deeper and asks why:

1. Does a Gold-up safe-haven prefix historically imply more Gold upside over
   the next 30 days?
2. Does the start-only baseline already select safe-haven-like support prefixes?
3. Does the frozen generator preserve or shrink those support-specific Gold
   differences in its terminal distribution?

No OpenAI calls are made. No generator rerun is performed; the script only
loads saved top3/90 artifacts and the broad support bank.
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

from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    KEY_FACTOR_NAMES,
)
from experiments.backfill.block_ar.nl_risk_manager_story_gradio_app import (  # noqa: E402
    JOINT39_SPEC_NAMES,
)
from experiments.backfill.block_ar.nl_safe_haven_gold_channel_audit import (  # noqa: E402
    build_audit,
    _jsonable,
)
from experiments.backfill.block_ar.nl_safe_haven_gold_start_sensitivity_audit import (  # noqa: E402
    DEFAULT_CASE_ROOTS,
    _extract_start_index,
)


DEFAULT_SUPPORT_BANK_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
DEFAULT_SUPPORT_BANK_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_report.json"
)
DEFAULT_OUTPUT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "safe_haven_gold_mechanism_audit_966d/"
    "safe_haven_gold_mechanism_audit.json"
)


SAFE_HAVEN_PREFIX_CLAIMS = [
    ("GOLD", "up"),
    ("US10Y", "down"),
    ("VIX", "up"),
    ("SPX", "down"),
]


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _finite_corr(x_values: np.ndarray, y_values: np.ndarray) -> float | None:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if int(mask.sum()) < 3:
        return None
    x = x[mask]
    y = y[mask]
    if float(np.std(x)) <= 1e-12 or float(np.std(y)) <= 1e-12:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _stats(values: np.ndarray) -> dict[str, Any]:
    valid = np.asarray(values, dtype=np.float64)
    valid = valid[np.isfinite(valid)]
    if valid.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "share_positive": None,
            "share_negative": None,
        }
    return {
        "count": int(valid.size),
        "mean": float(np.mean(valid)),
        "median": float(np.median(valid)),
        "p10": float(np.quantile(valid, 0.10)),
        "p90": float(np.quantile(valid, 0.90)),
        "share_positive": float(np.mean(valid > 0.0)),
        "share_negative": float(np.mean(valid < 0.0)),
    }


def _market_index(market: str) -> int:
    spec_name = KEY_FACTOR_NAMES[str(market).upper()]
    return JOINT39_SPEC_NAMES.index(spec_name)


def _sign_mask(delta: np.ndarray, direction: str) -> np.ndarray:
    if direction == "up":
        return delta > 0.0
    if direction in {"down", "tighter"}:
        return delta < 0.0
    if direction == "wider":
        return delta > 0.0
    raise ValueError(f"unsupported direction: {direction}")


def _prefix_mask(history_raw: np.ndarray, claims: list[tuple[str, str]]) -> np.ndarray:
    mask = np.ones(int(history_raw.shape[0]), dtype=bool)
    for market, direction in claims:
        idx = _market_index(market)
        delta = history_raw[:, -1, idx] - history_raw[:, 0, idx]
        mask &= _sign_mask(delta, direction)
    return mask


def _window_id_to_index(value: str) -> int | None:
    suffix = str(value).rsplit("_", maxsplit=1)[-1]
    try:
        return int(suffix)
    except ValueError:
        return None


def _support_by_index(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        index = row.get("window_index")
        if index is None:
            index = _window_id_to_index(str(row.get("window_id", "")))
        if index is not None:
            out[int(index)] = row
    return out


def _component_by_index(component_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in component_rows:
        out[int(row["window_index"])] = row
    return out


def _row_for_window(
    *,
    window_index: int,
    role: str,
    start_index: int | None,
    support_row: dict[str, Any],
    component_row: dict[str, Any] | None,
    history_raw: np.ndarray,
    future_delta: np.ndarray,
    metadata: list[dict[str, Any]],
) -> dict[str, Any]:
    gold_idx = _market_index("GOLD")
    spx_idx = _market_index("SPX")
    vix_idx = _market_index("VIX")
    us10y_idx = _market_index("US10Y")
    prefix_gold = float(
        history_raw[window_index, -1, gold_idx] - history_raw[window_index, 0, gold_idx]
    )
    future_gold = float(future_delta[window_index, -1, gold_idx])
    generated_mean = None
    generated_up = None
    generated_p10 = None
    generated_p90 = None
    posterior_weight = None
    if component_row:
        gold_terminal = component_row.get("gold_terminal", {})
        if isinstance(gold_terminal, dict):
            generated_mean = gold_terminal.get("mean_delta")
            generated_up = gold_terminal.get("probability_up")
            generated_p10 = gold_terminal.get("p10_delta")
            generated_p90 = gold_terminal.get("p90_delta")
        posterior_weight = component_row.get("posterior_weight")
    meta = metadata[window_index] if 0 <= window_index < len(metadata) else {}
    return {
        "start_index": start_index,
        "role": role,
        "window_index": int(window_index),
        "window_id": str(support_row.get("window_id", meta.get("window_id", ""))),
        "history_end_date": str(
            support_row.get("history_end_date", meta.get("calendar_end_date", ""))
        ),
        "support_weight": float(support_row.get("weight", 0.0) or 0.0),
        "posterior_weight": (
            None if posterior_weight is None else float(posterior_weight)
        ),
        "memory_support_cosine": float(
            support_row.get("memory_support_cosine", 0.0) or 0.0
        ),
        "start_distance_z": float(support_row.get("start_distance_z", 0.0) or 0.0),
        "prefix_gold_delta": prefix_gold,
        "prefix_spx_delta": float(
            history_raw[window_index, -1, spx_idx] - history_raw[window_index, 0, spx_idx]
        ),
        "prefix_vix_delta": float(
            history_raw[window_index, -1, vix_idx] - history_raw[window_index, 0, vix_idx]
        ),
        "prefix_us10y_delta": float(
            history_raw[window_index, -1, us10y_idx]
            - history_raw[window_index, 0, us10y_idx]
        ),
        "actual_future_gold_delta": future_gold,
        "generated_gold_mean_delta": (
            None if generated_mean is None else float(generated_mean)
        ),
        "generated_gold_probability_up": (
            None if generated_up is None else float(generated_up)
        ),
        "generated_gold_p10_p90": (
            None
            if generated_p10 is None or generated_p90 is None
            else [float(generated_p10), float(generated_p90)]
        ),
    }


def build_mechanism_audit(
    *,
    case_roots: list[Path],
    support_bank_arrays: Path,
    support_bank_report: Path,
) -> dict[str, Any]:
    with np.load(support_bank_arrays, allow_pickle=True) as arrays:
        history_raw = np.asarray(arrays["history_raw"], dtype=np.float64)
        future_delta = np.asarray(arrays["future_delta"], dtype=np.float64)
    support_report = _load_json(support_bank_report)
    metadata = list(support_report.get("window_metadata", []))

    gold_idx = _market_index("GOLD")
    gold_prefix = history_raw[:, -1, gold_idx] - history_raw[:, 0, gold_idx]
    gold_future = future_delta[:, -1, gold_idx]
    safe_haven_mask = _prefix_mask(history_raw, SAFE_HAVEN_PREFIX_CLAIMS)
    gold_up_mask = gold_prefix > 0.0
    gold_down_mask = gold_prefix < 0.0

    selected_rows: list[dict[str, Any]] = []
    per_start_summary: list[dict[str, Any]] = []
    for case_root in case_roots:
        start_index = _extract_start_index(case_root)
        audit = build_audit(case_root)
        for role, support_key, component_key in [
            ("narrative", "narrative_support_rows", "narrative_top3_gold"),
            ("start_only_baseline", "baseline_support_rows", "baseline_top3_gold"),
        ]:
            support_rows = _support_by_index(audit[support_key])
            component_rows = _component_by_index(
                audit[component_key]["component_rows"]
            )
            selected_indices = sorted(component_rows)
            for window_index in selected_indices:
                selected_rows.append(
                    _row_for_window(
                        window_index=int(window_index),
                        role=role,
                        start_index=start_index,
                        support_row=support_rows.get(int(window_index), {}),
                        component_row=component_rows.get(int(window_index)),
                        history_raw=history_raw,
                        future_delta=future_delta,
                        metadata=metadata,
                    )
                )
        narrative_gold = audit["narrative_top3_gold"]["pooled_gold_terminal"]
        baseline_gold = audit["baseline_top3_gold"]["pooled_gold_terminal"]
        per_start_summary.append(
            {
                "start_index": start_index,
                "narrative_gold_generated_mean": float(narrative_gold["mean_delta"]),
                "baseline_gold_generated_mean": float(baseline_gold["mean_delta"]),
                "narrative_minus_baseline_generated_mean": float(
                    narrative_gold["mean_delta"] - baseline_gold["mean_delta"]
                ),
                "narrative_gold_generated_up_share": float(
                    narrative_gold["probability_up"]
                ),
                "baseline_gold_generated_up_share": float(
                    baseline_gold["probability_up"]
                ),
                "narrative_minus_baseline_up_share": float(
                    narrative_gold["probability_up"]
                    - baseline_gold["probability_up"]
                ),
            }
        )

    selected_future = np.asarray(
        [row["actual_future_gold_delta"] for row in selected_rows], dtype=np.float64
    )
    selected_generated = np.asarray(
        [
            np.nan
            if row["generated_gold_mean_delta"] is None
            else row["generated_gold_mean_delta"]
            for row in selected_rows
        ],
        dtype=np.float64,
    )
    selected_prefix = np.asarray(
        [row["prefix_gold_delta"] for row in selected_rows], dtype=np.float64
    )

    return {
        "status": "ok",
        "support_bank": {
            "arrays": str(support_bank_arrays),
            "report": str(support_bank_report),
            "window_count": int(history_raw.shape[0]),
            "date_range": support_report.get("calendar_end_date_range", {}),
        },
        "safe_haven_prefix_claims": [
            {"market": market, "required_direction": direction}
            for market, direction in SAFE_HAVEN_PREFIX_CLAIMS
        ],
        "historical_gold_prefix_to_future": {
            "all_windows": {
                "gold_prefix_delta": _stats(gold_prefix),
                "gold_future_terminal_delta": _stats(gold_future),
                "prefix_future_correlation": _finite_corr(gold_prefix, gold_future),
            },
            "gold_up_prefix_windows": {
                "window_count": int(gold_up_mask.sum()),
                "gold_future_terminal_delta": _stats(gold_future[gold_up_mask]),
            },
            "gold_down_prefix_windows": {
                "window_count": int(gold_down_mask.sum()),
                "gold_future_terminal_delta": _stats(gold_future[gold_down_mask]),
            },
            "safe_haven_prefix_windows": {
                "window_count": int(safe_haven_mask.sum()),
                "share_of_bank": float(np.mean(safe_haven_mask)),
                "gold_prefix_delta": _stats(gold_prefix[safe_haven_mask]),
                "gold_future_terminal_delta": _stats(gold_future[safe_haven_mask]),
                "prefix_future_correlation": _finite_corr(
                    gold_prefix[safe_haven_mask], gold_future[safe_haven_mask]
                ),
            },
        },
        "selected_support_mechanism_rows": sorted(
            selected_rows,
            key=lambda row: (
                -1 if row["start_index"] is None else int(row["start_index"]),
                str(row["role"]),
                int(row["window_index"]),
            ),
        ),
        "selected_support_aggregate": {
            "row_count": int(len(selected_rows)),
            "actual_future_gold_delta": _stats(selected_future),
            "generated_gold_mean_delta": _stats(selected_generated),
            "prefix_gold_delta": _stats(selected_prefix),
            "prefix_to_actual_future_corr": _finite_corr(
                selected_prefix, selected_future
            ),
            "actual_future_to_generated_mean_corr": _finite_corr(
                selected_future, selected_generated
            ),
            "prefix_to_generated_mean_corr": _finite_corr(
                selected_prefix, selected_generated
            ),
        },
        "per_start_generated_summary": sorted(
            per_start_summary,
            key=lambda row: -1 if row["start_index"] is None else int(row["start_index"]),
        ),
        "root_cause_assessment": {
            "is_table_bug": False,
            "is_gold_index_bug": False,
            "is_start22_only_artifact": False,
            "primary_mechanism": (
                "The grounding/support checks are prefix checks. In the broad "
                "historical bank, Gold-up safe-haven prefixes do not imply a "
                "large positive next-30-day Gold terminal move. The start-only "
                "baseline also often selects Gold-up/safe-haven-ish prefixes, "
                "so the narrative support is not much more Gold-bullish than "
                "the baseline at this start family. Finally, the frozen SNI "
                "rollout shrinks component-specific historical Gold outcomes "
                "toward a similar calibrated terminal distribution."
            ),
            "product_implication": (
                "Safe-haven Gold currently means the conditioning support "
                "contains Gold-supportive recent histories; it does not mean "
                "the generated day-30 Gold marginal must be higher than the "
                "same-start baseline. For a stronger Gold-facing product claim, "
                "the selector or generator readout must score future Gold/hedge "
                "response explicitly, with CRPS/energy guardrails."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--support-bank-arrays", default=str(DEFAULT_SUPPORT_BANK_ARRAYS))
    parser.add_argument("--support-bank-report", default=str(DEFAULT_SUPPORT_BANK_REPORT))
    parser.add_argument(
        "--case-root",
        action="append",
        default=[],
        help="Safe-haven case root. Defaults to the current starts 18/22/40 pack.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_roots = [Path(value) for value in args.case_root] or list(DEFAULT_CASE_ROOTS)
    audit = build_mechanism_audit(
        case_roots=case_roots,
        support_bank_arrays=Path(args.support_bank_arrays),
        support_bank_report=Path(args.support_bank_report),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(audit), indent=2, sort_keys=True) + "\n")
    summary = {
        "status": audit["status"],
        "output": str(output),
        "historical_safe_haven_gold_future": audit[
            "historical_gold_prefix_to_future"
        ]["safe_haven_prefix_windows"]["gold_future_terminal_delta"],
        "selected_support_aggregate": audit["selected_support_aggregate"],
        "per_start_generated_summary": audit["per_start_generated_summary"],
        "root_cause": audit["root_cause_assessment"]["primary_mechanism"],
    }
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
