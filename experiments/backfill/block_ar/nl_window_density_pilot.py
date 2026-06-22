#!/usr/bin/env python
"""Measure narrative redundancy across overlapping 30-day episode windows.

The pilot compares anchor windows with +1, +5, and half-overlap style offsets
using existing Codex-authored episode cards plus local structured factor
signatures. It does not author new narrative prose locally.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar import nl_14_view_variant_pilot as pilot  # noqa: E402
from experiments.backfill.block_ar.nl_hard_negative_bank_regenerate import (  # noqa: E402
    FACTORS,
    _mechanical_summary,
    _sign_vector,
    _window_number,
)
from experiments.backfill.block_ar.nl_sparse_variant_pilot import (  # noqa: E402
    DEFAULT_CARDS_JSONL,
    _read_jsonl,
    _token_jaccard,
)


DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "window_density_pilot_987a"
)
DEFAULT_OFFSETS = (0, 1, 5, 15)
VIEW_NAMES_FOR_SIMILARITY = tuple(
    view for view in pilot.OLD_EIGHT_VIEW_NAMES if view != "old_factor_baseline"
)
SPREAD_FACTORS = {"BBB_OAS", "AAA_OAS"}


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _compact(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _window_id_for_number(number: int) -> str:
    return f"joint39_train_{int(number):04d}"


def _sign_label(factor: str, sign: int) -> str:
    sign = int(sign)
    if sign == 0:
        return "flat"
    if factor in SPREAD_FACTORS:
        return "wider" if sign > 0 else "tighter"
    return "up" if sign > 0 else "down"


def _view_text(card: dict[str, Any], view_name: str) -> str:
    views = card.get("views") if isinstance(card.get("views"), dict) else {}
    return _compact(views.get(view_name))


def average_view_similarity(anchor: dict[str, Any], offset_card: dict[str, Any]) -> float:
    """Average token Jaccard similarity across authored narrative views."""

    scores: list[float] = []
    for view_name in VIEW_NAMES_FOR_SIMILARITY:
        left = _view_text(anchor, view_name)
        right = _view_text(offset_card, view_name)
        if left and right:
            scores.append(float(_token_jaccard(left, right)))
    if not scores:
        return 0.0
    return float(np.mean(scores))


def _factor_change_rows(left: Iterable[int], right: Iterable[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for factor, left_sign, right_sign in zip(FACTORS, left, right):
        if int(left_sign) == int(right_sign):
            continue
        rows.append(
            {
                "factor": factor,
                "anchor_direction": _sign_label(factor, int(left_sign)),
                "offset_direction": _sign_label(factor, int(right_sign)),
                "is_opposite": bool(
                    int(left_sign) != 0
                    and int(right_sign) != 0
                    and int(left_sign) * int(right_sign) < 0
                ),
            }
        )
    return rows


def _notable_changes(change_rows: list[dict[str, Any]], *, limit: int = 5) -> list[str]:
    ordered = sorted(change_rows, key=lambda row: (not row["is_opposite"], row["factor"]))
    return [
        (
            f"{row['factor']}: {row['anchor_direction']} -> "
            f"{row['offset_direction']}"
        )
        for row in ordered[:limit]
    ]


def _human_label(
    *,
    offset: int,
    view_similarity: float,
    sign_change_count: int,
    opposite_sign_count: int,
    same_archetype: bool,
) -> str:
    if int(offset) == 0:
        return "anchor"
    if (
        view_similarity >= 0.68
        and sign_change_count <= 2
        and opposite_sign_count <= 1
        and same_archetype
    ):
        return "mostly redundant"
    if opposite_sign_count >= 2 or (not same_archetype and view_similarity < 0.55):
        return "distinct regime"
    return "partial change"


def build_density_records(
    *,
    cards: list[dict[str, Any]],
    anchors: list[str],
    offsets: list[int],
    sign_vector_fn: Callable[[dict[str, Any]], Iterable[int]] = _sign_vector,
) -> list[dict[str, Any]]:
    """Build per-anchor/per-offset comparison records."""

    by_number = {_window_number(str(card.get("window_id", ""))): card for card in cards}
    by_id = {str(card.get("window_id", "")): card for card in cards}
    records: list[dict[str, Any]] = []
    for anchor_id in anchors:
        anchor = by_id[str(anchor_id)]
        anchor_number = _window_number(anchor_id)
        anchor_signs = np.asarray(list(sign_vector_fn(anchor)), dtype=np.int8)
        for offset in offsets:
            offset_number = anchor_number + int(offset)
            offset_card = by_number.get(offset_number)
            if offset_card is None:
                continue
            offset_signs = np.asarray(list(sign_vector_fn(offset_card)), dtype=np.int8)
            products = anchor_signs * offset_signs
            change_rows = _factor_change_rows(anchor_signs, offset_signs)
            view_similarity = average_view_similarity(anchor, offset_card)
            same_archetype = str(anchor.get("archetype", "")) == str(
                offset_card.get("archetype", "")
            )
            sign_change_count = int(np.sum(anchor_signs != offset_signs))
            opposite_sign_count = int(np.sum(products < 0))
            label = _human_label(
                offset=int(offset),
                view_similarity=view_similarity,
                sign_change_count=sign_change_count,
                opposite_sign_count=opposite_sign_count,
                same_archetype=same_archetype,
            )
            records.append(
                {
                    "anchor_window_id": str(anchor.get("window_id", "")),
                    "offset_window_id": str(offset_card.get("window_id", "")),
                    "offset": int(offset),
                    "calendar_overlap_days": max(0, 30 - int(offset)),
                    "calendar_overlap_fraction": max(0.0, (30 - int(offset)) / 30.0),
                    "anchor_archetype": _compact(anchor.get("archetype")),
                    "offset_archetype": _compact(offset_card.get("archetype")),
                    "same_archetype": same_archetype,
                    "anchor_title": _compact(anchor.get("scenario_title")),
                    "offset_title": _compact(offset_card.get("scenario_title")),
                    "anchor_mechanical_summary": _mechanical_summary(anchor),
                    "offset_mechanical_summary": _mechanical_summary(offset_card),
                    "view_similarity": _round(view_similarity),
                    "sign_change_count": sign_change_count,
                    "opposite_sign_count": opposite_sign_count,
                    "factor_changes": change_rows,
                    "notable_changes": _notable_changes(change_rows),
                    "human_label": label,
                }
            )
    return records


def select_density_anchors(
    *,
    cards: list[dict[str, Any]],
    target_count: int,
    max_per_archetype: int,
    offsets: list[int],
) -> list[str]:
    """Select anchor windows spread across archetypes and time."""

    max_offset = max(int(offset) for offset in offsets)
    by_number = {_window_number(str(card.get("window_id", ""))): card for card in cards}
    feasible = [
        card
        for card in cards
        if all((_window_number(str(card.get("window_id", ""))) + int(offset)) in by_number for offset in offsets)
        and _window_number(str(card.get("window_id", ""))) + max_offset <= max(by_number)
    ]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for card in feasible:
        grouped[_compact(card.get("archetype")) or "unknown"].append(card)
    for rows in grouped.values():
        rows.sort(key=lambda card: _window_number(str(card.get("window_id", ""))))

    selected: list[str] = []
    selected_set: set[str] = set()
    archetypes = sorted(grouped, key=lambda name: (-len(grouped[name]), name))
    for archetype in archetypes:
        if len(selected) >= int(target_count):
            break
        rows = grouped[archetype]
        quota = min(int(max_per_archetype), len(rows), int(target_count) - len(selected))
        if quota <= 0:
            continue
        positions = np.linspace(0, len(rows) - 1, num=quota)
        for pos in positions:
            window_id = str(rows[int(round(float(pos)))].get("window_id", ""))
            if window_id and window_id not in selected_set:
                selected.append(window_id)
                selected_set.add(window_id)

    if len(selected) < int(target_count):
        feasible_sorted = sorted(
            feasible, key=lambda card: _window_number(str(card.get("window_id", "")))
        )
        positions = np.linspace(0, len(feasible_sorted) - 1, num=int(target_count))
        for pos in positions:
            if len(selected) >= int(target_count):
                break
            window_id = str(feasible_sorted[int(round(float(pos)))].get("window_id", ""))
            if window_id and window_id not in selected_set:
                selected.append(window_id)
                selected_set.add(window_id)
    return selected[: int(target_count)]


def summarize_density(records: list[dict[str, Any]], *, anchor_count: int) -> dict[str, Any]:
    """Summarize density records by offset."""

    by_offset: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        by_offset[int(row["offset"])].append(row)
    offsets: dict[int, dict[str, Any]] = {}
    for offset, rows in sorted(by_offset.items()):
        labels = Counter(str(row.get("human_label", "unknown")) for row in rows)
        offsets[offset] = {
            "case_count": len(rows),
            "mean_calendar_overlap_fraction": _round(
                np.mean([float(row["calendar_overlap_fraction"]) for row in rows])
            ),
            "mean_view_similarity": _round(
                np.mean([float(row["view_similarity"]) for row in rows])
            ),
            "median_view_similarity": _round(
                np.median([float(row["view_similarity"]) for row in rows])
            ),
            "mean_sign_change_count": _round(
                np.mean([float(row["sign_change_count"]) for row in rows])
            ),
            "mean_opposite_sign_count": _round(
                np.mean([float(row["opposite_sign_count"]) for row in rows])
            ),
            "same_archetype_share": _round(
                np.mean([1.0 if row["same_archetype"] else 0.0 for row in rows])
            ),
            "distinct_regime_share": _round(
                labels.get("distinct regime", 0) / max(1, len(rows))
            ),
            "label_counts": dict(sorted(labels.items())),
        }
    return {
        "anchor_count": int(anchor_count),
        "record_count": len(records),
        "offsets": offsets,
    }


def _offset(summary: dict[str, Any], offset: int) -> dict[str, Any] | None:
    offsets = summary.get("offsets", {})
    return offsets.get(offset) or offsets.get(str(offset))


def recommend_stride(summary: dict[str, Any]) -> dict[str, Any]:
    """Recommend first production-bank stride from offset summary."""

    one = _offset(summary, 1) or {}
    five = _offset(summary, 5) or {}
    fifteen = _offset(summary, 15) or {}
    one_redundant = (
        float(one.get("mean_view_similarity", 0.0)) >= 0.70
        and float(one.get("mean_sign_change_count", 99.0)) <= 2.0
        and float(one.get("distinct_regime_share", 1.0)) < 0.25
    )
    five_redundant = (
        float(five.get("mean_view_similarity", 0.0)) >= 0.70
        and float(five.get("mean_sign_change_count", 99.0)) <= 2.0
        and float(five.get("distinct_regime_share", 1.0)) < 0.25
    )
    five_distinct_enough = (
        float(five.get("mean_view_similarity", 1.0)) <= 0.68
        or float(five.get("mean_sign_change_count", 0.0)) >= 2.5
        or float(five.get("mean_opposite_sign_count", 0.0)) >= 1.0
        or float(five.get("distinct_regime_share", 0.0)) >= 0.30
    )
    if five_redundant and fifteen:
        return {
            "recommended_stride_days": 15,
            "decision": "Use half-overlapping windows for the first production bank.",
            "rationale": (
                "Daily windows are redundant, and 5-day windows still look redundant "
                "on narrative/factor movement. Half-overlap gives a cleaner first "
                "bank with less repeated prose."
            ),
            "follow_up": (
                "Backfill denser windows only around detected regime transitions or "
                "retrieval coverage gaps."
            ),
        }
    if one_redundant and five_distinct_enough:
        return {
            "recommended_stride_days": 5,
            "decision": "Generate every 5 trading days first.",
            "rationale": (
                "The daily offset is mostly redundant, while the 5-day offset changes "
                "enough factor/narrative content to justify a denser bank than pure "
                "half-overlap. This keeps coverage without paying for near-duplicate "
                "daily prose."
            ),
            "follow_up": (
                "Keep all mechanical daily sidecars and selectively fill daily windows "
                "near high-change transition clusters."
            ),
        }
    return {
        "recommended_stride_days": 5,
        "decision": "Use every 5 trading days as the conservative first bank.",
        "rationale": (
            "Daily shifts show substantial prose/title churn, but much less hard "
            "directional movement than wider offsets. Five-day spacing captures "
            "more factor movement at roughly one-fifth of daily narrative cost, "
            "while half-overlap is too sparse for the first retrieval bank."
        ),
        "follow_up": (
            "Audit retrieval misses after the first bank and expand locally where "
            "needed."
        ),
    }


def estimate_bank_sizes(*, total_windows: int, stride_days: int) -> dict[str, int]:
    anchor_count = int(math.ceil(total_windows / max(1, int(stride_days))))
    return {
        "anchor_count": anchor_count,
        "positive_narratives": anchor_count * 14,
        "negative_narratives": anchor_count * 14,
        "total_narratives": anchor_count * 28,
    }


def _select_review_records(records: list[dict[str, Any]], *, max_records: int) -> list[dict[str, Any]]:
    if len(records) <= int(max_records):
        return records
    non_anchor = [row for row in records if int(row.get("offset", 0)) != 0]
    label_order = {"mostly redundant": 0, "partial change": 1, "distinct regime": 2}
    selected: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, int]] = set()
    offsets = sorted({int(row.get("offset", 0)) for row in non_anchor})
    per_offset_reserve = max(1, min(2, int(max_records) // max(1, len(offsets))))
    ranked_by_offset: dict[int, list[dict[str, Any]]] = {}
    for offset in offsets:
        candidates = [row for row in non_anchor if int(row.get("offset", 0)) == offset]
        candidates.sort(
            key=lambda row: (
                label_order.get(str(row.get("human_label")), 99),
                abs(float(row.get("view_similarity", 0.0)) - 0.4),
                str(row.get("anchor_window_id")),
            )
        )
        ranked_by_offset[offset] = candidates
    for rank in range(per_offset_reserve):
        for offset in offsets:
            if len(selected) >= int(max_records):
                break
            candidates = ranked_by_offset.get(offset, [])
            if rank >= len(candidates):
                continue
            row = candidates[rank]
            key = (str(row["anchor_window_id"]), int(row["offset"]))
            if key not in seen_keys:
                selected.append(row)
                seen_keys.add(key)
    for label in ("mostly redundant", "partial change", "distinct regime"):
        candidates = [row for row in non_anchor if row.get("human_label") == label]
        candidates.sort(
            key=lambda row: (
                int(row.get("offset", 0)),
                -float(row.get("view_similarity", 0.0))
                if label == "mostly redundant"
                else float(row.get("view_similarity", 0.0)),
            )
        )
        for row in candidates[:2]:
            key = (str(row["anchor_window_id"]), int(row["offset"]))
            if key not in seen_keys:
                selected.append(row)
                seen_keys.add(key)
    if len(selected) < int(max_records):
        for row in sorted(
            non_anchor,
            key=lambda item: (
                label_order.get(str(item.get("human_label")), 99),
                int(item.get("offset", 0)),
                str(item.get("anchor_window_id")),
            ),
        ):
            key = (str(row["anchor_window_id"]), int(row["offset"]))
            if key in seen_keys:
                continue
            selected.append(row)
            seen_keys.add(key)
            if len(selected) >= int(max_records):
                break
    return selected[: int(max_records)]


def _estimate_markdown(total_windows: int, recommendation: dict[str, Any]) -> list[str]:
    rows = []
    for stride in (1, 5, 15, int(recommendation["recommended_stride_days"])):
        if stride in {row["stride"] for row in rows}:
            continue
        estimate = estimate_bank_sizes(total_windows=total_windows, stride_days=stride)
        rows.append({"stride": stride, **estimate})
    lines = [
        "| Stride | Anchor episodes | Positives | Negatives | Total narratives |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['stride']} | {row['anchor_count']} | "
            f"{row['positive_narratives']} | {row['negative_narratives']} | "
            f"{row['total_narratives']} |"
        )
    return lines


def build_human_review_markdown(report: dict[str, Any]) -> str:
    """Build a low-cognitive-load review packet."""

    recommendation = report["recommendation"]
    summary = report["summary"]
    total_windows = int(report.get("total_windows", 4010))
    lines: list[str] = [
        "# Window-Density Pilot",
        "",
        f"Generated: {report.get('generated_date', date.today().isoformat())}",
        "",
        "This pilot compares overlapping 30-day historical condition windows. "
        "It uses existing Codex-authored episode cards plus local structured "
        "factor signatures; it does not generate new training prose.",
        "",
        "Interpretation note: raw prose similarity is a harsh diagnostic because "
        "the same nearby market state can be paraphrased differently. For stride "
        "selection, put more weight on sign changes, opposite signs, and review "
        "tiles than on text similarity alone.",
        "",
        "## Decision Card",
        "",
        f"- Recommendation: **{recommendation['decision']}**",
        f"- Recommended stride: `{recommendation['recommended_stride_days']}` trading days",
        f"- Why: {recommendation['rationale']}",
        f"- Follow-up: {recommendation.get('follow_up', 'Review retrieval coverage after the first bank.')}",
        "",
        "Read this section and the offset summary first. The review tiles are "
        "only for sanity checking representative examples.",
        "",
        "## Bank Size Impact",
        "",
        *_estimate_markdown(total_windows, recommendation),
        "",
        "## Offset Summary",
        "",
        f"- Anchor count: `{summary['anchor_count']}`",
        f"- Comparison rows: `{summary.get('record_count', len(report.get('records', [])))}`",
        "",
        "| Offset | 30-day overlap | Mean view similarity | Mean sign changes | Mean opposite signs | Same archetype | Distinct-regime share | Label counts |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for offset, row in sorted(summary["offsets"].items(), key=lambda item: int(item[0])):
        overlap_fraction = row.get(
            "mean_calendar_overlap_fraction",
            max(0.0, (30 - int(offset)) / 30.0),
        )
        lines.append(
            f"| {offset} | {overlap_fraction:.2f} | "
            f"{row['mean_view_similarity']:.2f} | "
            f"{row['mean_sign_change_count']:.2f} | "
            f"{row['mean_opposite_sign_count']:.2f} | "
            f"{row.get('same_archetype_share', 0.0):.2f} | "
            f"{row['distinct_regime_share']:.2f} | "
            f"`{json.dumps(row.get('label_counts', {}), sort_keys=True)}` |"
        )
    lines.extend(
        [
            "",
            "## Low-Load Review Tiles",
            "",
            "Each tile compares one anchor condition with one shifted condition. "
            "The goal is to see whether a shifted window still sounds like the "
            "same condition or has moved enough to deserve its own authored "
            "14+14 packet.",
            "",
        ]
    )
    for idx, row in enumerate(report.get("review_records", []), 1):
        lines.extend(
            [
                f"### Tile {idx}: {row['anchor_window_id']} -> {row['offset_window_id']} (+{row['offset']})",
                "",
                f"- Label: `{row['human_label']}`",
                f"- View similarity: `{row['view_similarity']}`",
                f"- Sign changes: `{row['sign_change_count']}`; opposite signs: `{row['opposite_sign_count']}`",
                f"- Same archetype: `{row['same_archetype']}`",
                "",
                "Anchor:",
                "",
                f"> {row['anchor_title']}",
                "",
                f"> {row['anchor_mechanical_summary']}",
                "",
                "Shifted window:",
                "",
                f"> {row['offset_title']}",
                "",
                f"> {row['offset_mechanical_summary']}",
                "",
                "Notable changes:",
                "",
            ]
        )
        if row.get("notable_changes"):
            lines.extend(f"- {item}" for item in row["notable_changes"])
        else:
            lines.append("- No factor-direction changes.")
        lines.append("")
    lines.extend(
        [
            "## Full Row Data",
            "",
            "The Markdown intentionally omits the full comparison table to keep review "
            "load low. The machine-readable JSON report contains every row.",
            "",
            f"- JSON report: `{report.get('artifact_paths', {}).get('report', '')}`",
        ]
    )
    lines.append("")
    return "\n".join(lines)


def run_window_density_pilot(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = _resolve(Path(args.output_dir))
    cards = _read_jsonl(Path(args.cards_jsonl))
    offsets = [int(item) for item in args.offsets]
    if args.anchor_window_ids:
        anchors = [str(item) for item in args.anchor_window_ids]
    else:
        anchors = select_density_anchors(
            cards=cards,
            target_count=int(args.anchor_count),
            max_per_archetype=int(args.max_per_archetype),
            offsets=offsets,
        )
    records = build_density_records(cards=cards, anchors=anchors, offsets=offsets)
    summary = summarize_density(records, anchor_count=len(anchors))
    recommendation = recommend_stride(summary)
    review_records = _select_review_records(
        records,
        max_records=int(args.max_review_tiles),
    )
    report_path = output_dir / "window_density_pilot_report.json"
    markdown_path = output_dir / "window_density_pilot_review.md"
    report = {
        "schema_version": "window_density_pilot_v1",
        "status": "pass" if records else "fail",
        "generated_date": date.today().isoformat(),
        "cards_jsonl": str(args.cards_jsonl),
        "total_windows": len(cards),
        "offsets": offsets,
        "anchors": anchors,
        "summary": summary,
        "recommendation": recommendation,
        "records": records,
        "review_records": review_records,
        "local_prose_generated": False,
        "artifact_paths": {
            "report": str(report_path),
            "markdown": str(markdown_path),
        },
    }
    _write_json(report_path, report)
    _write_text(markdown_path, build_human_review_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-jsonl", type=Path, default=DEFAULT_CARDS_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--anchor-count", type=int, default=20)
    parser.add_argument("--max-per-archetype", type=int, default=2)
    parser.add_argument("--offsets", type=int, nargs="+", default=list(DEFAULT_OFFSETS))
    parser.add_argument("--anchor-window-ids", nargs="*", default=[])
    parser.add_argument("--max-review-tiles", type=int, default=8)
    args = parser.parse_args()
    report = run_window_density_pilot(args)
    print(
        json.dumps(
            {
                "status": report["status"],
                "anchor_count": len(report["anchors"]),
                "recommended_stride_days": report["recommendation"][
                    "recommended_stride_days"
                ],
                "report": report["artifact_paths"]["report"],
                "markdown": report["artifact_paths"]["markdown"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
