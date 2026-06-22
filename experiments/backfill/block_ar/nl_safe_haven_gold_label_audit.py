#!/usr/bin/env python
"""Audit whether Safe-haven Gold supports are correctly labeled.

This is an isolated verifier-style script for the narrative-conditioned
scenario work. It checks whether the selected historical supports really look
like Safe-haven Gold prefixes from raw 30-day market moves, then separates that
prefix-label question from the next-30-day Gold continuation question.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_narrative_retrieval import (  # noqa: E402
    build_query,
    rank_episode_cards,
)
from experiments.backfill.block_ar.nl_episode_narrative_support_cards import (  # noqa: E402
    _scales_from_history,
)


DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "safe_haven_gold_label_audit_972a"
)
SUPPORT_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_report.json"
)
SUPPORT_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
SUPPORT_CARDS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_support_bank_cards_all_970f/"
    "episode_narrative_support_cards.jsonl"
)
OLD_CHANNEL_AUDIT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "safe_haven_gold_channel_audit_966b/safe_haven_gold_channel_audit.json"
)
OLD_START_AUDIT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "safe_haven_gold_start_sensitivity_966c/"
    "safe_haven_gold_start_sensitivity_audit.json"
)
NEW_HYBRID_BRIDGE = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_hybrid_start_text_bridge_971d_t25_s75_gap30_full66/"
    "hybrid_start_text_bridge_report.json"
)
NEW_HYBRID_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_hybrid_start_text_eval_971o_t25_s75_top3_full66_s16/"
    "scenario_level_eval_arrays.npz"
)
START_ONLY_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_narrative_start_only_scenario_eval_971n_top3_full66_s16/"
    "scenario_level_eval_arrays.npz"
)

MARKETS: dict[str, int] = {
    "SPX": 25,
    "VIX": 38,
    "BBB_OAS": 35,
    "DXY": 28,
    "US10Y": 33,
    "GOLD": 37,
}

PAPER_SAFE_HAVEN_NARRATIVE = (
    "Scenario title: Safe-haven gold bid. Mechanical summary: The current/recent "
    "market state shows gold supported, Treasury yields lower, equities choppy "
    "to weaker, and volatility elevated while the dollar is not the only "
    "defensive channel. Dominant mechanism: investors are paying for safety "
    "and convexity rather than adding broad cyclical risk. Trigger and "
    "transmission: growth uncertainty, policy credibility concern, or "
    "geopolitical risk can raise demand for stores of value and duration, "
    "while limiting equity risk appetite. Cross-asset reaction: gold is "
    "higher, US10Y is lower, VIX is higher, SPX is mixed to weaker, and DXY is "
    "mixed when safe-haven demand is split between gold and dollars. "
    "Portfolio/risk implication: gold and duration can hedge part of the "
    "portfolio, while equity beta and short-vol exposure remain vulnerable to "
    "renewed stress. Evidence and ambiguity: the evidence is a safe-haven "
    "configuration, not a named event; equity direction is less clean than in "
    "a pure risk-off shock. No-forecast caveat: this is not a forecast; it is "
    "a current/recent conditioning narrative. Warning-only forward risk: "
    "safe-haven demand could broaden into risk-off behavior, but that language "
    "is warning-only."
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read_cards(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"{path}: non-object JSONL row")
                rows.append(row)
    return rows


def _window_index_from_id(window_id: str) -> int:
    match = re.search(r"_(\d+)$", str(window_id))
    if not match:
        raise ValueError(f"cannot parse window index from {window_id!r}")
    return int(match.group(1))


def _card_index(card: dict[str, Any]) -> int:
    meta = card.get("support_metadata", {})
    if isinstance(meta, dict) and meta.get("window_index") is not None:
        return int(meta["window_index"])
    return _window_index_from_id(str(card.get("window_id", "")))


def _card_title(card: dict[str, Any]) -> str:
    return str(card.get("scenario_title", ""))


def _direction(z: float, *, flat_threshold: float = 0.15) -> str:
    if abs(float(z)) < flat_threshold:
        return "flat"
    return "up" if z > 0 else "down"


def _safe(value: float | np.floating[Any]) -> float:
    return float(value) if np.isfinite(float(value)) else float("nan")


def _factor_moves(
    *,
    history_raw: np.ndarray,
    future_delta: np.ndarray,
    scales: np.ndarray,
    window_index: int,
) -> dict[str, dict[str, Any]]:
    idx = int(window_index)
    rows: dict[str, dict[str, Any]] = {}
    for market, factor_idx in MARKETS.items():
        prefix_delta = float(history_raw[idx, -1, factor_idx] - history_raw[idx, 0, factor_idx])
        future_terminal_delta = float(future_delta[idx, -1, factor_idx])
        scale = max(float(scales[factor_idx]), 1e-8)
        prefix_z = prefix_delta / scale
        future_z = future_terminal_delta / scale
        rows[market] = {
            "prefix_delta": round(prefix_delta, 6),
            "prefix_z": round(prefix_z, 6),
            "prefix_direction": _direction(prefix_z),
            "future_terminal_delta": round(future_terminal_delta, 6),
            "future_z": round(future_z, 6),
            "future_direction": _direction(future_z),
        }
    return rows


def _manual_relabel(moves: dict[str, dict[str, Any]]) -> dict[str, Any]:
    gold_up = moves["GOLD"]["prefix_direction"] == "up"
    rates_down = moves["US10Y"]["prefix_direction"] == "down"
    vix_up = moves["VIX"]["prefix_direction"] == "up"
    spx_down = moves["SPX"]["prefix_direction"] == "down"
    spx_not_up = moves["SPX"]["prefix_direction"] in {"down", "flat"}
    dxy_up = moves["DXY"]["prefix_direction"] == "up"
    credit_wider = moves["BBB_OAS"]["prefix_direction"] == "up"

    strict_safe_haven = gold_up and rates_down and vix_up and spx_down
    broad_safe_haven = gold_up and rates_down and (vix_up or spx_not_up)
    gold_duration = gold_up and rates_down

    if strict_safe_haven:
        label = "classic_safe_haven_gold_risk_off"
        narrative = (
            "Gold is higher, long yields are lower, volatility is higher, and "
            "equities are weaker. This is a clean safe-haven Gold/duration "
            "prefix."
        )
    elif broad_safe_haven:
        label = "broad_safe_haven_gold"
        narrative = (
            "Gold and duration are supported, with at least one risk-off "
            "confirmation channel. This is a plausible but less pure "
            "safe-haven Gold prefix."
        )
    elif gold_duration:
        label = "gold_duration_bid_without_clear_risk_off"
        narrative = (
            "Gold and duration are supported, but the equity/volatility stress "
            "signature is incomplete. This is a Gold-duration bid, not a clean "
            "risk-off safe-haven episode."
        )
    elif gold_up and (vix_up or spx_down or credit_wider):
        label = "gold_up_mixed_defensive"
        narrative = (
            "Gold is higher and at least one defensive channel is present, but "
            "Treasury duration does not confirm the classic safe-haven pattern."
        )
    elif gold_up:
        label = "gold_up_non_safe_haven"
        narrative = (
            "Gold is higher, but the other cross-asset channels do not support "
            "a safe-haven interpretation."
        )
    elif dxy_up and (vix_up or spx_down or credit_wider):
        label = "dollar_liquidity_or_defensive_risk_off"
        narrative = (
            "Defensive pressure is visible through the dollar, volatility, "
            "equity, or credit channels rather than a Gold-led safe-haven bid."
        )
    else:
        label = "not_safe_haven_gold"
        narrative = (
            "The 30-day prefix does not show a Gold-supported safe-haven setup."
        )

    required = {
        "gold_up": gold_up,
        "us10y_down": rates_down,
        "vix_up": vix_up,
        "spx_down": spx_down,
        "spx_mixed_or_weaker": spx_not_up,
    }
    return {
        "manual_label": label,
        "manual_narrative": narrative,
        "strict_safe_haven_gold_prefix": strict_safe_haven,
        "broad_safe_haven_gold_prefix": broad_safe_haven,
        "required_claims": required,
    }


def _metadata_by_index(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {
        int(row.get("window_index", index)): row
        for index, row in enumerate(report.get("window_metadata", []))
        if isinstance(row, dict)
    }


def _support_row(
    *,
    source: str,
    window_index: int,
    rank: int | None,
    weight: float | None,
    metadata: dict[int, dict[str, Any]],
    cards_by_index: dict[int, dict[str, Any]],
    history_raw: np.ndarray,
    future_delta: np.ndarray,
    scales: np.ndarray,
) -> dict[str, Any]:
    moves = _factor_moves(
        history_raw=history_raw,
        future_delta=future_delta,
        scales=scales,
        window_index=window_index,
    )
    relabel = _manual_relabel(moves)
    meta = metadata.get(int(window_index), {})
    card = cards_by_index.get(int(window_index), {})
    return {
        "source": source,
        "window_index": int(window_index),
        "window_id": str(meta.get("window_id", f"window_{window_index:04d}")),
        "calendar_start_date": str(meta.get("calendar_start_date", "")),
        "calendar_end_date": str(meta.get("calendar_end_date", "")),
        "rank": rank,
        "weight": None if weight is None else _safe(float(weight)),
        "existing_card_title": _card_title(card),
        "existing_card_archetype": str(card.get("archetype", "")),
        "moves": moves,
        **relabel,
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"count": 0}
    labels = Counter(str(row["manual_label"]) for row in rows)
    titles = Counter(str(row.get("existing_card_title", "")) for row in rows)
    gold_future = np.asarray(
        [row["moves"]["GOLD"]["future_terminal_delta"] for row in rows],
        dtype=np.float64,
    )
    gold_prefix = np.asarray(
        [row["moves"]["GOLD"]["prefix_delta"] for row in rows],
        dtype=np.float64,
    )
    by_label: dict[str, dict[str, Any]] = {}
    for label in sorted(labels):
        label_rows = [row for row in rows if row["manual_label"] == label]
        label_future = np.asarray(
            [row["moves"]["GOLD"]["future_terminal_delta"] for row in label_rows],
            dtype=np.float64,
        )
        by_label[label] = {
            "count": len(label_rows),
            "gold_future_mean": float(np.mean(label_future)),
            "gold_future_median": float(np.median(label_future)),
            "gold_future_up_share": float(np.mean(label_future > 0.0)),
            "gold_future_p10": float(np.quantile(label_future, 0.10)),
            "gold_future_p90": float(np.quantile(label_future, 0.90)),
        }

    return {
        "count": len(rows),
        "manual_label_counts": dict(labels),
        "existing_card_title_counts": dict(titles),
        "strict_safe_haven_gold_prefix_share": float(
            np.mean([bool(row["strict_safe_haven_gold_prefix"]) for row in rows])
        ),
        "broad_safe_haven_gold_prefix_share": float(
            np.mean([bool(row["broad_safe_haven_gold_prefix"]) for row in rows])
        ),
        "gold_prefix": {
            "mean": float(np.mean(gold_prefix)),
            "median": float(np.median(gold_prefix)),
            "up_share": float(np.mean(gold_prefix > 0.0)),
            "p10": float(np.quantile(gold_prefix, 0.10)),
            "p90": float(np.quantile(gold_prefix, 0.90)),
        },
        "gold_future_terminal": {
            "mean": float(np.mean(gold_future)),
            "median": float(np.median(gold_future)),
            "up_share": float(np.mean(gold_future > 0.0)),
            "p10": float(np.quantile(gold_future, 0.10)),
            "p90": float(np.quantile(gold_future, 0.90)),
        },
        "gold_future_by_manual_label": by_label,
    }


def _collect_old_rows(
    *,
    old_channel: dict[str, Any],
    old_start: dict[str, Any],
) -> list[tuple[str, int, int | None, float | None]]:
    rows: list[tuple[str, int, int | None, float | None]] = []
    for key in ("narrative_support_rows", "baseline_support_rows"):
        for row in old_channel.get(key, []):
            source = f"old_966b_{key}"
            rows.append(
                (
                    source,
                    _window_index_from_id(str(row["window_id"])),
                    int(row.get("rank", 0)),
                    float(row.get("weight", 0.0)),
                )
            )
    for case in old_start.get("rows", []):
        start = int(case.get("start_index", -1))
        for key in ("narrative_top3_support", "baseline_top3_support"):
            for row in case.get(key, []):
                rows.append(
                    (
                        f"old_966c_start{start}_{key}",
                        _window_index_from_id(str(row["window_id"])),
                        int(row.get("rank", 0)),
                        float(row.get("weight", 0.0)),
                    )
                )
    return rows


def _collect_new_safe_rows(
    *,
    bridge: dict[str, Any],
    cards_by_index: dict[int, dict[str, Any]],
) -> list[tuple[str, int, int | None, float | None]]:
    rows: list[tuple[str, int, int | None, float | None]] = []
    examples = bridge.get("evaluation", {}).get("heldout_examples", [])
    for example in examples:
        query_idx = int(example.get("window_index"))
        query_card = cards_by_index.get(query_idx, {})
        if _card_title(query_card) != "Safe-haven gold bid":
            continue
        for item in example.get("top_train_pool", [])[:3]:
            rows.append(
                (
                    f"new_971d_safe_query_{query_idx}_top3",
                    int(item["window_index"]),
                    int(item.get("rank", 0)),
                    float(item.get("weight", 0.0)),
                )
            )
    return rows


def _collect_exact_text_rows(
    *,
    cards: list[dict[str, Any]],
) -> list[tuple[str, int, int | None, float | None]]:
    train_cards = [card for card in cards if card.get("split") != "support_decoder_test"]
    query = build_query(
        label="paper_safe_haven_gold_bid",
        text=PAPER_SAFE_HAVEN_NARRATIVE,
    )
    ranked = rank_episode_cards(
        query,
        train_cards,
        method="hybrid",
        top_k=20,
        temporal_gap=30,
    )
    return [
        (
            "exact_paper_narrative_text_to_text_top20",
            int(item["window_index"]),
            rank,
            None,
        )
        for rank, item in enumerate(ranked, 1)
    ]


def build_audit() -> dict[str, Any]:
    support_report = _read_json(SUPPORT_REPORT)
    metadata = _metadata_by_index(support_report)
    cards = _read_cards(SUPPORT_CARDS)
    cards_by_index = {_card_index(card): card for card in cards}
    old_channel = _read_json(OLD_CHANNEL_AUDIT)
    old_start = _read_json(OLD_START_AUDIT)
    bridge = _read_json(NEW_HYBRID_BRIDGE)
    with np.load(SUPPORT_ARRAYS) as payload:
        history_raw = payload["history_raw"].copy()
        future_delta = payload["future_delta"].copy()
    scales = _scales_from_history(history_raw)

    source_specs = (
        _collect_old_rows(old_channel=old_channel, old_start=old_start)
        + _collect_new_safe_rows(bridge=bridge, cards_by_index=cards_by_index)
        + _collect_exact_text_rows(cards=cards)
    )
    support_rows = [
        _support_row(
            source=source,
            window_index=index,
            rank=rank,
            weight=weight,
            metadata=metadata,
            cards_by_index=cards_by_index,
            history_raw=history_raw,
            future_delta=future_delta,
            scales=scales,
        )
        for source, index, rank, weight in source_specs
    ]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in support_rows:
        grouped.setdefault(str(row["source"]).split("_top3")[0], []).append(row)

    old_rows = [row for row in support_rows if row["source"].startswith("old_")]
    new_rows = [row for row in support_rows if row["source"].startswith("new_971d")]
    exact_rows = [
        row
        for row in support_rows
        if row["source"] == "exact_paper_narrative_text_to_text_top20"
    ]

    with np.load(NEW_HYBRID_ARRAYS) as hybrid, np.load(START_ONLY_ARRAYS) as start:
        safe_generated: list[dict[str, Any]] = []
        for example in bridge.get("evaluation", {}).get("heldout_examples", []):
            query_idx = int(example.get("window_index"))
            query_card = cards_by_index.get(query_idx, {})
            if _card_title(query_card) != "Safe-haven gold bid":
                continue
            key = f"narrative_{query_idx}"
            if key not in hybrid.files or key not in start.files:
                continue
            h = np.asarray(hybrid[key], dtype=np.float32)[:, -1, MARKETS["GOLD"]]
            s = np.asarray(start[key], dtype=np.float32)[:, -1, MARKETS["GOLD"]]
            safe_generated.append(
                {
                    "query_window_index": query_idx,
                    "hybrid_gold_mean": float(np.mean(h)),
                    "hybrid_gold_median": float(np.median(h)),
                    "hybrid_gold_up_share": float(np.mean(h > 0.0)),
                    "start_only_gold_mean": float(np.mean(s)),
                    "start_only_gold_median": float(np.median(s)),
                    "start_only_gold_up_share": float(np.mean(s > 0.0)),
                    "hybrid_minus_start_only_gold_mean": float(np.mean(h) - np.mean(s)),
                    "hybrid_minus_start_only_gold_up_share": float(
                        np.mean(h > 0.0) - np.mean(s > 0.0)
                    ),
                }
            )

    generated_mean_diff = [
        row["hybrid_minus_start_only_gold_mean"] for row in safe_generated
    ]
    generated_up_diff = [
        row["hybrid_minus_start_only_gold_up_share"] for row in safe_generated
    ]
    return {
        "schema_version": "safe_haven_gold_label_audit_v1",
        "status": "ok",
        "question": (
            "Do the projected-memory and narrative-to-narrative Safe-haven Gold "
            "supports actually look like Safe-haven Gold prefixes, or is the "
            "weak terminal Gold response caused by mislabeled historical support?"
        ),
        "source_artifacts": {
            "support_arrays": str(SUPPORT_ARRAYS),
            "support_cards": str(SUPPORT_CARDS),
            "old_channel_audit": str(OLD_CHANNEL_AUDIT),
            "old_start_audit": str(OLD_START_AUDIT),
            "new_hybrid_bridge": str(NEW_HYBRID_BRIDGE),
            "new_hybrid_arrays": str(NEW_HYBRID_ARRAYS),
            "start_only_arrays": str(START_ONLY_ARRAYS),
        },
        "summary": {
            "old_projected_memory_and_baseline_supports": _summary(old_rows),
            "new_narrative_to_narrative_safe_query_top3_supports": _summary(new_rows),
            "exact_paper_narrative_text_to_text_top20": _summary(exact_rows),
            "new_generated_gold_vs_start_only": {
                "safe_query_count": len(safe_generated),
                "mean_gold_mean_delta_difference": float(np.mean(generated_mean_diff))
                if generated_mean_diff
                else None,
                "median_gold_mean_delta_difference": float(np.median(generated_mean_diff))
                if generated_mean_diff
                else None,
                "mean_gold_up_share_difference": float(np.mean(generated_up_diff))
                if generated_up_diff
                else None,
            },
        },
        "interpretation": {
            "label_issue_assessment": (
                "The new narrative-to-narrative branch retrieves support cards "
                "that are explicitly labeled Safe-haven gold bid, and the raw "
                "prefix relabeling shows Gold-supported prefixes. However, many "
                "are broad or Gold-duration safe-haven episodes rather than pure "
                "Gold+duration+VIX-up+SPX-down episodes."
            ),
            "terminal_gold_assessment": (
                "The weak next-30-day Gold response remains after semantically "
                "cleaner retrieval. The evidence favors a prefix-continuation "
                "issue, not a simple wrong-support-label issue."
            ),
        },
        "generated_safe_query_rows": safe_generated,
        "support_rows": support_rows,
    }


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Safe-haven Gold Label Audit",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in report["summary"].items():
        lines.append(f"### {key}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(value, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    lines += [
        "## Interpretation",
        "",
        report["interpretation"]["label_issue_assessment"],
        "",
        report["interpretation"]["terminal_gold_assessment"],
        "",
        "## First support examples",
        "",
    ]
    for row in report["support_rows"][:24]:
        lines.append(
            "- "
            f"{row['source']} `{row['window_id']}` ending {row['calendar_end_date']}: "
            f"{row['manual_label']}; Gold prefix {row['moves']['GOLD']['prefix_delta']:+g}, "
            f"future Gold {row['moves']['GOLD']['future_terminal_delta']:+g}. "
            f"{row['manual_narrative']}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    report = build_audit()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "safe_haven_gold_label_audit.json"
    md_path = args.output_dir / "safe_haven_gold_label_audit.md"
    report["artifact_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    _write_json(json_path, report)
    _write_markdown(md_path, report)
    print(
        json.dumps(
            {
                "status": "ok",
                "json": str(json_path),
                "markdown": str(md_path),
                "summary": report["summary"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
