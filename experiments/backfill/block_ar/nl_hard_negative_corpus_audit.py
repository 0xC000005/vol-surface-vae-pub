#!/usr/bin/env python
"""Audit hard-negative coverage for NL episode-card training.

This script does not regenerate narratives. It checks whether the current
Codex/GPT-authored episode-card corpus and bridge reports satisfy the explicit
hard-negative setup described in the paper/presentation.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]

DEFAULT_CARDS_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_multiformat_982g_sharded/final/"
    "multiformat_episode_cards.jsonl"
)
DEFAULT_CODEX_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_multiformat_982g_sharded/final/"
    "final_codex_corpus_report.json"
)
DEFAULT_TEXT_MEMORY_REPORTS = [
    Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "episode_card_v3_full_codex_982g_text_memory_grounded_top3_90_66q/"
        "text_memory_bridge_report.json"
    ),
    Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "episode_card_v3_full_codex_982g_text_memory_bridge_large_66q/"
        "text_memory_bridge_report.json"
    ),
]
DEFAULT_TEXT_SPACE_REPORTS = [
    Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "episode_card_v3_full_codex_982g_text_space_contrastive_983b/"
        "text_space_contrastive_retriever_report.json"
    ),
    Path(
        "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
        "episode_card_v3_full_codex_982g_grounded_text_preference_reranker_984a/"
        "grounded_text_preference_reranker_report.json"
    ),
]
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "hard_negative_corpus_audit_985a"
)

POSITIVE_TRAINING_VIEWS = (
    "sparse_user_query",
    "weekly_risk_monitor",
    "risk_manager_memo",
    "institutional_risk_committee_note",
    "technical_factor_evidence",
    "mechanism_first",
    "full_professional",
    "factor_list_baseline",
)
LINK_KEY_PATTERNS = (
    "negative_window",
    "hard_negative_window",
    "negative_support",
    "hard_negative_support",
    "linked_negative",
)
LEAKAGE_PATTERNS = (
    re.compile(r"\bnext\s+\d+\s+(day|days|trading\s+day|trading\s+days)\b", re.I),
    re.compile(r"\brealized\s+(future|post[- ]window)\b", re.I),
    re.compile(r"\bterminal\s+(path|value|level|return|move|distribution)\b", re.I),
    re.compile(r"\bwill\s+(rally|fall|rise|drop|sell off|tighten|widen)\b", re.I),
)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    with _resolve(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with _resolve(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text or ""))


def _leakage_hits(text: str) -> list[str]:
    hits: list[str] = []
    for pattern in LEAKAGE_PATTERNS:
        if pattern.search(text or ""):
            hits.append(pattern.pattern)
    return hits


def _flatten_keys(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        keys: list[str] = []
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            keys.append(name)
            keys.extend(_flatten_keys(item, name))
        return keys
    if isinstance(value, list):
        keys = []
        for pos, item in enumerate(value[:5]):
            keys.extend(_flatten_keys(item, f"{prefix}[{pos}]"))
        return keys
    return []


def _linked_negative_keys(card: dict[str, Any]) -> list[str]:
    keys = _flatten_keys(card)
    lowered = [(key, key.lower()) for key in keys]
    return [
        key
        for key, low in lowered
        if any(pattern in low for pattern in LINK_KEY_PATTERNS)
    ]


def _hard_negative_texts(card: dict[str, Any]) -> list[str]:
    views = card.get("views") if isinstance(card.get("views"), dict) else {}
    value = views.get("hard_negative_views")
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []


def _role_counts(report: dict[str, Any]) -> Counter[str]:
    rows = report.get("example_rows")
    if not isinstance(rows, list):
        rows = report.get("training_rows")
    counts: Counter[str] = Counter()
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                counts[str(row.get("role", ""))] += 1
    return counts


def _used_views(report: dict[str, Any]) -> list[str]:
    rows = report.get("example_rows")
    if not isinstance(rows, list):
        rows = report.get("training_rows")
    views: set[str] = set()
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict) and row.get("view"):
                views.add(str(row["view"]))
    return sorted(views)


def _audit_training_report(path: Path) -> dict[str, Any]:
    resolved = _resolve(path)
    if not resolved.exists():
        return {"path": str(path), "exists": False}
    report = _read_json(path)
    roles = _role_counts(report)
    role_lower = {role.lower(): count for role, count in roles.items()}
    explicit_negative_rows = sum(
        count for role, count in role_lower.items() if "negative" in role
    )
    adapter = report.get("adapter_training") or report.get("training") or {}
    losses = []
    if isinstance(adapter, dict):
        losses = adapter.get("loss_trace") or []
    return {
        "path": str(path),
        "exists": True,
        "schema_version": report.get("schema_version"),
        "status": report.get("status"),
        "role_counts": dict(roles),
        "explicit_stored_hard_negative_rows": int(explicit_negative_rows),
        "uses_explicit_stored_hard_negatives": explicit_negative_rows > 0,
        "used_views": _used_views(report),
        "training_loss_names": sorted(
            {
                key
                for row in losses
                if isinstance(row, dict)
                for key in row.keys()
                if key not in {"step"}
            }
        ),
        "train_example_count": adapter.get("train_example_count")
        if isinstance(adapter, dict)
        else None,
    }


def audit(
    *,
    cards_jsonl: Path,
    codex_report: Path,
    text_memory_reports: list[Path],
    text_space_reports: list[Path],
) -> dict[str, Any]:
    cards = _read_jsonl(cards_jsonl)
    codex = _read_json(codex_report) if _resolve(codex_report).exists() else {}

    view_key_counts: Counter[str] = Counter()
    hard_negative_counts: Counter[int] = Counter()
    authoring_counts: Counter[str] = Counter()
    missing_positive_view_counts: Counter[str] = Counter()
    linked_key_cards = 0
    linked_key_examples: list[dict[str, Any]] = []
    short_negative_cards = 0
    leakage_cards: list[dict[str, Any]] = []
    word_counts: list[int] = []

    for card in cards:
        views = card.get("views") if isinstance(card.get("views"), dict) else {}
        view_key_counts.update(str(key) for key in views.keys())
        authoring_counts[str(card.get("narrative_authoring", ""))] += 1
        for name in POSITIVE_TRAINING_VIEWS:
            if not str(views.get(name, "")).strip():
                missing_positive_view_counts[name] += 1
        negatives = _hard_negative_texts(card)
        hard_negative_counts[len(negatives)] += 1
        counts = [_word_count(text) for text in negatives]
        word_counts.extend(counts)
        if negatives and min(counts) < 12:
            short_negative_cards += 1
        card_leakage = [
            {"text": text, "patterns": _leakage_hits(text)}
            for text in negatives
            if _leakage_hits(text)
        ]
        if card_leakage:
            leakage_cards.append(
                {
                    "window_id": card.get("window_id"),
                    "leakage": card_leakage[:3],
                }
            )
        linked_keys = _linked_negative_keys(card)
        if linked_keys:
            linked_key_cards += 1
            if len(linked_key_examples) < 5:
                linked_key_examples.append(
                    {"window_id": card.get("window_id"), "keys": linked_keys[:10]}
                )

    target_negative_views_per_window = len(POSITIVE_TRAINING_VIEWS)
    cards_with_enough_negatives = sum(
        count for neg_count, count in hard_negative_counts.items()
        if neg_count >= target_negative_views_per_window
    )
    text_memory_usage = [_audit_training_report(path) for path in text_memory_reports]
    text_space_usage = [_audit_training_report(path) for path in text_space_reports]
    explicit_usage = any(
        item.get("uses_explicit_stored_hard_negatives")
        for item in [*text_memory_usage, *text_space_usage]
    )

    failures = []
    if cards_with_enough_negatives != len(cards):
        failures.append(
            "hard_negative_text_view_coverage_below_positive_training_view_count"
        )
    if linked_key_cards != len(cards):
        failures.append("hard_negatives_not_linked_to_real_incompatible_windows")
    if not explicit_usage:
        failures.append("current_training_reports_do_not_use_explicit_stored_negatives")
    if leakage_cards:
        failures.append("hard_negative_text_leakage_detected")

    if word_counts:
        sorted_counts = sorted(word_counts)
        mid = len(sorted_counts) // 2
        median_words = (
            sorted_counts[mid]
            if len(sorted_counts) % 2
            else 0.5 * (sorted_counts[mid - 1] + sorted_counts[mid])
        )
    else:
        median_words = 0

    return {
        "schema_version": "hard_negative_corpus_audit_v1",
        "status": "pass" if not failures else "fail_needs_regeneration",
        "failures": failures,
        "source": {
            "cards_jsonl": str(cards_jsonl),
            "codex_report": str(codex_report),
            "codex_authoring": codex.get("authoring"),
            "codex_card_count": codex.get("card_count"),
            "codex_local_template_prose_used": codex.get("local_template_prose_used"),
        },
        "target_standard": {
            "positive_training_views": list(POSITIVE_TRAINING_VIEWS),
            "positive_training_view_count": len(POSITIVE_TRAINING_VIEWS),
            "hard_negative_text_views_per_window_min": target_negative_views_per_window,
            "linked_negative_historical_windows_required": True,
            "direct_codex_or_trusted_human_authoring_required": True,
            "future_leakage_allowed": False,
        },
        "corpus_audit": {
            "card_count": len(cards),
            "view_key_counts": dict(sorted(view_key_counts.items())),
            "narrative_authoring_counts": dict(sorted(authoring_counts.items())),
            "missing_positive_view_counts": dict(sorted(missing_positive_view_counts.items())),
            "hard_negative_text_count_distribution": {
                str(key): value for key, value in sorted(hard_negative_counts.items())
            },
            "cards_with_enough_hard_negative_texts": int(cards_with_enough_negatives),
            "linked_negative_key_card_count": int(linked_key_cards),
            "linked_negative_key_examples": linked_key_examples,
            "hard_negative_word_count": {
                "min": min(word_counts) if word_counts else 0,
                "median": median_words,
                "max": max(word_counts) if word_counts else 0,
                "cards_with_any_negative_under_12_words": int(short_negative_cards),
            },
            "hard_negative_leakage_card_count": len(leakage_cards),
            "hard_negative_leakage_examples": leakage_cards[:10],
        },
        "training_usage_audit": {
            "text_memory_reports": text_memory_usage,
            "text_space_reports": text_space_usage,
            "any_current_report_uses_explicit_stored_hard_negatives": explicit_usage,
        },
        "recommended_next_actions": [
            "regenerate_or_enrich_hard_negative_views_with_codex_gpt_only",
            "link_each_negative_text_to_real_incompatible_historical_window_memory",
            "validate_contradiction_leakage_authoring_and_split_safety",
            "retrain_projected_memory_and_text_space_retrievers_with_explicit_negative_rows_only_after_validation_passes",
        ],
    }


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    audit_block = report["corpus_audit"]
    training = report["training_usage_audit"]
    lines = [
        "# Hard-Negative Corpus Audit",
        "",
        f"Status: `{report['status']}`",
        "",
        "## Key Findings",
        "",
        f"- Cards audited: `{audit_block['card_count']}`.",
        "- Positive training views expected per window: "
        f"`{report['target_standard']['positive_training_view_count']}`.",
        "- Hard-negative text count distribution: "
        f"`{audit_block['hard_negative_text_count_distribution']}`.",
        "- Cards with enough hard-negative texts for the explicit standard: "
        f"`{audit_block['cards_with_enough_hard_negative_texts']}`.",
        "- Cards with linked negative-window metadata: "
        f"`{audit_block['linked_negative_key_card_count']}`.",
        "- Current reports use explicit stored hard negatives: "
        f"`{training['any_current_report_uses_explicit_stored_hard_negatives']}`.",
        "",
        "## Failures",
        "",
    ]
    if report["failures"]:
        lines.extend(f"- `{item}`" for item in report["failures"])
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The 982g positives are direct Codex/GPT-authored and remain usable as "
            "positive narrative views. The hard-negative layer is not yet the "
            "explicit paired hard-negative corpus described in the paper: the "
            "stored negatives are short text labels, they are not linked to real "
            "incompatible support windows, and current bridge reports do not "
            "consume them as explicit negative rows.",
            "",
            "## Required Next Step",
            "",
            "Generate a new hard-negative bank with matched narrative views and "
            "linked incompatible historical windows before retraining or making "
            "paper-facing hard-negative training claims.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cards-jsonl", type=Path, default=DEFAULT_CARDS_JSONL)
    parser.add_argument("--codex-report", type=Path, default=DEFAULT_CODEX_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report = audit(
        cards_jsonl=args.cards_jsonl,
        codex_report=args.codex_report,
        text_memory_reports=DEFAULT_TEXT_MEMORY_REPORTS,
        text_space_reports=DEFAULT_TEXT_SPACE_REPORTS,
    )
    out_dir = _resolve(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "hard_negative_corpus_audit.json"
    md_path = out_dir / "hard_negative_corpus_audit.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, report)
    print(json.dumps({"status": report["status"], "json": str(json_path), "md": str(md_path)}, indent=2))


if __name__ == "__main__":
    main()
