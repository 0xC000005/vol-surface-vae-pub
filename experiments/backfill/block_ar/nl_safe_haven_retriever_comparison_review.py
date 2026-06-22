#!/usr/bin/env python
"""Build a human-review packet for Safe-haven Gold retriever comparison.

This utility compares two support-retrieval families for the same Safe-haven
Gold casebook narrative:

1. OpenAI text-embedding retrieval with generic grounding and the Stage-2
   frozen-SNI replay preference reranker.
2. Text-embedding -> SNI-memory projection trained on the rich Codex-authored
   corpus, with generic grounding and top3/90 support assembly.

It does not generate narratives, change the current paper/demo default, or add
any Safe-haven-specific response target.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.nl_episode_grounded_text_support_preference_reranker import (  # noqa: E402
    FEATURE_NAMES,
    TextSupportPreferenceModel,
    _apply_learned_top3_90_selection,
    _feature_stats,
    _rank_grounded_text_pool,
    _score_candidates,
    build_delta_scale,
)
from experiments.backfill.block_ar.nl_episode_narrative_bridge_report import (  # noqa: E402
    _load_json,
    _read_jsonl,
    _split_indices_from_support_report,
    _window_index,
)
from experiments.backfill.block_ar.nl_episode_narrative_embedding_bridge_report import (  # noqa: E402
    _candidate_view_rows,
)
from experiments.backfill.block_ar.nl_episode_narrative_retrieval import (  # noqa: E402
    DEFAULT_VIEW_NAMES,
    assert_cards_allowed_for_retrieval,
)


DEFAULT_CARDS_JSONL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_multiformat_982g_sharded/final/"
    "multiformat_episode_cards.jsonl"
)
DEFAULT_CASEBOOK_JSON = Path(
    "paper/narrative_grounded_scenarios/figures/"
    "narrative_qualitative_casebook_summary.json"
)
DEFAULT_QUERY_EMBEDDING = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_text_memory_grounded_top3_90_66q/"
    "safe_haven_gold_case_query_embedding/embedding_cache/"
    "openai_text_embedding_3_large_a2e53e091adcd1d1549d0bf4.npz"
)
DEFAULT_EMBEDDING_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_embedding_bridge_hybrid_66q/"
    "embedding_bridge_arrays.npz"
)
DEFAULT_SUPPORT_REPORT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_report.json"
)
DEFAULT_SUPPORT_ARRAYS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_support_bank_train_all_939a/support_bank_arrays.npz"
)
DEFAULT_RERANKER_MODEL = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_grounded_text_preference_reranker_984a/"
    "grounded_text_preference_model.json"
)
DEFAULT_PROJECTED_SUPPORTS = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_text_memory_grounded_top3_90_66q/"
    "safe_haven_gold_case_projected_memory_top3_90_supports.json"
)
DEFAULT_TEXT_PREF_SCENARIO = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_scenario_grounded_text_preference_984a_s16/"
    "scenario_level_eval_report.json"
)
DEFAULT_PROJECTED_SCENARIO = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_scenario_text_memory_grounded_top3_90_66q_s16/"
    "scenario_level_eval_report.json"
)
DEFAULT_START_ONLY_SCENARIO = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_scenario_start_only_66q_s16/"
    "scenario_level_eval_report.json"
)
DEFAULT_TEXT_PREF_LIFT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_lift_grounded_text_preference_984a_vs_start/"
    "conditionality_lift_report.json"
)
DEFAULT_PROJECTED_LIFT = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "episode_card_v3_full_codex_982g_lift_text_memory_grounded_top3_90_vs_start/"
    "conditionality_lift_report.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "safe_haven_gold_retriever_comparison_review_984b"
)
REVIEW_VIEW_ORDER = (
    "sparse_user_query",
    "weekly_risk_monitor",
    "institutional_risk_committee_note",
    "mechanism_first",
    "risk_manager_memo",
    "full_professional",
    "technical_factor_evidence",
    "factor_list_baseline",
    "hard_negative_views",
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {name: payload[name].copy() for name in payload.files}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _load_model(path: Path) -> TextSupportPreferenceModel:
    payload = _load_json(path)
    return TextSupportPreferenceModel(
        coefficients=np.asarray(payload["coefficients"], dtype=np.float64),
        intercept=float(payload["intercept"]),
        feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
        feature_std=np.asarray(payload["feature_std"], dtype=np.float64),
        feature_names=list(payload.get("feature_names", FEATURE_NAMES)),
        ridge_alpha=float(payload["ridge_alpha"]),
    )


def _casebook_case(path: Path, case_name: str) -> dict[str, Any]:
    payload = _load_json(path)
    for row in payload.get("case_summaries", []):
        if str(row.get("case_name", "")) == str(case_name):
            return row
    raise ValueError(f"{path}: no case_summaries entry named {case_name!r}")


def _metadata_by_index(support_report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows = support_report.get("window_metadata", [])
    return {
        int(row["window_index"]): row
        for row in rows
        if isinstance(row, dict) and "window_index" in row
    }


def _card_summary(
    *,
    card: dict[str, Any],
    support: dict[str, Any],
    metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    views = card.get("views", {})
    if not isinstance(views, dict):
        views = {}
    fields = card.get("caption_fields", {})
    if not isinstance(fields, dict):
        fields = {}
    selected_views: dict[str, Any] = {}
    for view_name in REVIEW_VIEW_ORDER:
        value = views.get(view_name)
        if value is None and view_name == "hard_negative_views":
            value = fields.get("hard_negative_views") or fields.get(
                "contrastive_hard_negatives"
            )
        if value is not None:
            selected_views[view_name] = value
    direction_check = support.get("direction_check")
    if direction_check is None:
        direction_check = support.get("score_components", {}).get("direction_check", {})
    return {
        "window_id": str(card.get("window_id", support.get("window_id", ""))),
        "window_index": int(support.get("window_index", _window_index(card))),
        "calendar_start_date": None if metadata is None else metadata.get("calendar_start_date"),
        "calendar_end_date": None if metadata is None else metadata.get("calendar_end_date"),
        "forecast_start_date": None if metadata is None else metadata.get("forecast_start_date"),
        "forecast_end_date": None if metadata is None else metadata.get("forecast_end_date"),
        "rank": int(support.get("rank", 0) or 0),
        "weight": float(support.get("weight", support.get("posterior_weight", 0.0)) or 0.0),
        "score": float(
            support.get(
                "learned_support_score",
                support.get("score", support.get("retrieval_score", support.get("memory_score", 0.0))),
            )
            or 0.0
        ),
        "embedding_score": support.get("cosine")
        or support.get("retrieval_score")
        or support.get("score_components", {}).get("embedding_score"),
        "memory_score": support.get("memory_score"),
        "title": str(card.get("scenario_title", support.get("scenario_title", ""))),
        "archetype": str(card.get("archetype", "")),
        "confidence": str(card.get("archetype_confidence", "")),
        "direction_check": direction_check,
        "fast_read": str(
            views.get("risk_manager_memo")
            or views.get("full_professional")
            or fields.get("risk_manager_implication")
            or ""
        ),
        "views": selected_views,
        "ambiguity_flags": fields.get("ambiguity_flags", []),
        "evidence_used": fields.get("evidence_used", []),
    }


def _metric_block(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    summary = payload.get("summary", {})
    block = summary.get("narrative_generator_topk", {})
    return {
        "report": str(path),
        "window_count": block.get("window_count"),
        "crps_improvement_vs_persistence": block.get(
            "ensemble_crps_z_improvement_vs_persistence"
        ),
        "energy_improvement_vs_persistence": block.get(
            "energy_score_z_improvement_vs_persistence"
        ),
        "coverage80": block.get("coverage_80_mean"),
        "crps_mean": block.get("ensemble_crps_z_mean"),
        "energy_mean": block.get("energy_score_z_mean"),
    }


def _lift_block(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    aggregate = payload.get("aggregate", {})
    quality = payload.get("quality_guardrail", {})
    return {
        "report": str(path),
        "verdict": payload.get("verdict"),
        "window_count": payload.get("window_count"),
        "mean_support_jaccard_distance": aggregate.get("mean_support_jaccard_distance"),
        "mean_terminal_factor_ks": aggregate.get("mean_terminal_factor_ks"),
        "mean_path_energy_distance_z": aggregate.get("mean_path_energy_distance_z"),
        "mean_abs_terminal_mean_shift_z": aggregate.get("mean_abs_terminal_mean_shift_z"),
        "crps_delta_vs_start_only": quality.get("crps_delta_vs_start_only"),
        "energy_delta_vs_start_only": quality.get("energy_delta_vs_start_only"),
    }


def _build_text_preference_supports(args: argparse.Namespace) -> dict[str, Any]:
    cards = _read_jsonl(Path(args.cards_jsonl))
    assert_cards_allowed_for_retrieval(cards, path=Path(args.cards_jsonl))
    support_report = _load_json(Path(args.support_report))
    support_arrays = _load_npz(Path(args.support_arrays))
    embedding_arrays = _load_npz(Path(args.embedding_arrays))
    projected_supports = _load_json(Path(args.projected_supports))
    query_embedding = _load_npz(Path(args.query_embedding))["embeddings"][0].astype(np.float32)
    query_embedding = query_embedding / max(float(np.linalg.norm(query_embedding)), 1e-8)
    cards_by_index = {_window_index(card): card for card in cards}
    train_indices, _test_indices = _split_indices_from_support_report(support_report)
    train_cards = [cards_by_index[idx] for idx in train_indices if idx in cards_by_index]
    candidate_rows, candidate_texts = _candidate_view_rows(
        train_cards, tuple(DEFAULT_VIEW_NAMES)
    )
    candidate_count = int(np.asarray(embedding_arrays["candidate_text_count"])[0])
    if candidate_count != len(candidate_rows) or candidate_count != len(candidate_texts):
        raise ValueError("candidate embedding arrays do not match rebuilt rows")
    candidate_vectors = np.asarray(embedding_arrays["text_embeddings"], dtype=np.float32)[
        :candidate_count
    ]
    claims = projected_supports["query_claims"]
    candidate_pool = _rank_grounded_text_pool(
        query_vector=query_embedding,
        query_index=int(args.accepted_start_index),
        query_claims=claims,
        candidate_vectors=candidate_vectors,
        candidate_rows=candidate_rows,
        cards_by_index=cards_by_index,
        support_pool_size=int(args.support_pool_size),
        temporal_gap=int(args.temporal_gap),
        max_grounding_mismatches=int(args.max_grounding_mismatches),
        initial_scan=int(args.initial_scan),
    )
    future_delta = np.asarray(support_arrays["future_delta"], dtype=np.float32)
    history_raw = np.asarray(support_arrays["history_raw"], dtype=np.float32)
    train_np = np.asarray(support_arrays["train_indices"], dtype=np.int64)
    delta_scale = build_delta_scale(future_delta, train_np)
    stats = _feature_stats(history_raw, train_np)
    model = _load_model(Path(args.reranker_model))
    scored = _score_candidates(
        query_window_index=int(args.accepted_start_index),
        candidates=candidate_pool,
        stats=stats,
        future_delta=future_delta,
        delta_scale=delta_scale,
        model=model,
    )
    selected = _apply_learned_top3_90_selection(
        scored, weight_temperature=float(args.weight_temperature)
    )
    return {
        "query_claims": claims,
        "candidate_pool": candidate_pool,
        "preference_reranked_candidate_pool": scored,
        "selected_top3_90": selected,
    }


def _method_review(
    *,
    method_name: str,
    selected_supports: list[dict[str, Any]],
    cards_by_index: dict[int, dict[str, Any]],
    metadata_by_index: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for support in selected_supports:
        idx = int(support["window_index"])
        card = cards_by_index[idx]
        rows.append(
            _card_summary(
                card=card,
                support=support,
                metadata=metadata_by_index.get(idx),
            )
        )
    return {
        "method": method_name,
        "support_count": len(rows),
        "direction_pass_count": sum(
            1
            for row in rows
            if str(row.get("direction_check", {}).get("status", "")) == "pass"
        ),
        "high_confidence_count": sum(
            1 for row in rows if str(row.get("confidence", "")).lower() == "high"
        ),
        "financial_accident_count": sum(
            1
            for row in rows
            if str(row.get("archetype", "")).lower() == "financial_accident"
        ),
        "support_titles": [row["title"] for row in rows],
        "support_window_ids": [row["window_id"] for row in rows],
        "supports": rows,
    }


def _md_table(rows: list[list[Any]], headers: list[str]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(out)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _markdown(report: dict[str, Any]) -> str:
    case = report["case"]
    comparison = report["comparison_summary"]
    metric_rows = []
    for method, metrics in report["metric_comparison"].items():
        scenario = metrics["scenario"]
        lift = metrics["lift"]
        metric_rows.append(
            [
                method,
                _fmt(scenario["crps_improvement_vs_persistence"]),
                _fmt(scenario["energy_improvement_vs_persistence"]),
                _fmt(scenario["coverage80"]),
                _fmt(lift["mean_terminal_factor_ks"]),
                _fmt(lift["mean_path_energy_distance_z"]),
                _fmt(lift["crps_delta_vs_start_only"]),
                _fmt(lift["energy_delta_vs_start_only"]),
            ]
        )
    method_rows = []
    for method in report["methods"]:
        method_rows.append(
            [
                method["method"],
                method["direction_pass_count"],
                method["high_confidence_count"],
                method["financial_accident_count"],
                "<br>".join(method["support_window_ids"]),
                "<br>".join(method["support_titles"]),
            ]
        )
    lines = [
        "# Safe-haven Gold Retriever Comparison Review",
        "",
        "## Purpose",
        "",
        "This packet is for human review of support quality. It does not add a "
        "Safe-haven-specific response target and does not change paper/demo defaults.",
        "",
        "Question: for the same Safe-haven Gold casebook narrative, which retrieval "
        "family finds more plausible historical support?",
        "",
        "## Case",
        "",
        f"- Case name: `{case['case_name']}`",
        f"- Label: {case['label']}",
        f"- Accepted start index used by the text-preference reranker features: `{report['accepted_start_index']}`",
        f"- Grounded implications: {case['grounded_implications']}",
        "",
        "Narrative:",
        "",
        "> " + str(case["narrative_text"]).replace("\n", "\n> "),
        "",
        "## Method Definitions",
        "",
        "1. `text_embedding_grounded_preference_984a`: OpenAI text embedding retrieves "
        "a grounded text candidate pool; the frozen-SNI historical replay preference "
        "model reranks that pool; unchanged top3/90 is applied.",
        "",
        "2. `projected_memory_rich_narrative`: OpenAI text embedding is projected into "
        "SNI final-hidden memory space using the rich Codex-authored training corpus; "
        "grounding and unchanged top3/90 are applied.",
        "",
        "## One-Page Decision Summary",
        "",
        _md_table(
            [
                ["Top-3 overlap", comparison["top3_overlap"]],
                [
                    "Preliminary qualitative read",
                    comparison["preliminary_qualitative_read"],
                ],
                [
                    "Conditionality read",
                    comparison["conditionality_read"],
                ],
            ],
            ["Item", "Value"],
        ),
        "",
        "## Support Summary",
        "",
        _md_table(
            method_rows,
            [
                "Method",
                "Direction pass",
                "High confidence",
                "Financial accident",
                "Support IDs",
                "Titles",
            ],
        ),
        "",
        "## Broad 66-Window Metric Context",
        "",
        _md_table(
            metric_rows,
            [
                "Method",
                "CRPS imp vs persistence",
                "Energy imp vs persistence",
                "Coverage80",
                "Terminal KS vs start",
                "Path energy vs start",
                "CRPS delta vs start",
                "Energy delta vs start",
            ],
        ),
        "",
        "Interpretation: positive CRPS/Energy improvement versus persistence is good. "
        "For deltas versus start-only, lower is better; positive means worse than "
        "start-only on absolute historical fidelity. Terminal KS and path energy "
        "measure narrative-vs-start-only distributional lift.",
        "",
        "## Reviewer Checklist",
        "",
        "- Does the support describe a current/recent prefix that plausibly matches "
        "Safe-haven Gold?",
        "- Does it separate a clean safe-haven risk-off prefix from a mixed gold-duration "
        "or gold-in-risk-on prefix?",
        "- Are ambiguity flags explicit rather than hidden?",
        "- Do not require the support future to continue Gold upward; the narrative "
        "describes the conditioning prefix, not the terminal path.",
        "- Prefer the method whose supports are coherent without relying on a "
        "Safe-haven-specific response gate.",
        "",
    ]
    for method in report["methods"]:
        lines += [
            f"## Support Narratives: {method['method']}",
            "",
        ]
        for support in method["supports"]:
            lines += [
                f"### Rank {support['rank']}: {support['window_id']} - {support['title']}",
                "",
                _md_table(
                    [
                        ["Dates", f"{support['calendar_start_date']} to {support['calendar_end_date']}"],
                        ["Weight", _fmt(support["weight"])],
                        ["Score", _fmt(support["score"])],
                        ["Archetype", support["archetype"]],
                        ["Confidence", support["confidence"]],
                        [
                            "Direction check",
                            str(support.get("direction_check", {}).get("status", "")),
                        ],
                    ],
                    ["Field", "Value"],
                ),
                "",
                "**Fast read**",
                "",
                str(support["fast_read"]),
                "",
            ]
            if support.get("ambiguity_flags"):
                lines += [
                    "**Ambiguity flags**",
                    "",
                    *[f"- {item}" for item in support["ambiguity_flags"]],
                    "",
                ]
            if support.get("evidence_used"):
                lines += [
                    "**Evidence used**",
                    "",
                    *[f"- {item}" for item in support["evidence_used"]],
                    "",
                ]
            lines += ["**Narrative views**", ""]
            for view_name in REVIEW_VIEW_ORDER:
                if view_name not in support["views"]:
                    continue
                value = support["views"][view_name]
                lines += [f"#### {view_name}", ""]
                if isinstance(value, list):
                    lines += [f"- {item}" for item in value]
                else:
                    lines += [str(value)]
                lines += [""]
    return "\n".join(lines).rstrip() + "\n"


def build_review(args: argparse.Namespace) -> dict[str, Any]:
    cards = _read_jsonl(Path(args.cards_jsonl))
    assert_cards_allowed_for_retrieval(cards, path=Path(args.cards_jsonl))
    support_report = _load_json(Path(args.support_report))
    cards_by_index = {_window_index(card): card for card in cards}
    metadata_by_index = _metadata_by_index(support_report)
    case = _casebook_case(Path(args.casebook_json), str(args.case_name))
    projected_payload = _load_json(Path(args.projected_supports))
    text_pref_payload = _build_text_preference_supports(args)
    text_pref_method = _method_review(
        method_name="text_embedding_grounded_preference_984a",
        selected_supports=text_pref_payload["selected_top3_90"],
        cards_by_index=cards_by_index,
        metadata_by_index=metadata_by_index,
    )
    projected_method = _method_review(
        method_name="projected_memory_rich_narrative",
        selected_supports=projected_payload["selected_top3_90"],
        cards_by_index=cards_by_index,
        metadata_by_index=metadata_by_index,
    )
    text_ids = set(text_pref_method["support_window_ids"])
    projected_ids = set(projected_method["support_window_ids"])
    high_conf_text = int(text_pref_method["high_confidence_count"])
    high_conf_projected = int(projected_method["high_confidence_count"])
    if high_conf_text > high_conf_projected:
        qualitative = "text-preference has more high-confidence selected supports"
    elif high_conf_projected > high_conf_text:
        qualitative = "projected-memory has more high-confidence selected supports"
    else:
        qualitative = "both methods have the same high-confidence selected-support count; inspect narratives below"
    report = {
        "schema_version": "safe_haven_gold_retriever_comparison_review_v1",
        "status": "ok",
        "scope_note": (
            "Human-review packet only. No narrative generation, no paper/demo "
            "default change, and no Safe-haven-specific response target."
        ),
        "accepted_start_index": int(args.accepted_start_index),
        "inputs": {
            "cards_jsonl": str(args.cards_jsonl),
            "casebook_json": str(args.casebook_json),
            "query_embedding": str(args.query_embedding),
            "reranker_model": str(args.reranker_model),
            "projected_supports": str(args.projected_supports),
        },
        "case": {
            "case_name": str(case.get("case_name", "")),
            "label": str(case.get("label", "")),
            "grounded_implications": str(case.get("grounded_implications", "")),
            "narrative_text": str(case.get("narrative_text", "")),
        },
        "comparison_summary": {
            "top3_overlap": len(text_ids & projected_ids),
            "text_preference_only": sorted(text_ids - projected_ids),
            "projected_memory_only": sorted(projected_ids - text_ids),
            "preliminary_qualitative_read": qualitative,
            "conditionality_read": (
                "Projected-memory is marginally stronger on the aggregate "
                "terminal-KS/path-energy lift metrics, while 984a is stronger "
                "on broad 66-window CRPS/Energy and has a smaller absolute "
                "fidelity regression versus start-only. Both remain worse than "
                "start-only on absolute CRPS/Energy."
            ),
        },
        "metric_comparison": {
            "text_embedding_grounded_preference_984a": {
                "scenario": _metric_block(Path(args.text_pref_scenario_report)),
                "lift": _lift_block(Path(args.text_pref_lift_report)),
            },
            "projected_memory_rich_narrative": {
                "scenario": _metric_block(Path(args.projected_scenario_report)),
                "lift": _lift_block(Path(args.projected_lift_report)),
            },
        },
        "text_preference_exact_case_retrieval": text_pref_payload,
        "projected_memory_exact_case_retrieval": projected_payload,
        "methods": [text_pref_method, projected_method],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cards-jsonl", type=Path, default=DEFAULT_CARDS_JSONL)
    parser.add_argument("--casebook-json", type=Path, default=DEFAULT_CASEBOOK_JSON)
    parser.add_argument("--case-name", default="safe_haven_gold")
    parser.add_argument("--query-embedding", type=Path, default=DEFAULT_QUERY_EMBEDDING)
    parser.add_argument("--embedding-arrays", type=Path, default=DEFAULT_EMBEDDING_ARRAYS)
    parser.add_argument("--support-report", type=Path, default=DEFAULT_SUPPORT_REPORT)
    parser.add_argument("--support-arrays", type=Path, default=DEFAULT_SUPPORT_ARRAYS)
    parser.add_argument("--reranker-model", type=Path, default=DEFAULT_RERANKER_MODEL)
    parser.add_argument("--projected-supports", type=Path, default=DEFAULT_PROJECTED_SUPPORTS)
    parser.add_argument("--text-pref-scenario-report", type=Path, default=DEFAULT_TEXT_PREF_SCENARIO)
    parser.add_argument("--projected-scenario-report", type=Path, default=DEFAULT_PROJECTED_SCENARIO)
    parser.add_argument("--start-only-scenario-report", type=Path, default=DEFAULT_START_ONLY_SCENARIO)
    parser.add_argument("--text-pref-lift-report", type=Path, default=DEFAULT_TEXT_PREF_LIFT)
    parser.add_argument("--projected-lift-report", type=Path, default=DEFAULT_PROJECTED_LIFT)
    parser.add_argument("--accepted-start-index", type=int, default=22)
    parser.add_argument("--support-pool-size", type=int, default=8)
    parser.add_argument("--temporal-gap", type=int, default=30)
    parser.add_argument("--max-grounding-mismatches", type=int, default=0)
    parser.add_argument("--initial-scan", type=int, default=512)
    parser.add_argument("--weight-temperature", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    report = build_review(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "safe_haven_gold_retriever_comparison_review.json"
    md_path = args.output_dir / "safe_haven_gold_retriever_comparison_review.md"
    report["artifact_paths"] = {"json": str(json_path), "markdown": str(md_path)}
    _write_json(json_path, report)
    md_path.write_text(_markdown(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "ok",
                "json": str(json_path),
                "markdown": str(md_path),
                "text_preference_supports": report["methods"][0]["support_window_ids"],
                "projected_memory_supports": report["methods"][1]["support_window_ids"],
                "top3_overlap": report["comparison_summary"]["top3_overlap"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
