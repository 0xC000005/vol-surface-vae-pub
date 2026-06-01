#!/usr/bin/env python
"""Fixed-start support-policy bake-off for narrative conditionality.

This diagnostic makes no OpenAI calls and does not run the frozen generator.
It tests whether the support pool itself is dominated by the supplied starting
level or whether the narrative can select distinct regime support under the
same fixed start.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.backfill.block_ar.evaluate_662a_state_aware_normalized_innovation_flow import (  # noqa: E402
    build_val_block,
)
from experiments.backfill.block_ar.nl_narrative_grounded_scenario_pipeline import (  # noqa: E402
    DEFAULT_CHECKPOINT,
)
from experiments.backfill.block_ar.nl_prefix_latent_analogue_mixture_prior import (  # noqa: E402
    build_mixture_memory_prior,
    candidate_support_table,
)
from experiments.backfill.block_ar.nl_prefix_latent_oracle_autoencoder import (  # noqa: E402
    selected_bridge_window_indices,
    split_indices_from_bridge_report,
)


DEFAULT_OUTPUT_DIR = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "fixed_start_support_policy_bakeoff_930a"
)
DEFAULT_BRIDGE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_full_906b_all_windows/bridge_eval_report.json"
)
DEFAULT_BRIDGE_ARRAYS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_full_906b_all_windows/bridge_eval_arrays.npz"
)
DEFAULT_REPRESENTATIVE_BRIDGE_REPORT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_openai_schema_v2_representative_220/bridge_eval_report.json"
)
DEFAULT_REPRESENTATIVE_BRIDGE_ARRAYS = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "manifest_bridge_eval_openai_schema_v2_representative_220/bridge_eval_arrays.npz"
)
DEFAULT_CONDITION_ROOT = (
    "experiments/backfill/block_ar/nl_scenario_demo_outputs/"
    "prefix_latent_condition_only_full906b_914a"
)

CASE_NAMES = (
    "fragile_risk_on_start18",
    "defensive_risk_off_start18",
    "commodity_inflation_start18",
    "dollar_liquidity_start18",
    "rates_selloff_start18",
    "safe_haven_gold_start18",
)


@dataclass(frozen=True)
class SupportPolicy:
    name: str
    prior_mode: str
    start_distance_penalty: float
    implication_alignment_weight: float
    diverse_min_index_gap: int
    description: str


POLICIES = (
    SupportPolicy(
        name="current_start_checked_gap30",
        prior_mode="diverse_topk_narrative_start_checked",
        start_distance_penalty=0.02,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        description=(
            "Current paper/demo selector: narrative-memory score, start-distance "
            "penalty, hard direction gate, and 30-window temporal gap."
        ),
    ),
    SupportPolicy(
        name="narrative_first_hard_direction_gap30",
        prior_mode="diverse_topk_narrative_start_checked",
        start_distance_penalty=0.0,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        description=(
            "Narrative-first selector: removes start-distance ranking pressure "
            "but keeps the hard direction gate and temporal diversity."
        ),
    ),
    SupportPolicy(
        name="narrative_first_soft_direction_gap30",
        prior_mode="diverse_topk_combined",
        start_distance_penalty=0.0,
        implication_alignment_weight=0.25,
        diverse_min_index_gap=30,
        description=(
            "Narrative-first selector with soft direction scoring: direction "
            "alignment affects ranking but does not hard-filter supports."
        ),
    ),
    SupportPolicy(
        name="narrative_only_diverse_gap30",
        prior_mode="diverse_topk_narrative_start",
        start_distance_penalty=0.0,
        implication_alignment_weight=0.0,
        diverse_min_index_gap=30,
        description=(
            "Narrative-only ablation: raw text-memory support with temporal "
            "diversity and no start or direction contribution."
        ),
    ),
    SupportPolicy(
        name="start_only_topk",
        prior_mode="soft_topk_start_only",
        start_distance_penalty=0.02,
        implication_alignment_weight=0.0,
        diverse_min_index_gap=0,
        description=(
            "Start-only null: selects nearby starting levels without narrative "
            "or direction semantics."
        ),
    ),
)


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected JSON object")
    return payload


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_text(path: str | Path, text: str) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _spec_names(specs: list[Any]) -> list[str]:
    names: list[str] = []
    for item in specs:
        if hasattr(item, "name"):
            names.append(str(item.name))
        elif isinstance(item, dict):
            names.append(str(item.get("name", "")))
    if not names:
        raise ValueError("empty state spec names")
    return names


def _bridge_metadata(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    metadata = report.get("window_metadata", [])
    if not isinstance(metadata, list):
        return rows
    for local_idx, item in enumerate(metadata):
        if not isinstance(item, dict):
            continue
        rows[int(local_idx)] = dict(item)
    return rows


def _window_year(row: dict[str, Any] | None) -> str:
    if not row:
        return "unknown"
    date = str(row.get("calendar_end_date") or row.get("forecast_start_date") or "")
    return date[:4] if len(date) >= 4 else "unknown"


def _shannon_entropy(counts: Counter[str]) -> float:
    total = float(sum(counts.values()))
    if total <= 0.0:
        return 0.0
    entropy = 0.0
    for value in counts.values():
        p = float(value) / total
        if p > 0.0:
            entropy -= p * math.log(p)
    return float(entropy)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float32).reshape(-1)
    b = np.asarray(right, dtype=np.float32).reshape(-1)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), 1e-8)
    return float(np.dot(a, b) / denom)


def _load_condition(
    case: str, condition_root: Path
) -> tuple[dict[str, Any], np.ndarray]:
    report_path = condition_root / case / "condition_only_report.json"
    arrays_path = condition_root / case / "condition_only_report_arrays.npz"
    report = _load_json(report_path)
    arrays = np.load(arrays_path)
    if "text_memory" not in arrays:
        raise ValueError(f"{arrays_path}: missing text_memory")
    memory = np.asarray(arrays["text_memory"], dtype=np.float32)
    if memory.shape[0] < 1:
        raise ValueError(f"{arrays_path}: empty text_memory")
    return report, memory[0]


def _build_history_level(
    *,
    checkpoint: str | Path,
    bridge_report: dict[str, Any],
) -> tuple[np.ndarray, list[str]]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args = argparse.Namespace(
        clean_nonpositive_log_levels=True,
        positive_level_policy="reference_based",
        iv_transform="log_level",
        iv_lower_bound=1e-4,
        iv_upper_bound=1.0,
        scale_half_life=0.0,
        scale_floor=1e-4,
        center_mode="zero",
        drift_feature_mode="none",
        state_scope="joint38",
        iv_count=25,
        test_start=4511,
        val_size=441,
        max_windows=441,
        eval_split="val",
    )
    history_level, _history_norm, _center, _scale, _drift, _raw, specs, _block = (
        build_val_block(args, payload)
    )
    selected = selected_bridge_window_indices(bridge_report)
    return history_level[selected].astype(np.float32), _spec_names(specs)


def _selected_support_rows(
    *,
    prior: dict[str, Any],
    metadata: dict[int, dict[str, Any]],
    candidate_by_index: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    indices = [int(idx) for idx in prior.get("window_indices", [])]
    weights = [float(w) for w in prior.get("weights", [])]
    for rank, idx in enumerate(indices, start=1):
        info = metadata.get(idx, {})
        candidate = candidate_by_index.get(idx, {})
        rows.append(
            {
                "rank": rank,
                "window_index": idx,
                "window_id": str(info.get("window_id", f"window_{idx:04d}")),
                "history_end": str(info.get("calendar_end_date", "")),
                "year": _window_year(info),
                "weight": float(weights[rank - 1]) if rank - 1 < len(weights) else 0.0,
                "memory_cosine": float(candidate.get("memory_support_cosine", 0.0)),
                "start_distance_z": float(candidate.get("start_distance_z", 0.0)),
                "alignment_score": float(
                    candidate.get("recent_prefix_alignment_score", 0.0)
                ),
                "direction_mismatches": int(
                    candidate.get("recent_prefix_mismatches", 0) or 0
                ),
                "direction_checked": int(
                    candidate.get("recent_prefix_checked", 0) or 0
                ),
            }
        )
    return rows


def _case_metrics(
    *,
    support_rows: list[dict[str, Any]],
    start_only_indices: set[int],
    memory_targets: np.ndarray,
    prior: dict[str, Any],
) -> dict[str, Any]:
    indices = [int(row["window_index"]) for row in support_rows]
    weights = np.asarray(
        [float(row["weight"]) for row in support_rows], dtype=np.float64
    )
    if weights.size and float(weights.sum()) > 0.0:
        weights = weights / float(weights.sum())
    elif weights.size:
        weights = np.ones(weights.shape[0], dtype=np.float64) / float(weights.shape[0])
    weighted_start = 0.0
    weighted_memory = 0.0
    weighted_direction_mismatches = 0.0
    weighted_direction_checked = 0.0
    for row, weight in zip(support_rows, weights, strict=False):
        weighted_start += float(weight) * float(row["start_distance_z"])
        weighted_memory += float(weight) * float(row["memory_cosine"])
        weighted_direction_mismatches += float(weight) * float(
            row["direction_mismatches"]
        )
        weighted_direction_checked += float(weight) * float(row["direction_checked"])
    years = Counter(str(row["year"]) for row in support_rows)
    min_gap = None
    if len(indices) >= 2:
        gaps = [
            abs(int(a) - int(b))
            for pos, a in enumerate(indices)
            for b in indices[pos + 1 :]
        ]
        min_gap = int(min(gaps))
    pairwise_memory_cosine = None
    if len(indices) >= 2:
        target = np.asarray(memory_targets, dtype=np.float32)
        cosines = [
            _cosine(target[int(a)], target[int(b)])
            for pos, a in enumerate(indices)
            for b in indices[pos + 1 :]
        ]
        pairwise_memory_cosine = float(np.mean(cosines))
    support = set(indices)
    start_overlap = len(support & start_only_indices) / max(
        len(support | start_only_indices), 1
    )
    direction_check = prior.get("direction_check", {})
    if not isinstance(direction_check, dict):
        direction_check = {}
    return {
        "support_count": int(len(indices)),
        "unique_year_count": int(len(years)),
        "year_counts": dict(sorted(years.items())),
        "year_entropy": _shannon_entropy(years),
        "min_index_gap": min_gap,
        "mean_pairwise_memory_cosine": pairwise_memory_cosine,
        "weighted_start_distance_z": float(weighted_start),
        "weighted_memory_cosine": float(weighted_memory),
        "weighted_direction_mismatches": float(weighted_direction_mismatches),
        "weighted_direction_checked": float(weighted_direction_checked),
        "start_only_jaccard": float(start_overlap),
        "direction_check_status": str(direction_check.get("status", "")),
        "direction_check_reason": str(direction_check.get("reason", "")),
    }


def _pairwise_support_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    labels = []
    for i, left in enumerate(case_results):
        left_set = {int(row["window_index"]) for row in left["support"]}
        for right in case_results[i + 1 :]:
            right_set = {int(row["window_index"]) for row in right["support"]}
            jaccard = len(left_set & right_set) / max(len(left_set | right_set), 1)
            values.append(float(jaccard))
            labels.append(
                {
                    "left": str(left["case"]),
                    "right": str(right["case"]),
                    "support_jaccard": float(jaccard),
                }
            )
    if not values:
        return {"mean_support_jaccard": None, "pairs": []}
    return {
        "mean_support_jaccard": float(np.mean(values)),
        "max_support_jaccard": float(np.max(values)),
        "min_support_jaccard": float(np.min(values)),
        "pairs": labels,
    }


def _policy_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    counts = [int(case["metrics"]["support_count"]) for case in case_results]
    years = [int(case["metrics"]["unique_year_count"]) for case in case_results]
    start_jaccard = [
        float(case["metrics"]["start_only_jaccard"]) for case in case_results
    ]
    direction_status = Counter(
        str(case["metrics"]["direction_check_status"]) for case in case_results
    )
    all_indices = {
        int(row["window_index"])
        for case in case_results
        for row in case.get("support", [])
    }
    pairwise = _pairwise_support_summary(case_results)
    return {
        "case_count": int(len(case_results)),
        "mean_support_count": float(np.mean(counts)) if counts else 0.0,
        "min_support_count": int(min(counts)) if counts else 0,
        "max_support_count": int(max(counts)) if counts else 0,
        "mean_unique_year_count": float(np.mean(years)) if years else 0.0,
        "distinct_support_windows_across_cases": int(len(all_indices)),
        "mean_start_only_jaccard": (
            float(np.mean(start_jaccard)) if start_jaccard else 0.0
        ),
        "direction_status_counts": dict(sorted(direction_status.items())),
        **pairwise,
    }


def _market_implication_rows(grounding: dict[str, Any]) -> list[dict[str, Any]]:
    nested = grounding.get("condition_only_grounding", grounding)
    if not isinstance(nested, dict):
        return []
    rows = nested.get("current_market_state_implications", [])
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if isinstance(row, dict):
            out.append(
                {
                    "market": row.get("market"),
                    "direction": row.get("direction"),
                    "confidence": row.get("confidence"),
                    "evidence": row.get("evidence"),
                }
            )
    return out


def evaluate_bank(
    *,
    bank_name: str,
    bridge_report_path: Path,
    bridge_arrays_path: Path,
    condition_root: Path,
    checkpoint: Path,
    start_window_index: int,
    top_k: int,
    temperature: float,
    start_distance_threshold_z: float,
    diverse_max_pairwise_cosine: float,
) -> dict[str, Any]:
    bridge_report = _load_json(bridge_report_path)
    bridge_arrays = np.load(bridge_arrays_path)
    history_level, spec_names = _build_history_level(
        checkpoint=checkpoint,
        bridge_report=bridge_report,
    )
    memory_targets = np.asarray(bridge_arrays["memory_targets"], dtype=np.float32)
    train_indices, _test_indices = split_indices_from_bridge_report(bridge_report)
    if history_level.shape[0] != memory_targets.shape[0]:
        raise ValueError(
            f"{bank_name}: history_level {history_level.shape} and memory_targets "
            f"{memory_targets.shape} disagree"
        )
    metadata = _bridge_metadata(bridge_report)
    if int(start_window_index) < 0 or int(start_window_index) >= history_level.shape[0]:
        raise IndexError(
            f"start_window_index {start_window_index} outside {history_level.shape[0]}"
        )
    query_start_state = history_level[int(start_window_index), -1, :]
    policy_cases: dict[str, list[dict[str, Any]]] = {
        policy.name: [] for policy in POLICIES
    }
    case_inputs = []
    for case in CASE_NAMES:
        condition_report, query_memory = _load_condition(case, condition_root)
        cached = condition_report.get("cached_query", {})
        grounding = cached.get("grounding", {})
        if not isinstance(grounding, dict):
            grounding = {}
        base_candidates = candidate_support_table(
            query_memory=query_memory,
            memory_targets=memory_targets,
            history_level=history_level,
            train_indices=train_indices,
            query_window_index=int(start_window_index),
            query_start_state=query_start_state,
            grounding=grounding,
            spec_names=spec_names,
            start_distance_threshold_z=float(start_distance_threshold_z),
            start_distance_penalty=0.02,
            implication_alignment_weight=0.25,
        )
        start_only_prior = build_mixture_memory_prior(
            query_memory=query_memory,
            memory_targets=memory_targets,
            history_level=history_level,
            train_indices=train_indices,
            query_window_index=int(start_window_index),
            query_start_state=query_start_state,
            grounding=grounding,
            spec_names=spec_names,
            mode="soft_topk_start_only",
            top_k=int(top_k),
            temperature=float(temperature),
            start_distance_threshold_z=float(start_distance_threshold_z),
            start_distance_penalty=0.02,
            implication_alignment_weight=0.0,
            diverse_max_pairwise_cosine=float(diverse_max_pairwise_cosine),
            diverse_min_index_gap=0,
        )
        start_only_indices = {int(idx) for idx in start_only_prior["window_indices"]}
        case_inputs.append(
            {
                "case": case,
                "window_id": str(cached.get("window_id", "")),
                "query_text": str(cached.get("query_text", "")),
                "implications": _market_implication_rows(grounding),
                "direction_passing_count": int(
                    sum(
                        1
                        for row in base_candidates
                        if int(row.get("recent_prefix_checked", 0) or 0) > 0
                        and int(row.get("recent_prefix_mismatches", 0) or 0) == 0
                    )
                ),
            }
        )
        for policy in POLICIES:
            prior = build_mixture_memory_prior(
                query_memory=query_memory,
                memory_targets=memory_targets,
                history_level=history_level,
                train_indices=train_indices,
                query_window_index=int(start_window_index),
                query_start_state=query_start_state,
                grounding=grounding,
                spec_names=spec_names,
                mode=policy.prior_mode,
                top_k=int(top_k),
                temperature=float(temperature),
                start_distance_threshold_z=float(start_distance_threshold_z),
                start_distance_penalty=float(policy.start_distance_penalty),
                implication_alignment_weight=float(policy.implication_alignment_weight),
                diverse_max_pairwise_cosine=float(diverse_max_pairwise_cosine),
                diverse_min_index_gap=int(policy.diverse_min_index_gap),
            )
            candidates = candidate_support_table(
                query_memory=query_memory,
                memory_targets=memory_targets,
                history_level=history_level,
                train_indices=train_indices,
                query_window_index=int(start_window_index),
                query_start_state=query_start_state,
                grounding=grounding,
                spec_names=spec_names,
                start_distance_threshold_z=float(start_distance_threshold_z),
                start_distance_penalty=float(policy.start_distance_penalty),
                implication_alignment_weight=float(policy.implication_alignment_weight),
            )
            by_index = {int(row["window_index"]): row for row in candidates}
            support = _selected_support_rows(
                prior=prior,
                metadata=metadata,
                candidate_by_index=by_index,
            )
            policy_cases[policy.name].append(
                {
                    "case": case,
                    "window_id": str(cached.get("window_id", "")),
                    "support": support,
                    "metrics": _case_metrics(
                        support_rows=support,
                        start_only_indices=start_only_indices,
                        memory_targets=memory_targets,
                        prior=prior,
                    ),
                }
            )
    policy_payloads = []
    policy_by_name = {policy.name: policy for policy in POLICIES}
    for name, cases in policy_cases.items():
        policy = policy_by_name[name]
        policy_payloads.append(
            {
                "policy": name,
                "description": policy.description,
                "prior_mode": policy.prior_mode,
                "start_distance_penalty": policy.start_distance_penalty,
                "implication_alignment_weight": policy.implication_alignment_weight,
                "diverse_min_index_gap": policy.diverse_min_index_gap,
                "summary": _policy_summary(cases),
                "cases": cases,
            }
        )
    return {
        "bank": bank_name,
        "bridge_report": str(bridge_report_path),
        "bridge_arrays": str(bridge_arrays_path),
        "window_count": int(history_level.shape[0]),
        "train_count": int(train_indices.size),
        "start_window_index": int(start_window_index),
        "start_window_metadata": metadata.get(int(start_window_index), {}),
        "train_year_counts": dict(
            sorted(
                Counter(
                    _window_year(metadata.get(int(idx))) for idx in train_indices
                ).items()
            )
        ),
        "case_inputs": case_inputs,
        "policies": policy_payloads,
    }


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Fixed-Start Support Policy Bake-Off",
        "",
        "This diagnostic makes no OpenAI calls and does not run the SNI rollout. "
        "It asks whether the support pool under one fixed starting level is "
        "selected by the narrative or mostly by the start level.",
        "",
        "## Interpretation",
        "",
        "- Lower mean support Jaccard across narratives means stronger narrative-conditioned support separation.",
        "- Lower start-only Jaccard means the policy is less dominated by nearest starting levels.",
        "- More distinct support windows and years mean better regime breadth, subject to direction checks.",
        "- Direction pass/warn remains an audit; it should not collapse the support pool to one local episode.",
        "",
    ]
    for bank in report["banks"]:
        lines.extend(
            [
                f"## Bank: {bank['bank']}",
                "",
                f"- Windows: `{bank['window_count']}`; train supports: `{bank['train_count']}`.",
                f"- Fixed start: `{bank['start_window_index']}` / `{bank['start_window_metadata'].get('window_id', '')}` / `{bank['start_window_metadata'].get('calendar_end_date', '')}`.",
                f"- Train year counts: `{bank['train_year_counts']}`.",
                "",
                "| Policy | Mean supports | Distinct supports | Mean support Jaccard | Mean start-only Jaccard | Mean unique years | Direction statuses |",
                "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for policy in bank["policies"]:
            summary = policy["summary"]
            lines.append(
                "| {policy} | {mean_support:.2f} | {distinct} | {jaccard:.3f} | "
                "{start_jaccard:.3f} | {years:.2f} | `{statuses}` |".format(
                    policy=policy["policy"],
                    mean_support=float(summary["mean_support_count"]),
                    distinct=int(summary["distinct_support_windows_across_cases"]),
                    jaccard=float(summary.get("mean_support_jaccard") or 0.0),
                    start_jaccard=float(summary["mean_start_only_jaccard"]),
                    years=float(summary["mean_unique_year_count"]),
                    statuses=summary["direction_status_counts"],
                )
            )
        lines.append("")
        for policy in bank["policies"]:
            lines.extend([f"### {bank['bank']} / {policy['policy']}", ""])
            lines.append(policy["description"])
            lines.append("")
            lines.append(
                "| Case | Supports | Years | Start-only overlap | Top support dates |"
            )
            lines.append("| --- | ---: | --- | ---: | --- |")
            for case in policy["cases"]:
                metrics = case["metrics"]
                top_dates = ", ".join(
                    f"{row['window_id']}@{row['history_end']} w={row['weight']:.2f}"
                    for row in case["support"][:4]
                )
                lines.append(
                    f"| {case['case']} | {metrics['support_count']} | "
                    f"`{metrics['year_counts']}` | "
                    f"{metrics['start_only_jaccard']:.3f} | {top_dates} |"
                )
            lines.append("")
    return "\n".join(lines)


def run_bakeoff(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    banks = [
        evaluate_bank(
            bank_name="full_906b_380_window_bank",
            bridge_report_path=Path(args.bridge_report),
            bridge_arrays_path=Path(args.bridge_arrays),
            condition_root=Path(args.condition_root),
            checkpoint=Path(args.checkpoint),
            start_window_index=int(args.start_window_index),
            top_k=int(args.top_k),
            temperature=float(args.temperature),
            start_distance_threshold_z=float(args.start_distance_threshold_z),
            diverse_max_pairwise_cosine=float(args.diverse_max_pairwise_cosine),
        ),
        evaluate_bank(
            bank_name="representative_220_182_window_bank",
            bridge_report_path=Path(args.representative_bridge_report),
            bridge_arrays_path=Path(args.representative_bridge_arrays),
            condition_root=Path(args.condition_root),
            checkpoint=Path(args.checkpoint),
            start_window_index=int(args.start_window_index),
            top_k=int(args.top_k),
            temperature=float(args.temperature),
            start_distance_threshold_z=float(args.start_distance_threshold_z),
            diverse_max_pairwise_cosine=float(args.diverse_max_pairwise_cosine),
        ),
    ]
    report = {
        "status": "ok",
        "scope_note": (
            "Fixed-start support-policy bake-off. No OpenAI calls and no SNI "
            "rollout. The goal is to diagnose whether starting-level similarity "
            "dominates narrative-conditioned support selection."
        ),
        "top_k": int(args.top_k),
        "temperature": float(args.temperature),
        "diverse_max_pairwise_cosine": float(args.diverse_max_pairwise_cosine),
        "artifact_paths": {
            "json": str(output_dir / "fixed_start_support_policy_bakeoff.json"),
            "markdown": str(output_dir / "fixed_start_support_policy_bakeoff.md"),
        },
        "banks": banks,
    }
    _write_json(report["artifact_paths"]["json"], report)
    _write_text(report["artifact_paths"]["markdown"], _render_markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bridge-report", default=DEFAULT_BRIDGE_REPORT)
    parser.add_argument("--bridge-arrays", default=DEFAULT_BRIDGE_ARRAYS)
    parser.add_argument(
        "--representative-bridge-report", default=DEFAULT_REPRESENTATIVE_BRIDGE_REPORT
    )
    parser.add_argument(
        "--representative-bridge-arrays", default=DEFAULT_REPRESENTATIVE_BRIDGE_ARRAYS
    )
    parser.add_argument("--condition-root", default=DEFAULT_CONDITION_ROOT)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--start-window-index", type=int, default=18)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--start-distance-threshold-z", type=float, default=15.0)
    parser.add_argument("--diverse-max-pairwise-cosine", type=float, default=0.95)
    args = parser.parse_args()
    report = run_bakeoff(args)
    compact = {
        "status": report["status"],
        "json": report["artifact_paths"]["json"],
        "markdown": report["artifact_paths"]["markdown"],
        "banks": [
            {
                "bank": bank["bank"],
                "window_count": bank["window_count"],
                "policies": {
                    policy["policy"]: {
                        "mean_support_count": policy["summary"]["mean_support_count"],
                        "distinct_support_windows": policy["summary"][
                            "distinct_support_windows_across_cases"
                        ],
                        "mean_support_jaccard": policy["summary"][
                            "mean_support_jaccard"
                        ],
                        "mean_start_only_jaccard": policy["summary"][
                            "mean_start_only_jaccard"
                        ],
                    }
                    for policy in bank["policies"]
                },
            }
            for bank in report["banks"]
        ],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
