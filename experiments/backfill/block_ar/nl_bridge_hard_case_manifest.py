#!/usr/bin/env python
"""Build downstream hard-case manifests from bridge failure diagnostics.

This script is offline. It consumes the saved bridge diagnostics and the
risk-manager casebook, then preserves the original window identity fields needed
by downstream bridge training and scenario-level evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


SUBSET_RULES = {
    "bridge_hard_case_validation": {
        "actions": {"add_to_bridge_hard_case_validation_set"},
        "modes": set(),
    },
    "mixed_regime_contrastive": {
        "actions": {"add_mixed_regime_contrastive_bridge_examples"},
        "modes": {"mixed_regime_semantic_ambiguity"},
    },
    "rank_metric_review": {
        "actions": {"review_rank_metric_against_analogue_acceptance"},
        "modes": {"dense_neighbor_rank_metric_strictness"},
    },
    "label_repair": {
        "actions": {"repair_or_regenerate_grounded_labels"},
        "modes": {"label_quality_review"},
    },
    "bridge_model_hard_case": {
        "actions": {"targeted_bridge_retraining_or_bakeoff_candidate"},
        "modes": {"bridge_model_hard_case"},
    },
}

IDENTITY_KEYS = (
    "manifest_window_index",
    "bridge_local_window_index",
    "block_window_index",
    "source_index",
)

DOWNSTREAM_INDEX_KEYS = {
    "manifest_window_index": "manifest_window_indices",
    "bridge_local_window_index": "bridge_local_window_indices",
    "block_window_index": "block_window_indices",
    "source_index": "source_indices",
}

BRIDGE_METRIC_KEYS = (
    "target_cosine",
    "target_mse",
    "true_rank_test_pool",
    "true_rank_full_pool",
    "hard_negative_gap",
    "hard_negative_positive_mean_cosine",
    "hard_negative_negative_mean_cosine",
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


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _index_cases_by_window(casebook: dict[str, Any]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in _as_list(casebook.get("cases")):
        if isinstance(row, dict) and row.get("window_id"):
            indexed[str(row["window_id"])] = row
    return indexed


def _mode_codes(case: dict[str, Any]) -> list[str]:
    codes: list[str] = []
    for mode in _as_list(case.get("failure_modes")):
        if isinstance(mode, dict):
            code = mode.get("code")
        else:
            code = mode
        if code is not None:
            codes.append(str(code))
    return codes


def _recommended_actions(case: dict[str, Any]) -> list[str]:
    return [str(action) for action in _as_list(case.get("recommended_actions"))]


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return None
    if raw != raw or raw in {float("inf"), float("-inf")}:
        return None
    return raw


def _compact_bridge_metrics(case: dict[str, Any]) -> dict[str, Any]:
    bridge = _as_dict(case.get("bridge"))
    compact: dict[str, Any] = {}
    for key in BRIDGE_METRIC_KEYS:
        value = bridge.get(key)
        if key in {"true_rank_test_pool", "true_rank_full_pool"}:
            compact[key] = _int_or_none(value)
        else:
            compact[key] = _float_or_none(value)
    return compact


def _priority(
    *,
    acceptance_status: Any,
    failure_modes: list[str],
    recommended_actions: list[str],
) -> str:
    if "repair_or_regenerate_grounded_labels" in recommended_actions:
        return "label_first"
    if (
        str(acceptance_status) == "fail"
        or "bridge_model_hard_case" in failure_modes
        or "weak_target_alignment" in failure_modes
    ):
        return "high"
    if "add_to_bridge_hard_case_validation_set" in recommended_actions:
        return "medium"
    return "review"


def _manifest_case(
    diagnostic_case: dict[str, Any],
    casebook_case: dict[str, Any],
) -> dict[str, Any]:
    failure_modes = _mode_codes(diagnostic_case)
    recommended_actions = _recommended_actions(diagnostic_case)
    row: dict[str, Any] = {
        "window_id": str(diagnostic_case.get("window_id", "")),
        "acceptance_status": diagnostic_case.get("acceptance_status"),
        "priority": _priority(
            acceptance_status=diagnostic_case.get("acceptance_status"),
            failure_modes=failure_modes,
            recommended_actions=recommended_actions,
        ),
        "regime_tags": [
            str(tag) for tag in _as_list(diagnostic_case.get("regime_tags"))
        ],
        "input_narrative": str(diagnostic_case.get("input_narrative", "")),
        "observed_fact_tokens": str(diagnostic_case.get("observed_fact_tokens", "")),
        "bridge": _compact_bridge_metrics(diagnostic_case),
        "failure_modes": failure_modes,
        "recommended_actions": recommended_actions,
    }
    for key in IDENTITY_KEYS:
        row[key] = _int_or_none(casebook_case.get(key))
    for key in ("calendar", "manifest_split", "selection_reasons"):
        if key in casebook_case:
            row[key] = casebook_case[key]
    return row


def _belongs_to_subset(case: dict[str, Any], subset_name: str) -> bool:
    rule = SUBSET_RULES[subset_name]
    actions = set(_as_list(case.get("recommended_actions")))
    modes = set(_as_list(case.get("failure_modes")))
    return bool(actions & rule["actions"] or modes & rule["modes"])


def _build_subsets(cases: list[dict[str, Any]]) -> dict[str, list[str]]:
    subsets: dict[str, list[str]] = {name: [] for name in SUBSET_RULES}
    for case in cases:
        window_id = str(case.get("window_id", ""))
        for subset_name in SUBSET_RULES:
            if _belongs_to_subset(case, subset_name):
                subsets[subset_name].append(window_id)
    return subsets


def _downstream_indices(
    subsets: dict[str, list[str]],
    cases: list[dict[str, Any]],
) -> dict[str, dict[str, list[Any]]]:
    cases_by_id = {str(case.get("window_id")): case for case in cases}
    downstream: dict[str, dict[str, list[Any]]] = {}
    for subset_name, window_ids in subsets.items():
        row: dict[str, list[Any]] = {"window_ids": list(window_ids)}
        for source_key, output_key in DOWNSTREAM_INDEX_KEYS.items():
            row[output_key] = [
                cases_by_id[window_id][source_key]
                for window_id in window_ids
                if window_id in cases_by_id
                and cases_by_id[window_id].get(source_key) is not None
            ]
        downstream[subset_name] = row
    return downstream


def build_bridge_hard_case_manifest(
    diagnostics: dict[str, Any],
    casebook: dict[str, Any],
    *,
    title: str = "Bridge Hard-Case Validation Manifest",
) -> dict[str, Any]:
    """Create a downstream-ready bridge hard-case manifest."""

    casebook_by_window = _index_cases_by_window(casebook)
    cases: list[dict[str, Any]] = []
    missing_casebook_rows: list[str] = []
    for diagnostic_case in _as_list(diagnostics.get("cases")):
        if not isinstance(diagnostic_case, dict) or not diagnostic_case.get(
            "window_id"
        ):
            continue
        window_id = str(diagnostic_case["window_id"])
        casebook_case = casebook_by_window.get(window_id, {})
        if not casebook_case:
            missing_casebook_rows.append(window_id)
        cases.append(_manifest_case(diagnostic_case, casebook_case))

    subsets = _build_subsets(cases)
    action_counts = Counter(
        action for case in cases for action in _as_list(case.get("recommended_actions"))
    )
    failure_mode_counts = Counter(
        mode for case in cases for mode in _as_list(case.get("failure_modes"))
    )
    priority_counts = Counter(str(case.get("priority", "review")) for case in cases)
    summary = {
        "case_count": len(cases),
        "subset_counts": {name: len(ids) for name, ids in sorted(subsets.items())},
        "priority_counts": dict(sorted(priority_counts.items())),
        "action_counts": dict(sorted(action_counts.items())),
        "failure_mode_counts": dict(sorted(failure_mode_counts.items())),
        "missing_casebook_rows": missing_casebook_rows,
        "recommended_next_step": _as_dict(diagnostics.get("summary")).get(
            "recommended_next_step",
            "Use the bridge hard-case subsets for targeted bridge validation.",
        ),
    }
    return {
        "title": title,
        "status": "bridge_hard_case_manifest",
        "summary": summary,
        "subsets": subsets,
        "downstream_indices": _downstream_indices(subsets, cases),
        "cases": cases,
    }


def _fmt_float(value: Any, digits: int = 3) -> str:
    raw = _float_or_none(value)
    return "n/a" if raw is None else f"{raw:.{digits}f}"


def _comma_list(values: list[Any]) -> str:
    return ", ".join(str(value) for value in values) if values else "none"


def render_bridge_hard_case_manifest_markdown(manifest: dict[str, Any]) -> str:
    """Render a hard-case manifest as Markdown."""

    summary = _as_dict(manifest.get("summary"))
    subsets = _as_dict(manifest.get("subsets"))
    lines = [
        f"# {manifest.get('title') or 'Bridge Hard-Case Validation Manifest'}",
        "",
        "## Summary",
        "",
        f"- Cases: {summary.get('case_count', 0)}",
        f"- Subset counts: {json.dumps(summary.get('subset_counts', {}), sort_keys=True)}",
        f"- Priorities: {json.dumps(summary.get('priority_counts', {}), sort_keys=True)}",
        f"- Recommended next step: {summary.get('recommended_next_step', '')}",
        "",
        "## Downstream Subsets",
        "",
    ]
    for subset_name in SUBSET_RULES:
        lines.append(
            f"- `{subset_name}`: {_comma_list(_as_list(subsets.get(subset_name)))}"
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Window | Priority | Status | Manifest Idx | Bridge Idx | Block Idx | Source Idx | Target Cosine | Test Rank | Modes | Actions |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for case in _as_list(manifest.get("cases")):
        if not isinstance(case, dict):
            continue
        bridge = _as_dict(case.get("bridge"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(case.get("window_id", "")),
                    str(case.get("priority", "")),
                    str(case.get("acceptance_status", "")),
                    str(case.get("manifest_window_index", "n/a")),
                    str(case.get("bridge_local_window_index", "n/a")),
                    str(case.get("block_window_index", "n/a")),
                    str(case.get("source_index", "n/a")),
                    _fmt_float(bridge.get("target_cosine")),
                    str(bridge.get("true_rank_test_pool", "n/a")),
                    _comma_list(_as_list(case.get("failure_modes"))),
                    _comma_list(_as_list(case.get("recommended_actions"))),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Case Details", ""])
    for case in _as_list(manifest.get("cases")):
        if not isinstance(case, dict):
            continue
        lines.extend(
            [
                f"### {case.get('window_id')}",
                "",
                f"- Priority: {case.get('priority')}",
                f"- Regimes: {_comma_list(_as_list(case.get('regime_tags')))}",
                f"- Source identity: manifest={case.get('manifest_window_index')}, bridge={case.get('bridge_local_window_index')}, block={case.get('block_window_index')}, source={case.get('source_index')}",
                f"- Narrative: {case.get('input_narrative', '')}",
                f"- Observed facts: {case.get('observed_fact_tokens', '')}",
                f"- Actions: {_comma_list(_as_list(case.get('recommended_actions')))}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True)
    parser.add_argument("--casebook", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--title", default="Bridge Hard-Case Validation Manifest")
    args = parser.parse_args()

    manifest = build_bridge_hard_case_manifest(
        _load_json(args.diagnostics),
        _load_json(args.casebook),
        title=args.title,
    )
    output_dir = Path(args.output_dir)
    json_path = output_dir / "bridge_hard_case_manifest.json"
    markdown_path = output_dir / "bridge_hard_case_manifest.md"
    _write_json(json_path, manifest)
    _write_text(markdown_path, render_bridge_hard_case_manifest_markdown(manifest))
    print(
        json.dumps(
            {
                "manifest_json": str(json_path),
                "manifest_markdown": str(markdown_path),
                "summary": manifest["summary"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
