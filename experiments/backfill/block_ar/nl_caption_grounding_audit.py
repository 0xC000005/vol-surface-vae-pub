"""Audit narrative label and hard-negative grounding hard cases.

The purpose is label-quality triage, not model tuning. It joins rejected label
validation rows with bridge hard-negative separation failures so that the next
research step can repair captions before adding more bridge architecture.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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


def _round(value: Any, digits: int = 12) -> float | None:
    raw = _float_or_none(value)
    return None if raw is None else round(raw, digits)


def _bundle_by_window(pipeline_report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for bundle in _as_list(pipeline_report.get("narrative_bundles")):
        if isinstance(bundle, dict) and bundle.get("window_id"):
            rows[str(bundle["window_id"])] = bundle
    return rows


def _validation_by_window(
    pipeline_report: dict[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {}
    for item in _as_list(pipeline_report.get("label_validation")):
        if not isinstance(item, dict) or not item.get("window_id"):
            continue
        errors = [err for err in _as_list(item.get("errors")) if isinstance(err, dict)]
        if errors:
            rows[str(item["window_id"])] = errors
    return rows


def _primary_narrative(bundle: dict[str, Any]) -> dict[str, Any]:
    narratives = _as_list(bundle.get("narratives"))
    if narratives and isinstance(narratives[0], dict):
        return narratives[0]
    return {}


def _catalyst_labels(bundle: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for key in ("narrative_catalysts",):
        for item in _as_list(bundle.get(key)):
            if isinstance(item, dict):
                label = item.get("label") or item.get("description")
                if label:
                    labels.append(str(label))
    for item in _as_list(_primary_narrative(bundle).get("narrative_catalysts")):
        if isinstance(item, dict):
            label = item.get("label") or item.get("description")
            if label:
                labels.append(str(label))
    deduped: list[str] = []
    seen: set[str] = set()
    for label in labels:
        if label not in seen:
            seen.add(label)
            deduped.append(label)
    return deduped


def _case_from_bundle(
    window_id: str,
    *,
    bundle: dict[str, Any],
    validation_errors: list[dict[str, Any]],
    hard_row: dict[str, Any] | None,
    reason_codes: list[str],
) -> dict[str, Any]:
    primary = _primary_narrative(bundle)
    hard = hard_row or {}
    if validation_errors:
        action = "regenerate_or_repair_label"
    elif "low_hard_negative_margin" in reason_codes:
        action = "inspect_hard_negatives"
    else:
        action = "review"
    return {
        "window_id": window_id,
        "reason_codes": reason_codes,
        "recommended_action": action,
        "hard_margin": _round(hard.get("hard_margin")),
        "negative_gap": _round(hard.get("negative_gap")),
        "positive_mean_cosine": _round(hard.get("positive_mean_cosine")),
        "negative_mean_cosine": _round(hard.get("negative_mean_cosine")),
        "validation_errors": validation_errors,
        "primary_narrative": str(primary.get("text", "")),
        "observed_fact_tokens": str(primary.get("observed_fact_tokens", "")),
        "unsupported_claim_count": len(_as_list(primary.get("unsupported_claims"))),
        "catalyst_labels": _catalyst_labels(bundle),
    }


def _hard_negative_rows(bridge_report: dict[str, Any]) -> list[dict[str, Any]]:
    hard = _as_dict(
        _as_dict(bridge_report.get("evaluation")).get("hard_negative_separation")
    )
    return [row for row in _as_list(hard.get("windows")) if isinstance(row, dict)]


def build_caption_grounding_audit(
    pipeline_report: dict[str, Any],
    bridge_report: dict[str, Any],
    *,
    low_margin_threshold: float = 0.50,
    low_gap_threshold: float = 0.50,
) -> dict[str, Any]:
    bundles = _bundle_by_window(pipeline_report)
    validation = _validation_by_window(pipeline_report)
    rejected = {
        str(item) for item in _as_list(pipeline_report.get("rejected_label_windows"))
    }
    hard_by_window = {
        str(row.get("window_id")): row
        for row in _hard_negative_rows(bridge_report)
        if row.get("window_id")
    }

    case_ids: set[str] = set(validation) | rejected
    low_margin_ids: set[str] = set()
    low_gap_ids: set[str] = set()
    for window_id, row in hard_by_window.items():
        hard_margin = _float_or_none(row.get("hard_margin"))
        negative_gap = _float_or_none(row.get("negative_gap"))
        if hard_margin is not None and hard_margin < float(low_margin_threshold):
            low_margin_ids.add(window_id)
            case_ids.add(window_id)
        if negative_gap is not None and negative_gap < float(low_gap_threshold):
            low_gap_ids.add(window_id)
            case_ids.add(window_id)

    cases: list[dict[str, Any]] = []
    for window_id in sorted(case_ids):
        reason_codes: list[str] = []
        if window_id in rejected or window_id in validation:
            reason_codes.append("label_validation_error")
        if window_id in low_margin_ids:
            reason_codes.append("low_hard_negative_margin")
        if window_id in low_gap_ids:
            reason_codes.append("low_hard_negative_gap")
        cases.append(
            _case_from_bundle(
                window_id,
                bundle=bundles.get(window_id, {}),
                validation_errors=validation.get(window_id, []),
                hard_row=hard_by_window.get(window_id),
                reason_codes=reason_codes,
            )
        )

    cases.sort(
        key=lambda row: (
            0 if "label_validation_error" in row["reason_codes"] else 1,
            float("inf") if row["hard_margin"] is None else float(row["hard_margin"]),
            row["window_id"],
        )
    )
    return {
        "status": "ok",
        "thresholds": {
            "low_margin_threshold": float(low_margin_threshold),
            "low_gap_threshold": float(low_gap_threshold),
        },
        "summary": {
            "case_count": len(cases),
            "rejected_label_count": len(rejected | set(validation)),
            "low_margin_count": len(low_margin_ids),
            "low_gap_count": len(low_gap_ids),
        },
        "cases": cases,
    }


def _write_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Caption Grounding Audit",
        "",
        f"- Cases: `{audit['summary']['case_count']}`",
        f"- Rejected labels: `{audit['summary']['rejected_label_count']}`",
        f"- Low hard-negative margin: `{audit['summary']['low_margin_count']}`",
        f"- Low hard-negative gap: `{audit['summary']['low_gap_count']}`",
        "",
        "## Cases",
        "",
    ]
    for case in audit["cases"]:
        lines.extend(
            [
                f"### {case['window_id']}",
                "",
                f"- Reasons: `{', '.join(case['reason_codes'])}`",
                f"- Action: `{case['recommended_action']}`",
                f"- Hard margin: `{case['hard_margin']}`",
                f"- Negative gap: `{case['negative_gap']}`",
                f"- Observed facts: `{case['observed_fact_tokens']}`",
                f"- Unsupported claims: `{case['unsupported_claim_count']}`",
                f"- Catalysts: `{', '.join(case['catalyst_labels'])}`",
                f"- Narrative: {case['primary_narrative']}",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-report", required=True)
    parser.add_argument("--bridge-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--low-margin-threshold", type=float, default=0.50)
    parser.add_argument("--low-gap-threshold", type=float, default=0.50)
    args = parser.parse_args()

    pipeline_report = json.loads(Path(args.pipeline_report).read_text(encoding="utf-8"))
    bridge_report = json.loads(Path(args.bridge_report).read_text(encoding="utf-8"))
    audit = build_caption_grounding_audit(
        pipeline_report,
        bridge_report,
        low_margin_threshold=float(args.low_margin_threshold),
        low_gap_threshold=float(args.low_gap_threshold),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "caption_grounding_audit.json"
    markdown_path = output_dir / "caption_grounding_audit.md"
    json_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_markdown(markdown_path, audit)
    print(
        json.dumps(
            {"json": str(json_path), "markdown": str(markdown_path)}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
